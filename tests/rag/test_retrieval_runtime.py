"""检索运行时、仓储、退款工具、KB 调试 API 与显式索引准备的接入测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager
import importlib

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, KnowledgeUnavailable
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import VectorIndexUnavailable
from app.repositories.support import SupportRepository
from app.schemas.chat import ChatRequest
from app.tools.registry import build_registry
from tests.chat_fakes import make_chat_service
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, make_settings
from tests.rag.test_retrieval import _chunk, _search


def test_numeric_retrieval_settings_accept_environment_strings_but_reject_boolean(monkeypatch) -> None:
    """真实 .env 字符串须可解析，布尔值不能充当数量。参数：替换器；返回：None；异常：断言失败。"""
    from pydantic import ValidationError

    source = importlib.import_module("app.config")
    for key, value in {"RECALL_TOP_K": "50", "RERANK_TOP_K": "10", "RERANK_BATCH_SIZE": "4"}.items():
        monkeypatch.setenv(key, value)
    settings = source.Settings(_env_file=None, chat_base_url="http://test.invalid/v1", chat_model="test-model", chat_api_key="test-key")
    assert (settings.recall_top_k, settings.rerank_top_k, settings.rerank_batch_size) == (50, 10, 4)
    with pytest.raises(ValidationError):
        make_settings(recall_top_k=True)


async def test_support_faq_default_is_ten_and_explicit_limit_is_preserved() -> None:
    """仓储不得以旧 limit=3 覆盖新检索默认 10。参数：无；返回：None；异常：断言失败。"""
    search, _, _, _ = _search([_chunk(identifier) for identifier in range(1, 16)])
    repository = SupportRepository(None, search)
    assert len(await repository.query_faq("退款处理时限")) == 10
    assert len(await repository.query_faq("退款处理时限", limit=5)) == 5


async def test_refund_tool_sends_thirty_authoritative_hits_to_answer_model() -> None:
    """退款 30 条必须经过真实工具/仓储/聊天消息，不能被 FAQ 限额再裁十条。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import expand_policy_queries

    query = "商品损坏如何退款"
    queries = expand_policy_queries(query)
    groups = {query: list(range(number * 10 + 1, number * 10 + 11)) for number, query in enumerate(queries)}
    search, _, _, _ = _search([_chunk(identifier) for identifier in range(1, 31)], groups=groups)
    decision = AIMessage(content="", tool_calls=[{"id": "refund", "name": "query_refund_policy", "args": {"query": query}}])
    service, persisted, _, model = make_chat_service(decision, chunks=["需按查到的政策申请，退款尚未执行。"])
    service.executor.support = SupportRepository(None, search)
    events = [event async for event in service.stream_turn(ChatRequest(session_id="refund-30", message=query))]
    assert events[-1].event == "done"
    result = next(iter(persisted.results.values()))
    assert result.ok and len(result.data["hits"]) == 30
    answer_input = next(messages for phase, messages in model.inputs if phase == "answer")
    facts = [json.loads(message.content)["data"]["hits"] for message in answer_input if isinstance(message, ToolMessage)]
    assert len(facts[0]) == 30
    assert {hit["id"]: hit["answer"] for hit in facts[0]} == {
        identifier: f"权威答案{identifier}。" for identifier in range(1, 31)
    }
    assert [hit["id"] for hit in facts[0]] == [*range(1, 31, 2), *range(30, 1, -2)]


async def test_hybrid_faq_tool_uses_model_keyword_for_clause_split() -> None:
    """混合路径用模型提交的检索词，并在该词内部继续做子句拆分。

    参数：无。
    返回：None。
    异常：AssertionError，检索词被改回本轮原话或子句拆分失效。
    """
    search, _, index, reranker = _search([_chunk(1)])
    decision = AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq",
                                                  "args": {"keyword": "退货申请条件，退款处理时限"}}])
    service, _, _, _ = make_chat_service(decision)
    service.executor.support = SupportRepository(None, search)
    events = [event async for event in service.stream_turn(
        ChatRequest(session_id="model-keyword", message="会员权益怎么算"))]
    assert events[-1].event == "done"
    # 检索词来自模型 keyword，不是本轮原话；子句拆分在 keyword 内部照常生效。
    assert [call[0] for call in index.calls] == ["退货申请条件", "退款处理时限"]
    assert len(reranker.calls) == 2


def test_kb_debug_api_default_five_and_explicit_top_k() -> None:
    """调试 API 默认值必须显式传 5，而不是使用聊天默认 10。参数：无；返回：None；异常：断言失败。"""
    from app.main import create_app

    search, _, _, reranker = _search([_chunk(identifier) for identifier in range(1, 13)])
    service, _, _, _ = make_chat_service()
    service.executor.support = SupportRepository(None, search)

    @asynccontextmanager
    async def runtime():
        """仅替换外部启动依赖。参数：无；返回：真实聊天服务；异常：无。"""
        yield service

    with TestClient(create_app(runtime_factory=runtime)) as client:
        first = client.get("/kb/search", params={"query": "退款处理时限"})
        assert first.status_code == 200 and len(first.json()["hits"]) == 5
        assert len(client.get("/kb/search", params={"query": "退款处理时限", "top_k": 7}).json()["hits"]) == 7
        assert client.get("/kb/search", params={"query": "退款", "top_k": 0}).status_code == 422
        assert client.get("/kb/search", params={"query": "   "}).status_code == 422
    assert [call[2] for call in reranker.calls] == [5, 7]


def test_kb_backend_error_is_safe_503_not_empty_success() -> None:
    """索引错误不能在调试 API 伪装成 200 空命中或泄露异常。参数：无；返回：None；异常：断言失败。"""
    from app.main import create_app

    search, _, index, _ = _search([_chunk(1)])

    async def fail(*args, **kwargs):
        """模拟内部故障。参数：请求；返回：不返回；异常：VectorIndexUnavailable。"""
        raise VectorIndexUnavailable("secret backend address")

    index.hybrid_search = fail
    service, _, _, _ = make_chat_service()
    service.executor.support = SupportRepository(None, search)

    @asynccontextmanager
    async def runtime():
        """提供隔离运行时。参数：无；返回：服务；异常：无。"""
        yield service

    with TestClient(create_app(runtime_factory=runtime)) as client:
        response = client.get("/kb/search", params={"query": "退款处理时限"})
        assert response.status_code == 503 and "secret" not in response.text


async def test_faq_has_independent_timeout_from_other_tools() -> None:
    """本地多次重排不应误用订单的短期限，业务工具期限不能被全局放宽。参数：无；返回：None；异常：断言失败。"""
    from app.schemas.tools import ToolContext
    from app.tools.executor import ToolExecutor
    from tests.chat_fakes import MemoryRepository

    class SlowSupport:
        """模拟慢知识查询的外部边界。"""

        async def query_faq(self, keyword):
            """短暂等待。参数：关键词；返回：空命中；异常：取消传播。"""
            await asyncio.sleep(0.03)
            return []

    repository = MemoryRepository()
    executor = ToolExecutor(build_registry(), repository, SlowSupport(), timeout_seconds=0.005,
                            max_retries=0, knowledge_timeout_seconds=0.2)
    context = await repository.start_turn(ToolContext(session_id="timeout", user_id="u", turn_id="t", user_message="退款条件"), "退款条件")
    context = context.model_copy(update={"tool_call_id": "faq"})

    async def status(frame):
        """接收状态。参数：帧；返回：None；异常：无。"""

    result = await executor.execute({"id": "faq", "name": "query_faq", "args": {"keyword": "退款条件"}}, context, status)
    assert result.ok and executor.timeout == 0.005


async def test_hybrid_startup_missing_index_fails_and_releases_without_dense_fallback(monkeypatch) -> None:
    """新路径故障须阻止启动并关闭已创建资源，不能回退旧索引。参数：替换器；返回：None；异常：断言失败。"""
    from app.knowledge import runtime

    encoder = FakeEncoder()

    class BrokenIndex:
        """模拟新集合未准备的 SDK 边界。"""

        closed = False

        async def ensure_ready(self, create=False):
            """明确报告缺集合。参数：创建开关；返回：不返回；异常：VectorIndexUnavailable。"""
            assert not create
            raise VectorIndexUnavailable("集合尚未准备")

        async def aclose(self):
            """标记关闭。参数：无；返回：None；异常：无。"""
            self.closed = True

    index = BrokenIndex()
    monkeypatch.setattr(runtime, "BgeM3Encoder", lambda config: encoder)
    monkeypatch.setattr(runtime, "MilvusHybridIndex", lambda config: index)

    def forbid(config):
        """禁止旧 Dense 回退。参数：配置；返回：不返回；异常：AssertionError。"""
        raise AssertionError("不能隐式回退")

    monkeypatch.setattr(runtime, "MilvusVectorIndex", forbid)
    with pytest.raises(VectorIndexUnavailable):
        async with runtime.online_search_runtime(make_settings(), MemoryKnowledgeStore()):
            pytest.fail("错误启动不能进入服务")
    assert encoder.closed and index.closed


async def test_cancelled_hybrid_warmup_closes_all_resources(monkeypatch) -> None:
    """启动预热取消也要释放编码器/重排/索引，不能留显存和连接。参数：替换器；返回：None；异常：断言失败。"""
    from app.knowledge import runtime
    from tests.knowledge.fakes import MemoryVectorIndex

    entered = asyncio.Event()
    encoder, index = FakeEncoder(), MemoryVectorIndex()

    class Reranker:
        """仅替换正在预热的外部模型。"""

        closed = False

        async def rank(self, *args, **kwargs):
            """等待取消。参数：预热输入；返回：不返回；异常：CancelledError。"""
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            """标记模型资源关闭。参数：无；返回：None；异常：无。"""
            self.closed = True

    ranking = Reranker()
    monkeypatch.setattr(runtime, "BgeM3Encoder", lambda config: encoder)
    monkeypatch.setattr(runtime, "MilvusHybridIndex", lambda config: index)
    monkeypatch.setattr(runtime, "BgeReranker", lambda config: ranking)

    async def check(config):
        """替代已验证的质量表外部依赖。参数：配置；返回：None；异常：无。"""

    monkeypatch.setattr(runtime, "check_quality_tables", check)

    async def start():
        """进入实际运行时。参数：无；返回：None；异常：原启动取消。"""
        async with runtime.online_search_runtime(make_settings(), MemoryKnowledgeStore()):
            pytest.fail("取消预热不能启动成功")

    job = asyncio.create_task(start())
    await entered.wait()
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert encoder.closed and index.closed and ranking.closed


async def test_refund_over_context_budget_reports_error_without_dropping_evidence() -> None:
    """30 条完整原文放不下时明确失败，不能默默只用十条回答。参数：无；返回：None；异常：断言失败。"""
    rows = []
    for number in range(1, 31):
        text = KnowledgeText(category="售后", questions=f"问题{number}", answer="必须验货并保留全部配件。" * 100)
        rows.append(_chunk(number).model_copy(update={"text": text, "content_hash": content_fingerprint(text)}))
    from app.knowledge.retrieval import expand_policy_queries
    query = "商品损坏如何退款"
    groups = {text: list(range(index * 10 + 1, index * 10 + 11)) for index, text in enumerate(expand_policy_queries(query))}
    search, _, _, _ = _search(rows, groups=groups)
    decision = AIMessage(content="", tool_calls=[{"id": "refund", "name": "query_refund_policy", "args": {"query": query}}])
    service, repository, _, model = make_chat_service(decision)
    service.executor.support = SupportRepository(None, search)
    events = [event async for event in service.stream_turn(ChatRequest(session_id="refund-budget", message=query))]
    assert any(event.event == "error" for event in events) and not any(event.event == "done" for event in events)
    assert len(next(iter(repository.results.values())).data["hits"]) == 30
    assert not any(phase == "answer" for phase, _ in model.inputs)


async def test_dense_mode_document_runtime_retains_legacy_token_chunking(monkeypatch) -> None:
    """显式 CH03 回退也必须保留旧文档边界，不在旧索引重新发布字符切分。参数：替换器；返回：None；异常：断言失败。"""
    from app.knowledge import runtime
    from app.knowledge.manifest import DocumentSpec
    from tests.knowledge.fakes import MemoryVectorIndex
    from types import SimpleNamespace

    @asynccontextmanager
    async def storage(config):
        """替代外部存储连接。参数：配置；返回：真实内存 store；异常：无。"""
        yield MemoryKnowledgeStore()

    monkeypatch.setattr(runtime, "knowledge_storage", storage)
    monkeypatch.setattr(runtime, "BgeM3Encoder", lambda config: FakeEncoder())
    monkeypatch.setattr(runtime, "MilvusVectorIndex", lambda config: MemoryVectorIndex())
    monkeypatch.setattr(runtime, "load_relation_policy", lambda *args, **kwargs: SimpleNamespace(min_score=0.0))
    source = "# 政策\n## 条件\n" + "长" * 600 + "。"
    spec = __import__("app.knowledge.manifest", fromlist=["DocumentSpec"]).DocumentSpec(
        path=Path("a.md"), source_key="doc:a.md", content_type="policy", key_sections=[])
    async with runtime.knowledge_runtime(make_settings(knowledge_retrieval_mode="dense")) as jobs:
        chunks = jobs.document_chunker(source, spec, jobs.encoder.count_tokens)
        assert len(chunks) == 1 and len(chunks[0].text.answer) == 601


def test_cli_explicit_hybrid_test_import_targets_ch04_not_ch03(monkeypatch) -> None:
    """新链路测试不能写旧章测试库，--mode hybrid 同时隔离两类资源。参数：替换器；返回：None；异常：断言失败。"""
    import sys
    from app.knowledge import cli
    from app.knowledge.schemas import SyncReport

    observed = []
    result = SyncReport(scanned=0, upserted=0, marked_done=0, deleted_vectors=0, remaining_pending=0)

    class Jobs:
        """仅替换外部同步任务。"""

        async def sync_pending(self):
            """返回同步报告。参数：无；返回：SyncReport；异常：无。"""
            return result

    @asynccontextmanager
    async def runtime(config):
        """观察 CLI 的真实配置路由。参数：配置；返回：任务；异常：无。"""
        observed.append((config.mysql_database, config.milvus_database, config.knowledge_retrieval_mode))
        yield Jobs()

    monkeypatch.setattr(cli, "knowledge_runtime", runtime)
    monkeypatch.setattr(sys, "argv", ["knowledge", "sync", "--test", "--mode", "hybrid"])
    cli.main()
    assert observed == [("mewhelp_ch04_test", "mewhelp_ch04_test", "hybrid")]


async def test_prepare_hybrid_holds_writer_lock_and_only_syncs_new_index(monkeypatch) -> None:
    """准备命令必须持写锁，不能创建/清理旧 Dense 集合。参数：替换器；返回：None；异常：断言失败。"""
    from app.knowledge import runtime
    from tests.knowledge.fakes import MemoryVectorIndex

    row = _chunk(1).model_copy(update={"vectorize_status": "pending", "vector_id": None})
    store, encoder = MemoryKnowledgeStore(snapshots=[row]), FakeEncoder()
    held = []
    original_writer = store.writer

    @asynccontextmanager
    async def writer():
        """暴露真实同步期间的租约边界。参数：无；返回：写入器；异常：同步异常。"""
        async with original_writer() as value:
            held.append(True)
            try:
                yield value
            finally:
                held.pop()

    store.writer = writer

    class Index(MemoryVectorIndex):
        """仅模拟外部索引，验证新索引在持锁范围准备。"""

        async def ensure_ready(self, create=False, *, confirmed_database=None):
            """检查锁。参数：创建/确认；返回：None；异常：断言失败。"""
            assert held and create and confirmed_database is None

    index = Index()

    @asynccontextmanager
    async def storage(config):
        """替换 MySQL 外部连接。参数：配置；返回：权威 store；异常：无。"""
        yield store

    monkeypatch.setattr(runtime, "knowledge_storage", storage)
    monkeypatch.setattr(runtime, "BgeM3Encoder", lambda config: encoder)
    monkeypatch.setattr(runtime, "MilvusHybridIndex", lambda config: index)
    report = await runtime.prepare_hybrid_index(make_settings())
    assert report.marked_done == 1 and index.ids == {1}
    assert index.records[1].text == "分类：售后\n问法：问法1\n答案：权威答案1。"
    assert encoder.closed and index.closed and not held


def test_cli_prepare_hybrid_exists_and_test_targets_are_isolated(monkeypatch, capsys) -> None:
    """新准备 CLI 不复用 CH03 测试库，不能只切半边资源。参数：替换器/输出；返回：None；异常：断言失败。"""
    import sys
    from app.knowledge import cli
    from app.knowledge.schemas import SyncReport

    observed = []

    async def prepare(config, *, confirmed_database=None):
        """替换索引外部准备。参数：配置/确认；返回：同步报告；异常：无。"""
        observed.append((config.mysql_database, config.milvus_database, config.milvus_hybrid_collection))
        return SyncReport(scanned=0, upserted=0, marked_done=0, deleted_vectors=0, remaining_pending=0)

    monkeypatch.setattr(cli, "prepare_hybrid_index", prepare)
    monkeypatch.setattr(sys, "argv", ["knowledge", "prepare-hybrid", "--test"])
    cli.main()
    assert observed == [("mewhelp_ch04_test", "mewhelp_ch04_test", "knowledge_ch04")]
