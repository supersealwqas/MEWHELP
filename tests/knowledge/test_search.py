"""在线 Dense 检索与 SupportRepository 注入契约测试。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.knowledge.errors import KnowledgeStoreUnavailable
from app.knowledge.schemas import (
    ChunkSnapshot,
    KnowledgeText,
    KnowledgeUnavailable,
    RetrievalPolicy,
    VectorRecord,
)
from app.knowledge.search import KnowledgeSearch, KnowledgeSearchUnavailable
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import VectorIndexUnavailable
from app.repositories.support import SupportRepository
from app.schemas.chat import ChatRequest
from app.schemas.tools import ToolResult
from app.tools.registry import build_registry
from tests.chat_fakes import make_chat_service
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, MemoryVectorIndex

REVISION = "fake-encoder-v1"
QUERY_VECTOR = [1.0] + [0.0] * 1023


def _policy(min_score: float = 0.0) -> RetrievalPolicy:
    """构造检索策略。参数：阈值；返回：策略；异常：校验失败。"""
    return RetrievalPolicy(
        min_score=min_score,
        embedding_revision=REVISION,
        chunking_version="ch03-v2",
        calibration_digest="test-only",
    )


def _snapshot(
    identifier: int,
    answer: str,
    *,
    active: bool = True,
    vector_id: str | None = None,
    revision: str = REVISION,
) -> ChunkSnapshot:
    """构造知识快照。参数：主键/答案/发布与向量状态；返回：快照；异常：校验失败。"""
    text = KnowledgeText(category="物流", questions=f"问法{identifier}", answer=answer)
    return ChunkSnapshot(
        id=identifier,
        source_key=f"doc:{identifier}.md",
        ordinal=0,
        text=text,
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
        content_hash=content_fingerprint(text),
        embedding_revision=revision,
        is_active=active,
        vector_id=str(identifier) if vector_id is None and active else vector_id,
        vectorize_status="done",
    )


def _record(snapshot: ChunkSnapshot, vector: list[float]) -> VectorRecord:
    """由快照与向量构造索引记录。参数：快照/向量；返回：记录；异常：校验失败。"""
    return VectorRecord(
        id=snapshot.id,
        vector=vector,
        content_hash=snapshot.content_hash,
        embedding_revision=snapshot.embedding_revision,
    )


def _search(
    snapshots: list[ChunkSnapshot],
    records: list[VectorRecord],
    *,
    min_score: float = 0.0,
) -> tuple[KnowledgeSearch, FakeEncoder, MemoryVectorIndex]:
    """组装内存检索。参数：快照/记录/阈值；返回：搜索器与依赖；异常：无。"""
    store = MemoryKnowledgeStore(snapshots=snapshots)
    index = MemoryVectorIndex()
    for record in records:
        index.records[record.id] = record
    encoder = FakeEncoder()
    return KnowledgeSearch(store, encoder, index, _policy(min_score)), encoder, index


async def test_keyword_encoded_verbatim() -> None:
    """关键词原样编码，不做同义改写。参数：无；返回：None；异常：无。"""
    snapshot = _snapshot(1, "未满99元收6元。")
    search, encoder, _ = _search([snapshot], [_record(snapshot, QUERY_VECTOR)])
    hits = await search.query_faq("邮费")
    assert encoder.inputs == ["邮费"]
    assert hits[0]["id"] == 1


async def test_results_follow_vector_order_and_mysql_text() -> None:
    """结果按向量顺序并回读 MySQL 正文。参数：无；返回：None；异常：无。"""
    first = _snapshot(1, "第一条权威答案。")
    second = _snapshot(2, "第二条权威答案。")
    third = _snapshot(3, "第三条权威答案。")
    search, _, _ = _search(
        [first, second, third],
        [
            _record(first, QUERY_VECTOR),
            _record(second, [0.8, 0.6] + [0.0] * 1022),
            _record(third, [0.0, 1.0] + [0.0] * 1022),
        ],
    )
    hits = await search.query_faq("邮费")
    assert [row["id"] for row in hits] == [1, 2, 3]
    assert set(hits[0]) == {"id", "question", "answer", "category"}
    assert hits[0]["answer"] == "第一条权威答案。"


@pytest.mark.parametrize("case", ["inactive", "mismatched"])
async def test_inactive_or_mismatched_hits_never_return(case: str) -> None:
    """失效/指纹失配命中不能被当作正常结果。参数：失配类型；返回：None；异常：无。"""
    if case == "inactive":
        snapshot = _snapshot(1, "不应返回。", active=False)
    else:
        snapshot = _snapshot(1, "不应返回。", vector_id="wrong-id")
    broken = _record(snapshot, QUERY_VECTOR)
    search, _, _ = _search([snapshot], [broken])
    with pytest.raises(KnowledgeSearchUnavailable):
        await search.query_faq("邮费")


async def test_backend_error_not_empty_hits() -> None:
    """索引故障不能伪装成空命中。参数：无；返回：None；异常：无。"""
    class _BrokenIndex:
        """搜索即失败的索引替身。"""

        async def search(self, vector, limit):
            """模拟索引不可用。参数：向量/数量；返回：不返回；异常：VectorIndexUnavailable。"""
            raise VectorIndexUnavailable("索引不可用")

    store = MemoryKnowledgeStore(snapshots=[_snapshot(1, "答案。")])
    search = KnowledgeSearch(store, FakeEncoder(), _BrokenIndex(), _policy())
    with pytest.raises(KnowledgeUnavailable):
        await search.query_faq("邮费")


async def test_unrelated_scores_empty_but_no_like_fallback() -> None:
    """低于校准阈值返回空，不调用旧 FAQ LIKE。参数：无；返回：None；异常：无。"""
    snapshot = _snapshot(1, "无关答案。")
    search, _, _ = _search(
        [snapshot],
        [_record(snapshot, [0.0, 1.0] + [0.0] * 1022)],
        min_score=0.9,
    )
    assert await search.query_faq("邮费") == []


@pytest.mark.parametrize("status", ["done", "pending"])
async def test_empty_index_with_active_knowledge_reports_sync_failure(status: str) -> None:
    """仍有有效原文时，空索引不能被当作正常未命中。

    参数：status 为原文的 done 或 pending 同步状态。
    返回：None，断言显式报告同步故障。
    异常：断言失败。
    """
    snapshot = _snapshot(1, "未满99元收6元。").model_copy(update={"vectorize_status": status})
    search, _, _ = _search([snapshot], [])
    with pytest.raises(KnowledgeSearchUnavailable):
        await search.query_faq("邮费")


@pytest.mark.parametrize("has_inactive", [False, True], ids=["empty-database", "only-withdrawn"])
async def test_empty_index_without_active_knowledge_remains_empty(has_inactive: bool) -> None:
    """真正空库或只有已撤回知识时，允许正常空结果。

    参数：has_inactive 是否保留一条已撤回原文。
    返回：None，断言查询没有误报同步故障。
    异常：断言失败。
    """
    snapshots = [_snapshot(1, "已撤回的答案。", active=False)] if has_inactive else []
    search, _, _ = _search(snapshots, [])
    assert await search.query_faq("邮费") == []


async def test_empty_index_cannot_hide_unavailable_source_status(monkeypatch) -> None:
    """无法核实原文状态时，不能把空索引包装成正常未命中。

    参数：monkeypatch 用于模拟原文状态读取故障。
    返回：None，断言原文读取错误向上层传播。
    异常：断言失败。
    """
    search, _, _ = _search([], [])

    async def unavailable_page(after_id: int, limit: int):
        """模拟原文读取失败。参数：游标、页大小；返回：不返回；异常：KnowledgeStoreUnavailable。"""
        raise KnowledgeStoreUnavailable("原文状态暂时不可用")

    monkeypatch.setattr(search.store, "active_page", unavailable_page)
    with pytest.raises(KnowledgeStoreUnavailable):
        await search.query_faq("邮费")


async def test_missing_vectors_report_failure_until_same_id_is_restored() -> None:
    """丢失向量后先报告故障，显式同步恢复后重新返回原知识。

    参数：无。
    返回：None，断言完整、丢失、恢复三个状态的查询结果。
    异常：断言失败。
    """
    snapshot = _snapshot(1, "未满99元收6元。")
    search, encoder, index = _search([snapshot], [_record(snapshot, QUERY_VECTOR)])
    assert [hit["id"] for hit in await search.query_faq("邮费")] == [1]
    # 只删除外部索引替身的数据，原文保持有效，真实检索和恢复逻辑继续运行。
    index.records.clear()
    with pytest.raises(KnowledgeSearchUnavailable):
        await search.query_faq("邮费")
    async with search.store.writer() as writer:
        await VectorSynchronizer(search.store, encoder, index).run(writer)
    hits = await search.query_faq("邮费")
    assert [hit["id"] for hit in hits] == [1]
    assert hits[0]["answer"] == "未满99元收6元。"


async def test_empty_index_reaches_chat_as_failed_tool_fact() -> None:
    """聊天层须保存并回传失败事实，不能生成成功的零命中事实。

    参数：无。
    返回：None，核对工具状态、持久化结果及回答阶段收到的事实。
    异常：断言失败。
    """
    search, _, _ = _search([_snapshot(1, "未满99元收6元。")], [])
    service, repository, _, model = make_chat_service(
        AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "邮费"}}]),
        chunks=["知识查询暂时不可用，请稍后重试。"],
    )
    service.executor.support = SupportRepository(None, search)
    events = [event async for event in service.stream_turn(ChatRequest(session_id="empty-index", message="邮费是多少"))]
    assert [event.data["status"] for event in events if event.event == "tool_status"] == ["running", "failed"]
    saved = next(iter(repository.results.values()))
    assert not saved.ok and saved.data is None
    answer_facts = [message for message in model.inputs[-1][1] if isinstance(message, ToolMessage)]
    assert len(answer_facts) == 1 and not ToolResult.model_validate_json(answer_facts[0].content).ok
    assert [phase for phase, _ in model.inputs] == ["decide", "answer"]
    assert events[-1].event == "done"


async def test_query_faq_rejects_invalid_limit_and_blank_keyword() -> None:
    """仓储层拒绝空白关键词、非法条数，并在无检索时失败关闭。参数：无；返回：None；异常：无。"""
    repository = SupportRepository(None)
    with pytest.raises(ValueError):
        await repository.query_faq("邮费", limit=0)
    with pytest.raises(ValueError):
        await repository.query_faq("   ")
    with pytest.raises(KnowledgeUnavailable):
        await repository.query_faq("邮费")


def test_public_tool_schema_and_envelope_unchanged() -> None:
    """query_faq 仍只暴露 keyword 参数。参数：无；返回：None；异常：无。"""
    schema = build_registry().get("query_faq").tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"keyword"}
