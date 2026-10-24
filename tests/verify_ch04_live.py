"""CH04真实链路验收：仅在具名测试资源中记录SSE、数据库与引用证据。

使用 --test 显式选定 mewhelp_ch04_test，--prepare 仅向空知识表导入真实
data/kb；运行记录、随机会话、问题池与工单全部保留，不删除旧数据。
订单和物流沿用项目的演示数据源，报告不把它们称为真实商家业务数据。
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DATABASE = "mewhelp_ch04_test"
TEST_COLLECTION = "knowledge_ch04_live"
MANIFEST = ROOT / "data/kb/manifest.json"
EVIDENCE_FIELDS = ("n", "chunk_id", "question", "answer", "section_path")


def inspect_exchange(case: dict, observed: dict) -> dict:
    """独立检查逐轮流、落库和模型证据，不改变采样。

    参数：case 固定案例；observed JSON兼容的真实采样。
    返回：dict，保留原证据并附完整失败码列表，模型失败不转换为通过。
    异常：畸形采样类型由采样调用者捕获并留作逐例错误。
    """
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL
    from app.schemas.tools import ToolResult

    failures = list(observed.get("failures", []))
    events, messages = observed.get("events", []), observed.get("messages", [])
    answer = "".join(event["data"]["delta"] for event in events if event.get("event") == "delta"
                     and isinstance(event.get("data", {}).get("delta"), str))
    if observed.get("http_status") != 200 or not observed.get("content_type", "").startswith("text/event-stream"):
        failures.append("invalid_http_sse")
    if not answer.strip():
        failures.append("missing_answer_delta")
    if any(event.get("event") not in {"delta", "tool_status", "answer_metadata", "done"} for event in events):
        failures.append("error_or_unknown_sse_event")
    if [i for i, event in enumerate(events) if event.get("event") == "done"] != [len(events) - 1]:
        failures.append("missing_or_misordered_done")
    metadata_events = [(i, event.get("data")) for i, event in enumerate(events)
                       if event.get("event") == "answer_metadata"]
    metadata = metadata_events[0][1] if metadata_events and isinstance(metadata_events[0][1], dict) else {}
    if len(metadata_events) != 1 or metadata_events[0][0] != len(events) - 2:
        failures.append("missing_or_misordered_metadata")
    if (metadata.get("status") != "completed" or metadata.get("session_id") != observed.get("session_id")
            or not isinstance(metadata.get("citations"), list) or not metadata.get("turn_id")):
        failures.append("invalid_metadata_identity")
    users = [row for row in messages if row.get("role") == "user"]
    finals = [row for row in messages if row.get("role") == "assistant" and not row.get("tool_calls")]
    if len(users) != 1 or users[0].get("content") != case["query"]:
        failures.append("persisted_question_mismatch")
    if not messages or any(row.get("turn_status") != "completed" or row.get("turn_id") != metadata.get("turn_id") for row in messages):
        failures.append("persisted_turn_mismatch")
    if len(finals) != 1 or finals[0].get("content") != answer:
        failures.append("persisted_answer_mismatch")
    if len(finals) != 1 or finals[0].get("rag_metadata") != metadata:
        failures.append("persisted_metadata_mismatch")

    calls = [call for row in messages for call in (row.get("tool_calls") or [])]
    if case["kind"] in {"knowledge", "unknown_model"}:
        # 原业务允许以不同关键词继续查FAQ；验收按实际配置限额检查，不要求恰好一次。
        limit = observed.get("tool_max_calls_per_turn")
        valid_path = (type(limit) is int and 1 <= len(calls) <= limit
                      and all(call.get("name") == "query_faq" for call in calls))
    else:
        valid_path = [call.get("name") for call in calls] == case.get("tools", [])
    if not valid_path:
        failures.append("unexpected_tool_path")
    tools = [row for row in messages if row.get("role") == "tool"]
    if len(tools) != len(calls) or len({call.get("id") for call in calls}) != len(calls):
        failures.append("tool_result_pairing_mismatch")
    pairs = []
    for call in calls:
        paired = [row for row in tools if row.get("tool_call_id") == call.get("id")]
        if len(paired) != 1:
            failures.append("tool_result_pairing_mismatch")
            continue
        try:
            payload = ToolResult.model_validate_json(paired[0]["content"])
            expected_mock = call["name"] in {"query_order", "query_logistics", "query_product"}
            if not payload.ok or payload.error is not None or payload.is_mock != expected_mock:
                failures.append("tool_failed_or_wrong_data_origin")
            pairs.append({"call": call, "result": payload.model_dump(mode="json")})
        except (ValueError, TypeError):
            failures.append("invalid_tool_result")
        succeeded = [i for i, event in enumerate(events) if event.get("event") == "tool_status"
                     and event.get("data", {}).get("tool_call_id") == call.get("id")
                     and event["data"].get("name") == call.get("name") and event["data"].get("status") == "succeeded"]
        deltas = [i for i, event in enumerate(events) if event.get("event") == "delta"]
        if len(succeeded) != 1 or not deltas or succeeded[0] >= min(deltas):
            failures.append("tool_success_not_before_answer")
    for event in events:
        if event.get("event") == "tool_status" and not any(
                call.get("id") == event["data"].get("tool_call_id") and call.get("name") == event["data"].get("name") for call in calls):
            failures.append("unknown_tool_status")

    citations = metadata.get("citations", [])
    checks = observed.get("model_checks", [])
    if checks:
        actual = checks[-1].get("evidence", [])
        numbers = [item.get("n") for item in actual]
        # Prompt按首尾排列、快照按稳定编号保存；只在核验时排序，不改写原始采样。
        # 完整数量和编号唯一性仍是门禁，布尔值不能冒充整数引用编号。
        if (len(checks) != 1 or len(actual) != len(citations)
                or any(type(number) is not int for number in numbers)
                or len(set(numbers)) != len(actual)
                or any(any(expected.get(key) != saved.get(key) for key in EVIDENCE_FIELDS)
                       for expected, saved in zip(sorted(actual, key=lambda item: item["n"]), citations))):
            failures.append("model_evidence_mismatch")
    elif citations:
        failures.append("model_evidence_mismatch")
    if [item.get("n") for item in citations] != list(range(1, len(citations) + 1)):
        failures.append("invalid_citation_numbering")
    for citation in citations:
        if citation.get("content_hash") != hashlib.sha256(citation.get("answer", "").encode()).hexdigest():
            failures.append("citation_content_hash_mismatch")
    responses = observed.get("citation_responses", [])
    if [item.get("n") for item in responses] != [item.get("n") for item in citations]:
        failures.append("citation_api_coverage_mismatch")
    for citation, response in zip(citations, responses):
        if response.get("http_status") != 200 or response.get("snapshot") != citation:
            failures.append("citation_snapshot_mismatch")
        expected, source = response.get("expected_source"), response.get("source") or {}
        if (not expected or response.get("source_status") != 200
                or source.get("content") != expected.get("content")
                or source.get("source_revision") != expected.get("revision")
                or citation.get("source_revision") != expected.get("revision")
                or source.get("section_path") != citation.get("section_path")
                or source.get("section_line") != expected.get("section_line")
                or not source.get("section_line") or source.get("source_changed") is not False):
            failures.append("source_mismatch")
    access = observed.get("access_checks", [])
    if citations and not {"other_user", "other_user_source"}.issubset({item.get("kind") for item in access}):
        failures.append("missing_citation_access_checks")
    if any(item.get("http_status") != 404 for item in access):
        failures.append("citation_access_not_404")

    decision = checks[-1].get("decision") or {} if checks else {}
    pool = observed.get("pool_rows", [])
    if decision.get("useful") is False:
        reason = (decision.get("reason") or "").strip() or "模型自评现有证据不足"
        if (len(pool) != 1 or pool[0].get("raw_question") != case["query"]
                or pool[0].get("source") != "self_check" or pool[0].get("reason") != reason
                or not answer.endswith(FIXED_KNOWLEDGE_REFUSAL)):
            failures.append("self_check_pool_mismatch")
    elif pool:
        failures.append("pool_without_useful_false")
    if case["kind"] == "unknown_model" and (decision.get("useful") is not False or answer != FIXED_KNOWLEDGE_REFUSAL):
        failures.append("unknown_model_not_refused")
    if case["kind"] == "knowledge" and (decision.get("useful") is not True or not citations
                                         or not re.search(r"\[\d+\]", answer)):
        failures.append("knowledge_not_answered_with_citations")
    for pattern in case.get("answer_patterns", []):
        if not re.search(pattern, answer, re.IGNORECASE):
            failures.append("missing_answer_fact:" + pattern)
    if case["kind"] == "logistics":
        from app.tools.mock_data import logistics_snapshot, order_snapshot

        expected_order = order_snapshot(case["order_id"])
        expected_logistics = logistics_snapshot(expected_order["tracking_no"])
        if ([pair["result"].get("data") for pair in pairs] != [expected_order, expected_logistics]
                or len(calls) != 2 or calls[1].get("args", {}).get("tracking_no") != expected_order["tracking_no"]):
            failures.append("order_logistics_fact_mismatch")
    tickets = observed.get("tickets", [])
    if case["kind"] == "ticket":
        data = pairs[0]["result"].get("data") or {} if pairs else {}
        if (len(tickets) != 1 or tickets[0].get("ticket_no") != data.get("ticket_no")
                or tickets[0].get("status") != "待处理" or tickets[0].get("ticket_type") != "咨询"
                or not data.get("ticket_no") or data["ticket_no"] not in answer or "待处理" not in answer):
            failures.append("ticket_receipt_mismatch")
    elif tickets:
        failures.append("unexpected_ticket_created")
    return {**observed, "answer": answer, "tool_pairs": pairs, "failures": list(dict.fromkeys(failures))}


def validate_resources(config, *, test: bool) -> None:
    """限制具名测试库和独占集合。参数：config 最终配置、test 显式开关；返回：None；异常：ValueError，越界。"""
    if (not test or config.mysql_database != TEST_DATABASE or config.milvus_database != TEST_DATABASE
            or config.milvus_hybrid_collection != TEST_COLLECTION or config.milvus_collection == TEST_COLLECTION
            or config.knowledge_retrieval_mode != "hybrid"):
        raise ValueError("真实验收必须显式使用--test、mewhelp_ch04_test和knowledge_ch04_live")


def parser() -> argparse.ArgumentParser:
    """定义显式运行入口。参数：无；返回：ArgumentParser；异常：无。"""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--test", action="store_true", required=True, help="明确只操作mewhelp_ch04_test")
    result.add_argument("--prepare", action="store_true", help="显式补齐兼容表、冻结真实资料和专用索引，然后验收")
    result.add_argument("--env-file", type=Path, default=ROOT / ".env")
    result.add_argument("--report-dir", type=Path, required=True, help="必须是尚不存在的新报告目录")
    for name in ("mysql-host", "mysql-user", "milvus-uri", "knowledge-model-path", "rerank-model-path", "rerank-device",
                 "chat-model", "chat-base-url"):
        result.add_argument("--" + name)
    result.add_argument("--mysql-port", type=int)
    result.add_argument("--token-budget", type=int, help="显式覆盖会话消息估算预算，包括旧CHAT_TOKEN_BUDGET配置")
    result.add_argument("--request-token-budget", type=int, help="显式覆盖含提示词和工具定义的完整请求估算预算")
    result.add_argument("--rerank-min-score", type=float)
    result.add_argument("--knowledge-timeout-seconds", type=float)
    result.add_argument("--http-timeout-seconds", type=float, default=300)
    return result


def build_config(args):
    """加载env后应用显式参数，库名仍固定。参数：args 命令行结果；返回：Settings；异常：配置/文件错误。"""
    from dotenv import load_dotenv

    if not args.env_file.is_file():
        raise ValueError("指定env文件不存在")
    # 首次导入app.config会创建单例，先补齐其必需凭证；参数随后显式传入，优先级高于env。
    load_dotenv(args.env_file, override=False)
    from app.config import Settings

    explicit = {key: value for key, value in vars(args).items() if key in Settings.model_fields and value is not None}
    explicit.update(mysql_database=TEST_DATABASE, milvus_database=TEST_DATABASE,
                    milvus_hybrid_collection=TEST_COLLECTION, milvus_collection="knowledge_live_legacy_unused",
                    knowledge_retrieval_mode="hybrid")
    config = Settings(_env_file=args.env_file, **explicit)
    validate_resources(config, test=args.test)
    if args.http_timeout_seconds <= 0:
        raise ValueError("HTTP超时必须为正数")
    return config


def verify_corpus(existing: list, expected: list) -> None:
    """拒绝覆盖异构、隐藏或已修改的知识记录。参数：existing 全部现有快照、expected 冻结原文；返回：None；异常：ValueError。"""
    from app.evaluation.policy import corpus_digest

    if existing and (any(not row.is_active for row in existing) or corpus_digest(existing) != corpus_digest(expected)):
        raise ValueError("专用测试库已有不同知识，拒绝覆盖或删除；请保留旧数据并另行处理资源归属")


def verify_vector_subset(chunks: list, fingerprints: dict) -> None:
    """核验既有向量归属。参数：权威语料与现有向量指纹；返回：None；异常：ValueError。"""
    expected = {row.id: (row.content_hash, row.embedding_revision) for row in chunks}
    if any(identifier not in expected or value != expected[identifier] for identifier, value in fingerprints.items()):
        raise ValueError("专用集合含外来或不同正文向量，禁止同步删除或覆盖旧数据")


def _check_tables(connection) -> None:
    """仅接受现有项目表及兼容列，防止同名库中外来结构。参数：connection 同步连接；返回：None；异常：ValueError。"""
    from sqlalchemy import inspect
    from app.models.customer_service import Base
    from app.models.knowledge import KnowledgeBase
    from app.models.rag_quality import QualityBase

    inspector = inspect(connection)
    expected = {**Base.metadata.tables, **KnowledgeBase.metadata.tables, **QualityBase.metadata.tables}
    tables = set(inspector.get_table_names())
    if tables - expected.keys():
        raise ValueError("专用测试库含外来表，拒绝接管")
    for name in tables:
        actual = {item["name"]: item for item in inspector.get_columns(name)}
        columns = {column.name: column for column in expected[name].columns}
        # 只允许显式引用迁移补齐新列，其余缺列/外来列不自行修复。
        if name == "messages" and "rag_metadata" not in actual:
            columns.pop("rag_metadata")
        if set(actual) != set(columns) or any(
                actual[key]["type"].as_generic()._type_affinity != column.type._type_affinity
                or getattr(actual[key]["type"], "enums", None) != getattr(column.type, "enums", None)
                for key, column in columns.items()):
            raise ValueError(f"现有{name}表结构不兼容，拒绝接管")


@asynccontextmanager
async def resource_lock(config, *, prepare: bool):
    """持有专用命名锁并核验实际数据库，绝不恢复旧会话或清库。

    参数：config 测试配置；prepare 是否允许创建不存在的空库。
    返回：异步连接，仅用于资源检查。
    异常：ValueError，资源忙/不存在/异构；数据库连接错误。
    """
    import anyio
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.db.session import create_database

    validate_resources(config, test=True)
    target, _ = create_database(config)
    server = create_async_engine(target.url._replace(database=None), connect_args={"connect_timeout": 5})
    lock_name = "mewhelp:ch04:live:exclusive"
    try:
        async with server.connect() as connection:
            if await connection.scalar(text("SELECT GET_LOCK(:name,0)"), {"name": lock_name}) != 1:
                raise ValueError("另一真实验收正在使用专用测试资源")
            try:
                exists = await connection.scalar(text("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME=:name"), {"name": TEST_DATABASE})
                if not exists:
                    if not prepare:
                        raise ValueError("专用测试库不存在，请显式--prepare")
                    await connection.execute(text("CREATE DATABASE `mewhelp_ch04_test` CHARACTER SET utf8mb4"))
                await connection.execute(text("USE `mewhelp_ch04_test`"))
                if await connection.scalar(text("SELECT DATABASE()")) != TEST_DATABASE:
                    raise ValueError("实际MySQL连接不是专用测试库")
                await connection.commit()
                # 服务器连接首次没有默认schema；另开具名连接让SQLAlchemy正确反射，不修改其内部缓存。
                async with target.connect() as target_connection:
                    if await target_connection.scalar(text("SELECT DATABASE()")) != TEST_DATABASE:
                        raise ValueError("具名MySQL连接不是专用测试库")
                    await target_connection.run_sync(_check_tables)
                    await target_connection.commit()
                    yield target_connection
            finally:
                with anyio.CancelScope(shield=True):
                    if connection.in_transaction():
                        await connection.rollback()
                    await connection.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": lock_name})
                    await connection.commit()
    finally:
        with anyio.CancelScope(shield=True):
            await server.dispose()
            await target.dispose()


async def prepare_resources(config, connection, *, prepare: bool) -> dict:
    """复用真实切分/存储/同步器，空表首次导入后按摘要冻结，不走LLM知识合并。

    参数：config 配置；connection 已验证持锁连接；prepare 显式建表/导入许可。
    返回：dict，语料摘要、数量与同步结果。
    异常：ValueError，已有异构知识/未准备/向量不一致；迁移/模型/数据库错误。
    """
    import anyio
    from sqlalchemy import select
    from app.db.migrate_citations import migrate_citation_metadata
    from app.db.migrate_rag_quality import migrate_quality_tables
    from app.db.session import create_database
    from app.evaluation.policy import corpus_digest
    from app.evaluation.runtime import local_corpus, _verify_vectors
    from app.knowledge.embedding import BgeM3Encoder
    from app.knowledge.hybrid_index import MilvusHybridIndex
    from app.knowledge.runtime import knowledge_storage
    from app.knowledge.store import _snapshot_from_row
    from app.knowledge.sync import VectorSynchronizer
    from app.models.customer_service import Base
    from app.models.knowledge import KnowledgeBase, KnowledgeChunk

    if prepare:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(KnowledgeBase.metadata.create_all)
        await connection.commit()
        await migrate_citation_metadata(config)
        await migrate_quality_tables(config)
    encoder, index = BgeM3Encoder(config), MilvusHybridIndex(config)
    engine, factory = create_database(config)
    try:
        expected = local_corpus(MANIFEST, count_tokens=encoder.count_tokens)
        sync = None
        async with knowledge_storage(config) as store:
            async with store.writer() as writer:
                async with factory.begin() as session:
                    existing = [_snapshot_from_row(row) for row in (await session.scalars(select(KnowledgeChunk))).all()]
                    verify_corpus(existing, expected)
                    if not existing:
                        if not prepare:
                            raise ValueError("专用知识表为空，请显式--prepare")
                        for row in expected:
                            session.add(KnowledgeChunk(source_key=row.source_key, ordinal=row.ordinal,
                                category=row.text.category, questions=row.text.questions, answer=row.text.answer,
                                section_path=row.section_path, content_type=row.content_type, is_key_clause=row.is_key_clause,
                                content_hash=row.content_hash, embedding_revision=row.embedding_revision,
                                is_active=True, vectorize_status="pending"))
                await index.ensure_ready(create=prepare, confirmed_database=TEST_DATABASE if prepare else None)
                async with factory() as session:
                    actual = [_snapshot_from_row(row) for row in (await session.scalars(select(KnowledgeChunk))).all()]
                # 原同步器会清理孤立向量；先拒绝任何外来ID/异文指纹，确保这里不会触碰旧数据。
                async for identifiers in index.iter_ids(500):
                    fingerprints = await index.get_fingerprints(identifiers)
                    verify_vector_subset(actual, {identifier: fingerprints.get(identifier) for identifier in identifiers})
                if prepare:
                    sync = await VectorSynchronizer(store, encoder, index).run(writer)
                async with factory() as session:
                    actual = [_snapshot_from_row(row) for row in (await session.scalars(select(KnowledgeChunk))).all()]
                verify_corpus(actual, expected)
                await _verify_vectors(index, actual)
        return {"corpus_digest": corpus_digest(actual), "chunk_count": len(actual),
                "sync": sync.model_dump(mode="json") if sync is not None else None}
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await encoder.aclose()
            finally:
                try:
                    await index.aclose()
                finally:
                    await engine.dispose()


@asynccontextmanager
async def live_runtime(config):
    """装配原服务并记录真实自评输入输出，模型不替换、不强迫通过。

    参数：config 已验证的最终测试配置。
    返回：服务、会话工厂和每轮模型证据记录。
    异常：在线检索/模型/数据库异常原样传播，退出释放资源。
    """
    from app.core import llm
    from app.db.session import create_database
    from app.knowledge.runtime import knowledge_storage, online_search_runtime
    from app.repositories.chat import ChatRepository
    from app.repositories.support import SupportRepository
    from app.services.chat import ChatService
    from app.tools.registry import build_registry

    class RecordingModel(llm.ChatModelAdapter):
        """只观测真实模型调用的输入和自评，用于证明useful=false入池条件。"""

        def __init__(self):
            """初始化逐轮采样容器。参数：无；返回：None；异常：无。"""
            self.checks = []  # 当前唯一串行请求的实际自评记录，不记录上游凭证。

        async def answer_knowledge(self, messages):
            """记录实际请求中的编号知识，随后调用原模型。参数：messages 完整真实输入；返回：KnowledgeAnswer；异常：原模型错误。"""
            evidence = {}
            for message in messages:
                if getattr(message, "type", None) != "tool":
                    continue
                payload = json.loads(message.content)
                for hit in (payload.get("data") or {}).get("hits", []):
                    if "n" in hit:
                        evidence.setdefault(hit["id"], {"n": hit["n"], "chunk_id": hit["id"], "question": hit["question"],
                            "answer": hit["answer"], "section_path": hit.get("section_path"), "source_key": hit.get("source_key")})
            record = {"evidence": list(evidence.values()), "decision": None}
            self.checks.append(record)
            decision = await super().answer_knowledge(messages)
            record["decision"] = decision.model_dump(mode="json")
            return decision

    engine, factory = create_database(config)
    original = llm.settings
    # LLM工厂使用模块配置；必须替换为CLI已覆盖的实例，避免只给数据库更新配置。
    llm.settings = config
    try:
        async with knowledge_storage(config) as store, online_search_runtime(config, store) as search:
            model = RecordingModel()
            service = ChatService(ChatRepository(factory), SupportRepository(factory, search), model, build_registry(), config)
            yield SimpleNamespace(service=service, factory=factory, model=model)
    finally:
        llm.settings = original
        import anyio

        with anyio.CancelScope(shield=True):
            await engine.dispose()


def live_cases() -> list[dict]:
    """定义最小真实验收集，订单选择已发货演示样本以覆盖两步查询。

    参数：无。
    返回：五个固定目的案例，业务数据来源在kind和说明中明确。
    异常：StopIteration，项目演示订单全部无物流号。
    """
    from app.tools.mock_data import order_snapshot

    order = next(order_snapshot(str(number)) for number in range(1001, 1008)
                 if order_snapshot(str(number))["tracking_no"])
    return [
        {"id": "greeting", "kind": "greeting", "query": "你好", "tools": []},
        {"id": "document_citation", "kind": "knowledge", "query": "MH-W20水箱容量是多少？", "tools": ["query_faq"],
         "answer_patterns": [r"(?<!\d)2\s*(?:L|升)"]},
        {"id": "unknown_model", "kind": "unknown_model", "query": "MH-W99水箱容量是多少？", "tools": ["query_faq"]},
        {"id": "order_logistics", "kind": "logistics", "order_id": order["order_id"],
         "query": f"请查询订单{order['order_id']}的物流单号和最新物流状态。", "tools": ["query_order", "query_logistics"],
         "answer_patterns": [re.escape(order["tracking_no"])]},
        {"id": "create_ticket", "kind": "ticket", "query": "请帮我创建咨询工单，问题是会员规则需要人工解释。", "tools": ["create_ticket"]},
    ]


async def sample_exchange(runtime, url: str, case: dict, *, session_id: str, documents: dict, timeout: float) -> dict:
    """通过本机TCP执行真实聊天，回读仅当前随机会话并请求引用API。

    参数：runtime 真实服务；url 独占TCP地址；case 案例；session_id 本次随机标识；documents 冻结全文；timeout HTTP秒数。
    返回：dict，包含原始事件、消息、模型自评、问题池、工单和HTTP引用证据。
    异常：网络/采样错误转为逐例失败码，不丢弃已经得到的证据。
    """
    import httpx
    from sqlalchemy import select
    from app.knowledge.sources import section_line
    from app.models.customer_service import Conversation, Message, Ticket
    from app.models.rag_quality import LowConfidenceQuestion
    from tests.ch03_live_runtime import parse_frame

    observed = {"case": case, "session_id": session_id, "user_id": "demo-user", "transport": "localhost TCP /api/chat",
                "events": [], "messages": [], "model_checks": [], "pool_rows": [], "tickets": [],
                "citation_responses": [], "access_checks": [], "failures": [],
                "tool_max_calls_per_turn": runtime.service.settings.tool_max_calls_per_turn}
    runtime.model.checks.clear()
    started, lines = time.monotonic(), []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10), trust_env=False) as client:
            async with client.stream("POST", url + "/api/chat", json={"session_id": session_id, "user_id": "demo-user", "message": case["query"]}) as response:
                observed.update(http_status=response.status_code, content_type=response.headers.get("content-type", ""))
                async for line in response.aiter_lines():
                    if line:
                        lines.append(line)
                    elif lines:
                        observed["events"].append(parse_frame(lines, time.monotonic() - started))
                        lines = []
                if lines:
                    observed["failures"].append("unterminated_sse_frame")
    except Exception as error:
        # 上游网络异常可能含鉴权内容，只保留类型；服务端SSE已提供安全错误说明。
        observed["failures"].append("http_sse_error:" + type(error).__name__)
    observed["elapsed_seconds"] = time.monotonic() - started
    observed["model_checks"] = json.loads(json.dumps(runtime.model.checks, ensure_ascii=False))
    try:
        async with runtime.factory() as session:
            conversation = await session.scalar(select(Conversation).where(Conversation.session_id == session_id))
            if conversation is not None:
                observed["conversation_id"] = conversation.id
                if conversation.user_id != "demo-user":
                    observed["failures"].append("conversation_owner_mismatch")
                messages = (await session.scalars(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id))).all()
                observed["messages"] = [{name: getattr(row, name) for name in
                    ("id", "turn_id", "role", "content", "tool_calls", "tool_call_id", "turn_status", "rag_metadata")} for row in messages]
                rows = (await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == conversation.id))).all()
                observed["pool_rows"] = [{name: getattr(row, name) for name in ("id", "raw_question", "source", "reason")} for row in rows]
                tickets = (await session.scalars(select(Ticket).where(Ticket.conversation_id == conversation.id))).all()
                observed["tickets"] = [{name: getattr(row, name) for name in ("ticket_no", "description", "ticket_type", "status")} for row in tickets]
    except Exception as error:
        observed["failures"].append("persistence_read_error:" + type(error).__name__)
    finals = [row for row in observed["messages"] if row["role"] == "assistant" and not row.get("tool_calls")]
    metadata = finals[-1].get("rag_metadata") or {} if finals else {}
    turn_id = metadata.get("turn_id")
    if turn_id:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10), trust_env=False) as client:
                path = f"{url}/api/chat/{session_id}/turns/{turn_id}/citations/"
                for citation in metadata.get("citations", []):
                    result = {"n": citation["n"]}
                    response = await client.get(path + str(citation["n"]), params={"user_id": "demo-user"})
                    result.update(http_status=response.status_code, snapshot=response.json())
                    source = await client.get(path + str(citation["n"]) + "/source", params={"user_id": "demo-user"})
                    result.update(source_status=source.status_code, source=source.json())
                    document = documents.get(citation.get("source_key"))
                    result["expected_source"] = ({**document, "section_line": section_line(document["content"], citation.get("section_path"))}
                                                 if document else None)
                    observed["citation_responses"].append(result)
                # 正常、跨用户、跨会话及未知编号均走相同公开路由；不存在不应泄露归属。
                probes = [("other_user", path + "1", "ch04-other-" + uuid4().hex),
                          ("other_user_source", path + "1/source", "ch04-other-" + uuid4().hex),
                          ("unknown_number", path + "999999", "demo-user"),
                          ("unknown_number_source", path + "999999/source", "demo-user"),
                          ("other_session", path.replace(session_id, "ch04-missing-" + uuid4().hex) + "1", "demo-user")]
                for kind, endpoint, user in probes:
                    response = await client.get(endpoint, params={"user_id": user})
                    observed["access_checks"].append({"kind": kind, "http_status": response.status_code})
        except Exception as error:
            observed["failures"].append("citation_api_error:" + type(error).__name__)
    try:
        return inspect_exchange(case, observed)
    except Exception as error:
        observed["failures"].append("verifier_error:" + type(error).__name__)
        return observed


def write_json(path: Path, value: dict) -> None:
    """原子替换当前运行报告，不覆盖其他运行目录。参数：path 输出文件、value JSON内容；返回：None；异常：文件错误。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


async def run(args, config, summary: dict) -> int:
    """顺序执行五例，每例立即落盘，结束继续保留全部测试记录。

    参数：args CLI参数；config 最终配置；summary 当前运行可原子保存的报告。
    返回：0全部通过，1至少一例失败；准备异常交由main写入报告后返回2。
    异常：准备/运行时启动/文件错误，main统一保留阶段和异常类型。
    """
    import os
    from app.evaluation.runtime import file_digest
    from app.knowledge.manifest import load_manifest
    from tests.ch03_live_runtime import tcp_server

    documents = {}
    for document in load_manifest(MANIFEST, root=MANIFEST.parent):
        raw = document.path.read_bytes()
        documents[document.source_key] = {"content": raw.decode("utf-8"), "revision": hashlib.sha256(raw).hexdigest()}
    write_json(args.report_dir / "sources.json", documents)
    summary["source_files"] = {path.relative_to(ROOT).as_posix(): file_digest(path)
        for path in [*sorted((ROOT / "app").rglob("*.py")), Path(__file__), ROOT / "uv.lock"]}
    summary["stage"] = "resources"
    write_json(args.report_dir / "summary.json", summary)
    async with resource_lock(config, prepare=args.prepare) as connection:
        summary["preparation"] = await prepare_resources(config, connection, prepare=args.prepare)
        summary["stage"] = "online_startup"
        write_json(args.report_dir / "summary.json", summary)
        async with live_runtime(config) as runtime, tcp_server(runtime) as url:
            summary["stage"] = "cases"
            with (args.report_dir / "cases.jsonl").open("x", encoding="utf-8") as output:
                for case in live_cases():
                    session_id = "ch04-live-" + uuid4().hex
                    # 请求发出前记录随机会话，意外中断也能追查可能已经创建的工单。
                    summary["current_case"] = {"id": case["id"], "session_id": session_id}
                    write_json(args.report_dir / "summary.json", summary)
                    row = await sample_exchange(runtime, url, case, session_id=session_id,
                                                documents=documents, timeout=args.http_timeout_seconds)
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    summary["results"].append({"id": case["id"], "session_id": session_id,
                        "conversation_id": row.get("conversation_id"), "passed": not row["failures"], "failures": row["failures"]})
                    write_json(args.report_dir / "summary.json", summary)
                    print(json.dumps({"case": case["id"], "passed": not row["failures"], "failures": row["failures"]}, ensure_ascii=False), flush=True)
    summary.pop("current_case", None)
    summary.update(status="completed", stage="completed", completed_at=datetime.now(timezone.utc).isoformat())
    summary["passed"] = sum(item["passed"] for item in summary["results"])
    summary["failed"] = len(summary["results"]) - summary["passed"]
    write_json(args.report_dir / "summary.json", summary)
    return 1 if summary["failed"] else 0


def main() -> int:
    """显式启动并记录失败阶段，不打印连接凭证。参数：命令行；返回：退出码；异常：目录/参数错误。"""
    args = parser().parse_args()
    # 新目录是执行前置条件：旧报告不会被下一轮通过结果覆盖。
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "stage": "configuration", "started_at": datetime.now(timezone.utc).isoformat(),
               "results": [], "retention": "保留本轮全部随机会话、消息、问题池与工单；不清库、不删除旧记录",
               "business_data_origin": "query_order/query_logistics沿用项目演示数据；知识与工单使用真实MySQL链路"}
    write_json(args.report_dir / "summary.json", summary)
    try:
        config = build_config(args)
        summary["config"] = {name: getattr(config, name) for name in ("mysql_host", "mysql_port", "mysql_database",
            "milvus_database", "milvus_hybrid_collection", "chat_model", "knowledge_model_path", "rerank_model_path",
            "rerank_device", "recall_top_k", "rerank_top_k", "subquery_split", "rerank_min_score", "knowledge_timeout_seconds",
            "token_budget", "request_token_budget", "tool_max_calls_per_turn")}
        summary["config"]["milvus_endpoint_digest"] = hashlib.sha256(config.milvus_uri.encode()).hexdigest()
        summary["config"]["chat_endpoint_digest"] = hashlib.sha256(config.chat_base_url.encode()).hexdigest()
        exit_code = asyncio.run(run(args, config, summary))
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else "error",
                       error_type=type(error).__name__, completed_at=datetime.now(timezone.utc).isoformat())
        trace = error.__traceback__
        while trace and trace.tb_next:
            trace = trace.tb_next
        # 只保存本脚本主动产生的受控校验文字，外部错误原文可能携带连接凭证。
        if isinstance(error, ValueError) and trace and Path(trace.tb_frame.f_code.co_filename) == Path(__file__):
            summary["error_message"] = str(error)
        write_json(args.report_dir / "summary.json", summary)
        exit_code = 2
    print(json.dumps({"status": summary["status"], "exit_code": exit_code, "report_dir": str(args.report_dir)}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
