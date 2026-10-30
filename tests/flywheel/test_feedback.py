"""原轮反馈身份校验、严格请求和全部公开元数据投影测试。"""

import importlib
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.api.chat import get_chat_service
from app.models.customer_service import Message
from app.models.rag_quality import LowConfidenceQuestion
from app.repositories.chat import ChatRepository
from app.repositories.context import ContextRepository
from tests.flywheel.test_pool import start
from tests.workflow.service_fakes import service_runtime


async def test_feedback_targets_original_turn_not_latest(tmp_path):
    """按最近问题反馈会污染问题池；原问只能从服务器读取。参数：目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (_, deps, factory):
        first = await start(deps.chat, question="第一轮原问")
        await deps.chat.finish_turn(first, "原轮回答")
        latest = await start(deps.chat, turn="latest", question="最新轮原问")
        await deps.chat.finish_turn(latest, "最新回答")
        assert hasattr(deps.chat, "record_unresolved_feedback"), "缺少服务端原轮反馈入口"
        receipt = await deps.chat.record_unresolved_feedback("s", "u", "first")
        assert await deps.chat.record_unresolved_feedback("s", "u", "first") == receipt
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
            assert len(rows) == 1 and rows[0].raw_question == "第一轮原问" != latest.user_message
            assert receipt.pool_id == str(rows[0].id) and rows[0].retrieved_chunks is None


async def test_feedback_lost_commit_response_retry_returns_same_receipt(tmp_path):
    """提交响应丢失重发不能创建第二条。参数：目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        await deps.chat.finish_turn(ctx, "原答")
        class LostResponse:
            """在真实提交成功后中断一次反馈返回。"""
            @asynccontextmanager
            async def begin(self):
                """参数：无；返回：真实事务；异常：ConnectionError模拟响应丢失。"""
                async with factory.begin() as session:
                    yield session
                raise ConnectionError("lost response")
        assert hasattr(deps.chat, "record_unresolved_feedback"), "缺少反馈入池实现"
        with pytest.raises(ConnectionError):
            await ChatRepository(LostResponse()).record_unresolved_feedback("s", "u", "first")
        receipt = await deps.chat.record_unresolved_feedback("s", "u", "first")
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
            assert len(rows) == 1 and receipt.pool_id == str(rows[0].id)


async def test_feedback_api_rejects_foreign_unfinished_and_client_evidence(tmp_path):
    """客户端原话/快照、越权或半完成轮不能进入问题池。参数：目录；返回：None；异常：断言失败。"""
    assert importlib.util.find_spec("app.api.feedback"), "缺少反馈HTTP入口"
    async with service_runtime(tmp_path) as (service, deps, _):
        ctx = await start(deps.chat)
        app = FastAPI()
        app.include_router(importlib.import_module("app.api.feedback").router)
        app.dependency_overrides[get_chat_service] = lambda: service
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            path = "/api/chat/s/turns/first/feedback"
            assert (await client.post(path, json={"user_id": "other", "resolved": False})).status_code == 404
            assert (await client.post(path, json={"user_id": "u", "resolved": False})).status_code == 409
            for extra in ({"question": "伪造原问"}, {"retrieval_snapshot": {}}, {"resolved": True}, {"resolved": 0}):
                assert (await client.post(path, json={"user_id": "u", "resolved": False, **extra})).status_code == 422
            await deps.chat.finish_turn(ctx, "完成")
            first = await client.post(path, json={"user_id": "u", "resolved": False})
            assert first.status_code == 200
            assert (await client.post(path, json={"user_id": "u", "resolved": False})).json() == first.json()


async def test_internal_metadata_absent_from_all_public_responses(tmp_path):
    """历史列表/动作回读/完成重放若直接暴露内部JSON则失败。参数：目录；返回：None；异常：断言失败。"""
    from app.schemas import chat as schema
    assert hasattr(schema, "public_rag_metadata"), "缺少统一公开投影"
    private = {"retrieval_snapshot": {"secret": "正文"}, "knowledge_pool": {"pool_ids": {}}, "trace_identity": {"trace_id": "secret"}}
    async with service_runtime(tmp_path) as (service, deps, factory):
        ctx = await start(deps.chat)
        await deps.chat.finish_turn(ctx, "原答")
        async with factory.begin() as session:
            for row in (await session.scalars(select(Message))).all():
                row.rag_metadata = {**(row.rag_metadata or {}), **private}
        page = await ContextRepository(factory).list_messages("u", ctx.conversation_id)
        current = await service.get_current_actions("s", "u")
        original = await deps.chat.get_workflow_turn("s", "u", "first")
        replay = [event async for event in service._ticket_replay(None, original)]
        for metadata in [*(r["rag_metadata"] for r in page["items"]), current.completion_metadata,
                         next(event.data for event in replay if event.event == "answer_metadata"),
                         schema.public_rag_metadata({"status": "completed", **private})]:
            assert set(private).isdisjoint(metadata or {})


async def test_feedback_on_unmarked_old_turn_keeps_unknown_snapshot(tmp_path):
    """没有现代标记的旧轮不能被当成已确认没检索。参数：目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        await deps.chat.finish_turn(ctx, "旧答复")
        async with factory.begin() as session:
            source = await session.scalar(select(Message).where(Message.role == "user"))
            source.rag_metadata = None
        await deps.chat.record_unresolved_feedback("s", "u", "first")
        async with factory() as session:
            pool = await session.scalar(select(LowConfidenceQuestion))
            assert pool.retrieved_chunks["status"] == "legacy_unavailable"


async def test_public_tool_content_hides_internal_snapshot_but_keeps_original_storage(tmp_path):
    """公开工具JSON泄露内部快照或投影污染原库/内部历史时失败。参数：目录；返回：None；异常：断言失败。"""
    import json
    from app.schemas.flywheel import RetrievalSnapshot
    from app.schemas.tools import ToolResult, VerifiedKnowledgeOutcome
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        await deps.chat.save_tool_calls(ctx, [{"id": "faq", "name": "query_faq", "args": {"keyword": "政策"}}])
        result = ToolResult(ok=True, data={"message": "公开检索状态"}, verified_knowledge=VerifiedKnowledgeOutcome(
            evidence=[], gate={"passed": False}, evidence_scores=[], expanded_queries=["政策"], budget={},
            retrieval_snapshot=RetrievalSnapshot(status="legacy_unavailable")))
        await deps.chat.save_tool_result(ctx.model_copy(update={"tool_call_id": "faq"}), result)
        original = result.model_dump_json()
        page = await ContextRepository(factory).list_messages("u", ctx.conversation_id)
        public = json.loads(next(r["content"] for r in page["items"] if r["role"] == "tool"))
        assert "verified_knowledge" not in public and public["data"] == {"message": "公开检索状态"}
        async with factory() as session:
            assert (await session.scalar(select(Message).where(Message.role == "tool"))).content == original
        internal = await deps.chat.load_turns(ctx.conversation_id)
        assert next(r.content for r in internal[0].messages if r.role == "tool") == original


async def test_finish_preserves_explicit_unknown_snapshot_for_later_feedback(tmp_path):
    """已明确记录历史未知不能在无入池信号收尾时变成确认没检索。参数：目录；返回：None；异常：断言失败。"""
    from app.schemas.flywheel import RetrievalSnapshot
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        await deps.chat.save_retrieval_snapshot(ctx, RetrievalSnapshot(status="legacy_unavailable"))
        await deps.chat.finish_turn(ctx, "原答")
        await deps.chat.record_unresolved_feedback("s", "u", "first")
        async with factory() as session:
            pool = await session.scalar(select(LowConfidenceQuestion))
            assert pool.retrieved_chunks is not None and pool.retrieved_chunks["status"] == "legacy_unavailable"
