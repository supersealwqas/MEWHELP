"""CH07本人会话只读API：真实仓储分页、完整原文及BIGINT边界，不接真实模型。"""

from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import text

from app.api.chat import get_chat_service
from app.main import create_app
from app.repositories.chat import ChatRepository
from app.repositories.context import ContextRepository
from tests.test_context_repository import context_db, seed_context


def application(factory):
    """只替换应用外部资源，API走真实仓储。参数：factory；返回：应用；异常：配置错误。"""
    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: SimpleNamespace(chat=ChatRepository(factory))
    return app


async def test_owned_list_and_full_messages_are_paginated_without_writes(context_db):
    """本人会话可遍历、后台摘要不重排、原文不截短且只读。参数：真实离线库；返回：None；异常：断言失败。"""
    old, _ = await seed_context(context_db, session_id="older")
    new, _ = await seed_context(context_db, session_id="newer")
    await seed_context(context_db, user_id="stranger", session_id="foreign")
    storage = ContextRepository(context_db)
    await storage.advance_layer1(old.conversation_id, expected_summary_upto=0, upto_msg_id=2)
    await storage.append_summary(old.conversation_id, expected_summary_upto=0, from_msg_id=1,
                                upto_msg_id=2, content="旧事实", projection_max_tokens=500)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application(context_db)), base_url="http://test") as client:
        first = await client.get("/api/conversations", params={"user_id":"owner", "limit":1})
        assert first.status_code == 200
        assert first.json()["items"][0]["id"] == str(new.conversation_id)
        second = await client.get("/api/conversations", params={"user_id":"owner", "limit":1,
                                                              "before_id":first.json()["next_before_id"]})
        assert second.json()["items"][0]["id"] == str(old.conversation_id)
        assert second.json()["items"][0]["has_summary"] is True
        assert second.json()["next_before_id"] is None
        rows, after = [], "0"
        while after is not None:
            response = await client.get(f"/api/conversations/{old.conversation_id}/messages",
                params={"user_id":"owner", "after_id":after, "limit":2})
            assert response.status_code == 200 and response.json()["session_id"] == "older"
            rows.extend(response.json()["items"])
            after = response.json()["next_after_id"]
        assert [row["id"] for row in rows] == [str(i) for i in range(1,7)]
        assert rows[0]["content"] == "第0轮原问\n  " and rows[1]["content"] == "完整客服原文" * 100
        assert all(row["turn_status"] == "completed" for row in rows)
        for identifier in (old.conversation_id, 999):
            response = await client.get(f"/api/conversations/{identifier}/messages", params={"user_id":"stranger-other"})
            assert response.status_code == 404 and response.json()["detail"] == "会话不存在或不可访问"
    async with context_db() as session:
        assert await session.scalar(text("SELECT COUNT(*) FROM conversations")) == 3
        assert await session.scalar(text("SELECT COUNT(*) FROM messages")) == 18
        assert await session.scalar(text("SELECT COUNT(*) FROM conversation_summaries")) == 1


async def test_bigint_and_metadata_survive_json_boundary(context_db):
    """ID超过JS安全整数仍精确，引用和状态完整保留。参数：真实离线库；返回：None；异常：断言失败。"""
    ctx, _ = await seed_context(context_db, count=1)
    large = 9007199254740993
    async with context_db.begin() as session:
        await session.execute(text("UPDATE conversations SET id=:large WHERE id=:old"), {"large":large,"old":ctx.conversation_id})
        await session.execute(text("UPDATE messages SET id=id+:large,conversation_id=:large,rag_metadata=:metadata"),
            {"large":large,"metadata":'{"citations":[{"n":1,"answer":"引用全文"}],"status":"completed"}'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application(context_db)), base_url="http://test") as client:
        listing = await client.get("/api/conversations", params={"user_id":"owner"})
        assert listing.json()["items"][0]["id"] == str(large)
        response = await client.get(f"/api/conversations/{large}/messages", params={"user_id":"owner", "limit":1})
        assert response.status_code == 200
        row = response.json()["items"][0]
        assert row["id"] == str(large+1) and row["conversation_id"] == str(large)
        assert response.json()["next_after_id"] == str(large+1)
        assert row["rag_metadata"]["citations"][0]["answer"] == "引用全文"


@pytest.mark.parametrize("identifier", ["-1", "1e2", "0", "18446744073709551616", "abc"])
async def test_invalid_ids_never_query_or_create_conversation(context_db, identifier):
    """拒绝非法主键，不从数值近似或空会话补救。参数：库/坏ID；返回：None；异常：断言失败。"""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application(context_db)), base_url="http://test") as client:
        response = await client.get(f"/api/conversations/{identifier}/messages")
        assert response.status_code == 422
    async with context_db() as session:
        assert await session.scalar(text("SELECT COUNT(*) FROM conversations")) == 0
