"""CH07仓储事务与原文查询测试；SQLite验证提交契约，MySQL另验证行锁与DDL。"""

import importlib
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import text

from app.repositories.chat import ChatRepository
from app.schemas.tools import ToolContext
from tests.recovery_fakes import offline_recovery_database


@pytest.fixture
async def context_db(tmp_path):
    """提供独占离线库。参数：tmp_path测试目录；返回：factory；异常：数据库或模块缺失。"""
    async with offline_recovery_database(tmp_path) as (engine, factory):
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE conversation_summaries (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, seq INTEGER NOT NULL, from_msg_id INTEGER NOT NULL, upto_msg_id INTEGER NOT NULL, content TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, UNIQUE(conversation_id, seq))"))
        yield factory


async def seed_context(factory, *, session_id="ch07", user_id="owner", count=3):
    """通过真实仓储保存多轮。参数：factory、会话/用户、轮数；返回：context与完整轮；异常：数据库错误。"""
    chat = ChatRepository(factory)
    for index in range(count):
        ctx = await chat.start_turn(ToolContext(session_id=session_id, user_id=user_id,
            turn_id=f"turn-{index}", user_message=f"第{index}轮原问\n  "), f"第{index}轮原问\n  ")
        await chat.finish_turn(ctx, "完整客服原文" * 100)
    return ctx, await chat.load_turns(ctx.conversation_id)


def repo(factory):
    """构造真实仓储。参数：factory会话工厂；返回：仓储；异常：模块缺失。"""
    return importlib.import_module("app.repositories.context").ContextRepository(factory)


async def test_full_messages_have_real_ids_and_read_pagination_keeps_every_character(context_db):
    """主键缺失或历史回载使用截短视图会失败。参数：离线库；返回：None；异常：断言失败。"""
    ctx, turns = await seed_context(context_db)
    assert [m.id for turn in turns for m in turn.messages] == [1, 2, 3, 4, 5, 6]
    first = await repo(context_db).list_messages("owner", ctx.conversation_id, after_id=0, limit=3)
    second = await repo(context_db).list_messages("owner", ctx.conversation_id, after_id=first["next_after_id"], limit=3)
    items = first["items"] + second["items"]
    assert [row["id"] for row in items] == [1, 2, 3, 4, 5, 6]
    assert items[0]["content"] == "第0轮原问\n  " and items[1]["content"] == "完整客服原文" * 100
    assert second["next_after_id"] is None


async def test_summary_appends_only_frozen_range_and_preserves_previous_segments(context_db):
    """慢摘要不能越过冻结范围，旧段不能被后续重新生成覆盖。参数：离线库；返回：None；异常：断言失败。"""
    ctx, _ = await seed_context(context_db)
    storage = repo(context_db)
    assert await storage.advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=4)
    assert await storage.advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=6)
    first = await storage.append_summary(ctx.conversation_id, expected_summary_upto=0,
        from_msg_id=1, upto_msg_id=4, content="第一段订单诉求", projection_max_tokens=500)
    state = await storage.load_context(ctx.conversation_id)
    assert state["summary_upto_msg_id"] == 4 and state["layer1_from_msg_id"] == 6
    second = await storage.append_summary(ctx.conversation_id, expected_summary_upto=4,
        from_msg_id=5, upto_msg_id=6, content="第二段", projection_max_tokens=500)
    assert [first["seq"], second["seq"]] == [1, 2]
    state = await storage.load_context(ctx.conversation_id)
    assert [row["content"] for row in state["summaries"]] == ["第一段订单诉求", "第二段"]
    duplicate = await storage.append_summary(ctx.conversation_id, expected_summary_upto=0,
        from_msg_id=1, upto_msg_id=4, content="不应覆盖旧段", projection_max_tokens=500)
    assert duplicate == first
    assert state["summary"] == "第一段订单诉求\n第二段"


@pytest.mark.parametrize("upto", [1, 3, 99])
async def test_advancing_boundary_rejects_partial_turn_and_foreign_or_missing_id(context_db, upto):
    """只按数值范围移动会覆盖半轮或他会话。参数：离线库及坏边界；返回：None；异常：断言失败。"""
    ctx, _ = await seed_context(context_db)
    with pytest.raises(ValueError):
        await repo(context_db).advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=upto)
    assert (await repo(context_db).load_context(ctx.conversation_id))["layer1_from_msg_id"] == 0


async def test_waiting_turn_and_skipped_first_turn_cannot_be_covered(context_db):
    """等待轮和跳过首段都不能前移摘要。参数：离线库；返回：None；异常：断言失败。"""
    ctx, _ = await seed_context(context_db)
    storage = repo(context_db)
    async with context_db.begin() as session:
        await session.execute(text("UPDATE messages SET turn_status='awaiting_input' WHERE id IN (1,2)"))
    with pytest.raises(ValueError):
        await storage.advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=4)
    async with context_db.begin() as session:
        await session.execute(text("UPDATE messages SET turn_status='completed' WHERE id IN (1,2)"))
    await storage.advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=4)
    with pytest.raises(ValueError):
        await storage.append_summary(ctx.conversation_id, expected_summary_upto=0, from_msg_id=3,
            upto_msg_id=4, content="漏过首轮", projection_max_tokens=500)
    assert (await storage.load_context(ctx.conversation_id))["summaries"] == []


@pytest.mark.parametrize("committed", [False, True])
async def test_summary_transaction_rollback_and_lost_commit_response(context_db, committed):
    """不能只保存段或只移锚点；丢回执时只认真实回读。参数：离线库、committed；返回：None；异常：断言失败。"""
    ctx, _ = await seed_context(context_db)
    await repo(context_db).advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=4)

    class FaultFactory:
        """只在事务提交前或提交后注入异常，普通读取用真实新session。"""

        def __call__(self):
            """参数：无；返回：独立session；异常：连接失败。"""
            return context_db()

        @asynccontextmanager
        async def begin(self):
            """参数：无；返回：写事务；异常：模拟丢失回执或提交前错误。"""
            async with context_db.begin() as session:
                yield session
                if not committed:
                    raise ConnectionError("rollback")
            raise ConnectionError("lost response")

    operation = repo(FaultFactory()).append_summary(ctx.conversation_id, expected_summary_upto=0,
        from_msg_id=1, upto_msg_id=4, content="原始事实", projection_max_tokens=500)
    if committed:
        assert (await operation)["seq"] == 1
    else:
        with pytest.raises(ConnectionError):
            await operation
    state = await repo(context_db).load_context(ctx.conversation_id)
    assert state["summary_upto_msg_id"] == (4 if committed else 0)
    assert len(state["summaries"]) == (1 if committed else 0)


async def test_conversation_listing_is_owned_stable_and_reports_committed_summary(context_db):
    """侧栏不能越权，后台摘要不能把旧会话顶到前面。参数：离线库；返回：None；异常：断言失败。"""
    older, _ = await seed_context(context_db, session_id="older")
    newer, _ = await seed_context(context_db, session_id="newer")
    foreign, _ = await seed_context(context_db, session_id="foreign", user_id="stranger")
    storage = repo(context_db)
    await storage.advance_layer1(older.conversation_id, expected_summary_upto=0, upto_msg_id=2)
    await storage.append_summary(older.conversation_id, expected_summary_upto=0, from_msg_id=1,
        upto_msg_id=2, content="", projection_max_tokens=500)
    first = await storage.list_conversations("owner", before_id=None, limit=1)
    second = await storage.list_conversations("owner", before_id=first["next_before_id"], limit=1)
    assert first["items"][0]["id"] == newer.conversation_id
    assert second["items"][0]["id"] == older.conversation_id
    assert second["items"][0]["has_summary"] is True
    assert second["items"][0]["preview"] == "第0轮原问\n  "
    for identifier in (foreign.conversation_id, 999):
        with pytest.raises(LookupError):
            await storage.list_messages("owner", identifier, after_id=0, limit=10)
