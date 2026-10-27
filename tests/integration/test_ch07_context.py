"""CH07真实MySQL验收：只操作mewhelp_ch04_test，按随机会话登记清理，不重建已有表。"""

import asyncio
import importlib
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, text

from app.config import Settings
from app.db.session import create_database
from app.models.customer_service import Conversation, Message
from tests.test_context_repository import seed_context

pytestmark = pytest.mark.mysql


@pytest.fixture
async def ch07_database():
    """准备增量结构并登记本例会话。参数：无；返回：factory/context；异常：连接或目标校验失败。"""
    migration = importlib.import_module("app.db.migrate_ch07")
    models = importlib.import_module("app.models.customer_service")
    settings = Settings(_env_file=Path(__file__).resolve().parents[2] / ".env")
    engine, factory = create_database(settings, database="mewhelp_ch04_test")
    session_id, user_id = "ch07-test-" + uuid4().hex, "ch07-test-owner"
    try:
        async with engine.begin() as conn:
            assert await conn.scalar(text("SELECT DATABASE()")) == "mewhelp_ch04_test"
            for table in (Conversation.__table__, Message.__table__):
                await conn.run_sync(lambda sync, table=table: table.create(sync, checkfirst=True))
            await conn.run_sync(lambda sync: migration.migrate_connection(sync, apply=True))
            assert await conn.run_sync(lambda sync: migration.migrate_connection(sync, apply=True)) == []
        ctx, _ = await seed_context(factory, session_id=session_id, user_id=user_id)
        yield factory, ctx
    finally:
        # 即使seed中途失败，也按事先登记的随机session及身份精确清理本例。
        async with factory.begin() as session:
            conv = await session.scalar(select(Conversation).where(Conversation.session_id == session_id))
            if conv is not None:
                assert conv.user_id == user_id
                turns = set((await session.scalars(select(Message.turn_id).where(Message.conversation_id == conv.id))).all())
                assert turns <= {"turn-0", "turn-1", "turn-2"}
                await session.execute(delete(models.ConversationSummary).where(models.ConversationSummary.conversation_id == conv.id))
                await session.execute(delete(Message).where(Message.conversation_id == conv.id))
                await session.execute(delete(Conversation).where(Conversation.id == conv.id))
        await engine.dispose()


async def test_mysql_competing_summaries_append_one_segment_and_keep_range(ch07_database):
    """真实行锁下两任务只落一段，继续降级不扩大冻结覆盖。参数：隔离库；返回：None；异常：断言失败。"""
    from app.repositories.context import ContextRepository
    from app.repositories.chat import ChatRepository
    factory, ctx = ch07_database
    turns = await ChatRepository(factory).load_turns(ctx.conversation_id)
    first_id, upto = turns[0].messages[0].id, turns[1].messages[-1].id
    storage = ContextRepository(factory)
    assert await storage.advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=turns[-1].messages[-1].id)
    first, second = await asyncio.gather(*[storage.append_summary(ctx.conversation_id,
        expected_summary_upto=0, from_msg_id=first_id, upto_msg_id=upto,
        content="订单问题待处理", projection_max_tokens=500) for _ in range(2)])
    assert first == second
    loaded = await storage.load_context(ctx.conversation_id)
    assert len(loaded["summaries"]) == 1 and loaded["summary_upto_msg_id"] == upto
    assert loaded["layer1_from_msg_id"] == turns[-1].messages[-1].id


async def test_mysql_summary_rollback_keeps_boundary(ch07_database):
    """真实数据库提交前故障不能留下半份摘要。参数：隔离库；返回：None；异常：断言失败。"""
    from contextlib import asynccontextmanager
    from app.repositories.context import ContextRepository
    from app.repositories.chat import ChatRepository
    factory, ctx = ch07_database
    turns = await ChatRepository(factory).load_turns(ctx.conversation_id)
    upto = turns[0].messages[-1].id
    await ContextRepository(factory).advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=upto)

    class FaultFactory:
        """真实写事务注入回滚，读取独立连接。"""

        def __call__(self):
            """参数：无；返回：session；异常：连接失败。"""
            return factory()

        @asynccontextmanager
        async def begin(self):
            """参数：无；返回：session；异常：提交前故障。"""
            async with factory.begin() as session:
                yield session
                raise ConnectionError("rollback")

    with pytest.raises(ConnectionError):
        await ContextRepository(FaultFactory()).append_summary(ctx.conversation_id, expected_summary_upto=0,
            from_msg_id=turns[0].messages[0].id, upto_msg_id=upto, content="原始事实", projection_max_tokens=500)
    loaded = await ContextRepository(factory).load_context(ctx.conversation_id)
    assert loaded["summary_upto_msg_id"] == 0 and loaded["summaries"] == []


async def test_mysql_manager_cancel_then_restart_from_committed_range(ch07_database):
    """真实MySQL上取消模型后重启按DB恢复，不依赖旧内存队列。参数：隔离库；返回：None；异常：断言或数据库错误。"""
    from app.repositories.chat import ChatRepository
    from app.repositories.context import ContextRepository
    from tests.test_summaries import ControlledGenerator, manager, wait_idle
    factory, ctx = ch07_database
    turns = await ChatRepository(factory).load_turns(ctx.conversation_id)
    upto = turns[-1].messages[-1].id
    await ContextRepository(factory).advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=upto)
    generator = ControlledGenerator(blocked=True)
    worker = manager(factory, generator, close_seconds=0.01)
    try:
        await worker.maybe_schedule(ctx.conversation_id)
        await asyncio.wait_for(generator.started.wait(), 3)
    finally:
        await worker.close()
    assert (await ContextRepository(factory).load_context(ctx.conversation_id))["summary_upto_msg_id"] == 0
    resumed = manager(factory, ControlledGenerator())
    try:
        await resumed.recover()
        await wait_idle(resumed)
        loaded = await ContextRepository(factory).load_context(ctx.conversation_id)
        assert loaded["summary_upto_msg_id"] == upto and len(loaded["summaries"]) == 1
    finally:
        await resumed.close()
