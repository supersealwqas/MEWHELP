"""CH06真实MySQL事务验收：仅mewhelp_ch04_test，随机会话，按登记主键清理。"""

import asyncio
from pathlib import Path
from uuid import uuid4
import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import DBAPIError
from app.config import Settings
from app.db.session import create_database
from app.db.migrate_ch06 import migrate_connection
from app.models.customer_service import Conversation, Message, Ticket
from app.repositories.actions import ActionRepository
from app.schemas.workflow import OrderSelectionConflict
from tests.workflow.test_order_wait_state import prepare_wait, selection

pytestmark = pytest.mark.mysql


@pytest.fixture
async def ch06_database():
    """创建缺失结构，登记随机会话后只清理本例数据。

    参数：无；返回：factory及唯一session_id。
    异常：数据库不可用或越过测试库边界时失败，不回退应用库、不drop_all。
    """
    config = Settings(_env_file=Path(__file__).resolve().parents[2] / ".env")
    engine, factory = create_database(config, database="mewhelp_ch04_test")
    session_id = "ch06-test-" + uuid4().hex
    registered = None
    try:
        async with engine.begin() as conn:
            assert await conn.scalar(text("SELECT DATABASE()")) == "mewhelp_ch04_test"
            for table in (Conversation.__table__, Message.__table__, Ticket.__table__):
                await conn.run_sync(lambda sync, table=table: table.create(sync, checkfirst=True))
            await conn.run_sync(lambda sync: migrate_connection(sync, apply=True))
            assert await conn.run_sync(lambda sync: migrate_connection(sync, apply=True)) == []
        async with factory() as session:
            assert await session.scalar(select(Conversation.id).where(Conversation.session_id == session_id)) is None
        chat, ctx, action = await prepare_wait(factory, session_id)
        registered = ctx.conversation_id
        yield factory, chat, ctx, action
    finally:
        if registered is not None:
            async with factory.begin() as session:
                row = await session.get(Conversation, registered)
                assert row.session_id == session_id and row.user_id == "demo-user"
                turns = set((await session.scalars(select(Message.turn_id).where(Message.conversation_id == registered))).all())
                assert turns <= {"original"}, "未登记轮次禁止清理"
                await session.execute(delete(Message).where(Message.conversation_id == registered))
                await session.execute(delete(Conversation).where(Conversation.id == registered))
        await engine.dispose()


async def test_mysql_order_wait_constraints_and_duplicate_accept(ch06_database):
    """真实CHECK拒坏值，两个并发确认只固定一个订单。参数：ch06_database隔离库；返回：None；异常：断言失败。"""
    factory, chat, ctx, action = ch06_database
    await chat.save_order_wait(ctx, action)
    with pytest.raises(DBAPIError):
        async with factory.begin() as session:
            await session.execute(update(Message).where(Message.conversation_id == ctx.conversation_id).values(turn_status="bad"))
    repo = ActionRepository(factory)
    first, second = await asyncio.gather(*[repo.accept_order_selection(selection(ctx, action)) for _ in range(2)])
    assert first == second and first.accepted_order_id == "1001"
    with pytest.raises(OrderSelectionConflict):
        await repo.accept_order_selection(selection(ctx, action, order_id="1002"))


async def test_mysql_accept_cancel_has_single_winner(ch06_database):
    """真实行锁下取消和接受只有一方获胜。参数：ch06_database隔离库；返回：None；异常：断言失败。"""
    factory, chat, ctx, action = ch06_database
    await chat.save_order_wait(ctx, action)
    result = await asyncio.gather(ActionRepository(factory).accept_order_selection(selection(ctx, action)),
                                  chat.cancel_order_wait(ctx.session_id, ctx.user_id), return_exceptions=True)
    async with factory() as session:
        row = await session.scalar(select(Message).where(Message.conversation_id == ctx.conversation_id))
        saved = row.rag_metadata["order_selection"]
        if isinstance(result[0], OrderSelectionConflict):
            assert row.turn_status == saved["status"] == "cancelled"
        else:
            assert not isinstance(result[0], Exception)
            assert row.turn_status == "running" and saved["status"] == "accepted"


async def test_mysql_accept_rollback_preserves_wait(ch06_database):
    """提交前异常回滚不能残留accepted。参数：ch06_database隔离库；返回：None；异常：断言失败。"""
    from contextlib import asynccontextmanager
    factory, chat, ctx, action = ch06_database
    await chat.save_order_wait(ctx, action)

    class RollbackFactory:
        """真实事务在业务方法返回前注入故障。"""

        @asynccontextmanager
        async def begin(self):
            """参数：无；返回：真实session；异常：RuntimeError强制回滚。"""
            async with factory.begin() as session:
                yield session
                raise RuntimeError("injected rollback")

    with pytest.raises(RuntimeError):
        await ActionRepository(RollbackFactory()).accept_order_selection(selection(ctx, action))
    async with factory() as session:
        row = await session.scalar(select(Message).where(Message.conversation_id == ctx.conversation_id))
        assert row.turn_status == row.rag_metadata["order_selection"]["status"] == "awaiting_input"
    accepted = await ActionRepository(factory).accept_order_selection(selection(ctx, action))
    assert accepted.accepted_order_id == "1001"
