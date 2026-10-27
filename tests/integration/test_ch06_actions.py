"""CH06真实MySQL事务验收：仅mewhelp_ch04_test，随机会话，按登记主键清理。"""

import asyncio
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL
import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import DBAPIError
from app.config import Settings
from app.db.session import create_database
from app.db.migrate_ch06 import migrate_connection
from app.models.customer_service import Conversation, Message, Ticket, RefundApplication
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
                refunds = (await session.scalars(select(RefundApplication).where(RefundApplication.conversation_id == registered))).all()
                expected_action = str(uuid5(NAMESPACE_URL, f"mewhelp:ch06:{registered}:original:submit_refund"))
                assert all(row.action_id == expected_action and row.source_turn_id == "original" for row in refunds)
                await session.execute(delete(RefundApplication).where(RefundApplication.conversation_id == registered,
                                                                     RefundApplication.action_id == expected_action))
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


async def prepare_refund(ch06_database):
    """保存完整查询配对和退款快照。参数：ch06_database安全夹具；返回：工厂及确认请求；异常：数据库错误。"""
    from app.schemas.workflow import RefundAction, RefundSubmission
    from app.schemas.tools import ToolResult
    factory, chat, ctx, _ = ch06_database
    call_id = f"ch06:order:{ctx.turn_id}:1001"
    await chat.save_tool_calls(ctx, [{"id": call_id, "name": "query_order", "args": {"order_id": "1001"}}], deduplicate=True)
    await chat.save_tool_result(ctx.model_copy(update={"tool_call_id": call_id}), ToolResult.success({"order_id": "1001"}))
    action = RefundAction(action_id=str(uuid5(NAMESPACE_URL, f"mewhelp:ch06:{ctx.conversation_id}:{ctx.turn_id}:submit_refund")),
                          source_turn_id=ctx.turn_id, question=ctx.user_message, order_id="1001")
    await chat.finish_turn(ctx, "本地申请待处理，不代表退款批准", refund_action=action,
                           workflow_metadata={"intent": "退款退货", "route": "order_policy", "order_id": "1001"})
    return factory, RefundSubmission(session_id=ctx.session_id, user_id=ctx.user_id,
        source_turn_id=ctx.turn_id, action_id=action.action_id, reason_code="quality_issue", confirmed=True)


async def test_mysql_refund_unique_constraint_and_concurrent_receipt(ch06_database):
    """真实并发与独立唯一键都只允许一份退款申请。参数：安全夹具；返回：None；异常：断言失败。"""
    factory, request = await prepare_refund(ch06_database)
    repo = ActionRepository(factory)
    first, second = await asyncio.gather(repo.submit_refund(request), repo.submit_refund(request))
    assert first == second and first.status == "待处理"
    async with factory() as session:
        rows = (await session.scalars(select(RefundApplication).where(RefundApplication.action_id == request.action_id))).all()
        assert len(rows) == 1
        row = rows[0]
        assert await session.scalar(select(Ticket.ticket_no).where(Ticket.conversation_id == row.conversation_id)) is None
        values = {column.name: getattr(row, column.name) for column in RefundApplication.__table__.columns}
    with pytest.raises(DBAPIError, match="uq_refund_action"):
        async with factory.begin() as session:
            session.add(RefundApplication(**{**values, "refund_no": "RF" + "0" * 32}))
    with pytest.raises(OrderSelectionConflict):
        await repo.submit_refund(request.model_copy(update={"reason_code": "other"}))


@pytest.mark.parametrize("committed", [False, True])
async def test_mysql_refund_rollback_and_lost_commit_response(ch06_database, committed):
    """回滚可重试，提交响应丢失回读同编号。参数：安全夹具、committed是否已提交；返回：None；异常：断言失败。"""
    from contextlib import asynccontextmanager
    factory, request = await prepare_refund(ch06_database)

    class FaultFactory:
        """只在写事务提交边界注入故障，回读使用真正新session。"""

        def __call__(self):
            """参数：无；返回：新session；异常：连接错误。"""
            return factory()

        @asynccontextmanager
        async def begin(self):
            """参数：无；返回：写session；异常：提交前/后ConnectionError。"""
            async with factory.begin() as session:
                yield session
                if not committed:
                    raise ConnectionError("rollback")
            raise ConnectionError("commit response lost")

    repo = ActionRepository(FaultFactory())
    if committed:
        first = await repo.submit_refund(request)
        assert first == await ActionRepository(factory).submit_refund(request)
    else:
        with pytest.raises(ConnectionError):
            await repo.submit_refund(request)
        async with factory() as session:
            assert await session.scalar(select(RefundApplication).where(RefundApplication.action_id == request.action_id)) is None
        assert (await ActionRepository(factory).submit_refund(request)).status == "待处理"
