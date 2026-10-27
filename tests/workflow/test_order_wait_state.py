"""真实仓储的订单等待状态测试；SQLite证明事务契约，MySQL行锁另行验收。"""

from uuid import NAMESPACE_URL, uuid5
import pytest
from sqlalchemy import select
from app.models.customer_service import Message
from app.repositories.chat import ChatRepository, PersistenceUncertain
from app.repositories.actions import ActionRepository, ActionNotFound
from app.schemas.tools import ToolContext
from app.schemas import workflow
from app.tools.business import list_user_orders
from tests.recovery_fakes import offline_recovery_database


async def prepare_wait(factory, session_id="wait"):
    """创建原轮和稳定选单。参数：factory隔离仓储工厂；返回：仓储/上下文/动作；异常：数据库错误。"""
    chat = ChatRepository(factory)
    ctx = await chat.start_turn(ToolContext(session_id=session_id, user_id="demo-user", turn_id="original", user_message="退款"), "退款")
    action = workflow.OrderSelectionAction(type="select_order", action_id=str(uuid5(NAMESPACE_URL,
        f"mewhelp:ch06:{ctx.conversation_id}:{ctx.turn_id}:select_order")), source_turn_id=ctx.turn_id,
        orders=list_user_orders(ctx.user_id))
    return chat, ctx, action


def selection(ctx, action, **override):
    """创建严格请求。参数：ctx原轮、action快照、override覆盖；返回：OrderSelection；异常：请求非法。"""
    return workflow.OrderSelection(**{"session_id": ctx.session_id, "user_id": ctx.user_id,
        "source_turn_id": ctx.turn_id, "action_id": action.action_id, "order_id": "1001", **override})


async def test_save_wait_is_idempotent(tmp_path):
    """保存同等待两次不丢元数据。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    async with offline_recovery_database(tmp_path) as (_, factory):
        chat, ctx, action = await prepare_wait(factory)
        async with factory.begin() as session:
            row = await session.scalar(select(Message))
            row.rag_metadata = {"keep": "原数据"}
        await chat.save_order_wait(ctx, action)
        await chat.save_order_wait(ctx, action)
        async with factory() as session:
            row = await session.scalar(select(Message))
            assert row.turn_status == "awaiting_input" and row.rag_metadata["keep"] == "原数据"
            assert row.rag_metadata["order_selection"]["status"] == "awaiting_input"
        assert await chat.get_turn_context("wait", "demo-other-user", ctx.turn_id) is None
        assert await chat.get_turn_context("wait", "demo-user", ctx.turn_id) == ctx


@pytest.mark.parametrize("finish", [False, True])
async def test_saved_wait_cannot_regress_accepted_action(tmp_path, finish):
    """接受/完成后旧保存不能倒退，重复同键返回。参数：tmp_path目录、finish是否完成；返回：None；异常：断言失败。"""
    async with offline_recovery_database(tmp_path) as (_, factory):
        chat, ctx, action = await prepare_wait(factory)
        await chat.save_order_wait(ctx, action)
        repo = ActionRepository(factory)
        request = selection(ctx, action)
        first = await repo.accept_order_selection(request)
        if finish:
            await chat.finish_turn(ctx, "已处理")
        with pytest.raises(PersistenceUncertain):
            await chat.save_order_wait(ctx, action)
        await chat.cancel_order_wait(ctx.session_id, ctx.user_id)
        retry = await repo.accept_order_selection(request)
        assert first.accepted_order_id == retry.accepted_order_id == "1001"
        assert retry.turn_status == ("completed" if finish else "running")
        with pytest.raises(workflow.OrderSelectionConflict):
            await repo.accept_order_selection(selection(ctx, action, order_id="1002"))


async def test_cancelled_card_and_other_identity_are_rejected(tmp_path):
    """旧卡、他人身份和非本人订单均不接受。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with offline_recovery_database(tmp_path) as (_, factory):
        chat, ctx, action = await prepare_wait(factory)
        await chat.save_order_wait(ctx, action)
        repo = ActionRepository(factory)
        for override in ({"user_id": "demo-other-user"}, {"order_id": "2003"}, {"action_id": str(uuid5(NAMESPACE_URL, "fake"))}):
            with pytest.raises(ActionNotFound):
                await repo.accept_order_selection(selection(ctx, action, **override))
        await chat.cancel_order_wait(ctx.session_id, ctx.user_id)
        with pytest.raises(workflow.OrderSelectionConflict):
            await repo.accept_order_selection(selection(ctx, action))


async def test_deterministic_order_call_is_stored_once(tmp_path):
    """重入不重复申请，同ID冲突须中止。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with offline_recovery_database(tmp_path) as (_, factory):
        chat, ctx, _ = await prepare_wait(factory)
        call = {"id": "stable", "name": "query_order", "args": {"order_id": "1001"}}
        await chat.save_tool_calls(ctx, [call], deduplicate=True)
        await chat.save_tool_calls(ctx, [call], deduplicate=True)
        with pytest.raises(PersistenceUncertain):
            await chat.save_tool_calls(ctx, [{**call, "args": {"order_id": "1002"}}], deduplicate=True)
        async with factory() as session:
            rows = (await session.scalars(select(Message).where(Message.role == "assistant"))).all()
            assert len(rows) == 1 and rows[0].tool_calls == [call]


@pytest.mark.parametrize("override", [{"budget": {}}, {"order_id": " "}, {"source_turn_id": ""}, {"action_id": "bad"}])
def test_order_selection_rejects_client_state(override):
    """客户端不能提交图状态或非法键。参数：override非法字段；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        workflow.OrderSelection(**{"session_id": "wait", "user_id": "demo-user", "source_turn_id": "original",
            "action_id": str(uuid5(NAMESPACE_URL, "test")), "order_id": "1001", **override})
