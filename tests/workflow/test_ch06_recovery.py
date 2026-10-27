"""CH06启动对账：真实SQLite检查点和真实仓储，恢复阶段不调用模型。"""

import pytest
from sqlalchemy import update
from app.models.customer_service import Message
from app.repositories.actions import ActionRepository
from app.schemas.tools import ToolContext
from tests.workflow.service_fakes import service_runtime, collect
from tests.workflow.test_order_selection import resume_request


@pytest.mark.parametrize("database_state", ["awaiting_input", "running", "accepted", "completed"])
async def test_restart_reconciles_wait_without_reclassifying(tmp_path, database_state):
    """暂停与跨库窗口重启后保留原动作。参数：目录、数据库阶段；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"]) as (service, deps, factory):
        events = await collect(service, "我要退款")
        action = events[-1].data
        request = resume_request(action)
        if database_state == "running":
            async with factory.begin() as session:
                await session.execute(update(Message).values(turn_status="running", rag_metadata=None))
        elif database_state in {"accepted", "completed"}:
            accepted = await ActionRepository(factory).accept_order_selection(request)
            if database_state == "completed":
                await deps.chat.finish_turn(accepted.context, "已保存回答")
    async with service_runtime(tmp_path) as (service, deps, _):
        await service.reconcile_waiting_turns()
        current = await service.get_current_actions("s", "demo-user")
        assert current.source_turn_id == action["source_turn_id"] and not deps.model.inputs
        if database_state == "accepted":
            assert current.resume_required and current.turn_status == "running" and current.accepted_order_id == "1001"
        elif database_state == "completed":
            assert current.turn_status == "completed" and current.answer == "已保存回答" and not current.resume_required
        else:
            assert current.turn_status == "awaiting_input" and current.order_selection.action_id == action["action_id"]


async def test_orphan_wait_fails_but_real_running_is_cancelled(tmp_path):
    """等待没有对应interrupt时标不一致，普通遗留运行轮取消。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (service, deps, _):
        from tests.workflow.test_order_wait_state import prepare_wait
        chat, ctx, action = await prepare_wait(deps.chat.factory, "orphan")
        await chat.save_order_wait(ctx, action)
        await chat.start_turn(ToolContext(session_id="unfinished", user_id="demo-user", turn_id="old", user_message="旧问题"), "旧问题")
        await service.reconcile_waiting_turns()
        assert (await chat.load_turns(ctx.conversation_id))[0].turn_status == "failed"
        assert (await chat.get_workflow_turn("unfinished", "demo-user"))["turn_status"] == "cancelled"


async def test_checkpoint_read_error_prevents_recovery_cleanup(tmp_path):
    """读取失败不得当作没有等待然后取消原轮。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"]) as (service, deps, _):
        await collect(service, "我要退款")

        async def fail(*args):
            """模拟保存器读取故障。参数：原参数；返回：无；异常：ConnectionError。"""
            raise ConnectionError("checkpoint unavailable")

        service.graph.aget_state = fail
        with pytest.raises(ConnectionError):
            await service.reconcile_waiting_turns()
        assert (await deps.chat.load_turns(1))[0].turn_status == "awaiting_input"
