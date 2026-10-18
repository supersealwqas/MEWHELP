"""真实 MySQL 故障注入：提交确认丢失、核实不可用和取消边界。"""

import asyncio
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSessionTransaction
from app.models.customer_service import Message, Ticket
from app.repositories.chat import PersistenceUncertain
from app.schemas.tools import ToolContext
from app.tools.registry import build_registry
from app.tools.executor import ToolExecutor

pytestmark = pytest.mark.mysql


async def setup_call(repositories):
    """准备原始调用并存申请。参数：真实仓库；返回：上下文、申请；异常：数据库错误。"""
    ctx = ToolContext(session_id="retry-session", user_id="demo-user", turn_id="retry-turn",
                      tool_call_id="ticket-call", user_message="请建人工工单")
    call = {"id": ctx.tool_call_id, "name": "create_ticket",
            "args": {"description": "订单延迟", "ticket_type": "咨询"}}
    ctx = await repositories[0].start_turn(ctx, ctx.user_message)
    await repositories[0].save_tool_calls(ctx, [call])
    return ctx, call


async def ignore_status(frame):
    """不展示测试状态。参数：状态帧；返回：None；异常：无。"""


async def test_executor_reuses_ticket_after_commit_ack_lost(database, repositories, monkeypatch):
    """提交后断线时返回唯一原结果。参数：数据库、仓库和替换器；返回：None；异常：断言失败。"""
    ctx, call = await setup_call(repositories)
    original = AsyncSessionTransaction.__aexit__
    commits = []
    async def commit_then_fail(transaction, *args):
        """真实提交后抛断线。参数：事务和退出参数；返回：不返回；异常：OperationalError。"""
        await original(transaction, *args)
        commits.append(1)
        raise OperationalError(None, None, RuntimeError("lost acknowledgement"), connection_invalidated=True)
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSessionTransaction, "__aexit__", commit_then_fail)
        result = await ToolExecutor(build_registry(), *repositories).execute(call, ctx, ignore_status)
    assert result.ok and result.data["ticket_no"].startswith("T")
    # 第二个短事务只是结果复用；业务工单仍只有一张。
    async with database[1]() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 1
        assert await session.scalar(select(func.count()).select_from(Message).where(Message.role == "tool")) == 1


async def test_no_blind_retry_when_verification_unavailable(database, repositories, monkeypatch):
    """提交状态无法核实则不追加相反结果。参数：数据库与替换器；返回：None；异常：断言失败。"""
    ctx, call = await setup_call(repositories)
    chat, support = repositories
    original_exit, original_read = AsyncSessionTransaction.__aexit__, chat.get_tool_result
    commit_happened = asyncio.Event()
    async def commit_then_fail(transaction, *args):
        """完成真实提交再模拟断线。参数：事务退出信息；返回：不返回；异常：OperationalError。"""
        await original_exit(transaction, *args)
        commit_happened.set()
        raise OperationalError(None, None, RuntimeError("lost"), connection_invalidated=True)
    async def unavailable_after_commit(context):
        """第一次读取正常，提交后核实不可用。参数：上下文；返回：原结果；异常：ConnectionError。"""
        if commit_happened.is_set():
            raise ConnectionError("database down")
        return await original_read(context)
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSessionTransaction, "__aexit__", commit_then_fail)
        patch.setattr(chat, "get_tool_result", unavailable_after_commit)
        with pytest.raises(PersistenceUncertain):
            await ToolExecutor(build_registry(), chat, support).execute(call, ctx, ignore_status)
    original_result = await chat.get_tool_result(ctx)
    assert original_result.ok
    async with database[1]() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 1
        assert await session.scalar(select(func.count()).select_from(Message).where(Message.role == "tool")) == 1


async def test_cancel_waits_for_commit_boundary(database, repositories, monkeypatch):
    """提交阶段取消时先等待事务结局，不能遗留后台写入。参数：真实库与替换器；返回：None；异常：断言失败。"""
    ctx, call = await setup_call(repositories)
    original = AsyncSessionTransaction.__aexit__
    committing, release = asyncio.Event(), asyncio.Event()
    async def delayed_commit(transaction, *args):
        """控制提交边界。参数：事务退出信息；返回：原退出结果；异常：数据库错误。"""
        committing.set()
        await release.wait()
        return await original(transaction, *args)
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSessionTransaction, "__aexit__", delayed_commit)
        task = asyncio.create_task(ToolExecutor(build_registry(), *repositories).execute(call, ctx, ignore_status))
        await asyncio.wait_for(committing.wait(), 2)
        task.cancel()
        # 只检测是否在提交仍受阻时过早宣布清理完成，不依赖任意固定睡眠。
        done, _ = await asyncio.wait({task}, timeout=0.03)
        release.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await asyncio.sleep(0)
        assert not done
    assert (await repositories[0].get_tool_result(ctx)).ok


async def test_retry_after_confirmed_rollback(database, repositories, monkeypatch):
    """确认回滚后才重试整笔工单事务。参数：测试库与替换器；返回：None；异常：断言失败。"""
    from app.repositories import support as module
    ctx, call = await setup_call(repositories)
    original_message, attempts, states = module.Message, [], []
    def fail_once(**kwargs):
        """第一次写结果前模拟连接失败。参数：消息字段；返回：Message；异常：OperationalError。"""
        attempts.append(1)
        if len(attempts) == 1:
            raise OperationalError(None, None, RuntimeError("before commit"), connection_invalidated=True)
        return original_message(**kwargs)
    async def status(frame):
        """记录重试状态。参数：状态帧；返回：None；异常：无。"""
        states.append(frame["status"])
    with monkeypatch.context() as patch:
        patch.setattr(module, "Message", fail_once)
        result = await ToolExecutor(build_registry(), *repositories, retry_delay_seconds=0).execute(call, ctx, status)
    assert result.ok and len(attempts) == 2
    assert states == ["running", "retrying", "succeeded"]
    async with database[1]() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 1
        assert await session.scalar(select(func.count()).select_from(Message).where(Message.role == "tool")) == 1
