"""客户端取消及ASGI发送失败必须即时关闭上游并释放会话占用。"""

import asyncio
import pytest

from app.api.chat import chat
from app.schemas.chat import ChatRequest
from app.schemas.workflow import AgentDecision
from tests.workflow.service_fakes import collect, service_runtime


@pytest.mark.parametrize("spec", ["2.0", "2.4"])
async def test_send_failure_closes_suspended_model_and_releases_guard(tmp_path, spec):
    """网络send失败只等待GC清理时失败。参数：tmp_path目录、spec协议；返回：None；异常：受控网络错误。"""
    async with service_runtime(tmp_path, intents=["订单", "闲聊"], decisions=[AgentDecision(kind="answer")]) as (service, deps, _):
        closed = asyncio.Event()

        async def waiting_answer(messages, ledger):
            """输出一块后等待取消。参数：messages/ledger模型契约；返回：正文流；异常：取消。"""
            try:
                yield "部分答案"
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def receive():
            """保持接收端不主动断连。参数：无；返回：不返回；异常：取消。"""
            await asyncio.Event().wait()

        sent = []

        async def send(message):
            """首正文发送失败。参数：message响应帧；返回：None；异常：OSError。"""
            sent.append(message)
            if message["type"] == "http.response.body":
                raise OSError("连接已关闭")

        deps.model.stream_answer = waiting_answer
        response = await chat(ChatRequest(session_id="s", message="查订单"), service)
        with pytest.raises(Exception):
            await response({"type": "http", "asgi": {"spec_version": spec}}, receive, send)
        assert closed.is_set()
        assert (await deps.chat.load_turns(1))[0].turn_status == "cancelled"
        assert not any(b"[DONE]" in item.get("body", b"") for item in sent)
        assert (await collect(service))[-1].event == "done"


@pytest.mark.parametrize("stage", ["classify", "answer", "tool"])
async def test_cancel_while_waiting_closes_upstream_and_never_replays_pending(tmp_path, stage):
    """首帧前或工具等待取消遗留任务/申请时失败。参数：tmp_path目录、stage阻塞阶段；返回：None；异常：受控取消。"""
    from tests.workflow.graph_fakes import call
    decisions = [AgentDecision(kind="tools", tool_calls=[call("old", order_id="1001")])] if stage == "tool" else [AgentDecision(kind="answer")]
    async with service_runtime(tmp_path, intents=["订单"], decisions=decisions) as (service, deps, _):
        entered, closed = asyncio.Event(), asyncio.Event()

        async def block(*args, **kwargs):
            """阻塞外部边界并记录取消。参数：原边界参数；返回：不返回；异常：CancelledError。"""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def answer(messages, ledger):
            """在首正文前等待。参数：模型契约；返回：异步流；异常：取消。"""
            await block()
            yield "不可到达"

        if stage == "classify":
            deps.model.classify = block
        elif stage == "answer":
            deps.model.stream_answer = answer
        else:
            deps.executor.execute = block
        task = asyncio.create_task(collect(service, "查订单"))
        await asyncio.wait_for(entered.wait(), 3)
        # 同一共享guard供Task6使用；并发请求必须被拒且不新增轮次。
        busy = await collect(service, "第二条")
        assert busy[-1].event == "error" and "当前会话" in busy[-1].data["error"]
        assert len(await deps.chat.load_turns(1)) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert (await deps.chat.load_turns(1))[0].turn_status == "cancelled"
    async with service_runtime(tmp_path, intents=["闲聊"]) as (service, deps, _):
        assert (await collect(service))[-1].event == "done"
        assert deps.executor.calls == []
        messages = deps.model.inputs[0]["messages"]
        assert not any(row.get("tool_calls") or row["role"] == "tool" for row in messages)


async def test_shared_guard_rejects_chat_while_confirmation_holds_session(tmp_path):
    """独立动作占用未能阻止聊天写库时失败。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["闲聊"]) as (service, deps, _):
        async with service.guard.hold("s"):
            assert (await collect(service))[-1].event == "error"
            assert await deps.chat.load_turns(1) == []
        assert (await collect(service))[-1].event == "done"
