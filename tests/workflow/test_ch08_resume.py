"""CH08 恢复协议与持久化决定测试，隔离外部模型并保留真实 HTTP 和仓储边界。"""

from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select

from app.api.chat import router
from app.models.customer_service import Message
from app.repositories.actions import ActionNotFound, ActionRepository
from app.schemas.chat import ChatEvent
from app.schemas.tools import ToolContext
from app.schemas.workflow import OrderSelectionConflict
from app.tools.contracts import WriteAuthorization, json_hash
from tests.workflow.service_fakes import service_runtime
from tests.workflow.ch08_ticket_fakes import ticket_runtime
from tests.workflow.test_ch08_ticket_interrupt import ticket_call, rows
from tests.workflow.service_fakes import collect


class ResumeProtocolService:
    """HTTP 编码测试的明确服务边界，仅输出已经解析的动作类型和值。"""

    async def validate_resume(self, request):
        """不依赖数据库验证编码边界。参数：已解析请求；返回：None；异常：无。"""

    async def stream_resume(self, request):
        """输出实际解析分支。参数：已解析请求；返回：语义事件流；异常：错误模型缺字段。"""
        value = request.order_id if hasattr(request, "order_id") else request.decision
        yield ChatEvent(event="delta", data={"delta": value})
        yield ChatEvent(event="done")


def resume_body(**changes):
    """构造浏览器允许提交的确认请求。参数：changes 字段变体；返回：JSON 请求；异常：无。"""
    return {"type": "confirm_ticket", "session_id": "ticket-session", "user_id": "demo-user",
            "source_turn_id": "ticket-turn", "action_id": str(uuid5(NAMESPACE_URL, "ticket-action")),
            "decision": "confirm", **changes}


async def post_resume(body):
    """通过真实项目路由解析并编码恢复请求。参数：body JSON；返回：HTTP 响应；异常：HTTP/应用错误。"""
    app = FastAPI()
    app.include_router(router)
    app.state.chat_service = ResumeProtocolService()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/api/chat/resume", json=body)


async def test_old_order_resume_body_is_accepted():
    """给旧请求强制新增 type 会破坏既有选单。参数：无；返回：None；异常：断言失败。"""
    body = {key: value for key, value in resume_body().items() if key not in {"type", "decision"}}
    response = await post_resume({**body, "order_id": "1001"})
    assert response.status_code == 200
    assert '"delta": "1001"' in response.text and "[DONE]" in response.text


@pytest.mark.parametrize("decision", ["confirm", "cancel"])
async def test_ticket_resume_dispatches_strict_decision(decision):
    """新请求落入旧选单分支或丢失取消会失败。参数：decision 两种决定；返回：None；异常：断言失败。"""
    response = await post_resume(resume_body(decision=decision))
    assert response.status_code == 200
    assert f'"delta": "{decision}"' in response.text and "[DONE]" in response.text


@pytest.mark.parametrize("extra", [
    {"description": "客户端替换问题"}, {"ticket_type": "投诉"}, {"confirmed": True},
    {"order_id": "1001"}, {"decision": True}, {"decision": 1}, {"decision": "true"},
    {"session_id": 123}, {"user_id": False}, {"source_turn_id": 9}, {"type": "select_order"},
])
async def test_ticket_resume_rejects_parameter_override(extra):
    """额外参数或宽松类型不能覆盖冻结预览。参数：extra 非法变体；返回：None；异常：断言失败。"""
    response = await post_resume(resume_body(**extra))
    assert response.status_code == 422


async def test_ticket_resume_requires_explicit_type():
    """确认请求省略类型不能隐式进入新分支。参数：无；返回：None；异常：断言失败。"""
    response = await post_resume({key: value for key, value in resume_body().items() if key != "type"})
    assert response.status_code == 422


async def saved_preview(chat, factory):
    """保存符合冻结契约的来源和工具申请，供真实仓储决定测试使用。

    参数：chat 真实聊天仓储，factory 独占数据库工厂；返回：上下文、预览和请求字典；异常：数据库错误。
    """
    context = await chat.start_turn(ToolContext(session_id="ticket-session", user_id="demo-user",
        turn_id="ticket-turn", tool_call_id="ticket-call", user_message="请建工单，商品破损"), "请建工单，商品破损")
    arguments = {"description": "商品破损", "ticket_type": "售后"}
    from app.tools.confirmation import ticket_preview
    from app.tools.registry import build_ticket_registry
    from app.tools.contracts import ToolIdentity, binding_dict
    binding = build_ticket_registry().get_definition(ToolIdentity("builtin", None, "create_ticket")).binding
    action = ticket_preview(context, arguments)
    await chat.save_tool_calls(context, [{"id": context.tool_call_id, "name": "create_ticket", "args": arguments}],
                               bindings={context.tool_call_id: binding_dict(binding)})
    pending = {"action": action.model_dump(mode="json"), "binding": binding_dict(binding),
               "arguments_hash": json_hash(arguments), "status": "awaiting_input", "decision": None,
               "execution_started": False}
    async with factory.begin() as session:
        rows = list((await session.scalars(select(Message).where(Message.conversation_id == context.conversation_id))).all())
        for row in rows:
            row.turn_status = "awaiting_input"
            if row.role == "user":
                row.rag_metadata = {"pending_ticket": pending, "ticket_write_state": "preview"}
    body = resume_body(action_id=action.action_id)
    return context, action, body


@pytest.mark.parametrize("change", [{"session_id": "other"}, {"user_id": "other"},
                                     {"source_turn_id": "other"}, {"action_id": str(uuid5(NAMESPACE_URL, "other"))}])
async def test_cross_session_action_is_rejected(tmp_path, change):
    """未核验原会话、用户、轮次或动作会使串单通过。参数：tmp_path 目录、change 身份变体；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (service, _, factory):
        context, _, body = await saved_preview(service.chat, factory)
        from app.schemas.workflow import TicketResume
        with pytest.raises(ActionNotFound):
            await ActionRepository(factory).accept_ticket_resume(TicketResume(**{**body, **change}))
        pending = await service.chat.get_pending_ticket(context)
        assert pending["decision"] is None


@pytest.mark.parametrize("decision", ["confirm", "cancel"])
async def test_first_decision_is_durable_and_cannot_be_reversed(tmp_path, decision):
    """接受结果未提交或可被反向覆盖会失败。参数：tmp_path 目录、decision 首次选择；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (service, _, factory):
        context, action, body = await saved_preview(service.chat, factory)
        from app.schemas.workflow import TicketResume
        request = TicketResume(**{**body, "decision": decision})
        first = await ActionRepository(factory).accept_ticket_resume(request)
        repeated = await ActionRepository(factory).accept_ticket_resume(request)
        assert isinstance(first, WriteAuthorization) and asdict(first) == asdict(repeated)
        assert first.decision == decision and first.action_id == action.action_id
        assert first.conversation_id == context.conversation_id and first.tool_call_id == "ticket-call"
        pending = await service.chat.get_pending_ticket(context)
        assert pending["decision"] == decision
        opposite = "cancel" if decision == "confirm" else "confirm"
        with pytest.raises(OrderSelectionConflict):
            await ActionRepository(factory).accept_ticket_resume(TicketResume(**{**body, "decision": opposite}))


async def open_preview(service):
    """通过真实服务启动直接建单请求并取得公开预览。

    参数：service 真实服务；返回：客户端恢复请求；异常：未正确暂停时断言失败。
    """
    events = await collect(service, "帮我建个工单，订单1001收到时破损", session="ticket-session")
    previews = [event.data for event in events if event.event == "confirm_ticket"]
    assert len(previews) == 1, [(event.event, event.data) for event in events]
    assert not any(event.event in {"error", "done"} for event in events)
    return resume_body(action_id=previews[0]["action_id"], source_turn_id=previews[0]["source_turn_id"])


@pytest.mark.parametrize("decision", ["confirm", "cancel"])
async def test_full_service_resumes_original_turn_once(tmp_path, decision):
    """确认或取消必须恢复原工具轮；重复请求不能新增用户消息、工单或审计。

    参数：tmp_path 独占目录、decision 两种决定；返回：None；异常：断言失败。
    """
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer")], answers=[["已记录。"]]) as (service, _, factory, _):
        body = await open_preview(service)
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []
        before = await rows(factory, "SELECT id, content FROM messages WHERE role='user'")
        request = TicketResume(**{**body, "decision": decision})
        first = [event async for event in service.stream_resume(request)]
        assert first[-1].event == "done", [(event.event, event.data) for event in first]
        repeated = [event async for event in service.stream_resume(request)]
        assert repeated[-1].event == "done"
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_decision == decision and not current.resume_required
        assert await rows(factory, "SELECT id, content FROM messages WHERE role='user'") == before
        tickets = await rows(factory, "SELECT ticket_no FROM tickets")
        assert len(tickets) == (1 if decision == "confirm" else 0)
        assert await rows(factory, "SELECT status FROM tool_audit_logs") == [("成功" if decision == "confirm" else "权限拒绝",)]
        if decision == "confirm":
            assert current.ticket_receipt.ticket_no == tickets[0][0]
            assert tickets[0][0] in "".join(event.data["delta"] for event in first if event.event == "delta")


@pytest.mark.parametrize("after_progress", [False, True])
async def test_accepted_resume_can_recover_after_disconnect(tmp_path, monkeypatch, after_progress):
    """决定提交后，进度帧前后断线都必须保留原决定并允许同键继续。

    参数：tmp_path 独占目录、monkeypatch 注入图流断线、after_progress 是否先发前序查询进度；返回：None；异常：断言失败。
    """
    import asyncio
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer")], answers=[["已记录。"]]) as (service, _, factory, _):
        body = await open_preview(service)
        original = service._run_graph

        async def disconnected(initial, context, outcome):
            """模拟确认已保存、建单执行前的取消。参数：原图参数；返回：可选查询进度；异常：CancelledError。"""
            if after_progress:
                yield ChatEvent(event="tool_status", data={"name": "query_order", "status": "succeeded"})
            raise asyncio.CancelledError()

        monkeypatch.setattr(service, "_run_graph", disconnected)
        with pytest.raises(asyncio.CancelledError):
            _ = [event async for event in service.stream_resume(TicketResume(**body))]
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_decision == "confirm" and current.resume_required
        assert current.turn_status == "running" and await rows(factory, "SELECT ticket_no FROM tickets") == []
        await service.reconcile_waiting_turns()
        monkeypatch.setattr(service, "_run_graph", original)
        result = [event async for event in service.stream_resume(TicketResume(**body))]
        assert result[-1].event == "done"
        assert len(await rows(factory, "SELECT ticket_no FROM tickets")) == 1


async def test_definite_ticket_failure_allows_next_chat(tmp_path, monkeypatch):
    """明确拒绝的已结束建单不能把下一句聊天永久拦截为结果未知。

    参数：tmp_path 独占目录、monkeypatch 注入执行前确定拒绝；返回：None；异常：断言失败。
    """
    from app.schemas.tools import ToolResult
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, intents=["订单", "闲聊"], decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer"), AgentDecision(kind="answer")],
            answers=[["本次工单未创建。"], ["你好。"]]) as (service, deps, factory, _):
        body = await open_preview(service)
        # 仅替换权限依赖，冻结动作、执行器结果保存和服务收尾仍走生产实现。
        class RevokedPermissions:
            """模拟预览之后被撤销的本地建单许可。"""

            def check(self, identity, arguments, context):
                """拒绝本次写操作。参数：来源、参数、身份；返回：确定拒绝；异常：无。"""
                return ToolResult.failed("PERMISSION_DENIED", "本地已撤销建单权限")

        monkeypatch.setattr(deps.executor, "permissions", RevokedPermissions())
        events = [event async for event in service.stream_resume(TicketResume(**body))]
        assert events[-1].event == "done", [(event.event, event.data) for event in events]
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []
        assert await rows(factory, "SELECT status FROM tool_audit_logs") == [("权限拒绝",)]
        following = await collect(service, "你好", session="ticket-session")
        assert not any(event.event == "error" for event in following), [(event.event, event.data) for event in following]
        assert following[-1].event == "done"


async def test_committed_receipt_survives_answer_failure(tmp_path):
    """工单已提交后正文异常不能把成功事实隐藏为普通错误。

    参数：tmp_path 独占目录；返回：None；异常：断言失败。
    """
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer")], answers=[[RuntimeError("正文故障")]]) as (service, _, factory, _):
        body = await open_preview(service)
        events = [event async for event in service.stream_resume(TicketResume(**body))]
        tickets = await rows(factory, "SELECT ticket_no FROM tickets")
        assert len(tickets) == 1
        assert tickets[0][0] in "".join(event.data["delta"] for event in events if event.event == "delta")
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_receipt.ticket_no == tickets[0][0]
        repeated = [event async for event in service.stream_resume(TicketResume(**body))]
        assert tickets[0][0] in "".join(event.data["delta"] for event in repeated if event.event == "delta")
        assert len(await rows(factory, "SELECT ticket_no FROM tickets")) == 1


@pytest.mark.parametrize("accepted", [False, True])
async def test_reopened_saver_restores_original_preview_and_decision(tmp_path, accepted):
    """重开真实 SQLite 保存器不能丢卡、重做决策或新建用户消息。

    参数：tmp_path 原资源目录、accepted 断开前是否已保存确认；返回：None；异常：断言失败。
    """
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()])]) as (service, _, factory, _):
        body = await open_preview(service)
        before = await rows(factory, "SELECT id, content FROM messages WHERE role='user'")
        if accepted:
            await ActionRepository(factory).accept_ticket_resume(TicketResume(**body))
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="answer")], answers=[["已记录。"]]) as (service, deps, factory, _):
        await service.reconcile_waiting_turns()
        saved = await rows(factory, "SELECT id, turn_status, rag_metadata FROM messages")
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_confirmation.action_id == body["action_id"]
        assert current.ticket_decision == ("confirm" if accepted else None)
        assert current.resume_required is accepted
        assert await rows(factory, "SELECT id, turn_status, rag_metadata FROM messages") == saved
        events = [event async for event in service.stream_resume(TicketResume(**body))]
        assert events[-1].event == "done"
        assert [item["stage"] for item in deps.model.inputs] == ["decide", "answer"]
        assert await rows(factory, "SELECT id, content FROM messages WHERE role='user'") == before
        assert len(await rows(factory, "SELECT ticket_no FROM tickets")) == 1


async def test_started_without_result_is_never_reexecuted(tmp_path):
    """执行开始但无可靠结果时，重发与启动对账不能靠再次建单核实。

    参数：tmp_path 独占目录；返回：None；异常：断言失败。
    """
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()])]) as (service, _, factory, _):
        body = await open_preview(service)
        request = TicketResume(**body)
        accepted = await ActionRepository(factory).accept_ticket_resume(request)
        context = (await service.validate_resume(request))["context"]
        await service.chat.mark_ticket_execution(context, accepted.action_id)
        await service.reconcile_waiting_turns()
        for _ in range(2):
            events = [event async for event in service.stream_resume(request)]
            assert events[-1].event == "error" and events[-1].data["outcome"] == "unknown"
            assert events[-1].data["retry_safe"] is False
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_decision == "confirm" and not current.resume_required
        assert current.ticket_receipt is None
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []
        assert await rows(factory, "SELECT tool_call_id FROM tool_audit_logs") == []
        following = await collect(service, "你好", session="ticket-session")
        assert following[-1].event == "error" and following[-1].data["status_code"] == 409


async def test_chat_receipt_is_not_replaced_by_later_button_suggestion(tmp_path):
    """模型追加旧按钮建议不能覆盖同轮真实聊天工单回执。

    参数：tmp_path 独占目录；返回：None；异常：断言失败。
    """
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer", suggested_actions=["create_ticket"])], answers=[["已记录。"]]) as (service, _, factory, _):
        body = await open_preview(service)
        events = [event async for event in service.stream_resume(TicketResume(**body))]
        assert events[-1].event == "done"
        ticket = (await rows(factory, "SELECT ticket_no FROM tickets"))[0][0]
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_receipt is not None
        assert current.ticket_receipt.ticket_no == ticket


@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_started_write_fault_remains_unknown_on_replay(tmp_path, monkeypatch, failure):
    """已经开始的写入故障不能在刷新或重发时变成确定未创建，更不能再次执行。

    参数：tmp_path 独占目录、monkeypatch 注入写处理器故障、failure 为超时或普通故障；返回：None；异常：断言失败。
    """
    import asyncio
    from app.schemas.workflow import AgentDecision, TicketResume
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()])]) as (service, deps, factory, _):
        body = await open_preview(service)
        attempts = []

        async def failed_write(context, description, ticket_type):
            """在进入真实执行边界后中断处理器。参数：原建单参数；返回：无；异常：超时取消或RuntimeError。"""
            attempts.append(context.tool_call_id)
            if failure == "timeout":
                await asyncio.sleep(1)
            raise RuntimeError("write failure")

        monkeypatch.setattr(deps.executor.support, "create_ticket", failed_write)
        deps.executor.timeout = 0.01
        request = TicketResume(**body)
        first = [event async for event in service.stream_resume(request)]
        saved = await service.chat.get_pending_ticket((await service.validate_resume(request))["context"])
        assert saved["status"] == "unknown"
        assert "待核实" in "".join(event.data.get("delta", "") for event in first)
        for _ in range(2):
            repeated = [event async for event in service.stream_resume(request)]
            assert repeated[-1].event == "error" and repeated[-1].data["outcome"] == "unknown"
            assert repeated[-1].data["retry_safe"] is False
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_status == "unknown" and current.ticket_receipt is None
        following = await collect(service, "你好", session="ticket-session")
        assert following[-1].event == "error" and following[-1].data["status_code"] == 409
        assert attempts == ["ticket-1"] and await rows(factory, "SELECT ticket_no FROM tickets") == []
