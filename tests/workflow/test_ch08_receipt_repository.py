"""CH08 真实仓储回执边界：弱证据正文只接受本轮真实工单，不允许伪造前缀。"""

import pytest
from sqlalchemy import delete, select

from app.models.customer_service import Message, Ticket
from app.repositories.actions import ActionRepository
from app.repositories.chat import PersistenceUncertain
from app.schemas.workflow import AgentDecision, RetrievalLowConfidence, TicketResume
from app.tools.contracts import ToolExecutionScope, binding_from_dict
from app.workflow.routing import fixed_reply
from tests.workflow.ch08_ticket_fakes import ticket_runtime
from tests.workflow.test_ch08_ticket_interrupt import begin, rows, ticket_call


async def create_confirmed_ticket(service, deps, factory):
    """经真实预览、确认、执行器和仓储写入工单，保留原轮待保存正文。

    参数：service 本地服务，deps 本地依赖，factory 独占 SQLite 工厂；
    返回：原轮上下文及真实回执；异常：预览缺失、执行失败或数据库错误。
    """
    context, _, snapshot = await begin(service.graph, service.chat)
    preview = snapshot.interrupts[0].value
    request = TicketResume(type="confirm_ticket", session_id=context.session_id,
        user_id=context.user_id, source_turn_id=context.turn_id,
        action_id=preview["action_id"], decision="confirm")
    actions = ActionRepository(factory)
    authorization = await actions.accept_ticket_resume(request)
    call_context = context.model_copy(update={"tool_call_id": preview["tool_call_id"]})
    pending = await service.chat.get_pending_ticket(call_context)
    await service.chat.mark_ticket_execution(call_context, request.action_id)

    async def authorize(binding, arguments, original_context):
        """提供已经由真实仓储保存的授权，不修改业务身份。

        参数：binding 原来源，arguments 原参数，original_context 原调用；
        返回：已保存的 WriteAuthorization；异常：无。
        """
        return authorization

    async def on_status(status):
        """消费执行状态，不参与业务结果判断。

        参数：status 执行器事件；返回：None；异常：无。
        """

    result = await deps.executor.execute(ticket_call(), call_context, on_status,
        scope=ToolExecutionScope(binding=binding_from_dict(pending["binding"]), authorize_write=authorize))
    assert result.ok
    await service.chat.finish_ticket_preview(call_context, result)
    receipt = await actions.get_chat_ticket_receipt(request)
    assert receipt is not None
    return context, receipt


async def test_low_evidence_accepts_current_receipt_and_pools_original_once(tmp_path):
    """拒绝本轮真实回执、丢失原问或重复入池时失败。

    参数：tmp_path 独占 SQLite 目录；返回：None；异常：断言或数据库错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()])]) as (service, deps, factory, _):
        context, receipt = await create_confirmed_ticket(service, deps, factory)
        answer = f"工单已创建，工单号：{receipt.ticket_no}，状态：待处理。\n\n" + fixed_reply("fallback")["answer"]
        low = RetrievalLowConfidence(reason="实际检索最高分低于门槛")
        for _ in range(2):
            await service.chat.finish_turn(context, answer, low_confidence=low)
        assert await rows(factory, "SELECT content FROM messages WHERE role='assistant' AND tool_calls IS NULL") == [(answer,)]
        assert await rows(factory, "SELECT raw_question, source FROM low_confidence_questions") == [(context.user_message, "retrieval_low_conf")]
        assert await rows(factory, "SELECT ticket_no FROM tickets") == [(receipt.ticket_no,)]
        assert await rows(factory, "SELECT status FROM tool_audit_logs") == [("成功",)]


@pytest.mark.parametrize("corruption", ["forged_number", "extra_text", "other_turn", "missing_ticket", "forged_result", "cancelled_decision"])
async def test_low_evidence_rejects_untrusted_receipt_without_saving(tmp_path, corruption):
    """仅凭固定答复后缀、其他轮票号或损坏成功记录放行时失败。

    参数：tmp_path 独占目录，corruption 伪造或不一致边界；
    返回：None；异常：断言或数据库错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()])]) as (service, deps, factory, _):
        context, receipt = await create_confirmed_ticket(service, deps, factory)
        number = "FORGED-TICKET" if corruption == "forged_number" else receipt.ticket_no
        prefix = f"工单已创建，工单号：{number}，状态：待处理。\n\n"
        if corruption == "extra_text":
            prefix += "退款肯定成功。\n\n"
        elif corruption == "other_turn":
            context = await service.chat.start_turn(context.model_copy(update={"turn_id": "another-turn"}), context.user_message)
        elif corruption in {"missing_ticket", "forged_result", "cancelled_decision"}:
            async with factory.begin() as session:
                if corruption == "missing_ticket":
                    await session.execute(delete(Ticket).where(Ticket.ticket_no == receipt.ticket_no))
                elif corruption == "forged_result":
                    result = await session.scalar(select(Message).where(Message.role == "tool"))
                    result.content = result.content.replace(receipt.ticket_no, "FORGED-TICKET")
                    prefix = prefix.replace(receipt.ticket_no, "FORGED-TICKET")
                else:
                    user = await session.scalar(select(Message).where(Message.role == "user"))
                    user.rag_metadata = {**user.rag_metadata, "pending_ticket": {
                        **user.rag_metadata["pending_ticket"], "decision": "cancel"}}
        with pytest.raises((ValueError, PersistenceUncertain)):
            await service.chat.finish_turn(context, prefix + fixed_reply("fallback")["answer"],
                low_confidence=RetrievalLowConfidence(reason="实际检索最高分低于门槛"))
        assert await rows(factory, "SELECT content FROM messages WHERE role='assistant' AND tool_calls IS NULL") == []
        assert await rows(factory, "SELECT raw_question FROM low_confidence_questions") == []
