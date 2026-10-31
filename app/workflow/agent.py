"""由decision/tools/answer节点编排的工具子图，核验建单确认并保留完整事实及停止边界。"""

import asyncio
import json
from copy import deepcopy
from contextlib import aclosing
from typing import TYPE_CHECKING

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.core.memory import ContextBudgetExceeded, valid_tool_calls
from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
from app.schemas.chat import InvalidToolDecision
from app.schemas.tools import ToolContext, ToolResult
from app.repositories.chat import PersistenceUncertain
from app.tools.confirmation import TicketIntentState, request_ticket_authorization, ticket_context_notes, ticket_write_rejection
from app.tools.contracts import KnowledgeCallContext, ToolExecutionScope, binding_dict, binding_from_dict
from app.tools.business import list_user_orders
from app.tools.knowledge import merge_knowledge_budget, retrieve_verified_knowledge, restore_turn_knowledge, unresolved_knowledge_faults
from app.workflow.citations import CitationStreamFilter
from app.workflow.answer_policy import query_fault_answer
from app.workflow.model import decision_tool_schemas
from app.workflow.prompts import ANSWER_PROMPT, DECISION_PROMPT
from app.workflow.state import ConversationState, message_row
from app.workflow.stages import publish_stage
from app.workflow.policy_context import verified_business_prefix, optional_probe_miss, knowledge_failure_reason
from app.workflow.routing import _active_order_tokens, knowledge_input_stop_reason, order_candidate, _trusted_history
from app.workflow.resilience import DEPENDENCY_ERRORS, degradation
from app.workflow.model import WorkflowOutputError
from app.workflow.context import assemble_context

if TYPE_CHECKING:
    from app.workflow.graph import WorkflowDependencies

_STOP_TEXT = {
    "budget_exceeded": "本轮问题和查询结果已达到处理额度，请拆分问题后继续。",
    "decision_limit": "本轮处理步骤已达到上限，请补充或拆分问题后继续。",
    "tool_call_limit": "本轮查询次数已达到上限，请分开查询剩余问题。",
    "duplicate_tool_call": "本轮已查询过相同信息，未重复查询。请补充需要继续处理的问题。",
    "invalid_decision": "抱歉，本次回复生成失败，请重试。",
    "tool_failed": "本轮查询未能完成，请稍后重试或选择人工客服协助。",
    "ticket_cancelled": "工单申请已取消，本次没有创建工单。",
    "ticket_unknown": "本次工单处理结果待核实，请继续核实原操作，勿重新建单。",
    "model_unavailable": "本次暂时无法完成问题处理，请稍后重试；您的问题已保留在当前对话中。",
    "retrieval_unavailable": "资料查询暂时不可用，相关规则目前无法核实。您可以稍后重试或联系人工客服继续处理。",
    "answer_unavailable": "本次回答未能完整生成，尚未说明的内容暂无法确认。请稍后重试或继续补充问题。",
}


def stopped_state(reason: str, ledger: BudgetLedger | None = None) -> ConversationState:
    """生成明确的固定停止出口。参数：reason稳定原因、ledger可选账本；返回：State部分更新；异常：KeyError，未知原因。"""
    update = {"stop_reason": reason, "answer": _STOP_TEXT[reason], "pending_calls": [], "suggested_kinds": []}
    if ledger is not None:
        update["budget"] = ledger.snapshot()
    return update


async def confirmed_ticket_prefix(state: ConversationState, chat) -> str:
    """从已完成建单的原调用回读票号，供普通答复与固定出口共用。

    参数：state 当前可信工作流状态；chat 保存原工具结果的聊天仓储。
    返回：str，成功回执前缀；没有已确认的成功结果时为空串。
    异常：数据库读取错误或回执字段校验错误；不重试写入。
    """
    from app.schemas.workflow import TicketReceipt
    pending = state.get("pending_ticket")
    if not pending or pending.get("status") != "completed":
        return ""
    # 使用原轮、原调用的持久化结果，模型正文和后续订单选择均不能提供票号。
    action = pending["action"]
    context = ToolContext.model_validate(state["context"]).model_copy(update={"tool_call_id": action["tool_call_id"]})
    result = await chat.get_tool_result(context)
    if result is None or not result.ok or not result.data:
        return ""
    return TicketReceipt(action_id=action["action_id"], ticket_no=result.data["ticket_no"],
        ticket_type=result.data["ticket_type"], status=result.data["status"]).answer_prefix()


def model_messages(state: ConversationState, settings, prompt: str, tools: list | None = None, *, registry=None) -> list[dict]:
    """复用共享历史组装模型副本。参数：state、settings、固定prompt、tools和共享registry；返回：ModelContext；异常：预算或身份错误。"""
    return assemble_context(state, settings, prompt, tools, registry=registry)


def _signature(call: dict) -> str:
    """按既有工具去空白规则计算去重键。参数：call严格工具申请；返回：稳定JSON；异常：ValueError，非法JSON数值。"""
    args = {key: value.strip() if isinstance(value, str) else value for key, value in call["args"].items()}
    return json.dumps([call["name"], args], sort_keys=True, ensure_ascii=False, allow_nan=False)


def build_agent_decision(deps: "WorkflowDependencies"):
    """建立根图和子图共用的决定节点。参数：deps唯一实例依赖；返回：异步节点；异常：无。"""
    async def decision(state: ConversationState) -> ConversationState:
        """执行有界内部决定，不发布内部JSON。参数：state当前事实；返回：申请或终止更新；异常：模型/审计错误。"""
        ledger = BudgetLedger.from_snapshot(state["budget"])
        if state["decision_count"] >= deps.settings.workflow_max_decisions:
            return stopped_state("decision_limit", ledger)
        try:
            if deps.permissions is not None:
                deps.permissions.reload()
            if deps.mcp is not None:
                await deps.mcp.refresh()
            tools = [tool for tool in deps.registry.snapshot()
                     if deps.permissions is None and tool.identity.source == "builtin"
                     or deps.permissions is not None and deps.permissions.allowed(tool.identity)]
            # 结束决策也使用工具协议；历史裁剪须计入与模型实际绑定一致的完整定义。
            ticket_intent = TicketIntentState.model_validate(state["ticket_intent"]) if state.get("ticket_intent") else None
            # 每次决定重新派生建单边界，避免确认后残留旧许可，也不向正文模型传技术操作提示。
            current = {**state, "context_notes": [*state.get("context_notes", []), *ticket_context_notes(ticket_intent)]}
            messages = model_messages(current, deps.settings, DECISION_PROMPT, decision_tool_schemas(tools), registry=deps.registry)
            # 先检查最少的调用/输出余量；适配器还按完整输入执行精确保守占额检查。
            ledger.ensure_capacity([("decide", 0, ledger.output_limit("decide")),
                ("answer", 0, ledger.output_limit("answer")), ("answer", 0, ledger.output_limit("answer"))])
            selected = await deps.model.decide(messages, tools, ledger)
            calls = selected.tool_calls
            if calls and (not valid_tool_calls(calls) or any(call["id"] in state["seen_call_ids"]
                    or call["name"] not in {tool.name for tool in tools} for call in calls)):
                raise InvalidToolDecision(tool_calls=calls)
        except (WorkflowBudgetExceeded, ContextBudgetExceeded):
            return stopped_state("budget_exceeded", ledger)
        except InvalidToolDecision as error:
            await deps.chat.save_tool_calls(ToolContext.model_validate(state["context"]), [error.audit_record])
            return {**stopped_state("invalid_decision", ledger), "decision_count": state["decision_count"] + 1}
        except DEPENDENCY_ERRORS as error:
            return {**stopped_state("model_unavailable", ledger), **degradation(state, "decide", error),
                    "decision_count": state["decision_count"] + 1}
        by_name = {tool.name: tool for tool in tools}
        frozen = {call["id"]: binding_dict(by_name[call["name"]].binding) for call in calls}
        # 原允许清单的读写属性随决定进入检查点；后续发现失败不能撤销已经发生的查询事实。
        call_facts = deepcopy(state.get("tool_call_facts", {}))
        for call in calls:
            call_facts.setdefault(call["id"], {"turn_id": state["context"]["turn_id"],
                "binding": deepcopy(frozen[call["id"]]), "access": by_name[call["name"]].access})
        suggested = [kind for kind in selected.suggested_actions
                     if kind != "create_ticket" or not state.get("ticket_write_state")]
        return {"budget": ledger.snapshot(), "decision_count": state["decision_count"] + 1,
                "pending_calls": calls, "pending_bindings": frozen, "pending_tool_bindings": frozen,
                "tool_call_facts": call_facts,
                "suggested_kinds": suggested,
                "stop_reason": selected.kind if selected.kind != "tools" else None}
    return decision


def build_agent_graph(deps: "WorkflowDependencies", *, decision=None):
    """编译共享State的无独立saver子图。参数：deps实例依赖、decision共用节点；返回：子图；异常：编译错误。"""
    decision = decision or build_agent_decision(deps)

    async def tools_node(state: ConversationState) -> ConversationState:
        """逐个执行已批准申请并写完整配对。参数：state待执行批次；返回：完整替换更新；异常：持久化错误/取消。"""
        context = ToolContext.model_validate(state["context"])
        calls, messages = state["pending_calls"], []
        bindings = state.get("pending_tool_bindings") or state.get("pending_bindings", {})
        ticket_intent = TicketIntentState.model_validate(state["ticket_intent"]) if state.get("ticket_intent") else None
        pending_ticket, ticket_write_state = state.get("pending_ticket"), state.get("ticket_write_state")
        ledger = BudgetLedger.from_snapshot(state["budget"])
        knowledge_state = deepcopy(state)
        knowledge_state.setdefault("knowledge_cache", {})
        if "retrieval_snapshot" not in knowledge_state or "pool_signals" not in knowledge_state:
            # 根Command.update不能改写已暂停子图的旧输入，消费边界也要恢复新增字段。
            knowledge_state.update(await restore_turn_knowledge(knowledge_state, deps.chat))
        knowledge_state["knowledge_faults"] = unresolved_knowledge_faults(knowledge_state)
        seen_ids, seen_signatures = list(state["seen_call_ids"]), list(state["seen_call_signatures"])
        count, reason, batch_failed = state["tool_count"], None, False
        writer = get_stream_writer()
        await deps.chat.save_tool_calls(context, calls, deduplicate=True, bindings=bindings)
        messages.append(message_row(context.turn_id, "assistant", "", calls=calls, bindings=bindings))

        async def on_status(frame: dict) -> None:
            """发布执行器安全状态。参数：frame状态字典；返回：None；异常：流写入错误。"""
            writer({"event": "tool_status", "data": frame})

        for call in calls:
            call_context = context.model_copy(update={"tool_call_id": call["id"]})
            signature = _signature(call)
            seen_ids.append(call["id"])
            if reason is None and count >= deps.settings.tool_max_calls_per_turn:
                reason = "tool_call_limit"
            if reason is None and signature in seen_signatures:
                reason = "duplicate_tool_call"
            rejection = ToolResult.failed("TOOL_SKIPPED", "前序申请未完成或已达到处理边界，该申请未执行") if reason or batch_failed else None
            if rejection is None:
                seen_signatures.append(signature)
                count += 1
            binding = binding_from_dict(bindings[call["id"]]) if call["id"] in bindings else None
            runtime = KnowledgeCallContext(knowledge_state, ledger, deps.search, deps.settings, deps.query_expander, deps.chat)
            builtin_ticket = (binding is not None and binding.identity.source == "builtin"
                              and binding.identity.server is None and binding.identity.name == "create_ticket")
            authorize = None
            if builtin_ticket:
                pending_ticket = await deps.chat.get_pending_ticket(context)
                rejection = ticket_write_rejection(ticket_intent, call["args"], call_context, pending_ticket) or rejection

                async def authorize(saved_binding, arguments, original_context):
                    """把当前服务器诉求和仓储绑定到唯一内置写工具。

                    参数：原来源、参数和上下文；返回：已保存决定；异常：中断、拒绝或仓储故障。
                    """
                    return await request_ticket_authorization(saved_binding, arguments, original_context,
                        intent=ticket_intent, repository=deps.chat)

            # 编排拒绝也进入执行器；中断、取消及提交未知由执行器原样透传，不能另造结果。
            try:
                result = await deps.executor.execute(call, call_context, on_status,
                    scope=ToolExecutionScope(binding=binding, rejection=rejection, knowledge=runtime, authorize_write=authorize))
            except PersistenceUncertain:
                if builtin_ticket:
                    await deps.chat.finish_ticket_preview(call_context, None)
                raise
            if builtin_ticket:
                pending_ticket = await deps.chat.finish_ticket_preview(call_context, result)
                if pending_ticket:
                    ticket_write_state = {"action_id": pending_ticket["action"]["action_id"],
                        "tool_call_id": pending_ticket["action"]["tool_call_id"], "status": pending_ticket["status"]}
                    if ticket_intent and pending_ticket["action"]["tool_call_id"] == call["id"]:
                        ticket_intent = ticket_intent.model_copy(update={"status": "unknown" if pending_ticket["status"] == "unknown"
                            else "confirmed" if result.ok else "cancelled"})
            if (result.verified_knowledge is not None and binding is not None
                    and binding.identity.source == "builtin" and call["name"] in {"query_faq", "query_refund_policy"}):
                outcome = result.verified_knowledge
                ledger = BudgetLedger.from_snapshot(merge_knowledge_budget(ledger.snapshot(), outcome.budget))
                from app.knowledge.snapshots import merge_snapshots
                from app.schemas.flywheel import RetrievalSnapshot
                saved = RetrievalSnapshot.model_validate(knowledge_state["retrieval_snapshot"]) if knowledge_state.get("retrieval_snapshot") else None
                merged = merge_snapshots(saved, outcome.retrieval_snapshot) if outcome.retrieval_snapshot is not None else saved
                knowledge_state.update(outcome.model_dump(mode="json"), retrieval_attempted=True)
                # 旧回执缺快照时保留已恢复的未知/原件，新回执也只能累加而不能覆盖前次尝试。
                knowledge_state["retrieval_snapshot"] = merged.model_dump(mode="json") if merged is not None else None
            if result.ok and call["name"] == "query_order" and result.data and result.data.get("order_id"):
                # 订单政策先确定办理对象；附带查询只保留在真实工具流水中，不能替换候选或提前生成退款卡。
                if (state.get("route") != "order_policy"
                        or order_candidate(knowledge_state, list_user_orders(context.user_id)) == result.data["order_id"]):
                    knowledge_state.update(order_id=result.data["order_id"], order_data=deepcopy(result.data))
            if not result.ok and result.error is not None:
                if (binding is not None and binding.identity.source == "builtin"
                        and binding.identity.name == "query_refund_policy" and result.error.code == "ORDER_REQUIRED"
                        and signature in seen_signatures):
                    # 本地前置闸尚未查询政策，核实订单后可用新ID重提同参数；调用ID和次数仍保留。
                    seen_signatures.remove(signature)
                batch_failed = True
                try:
                    write = binding is not None and deps.registry.get_definition(binding.identity).access == "write"
                except KeyError:
                    write = False
                if result.error.code in {"USER_CANCELLED", "WRITE_RESULT_UNKNOWN"} or write and result.error.category == "fault":
                    reason = ("ticket_cancelled" if result.error.code == "USER_CANCELLED" else
                              "ticket_unknown" if pending_ticket and pending_ticket.get("status") == "unknown" else "tool_failed")
            messages.append(message_row(context.turn_id, "tool", result.model_dump_json(), call_id=call["id"]))
        update = {"messages": messages, "pending_calls": [], "pending_bindings": {}, "pending_tool_bindings": {},
                  "pending_ticket": pending_ticket, "ticket_write_state": ticket_write_state,
                  "ticket_intent": ticket_intent.model_dump(mode="json") if ticket_intent else None, "seen_call_ids": seen_ids,
                  "seen_call_signatures": seen_signatures, "tool_count": count, "budget": ledger.snapshot(),
                  **{key: knowledge_state[key] for key in ("knowledge_cache", "evidence", "gate", "evidence_scores",
                     "expanded_queries", "retrieval_attempted", "order_id", "order_data", "retrieval_snapshot", "pool_signals", "knowledge_faults")}}
        if reason:
            update.update(stopped_state(reason))
        return update

    async def answer_node(state: ConversationState) -> ConversationState:
        """推送并累计相同正文，外部断流追加说明。参数：state事实；返回：完整正文和账本；异常：取消及程序错误。"""
        writer = get_stream_writer()
        publish_stage(state, "answer", "running")
        receipt_prefix = await confirmed_ticket_prefix(state, deps.chat)
        instructions = {"create_ticket": "请点击本条回复后的“创建工单”按钮，在弹窗中确认后提交。",
                        "handoff": "如需人工协助，可点击本条回复后的“转人工”按钮。"}
        action_text = ("本轮回复后可用按钮的操作说明如下：\n" +
            "\n".join(instructions[kind] for kind in state["suggested_kinds"])) if state["suggested_kinds"] else "本轮回复后没有可操作的按钮，请按已有事实回答或追问。"
        answer_state = {**state, "context_notes": [*state.get("context_notes", []), action_text]}
        failure_reason = knowledge_failure_reason(state)
        if failure_reason is not None:
            update = stopped_state(failure_reason)
            update.update(answer=await verified_business_prefix(state, deps.chat) + update["answer"],
                          evidence=[], pool_signals=[])
            writer({"event": "delta", "data": {"delta": update["answer"]}})
            publish_stage(state, "answer", "completed")
            return update
        # 已有本轮真实知识结果才进入自评；不以意图名称决定，也不把业务字段当政策证据。
        if (state.get("gate") is not None and not optional_probe_miss(state)
                and (not state["answer"] or state["gate"].get("passed") is False)):
            from app.knowledge.answering import EvidenceCitation
            from app.knowledge.pipeline import ProductionKnowledgePipeline
            ledger = BudgetLedger.from_snapshot(state["budget"])
            prefix = await verified_business_prefix(state, deps.chat)
            try:
                pipeline = ProductionKnowledgePipeline(deps.search, deps.settings, query_expander=deps.query_expander,
                    model=deps.model, state=answer_state)
                generated = await pipeline.generate({"ledger": ledger}, {
                    "selected": [EvidenceCitation.model_validate(item) for item in state["evidence"]], "gate": state["gate"]})
                answer = prefix + generated["answer"]
                signals = {item["source"]: item for item in state.get("pool_signals", [])}
                signals.update({item.source: item.model_dump(mode="json") for item in generated["pool_signals"]})
                update = {"answer": answer, "budget": ledger.snapshot(),
                    "evidence": [] if generated["refused"] else state["evidence"],
                    "pool_signals": list(signals.values())}
                if any(item.source == "self_check" for item in generated["pool_signals"]):
                    update["stop_reason"] = "self_check"
            except (WorkflowBudgetExceeded, ContextBudgetExceeded) as error:
                update = {**stopped_state("budget_exceeded", ledger), **degradation(state, "answer", error), "evidence": []}
                update["answer"] = prefix + update["answer"]
            except DEPENDENCY_ERRORS as error:
                update = {**stopped_state("answer_unavailable", ledger), **degradation(state, "answer", error), "evidence": []}
                update["answer"] = prefix + update["answer"]
            # 只有已经通过结构/自评/引用校验的最终正文能进入custom公共流。
            writer({"event": "delta", "data": {"delta": update["answer"]}})
            publish_stage(state, "answer", "completed")
            return update
        if state["answer"]:
            answer = receipt_prefix + state["answer"]
            writer({"event": "delta", "data": {"delta": answer}})
            return {"answer": answer}
        fault_answer = query_fault_answer(state, registry=deps.registry)
        if fault_answer is not None:
            # 固定正文与已提交票号累计同一份字符串，后续仍由根图统一保存和完成。
            answer = receipt_prefix + fault_answer
            writer({"event": "delta", "data": {"delta": answer}})
            publish_stage(state, "answer", "completed")
            return {"answer": answer}
        ledger = BudgetLedger.from_snapshot(state["budget"])
        parts = []
        if receipt_prefix:
            parts.append(receipt_prefix)
            writer({"event": "delta", "data": {"delta": receipt_prefix}})
        prefix_count = len(parts)
        citation_tool_names = {tool.name for tool in deps.registry.snapshot()} | {"submit_intent", "submit_decision"}
        citations = CitationStreamFilter({item["n"] for item in state["evidence"]}, citation_tool_names)

        def finish_failure(reason: str, error: Exception) -> ConversationState:
            """保留已发布文字并追加安全说明。参数：reason停止原因、error故障；返回：与流一致的完整更新；异常：流写入错误。"""
            update = stopped_state(reason, ledger)
            # 断流时丢弃尚未发布的半个引用，不能把不完整内部标记刷到页面。
            suffix = ("\n\n" if parts else "") + update["answer"]
            parts.append(suffix)
            writer({"event": "delta", "data": {"delta": suffix}})
            publish_stage(state, "answer", "completed")
            return {**update, **degradation(state, "answer", error), "answer": "".join(parts)}

        try:
            # 正文模型只需要中文按钮语义；不传内部字段、枚举、身份或动作ID。完整说明仍先过原预算。
            messages = model_messages(answer_state, deps.settings, ANSWER_PROMPT, registry=deps.registry)
            async with aclosing(deps.model.stream_answer(messages, ledger)) as stream:
                async for delta in stream:
                    visible = citations.feed(delta)
                    if visible:
                        parts.append(visible)
                        writer({"event": "delta", "data": {"delta": visible}})
            # 尾部仍需判定单独角标；发布与落库始终累计同一份过滤后正文。
            tail = citations.feed("", final=True)
            if tail:
                parts.append(tail)
                writer({"event": "delta", "data": {"delta": tail}})
            # 程序事实前缀不代表政策说明已生成；只有过滤后的有效正文才算完成。
            if not "".join(parts[prefix_count:]).strip():
                raise WorkflowOutputError("正文为空")
        except (WorkflowBudgetExceeded, ContextBudgetExceeded) as error:
            return finish_failure("budget_exceeded", error)
        except DEPENDENCY_ERRORS as error:
            return finish_failure("answer_unavailable", error)
        except RuntimeError as error:
            # 只有已提交票据的正文故障在这里收尾；其他程序错误继续交由外层定位。
            if not receipt_prefix:
                raise
            return finish_failure("answer_unavailable", error)
        publish_stage(state, "answer", "completed")
        return {"answer": "".join(parts), "budget": ledger.snapshot()}

    async def recover_knowledge(state: ConversationState) -> ConversationState:
        """业务分支准备结束但尚无业务对象或事实时，只补一次知识检索。

        参数：state本轮未执行工具且未检索的终止决定；返回：知识证据或原回答/澄清出口；异常：取消及程序错误。
        """
        ledger = BudgetLedger.from_snapshot(state["budget"])
        update = {"retrieval_attempted": True, "optional_knowledge_probe": True}
        if any(knowledge_input_stop_reason(state[key]) for key in ("raw_query", "resolved_query", "standard_query")):
            return update
        publish_stage(state, "policy", "running")
        try:
            async with asyncio.timeout(deps.settings.knowledge_timeout_seconds):
                knowledge_state = deepcopy(state)
                outcome = await retrieve_verified_knowledge(knowledge_state, ledger, search=deps.search,
                    settings=deps.settings, query_expander=deps.query_expander, query=state["resolved_query"],
                    prepared_queries=[state["standard_query"]], chat=deps.chat)
        except (WorkflowBudgetExceeded, ContextBudgetExceeded):
            return {**update, **stopped_state("budget_exceeded", ledger),
                    "retrieval_snapshot": knowledge_state.get("retrieval_snapshot"),
                    "knowledge_faults": unresolved_knowledge_faults(knowledge_state)}
        except DEPENDENCY_ERRORS as error:
            publish_stage(state, "policy", "completed")
            return {**update, **stopped_state("retrieval_unavailable", ledger),
                    **degradation(state, "recover_knowledge", error),
                    "retrieval_snapshot": knowledge_state.get("retrieval_snapshot"),
                    "knowledge_faults": unresolved_knowledge_faults(knowledge_state)}
        publish_stage(state, "policy", "completed")
        update.update(outcome.model_dump(mode="json"), knowledge_cache=knowledge_state.get("knowledge_cache", {}),
                      knowledge_faults=unresolved_knowledge_faults(knowledge_state))
        if outcome.gate["passed"]:
            update.update(stop_reason="answer", context_notes=[])
        # 低分不把原业务问题封死，也不冒充一次有效回答；保留已有澄清流程。
        return update

    def after_decision(state: ConversationState) -> str:
        """参数：state决定结果；返回：tools或answer；异常：无。"""
        if state["pending_calls"]:
            return "tools"
        if (state.get("stop_reason") == "answer" and state.get("route") in {"knowledge", "order_policy"}
                and not state.get("route_ready")):
            # 回到根图继续原政策检索/选单；普通工具事实不能把政策证据闸短路。
            return "route"
        if (state.get("route") == "business" and state.get("stop_reason") == "answer"
                and not state.get("retrieval_attempted") and not state["tool_count"]
                and not state.get("order_id") and not _active_order_tokens(state["raw_query"])
                and not _trusted_history(state["history_turns"])[1]
                # 已有业务调用或回执（包括未完成状态）可直接说明，不让知识查询干扰恢复对账。
                and not any(message.get("tool_calls") or message.get("role") == "tool"
                            for turn in state["history_turns"] for message in turn.get("messages", []))
                and any(char.isalnum() for char in state["standard_query"])
                and not state["suggested_kinds"]):
            return "recover_knowledge"
        return "answer"

    def after_tools(state: ConversationState) -> str:
        """参数：state工具结果；返回：decision或answer；异常：无。"""
        return "answer" if state["stop_reason"] else "decision"

    graph = StateGraph(ConversationState)
    graph.add_node("decision", decision)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_node)
    graph.add_node("recover_knowledge", recover_knowledge)
    graph.add_conditional_edges(START, lambda state: after_decision(state) if state.get("pending_calls") or state.get("stop_reason")
                                else "answer" if state.get("route_ready") else "decision",
                                {"tools": "tools", "answer": "answer", "recover_knowledge": "recover_knowledge", "route": END, "decision": "decision"})
    graph.add_conditional_edges("decision", after_decision,
                               {"tools": "tools", "answer": "answer", "recover_knowledge": "recover_knowledge", "route": END})
    graph.add_edge("recover_knowledge", "answer")
    graph.add_conditional_edges("tools", after_tools, {"decision": "decision", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()
