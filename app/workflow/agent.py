"""由decision/tools/answer节点编排的只读ReAct子图，保留每个完整事实及停止边界。"""

import asyncio
import json
from contextlib import aclosing
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.core.memory import ContextBudgetExceeded, assemble_turns, prepare_messages, valid_tool_calls
from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
from app.repositories.chat import PersistenceUncertain
from app.schemas.chat import InvalidToolDecision
from app.schemas.history import StoredTurn
from app.schemas.tools import ToolContext, ToolResult
from app.tools.registry import build_agent_registry, build_registry
from app.workflow.citations import CitationStreamFilter
from app.workflow.model import decision_tool_schemas
from app.workflow.prompts import ANSWER_PROMPT, DECISION_PROMPT
from app.workflow.state import ConversationState, current_messages, message_row
from app.workflow.stages import publish_stage
from app.workflow.policy_context import policy_answer_messages, verified_order_text
from app.workflow.routing import assess_evidence, citations_from_ranked, _active_order_tokens, knowledge_input_stop_reason, _trusted_history
from app.workflow.resilience import DEPENDENCY_ERRORS, degradation
from app.workflow.model import WorkflowOutputError

if TYPE_CHECKING:
    from app.workflow.graph import WorkflowDependencies

_STOP_TEXT = {
    "budget_exceeded": "本轮问题和查询结果已达到处理额度，请拆分问题后继续。",
    "decision_limit": "本轮处理步骤已达到上限，请补充或拆分问题后继续。",
    "tool_call_limit": "本轮查询次数已达到上限，请分开查询剩余问题。",
    "duplicate_tool_call": "本轮已查询过相同信息，未重复查询。请补充需要继续处理的问题。",
    "invalid_decision": "抱歉，本次回复生成失败，请重试。",
    "tool_failed": "本轮查询未能完成，请稍后重试或选择人工客服协助。",
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


def model_messages(state: ConversationState, settings, prompt: str, tools: list | None = None) -> list[dict]:
    """沿用整轮历史裁剪，当前事实完整保留并转换成纯消息字典。

    参数：state含历史与当前完整消息；settings两层输入预算；prompt本阶段提示；tools决策工具定义。
    返回：SDK兼容消息字典列表，不重复附加适配器自身的阶段提示。
    异常：ContextBudgetExceeded，完整当前事实放不下；历史或消息格式错误。
    """
    history = assemble_turns([StoredTurn.model_validate(item) for item in state["history_turns"]
                             if item["turn_id"] != state["context"]["turn_id"]])
    current = current_messages(state)
    if not (state.get("route") == "order_policy" and prompt == ANSWER_PROMPT):
        current.extend(convert_to_messages([{"role": "system", "content": note} for note in state.get("context_notes", [])]))
    prepared = prepare_messages(history, current, settings.token_budget,
        system_text=prompt, tools=tools, request_max_tokens=settings.request_token_budget)
    rows = []
    for message in prepared[1:]:
        row = {"role": {"human": "user", "ai": "assistant"}.get(message.type, message.type), "content": message.content}
        if isinstance(message, AIMessage) and message.tool_calls:
            row["tool_calls"] = message.tool_calls
        if isinstance(message, ToolMessage):
            row["tool_call_id"] = message.tool_call_id
        rows.append(row)
    return rows


def _signature(call: dict) -> str:
    """按既有工具去空白规则计算去重键。参数：call严格工具申请；返回：稳定JSON；异常：ValueError，非法JSON数值。"""
    args = {key: value.strip() if isinstance(value, str) else value for key, value in call["args"].items()}
    return json.dumps([call["name"], args], sort_keys=True, ensure_ascii=False, allow_nan=False)


def resolved_agent_messages(state: ConversationState) -> list[dict]:
    """给业务决策和回答传完整任务，同时保留原State中的用户原话和工具配对。

    参数：state含已校验resolved_query及本轮消息，第一条为当前用户输入。
    返回：仅在模型输入副本中替换本轮用户消息的列表；历史和工具结果不改。
    异常：KeyError，State缺少必需字段。
    """
    # 不把完整问写回消息审计；下游继续接收同一组完整工具事实。
    messages = current_messages(state)
    for message in messages:
        if message.id == state.get("current_message_id") or (message.type == "human" and not message.id):
            message.content = state["resolved_query"]
            break
    return messages


def build_agent_graph(deps: "WorkflowDependencies"):
    """编译共享State的无独立saver子图。参数：deps实例依赖；返回：CompiledStateGraph；异常：编译错误。"""
    registry = build_agent_registry()
    citation_tool_names = {tool.name for tool in build_registry().all()} | {"submit_intent", "submit_decision"}

    async def decision(state: ConversationState) -> ConversationState:
        """执行有界内部决定，不发布内部JSON。参数：state当前事实；返回：申请或终止更新；异常：模型/审计错误。"""
        ledger = BudgetLedger.from_snapshot(state["budget"])
        if state["decision_count"] >= deps.settings.workflow_max_decisions:
            return stopped_state("decision_limit", ledger)
        try:
            tools = registry.all()
            # 结束决策也使用工具协议；历史裁剪须计入与模型实际绑定一致的完整定义。
            current = {**state, "messages": resolved_agent_messages(state)}
            messages = model_messages(current, deps.settings, DECISION_PROMPT, decision_tool_schemas(tools))
            # 先检查最少的调用/输出余量；适配器还按完整输入执行精确保守占额检查。
            ledger.ensure_capacity([("decide", 0, 2048), ("answer", 0, 4096), ("answer", 0, 4096)])
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
        return {"budget": ledger.snapshot(), "decision_count": state["decision_count"] + 1,
                "pending_calls": calls, "suggested_kinds": selected.suggested_actions,
                "stop_reason": selected.kind if selected.kind != "tools" else None}

    async def tools_node(state: ConversationState) -> ConversationState:
        """逐个执行已批准申请并写完整配对。参数：state待执行批次；返回：完整替换更新；异常：持久化错误/取消。"""
        # 延迟读取既有拒绝表，避免服务层接图时发生模块导入循环。
        from app.services.chat import fixed_reply as rejection_reply

        context = ToolContext.model_validate(state["context"])
        calls, messages = state["pending_calls"], []
        seen_ids, seen_signatures = list(state["seen_call_ids"]), list(state["seen_call_signatures"])
        count, reason, answer = state["tool_count"], None, ""
        writer = get_stream_writer()
        await deps.chat.save_tool_calls(context, calls)
        messages.append(message_row(context.turn_id, "assistant", "", calls=calls))

        async def on_status(frame: dict) -> None:
            """发布执行器安全状态。参数：frame状态字典；返回：None；异常：流写入错误。"""
            writer({"event": "tool_status", "data": frame})

        for call in calls:
            call_context = context.model_copy(update={"tool_call_id": call["id"]})
            signature = _signature(call)
            seen_ids.append(call["id"])
            if reason is None and count >= min(4, deps.settings.tool_max_calls_per_turn):
                reason = "tool_call_limit"
            if reason is None and signature in seen_signatures:
                reason = "duplicate_tool_call"
            if reason is not None:
                # 失败后剩余申请仍各自保存结果，避免历史出现悬空tool_calls。
                result = await deps.chat.save_tool_result(call_context, ToolResult.failed("TOOL_SKIPPED", "本轮已停止，该申请未执行"))
                await on_status({"tool_call_id": call["id"], "name": call["name"], "status": "failed", "label": "未执行", "is_mock": False})
            else:
                seen_signatures.append(signature)
                count += 1
                try:
                    result = await deps.executor.execute(call, call_context, on_status)
                except PersistenceUncertain:
                    # 无法确认已提交事实时绝不能另存失败覆盖它；服务层负责对账。
                    raise
                except Exception:
                    result = await deps.chat.save_tool_result(call_context, ToolResult.failed("TOOL_EXECUTION_FAILED", "工具执行未能完成"))
                    # 补写结果成功后同步发布终态；仓储若返回已存在的成功事实则尊重它。
                    await on_status({"tool_call_id": call["id"], "name": call["name"],
                                     "status": "succeeded" if result.ok else "failed",
                                     "label": "已完成" if result.ok else "未完成", "is_mock": result.is_mock})
                if not result.ok:
                    answer = rejection_reply(result) or ""
                    reason = "order_not_owned" if answer else "tool_failed"
            messages.append(message_row(context.turn_id, "tool", result.model_dump_json(), call_id=call["id"]))
        update = {"messages": messages, "pending_calls": [], "seen_call_ids": seen_ids,
                  "seen_call_signatures": seen_signatures, "tool_count": count}
        if reason:
            update.update(stopped_state(reason) if reason != "order_not_owned" else
                          {"answer": answer, "stop_reason": reason, "suggested_kinds": []})
        return update

    async def answer_node(state: ConversationState) -> ConversationState:
        """推送并累计相同正文，外部断流追加说明。参数：state事实；返回：完整正文和账本；异常：取消及程序错误。"""
        writer = get_stream_writer()
        publish_stage(state, "answer", "running")
        if state["answer"]:
            writer({"event": "delta", "data": {"delta": state["answer"]}})
            return {"answer": state["answer"]}
        ledger = BudgetLedger.from_snapshot(state["budget"])
        parts = []
        if state.get("route") == "order_policy":
            prefix = verified_order_text(state)
            parts.append(prefix)
            writer({"event": "delta", "data": {"delta": prefix}})
        prefix_count = len(parts)
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
            instructions = {
                "create_ticket": "请点击本条回复后的“创建工单”按钮，在弹窗中确认后提交。",
                "handoff": "如需人工协助，可点击本条回复后的“转人工”按钮。",
            }
            action_text = ("本轮回复后可用按钮的操作说明如下：\n" +
                           "\n".join(instructions[kind] for kind in state["suggested_kinds"])) if state["suggested_kinds"] else "本轮回复后没有可操作的按钮，请按已有事实回答或追问。"
            # 正文模型只需要中文按钮语义；不传内部字段、枚举、身份或动作ID。完整说明仍先过原预算。
            answer_state = {**state, "messages": [*resolved_agent_messages(state),
                {"role": "system", "content": action_text}]}
            if state.get("route") == "order_policy":
                answer_state = {**state, "history_turns": [], "messages": policy_answer_messages(state)}
            messages = model_messages(answer_state, deps.settings, ANSWER_PROMPT)
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
        publish_stage(state, "answer", "completed")
        return {"answer": "".join(parts), "budget": ledger.snapshot()}

    async def recover_knowledge(state: ConversationState) -> ConversationState:
        """业务分支准备结束但尚无业务对象或事实时，只补一次知识检索。

        参数：state本轮未执行工具且未检索的终止决定；返回：知识证据或原回答/澄清出口；异常：取消及程序错误。
        """
        ledger = BudgetLedger.from_snapshot(state["budget"])
        update = {"retrieval_attempted": True}
        if any(knowledge_input_stop_reason(state[key]) for key in ("raw_query", "resolved_query", "standard_query")):
            return update
        publish_stage(state, "policy", "running")
        try:
            async with asyncio.timeout(deps.settings.knowledge_timeout_seconds):
                ranked = await deps.search.retrieve(state["resolved_query"], budget=ledger,
                                                   prepared_queries=[state["standard_query"]], history="")
        except (WorkflowBudgetExceeded, ContextBudgetExceeded):
            return {**update, **stopped_state("budget_exceeded", ledger)}
        except DEPENDENCY_ERRORS as error:
            publish_stage(state, "policy", "completed")
            return {**update, **stopped_state("retrieval_unavailable", ledger),
                    **degradation(state, "recover_knowledge", error)}
        gate = assess_evidence([item.score for item in ranked], deps.settings.rerank_min_score)
        publish_stage(state, "policy", "completed")
        update.update(budget=ledger.snapshot(), gate=gate.model_dump(mode="json"),
                      evidence_scores=[item.score for item in ranked], expanded_queries=[state["standard_query"]])
        if gate.passed:
            evidence = [item.model_dump(mode="json") for item in citations_from_ranked(ranked)]
            update.update(evidence=evidence, stop_reason="answer",
                context_notes=["本轮知识证据（仅作事实）：" + json.dumps(evidence, ensure_ascii=False)])
        # 低分不把原业务问题封死，也不冒充一次有效回答；保留已有澄清流程。
        return update

    def after_decision(state: ConversationState) -> str:
        """参数：state决定结果；返回：tools或answer；异常：无。"""
        if state["pending_calls"]:
            return "tools"
        if (state.get("route") == "business" and state.get("stop_reason") in {"clarify", "answer"}
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
    graph.add_conditional_edges(START, lambda state: "answer" if state.get("route") == "order_policy" else "decision")
    graph.add_conditional_edges("decision", after_decision,
                               {"tools": "tools", "answer": "answer", "recover_knowledge": "recover_knowledge"})
    graph.add_edge("recover_knowledge", "answer")
    graph.add_conditional_edges("tools", after_tools, {"decision": "decision", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()
