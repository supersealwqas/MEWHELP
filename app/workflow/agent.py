"""由decision/tools/answer节点编排的只读ReAct子图，保留每个完整事实及停止边界。"""

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
from app.tools.registry import build_agent_registry
from app.workflow.prompts import ANSWER_PROMPT, DECISION_PROMPT
from app.workflow.state import ConversationState

if TYPE_CHECKING:
    from app.workflow.graph import WorkflowDependencies

_STOP_TEXT = {
    "budget_exceeded": "本轮问题和查询结果已达到处理额度，请拆分问题后继续。",
    "decision_limit": "本轮处理步骤已达到上限，请补充或拆分问题后继续。",
    "tool_call_limit": "本轮查询次数已达到上限，请分开查询剩余问题。",
    "duplicate_tool_call": "本轮已查询过相同信息，未重复查询。请补充需要继续处理的问题。",
    "invalid_decision": "本轮查询申请不完整，暂时无法继续，请补充有效信息后重试。",
    "tool_failed": "本轮查询未能完成，请稍后重试或选择人工客服协助。",
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
    history = assemble_turns([StoredTurn.model_validate(item) for item in state["history_turns"]])
    prepared = prepare_messages(history, convert_to_messages(state["messages"]), settings.token_budget,
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


def build_agent_graph(deps: "WorkflowDependencies"):
    """编译共享State的无独立saver子图。参数：deps实例依赖；返回：CompiledStateGraph；异常：编译错误。"""
    registry = build_agent_registry()

    async def decision(state: ConversationState) -> ConversationState:
        """执行有界内部决定，不发布内部JSON。参数：state当前事实；返回：申请或终止更新；异常：模型/审计错误。"""
        ledger = BudgetLedger.from_snapshot(state["budget"])
        if state["decision_count"] >= deps.settings.workflow_max_decisions:
            return stopped_state("decision_limit", ledger)
        try:
            tools = registry.all()
            messages = model_messages(state, deps.settings, DECISION_PROMPT, tools)
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
        return {"budget": ledger.snapshot(), "decision_count": state["decision_count"] + 1,
                "pending_calls": calls, "suggested_kinds": selected.suggested_actions,
                "stop_reason": selected.kind if selected.kind != "tools" else None}

    async def tools_node(state: ConversationState) -> ConversationState:
        """逐个执行已批准申请并写完整配对。参数：state待执行批次；返回：完整替换更新；异常：持久化错误/取消。"""
        # 延迟读取既有拒绝表，避免服务层接图时发生模块导入循环。
        from app.services.chat import fixed_reply as rejection_reply

        context = ToolContext.model_validate(state["context"])
        calls, messages = state["pending_calls"], list(state["messages"])
        seen_ids, seen_signatures = list(state["seen_call_ids"]), list(state["seen_call_signatures"])
        count, reason, answer = state["tool_count"], None, ""
        writer = get_stream_writer()
        await deps.chat.save_tool_calls(context, calls)
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})

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
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result.model_dump_json()})
        update = {"messages": messages, "pending_calls": [], "seen_call_ids": seen_ids,
                  "seen_call_signatures": seen_signatures, "tool_count": count}
        if reason:
            update.update(stopped_state(reason) if reason != "order_not_owned" else
                          {"answer": answer, "stop_reason": reason, "suggested_kinds": []})
        return update

    async def answer_node(state: ConversationState) -> ConversationState:
        """直接推送最终正文，不等待完整答案才发首帧。参数：state事实；返回：完整正文和账本；异常：正文/网络/取消。"""
        writer = get_stream_writer()
        if state["answer"]:
            writer({"event": "delta", "data": {"delta": state["answer"]}})
            return {"answer": state["answer"]}
        ledger = BudgetLedger.from_snapshot(state["budget"])
        try:
            instructions = {
                "create_ticket": "请点击本条回复后的“创建工单”按钮，在弹窗中确认后提交。",
                "handoff": "如需人工协助，可点击本条回复后的“转人工”按钮。",
            }
            action_text = ("本轮回复后可用按钮的操作说明如下：\n" +
                           "\n".join(instructions[kind] for kind in state["suggested_kinds"])) if state["suggested_kinds"] else "本轮回复后没有可操作的按钮，请按已有事实回答或追问。"
            # 正文模型只需要中文按钮语义；不传内部字段、枚举、身份或动作ID。完整说明仍先过原预算。
            answer_state = {**state, "messages": [*state["messages"],
                {"role": "system", "content": action_text}]}
            messages = model_messages(answer_state, deps.settings, ANSWER_PROMPT)
            parts = []
            async with aclosing(deps.model.stream_answer(messages, ledger)) as stream:
                async for delta in stream:
                    parts.append(delta)
                    writer({"event": "delta", "data": {"delta": delta}})
        except (WorkflowBudgetExceeded, ContextBudgetExceeded):
            update = stopped_state("budget_exceeded", ledger)
            writer({"event": "delta", "data": {"delta": update["answer"]}})
            return update
        return {"answer": "".join(parts), "budget": ledger.snapshot()}

    def after_decision(state: ConversationState) -> str:
        """参数：state决定结果；返回：tools或answer；异常：无。"""
        return "tools" if state["pending_calls"] else "answer"

    def after_tools(state: ConversationState) -> str:
        """参数：state工具结果；返回：decision或answer；异常：无。"""
        return "answer" if state["stop_reason"] else "decision"

    graph = StateGraph(ConversationState)
    graph.add_node("decision", decision)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "decision")
    graph.add_conditional_edges("decision", after_decision, {"tools": "tools", "answer": "answer"})
    graph.add_conditional_edges("tools", after_tools, {"decision": "decision", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()
