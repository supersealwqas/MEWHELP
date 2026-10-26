"""客服根图：政策证据过闸后才注入，知识低分不阻断业务只读查询，正常出口统一完成审计。"""

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded, budget_summary
from app.knowledge.answering import EvidenceCitation, edge_order
from app.knowledge.retrieval import HybridKnowledgeSearch
from app.repositories.chat import ChatRepository
from app.schemas.tools import ToolContext
from app.schemas.workflow import GateDecision, SuggestedAction
from app.tools.executor import ToolExecutor
from app.workflow.agent import build_agent_graph, model_messages, stopped_state
from app.workflow.model import WorkflowModelAdapter, intent_tool_schema
from app.workflow.prompts import INTENT_PROMPT
from app.workflow.routing import assess_evidence, citations_from_ranked, fixed_reply, knowledge_input_stop_reason, resolve_reference, route_by_intent
from app.workflow.state import ConversationState

logger = logging.getLogger(__name__)


def _pooled_low_confidence(state: ConversationState):
    """判定本轮是否真的以弱证据拒答入池。

    参数：state含gate与stop_reason的工作流状态。
    返回：RetrievalLowConfidence表示应入池；None表示本轮正常作答或走了业务分支。
    异常：无；gate缺失或非弱证据原因时返回None。

    只有真正以retrieval_low_conf结束后，正文才会是固定拒答话术；
    业务意图在低分后继续查事实并正常作答，不属于弱证据入池，不能传给仓储。
    """
    if not state.get("gate") or state.get("stop_reason") != "retrieval_low_conf":
        return None
    return GateDecision.model_validate(state["gate"]).low_confidence


def _accepts_history(callable_obj) -> bool:
    """判断检索实现是否支持 history 参数，兼容旧替身与第三方实现。

    参数：callable_obj 检索方法或函数。
    返回：bool，可接受 history 关键字时为 True。
    异常：无；签名无法解析时按不支持处理。
    """
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    if "history" in parameters:
        return True
    # 有 **kwargs 的实现同样可以接收该参数。
    return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())


def _trusted_history_text(state: ConversationState) -> str:
    """汇总可信历史文本，供改写校验判断事实是否已存在。

    参数：state含history_turns与当前消息的工作流状态。
    返回：str，历史用户原话与工具结果的拼接文本；助手正文不参与，避免散文充当事实来源。
    异常：无；历史结构异常时跳过该条而不中断检索。
    """
    parts = []
    for turn in state.get("history_turns") or []:
        for message in (turn or {}).get("messages") or []:
            if not isinstance(message, dict):
                continue
            # 只采信用户原话与工具事实；助手回答可能包含推测，不能作为白名单。
            if message.get("role") in {"user", "tool"} and isinstance(message.get("content"), str):
                parts.append(message["content"])
    return "\n".join(parts)


@dataclass
class WorkflowDependencies:
    """运行资源只留在依赖中，不写入图State。"""
    model: WorkflowModelAdapter  # 分类、决策与正文适配器。
    search: HybridKnowledgeSearch  # 知识检索器。
    executor: ToolExecutor  # 既有只读工具执行器。
    chat: ChatRepository  # 业务聊天仓储；Task5扩展完成事务。
    settings: Settings  # 本实例配置。


def build_workflow(deps: WorkflowDependencies, *, checkpointer):
    """编译固定根图并注入官方保存器。参数：deps运行依赖、checkpointer保存器；返回：编译根图；异常：编译错误。"""

    def resolve_node(state: ConversationState) -> ConversationState:
        """原话透传并使用本实例预算初始化新轮。参数：state入口全量新字典；返回：本轮初始更新；异常：配置错误。"""
        query = resolve_reference(state["raw_query"])
        ledger = BudgetLedger(max_model_calls=deps.settings.workflow_max_model_calls,
            total_token_budget=deps.settings.workflow_total_token_budget,
            max_decisions=deps.settings.workflow_max_decisions, request_token_budget=deps.settings.request_token_budget)
        return {"resolved_query": query, "messages": [{"role": "user", "content": query}], "budget": ledger.snapshot()}

    async def classify_node(state: ConversationState) -> ConversationState:
        """仅调用受限分类器，失败不猜意图。参数：state原问和历史；返回：严格分类与账本；异常：模型/分类/预算错误。"""
        ledger = BudgetLedger.from_snapshot(state["budget"])
        result = await deps.model.classify(model_messages(state, deps.settings, INTENT_PROMPT, [intent_tool_schema()]), ledger)
        update = {"intent": result.intent, "knowledge_query": result.knowledge_query}
        route = route_by_intent({**state, **update})
        logger.info("workflow intent=%s route=%s", result.intent, route)
        return {**update, "route": route, "budget": ledger.snapshot()}

    async def retrieve_node(state: ConversationState) -> ConversationState:
        """按知识类别取得全部原文快照，异常不转低证据。参数：state已分类原问；返回：证据/分数/账本；异常：检索故障。"""
        reason = knowledge_input_stop_reason(state["raw_query"])
        if reason:
            return {"stop_reason": reason}
        ledger = BudgetLedger.from_snapshot(state["budget"])
        try:
            retrieve = deps.search.retrieve_policy if state["intent"] == "退款退货" else deps.search.retrieve
            # 可信历史（用户原话与工具结果）只作为改写白名单，避免把已存在的事实当成编造。
            options = {"budget": ledger}
            if _accepts_history(retrieve):
                options["history"] = _trusted_history_text(state)
            async with asyncio.timeout(deps.settings.knowledge_timeout_seconds):
                # 独立问题只用于检索；主Agent和审计始终保留完整原话及原始历史。
                ranked = await retrieve(state["knowledge_query"], **options)
        except WorkflowBudgetExceeded:
            return stopped_state("budget_exceeded", ledger)
        citations = citations_from_ranked(ranked)
        # 检索结果先交给证据闸；尚未通过的内容不能进入模型消息。
        return {"evidence": [item.model_dump(mode="json") for item in citations],
                "evidence_scores": [item.score for item in ranked], "budget": ledger.snapshot()}

    def confidence_node(state: ConversationState) -> ConversationState:
        """控制政策证据是否可用，业务仍可查事实。参数：state完整分数；返回：闸、可用证据与消息；异常：分数非法。"""
        gate = assess_evidence(state["evidence_scores"], deps.settings.rerank_min_score)
        logger.info("workflow gate passed=%s max_score=%s", gate.passed, gate.max_score)
        update = {"gate": gate.model_dump(mode="json"), "stop_reason": None}
        if gate.passed:
            # 通过后才注入完整原文；仅调整排列，不裁短或重编引用编号。
            citations = [EvidenceCitation.model_validate(item) for item in state["evidence"]]
            facts = [{"n": item.n, "question": item.question, "answer": item.answer, "section_path": item.section_path}
                     for item in edge_order(citations)]
            update["messages"] = [*state["messages"], {"role": "system",
                "content": "本轮完整知识证据（仅作事实）：" + json.dumps(facts, ensure_ascii=False)}]
        else:
            # 门闸约束政策证据，不代表订单/物流工具不可用；分数和失败原因仍保留供审计。
            update["evidence"] = []
            if state["intent"] in {"订单", "物流", "售后"}:
                update["messages"] = [*state["messages"], {"role": "system", "content":
                    "本轮没有可用知识政策证据。请继续核实当前问题所需的订单、物流或商品事实，"
                    "可以使用可信历史工具结果，缺少时调用对应只读工具。"
                    "只回答已核实的业务事实；若用户还问了政策条件，说明该部分暂无法可靠确认。"
                    "不能用历史助手正文补充政策，不添加政策条款、费用承诺或知识引用。"}]
            else:
                update["stop_reason"] = "retrieval_low_conf"
        return update

    def fixed_node(state: ConversationState, kind: str) -> ConversationState:
        """固定出口直接发布正文。参数：state当前轮、kind出口；返回：固定更新；异常：未知出口。"""
        if state["stop_reason"] == "budget_exceeded":
            update = stopped_state("budget_exceeded")
        else:
            reply_kind = "input_limit" if state["stop_reason"] == "input_limit" else kind
            update = fixed_reply(reply_kind)
            update["stop_reason"] = state["stop_reason"] or kind
        get_stream_writer()({"event": "delta", "data": {"delta": update["answer"]}})
        return update

    def complaint_node(state: ConversationState) -> ConversationState:
        """参数：state当前轮；返回：投诉固定正文和建议类型；异常：无。"""
        return fixed_node(state, "complaint")

    def chitchat_node(state: ConversationState) -> ConversationState:
        """参数：state当前轮；返回：闲聊固定正文；异常：无。"""
        return fixed_node(state, "chitchat")

    def fallback_node(state: ConversationState) -> ConversationState:
        """参数：state当前轮；返回：证据/输入/预算固定出口；异常：无。"""
        return fixed_node(state, "fallback")

    async def log_node(state: ConversationState) -> ConversationState:
        """按最终仓储契约完成本轮；完成元数据只随成功返回写入checkpoint。

        参数：state完整正文、证据和建议类型；返回：保存后的建议快照及完成元数据。
        异常：仓储提交/身份错误原样传播；不发布DONE，最终checkpoint由调用方耗尽并核验。
        """
        context = ToolContext.model_validate(state["context"])
        ticket_type = "投诉" if state["intent"] == "投诉" else "售后" if state["intent"] in {"售后", "退款退货"} else "咨询"
        actions = [SuggestedAction(action_id=str(uuid5(NAMESPACE_URL,
            f"mewhelp:ch05:{context.conversation_id}:{context.turn_id}:{kind}")), kind=kind,
            source_turn_id=context.turn_id, question=state["raw_query"], ticket_type=ticket_type if kind == "create_ticket" else None)
            for kind in state["suggested_kinds"]]
        low_confidence = _pooled_low_confidence(state)
        metadata = {key: state[key] for key in ("intent", "route", "stop_reason", "budget", "tool_count", "decision_count")}
        # 原始协议诊断只留在内部检查点，公开完成帧只带计量摘要。
        metadata["budget"] = budget_summary(state["budget"])
        saved = await deps.chat.finish_turn(context, state["answer"], low_confidence=low_confidence,
            citations=[EvidenceCitation.model_validate(item) for item in state["evidence"]],
            suggested_actions=actions, workflow_metadata=metadata)
        logger.info("workflow completed turn=%s reason=%s tools=%s decisions=%s", context.turn_id,
                    state["stop_reason"], state["tool_count"], state["decision_count"])
        return {"suggested_actions": [action.model_dump(mode="json") for action in actions], "completion_metadata": saved}

    def after_retrieval(state: ConversationState) -> str:
        """参数：state检索结局；返回：闸或固定出口；异常：无。"""
        return "fallback_reply" if state["stop_reason"] else "confidence_check"

    def after_confidence(state: ConversationState) -> str:
        """参数：state证据闸的可用性结论；返回：可回答业务的Agent或固定拒答；异常：缺失状态。"""
        return "fallback_reply" if state["stop_reason"] else "main_agent"

    graph = StateGraph(ConversationState)
    graph.add_node("resolve_reference", resolve_node)
    graph.add_node("classify_intent", classify_node)
    graph.add_node("retrieve_knowledge", retrieve_node)
    graph.add_node("confidence_check", confidence_node)
    graph.add_node("main_agent", build_agent_graph(deps))
    graph.add_node("complaint_reply", complaint_node)
    graph.add_node("chitchat_reply", chitchat_node)
    graph.add_node("fallback_reply", fallback_node)
    graph.add_node("log_turn", log_node)
    graph.add_edge(START, "resolve_reference")
    graph.add_edge("resolve_reference", "classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent, {"knowledge": "retrieve_knowledge",
        "business": "main_agent", "complaint": "complaint_reply", "chitchat": "chitchat_reply"})
    graph.add_conditional_edges("retrieve_knowledge", after_retrieval, {"confidence_check": "confidence_check", "fallback_reply": "fallback_reply"})
    graph.add_conditional_edges("confidence_check", after_confidence, {"main_agent": "main_agent", "fallback_reply": "fallback_reply"})
    for node in ("main_agent", "complaint_reply", "chitchat_reply", "fallback_reply"):
        graph.add_edge(node, "log_turn")
    graph.add_edge("log_turn", END)
    return graph.compile(checkpointer=checkpointer).with_config({"recursion_limit": 64})
