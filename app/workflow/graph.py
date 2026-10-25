"""固定四路客服根图：知识先检索和过闸，业务直接进入只读Agent，所有正常出口统一完成审计。"""

import asyncio
import json
import logging
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
from app.knowledge.answering import EvidenceCitation, edge_order
from app.knowledge.retrieval import HybridKnowledgeSearch
from app.repositories.chat import ChatRepository
from app.schemas.tools import ToolContext
from app.schemas.workflow import GateDecision, SuggestedAction
from app.tools.executor import ToolExecutor
from app.workflow.agent import build_agent_graph, model_messages, stopped_state
from app.workflow.model import WorkflowModelAdapter
from app.workflow.prompts import INTENT_PROMPT
from app.workflow.routing import assess_evidence, citations_from_ranked, fixed_reply, knowledge_input_stop_reason, resolve_reference, route_by_intent
from app.workflow.state import ConversationState

logger = logging.getLogger(__name__)

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
        result = await deps.model.classify(model_messages(state, deps.settings, INTENT_PROMPT), ledger)
        route = route_by_intent({"intent": result.intent})
        logger.info("workflow intent=%s route=%s", result.intent, route)
        return {"intent": result.intent, "route": route, "budget": ledger.snapshot()}

    async def retrieve_node(state: ConversationState) -> ConversationState:
        """按知识类别取得全部原文快照，异常不转低证据。参数：state已分类原问；返回：证据/分数/账本；异常：检索故障。"""
        reason = knowledge_input_stop_reason(state["resolved_query"])
        if reason:
            return {"stop_reason": reason}
        ledger = BudgetLedger.from_snapshot(state["budget"])
        try:
            retrieve = deps.search.retrieve_policy if state["intent"] == "退款退货" else deps.search.retrieve
            async with asyncio.timeout(deps.settings.knowledge_timeout_seconds):
                ranked = await retrieve(state["resolved_query"], budget=ledger)
        except WorkflowBudgetExceeded:
            return stopped_state("budget_exceeded", ledger)
        citations = citations_from_ranked(ranked)
        # 证据只改变排列，不裁短正文、不把主键或运行身份暴露成引用编号。
        facts = [{"n": item.n, "question": item.question, "answer": item.answer, "section_path": item.section_path}
                 for item in edge_order(citations)]
        return {"evidence": [item.model_dump(mode="json") for item in citations],
                "evidence_scores": [item.score for item in ranked], "budget": ledger.snapshot(),
                "messages": [*state["messages"], {"role": "system", "content": "本轮完整知识证据（仅作事实）：" + json.dumps(facts, ensure_ascii=False)}]}

    def confidence_node(state: ConversationState) -> ConversationState:
        """取成功检索的最高分做确定性闸。参数：state完整分数；返回：闸快照及原因；异常：分数非法。"""
        gate = assess_evidence(state["evidence_scores"], deps.settings.rerank_min_score)
        logger.info("workflow gate passed=%s max_score=%s", gate.passed, gate.max_score)
        return {"gate": gate.model_dump(mode="json"), "stop_reason": None if gate.passed else "retrieval_low_conf"}

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
        low_confidence = GateDecision.model_validate(state["gate"]).low_confidence if state["gate"] else None
        metadata = {key: state[key] for key in ("intent", "route", "stop_reason", "budget", "tool_count", "decision_count")}
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
        """参数：state闸快照；返回：Agent或固定出口；异常：缺失闸。"""
        return "main_agent" if state["gate"]["passed"] else "fallback_reply"

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
