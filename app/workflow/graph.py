"""客服根图：政策证据过闸后才注入，知识低分不阻断业务只读查询，正常出口统一完成审计。"""

import asyncio
import json
import logging
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.config import Settings
from app.core.memory import ContextBudgetExceeded
from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded, budget_summary
from app.knowledge.answering import EvidenceCitation, edge_order
from app.knowledge.retrieval import HybridKnowledgeSearch
from app.knowledge.query_rewrite import QueryExpander
from app.repositories.chat import ChatRepository
from app.schemas.tools import ToolContext
from app.schemas.workflow import GateDecision, SuggestedAction, RefundAction
from app.tools.executor import ToolExecutor
from app.workflow.agent import build_agent_graph, model_messages, stopped_state
from app.workflow.model import WorkflowModelAdapter, intent_tool_schema, understanding_tool_schema
from app.workflow.prompts import INTENT_PROMPT, UNDERSTANDING_PROMPT
from app.workflow.routing import assess_evidence, citations_from_ranked, fixed_reply, knowledge_input_stop_reason, route_by_intent, order_candidate
from app.tools.business import list_user_orders, owns_order
from app.workflow.state import ConversationState, full_history_messages, message_row
from app.workflow.stages import publish_stage
from app.workflow.policy_context import policy_query, verified_order_text
from app.workflow.resilience import DEPENDENCY_ERRORS, degradation
from app.workflow.routing import original_understanding
from app.workflow.context import history_snapshot, log_history, select_evidence
from app.repositories.context import ContextRepository

logger = logging.getLogger(__name__)
# 订单适用性只采用正式退换货/售后政策及其引用的运费、会员规则；商品规格不进入此分支。
ORDER_POLICY_SOURCES = ("doc:returns-policy.md", "doc:after-sales-manual.md",
                        "doc:billing-shipping.md", "doc:member-benefits.md")


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


@dataclass
class WorkflowDependencies:
    """运行资源只留在依赖中，不写入图State。"""
    model: WorkflowModelAdapter  # 分类、决策与正文适配器。
    search: HybridKnowledgeSearch  # 知识检索器。
    executor: ToolExecutor  # 既有只读工具执行器。
    chat: ChatRepository  # 业务聊天仓储；Task5扩展完成事务。
    settings: Settings  # 本实例配置。
    query_expander: QueryExpander  # 同一上游和账本控制的政策扩写器。


def build_workflow(deps: WorkflowDependencies, *, checkpointer):
    """编译固定根图并注入官方保存器。参数：deps运行依赖、checkpointer保存器；返回：编译根图；异常：编译错误。"""
    context_repository = ContextRepository(deps.chat.factory) if hasattr(deps.chat, "factory") else None

    async def resolve_node(state: ConversationState) -> ConversationState:
        """原话透传并使用本实例预算初始化新轮。参数：state入口全量新字典；返回：本轮初始更新；异常：配置错误。"""
        query = state["raw_query"]
        ledger = BudgetLedger.from_settings(deps.settings)
        snapshot = None
        publish_stage(state, "understand", "running")
        try:
            stored = await context_repository.load_context(state["context"]["conversation_id"], include_turns=False) if context_repository else None
            snapshot = history_snapshot(state, deps.settings, stored)
            log_history(state, snapshot)
            if snapshot["layer1_from_msg_id"] > snapshot["previous_layer1_from_msg_id"]:
                advanced = await context_repository.advance_layer1(state["context"]["conversation_id"],
                    expected_summary_upto=snapshot["summary_upto_msg_id"], upto_msg_id=snapshot["layer1_from_msg_id"]) if context_repository else False
                logger.info("层1 降级 %s→%s conversation=%s turns=%s tokens=%s→%s committed=%s",
                    snapshot["previous_layer1_from_msg_id"], snapshot["layer1_from_msg_id"], state["context"]["conversation_id"],
                    snapshot["demoted_turns"], snapshot["layer1_before_tokens"], snapshot["layer1_tokens"], advanced)
            current = {**state, "history_context": snapshot}
            understood = await deps.model.understand(model_messages(current, deps.settings, UNDERSTANDING_PROMPT,
                [understanding_tool_schema()]), ledger)
        except (WorkflowBudgetExceeded, ContextBudgetExceeded) as error:
            if snapshot is None:
                log_history(state, {"summary": "", "layer1": [], "layer2": []}, error_type=type(error).__name__)
            return {**stopped_state("budget_exceeded", ledger), "history_context": snapshot}
        except DEPENDENCY_ERRORS as error:
            # 理解只是优化；失败时保留完整原话，让后续分类仍有机会工作。
            understood = original_understanding(query)
            publish_stage(state, "understand", "completed", resolved_query=query, standard_query=query)
            return {**understood.model_dump(), "budget": ledger.snapshot(), "history_context": snapshot,
                    **degradation(state, "understand", error)}
        publish_stage(state, "understand", "completed", resolved_query=understood.resolved_query,
                      standard_query=understood.standard_query)
        return {"resolved_query": understood.resolved_query, "standard_query": understood.standard_query,
                "order_id": understood.order_id, "budget": ledger.snapshot(), "history_context": snapshot}

    async def classify_node(state: ConversationState) -> ConversationState:
        """仅调用受限分类器，失败不猜意图。参数：state原问和历史；返回：严格分类与账本；异常：模型/分类/预算错误。"""
        ledger = BudgetLedger.from_snapshot(state["budget"])
        publish_stage(state, "classify", "running")
        try:
            result = await deps.model.classify(model_messages(state, deps.settings, INTENT_PROMPT, [intent_tool_schema()]), ledger)
        except (WorkflowBudgetExceeded, ContextBudgetExceeded):
            return stopped_state("budget_exceeded", ledger)
        except DEPENDENCY_ERRORS as error:
            publish_stage(state, "classify", "completed")
            return {**stopped_state("model_unavailable", ledger), **degradation(state, "classify", error)}
        update = {"intent": result.intent, "intent_confidence": result.confidence}
        publish_stage(state, "classify", "completed", intent=result.intent, confidence=result.confidence)
        route = route_by_intent({**state, **update})
        logger.info("workflow intent=%s route=%s", result.intent, route)
        return {**update, "route": route, "budget": ledger.snapshot()}

    async def fetch_order(state: ConversationState) -> ConversationState:
        """核实本人订单后确定性查询；暂停前不写入。

        参数：state含可信身份、完整问题和候选号。
        返回：成功订单事实与完整工具审计消息；无订单时固定结束。
        异常：GraphInterrupt正常等待、ValueError非法恢复、数据库错误。
        """
        if any(knowledge_input_stop_reason(state[key]) for key in ("raw_query", "resolved_query", "standard_query")):
            return {"stop_reason": "input_limit"}
        context = ToolContext.model_validate(state["context"])
        orders = list_user_orders(context.user_id)
        if not orders:
            return {"answer": "暂未找到您的订单，请核对当前账号或联系人工客服。", "stop_reason": "no_owned_orders"}
        identifier = order_candidate(state, orders)
        action = {"type": "select_order", "action_id": str(uuid5(NAMESPACE_URL,
            f"mewhelp:ch06:{context.conversation_id}:{context.turn_id}:select_order")),
            "source_turn_id": context.turn_id, "orders": orders}
        if identifier is None:
            identifier = interrupt(action)
            action = {**action, "status": "accepted", "accepted_order_id": identifier}
        else:
            action = None
        # 恢复值也不可信，必须再次核对完整订单号及当前归属。
        if not isinstance(identifier, str) or identifier not in {row["order_id"] for row in orders} or not owns_order(context.user_id, identifier):
            raise ValueError("所选订单不属于当前用户")
        publish_stage(state, "order", "running")
        call = {"id": f"ch06:order:{context.turn_id}:{identifier}", "name": "query_order", "args": {"order_id": identifier}}
        await deps.chat.save_tool_calls(context, [call], deduplicate=True)

        async def on_status(frame: dict) -> None:
            """发布工具安全状态。参数：frame执行状态；返回：None；异常：流错误。"""
            get_stream_writer()({"event": "tool_status", "data": frame})

        result = await deps.executor.execute(call, context.model_copy(update={"tool_call_id": call["id"]}), on_status)
        messages = [message_row(context.turn_id, "assistant", "", calls=[call]),
                    message_row(context.turn_id, "tool", result.model_dump_json(), call_id=call["id"])]
        update = {"messages": messages, "order_id": identifier, "tool_count": state["tool_count"] + 1}
        if not result.ok or result.data.get("order_id") != identifier:
            return {**update, **stopped_state("tool_failed")}
        publish_stage(state, "order", "completed", order_id=identifier)
        return {**update, "order_data": result.data, "selection_action": action}

    async def retrieve_node(state: ConversationState) -> ConversationState:
        """按类别取得原文，依赖故障单独降级。参数：state已分类原问；返回：证据/分数/账本或安全回复；异常：取消及程序错误。"""
        reason = next((knowledge_input_stop_reason(state[key]) for key in ("raw_query", "resolved_query", "standard_query")
                       if knowledge_input_stop_reason(state[key])), None)
        if reason:
            return {"stop_reason": reason}
        ledger = BudgetLedger.from_snapshot(state["budget"])
        try:
            async with asyncio.timeout(deps.settings.knowledge_timeout_seconds):
                if state["route"] == "order_policy":
                    query = policy_query(state)
                    if knowledge_input_stop_reason(query):
                        return {"stop_reason": "input_limit"}
                    publish_stage(state, "expand", "running")
                    expanded = await deps.query_expander.expand(query, simple_query=state["standard_query"],
                                                                required_count=3, budget=ledger)
                    publish_stage(state, "expand", "completed", queries=expanded.queries)
                    prepared_queries = expanded.queries
                    retrieve = deps.search.retrieve_policy
                    options = {"filters": {"source_key": list(ORDER_POLICY_SOURCES)}}
                else:
                    query, prepared_queries, retrieve = state["resolved_query"], [state["standard_query"]], deps.search.retrieve
                    options = {}
                publish_stage(state, "policy", "running")
                ranked = await retrieve(query, prepared_queries=prepared_queries, budget=ledger, history="", **options)
        except (WorkflowBudgetExceeded, ContextBudgetExceeded):
            return stopped_state("budget_exceeded", ledger)
        except DEPENDENCY_ERRORS as error:
            # 依赖故障不能冒充低分证据，也不能把越界或失配原文交给模型。
            update = stopped_state("retrieval_unavailable", ledger)
            if state.get("order_data"):
                update["answer"] = verified_order_text(state) + update["answer"]
            publish_stage(state, "policy", "completed")
            return {**update, **degradation(state, "retrieve", error), "retrieval_attempted": True,
                    "evidence": [], "evidence_scores": [], "gate": None}
        citations = citations_from_ranked(ranked)
        publish_stage(state, "policy", "completed")
        # 检索结果先交给证据闸；尚未通过的内容不能进入模型消息。
        return {"evidence": [item.model_dump(mode="json") for item in citations],
                "evidence_scores": [item.score for item in ranked], "budget": ledger.snapshot(),
                "expanded_queries": prepared_queries, "retrieval_attempted": True}

    def confidence_node(state: ConversationState) -> ConversationState:
        """控制政策证据是否可用，业务仍可查事实。参数：state完整分数；返回：闸、可用证据与消息；异常：分数非法。"""
        gate = assess_evidence(state["evidence_scores"], deps.settings.rerank_min_score)
        logger.info("workflow gate passed=%s max_score=%s", gate.passed, gate.max_score)
        update = {"gate": gate.model_dump(mode="json"), "stop_reason": None}
        if gate.passed:
            # 只发送可完整容纳的块，引用仍使用原编号，不能引用未发给模型的资料。
            update["evidence"] = select_evidence(state["evidence"], deps.settings)
            update["context_notes"] = []
        else:
            # 门闸约束政策证据，不代表订单/物流工具不可用；分数和失败原因仍保留供审计。
            update["evidence"] = []
            if state["route"] == "order_policy":
                update["context_notes"] = [
                    "本轮没有可用知识政策证据。请继续核实当前问题所需的订单、物流或商品事实，"
                    "可以使用可信历史工具结果，缺少时调用对应只读工具。"
                    "只回答已核实的业务事实；若用户还问了政策条件，说明该部分暂无法可靠确认。"
                    "不能用历史助手正文补充政策，不添加政策条款、费用承诺或知识引用。"]
            else:
                update["stop_reason"] = "retrieval_low_conf"
        return update

    def fixed_node(state: ConversationState, kind: str) -> ConversationState:
        """固定出口直接发布正文。参数：state当前轮、kind出口；返回：固定更新；异常：未知出口。"""
        if state.get("answer"):
            update = {"answer": state["answer"]}
        elif state["stop_reason"] in {"budget_exceeded", "tool_failed"}:
            update = stopped_state(state["stop_reason"])
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
        metadata.update({"intent_confidence": state.get("intent_confidence"), "resolved_query": state["resolved_query"],
                         "standard_query": state["standard_query"], "order_id": state.get("order_id")})
        metadata["degraded_stages"] = [item["stage"] for item in state.get("degradations", [])]
        refund_action = None
        if state["intent"] == "退款退货" and state.get("order_data") and state.get("order_id"):
            refund_action = RefundAction(action_id=str(uuid5(NAMESPACE_URL,
                f"mewhelp:ch06:{context.conversation_id}:{context.turn_id}:submit_refund")),
                source_turn_id=context.turn_id, question=state["raw_query"], order_id=state["order_id"])
        saved = await deps.chat.finish_turn(context, state["answer"], low_confidence=low_confidence,
            citations=[EvidenceCitation.model_validate(item) for item in state["evidence"]],
            suggested_actions=actions, workflow_metadata=metadata, refund_action=refund_action)
        # 完成事务后增量回读，使用同一稳定消息ID补主键，不能另追加第二份客服回答。
        delta = await deps.chat.load_history_delta(context.conversation_id, state["audit_cursor"], refresh_turn_id=context.turn_id)
        history = {turn["turn_id"]: turn for turn in state["history_turns"]}
        history.update({turn.turn_id: turn.model_dump(mode="json") for turn in delta.turns})
        logger.info("workflow completed turn=%s reason=%s tools=%s decisions=%s", context.turn_id,
                    state["stop_reason"], state["tool_count"], state["decision_count"])
        return {"suggested_actions": [action.model_dump(mode="json") for action in actions], "completion_metadata": saved,
                "messages": full_history_messages([turn.model_dump(mode="json") for turn in delta.turns]),
                "history_turns": list(history.values()), "audit_cursor": delta.last_message_id,
                "refund_action": refund_action.model_dump(mode="json") if refund_action else None}

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
    graph.add_node("fetch_order", fetch_order)
    graph.add_node("retrieve_policy", retrieve_node)
    graph.add_node("confidence_check", confidence_node)
    graph.add_node("main_agent", build_agent_graph(deps))
    graph.add_node("complaint_reply", complaint_node)
    graph.add_node("chitchat_reply", chitchat_node)
    graph.add_node("fallback_reply", fallback_node)
    graph.add_node("log_turn", log_node)
    graph.add_edge(START, "resolve_reference")
    graph.add_conditional_edges("resolve_reference", lambda state: "fallback_reply" if state["stop_reason"] else "classify_intent")
    graph.add_conditional_edges("classify_intent", lambda state: "fallback_reply" if state["stop_reason"] else {
        "knowledge": "retrieve_knowledge", "business": "main_agent", "order_policy": "fetch_order",
        "complaint": "complaint_reply", "chitchat": "chitchat_reply"}[state["route"]])
    graph.add_conditional_edges("fetch_order", lambda state: "fallback_reply" if state["stop_reason"] else "retrieve_policy")
    graph.add_conditional_edges("retrieve_policy", after_retrieval, {"confidence_check": "confidence_check", "fallback_reply": "fallback_reply"})
    graph.add_conditional_edges("retrieve_knowledge", after_retrieval, {"confidence_check": "confidence_check", "fallback_reply": "fallback_reply"})
    graph.add_conditional_edges("confidence_check", after_confidence, {"main_agent": "main_agent", "fallback_reply": "fallback_reply"})
    for node in ("main_agent", "complaint_reply", "chitchat_reply", "fallback_reply"):
        graph.add_edge(node, "log_turn")
    graph.add_edge("log_turn", END)
    return graph.compile(checkpointer=checkpointer).with_config({"recursion_limit": 64})
