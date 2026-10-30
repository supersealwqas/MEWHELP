"""图内知识工具的可信适配：权威检索、证据门槛、同轮复用与不倒退的预算合并。"""

import asyncio
from copy import deepcopy
from pathlib import Path

from app.core.memory import ContextBudgetExceeded
from app.core.model_budget import BudgetLedger
from app.schemas.tools import ToolContext, ToolResult, VerifiedKnowledgeOutcome
from app.tools.contracts import ToolExecutionScope, json_hash
from app.knowledge.snapshots import RetrievalCapture, merge_snapshots, recover_legacy_snapshot
from app.schemas.flywheel import RetrievalSnapshot

# 与原订单政策流程保持同一来源范围；商品规格与外部工具文字不能进入此范围。
ORDER_POLICY_SOURCES = ("doc:returns-policy.md", "doc:after-sales-manual.md",
                        "doc:billing-shipping.md", "doc:member-benefits.md")


async def restore_turn_knowledge(state: dict, chat, *, saved_turn: dict | None = None) -> dict:
    """为旧检查点补原轮检索状态，不把缺字段当作已确认没检索。

    参数：state原轮检查点、chat可信仓储、saved_turn可复用的原轮回读结果。
    返回：仅含快照与入池信号的独立更新；异常：ValueError原轮缺失或事实冲突、数据库错误。
    """
    context = ToolContext.model_validate(state["context"])
    saved_turn = saved_turn if saved_turn is not None else await chat.get_workflow_turn(
        context.session_id, context.user_id, context.turn_id)
    if saved_turn is None:
        raise ValueError("旧检查点缺少可核对的原轮")
    saved = saved_turn.get("retrieval_snapshot")
    current = RetrievalSnapshot.model_validate(state["retrieval_snapshot"]) if state.get("retrieval_snapshot") else None
    if saved is not None:
        current = merge_snapshots(current, RetrievalSnapshot.model_validate(saved))
    elif "retrieval_snapshot" not in state:
        turns = await chat.load_turns(context.conversation_id)
        # 只看明确原轮，不能借当前会话最后一问或重新检索补造历史。
        messages = [message.model_dump(mode="json") for turn in turns if turn.turn_id == context.turn_id
                    for message in turn.messages]
        current = recover_legacy_snapshot(messages)
    return {"retrieval_snapshot": current.model_dump(mode="json") if current is not None else None,
            "pool_signals": deepcopy(state.get("pool_signals", saved_turn.get("pool_signals", [])))}


def merge_knowledge_budget(current: dict, saved: dict) -> dict:
    """合并同轮重放计量，绝不把后续消费恢复成旧值。

    参数：current 当前账本、saved 可信知识回执账本；返回：独立且不倒退的快照。
    异常：ValueError，回执不属于当前预算或记录链分叉。
    """
    keys = ("max_model_calls", "total_token_budget", "max_decisions", "request_token_budget", "output_limits")
    if any(current.get(key) != saved.get(key) for key in keys):
        raise ValueError("知识回执与当前轮预算不一致")
    left, right = current["records"], saved["records"]
    shared = min(len(left), len(right))
    if left[:shared] != right[:shared]:
        raise ValueError("知识回执预算记录不属于当前轮")
    return deepcopy(current if len(left) >= len(right) else saved)


async def retrieve_verified_knowledge(state: dict, ledger: BudgetLedger, *, search, settings,
                                     query_expander, query: str, policy: bool = False,
                                     filters: dict | None = None,
                                     prepared_queries: list[str] | None = None, stage_callback=None,
                                     chat=None) -> VerifiedKnowledgeOutcome:
    """运行原权威检索及证据闸，只有相同轮/查询/过滤才复用。

    参数：state 本轮状态、ledger 独占账本、search/settings/query_expander 实例依赖；
        query 完整查询、policy 是否三角度政策检索、filters 权威筛选、prepared_queries 已准备问法；
        stage_callback 可选同步进度回调，chat 可选原轮快照仓储；两者不进入结果或检查点。
    返回：VerifiedKnowledgeOutcome，弱证据保留分数/计量而不注入原文。
    异常：预算超限、输入非法、知识依赖故障及取消原样传播。
    """
    from app.workflow.routing import knowledge_input_stop_reason
    from app.knowledge.confidence import assess_evidence, load_evidence_policy
    from app.knowledge.pipeline import ProductionKnowledgePipeline, production_protocols

    source_query = state.get("resolved_query") or state["raw_query"]
    if any(knowledge_input_stop_reason(value) for value in (state["raw_query"], source_query, query)):
        raise ContextBudgetExceeded("本轮知识输入超出可处理范围")
    evidence_policy = load_evidence_policy(Path(settings.evidence_policy_path), protocols=production_protocols(settings))
    key = json_hash({"turn_id": state["context"]["turn_id"], "query": query, "source_query": source_query,
                     "filters": filters or {}, "policy": policy,
                     "prepared_queries": prepared_queries, "evidence_policy_version": evidence_policy.version})
    cached = state.get("knowledge_cache", {}).get(key)
    saved = RetrievalSnapshot.model_validate(state["retrieval_snapshot"]) if state.get("retrieval_snapshot") else None
    if cached is not None:
        outcome = VerifiedKnowledgeOutcome.model_validate(cached)
        merged = merge_snapshots(saved, outcome.retrieval_snapshot) if outcome.retrieval_snapshot else saved
        state["retrieval_snapshot"] = merged.model_dump(mode="json") if merged else None
        if stage_callback is not None:
            stage_callback("policy", "running")
            stage_callback("policy", "completed")
        return outcome.model_copy(update={"budget": merge_knowledge_budget(ledger.snapshot(), outcome.budget),
                                          "retrieval_snapshot": merged}, deep=True)
    # 缓存重放没有新尝试；真正重新执行时以原轮已保存次数区分故障后的再次检索。
    capture = RetrievalCapture(state["context"]["turn_id"], json_hash([key, len(saved.attempts) if saved else 0]))

    async def save_snapshot(incoming):
        """每次子查询完成或故障立即保存原轮。参数：本请求快照；返回：None；异常：仓储或冲突错误。"""
        current = RetrievalSnapshot.model_validate(state["retrieval_snapshot"]) if state.get("retrieval_snapshot") else None
        merged = merge_snapshots(current, incoming)
        state["retrieval_snapshot"] = merged.model_dump(mode="json")
        if chat is not None:
            await chat.save_retrieval_snapshot(ToolContext.model_validate(state["context"]), merged)

    capture.on_update = save_snapshot
    async with asyncio.timeout(settings.knowledge_timeout_seconds):
        pipeline = ProductionKnowledgePipeline(search, settings, query_expander=query_expander,
            state=state, policy=policy, prepared_queries=prepared_queries, stage_callback=stage_callback)
        prepared = await pipeline.prepare(query, filters or {}, messages=[], budget=ledger)
        result = await pipeline.retrieve(prepared, capture=capture)
    ranked, selected, prepared_queries = result["ranked"], result["selected"], result["queries"]
    gate = assess_evidence(ranked, selected, evidence_policy)
    evidence = [item.model_dump(mode="json") for item in selected] if gate.passed else []
    outcome = VerifiedKnowledgeOutcome(evidence=evidence, gate=gate.model_dump(mode="json"),
        evidence_scores=[item.score for item in ranked], expanded_queries=prepared_queries, budget=ledger.snapshot(),
        retrieval_snapshot=RetrievalSnapshot.model_validate(state["retrieval_snapshot"]) if state.get("retrieval_snapshot") else None)
    state["knowledge_cache"] = {**state.get("knowledge_cache", {}), key: outcome.model_dump(mode="json")}
    if stage_callback is not None:
        stage_callback("policy", "completed")
    return outcome


async def verified_knowledge_tool(arguments: dict, context: ToolContext, scope: ToolExecutionScope) -> ToolResult:
    """将两个知识工具接到同一证据和预算边界。

    参数：arguments 已由执行器校验的公开参数、context 可信身份、scope 当前知识资源。
    返回：ToolResult，公开数据只说明检索状态，可信原文留在内部 verified_knowledge。
    异常：ValueError，缺可信资源；预算、检索错误及取消原样传播给统一执行器。
    """
    from app.workflow.policy_context import policy_query

    runtime = scope.knowledge
    if runtime is None or runtime.search is None or runtime.settings is None or scope.binding is None:
        raise ValueError("知识工具缺少可信本轮资源")
    state = runtime.state
    if state["context"]["turn_id"] != context.turn_id:
        raise ValueError("知识工具轮次不一致")
    policy = scope.binding.identity.name == "query_refund_policy"
    query = arguments["query" if policy else "keyword"]
    filters = {key: arguments[key] for key in ("category", "content_type") if arguments.get(key) is not None}
    if policy:
        filters["source_key"] = list(ORDER_POLICY_SOURCES)
        if state.get("route") == "order_policy":
            if not state.get("order_data"):
                # 订单政策必须用核验后的商品与状态扩写，不能让前置申请抢占本轮唯一扩写额度。
                return ToolResult.failed("ORDER_REQUIRED", "请先核实本次订单，再按已核实的商品和状态查询退款政策。")
            query = policy_query(state)
    outcome = await retrieve_verified_knowledge(state, runtime.ledger, search=runtime.search, settings=runtime.settings,
        query_expander=runtime.query_expander, query=query, policy=policy, filters=filters,
        prepared_queries=None if policy else [query], chat=runtime.chat)
    return ToolResult(ok=True, data={"evidence_count": len(outcome.evidence),
        "message": "已取得可核实知识证据" if outcome.gate["passed"] else "本轮没有达到可靠门槛的知识证据"},
        verified_knowledge=outcome)
