"""图内知识工具的可信适配：权威检索、证据门槛、同轮复用与不倒退的预算合并。"""

import asyncio
from copy import deepcopy

from app.core.memory import ContextBudgetExceeded
from app.core.model_budget import BudgetLedger
from app.schemas.tools import ToolContext, ToolResult, VerifiedKnowledgeOutcome
from app.tools.contracts import ToolExecutionScope, json_hash

# 与原订单政策流程保持同一来源范围；商品规格与外部工具文字不能进入此范围。
ORDER_POLICY_SOURCES = ("doc:returns-policy.md", "doc:after-sales-manual.md",
                        "doc:billing-shipping.md", "doc:member-benefits.md")


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
                                     prepared_queries: list[str] | None = None, stage_callback=None) -> VerifiedKnowledgeOutcome:
    """运行原权威检索及证据闸，只有相同轮/查询/过滤才复用。

    参数：state 本轮状态、ledger 独占账本、search/settings/query_expander 实例依赖；
        query 完整查询、policy 是否三角度政策检索、filters 权威筛选、prepared_queries 已准备问法；
        stage_callback 可选同步进度回调，仅图入口提供，不进入结果或检查点。
    返回：VerifiedKnowledgeOutcome，弱证据保留分数/计量而不注入原文。
    异常：预算超限、输入非法、知识依赖故障及取消原样传播。
    """
    from app.workflow.context import select_evidence
    from app.workflow.routing import assess_evidence, citations_from_ranked, knowledge_input_stop_reason

    source_query = state.get("resolved_query") or state["raw_query"]
    if any(knowledge_input_stop_reason(value) for value in (state["raw_query"], source_query, query)):
        raise ContextBudgetExceeded("本轮知识输入超出可处理范围")
    key = json_hash({"turn_id": state["context"]["turn_id"], "query": query, "source_query": source_query,
                     "filters": filters or {}, "policy": policy,
                     "prepared_queries": prepared_queries})
    cached = state.get("knowledge_cache", {}).get(key)
    if cached is not None:
        outcome = VerifiedKnowledgeOutcome.model_validate(cached)
        if stage_callback is not None:
            stage_callback("policy", "running")
            stage_callback("policy", "completed")
        return outcome.model_copy(update={"budget": merge_knowledge_budget(ledger.snapshot(), outcome.budget)}, deep=True)
    async with asyncio.timeout(settings.knowledge_timeout_seconds):
        if prepared_queries is None:
            if policy:
                if stage_callback is not None:
                    stage_callback("expand", "running")
                expanded = await query_expander.expand(query, simple_query=state.get("standard_query") or source_query,
                                                       required_count=3, budget=ledger)
                prepared_queries = expanded.queries
                if stage_callback is not None:
                    stage_callback("expand", "completed", queries=prepared_queries)
            else:
                prepared_queries = [query]
        options = {"filters": filters} if filters else {}
        retrieve = search.retrieve_policy if policy else search.retrieve
        if not policy:
            options["source_query"] = source_query
        if stage_callback is not None:
            stage_callback("policy", "running")
        ranked = await retrieve(query, prepared_queries=prepared_queries, budget=ledger, history="", **options)
    gate = assess_evidence([item.score for item in ranked], settings.rerank_min_score)
    evidence = select_evidence([item.model_dump(mode="json") for item in citations_from_ranked(ranked)], settings) if gate.passed else []
    outcome = VerifiedKnowledgeOutcome(evidence=evidence, gate=gate.model_dump(mode="json"),
        evidence_scores=[item.score for item in ranked], expanded_queries=prepared_queries, budget=ledger.snapshot())
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
        if state.get("route") == "order_policy" and state.get("order_data"):
            query = policy_query(state)
    outcome = await retrieve_verified_knowledge(state, runtime.ledger, search=runtime.search, settings=runtime.settings,
        query_expander=runtime.query_expander, query=query, policy=policy, filters=filters,
        prepared_queries=None if policy else [query])
    return ToolResult(ok=True, data={"evidence_count": len(outcome.evidence),
        "message": "已取得可核实知识证据" if outcome.gate["passed"] else "本轮没有达到可靠门槛的知识证据"},
        verified_knowledge=outcome)
