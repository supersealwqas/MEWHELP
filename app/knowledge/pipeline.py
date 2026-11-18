"""线上与评估共用的知识理解、检索、预算选择、策略判定和生成流程，不负责聊天或问题池持久化。"""

import asyncio
from hashlib import sha256
import json

from app.core.memory import ContextBudgetExceeded
from app.core.model_budget import BudgetLedger
from app.knowledge.answering import EvidenceCitation
from app.knowledge.confidence import unique_ranked
from app.knowledge.rerank import RERANKER_REVISION
from app.knowledge.snapshots import RetrievalCapture

QUERY_PROTOCOL = "production-understanding:prepared-query:measured-condition-preservation:policy-expansion3:v2"
BUDGET_PROTOCOL = "complete-block-skip:context-evidence:window-div16:v1"
# 线上引用修复与离线生成协议身份共用同一文案，避免实际提示漂移仍沿用旧比较键。
KNOWLEDGE_CITATION_REPAIR_TEMPLATE = "上次知识回答引用校验未通过：{reason}。请基于同一完整证据重新提交唯一结果；不要使用未出现的编号。"


def production_protocols(settings) -> dict[str, str]:
    """绑定会影响制品判断的行为协议，语料摘要不在此处。

    参数：settings当前模型与预算设置；返回：查询/精排/预算协议摘要；异常：序列化错误。
    """
    from app.workflow.prompts import UNDERSTANDING_PROMPT, INTENT_PROMPT
    from app.knowledge.query_rewrite import QUERY_EXPANSION_RULES
    query = {"protocol": QUERY_PROTOCOL, "model": settings.chat_model, "intent_model": settings.intent_model,
             "classification": INTENT_PROMPT,
             "understanding": UNDERSTANDING_PROMPT, "expansion": QUERY_EXPANSION_RULES}
    budget = {"protocol": BUDGET_PROTOCOL, "limit": settings.context_evidence_max_tokens,
              "window": settings.model_context_window, "non_ascii_tokens": settings.context_non_ascii_tokens}
    return {"query": sha256(json.dumps(query, sort_keys=True).encode()).hexdigest(),
            "rerank": f"{RERANKER_REVISION}:recall{settings.recall_top_k}:top{settings.rerank_top_k}",
            "budget": sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest()}


async def understand_query(model, messages: list, budget: BudgetLedger):
    """沿用同一模型适配器及完整消息组装结果完成一次理解。

    参数：model正式适配器、messages可信ModelContext、budget本轮账本。
    返回：UnderstandingResult；异常：预算与模型依赖错误原样传播。
    """
    return await model.understand(messages, budget)


class ProductionKnowledgePipeline:
    """共享检索流程；请求状态由prepare返回，不保存在共享搜索器。"""

    def __init__(self, search, settings, *, query_expander, model=None, state=None,
                 policy=False, prepared_queries=None, stage_callback=None):
        """保存依赖及已核验的可选查询上下文。参数：检索/配置/模型/原轮及阶段参数；返回：None；异常：无。"""
        self.search, self.settings, self.query_expander, self.model = search, settings, query_expander, model
        self.state, self.policy, self.prepared_queries = state, policy, prepared_queries
        self.stage_callback = stage_callback

    async def prepare(self, query: str, filters: dict, *, messages: list, budget: BudgetLedger) -> dict:
        """准备同一次真实检索；已完成理解的线上原轮不重复模型调用。

        参数：query完整问、filters真实过滤、messages可信上下文、budget独占账本。
        返回：含实际查询、扩写、过滤与账本的请求私有字典。
        异常：输入/扩写/预算错误原样传播；不加载置信度制品。
        """
        from app.workflow.routing import knowledge_input_stop_reason, original_understanding
        from app.knowledge.retrieval import _validated_filters
        _validated_filters(filters)
        if knowledge_input_stop_reason(query):
            raise ContextBudgetExceeded("本轮知识输入超出可处理范围")
        if self.state is not None:
            source = self.state.get("resolved_query") or self.state["raw_query"]
            standard = self.state.get("standard_query") or source
        else:
            understood = await understand_query(self.model, messages, budget) if self.model else original_understanding(query)
            source, standard = understood.resolved_query, understood.standard_query
        if any(knowledge_input_stop_reason(value) for value in (source, standard)):
            raise ContextBudgetExceeded("本轮理解后的知识输入超出可处理范围")
        queries = self.prepared_queries
        if queries is None:
            if self.policy:
                if self.stage_callback:
                    self.stage_callback("expand", "running")
                expanded = await self.query_expander.expand(query, simple_query=standard, required_count=3, budget=budget)
                queries = expanded.queries
                if self.stage_callback:
                    self.stage_callback("expand", "completed", queries=queries)
            else:
                queries = [standard]
        return {"query": query, "source_query": source, "standard_query": standard, "queries": list(queries),
                "filters": dict(filters), "policy": self.policy, "ledger": budget}

    async def retrieve(self, prepared: dict, *, capture: RetrievalCapture | None = None) -> dict:
        """一次检索保留按最高分合并排序的完整精排并集与原候选预算结果，不依赖门禁。

        参数：prepared请求私有准备结果、capture可选审计快照；返回：ranked/selected/budget/查询/快照。
        异常：检索故障、原文冲突、非空候选全被预算排除；均不产生问题池信号。
        """
        from app.knowledge.retrieval import merge_ranked_chunks
        from app.workflow.context import select_evidence
        from app.workflow.routing import citations_from_ranked
        ledger, full_ranked = prepared["ledger"], []
        options = {"filters": prepared["filters"], "prepared_queries": prepared["queries"],
                   "budget": ledger, "history": "", "capture": capture, "ranked_capture": full_ranked}
        if not prepared["policy"]:
            options["source_query"] = prepared["source_query"]
        retrieve = self.search.retrieve_policy if prepared["policy"] else self.search.retrieve
        if self.stage_callback:
            self.stage_callback("policy", "running")
        async with asyncio.timeout(self.settings.knowledge_timeout_seconds):
            candidates = await retrieve(prepared["query"], **options)
        # 捕获列表保留各查询出现次序；评估名次须沿用真实合并排序，且不能裁成预算候选。
        ranked = merge_ranked_chunks([full_ranked])
        # 明确要求搜索器履行全候选协议，不能悄悄把已裁剪返回列表当完整并集。
        if candidates and not ranked:
            raise ValueError("检索器未提供跨查询裁剪前的完整ranked")
        candidates = unique_ranked(candidates)
        known = {row.chunk.id: row for row in ranked}
        if any(row.chunk.id not in known or row.chunk != known[row.chunk.id].chunk for row in candidates):
            raise ValueError("预算候选不属于本次完整精排原文")
        selected = [EvidenceCitation.model_validate(row) for row in select_evidence(
            [row.model_dump(mode="json") for row in citations_from_ranked(candidates)], self.settings)]
        if ranked and not selected:
            raise ContextBudgetExceeded("完整知识证据无法放入本轮预算")
        return {"ranked": ranked, "selected": selected, "budget": ledger.snapshot(),
                "query": prepared["query"], "source_query": prepared["source_query"],
                "standard_query": prepared["standard_query"], "queries": prepared["queries"],
                "snapshot": capture.snapshot if capture is not None else None}

    async def generate(self, prepared: dict, retrieved: dict) -> dict:
        """共用实际selected生成、自评和一次引用修复，不保存聊天或问题池。

        参数：prepared至少含原ledger；retrieved含selected及线上原gate或完整ranked。
        返回：answer/refused/citations/pool_signals/usage；用量缺失仍为未知。
        异常：预算、结构、引用、网络故障原样传播，不能冒充知识不足。
        """
        from pathlib import Path
        from app.knowledge.answering import (FIXED_KNOWLEDGE_REFUSAL, KnowledgeAnswerError,
            decide_answer, knowledge_answer_handoff, published_citations)
        from app.knowledge.confidence import assess_evidence, load_evidence_policy
        from app.schemas.flywheel import PoolSignal
        from app.workflow.context import assemble_context, select_evidence
        from app.workflow.model import knowledge_tool_schema
        from app.workflow.prompts import KNOWLEDGE_PROMPT

        ledger = prepared["ledger"]
        selected = [EvidenceCitation.model_validate(item) for item in retrieved["selected"]]
        gate = retrieved.get("gate")
        if gate is None:
            policy = load_evidence_policy(Path(self.settings.evidence_policy_path), protocols=production_protocols(self.settings))
            gate = assess_evidence(retrieved["ranked"], selected, policy).model_dump(mode="json")
        if not gate["passed"]:
            signal = PoolSignal(source="retrieval_low_conf", reason=(gate.get("low_confidence") or {}).get("reason", "本轮检索证据不足"))
            return {"answer": FIXED_KNOWLEDGE_REFUSAL, "refused": True, "citations": [],
                    "pool_signals": [signal], "usage": ledger.snapshot(), "gate": gate}
        rows = [item.model_dump(mode="json") for item in selected]
        # assemble_context的历史接口还会选择证据；先证明其结果相同，否则明确预算失败。
        # 不能把更小的集合带进模型却沿用原selected通过的检索判定。
        if select_evidence(rows, self.settings) != rows:
            raise ContextBudgetExceeded("生成阶段不能缩小已判闸的完整证据集合")
        if self.state is None or self.model is None:
            raise ValueError("知识生成缺少原轮状态或正式模型")
        state = {**self.state, "evidence": rows}
        messages = assemble_context(state, self.settings, KNOWLEDGE_PROMPT, [knowledge_tool_schema()])
        messages.append({"role": "user", "content": json.dumps(knowledge_answer_handoff(state["raw_query"]), ensure_ascii=False)})
        for attempt in range(2):
            result = await self.model.answer_knowledge(messages, ledger)
            try:
                answer, failure = decide_answer(result, selected)
            except KnowledgeAnswerError as error:
                ledger.record_failure("answer", {"type": "knowledge_citation", "reason": str(error)})
                if attempt:
                    raise
                # 同一answer阶段最多两次；完整修复说明也计入下一次请求预算。
                messages.append({"role": "user", "content": KNOWLEDGE_CITATION_REPAIR_TEMPLATE.format(reason=str(error))})
                continue
            signals = [PoolSignal(source=failure.source, reason=failure.reason)] if failure else []
            return {"answer": answer, "refused": failure is not None,
                    "citations": [] if failure else published_citations(answer, selected),
                    "pool_signals": signals, "usage": ledger.snapshot(), "gate": gate}
        raise KnowledgeAnswerError("知识回答无法通过校验")
