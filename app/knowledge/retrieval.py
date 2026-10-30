"""参考检索链路：两路各 50、RRF 50、每子句独立重排后轮转去重或按最高分合并。

普通入口最后最多 10 条，最高重排分低于 0.3 时 FAQ 正常空结果；退款与售后政策
支持已准备查询按 chunk ID 取最高分降序合并（最多 30 条）及旧入口兼容。所有答案正文必须
从 MySQL 权威回读，索引/模型故障绝不伪装成未命中。
"""

from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.knowledge.contracts import KnowledgeEncoder
from app.knowledge.limits import valid_dense_vector
from app.knowledge.query_rewrite import QueryRewrite, QueryRewriter, validate_prepared_queries, validate_rewrite
from app.knowledge.rerank import RankedChunk, RerankUnavailable
from app.knowledge.confidence import unique_ranked
from app.knowledge.schemas import ChunkSnapshot
from app.knowledge.search import KnowledgeSearchUnavailable
from app.knowledge.store import KnowledgeStore
from app.knowledge.snapshots import RetrievalCapture, chunk_snapshot
from app.schemas.flywheel import RetrievalAttempt

if TYPE_CHECKING:
    from app.core.model_budget import BudgetLedger

_MIN_CLAUSE = 4
_MAX_QUERY_CHARACTERS = 2000
# CH06 已准备查询每条最多保留十块，最多三条政策查询合计不超过三十块。
_PREPARED_QUERY_TOP_K = 10


def split_clauses(query: str) -> list[str]:
    """按参考中文标点拆有效子句，只有一个有效片段时保留原意完整查询。

    参数：query 为非空当前单轮查询；按 `，。；？` 分隔，短于 4 字片段不单独执行。
    返回：list[str]，多个有效子句或单个去首尾空白的原查询。
    异常：ValueError，原 query 为空。
    """
    if not query.strip():
        raise ValueError("检索 query 不能为空")
    parts = [part.strip() for part in re.split(r"[，。；？]+", query) if part.strip()]
    clauses = [part for part in parts if len(part) >= _MIN_CLAUSE]
    if len(clauses) <= 1:
        return [query.strip()]
    # 短条件不单独召回，但必须随每个子句进入编码、BM25和重排。
    conditions = "，".join(part for part in parts if len(part) < _MIN_CLAUSE)
    return [f"{conditions}，{clause}" if conditions else clause for clause in clauses]


def expand_policy_queries(query: str) -> list[str]:
    """保留原意形成三条退款政策检索角度，不生成任何政策事实。

    参数：query 为当前用户退款/售后问题原话。
    返回：list[str]，原话及追加申请条件/流程、时限/运费角度的三个查询。
    异常：ValueError，query 为空。
    """
    if not query.strip():
        raise ValueError("退款政策 query 不能为空")
    raw = query.strip()
    return [raw, f"{raw} 退款申请条件及流程", f"{raw} 退款处理时限及退换货运费承担"]


def merge_ranked_chunks(groups: list[list[RankedChunk]]) -> list[RankedChunk]:
    """跨查询组按 chunk ID 保留最高重排分，并按分数降序、同分主键升序排列。

    参数：groups 多条查询各自返回的 RankedChunk 列表。
    返回：list[RankedChunk]，去重后按 (-score, chunk.id) 排序的权威结果列表。
    异常：ValueError，同一稳定身份对应不同原文。
    """
    strongest = unique_ranked([item for group in groups for item in group])
    # 分数相同按原文主键固定次序，保证后续引用编号稳定。
    return sorted(strongest, key=lambda item: (-item.score, item.chunk.id))


def faq_rows(ranked: list[RankedChunk]) -> list[dict]:
    """映射模型可读权威结果。参数：ranked 为已校验排名；返回：原 FAQ 四字段列表；异常：无。"""
    return [{"id": item.chunk.id, "question": item.chunk.text.questions,
             "answer": item.chunk.text.answer, "category": item.chunk.text.category,
             "section_path": item.chunk.section_path} for item in ranked]


class HybridKnowledgeSearch:
    """无共享单轮状态的混合检索编排，逐子句调用但最终普通结果仍只取 Top-K。"""

    def __init__(self, store: KnowledgeStore, encoder: KnowledgeEncoder, index: Any,
                 reranker: Any, config: Settings, *, query_rewriter: QueryRewriter | None = None) -> None:
        """参数：store/encoder/index/reranker/config 检索依赖；query_rewriter 正式运行时改写器；返回：None；异常：无。"""
        self.store, self.encoder, self.index, self.reranker, self.config = store, encoder, index, reranker, config
        self.query_rewriter = query_rewriter

    async def _prepare(self, query: str, *, policy: bool = False, subquery_split: bool = True,
                       source_query: str | None = None, budget: BudgetLedger | None = None,
                       history: str = "", prepared_queries: list[str] | None = None) -> PreparedQuery:
        """一次准备整次查询；显式传入 prepared_queries 时只跳过旧改写与固定扩写，保留逐条确定性校验。

        参数：query 关键词或原问、policy 退款、subquery_split 拆句、source_query 原问、budget 请求独占账本、
            history 可信历史用户/工具文本（仅旧路径使用）、prepared_queries 已准备的1条或3条查询。
        返回：准备结果；异常：预算/改写/校验错误。
        """
        raw = source_query if source_query is not None else query
        if prepared_queries is not None:
            # 编码/召回前先做整组确定性校验，且不向prepared分支传递history。
            validate_prepared_queries(raw, prepared_queries)
            cleaned = [item.strip() for item in prepared_queries]
            # 已准备问法的型号、状态与问题必须整体检索，不再按标点拆散条件。
            groups = [[item] for item in cleaned]
            return PreparedQuery(raw=raw, policy=policy, groups=groups)
        if self.query_rewriter is None:
            # 未注入只供底层兼容；正式运行时在装配处强制提供改写器。
            return prepare_query(query, policy=policy, subquery_split=subquery_split)
        options = {"budget": budget} if budget is not None else {}
        rewrite = await self.query_rewriter.rewrite(raw, keyword=query if source_query is not None else None,
                                                   context=history, **options)
        return prepare_query(raw, policy=policy, subquery_split=subquery_split, rewrite=rewrite)

    def _validate_query(self, query: str) -> None:
        """校验单轮查询字符限制。参数：query 为 1–2000 非空字符；返回：None；异常：ValueError。"""
        if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY_CHARACTERS:
            raise ValueError("检索 query 必须为 1–2000 字符的非空文本，请拆分超长问题")

    async def _filter_scope(self, filters: dict, *, query: str, capture: RetrievalCapture | None = None) -> dict[int, ChunkSnapshot]:
        """分页取得权威白名单。参数：filters 已验证元数据、query 实际查询、capture 请求收集器；返回：主键快照；异常：存储故障。"""
        scope, after_id = {}, 0
        try:
            while rows := await self.store.active_page(after_id=after_id, limit=500):
                scope.update({row.id: row for row in rows if matches_filters(row, filters)})
                after_id = rows[-1].id
        except BaseException as error:
            if capture is not None:
                # 尚未得到召回总数，分页局部数量不能充当零候选或完整候选数。
                await capture.record(RetrievalAttempt(attempt_id=capture.attempt_id(query, scope=filters),
                    query=query, top_k=capture.top_k, status="fault", fault=type(error).__name__))
            raise
        return scope

    async def _retrieve_one(self, query: str, top_k: int,
                            scope: dict[int, ChunkSnapshot] | None = None, *,
                            bm25_query: str | None = None, rerank_query: str | None = None,
                            capture: RetrievalCapture | None = None) -> list[RankedChunk]:
        """记录实际查询、候选及精排。参数：查询/上限/范围、阶段查询、capture 请求收集器；返回：真实排名；异常：检索/快照保存错误透传。"""
        from app.observability.runtime import observe
        with observe("knowledge.retrieve", kind="retriever", input={"query": query, "top_k": top_k,
                "scope_ids": sorted(scope) if scope is not None else None,
                "bm25_query": bm25_query, "rerank_query": rerank_query}) as observation:
            facts = {"candidate_count": None, "chunks": [], "recall_scores": {}}
            selected = max(top_k, capture.top_k) if capture else top_k
            try:
                ranked = await self._retrieve_one_observed(query, selected, scope, bm25_query=bm25_query,
                    rerank_query=rerank_query, observation=observation, facts=facts)
            except BaseException as error:
                if capture is not None:
                    await capture.record(RetrievalAttempt(attempt_id=capture.attempt_id(query,
                        scope=sorted(scope) if scope is not None else None, bm25_query=bm25_query, rerank_query=rerank_query),
                        query=query, top_k=capture.top_k, candidate_count=facts["candidate_count"], status="fault",
                        chunks=facts["chunks"][:capture.top_k], fault=type(error).__name__))
                raise
            if capture is not None:
                await capture.record(RetrievalAttempt(attempt_id=capture.attempt_id(query,
                    scope=sorted(scope) if scope is not None else None, bm25_query=bm25_query, rerank_query=rerank_query),
                    query=query, top_k=capture.top_k, candidate_count=facts["candidate_count"],
                    status="ok" if ranked else "empty", chunks=[chunk_snapshot(item.chunk, i + 1,
                        recall_score=facts["recall_scores"].get(item.chunk.id), rerank_score=item.score)
                        for i, item in enumerate(sorted(ranked, key=lambda item: (-item.score, item.chunk.id))[:capture.top_k])]))
            observation.update(output={"ranked": [item.model_dump(mode="json") for item in ranked]})
            return ranked[:top_k]

    async def _retrieve_one_observed(self, query: str, top_k: int,
                                     scope: dict[int, ChunkSnapshot] | None = None, *,
                                     bm25_query: str | None, rerank_query: str | None, observation, facts) -> list[RankedChunk]:
        """执行一次完整混合召回、权威回读、原 chunk 重排。

        参数：query 标准子句；top_k 结果上限；scope 权威筛选快照；bm25_query/rerank_query 各阶段完整文本；
            observation 当前观测；facts 本次检索局部候选数、原文与召回分，用于故障保留。
        返回：list[RankedChunk]，最多 top_k 权威块；候选默认最多 50。
        异常：KnowledgeSearchUnavailable，空索引或权威状态失配；编码/索引/重排依赖异常。
        """
        if scope is not None and not scope:
            facts["candidate_count"] = 0
            return []
        vectors = await self.encoder.encode([query])
        if len(vectors) != 1 or not valid_dense_vector(vectors[0]):
            raise KnowledgeSearchUnavailable("查询编码结果不合法")
        recall = self.config.recall_top_k
        options = {"filter_ids": sorted(scope)} if scope is not None else {}
        hits = await self.index.hybrid_search(vectors[0], bm25_query if bm25_query is not None else query, dense_limit=recall,
                                              sparse_limit=recall, limit=recall, **options)
        facts.update(candidate_count=len(hits), recall_scores={hit.id: hit.score for hit in hits})
        observation.update(metadata={"recall_candidates": [hit.model_dump(mode="json") for hit in hits],
                                     "candidate_count": len(hits)})
        if not hits:
            # 真空库允许零结果；仍有有效原文却没有 Dense 候选属于同步故障。
            if scope or await self.store.active_page(after_id=0, limit=1):
                raise KnowledgeSearchUnavailable("知识混合索引为空，但仍有有效原文，需显式同步")
            return []
        if len(hits) > recall or len({hit.id for hit in hits}) != len(hits):
            raise KnowledgeSearchUnavailable("混合索引返回重复或超预算候选")
        rows = await self.store.get_chunks([hit.id for hit in hits])
        by_id = {row.id: row for row in rows}
        chunks: list[ChunkSnapshot] = []
        for hit in hits:
            row = by_id.get(hit.id)
            # 筛选后复核整份权威快照，拒绝任一路忽略filter或查询中途元数据变化。
            if scope is not None and (hit.id not in scope or row != scope[hit.id]):
                raise KnowledgeSearchUnavailable("索引返回过滤范围外候选或权威范围已变化")
            if (row is None or not row.is_active or row.vectorize_status != "done"
                    or row.vector_id != str(row.id) or row.content_hash != hit.content_hash
                    or row.embedding_revision != hit.embedding_revision):
                if scope is not None:
                    raise KnowledgeSearchUnavailable("过滤范围内索引与权威状态失配")
                continue
            chunks.append(row)
            facts["chunks"].append(chunk_snapshot(row, len(chunks), recall_score=hit.score))
        if not chunks:
            raise KnowledgeSearchUnavailable("知识混合索引与 MySQL 权威状态不一致")
        observation.update(metadata={"verified_candidates": [chunk.model_dump(mode="json") for chunk in chunks]})
        ranked = await self.reranker.rank(rerank_query if rerank_query is not None else query, chunks, top_n=top_k)
        expected = {row.id: row for row in chunks}
        ids = [item.chunk.id for item in ranked]
        # 即使注入模型返回类型正确，未知主键或替换正文也不能进入上下文。
        if (len(ids) != min(top_k, len(chunks)) or len(ids) != len(set(ids))
                or any(item.chunk != expected.get(item.chunk.id) for item in ranked)):
            raise RerankUnavailable("重排结果缺项、重复或不属于本次权威候选")
        return ranked

    async def retrieve(self, query: str, *, top_k: int | None = None,
                       subquery_split: bool | None = None, filters: dict | None = None,
                       source_query: str | None = None, budget: BudgetLedger | None = None,
                       history: str = "", prepared_queries: list[str] | None = None,
                       capture: RetrievalCapture | None = None,
                       ranked_capture: list[RankedChunk] | None = None) -> list[RankedChunk]:
        """逐子句跑完整链路，轮转合并或按已准备查询最高分合并后截到 Top-K。

        参数：query 当前原话或已验证完整问；top_k None 使用 config.rerank_top_k=10，显式值 1–recall_top_k；
            subquery_split None 使用配置默认 True，False 只检索完整查询；filters 为权威元数据筛选；
            source_query 为可信原问；budget 请求独占账本；history 可信历史用户/工具文本（仅旧路径使用）；
            prepared_queries 可选已准备的 1 条或 3 条查询，传入时跳过旧改写模型并按最高分合并；capture 请求局部闸前快照收集器。
            ranked_capture 可选请求局部列表，收集业务精排返回而非Top5审计额外候选，跨查询裁剪前保留。
        返回：list[RankedChunk]，旧路径最多 Top-K；已准备路径最多 min(Top-K, 10)，每条完整查询重排一次。
        异常：ValueError，参数非法；检索/模型/权威存储异常；取消直接传播。
        """
        self._validate_query(query)
        if source_query is not None:
            self._validate_query(source_query)
        selected = self.config.rerank_top_k if top_k is None else top_k
        if type(selected) is not int or not 1 <= selected <= self.config.recall_top_k:
            raise ValueError("top_k 必须为 1–recall_top_k 的整数，零值不表示默认值")
        split = self.config.subquery_split if subquery_split is None else subquery_split
        if type(split) is not bool:
            raise ValueError("subquery_split 必须为布尔值")
        normalized = _validated_filters(filters)
        prepared = await self._prepare(query, subquery_split=split, source_query=source_query, budget=budget,
                                       history=history, prepared_queries=prepared_queries)
        scope = await self._filter_scope(normalized, query=query, capture=capture) if normalized else None
        if prepared_queries is not None:
            selected = min(selected, _PREPARED_QUERY_TOP_K)
            groups = [await self._retrieve_one(group[0], selected, scope, capture=capture) for group in prepared.groups]
            if ranked_capture is not None:
                ranked_capture.extend(item for group in groups for item in group)
            # 全部查询完成后再按最高分合并裁剪，普通结果最多十块，较小显式上限仍有效。
            return merge_ranked_chunks(groups)[:selected]
        return await self._retrieve_group(prepared.groups[0], prepared, selected, scope, capture=capture, ranked_capture=ranked_capture)

    async def _retrieve_group(self, queries: list[str], prepared: PreparedQuery, selected: int,
                              scope: dict[int, ChunkSnapshot] | None, *, capture: RetrievalCapture | None = None,
                              ranked_capture: list[RankedChunk] | None = None) -> list[RankedChunk]:
        """合并一组子句排名。参数：queries 子句、prepared 共用文本、selected 上限、scope 权威范围、capture 请求收集器；返回：排名；异常：检索依赖错误。"""
        results = [await self._retrieve_one(clause, selected, scope,
                    bm25_query=prepared.bm25_queries.get(clause, clause),
                    rerank_query=prepared.rerank_queries.get(clause, clause), capture=capture) for clause in queries]
        if ranked_capture is not None:
            ranked_capture.extend(item for group in results for item in group)
        # 保持首次轮转位置，但同一证据跨子句相关度不同，门槛使用其最高有效分。
        strongest = {item.chunk.id: item for item in unique_ranked([item for group in results for item in group])}
        merged: list[RankedChunk] = []
        seen: set[int] = set()
        # 按轮次取每条子句的下一条，避免首句占满截断上限而完全丢掉后句证据。
        for row in zip_longest(*results):
            for item in row:
                if item is not None and item.chunk.id not in seen:
                    seen.add(item.chunk.id)
                    merged.append(strongest[item.chunk.id])
                    if len(merged) == selected:
                        return merged
        return merged

    async def query_faq(self, keyword: str, limit: int | None = None, *, filters: dict | None = None,
                        source_query: str | None = None) -> list[dict]:
        """普通 FAQ 在重排完成后以最高分闸判断是否可用，不把 RRF 分数当阈值。

        参数：keyword 当前查询；limit None 使用默认10；filters 元数据筛选；source_query 可信本轮原问。
        返回：list[dict]，最高重排分低于 0.3 时空，否则返回全部最多 limit 条。
        异常：ValueError，参数不合法；检索/模型/存储依赖错误。
        """
        ranked = await self.retrieve(keyword, top_k=limit, filters=filters, source_query=source_query)
        if not ranked or max(item.score for item in ranked) < self.config.rerank_min_score:
            return []
        return faq_rows(ranked)

    async def retrieve_policy(self, query: str, *, filters: dict | None = None,
                              budget: BudgetLedger | None = None, history: str = "",
                              prepared_queries: list[str] | None = None, capture: RetrievalCapture | None = None,
                              ranked_capture: list[RankedChunk] | None = None) -> list[RankedChunk]:
        """退款与售后政策多查询检索并按 chunk id 去重，默认最多 30 条。

        参数：query 当前退款问题或含已核实状态的种子；filters 可选元数据筛选；
            budget 请求独占账本；history 可信历史用户/工具文本（仅旧路径使用）；
            prepared_queries 可选已准备的 1 条或 3 条查询，传入时按 chunk ID 取最高分降序合并；capture 请求局部原文收集器。
            ranked_capture 收集各子查询既定业务上限的精排输出，最终候选裁剪前写入，不包含额外审计块。
        返回：list[RankedChunk]，旧路径默认最多 30 个原 chunk；已准备路径每条最多十块、合计最多三十块。
        异常：ValueError，参数/输入限制；检索、模型或存储异常。
        """
        self._validate_query(query)
        normalized = _validated_filters(filters)
        prepared = await self._prepare(query, policy=True, subquery_split=self.config.subquery_split,
                                       budget=budget, history=history, prepared_queries=prepared_queries)
        scope = await self._filter_scope(normalized, query=query, capture=capture) if normalized else None
        if prepared_queries is not None:
            selected = min(self.config.rerank_top_k, _PREPARED_QUERY_TOP_K)
            groups = [await self._retrieve_one(group[0], selected, scope, capture=capture) for group in prepared.groups]
            if ranked_capture is not None:
                ranked_capture.extend(item for group in groups for item in group)
            # 每条最多十块，最多三条查询；合并后不超过三十块，并保留更小的配置限制。
            return merge_ranked_chunks(groups)[:selected * 3]
        groups = [await self._retrieve_group(group, prepared, self.config.rerank_top_k, scope,
                    capture=capture, ranked_capture=ranked_capture) for group in prepared.groups]
        return unique_ranked([item for group in groups for item in group])


class RetrievalCandidate(BaseModel):
    """携带原策略分数的权威候选，BM25/Dense/RRF不套用sigmoid范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk: ChunkSnapshot = Field(description="已回核的原文快照")
    score: float = Field(allow_inf_nan=False, description="本策略原始相关度，跨策略不可比较")


class PreparedQuery(BaseModel):
    """同一道题四策略共享的确定查询与惰性Dense编码缓存。"""

    model_config = ConfigDict(extra="forbid")
    raw: str = Field(description="未改写的用户当前原问")
    groups: list[list[str]] = Field(description="普通问题一组，退款三组；每组为有效子句")
    policy: bool = Field(description="是否使用一致的三角度退款扩展")
    vectors: dict[str, list[float]] = Field(default_factory=dict, description="本题实际查询编码缓存；纯BM25不填充")
    bm25_queries: dict[str, str] = Field(default_factory=dict, description="标准子句对应的完整原问、标准问法及同义检索文本")
    rerank_queries: dict[str, str] = Field(default_factory=dict, description="标准子句对应的本轮原意重排文本，不含检索同义词")


class EvaluationRetrieval(BaseModel):
    """评估检索保留门禁前排名，并单列实际生成候选。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ranked: list[RetrievalCandidate] = Field(description="门禁前去重排名；重排策略每子句最多十条")
    selected: list[RetrievalCandidate] = Field(description="普通最多10、退款最多30的生成候选")


def prepare_query(raw: str, *, policy: bool = False, subquery_split: bool = True,
                  rewrite: QueryRewrite | None = None) -> PreparedQuery:
    """确定性准备一次原问，四策略使用相同子句与退款扩展。

    参数：raw 1–2000字符非空问题；policy 是否退款；subquery_split 是否拆子句；rewrite 可选本轮严格改写。
    返回：PreparedQuery，本题私有缓存，不读取答案、历史或其他题。
    异常：ValueError，输入或布尔参数非法。
    """
    if not isinstance(raw, str) or not raw.strip() or len(raw) > _MAX_QUERY_CHARACTERS:
        raise ValueError("评估问题必须为1–2000字符")
    if type(policy) is not bool or type(subquery_split) is not bool:
        raise ValueError("查询规则必须为布尔值")
    if rewrite is not None:
        validate_rewrite(raw, rewrite)
    standard = rewrite.standard_query if rewrite is not None else raw.strip()
    expanded = expand_policy_queries(standard) if policy else [standard]
    groups = [split_clauses(query) if subquery_split else [query] for query in expanded]
    bm25_queries, rerank_queries = {}, {}
    for group in groups:
        for query in group:
            # 同义词不进入Dense或重排；原问原样保留，不能被标准问法取代。
            bm25_queries[query] = "\n".join([raw, query, *rewrite.synonyms]) if rewrite is not None else query
            rerank_queries[query] = "\n".join([raw, query]) if rewrite is not None else query
    return PreparedQuery(raw=raw, policy=policy, groups=groups,
                         bm25_queries=bm25_queries, rerank_queries=rerank_queries)


def _validated_filters(filters: dict | None) -> dict:
    """校验过滤独立于语料是否存在。参数：filters 标量/列表映射；返回：规范列表映射；异常：ValueError。"""
    normalized = {}
    for key, value in (filters or {}).items():
        if key not in {"source_key", "category", "content_type", "is_key_clause"}:
            raise ValueError(f"不支持的过滤字段：{key}")
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        expected_type = bool if key == "is_key_clause" else str
        if not values or any(type(item) is not expected_type for item in values):
            raise ValueError("过滤值类型错误或列表为空")
        normalized[key] = values
    return normalized


def matches_filters(chunk: ChunkSnapshot, filters: dict | None) -> bool:
    """按冻结权威元数据精确筛选，不接受任意SQL/Milvus表达式。

    参数：chunk 原文；filters 可选source_key/category/content_type/is_key_clause，各项为标量或列表。
    返回：bool，各项之间AND，列表内OR。
    异常：ValueError，未知过滤字段或类型不合法。
    """
    for key, values in _validated_filters(filters).items():
        actual = chunk.text.category if key == "category" else getattr(chunk, key)
        if actual not in values:
            return False
    return True


def _interleave(groups: list[list[RetrievalCandidate]], limit: int) -> list[RetrievalCandidate]:
    """轮转去重保留最高同策略分数。参数：groups 子句排名、limit 正上限；返回：候选；异常：无。"""
    strongest = {}
    for group in groups:
        for item in group:
            if item.chunk.id not in strongest or strongest[item.chunk.id].score < item.score:
                strongest[item.chunk.id] = item
    merged = {}
    for row in zip_longest(*groups):
        for item in row:
            if item is not None:
                merged.setdefault(item.chunk.id, strongest[item.chunk.id])
                if len(merged) == limit:
                    return list(merged.values())
    return list(merged.values())


class EvaluationSearch:
    """独立四策略评估入口，共用已冻结语料，不改变应用默认门禁与返回协议。"""

    def __init__(self, store: KnowledgeStore, encoder: KnowledgeEncoder, index: Any,
                 reranker: Any, config: Settings) -> None:
        """保存外部依赖。参数：store/encoder/index/reranker/config；返回：None；异常：无。"""
        self.store, self.encoder, self.index, self.reranker, self.config = store, encoder, index, reranker, config

    async def _one(self, query: str, prepared: PreparedQuery, strategy: str,
                   scope: dict[int, ChunkSnapshot], filtered: bool) -> list[RetrievalCandidate]:
        """执行单子句原始召回与权威快照复核。

        参数：query 子句；prepared 本题缓存；strategy 策略；scope 冻结过滤语料；filtered 是否传原生筛选。
        返回：候选，重排仅hybrid_rerank调用，最多十条；其他策略最多五十。
        异常：KnowledgeSearchUnavailable，同步/过滤/语料失配；外部模型或索引错误。
        """
        if not scope:
            return []
        options = {"filter_ids": sorted(scope)} if filtered else {}
        if strategy != "bm25" and query not in prepared.vectors:
            vectors = await self.encoder.encode([query])
            if len(vectors) != 1 or not valid_dense_vector(vectors[0]):
                raise KnowledgeSearchUnavailable("查询编码结果不合法")
            prepared.vectors[query] = vectors[0]
        if strategy == "bm25":
            hits = await self.index.search_bm25(prepared.bm25_queries.get(query, query), 50, **options)
        elif strategy == "dense":
            hits = await self.index.search(prepared.vectors[query], 50, **options)
        else:
            hits = await self.index.hybrid_search(prepared.vectors[query], prepared.bm25_queries.get(query, query),
                                                  dense_limit=50, sparse_limit=50, limit=50, **options)
        if not hits:
            if strategy != "bm25":
                raise KnowledgeSearchUnavailable("过滤内存在权威语料但Dense索引没有候选")
            return []
        if len(hits) > 50 or len({hit.id for hit in hits}) != len(hits):
            raise KnowledgeSearchUnavailable("索引候选重复或超出50条预算")
        rows = {row.id: row for row in await self.store.get_chunks([hit.id for hit in hits])}
        candidates = []
        for hit in hits:
            row = rows.get(hit.id)
            if hit.id not in scope:
                raise KnowledgeSearchUnavailable("索引返回过滤范围外候选")
            if (row != scope[hit.id] or not row.is_active or row.vectorize_status != "done"
                    or row.vector_id != str(row.id) or row.content_hash != hit.content_hash
                    or row.embedding_revision != hit.embedding_revision):
                raise KnowledgeSearchUnavailable("评估权威语料或索引指纹已变化")
            candidates.append(RetrievalCandidate(chunk=row, score=hit.score))
        if strategy != "hybrid_rerank":
            return candidates
        ranked = await self.reranker.rank(prepared.rerank_queries.get(query, query), [item.chunk for item in candidates], top_n=10)
        expected = {item.chunk.id: item.chunk for item in candidates}
        if (len(ranked) != min(10, len(candidates)) or len({item.chunk.id for item in ranked}) != len(ranked)
                or any(item.chunk != expected.get(item.chunk.id) for item in ranked)):
            raise RerankUnavailable("重排结果缺项、重复或替换原文")
        return [RetrievalCandidate(chunk=item.chunk, score=item.score) for item in ranked]

    async def retrieve(self, prepared: PreparedQuery, *, strategy: str,
                       corpus: list[ChunkSnapshot], filters: dict | None = None) -> EvaluationRetrieval:
        """对同题执行四策略之一，排名保留到门禁之前。

        参数：prepared 一次准备的原问；strategy dense/bm25/hybrid/hybrid_rerank；corpus 冻结语料；filters 权威筛选。
        返回：EvaluationRetrieval，普通生成10、退款三组生成去重最多30；原始召回排名最多50。
        异常：ValueError，策略/过滤非法；检索/编码/重排错误。
        """
        if strategy not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError("未知评估策略")
        filters = _validated_filters(filters)
        scope = {chunk.id: chunk for chunk in corpus if chunk.is_active and matches_filters(chunk, filters)}
        ranking_groups, selected_groups = [], []
        for clauses in prepared.groups:
            results = [await self._one(clause, prepared, strategy, scope, bool(filters)) for clause in clauses]
            merged = _interleave(results, 10 if strategy == "hybrid_rerank" else 50)
            ranking_groups.append(merged)
            selected_groups.append(merged[:10])
        # 退款沿用三扩展的首次出现顺序，同时保留同一证据的最高门槛分数。
        selected = _interleave([sum(selected_groups, [])], 30 if prepared.policy else 10)
        # 排名必须以全部实际生成候选为前缀，不能让首个退款扩展的50条挤掉后续扩展证据。
        ranked = _interleave([selected + sum(ranking_groups, [])],
                             30 if prepared.policy and strategy == "hybrid_rerank" else 50)
        return EvaluationRetrieval(ranked=ranked, selected=selected)
