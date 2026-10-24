"""参考检索链路：两路各 50、RRF 50、每子句独立重排后轮转去重。

普通入口最后最多 10 条，最高重排分低于 0.3 时 FAQ 正常空结果；退款政策
三条扩展各检索一次，按主键去重后保留全部，默认最多 30。所有答案正文必须
从 MySQL 权威回读，索引/模型故障绝不伪装成未命中。
"""

from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.knowledge.contracts import KnowledgeEncoder
from app.knowledge.limits import valid_dense_vector
from app.knowledge.rerank import RankedChunk, RerankUnavailable
from app.knowledge.schemas import ChunkSnapshot
from app.knowledge.search import KnowledgeSearchUnavailable
from app.knowledge.store import KnowledgeStore

_MIN_CLAUSE = 4
_MAX_QUERY_CHARACTERS = 2000


def split_clauses(query: str) -> list[str]:
    """按参考中文标点拆有效子句，只有一个有效片段时保留原意完整查询。

    参数：query 为非空当前单轮查询；按 `，。；？` 分隔，短于 4 字片段不单独执行。
    返回：list[str]，多个有效子句或单个去首尾空白的原查询。
    异常：ValueError，原 query 为空。
    """
    if not query.strip():
        raise ValueError("检索 query 不能为空")
    clauses = [part.strip() for part in re.split(r"[，。；？]+", query) if len(part.strip()) >= _MIN_CLAUSE]
    return clauses if len(clauses) > 1 else [query.strip()]


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


def faq_rows(ranked: list[RankedChunk]) -> list[dict]:
    """映射模型可读权威结果。参数：ranked 为已校验排名；返回：原 FAQ 四字段列表；异常：无。"""
    return [{"id": item.chunk.id, "question": item.chunk.text.questions,
             "answer": item.chunk.text.answer, "category": item.chunk.text.category,
             "section_path": item.chunk.section_path} for item in ranked]


class HybridKnowledgeSearch:
    """无共享单轮状态的混合检索编排，逐子句调用但最终普通结果仍只取 Top-K。"""

    def __init__(self, store: KnowledgeStore, encoder: KnowledgeEncoder, index: Any,
                 reranker: Any, config: Settings) -> None:
        """保存外部依赖。参数：store 权威库、encoder 编码器、index 混合索引、reranker 相关度模型、config K/N；返回：None；异常：无。"""
        self.store, self.encoder, self.index, self.reranker, self.config = store, encoder, index, reranker, config

    def _validate_query(self, query: str) -> None:
        """校验单轮查询字符限制。参数：query 为 1–2000 非空字符；返回：None；异常：ValueError。"""
        if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY_CHARACTERS:
            raise ValueError("检索 query 必须为 1–2000 字符的非空文本，请拆分超长问题")

    async def _retrieve_one(self, query: str, top_k: int) -> list[RankedChunk]:
        """执行一次完整混合召回、权威回读、原 chunk 重排。

        参数：query 为完整子句；top_k 为已校验结果上限。
        返回：list[RankedChunk]，最多 top_k 权威块；候选默认最多 50。
        异常：KnowledgeSearchUnavailable，空索引或权威状态失配；编码/索引/重排依赖异常。
        """
        vectors = await self.encoder.encode([query])
        if len(vectors) != 1 or not valid_dense_vector(vectors[0]):
            raise KnowledgeSearchUnavailable("查询编码结果不合法")
        recall = self.config.recall_top_k
        hits = await self.index.hybrid_search(vectors[0], query, dense_limit=recall,
                                              sparse_limit=recall, limit=recall)
        if not hits:
            # 真空库允许零结果；仍有有效原文却没有 Dense 候选属于同步故障。
            if await self.store.active_page(after_id=0, limit=1):
                raise KnowledgeSearchUnavailable("知识混合索引为空，但仍有有效原文，需显式同步")
            return []
        if len(hits) > recall or len({hit.id for hit in hits}) != len(hits):
            raise KnowledgeSearchUnavailable("混合索引返回重复或超预算候选")
        rows = await self.store.get_chunks([hit.id for hit in hits])
        by_id = {row.id: row for row in rows}
        chunks: list[ChunkSnapshot] = []
        for hit in hits:
            row = by_id.get(hit.id)
            if (row is None or not row.is_active or row.vectorize_status != "done"
                    or row.vector_id != str(row.id) or row.content_hash != hit.content_hash
                    or row.embedding_revision != hit.embedding_revision):
                continue
            chunks.append(row)
        if not chunks:
            raise KnowledgeSearchUnavailable("知识混合索引与 MySQL 权威状态不一致")
        ranked = await self.reranker.rank(query, chunks, top_n=top_k)
        expected = {row.id: row for row in chunks}
        ids = [item.chunk.id for item in ranked]
        # 即使注入模型返回类型正确，未知主键或替换正文也不能进入上下文。
        if (len(ids) != min(top_k, len(chunks)) or len(ids) != len(set(ids))
                or any(item.chunk != expected.get(item.chunk.id) for item in ranked)):
            raise RerankUnavailable("重排结果缺项、重复或不属于本次权威候选")
        return ranked

    async def retrieve(self, query: str, *, top_k: int | None = None,
                       subquery_split: bool | None = None) -> list[RankedChunk]:
        """逐子句跑完整链路，轮转合并去重后最终截到 Top-K。

        参数：query 当前原话；top_k None 使用 config.rerank_top_k=10，显式值 1–recall_top_k；
            subquery_split None 使用配置默认 True，False 只检索完整原话。
        返回：list[RankedChunk]，普通路径最多 Top-K；N 个子句会分别调用 N 次重排。
        异常：ValueError，参数非法；检索/模型/权威存储异常；取消直接传播。
        """
        self._validate_query(query)
        selected = self.config.rerank_top_k if top_k is None else top_k
        if type(selected) is not int or not 1 <= selected <= self.config.recall_top_k:
            raise ValueError("top_k 必须为 1–recall_top_k 的整数，零值不表示默认值")
        split = self.config.subquery_split if subquery_split is None else subquery_split
        if type(split) is not bool:
            raise ValueError("subquery_split 必须为布尔值")
        queries = split_clauses(query) if split else [query.strip()]
        results = [await self._retrieve_one(clause, selected) for clause in queries]
        # 保持首次轮转位置，但同一证据跨子句相关度不同，门槛使用其最高有效分。
        strongest: dict[int, RankedChunk] = {}
        for group in results:
            for item in group:
                existing = strongest.get(item.chunk.id)
                if existing is None or item.score > existing.score:
                    strongest[item.chunk.id] = item
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

    async def query_faq(self, keyword: str, limit: int | None = None) -> list[dict]:
        """普通 FAQ 在重排完成后以最高分闸判断是否可用，不把 RRF 分数当阈值。

        参数：keyword 当前查询；limit None 使用默认 10，显式值覆盖。
        返回：list[dict]，最高重排分低于 0.3 时空，否则返回全部最多 limit 条。
        异常：ValueError，参数不合法；检索/模型/存储依赖错误。
        """
        ranked = await self.retrieve(keyword, top_k=limit)
        if not ranked or max(item.score for item in ranked) < self.config.rerank_min_score:
            return []
        return faq_rows(ranked)

    async def retrieve_policy(self, query: str) -> list[RankedChunk]:
        """退款政策三查询分别检索并按 chunk id 去重，合并后不再截为十条。

        参数：query 当前退款问题，原话保留到三条扩展中。
        返回：list[RankedChunk]，默认最多 30 个原 chunk，保留查询顺序。
        异常：ValueError，参数/输入限制；检索、模型或存储异常。
        """
        self._validate_query(query)
        merged: dict[int, RankedChunk] = {}
        for expanded in expand_policy_queries(query):
            for item in await self.retrieve(expanded):
                merged.setdefault(item.chunk.id, item)
        return list(merged.values())


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


class EvaluationRetrieval(BaseModel):
    """评估检索保留门禁前排名，并单列实际生成候选。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ranked: list[RetrievalCandidate] = Field(description="门禁前去重排名；重排策略每子句最多十条")
    selected: list[RetrievalCandidate] = Field(description="普通最多10、退款最多30的生成候选")


def prepare_query(raw: str, *, policy: bool = False, subquery_split: bool = True) -> PreparedQuery:
    """确定性准备一次原问，四策略使用相同子句与退款扩展。

    参数：raw 1–2000字符非空问题；policy 是否退款；subquery_split 是否拆子句。
    返回：PreparedQuery，本题私有缓存，不读取答案、历史或其他题。
    异常：ValueError，输入或布尔参数非法。
    """
    if not isinstance(raw, str) or not raw.strip() or len(raw) > _MAX_QUERY_CHARACTERS:
        raise ValueError("评估问题必须为1–2000字符")
    if type(policy) is not bool or type(subquery_split) is not bool:
        raise ValueError("查询规则必须为布尔值")
    expanded = expand_policy_queries(raw) if policy else [raw.strip()]
    return PreparedQuery(raw=raw, policy=policy,
                         groups=[split_clauses(query) if subquery_split else [query] for query in expanded])


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
            hits = await self.index.search_bm25(query, 50, **options)
        elif strategy == "dense":
            hits = await self.index.search(prepared.vectors[query], 50, **options)
        else:
            hits = await self.index.hybrid_search(prepared.vectors[query], query,
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
        ranked = await self.reranker.rank(query, [item.chunk for item in candidates], top_n=10)
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
