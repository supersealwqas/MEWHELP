"""参考检索链路：两路各 50、RRF 50、每子句独立重排后轮转去重。

普通入口最后最多 10 条，最高重排分低于 0.3 时 FAQ 正常空结果；退款政策
三条扩展各检索一次，按主键去重后保留全部，默认最多 30。所有答案正文必须
从 MySQL 权威回读，索引/模型故障绝不伪装成未命中。
"""

from __future__ import annotations

import re
from itertools import zip_longest
from typing import Any

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
             "answer": item.chunk.text.answer, "category": item.chunk.text.category} for item in ranked]


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
        merged: list[RankedChunk] = []
        seen: set[int] = set()
        # 按轮次取每条子句的下一条，避免首句占满截断上限而完全丢掉后句证据。
        for row in zip_longest(*results):
            for item in row:
                if item is not None and item.chunk.id not in seen:
                    seen.add(item.chunk.id)
                    merged.append(item)
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
