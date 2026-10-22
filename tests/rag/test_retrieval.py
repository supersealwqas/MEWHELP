"""参考混合检索调用链测试，锁定逐子句重排、轮转及退款证据上限。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, VectorHit
from app.knowledge.search import KnowledgeSearchUnavailable
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import VectorIndexUnavailable
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, make_settings


def _chunk(identifier: int) -> ChunkSnapshot:
    """构造有权威原文的知识记录。参数：主键；返回：快照；异常：校验失败。"""
    text = KnowledgeText(category="售后", questions=f"问法{identifier}", answer=f"权威答案{identifier}。")
    return ChunkSnapshot(id=identifier, source_key=f"doc:{identifier}.md", ordinal=0, text=text,
                         content_type="policy", content_hash=content_fingerprint(text),
                         embedding_revision="fake-encoder-v1", vectorize_status="done", vector_id=str(identifier))


class HybridBoundary:
    """仅替换 Milvus 外部边界，记录真实检索服务发出的召回预算。"""

    def __init__(self, chunks: list[ChunkSnapshot], groups: dict[str, list[int]] | None = None) -> None:
        """保存候选与 query 映射。参数：快照/映射；返回：None；异常：无。"""
        self.rows = {row.id: row for row in chunks}
        self.groups = groups or {}
        self.calls = []

    async def hybrid_search(self, vector, text, *, dense_limit, sparse_limit, limit):
        """返回 SDK 候选指纹。参数：向量/查询/两路及融合数量；返回：命中；异常：无。"""
        self.calls.append((text, dense_limit, sparse_limit, limit))
        ids = self.groups.get(text, list(self.rows))[:limit]
        return [VectorHit(id=identifier, score=0.01, content_hash=self.rows[identifier].content_hash,
                          embedding_revision=self.rows[identifier].embedding_revision) for identifier in ids]


class RerankBoundary:
    """替换外部相关度模型，不替换检索编排和权威回读。"""

    def __init__(self, scores: dict[int, float] | None = None) -> None:
        """保存相关度标注。参数：主键分数；返回：None；异常：无。"""
        self.scores = scores or {}
        self.calls = []

    async def rank(self, query, chunks, *, top_n):
        """按标注相关度输出。参数：原查询/权威快照/输出数；返回：排名；异常：校验失败。"""
        from app.knowledge.rerank import RankedChunk

        self.calls.append((query, [chunk.id for chunk in chunks], top_n))
        ranked = [RankedChunk(chunk=chunk, score=self.scores.get(chunk.id, 0.8)) for chunk in chunks]
        return sorted(ranked, key=lambda item: (-item.score, item.chunk.id))[:top_n]


def _search(chunks: list[ChunkSnapshot], *, groups=None, scores=None, **overrides):
    """组装真实检索编排。参数：快照/候选/分数/配置；返回：服务与外部边界；异常：配置错误。"""
    from app.knowledge.retrieval import HybridKnowledgeSearch

    encoder = FakeEncoder()
    index, reranker = HybridBoundary(chunks, groups), RerankBoundary(scores)
    service = HybridKnowledgeSearch(MemoryKnowledgeStore(snapshots=chunks), encoder, index, reranker, make_settings(**overrides))
    return service, encoder, index, reranker


async def test_default_recall_50_fused_50_rerank_10() -> None:
    """避免把两路共 100 或旧 FAQ 3 条用于重排链路。参数：无；返回：None；异常：断言失败。"""
    search, encoder, index, reranker = _search([_chunk(identifier) for identifier in range(1, 61)])
    result = await search.retrieve("退货申请条件")
    assert [item.chunk.id for item in result] == list(range(1, 11))
    assert index.calls == [("退货申请条件", 50, 50, 50)]
    assert reranker.calls == [("退货申请条件", list(range(1, 51)), 10)]
    assert encoder.inputs == ["退货申请条件"]


async def test_subqueries_each_run_full_pipeline_then_round_robin_merge() -> None:
    """每子句独立 rank，不能变成合并候选一次重排或顺序拼接截断。参数：无；返回：None；异常：断言失败。"""
    groups = {"退货申请条件": [1, 3, 5, 7, 9, 11], "退款处理时限": [2, 4, 6, 8, 10, 12]}
    search, encoder, index, reranker = _search([_chunk(identifier) for identifier in range(1, 13)], groups=groups)
    result = await search.retrieve("退货申请条件，退款处理时限？")
    assert [item.chunk.id for item in result] == list(range(1, 11))
    assert len(index.calls) == len(reranker.calls) == 2
    assert all(call[1:] == (50, 50, 50) for call in index.calls)
    assert all(call[2] == 10 for call in reranker.calls)
    assert encoder.inputs == ["退货申请条件", "退款处理时限"]


async def test_subqueries_deduplicate_chunk_ids_and_keep_other_evidence() -> None:
    """重复命中不能挤占最终 10 条位置。参数：无；返回：None；异常：断言失败。"""
    groups = {"退货申请条件": [1, 2, 3], "退款处理时限": [1, 4, 5]}
    search, _, _, _ = _search([_chunk(identifier) for identifier in range(1, 6)], groups=groups)
    assert [item.chunk.id for item in await search.retrieve("退货申请条件；退款处理时限")] == [1, 2, 4, 3, 5]


@pytest.mark.parametrize("query,expected", [
    ("退货条件，退款时限。", ["退货条件", "退款时限"]),
    ("退货，退款？", ["退货，退款？"]),
    ("退款条件，啊", ["退款条件，啊"]),
    ("退货条件,退款时限", ["退货条件,退款时限"]),
    ("甲；退货条件；退款时限", ["退货条件", "退款时限"]),
])
def test_clause_split_uses_four_character_minimum(query: str, expected: list[str]) -> None:
    """不足四字不发独立调用，不能形成多个子句时原句保留。参数：查询/预期；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import split_clauses

    assert split_clauses(query) == expected


async def test_explicit_top_k_and_no_split_override_defaults() -> None:
    """调试 top_k=5 及关闭拆分均须传到底层 rank。参数：无；返回：None；异常：断言失败。"""
    search, _, index, reranker = _search([_chunk(identifier) for identifier in range(1, 12)])
    result = await search.retrieve("退货申请条件，退款处理时限", top_k=5, subquery_split=False)
    assert len(result) == 5 and len(index.calls) == 1 and reranker.calls[0][2] == 5


@pytest.mark.parametrize("value", [0, -1, True, 51])
async def test_invalid_top_k_does_not_issue_retrieval(value) -> None:
    """非法显式值不能触发任何外部调用。参数：非法条数；返回：None；异常：断言失败。"""
    search, encoder, index, reranker = _search([_chunk(1)])
    with pytest.raises(ValueError):
        await search.retrieve("退款处理时限", top_k=value)
    assert not encoder.inputs and not index.calls and not reranker.calls


@pytest.mark.parametrize("score,allowed", [(0.299, False), (0.3, True), (0.8, True)])
async def test_faq_uses_highest_rerank_score_threshold(score: float, allowed: bool) -> None:
    """0.3 只用于最高重排分而非 RRF，边界等值允许。参数：分数/可用性；返回：None；异常：断言失败。"""
    search, _, _, _ = _search([_chunk(1), _chunk(2)], scores={1: score, 2: 0.01})
    result = await search.query_faq("退款处理时限")
    assert bool(result) is allowed
    if allowed:
        # 参考以最高分闸门，不逐条滤掉其他低分结果。
        assert len(result) == 2 and result[0]["answer"] == "权威答案1。"


async def test_refund_expands_three_queries_merges_up_to_30_without_final_cut() -> None:
    """退款证据不能沿用 FAQ 的最终十条截断。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import expand_policy_queries

    query = "商品损坏如何退款"
    queries = expand_policy_queries(query)
    groups = {text: list(range(index * 10 + 1, index * 10 + 11)) for index, text in enumerate(queries)}
    search, _, _, reranker = _search([_chunk(identifier) for identifier in range(1, 31)], groups=groups)
    result = await search.retrieve_policy(query)
    assert len(queries) == 3 and all(query in value for value in queries)
    assert len(result) == 30 and len({item.chunk.id for item in result}) == 30
    assert len(reranker.calls) == 3 and all(call[2] == 10 for call in reranker.calls)


async def test_refund_deduplicates_by_id_without_reranking_merged_results() -> None:
    """退款相同证据去重，三查询后不得再做第四次重排。参数：无；返回：None；异常：断言失败。"""
    search, _, _, reranker = _search([_chunk(identifier) for identifier in range(1, 11)])
    result = await search.retrieve_policy("退款条件有哪些")
    assert len(result) == 10 and len(reranker.calls) == 3


@pytest.mark.parametrize("fault", ["inactive", "pending", "vector_id", "hash", "revision"])
async def test_invalid_mysql_snapshot_is_never_returned(fault: str) -> None:
    """权威状态/指纹失配不得进入模型，全部失配是同步故障。参数：故障；返回：None；异常：断言失败。"""
    original = _chunk(1)
    search, _, index, _ = _search([original])
    changes = {"inactive": {"is_active": False}, "pending": {"vectorize_status": "pending"},
               "vector_id": {"vector_id": "7"}, "hash": {"content_hash": "b" * 64},
               "revision": {"embedding_revision": "stale"}}
    search.store._snapshots[1] = original.model_copy(update=changes[fault])
    with pytest.raises(KnowledgeSearchUnavailable):
        await search.retrieve("退款处理时限")


async def test_empty_index_with_active_text_is_failure_but_empty_database_is_normal() -> None:
    """空索引故障与真空库必须区分。参数：无；返回：None；异常：断言失败。"""
    query = "退款处理时限"
    broken, _, _, _ = _search([_chunk(1)], groups={query: []})
    with pytest.raises(KnowledgeSearchUnavailable):
        await broken.retrieve(query)
    empty, _, _, _ = _search([])
    assert await empty.retrieve(query) == []


async def test_backend_failure_is_not_empty_hits() -> None:
    """索引后端故障必须沿调用链传出，不能触发假拒答。参数：无；返回：None；异常：断言失败。"""
    search, _, index, _ = _search([_chunk(1)])

    async def broken(*args, **kwargs):
        """模拟 SDK 故障。参数：请求；返回：不返回；异常：VectorIndexUnavailable。"""
        raise VectorIndexUnavailable("测试故障")

    index.hybrid_search = broken
    with pytest.raises(VectorIndexUnavailable):
        await search.query_faq("退款处理时限")


@pytest.mark.parametrize("field,value", [("recall_top_k", 0), ("rerank_top_k", 0), ("rerank_min_score", 1.1)])
def test_retrieval_configuration_rejects_invalid_values(field: str, value) -> None:
    """配置参数不能接受无效数量/分数。参数：字段/值；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError):
        make_settings(**{field: value})
