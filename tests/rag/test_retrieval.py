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


async def test_duplicate_subquery_evidence_preserves_strongest_score_for_faq_gate() -> None:
    """同主键在另一子句高相关时不能因先遇低分而被错误拒答。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.rerank import RankedChunk

    search, _, _, reranker = _search([_chunk(1)])

    async def rank(query, chunks, *, top_n):
        """模拟同一证据对两子句的不同真实相关度。参数：查询/候选/数量；返回：排名；异常：无。"""
        return [RankedChunk(chunk=chunks[0], score=0.9 if query == "退款处理时限" else 0.1)]

    reranker.rank = rank
    result = await search.query_faq("退货申请条件，退款处理时限")
    assert [row["id"] for row in result] == [1]


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
    ("甲；退货条件；退款时限", ["甲，退货条件", "甲，退款时限"]),
])
def test_clause_split_uses_four_character_minimum(query: str, expected: list[str]) -> None:
    """不足四字不发独立调用，不能形成多个子句时原句保留。参数：查询/预期；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import split_clauses

    assert split_clauses(query) == expected


@pytest.mark.parametrize("top_k", [5, 20, 50])
async def test_explicit_top_k_and_no_split_override_defaults(top_k: int) -> None:
    """旧入口显式大小上限及关闭拆分均须传到底层重排。

    参数：top_k 既有合法范围内的显式结果上限，分别为 5、20、50。
    返回：None。
    异常：AssertionError，旧入口被新增十条上限影响或关闭拆句无效。
    """
    search, _, index, reranker = _search([_chunk(identifier) for identifier in range(1, 61)])
    result = await search.retrieve("退货申请条件，退款处理时限", top_k=top_k, subquery_split=False)
    assert len(result) == top_k and len(index.calls) == 1 and reranker.calls[0][2] == top_k


@pytest.mark.parametrize("value", [0, -1, True, 51])
@pytest.mark.parametrize("prepared_queries", [None, ["退款处理时限"]])
async def test_invalid_top_k_does_not_issue_retrieval(value, prepared_queries: list[str] | None) -> None:
    """新旧入口均先拒绝非法显式值，不能通过裁剪上限把非法值变为有效请求。

    参数：value 非法条数，含零、负数、布尔值及超过召回上限的整数；prepared_queries 为旧入口或单条有效查询。
    返回：None。
    异常：AssertionError，非法值触发了编码、索引或重排。
    """
    search, encoder, index, reranker = _search([_chunk(1)])
    with pytest.raises(ValueError):
        await search.retrieve("退款处理时限", top_k=value, prepared_queries=prepared_queries)
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


def test_merge_ranked_chunks_picks_highest_score_and_stable_tie_order() -> None:
    """跨查询合并按chunk ID保留最高rerank分，同分按chunk ID升序排列。

    参数：无。
    返回：None。
    异常：AssertionError，未保留后次更高分或同分排序不稳定。
    """
    from app.knowledge.rerank import RankedChunk
    from app.knowledge.retrieval import merge_ranked_chunks

    groups = [
        [RankedChunk(chunk=_chunk(7), score=0.4), RankedChunk(chunk=_chunk(3), score=0.6)],
        [RankedChunk(chunk=_chunk(7), score=0.9)],
    ]
    merged = merge_ranked_chunks(groups)
    assert [item.chunk.id for item in merged] == [7, 3]
    assert merged[0].score == 0.9  # chunk 7 的后次结果高于首次 0.4

    ties = [
        [RankedChunk(chunk=_chunk(5), score=0.8)],
        [RankedChunk(chunk=_chunk(2), score=0.8)],
    ]
    assert [item.chunk.id for item in merge_ranked_chunks(ties)] == [2, 5]


@pytest.mark.parametrize("entrypoint", ["retrieve", "retrieve_policy"])
async def test_prepared_queries_skip_rewriter_and_fixed_expansion_with_best_score_merge(entrypoint: str) -> None:
    """传入prepared_queries时不触发旧改写或固定退款扩写，且重复chunk取最高分并继续校验权威回读。

    参数：entrypoint 为普通或政策检索入口名称。
    返回：None。
    异常：AssertionError，触发了旧改写、保留了旧首条低分或权威回读异常被吞。
    """
    from app.knowledge.query_rewrite import QueryRewriter
    from app.knowledge.rerank import RankedChunk
    from tests.rag.test_query_rewrite import RewriteModel

    chunks = [_chunk(3), _chunk(7)]
    groups = {
        "MH-W99未拆封，已签收退货条件": [7, 3],
        "MH-W99未拆封；退货限制条款": [7],
        "MH-W99未拆封。退货运费承担规则": [3],
    }
    search, encoder, index, reranker = _search(chunks, groups=groups)
    rewriter_model = RewriteModel()
    search.query_rewriter = QueryRewriter(rewriter_model.factory)

    async def rank(query, candidates, *, top_n):
        """对不同完整查询返回不同相关度，验证后次高分覆盖首次低分。

        参数：query 非空完整查询；candidates 已回读的权威候选列表；top_n 正整数结果上限。
        返回：list[RankedChunk]，不超过 top_n 的候选排名。
        异常：StopIteration，测试夹具缺少应有的候选主键。
        """
        reranker.calls.append((query, [candidate.id for candidate in candidates], top_n))
        if query == "MH-W99未拆封，已签收退货条件":
            return [
                RankedChunk(chunk=next(c for c in candidates if c.id == 3), score=0.6),
                RankedChunk(chunk=next(c for c in candidates if c.id == 7), score=0.4),
            ][:top_n]
        if query == "MH-W99未拆封；退货限制条款":
            return [RankedChunk(chunk=next(c for c in candidates if c.id == 7), score=0.9)][:top_n]
        return [RankedChunk(chunk=next(c for c in candidates if c.id == 3), score=0.5)][:top_n]

    reranker.rank = rank
    prepared_list = list(groups)
    merged = await getattr(search, entrypoint)(
        "订单1001的MH-W99未拆封能退吗 已签收",
        prepared_queries=prepared_list,
    )
    rewrite_call_count = len(rewriter_model.inputs)
    assert rewrite_call_count == 0
    assert encoder.inputs == prepared_list
    assert [call[0] for call in index.calls] == prepared_list
    assert [call[0] for call in reranker.calls] == prepared_list
    assert [item.chunk.id for item in merged] == [7, 3]
    assert merged[0].score == 0.9

    # 权威原文回读失配时继续抛错，不伪装成空命中。
    search.store._snapshots[7] = chunks[1].model_copy(update={"is_active": False})
    search.store._snapshots[3] = chunks[0].model_copy(update={"is_active": False})
    with pytest.raises(KnowledgeSearchUnavailable):
        await getattr(search, entrypoint)(
            "订单1001的MH-W99未拆封能退吗 已签收",
            prepared_queries=prepared_list,
        )


@pytest.mark.parametrize("entrypoint", ["retrieve", "retrieve_policy"])
async def test_single_prepared_query_keeps_punctuation_model_and_status(entrypoint: str) -> None:
    """单条带标点、型号和状态的 prepared 查询在三个外部边界均完整传递且只调用一次。

    参数：entrypoint 为普通或政策检索入口名称。
    返回：None。
    异常：AssertionError，查询被拆开、状态丢失或单条结果突破十条。
    """
    query = "MH-W99未拆封，退货运费谁承担"
    search, encoder, index, reranker = _search(
        [_chunk(identifier) for identifier in range(1, 21)], rerank_top_k=20
    )
    result = await getattr(search, entrypoint)(query, prepared_queries=[query])
    assert encoder.inputs == [query]
    assert index.calls == [(query, 50, 50, 50)]
    assert reranker.calls == [(query, list(range(1, 21)), 10)]
    assert [item.chunk.id for item in result] == list(range(1, 11))


@pytest.mark.parametrize("entrypoint,configured,top_k,per_query,expected_ids", [
    ("retrieve", 20, None, 10, list(range(60, 50, -1))),
    ("retrieve", 10, 50, 10, list(range(60, 50, -1))),
    ("retrieve", 4, None, 4, [60, 59, 58, 57]),
    ("retrieve", 20, 5, 5, [60, 59, 58, 57, 56]),
    ("retrieve_policy", 20, None, 10, list(range(60, 50, -1)) + list(range(40, 30, -1)) + list(range(20, 10, -1))),
    ("retrieve_policy", 4, None, 4, [60, 59, 58, 57, 40, 39, 38, 37, 20, 19, 18, 17]),
])
async def test_prepared_limits_preserve_highest_scored_evidence(
    entrypoint: str, configured: int, top_k: int | None, per_query: int, expected_ids: list[int]
) -> None:
    """三条查询每条最多十条，普通最终十条、政策最终三十条，并尊重更小上限和跨查询高分。

    参数：entrypoint 普通或政策入口；configured 合法配置上限；top_k 普通入口显式上限或 None；
        per_query 每条预期重排数量；expected_ids 按手工标注相关度应保留的主键顺序。
    返回：None。
    异常：AssertionError，上限未生效、较小限制失效或高分结果被旧轮转截断。
    """
    queries = ["已签收商品退货条件", "商品退货限制条款", "退货运费承担规则"]
    groups = {queries[0]: list(range(1, 21)), queries[1]: list(range(21, 41)), queries[2]: list(range(41, 61))}
    search, encoder, index, reranker = _search(
        [_chunk(identifier) for identifier in range(1, 61)],
        groups=groups, scores={identifier: identifier / 100 for identifier in range(1, 61)},
        rerank_top_k=configured,
    )
    options = {"top_k": top_k} if entrypoint == "retrieve" else {}
    result = await getattr(search, entrypoint)("已签收商品退货规则", prepared_queries=queries, **options)
    assert [item.chunk.id for item in result] == expected_ids
    assert encoder.inputs == queries and [call[0] for call in index.calls] == queries
    assert [call[0] for call in reranker.calls] == queries
    assert [call[2] for call in reranker.calls] == [per_query, per_query, per_query]


@pytest.mark.parametrize("entrypoint,expected_count", [("retrieve", 20), ("retrieve_policy", 60)])
async def test_legacy_configured_large_limits_remain_available(entrypoint: str, expected_count: int) -> None:
    """prepared_queries=None 的旧入口仍接受大于十条的合法配置。

    参数：entrypoint 普通或政策入口；expected_count 旧入口配置二十条时应返回的总数。
    返回：None。
    异常：AssertionError，新分支的上限意外改变旧入口行为。
    """
    from app.knowledge.retrieval import expand_policy_queries

    query = "已签收商品退货条件"
    queries = expand_policy_queries(query) if entrypoint == "retrieve_policy" else [query]
    groups = {text: list(range(position * 20 + 1, position * 20 + 21)) for position, text in enumerate(queries)}
    search, _, _, reranker = _search(
        [_chunk(identifier) for identifier in range(1, 61)], groups=groups, rerank_top_k=20
    )
    result = await getattr(search, entrypoint)(query)
    assert len(result) == expected_count
    assert [call[2] for call in reranker.calls] == [20] * len(queries)


async def test_prepared_queries_still_validated() -> None:
    """prepared_queries仍执行逐条确定性校验且不借用history放行旧订单，任一条非法时零外部调用。

    参数：无。
    返回：None。
    异常：AssertionError，放行了历史订单1002或触发了编码/召回/重排调用。
    """
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError
    from tests.rag.test_query_rewrite import RewriteModel

    search, encoder, index, reranker = _search([_chunk(1), _chunk(2)])
    rewriter_model = RewriteModel()
    search.query_rewriter = QueryRewriter(rewriter_model.factory)

    # 1. 当前1001、历史1002，候选仅1002或“1001和1002”均拒绝（即使传入history也不得放行）
    for illegal_candidate in [["订单1002退货条件"], ["订单1001和1002退货条件"]]:
        with pytest.raises(QueryRewriteError):
            await search.retrieve_policy(
                "订单1001能退货吗 已签收",
                history="此前讨论过订单1002",
                prepared_queries=illegal_candidate,
            )

    # 2. 通用运费原问候选夹带历史1002也拒绝
    with pytest.raises(QueryRewriteError):
        await search.retrieve(
            "运费怎么算",
            history="此前讨论过订单1002",
            prepared_queries=["订单1002运费怎么算"],
        )

    # 3. 三条中只有第三条非法时，encoder/index/reranker全部零调用
    with pytest.raises(QueryRewriteError):
        await search.retrieve_policy(
            "订单1001能退货吗 已签收",
            history="此前讨论过订单1002",
            prepared_queries=["已签收退货申请条件", "退货运费承担规则", "订单1002退货时限"],
        )
    assert not encoder.inputs and not index.calls and not reranker.calls
    assert len(rewriter_model.inputs) == 0

    # 4. 合法查询通过且rewrite_call_count为零；超长、空列表或两条查询明确拒绝不裁短；真空库返回正常空列表
    for bad_shape in [[], ["查询1", "查询2"], ["退" * 2001]]:
        with pytest.raises(QueryRewriteError):
            await search.retrieve("运费怎么算", prepared_queries=bad_shape)
    assert not encoder.inputs and not index.calls and not reranker.calls

    empty_search, _, _, _ = _search([])
    assert await empty_search.retrieve_policy("运费怎么算", prepared_queries=["运费规则"]) == []

    valid_hits = await search.retrieve(
        "运费怎么算",
        history="此前讨论过订单1002",
        prepared_queries=["订单运费与包邮门槛规则"],
    )
    rewrite_call_count = len(rewriter_model.inputs)
    assert rewrite_call_count == 0
    assert len(valid_hits) == 2 and encoder.inputs == ["订单运费与包邮门槛规则"]
