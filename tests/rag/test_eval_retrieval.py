"""四策略共用查询、独立分数域及完整排名的检索回归。"""

import pytest

from app.knowledge.schemas import VectorHit
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, make_settings
from tests.rag.test_retrieval import RerankBoundary, _chunk


class IndexBoundary:
    """仅替换外部Milvus，保留过滤和调用轨迹。"""

    def __init__(self, chunks):
        """保存语料。参数：chunks 快照；返回：None；异常：无。"""
        self.chunks, self.calls = chunks, []

    def hits(self, strategy, limit, filter_ids):
        """返回明确不同分数空间。参数：策略/上限/过滤ID；返回：命中；异常：无。"""
        self.calls.append((strategy, limit, filter_ids))
        return [VectorHit(id=row.id, score=12.0 if strategy == "bm25" else 0.02,
                          content_hash=row.content_hash, embedding_revision=row.embedding_revision)
                for row in self.chunks if filter_ids is None or row.id in filter_ids][:limit]

    async def search(self, vector, limit, *, filter_ids=None):
        """模拟Dense边界。参数：向量/上限/过滤；返回：命中；异常：无。"""
        return self.hits("dense", limit, filter_ids)

    async def search_bm25(self, query, limit, *, filter_ids=None):
        """模拟BM25边界。参数：查询/上限/过滤；返回：命中；异常：无。"""
        return self.hits("bm25", limit, filter_ids)

    async def hybrid_search(self, vector, query, *, dense_limit, sparse_limit, limit, filter_ids=None):
        """模拟融合边界。参数：查询/两路及融合上限/过滤；返回：命中；异常：断言失败。"""
        assert dense_limit == sparse_limit == limit == 50
        return self.hits("hybrid", limit, filter_ids)


async def test_four_strategies_share_prepared_query_without_bm25_embedding() -> None:
    """纯BM25不编码且其他策略复用同次向量；BM25分数大于1不被裁剪。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import EvaluationSearch, prepare_query

    chunks = [_chunk(index) for index in range(1, 61)]
    encoder, index, reranker = FakeEncoder(), IndexBoundary(chunks), RerankBoundary()
    service = EvaluationSearch(MemoryKnowledgeStore(snapshots=chunks), encoder, index, reranker, make_settings())
    prepared = prepare_query("退货申请条件", policy=False)
    bm25 = await service.retrieve(prepared, strategy="bm25", corpus=chunks)
    assert not encoder.inputs
    assert len(bm25.ranked) == 50 and len(bm25.selected) == 10
    assert bm25.ranked[0].score == 12
    for strategy in ("dense", "hybrid", "hybrid_rerank"):
        result = await service.retrieve(prepared, strategy=strategy, corpus=chunks)
    assert encoder.inputs == ["退货申请条件"]
    assert len(reranker.calls) == 1 and reranker.calls[0][2] == 10
    assert len(result.ranked) == 10


async def test_filter_empty_is_normal_and_filter_is_checked_after_index() -> None:
    """筛选内空语料正常空结果，索引返回越权过滤项报错。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import EvaluationSearch, prepare_query
    from app.knowledge.search import KnowledgeSearchUnavailable

    chunks = [_chunk(1), _chunk(2)]
    index = IndexBoundary(chunks)
    service = EvaluationSearch(MemoryKnowledgeStore(snapshots=chunks), FakeEncoder(), index, RerankBoundary(), make_settings())
    empty = await service.retrieve(prepare_query("不存在的来源", policy=False), strategy="dense", corpus=chunks,
                                   filters={"source_key": "doc:absent.md"})
    assert empty.ranked == [] and index.calls == []

    async def wrong_scope(vector, limit, *, filter_ids=None):
        """模拟索引忽略筛选。参数：向量/上限/过滤；返回：全部；异常：无。"""
        return index.hits("dense", limit, None)

    index.search = wrong_scope
    with pytest.raises(KnowledgeSearchUnavailable, match="过滤"):
        await service.retrieve(prepare_query("退货申请条件", policy=False), strategy="dense", corpus=chunks,
                               filters={"source_key": "doc:1.md"})


async def test_adapter_filters_both_rrf_branches() -> None:
    """过滤条件须在两路召回之前生效，不能召回后丢弃。参数：无；返回：None；异常：断言失败。"""
    from tests.rag.test_hybrid_index import HybridClient, _index

    client = HybridClient()
    index = _index(client)
    try:
        await index.ensure_ready()
        await index.hybrid_search([1.0] + [0.0] * 1023, "退款申请", filter_ids=[7, 9])
        payload = next(payload for name, payload in client.calls if name == "hybrid")
        assert all(request.expr == "id in [7,9]" for request in payload["reqs"])
        await index.search_bm25("退款申请", filter_ids=[7])
        assert client.calls[-1][1]["filter"] == "id in [7]"
        with pytest.raises(ValueError):
            await index.search_bm25("退款申请", filter_ids=["7 or true"])
    finally:
        await index.aclose()


async def test_unknown_filter_rejected_even_if_corpus_is_empty() -> None:
    """空语料不能绕过过滤字段校验。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import EvaluationSearch, prepare_query

    service = EvaluationSearch(MemoryKnowledgeStore(), FakeEncoder(), IndexBoundary([]), RerankBoundary(), make_settings())
    with pytest.raises(ValueError, match="过滤"):
        await service.retrieve(prepare_query("一个测试问题"), strategy="dense", corpus=[], filters={"arbitrary": "x"})


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
async def test_policy_ranking_prefix_contains_all_generation_candidates(strategy) -> None:
    """三路各50个独立命中时，后两路生成证据也必须位于计分排名前缀。参数：strategy；返回：None；异常：断言失败。"""
    from app.knowledge.retrieval import EvaluationSearch, EvaluationRetrieval, RetrievalCandidate, prepare_query
    from app.evaluation.metrics import retrieval_metrics

    chunks = [_chunk(index) for index in range(1, 151)]
    index = IndexBoundary(chunks)
    service = EvaluationSearch(MemoryKnowledgeStore(snapshots=chunks), FakeEncoder(), index, RerankBoundary(), make_settings())
    prepared = prepare_query("我要退款", policy=True, subquery_split=False)

    async def recall(query, prepared, strategy, scope, filtered):
        """替换单路外部召回为三份独立排名。参数：当前查询及策略范围；返回：50条；异常：无。"""
        offset = [group[0] for group in prepared.groups].index(query) * 50
        return [RetrievalCandidate(chunk=row, score=0.5) for row in chunks[offset:offset + 50]]

    service._one = recall
    result = await service.retrieve(prepared, strategy=strategy, corpus=chunks)
    selected = [item.chunk.id for item in result.selected]
    ranking = [item.chunk.id for item in result.ranked]
    assert selected == [*range(1, 11), *range(51, 61), *range(101, 111)]
    assert ranking[:30] == selected and len(ranking) == 50
    assert retrieval_metrics(ranking, [{51}, {101}])["recall@50"] == 1
