"""知识浏览与分阶段检索的离线测试，真实服务只在数据库和模型边界使用替身。"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb import router
from app.knowledge.rerank import RankedChunk
from app.knowledge.retrieval import HybridKnowledgeSearch
from app.knowledge.store import KnowledgeStore
from app.knowledge.vector import VectorIndexUnavailable
from tests.knowledge.fakes import FakeEncoder, make_settings
from tests.rag.test_hybrid_index import HybridClient, _index


def test_collection_route_exists_and_requires_online_dependencies() -> None:
    """缺运行时应503，不能缺路由或初始化资源。参数：无；返回：None；异常：断言失败。"""
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.get("/kb/collection").status_code == 503


class IteratorBoundary:
    """短寿命 SDK 迭代器替身，核验最终关闭与不取大向量。"""

    def __init__(self, rows: list[dict], fail: bool = False) -> None:
        """参数：返回行及故障开关；返回：None；异常：无。"""
        self.rows, self.fail, self.closed = rows, fail, False

    def next(self) -> list[dict]:
        """参数：无；返回：行副本；异常：模拟SDK故障。"""
        if self.fail:
            raise RuntimeError("https://secret:credential@private/endpoint")
        return copy.deepcopy(self.rows)

    def close(self) -> None:
        """参数：无；返回：None；异常：无。"""
        self.closed = True


class InspectionClient(HybridClient):
    """保持真实 Adapter 执行，仅模拟 SDK 读操作与检索响应。"""

    def __init__(self, ids: list[int] | None = None) -> None:
        """参数：主键列表；返回：None；异常：无。"""
        super().__init__()
        self.rows = {identifier: {"id": identifier, "text": "分类：售后\n问法：退款\n答案：保留包装。",
            "content_hash": "2aac44e4291a69128a5ec522b22bff007f4ba0e1edcef221bcc6f3ec9a85fff1",
            "embedding_revision": "fake-encoder-v1", "vector": [1.0] + [0.0] * 1023}
            for identifier in (ids if ids is not None else list(range(1, 56)))}
        self.iterators: list[IteratorBoundary] = []
        self.iterator_fail = False
        self.get_change = False
        self.dense_ids = [1, 2]
        self.sparse_ids = [2, 3]
        self.fused_ids = [2, 1, 3]

    def query(self, name: str, **kwargs) -> list[dict]:
        """参数：查询载荷；返回：真实count形状；异常：无。"""
        self.calls.append(("query", kwargs))
        assert kwargs["output_fields"] == ["count(*)"]
        return [{"count(*)": len(self.rows)}]

    def query_iterator(self, name: str, **kwargs) -> IteratorBoundary:
        """参数：分页载荷；返回：短寿命迭代器；异常：无。"""
        self.calls.append(("iterator", kwargs))
        after = int(kwargs["filter"].split(">")[1])
        selected = [self.rows[key] for key in sorted(self.rows) if key > after][:kwargs["limit"]]
        fields = kwargs["output_fields"]
        iterator = IteratorBoundary([{key: row[key] for key in fields} for row in selected], self.iterator_fail)
        self.iterators.append(iterator)
        return iterator

    def get(self, name: str, ids: list[int], **kwargs) -> list[dict]:
        """参数：批量主键/明确字段；返回：原始行；异常：无。"""
        self.calls.append(("get", {"ids": ids, **kwargs}))
        rows = [{key: self.rows[identifier][key] for key in kwargs["output_fields"]}
                for identifier in ids if identifier in self.rows]
        if self.get_change and rows:
            rows[0]["content_hash"] = "c" * 64
        return rows

    def _hits(self, ids: list[int], score: float) -> list:
        """参数：主键/起始分数；返回：完整SDK命中；异常：无。"""
        return [[{"id": identifier, "distance": score - offset * .1,
                  "entity": {key: self.rows[identifier][key] for key in ("content_hash", "embedding_revision")}}
                 for offset, identifier in enumerate(ids)]]

    def search(self, name: str, **kwargs) -> list:
        """参数：单路载荷；返回：对应分数；异常：无。"""
        self.calls.append(("search", kwargs))
        return self._hits(self.dense_ids if kwargs["anns_field"] == "vector" else self.sparse_ids,
                          .8 if kwargs["anns_field"] == "vector" else 8.0)

    def hybrid_search(self, name: str, **kwargs) -> list:
        """参数：融合载荷；返回：融合分数；异常：无。"""
        self.calls.append(("hybrid", kwargs))
        return self._hits(self.fused_ids[:kwargs["limit"]], .9)


class SqlSessionBoundary:
    """真实 select 的执行边界，保留NULL、日期和全部18字段。"""

    def __init__(self, client: InspectionClient) -> None:
        """参数：SDK语料；返回：None；异常：无。"""
        self.rows = [{"id": identifier, "source_key": "doc:a", "ordinal": identifier - 1,
            "category": "售后", "questions": "退款", "answer": "保留包装。", "section_path": None,
            "content_type": None, "is_key_clause": False, "content_hash": row["content_hash"],
            "embedding_revision": "fake-encoder-v1", "is_active": True, "prev_chunk_id": None,
            "next_chunk_id": identifier + 1, "vector_id": str(identifier), "vectorize_status": "done",
            "created_at": datetime(2026, 11, 11, 8, 9, 10), "updated_at": datetime(2026, 11, 11, 8, 9, 11)}
            for identifier, row in client.rows.items()]
        self.statements = []
        self.fail = False

    async def __aenter__(self):
        """参数：无；返回：会话；异常：无。"""
        return self

    async def __aexit__(self, *args):
        """参数：上下文状态；返回：None；异常：无。"""

    async def execute(self, statement):
        """参数：真实SQL语句；返回：映射查询结果；异常：模拟连接错误。"""
        self.statements.append(statement)
        if self.fail:
            raise RuntimeError("mysql://root:password@private")
        ids = next(iter(statement.compile().params.values()))
        rows = [row for row in self.rows if row["id"] in ids]
        return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: copy.deepcopy(rows)))


class RerankerBoundary:
    """模拟外部模型评分，服务仍执行真实快照校验和Top-K。"""

    def __init__(self) -> None:
        """参数：无；返回：None；异常：无。"""
        self.calls = []
        self.error = None
        self.malformed = None

    async def rank(self, query, chunks, *, top_n):
        """参数：原问/完整合法候选/Top-K；返回：真实RankedChunk；异常：配置的模型错误。"""
        self.calls.append((query, chunks, top_n))
        if self.error:
            raise self.error
        ranked = [RankedChunk(chunk=chunk, score=.9 - offset * .1)
                  for offset, chunk in enumerate(reversed(chunks))][:top_n]
        if self.malformed:
            return self.malformed(ranked)
        return ranked


def environment():
    """参数：无；返回：真实服务与边界；异常：无。"""
    from app.knowledge.inspection import KnowledgeInspector
    client = InspectionClient()
    index = _index(client)
    index._client, index._ready = client, True
    session = SqlSessionBoundary(client)
    store = KnowledgeStore(lambda: session, None)
    encoder, reranker = FakeEncoder(), RerankerBoundary()
    search = HybridKnowledgeSearch(store, encoder, index, reranker, make_settings())
    return KnowledgeInspector(search), client, session, encoder, reranker


async def test_live_count_cursor_pages_and_raw_nulls() -> None:
    """漏页/读取向量/丢NULL/大ID精度错误均会失败。参数：无；返回：None；异常：断言失败。"""
    inspector, client, session, _, _ = environment()
    info = await inspector.collection()
    assert info["row_count"] == 55 and info["dense_dimension"] == 1024
    assert next(field for field in info["fields"] if field["name"] == "sparse_vector")["readable"] is False
    pages = [await inspector.records(after_id=0, limit=20), await inspector.records(after_id=20, limit=20),
             await inspector.records(after_id=40, limit=20)]
    assert [len(page["rows"]) for page in pages] == [20, 20, 15]
    assert [page["next_after_id"] for page in pages] == ["20", "40", "55"]
    assert [page["has_more"] for page in pages] == [True, True, False]
    assert [row["id"] for row in pages[1]["rows"]] == [str(value) for value in range(21, 41)]
    row = pages[0]["rows"][0]
    assert row["mysql"]["content_type"] is None and len(row["mysql"]) == 18
    assert row["mysql"]["next_chunk_id"] == "2"
    assert row["mysql"]["created_at"] == "2026-11-11T08:09:10"
    assert row["sync"]["status"] == "ok"
    assert all(iterator.closed for iterator in client.iterators)
    assert all(payload["output_fields"] == ["id", "text", "content_hash", "embedding_revision"]
               for name, payload in client.calls if name == "iterator")
    assert all(payload["batch_size"] == 21 and payload["limit"] == 21
               for name, payload in client.calls if name == "iterator")
    detail = await inspector.record(1)
    assert len(detail["milvus"]["vector"]) == 1024
    assert len(session.statements) == 4


async def test_iterator_failure_closes_and_sql_failure_degrades() -> None:
    """SQL故障保留Milvus行并显式告警，SDK失败仍关闭迭代器。参数：无；返回：None；异常：断言失败。"""
    inspector, client, session, _, _ = environment()
    session.fail = True
    page = await inspector.records(after_id=0, limit=20)
    assert page["warnings"] and page["rows"][0]["sync"]["status"] == "mysql_unavailable"
    client.iterator_fail = True
    with pytest.raises(VectorIndexUnavailable):
        await inspector.records(after_id=0, limit=20)
    assert client.iterators[-1].closed


@pytest.mark.parametrize("payload", [{"query": " "}, {"query": "x" * 2001}, {"query": "x", "dense_weight": True},
    {"query": "x", "dense_weight": float("nan")}, {"query": "x", "dense_weight": 0, "sparse_weight": 0},
    {"query": "x", "dense_limit": True}, {"query": "x", "candidate_limit": 51},
    {"query": "x", "rerank": 1}, {"query": "x", "extra": 1}])
def test_retrieval_request_rejects_coercions_and_out_of_bounds(payload: dict) -> None:
    """参数非法不能进入模型或RPC。参数：请求；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    with pytest.raises(ValueError):
        RetrievalDebugRequest.model_validate(payload)


async def test_weighted_recall_precedes_rerank_and_scores_all_legal_candidates() -> None:
    """候选只先裁Top-K、原问改写、把缺路分数写0均会失败。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, client, _, encoder, reranker = environment()
    request = RetrievalDebugRequest(query=" 新疆呢，金卡会员买2件能免邮吗？ ", dense_weight=.2, sparse_weight=.4, rerank_top_k=1)
    events = inspector.debug(request)
    seen = []
    async for event in events:
        seen.append(event)
        if event["event"] == "recall":
            assert not reranker.calls
            recall = event["data"]
    assert [item["event"] for item in seen] == ["status", "status", "recall", "status", "rerank", "complete"]
    assert encoder.inputs == ["新疆呢，金卡会员买2件能免邮吗？"]
    assert recall["parameters"]["dense_weight"] == pytest.approx(1 / 3)
    first_dense = recall["dense"][0]
    assert first_dense["cosine_score"] == .8 and first_dense["bm25_score"] is None
    assert recall["sparse"][0]["bm25_score"] == 8.0
    assert reranker.calls[0][0] == "新疆呢，金卡会员买2件能免邮吗？"
    assert len(reranker.calls[0][1]) == 3 and reranker.calls[0][2] == 1
    rerank = next(item["data"] for item in seen if item["event"] == "rerank")
    assert rerank["scored_ids"] == ["2", "1", "3"] and rerank["rows"][0]["id"] == "3"
    assert rerank["rows"][0]["rank_change"] == 2
    ranker = next(payload["ranker"] for name, payload in client.calls if name == "hybrid")
    assert ranker.dict()["params"]["norm_score"] is True


@pytest.mark.parametrize("strategy", ["dense", "bm25", "rrf"])
async def test_original_query_single_route_and_rrf(strategy: str) -> None:
    """纯BM25不编码，纯路不调用融合，RRF固定60。参数：策略；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, client, _, encoder, _ = environment()
    events = [event async for event in inspector.debug(RetrievalDebugRequest(query="不退款，签收7天后换货？", strategy=strategy, rerank=False))]
    assert events[-1]["event"] == "complete"
    assert encoder.inputs == ([] if strategy == "bm25" else ["不退款，签收7天后换货？"])
    searches = [payload for name, payload in client.calls if name == "search"]
    assert len(searches) == (2 if strategy == "rrf" else 1)
    if strategy == "rrf":
        assert next(payload["ranker"] for name, payload in client.calls if name == "hybrid").dict()["params"]["k"] == 60


@pytest.mark.parametrize("kind", ["hash", "text", "revision", "active", "vector_id", "pending", "changed"])
async def test_mismatched_candidates_are_excluded_before_rerank(kind: str) -> None:
    """正文/hash/发布资格/版本/回读变化任何失配不得进入模型。参数：错误类型；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, client, session, _, reranker = environment()
    row = session.rows[0]
    if kind == "hash": row["content_hash"] = "e" * 64
    elif kind == "text": row["answer"] = "新正文"
    elif kind == "revision": row["embedding_revision"] = "old"
    elif kind == "active": row["is_active"] = False
    elif kind == "vector_id": row["vector_id"] = "99"
    elif kind == "pending": row["vectorize_status"] = "pending"
    else: client.get_change = True
    events = [event async for event in inspector.debug(RetrievalDebugRequest(query="退款"))]
    assert events[-1]["event"] == "complete"
    assert 1 not in [chunk.id for chunk in reranker.calls[0][1]]


@pytest.mark.parametrize("kind", ["model", "overflow", "sql", "duplicate", "missing", "snapshot", "score"])
async def test_rerank_errors_preserve_recall_and_hide_secrets(kind: str) -> None:
    """精排失败只有error终止且保留完整召回，不泄露原异常。参数：故障；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, _, session, _, reranker = environment()
    if kind == "model": reranker.error = RuntimeError("https://secret:password@private")
    elif kind == "overflow": reranker.error = ValueError("查询超过重排128token")
    elif kind == "sql": session.fail = True
    elif kind == "duplicate": reranker.malformed = lambda rows: [rows[0]] * len(rows)
    elif kind == "missing": reranker.malformed = lambda rows: rows[:-1]
    elif kind == "snapshot": reranker.malformed = lambda rows: [rows[0].model_copy(update={"chunk": rows[0].chunk.model_copy(update={"section_path": "changed"})}), *rows[1:]]
    else: reranker.malformed = lambda rows: [rows[0].model_copy(update={"score": float("nan")}), *rows[1:]]
    events = [event async for event in inspector.debug(RetrievalDebugRequest(query="退款"))]
    assert "recall" in [event["event"] for event in events]
    assert events[-1]["event"] == "error" and all(event["event"] != "complete" for event in events)
    assert "password" not in str(events) and "private" not in str(events)


async def test_cancel_propagates_without_closing_shared_resources() -> None:
    """取消不转error、不关闭共享编码器。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, _, _, encoder, reranker = environment()
    reranker.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _ = [event async for event in inspector.debug(RetrievalDebugRequest(query="退款"))]
    assert not encoder.closed


async def test_zero_weight_branch_not_in_weighted_ranker_requests() -> None:
    """零权重路可以展示，但不能送入融合制造假贡献。参数：无；返回：None；异常：断言失败。"""
    client = InspectionClient()
    index = _index(client)
    index._client, index._ready = client, True
    await index.weighted_search([1.] + [0.] * 1023, "退款", dense_weight=0., sparse_weight=1.)
    payload = next(payload for name, payload in client.calls if name == "hybrid")
    assert len(payload["reqs"]) == 1 and payload["reqs"][0].anns_field == "sparse_vector"
    assert payload["ranker"].dict()["params"]["weights"] == [1.]


def api_environment():
    """参数：无；返回：真实路由与服务边界；异常：无。"""
    inspector, client, session, encoder, reranker = environment()
    app = FastAPI()
    app.include_router(router)
    app.state.chat_service = SimpleNamespace(executor=SimpleNamespace(support=SimpleNamespace(knowledge_search=inspector.search)))
    return app, inspector, client, session, encoder, reranker


def test_http_collection_details_validation_and_sse() -> None:
    """真实HTTP同时核验公开结构、404/422与SSE顺序。参数：无；返回：None；异常：断言失败。"""
    app, _, boundary, _, _, _ = api_environment()
    with TestClient(app) as client:
        info = client.get("/kb/collection")
        assert info.status_code == 200 and info.json()["row_count"] == 55
        assert client.get("/kb/records/1").json()["milvus"]["id"] == "1"
        assert client.get("/kb/records/999").status_code == 404
        for url in ("/kb/records?after_id=-1", "/kb/records?limit=101", "/kb/records/0", "/kb/records/9223372036854775808"):
            assert client.get(url).status_code == 422
        assert client.post("/kb/retrieval-debug", json={"query": "退款", "rerank": 1}).status_code == 422
        response = client.post("/kb/retrieval-debug", json={"query": "退款", "rerank_top_k": 1})
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        blocks = response.text.strip().split("\n\n")
        assert [block.split("\n")[0] for block in blocks] == ["event: status", "event: status", "event: recall", "event: status", "event: rerank", "event: complete"]
        assert json.loads(blocks[2].split("data: ")[1])["counts"]["eligible"] == 3
        boundary.iterator_fail = True
        failure = client.get("/kb/records")
        assert failure.status_code == 503 and "credential" not in failure.text


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_http_nonfinite_weights_return_safe_422(value: str) -> None:
    """非标准JSON非有限数值应422，不能触发编码错误500。参数：数字文本；返回：None；异常：断言失败。"""
    app, _, _, _, _, _ = api_environment()
    with TestClient(app) as client:
        response = client.post("/kb/retrieval-debug", content='{"query":"退款","dense_weight":' + value + '}', headers={"content-type": "application/json"})
        assert response.status_code == 422


def test_preflight_rejects_unready_and_nonhybrid_without_rpc() -> None:
    """预检错误开流前503，不能创建client或调用模型。参数：无；返回：None；异常：断言失败。"""
    app, inspector, boundary, _, encoder, reranker = api_environment()
    with TestClient(app) as client:
        inspector.search.index._ready = False
        assert client.post("/kb/retrieval-debug", json={"query": "退款"}).status_code == 503
        inspector.search.index._ready = True
        inspector.search.config = inspector.search.config.model_copy(update={"knowledge_retrieval_mode": "dense"})
        assert client.get("/kb/collection").status_code == 503
    assert not boundary.calls and not encoder.inputs and not reranker.calls


async def test_big_ids_empty_pages_and_missing_mysql_remain_distinct() -> None:
    """INT64最大主键/指针不损精度，缺字段与故障区别明确。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import KnowledgeInspector
    identifier = 9223372036854775807
    boundary = InspectionClient([identifier])
    index = _index(boundary)
    index._client, index._ready = boundary, True
    session = SqlSessionBoundary(boundary)
    session.rows[0]["prev_chunk_id"] = identifier - 1
    session.rows[0]["next_chunk_id"] = None
    inspector = KnowledgeInspector(HybridKnowledgeSearch(KnowledgeStore(lambda: session, None), FakeEncoder(), index, RerankerBoundary(), make_settings()))
    page = await inspector.records(after_id=identifier - 1, limit=1)
    assert page["rows"][0]["id"] == "9223372036854775807" and page["next_after_id"] == "9223372036854775807"
    assert page["rows"][0]["mysql"]["prev_chunk_id"] == "9223372036854775806"
    empty = await inspector.records(after_id=identifier, limit=1)
    assert empty["rows"] == [] and not empty["has_more"] and empty["next_after_id"] == "9223372036854775807"
    session.rows.clear()
    missing = await inspector.record(identifier)
    assert missing["mysql"] is None and missing["sync"]["status"] == "missing_mysql"


async def test_all_invalid_or_empty_candidates_skip_model_normally() -> None:
    """空候选或同步全部失配不能伪装模型错误。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, boundary, session, _, reranker = environment()
    for row in session.rows:
        row["is_active"] = False
    events = [event async for event in inspector.debug(RetrievalDebugRequest(query="退款"))]
    assert events[-1]["event"] == "complete" and not reranker.calls
    result = next(event["data"] for event in events if event["event"] == "rerank")
    assert result["rows"] == [] and result["scored_ids"] == [] and len(result["skipped"]) == 3
    boundary.dense_ids, boundary.sparse_ids, boundary.fused_ids = [], [], []
    empty = [event async for event in inspector.debug(RetrievalDebugRequest(query="没有匹配"))]
    assert empty[-1]["event"] == "complete" and not reranker.calls


async def test_debug_timeout_preserves_recall_and_does_not_close_model() -> None:
    """整次线上超时覆盖重排且保留召回，不销毁共享模型。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, _, _, encoder, reranker = environment()
    entered = asyncio.Event()

    async def slow_rank(*args, **kwargs):
        """参数：模型输入；返回：无；异常：外层超时取消。"""
        entered.set()
        await asyncio.sleep(10)

    reranker.rank = slow_rank
    inspector.search.config = inspector.search.config.model_copy(update={"knowledge_timeout_seconds": .1})
    events = [event async for event in inspector.debug(RetrievalDebugRequest(query="退款"))]
    assert entered.is_set() and events[-1]["event"] == "error" and events[-1]["data"]["stage"] == "rerank"
    assert any(event["event"] == "recall" for event in events) and not encoder.closed


async def test_sql_timeout_is_explicit_degradation(monkeypatch) -> None:
    """SQL超时不得当作缺原文，取消本次查询后保留Milvus页。参数：替换夹具；返回：None；异常：断言失败。"""
    from app.knowledge import inspection
    inspector, _, session, _, _ = environment()
    cancelled = asyncio.Event()

    async def slow_execute(statement):
        """参数：真实select；返回：无；异常：本次SQL等待预算取消。"""
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    session.execute = slow_execute
    monkeypatch.setattr(inspection, "_MYSQL_TIMEOUT_SECONDS", .01)
    page = await inspector.records(after_id=0, limit=1)
    assert cancelled.is_set() and len(page["rows"]) == 1
    assert page["warnings"] and page["rows"][0]["sync"]["status"] == "mysql_unavailable"


@pytest.mark.parametrize("failure", ["duplicate", "foreign", "missing_field"])
async def test_adapter_rejects_invalid_readback_identity(failure: str) -> None:
    """回读重复/范围外主键/缺字段必须失败，不能静默覆盖。参数：故障类型；返回：None；异常：断言失败。"""
    boundary = InspectionClient()
    index = _index(boundary)
    index._client, index._ready = boundary, True
    original = boundary.get

    def broken_get(*args, **kwargs):
        """参数：SDK读载荷；返回：损坏行；异常：无。"""
        rows = original(*args, **kwargs)
        if failure == "duplicate": rows.append(copy.deepcopy(rows[0]))
        elif failure == "foreign": rows[0]["id"] = 999
        else: rows[0].pop("text")
        return rows

    boundary.get = broken_get
    with pytest.raises(VectorIndexUnavailable):
        await index.inspection_records([1])


async def test_sse_send_failure_closes_stream_before_rerank() -> None:
    """网络发送召回失败时必须关闭本次迭代器，不能继续共享模型重排。参数：无；返回：None；异常：断言失败。"""
    from app.api.kb import retrieval_debug
    from app.knowledge.inspection import RetrievalDebugRequest
    inspector, _, _, encoder, reranker = environment()
    response = await retrieval_debug(RetrievalDebugRequest(query="退款"), inspector)

    async def send(message):
        """参数：ASGI发送消息；返回：None；异常：模拟断网。"""
        if message["type"] == "http.response.body" and b"event: recall" in message["body"]:
            raise RuntimeError("disconnected")

    async def receive():
        """参数：无；返回：无；异常：测试仅使用ASGI2.4直接send。"""
        await asyncio.sleep(10)

    with pytest.raises(RuntimeError, match="disconnected"):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert not reranker.calls and not encoder.closed
    with pytest.raises(StopAsyncIteration):
        await response.body_iterator.__anext__()
