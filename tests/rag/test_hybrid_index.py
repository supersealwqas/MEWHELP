"""原生 BM25/RRF Adapter 的离线边界测试，不用客户端词频模拟 Milvus。"""

from __future__ import annotations

import copy
import json

import pytest

from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, VectorRecord
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint, format_knowledge
from app.knowledge.vector import VectorIndexUnavailable
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, MemoryVectorIndex, make_settings


class HybridClient:
    """仅替换外部 SDK 连接，保留真实 schema/RRF 对象和 Adapter 逻辑。"""

    def __init__(self) -> None:
        """初始化 SDK 观测与合法描述。参数：无；返回：None；异常：无。"""
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.collections = ["knowledge_ch04"]
        self.description = {
            "description": "mewhelp-ch04:hybrid:char-reference:v1", "auto_id": False,
            "enable_dynamic_field": False, "consistency_level": "Strong",
            "fields": [
                {"name": "id", "type": 5, "is_primary": True, "auto_id": False},
                {"name": "vector", "type": 101, "params": {"dim": 1024}},
                {"name": "sparse_vector", "type": 104, "is_function_output": True},
                {"name": "content_hash", "type": 21, "params": {"max_length": 64}},
                {"name": "embedding_revision", "type": 21, "params": {"max_length": 128}},
                {"name": "text", "type": 21, "params": {
                    "max_length": 8192, "enable_analyzer": "true", "analyzer_params": '{"type":"chinese"}',
                }},
            ],
            "functions": [{"name": "text_bm25", "type": 1, "input_field_names": ["text"],
                           "output_field_names": ["sparse_vector"], "params": {}}],
        }

    def list_collections(self, **kwargs) -> list[str]:
        """读取集合。参数：SDK 超时；返回：集合列表；异常：无。"""
        return list(self.collections)

    def describe_collection(self, name: str, **kwargs) -> dict:
        """读取集合结构。参数：集合名/SDK 参数；返回：描述副本；异常：无。"""
        return copy.deepcopy(self.description)

    def list_indexes(self, name: str, field_name: str, **kwargs) -> list[str]:
        """读取字段索引。参数：集合/字段/超时；返回：索引名；异常：无。"""
        return [field_name]

    def describe_index(self, name: str, index_name: str, **kwargs) -> dict:
        """读取索引度量。参数：集合/索引/超时；返回：描述；异常：无。"""
        return {"field_name": index_name, "index_type": "AUTOINDEX" if index_name == "vector" else "SPARSE_INVERTED_INDEX",
                "metric_type": "COSINE" if index_name == "vector" else "BM25"}

    def load_collection(self, name: str, **kwargs) -> None:
        """记录加载。参数：集合/超时；返回：None；异常：无。"""
        self.calls.append(("load", kwargs))

    def create_collection(self, name: str, **kwargs) -> None:
        """记录真实 schema 的创建载荷。参数：集合/SDK 载荷；返回：None；异常：无。"""
        self.calls.append(("create", kwargs))
        self.collections.append(name)

    def hybrid_search(self, name: str, **kwargs) -> list:
        """记录两路 SDK 载荷并返回有同步字段的候选。参数：集合/请求；返回：SDK 命中；异常：无。"""
        self.calls.append(("hybrid", kwargs))
        return [[{"id": 7, "distance": 0.02, "entity": {"content_hash": "a" * 64, "embedding_revision": "rev"}}]]

    def search(self, name: str, **kwargs) -> list:
        """记录纯 Dense/BM25 请求。参数：集合/请求；返回：SDK 命中；异常：无。"""
        self.calls.append(("search", kwargs))
        return []

    def upsert(self, name: str, data: list, **kwargs) -> None:
        """记录写入正文载荷。参数：集合/记录/超时；返回：None；异常：无。"""
        self.calls.append(("upsert", {"data": data, **kwargs}))

    def close(self) -> None:
        """关闭边界。参数：无；返回：None；异常：无。"""
        self.closed = True


class Bootstrap:
    """目标数据库已存在的 SDK 替身。"""

    def list_databases(self, **kwargs) -> list[str]:
        """列出库。参数：超时；返回：库名；异常：无。"""
        return ["mewhelp_ch03"]

    def close(self) -> None:
        """结束引导连接。参数：无；返回：None；异常：无。"""


def _index(client: HybridClient):
    """构造真实混合 Adapter。参数：外部 SDK 边界；返回：索引；异常：无。"""
    from app.knowledge.hybrid_index import MilvusHybridIndex

    return MilvusHybridIndex(make_settings(), client_factory=lambda _: client,
                             bootstrap_factory=lambda _: Bootstrap())


async def test_hybrid_passes_two_50_requests_and_fuses_only_50() -> None:
    """两路各 50 而融合不能变 100，RRF 与同步字段正确传入。参数：无；返回：None；异常：断言失败。"""
    client = HybridClient()
    index = _index(client)
    try:
        await index.ensure_ready()
        hits = await index.hybrid_search([1.0] + [0.0] * 1023, "退款如何申请")
        payload = next(payload for name, payload in client.calls if name == "hybrid")
        assert payload["limit"] == 50
        assert [request.limit for request in payload["reqs"]] == [50, 50]
        assert [request.anns_field for request in payload["reqs"]] == ["vector", "sparse_vector"]
        assert payload["reqs"][1].data == ["退款如何申请"]
        assert payload["ranker"].dict() == {"strategy": "rrf", "params": {"k": 60}}
        assert payload["timeout"] > 0 and payload["consistency_level"] == "Strong"
        assert [hit.id for hit in hits] == [7]
        assert hits[0].content_hash == "a" * 64
    finally:
        await index.aclose()


def test_created_schema_has_chinese_bm25_function_and_index() -> None:
    """错误 analyzer 或函数绑定会让中文原生全文检索失效。参数：无；返回：None；异常：断言失败。"""
    client = HybridClient()
    _index(client)._create_collection(client)
    payload = next(payload for name, payload in client.calls if name == "create")
    schema = payload["schema"].to_dict()
    fields = {field["name"]: field for field in schema["fields"]}
    assert fields["text"]["params"]["enable_analyzer"] is True
    assert json.loads(fields["text"]["params"]["analyzer_params"]) == {"type": "chinese"}
    assert fields["vector"]["params"]["dim"] == 1024
    function = schema["functions"][0]
    assert function["input_field_names"] == ["text"] and function["output_field_names"] == ["sparse_vector"]
    assert function["type"] == 1
    assert payload["consistency_level"] == "Strong" and payload["timeout"] > 0
    params = [item.to_dict() for item in payload["index_params"]]
    assert any(item["field_name"] == "sparse_vector" and item["metric_type"] == "BM25" for item in params)


@pytest.mark.parametrize("fault", ["owner", "analyzer", "function", "sparse"])
async def test_incompatible_hybrid_resources_fail_without_create_or_drop(fault: str) -> None:
    """同名异构和外来集合不能被自动重建接管。参数：故障类型；返回：None；异常：断言失败。"""
    client = HybridClient()
    if fault == "owner":
        client.description["description"] = "foreign-owner"
    elif fault == "analyzer":
        client.description["fields"][-1]["params"]["analyzer_params"] = '{"type":"standard"}'
    elif fault == "function":
        client.description["functions"] = []
    else:
        client.description["fields"][2]["type"] = 101
    index = _index(client)
    try:
        with pytest.raises(VectorIndexUnavailable):
            await index.ensure_ready(create=True)
        assert not any(name == "create" for name, _ in client.calls)
        assert client.closed
    finally:
        await index.aclose()


async def test_dense_and_bm25_routes_select_their_own_vector_field() -> None:
    """两个向量字段存在时必须明确选择 anns_field，BM25 只传文本。参数：无；返回：None；异常：断言失败。"""
    client = HybridClient()
    index = _index(client)
    try:
        await index.ensure_ready()
        await index.search([1.0] + [0.0] * 1023, 50)
        await index.search_bm25("MH-LP50 不支持 App", 50)
        payloads = [payload for name, payload in client.calls if name == "search"]
        assert [payload["anns_field"] for payload in payloads] == ["vector", "sparse_vector"]
        assert payloads[1]["data"] == ["MH-LP50 不支持 App"]
        assert all(payload["limit"] == 50 and payload["timeout"] > 0 for payload in payloads)
    finally:
        await index.aclose()


async def test_hybrid_upsert_requires_nonempty_authoritative_text() -> None:
    """新 BM25 索引不能接受没有正文的旧格式记录。参数：无；返回：None；异常：断言失败。"""
    client = HybridClient()
    index = _index(client)
    try:
        await index.ensure_ready()
        record = VectorRecord(id=1, vector=[1.0] + [0.0] * 1023, content_hash="a" * 64, embedding_revision="rev")
        with pytest.raises(VectorIndexUnavailable, match="正文"):
            await index.upsert([record])
        assert not any(name == "upsert" for name, _ in client.calls)
    finally:
        await index.aclose()


async def test_synchronizer_writes_the_same_text_used_for_dense_embedding() -> None:
    """同步丢掉 text 会令 BM25 无法入库，必须与 Dense 三格输入完全相同。参数：无；返回：None；异常：断言失败。"""
    text = KnowledgeText(category="售后", questions="退款条件", answer="必须保留包装。")
    row = ChunkSnapshot(id=1, source_key="doc:a.md", ordinal=0, text=text, content_type="policy",
                        content_hash=content_fingerprint(text), embedding_revision="fake-encoder-v1")
    store, encoder, index = MemoryKnowledgeStore(snapshots=[row]), FakeEncoder(), MemoryVectorIndex()
    async with store.writer() as writer:
        await VectorSynchronizer(store, encoder, index).run(writer)
    assert index.records[1].text == "分类：售后\n问法：退款条件\n答案：必须保留包装。"
    assert encoder.inputs == [index.records[1].text]


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"dense_limit": True}, {"sparse_limit": -1}])
async def test_invalid_hybrid_limits_are_rejected_before_rpc(kwargs: dict) -> None:
    """非法召回数量不能被默认值掩盖。参数：限额；返回：None；异常：断言失败。"""
    index = _index(HybridClient())
    with pytest.raises(ValueError):
        await index.hybrid_search([1.0] + [0.0] * 1023, "退款如何申请", **kwargs)
