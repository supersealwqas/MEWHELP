"""真实 Milvus 2.4.10 集合的写入、检索、分页与所有权测试。"""

from __future__ import annotations

import uuid

import pytest

from app.knowledge.schemas import VectorRecord
from app.knowledge.vector import MilvusVectorIndex, VectorIndexUnavailable

pytestmark = pytest.mark.knowledge_integration


def _vector(first: float, second: float = 0.0) -> list[float]:
    """构造 1024 维测试向量。参数：前两维；返回：向量；异常：无。"""
    return [first, second] + [0.0] * 1022


async def test_upsert_search_iter_and_delete(milvus_index) -> None:
    """真实集合完成写入、COSINE 检索、主键分页与删除。

    参数:
        milvus_index: 真实索引夹具。
    返回值:
        None
    异常:
        无
    """
    await milvus_index.upsert(
        [
            VectorRecord(
                id=7101,
                vector=_vector(1.0),
                content_hash="a" * 64,
                embedding_revision="milvus-test-v1",
            ),
            VectorRecord(
                id=7102,
                vector=_vector(0.8, 0.6),
                content_hash="b" * 64,
                embedding_revision="milvus-test-v1",
            ),
        ]
    )
    hits = await milvus_index.search(_vector(1.0), 2)
    assert hits[0].id == 7101
    assert hits[0].score > hits[1].score
    ids = []
    async for batch in milvus_index.iter_ids(100):
        ids.extend(batch)
    assert sorted(ids) == [7101, 7102]
    await milvus_index.delete([7101, 7102])
    assert await milvus_index.search(_vector(1.0), 2) == []


async def test_same_primary_key_overwrites_fact(milvus_index) -> None:
    """同一主键覆盖后检索只反映当前向量。

    参数:
        milvus_index: 真实索引夹具。
    返回值:
        None
    异常:
        无
    """
    await milvus_index.upsert(
        [
            VectorRecord(
                id=7201,
                vector=_vector(1.0),
                content_hash="a" * 64,
                embedding_revision="v1",
            ),
            VectorRecord(
                id=7202,
                vector=_vector(0.8, 0.6),
                content_hash="b" * 64,
                embedding_revision="v1",
            ),
        ]
    )
    await milvus_index.upsert(
        [
            VectorRecord(
                id=7201,
                vector=_vector(0.0, 0.9),
                content_hash="c" * 64,
                embedding_revision="v1",
            )
        ]
    )
    hits = await milvus_index.search(_vector(1.0), 2)
    assert hits[0].id == 7202
    ids = []
    async for batch in milvus_index.iter_ids(100):
        ids.extend(batch)
    assert sorted(ids) == [7201, 7202]


async def test_owner_mismatch_never_drops_foreign_collection(
    milvus_test_config,
) -> None:
    """归属不匹配的集合被拒绝且不被删除。

    参数:
        milvus_test_config: 独立 Milvus 测试配置。
    返回值:
        None
    异常:
        无
    """
    from pymilvus import MilvusClient

    name = f"foreign_{uuid.uuid4().hex[:12]}"
    client = MilvusClient(
        uri=milvus_test_config.milvus_uri,
        db_name=milvus_test_config.milvus_database,
        token=milvus_test_config.milvus_token.get_secret_value(),
        timeout=10.0,
    )
    try:
        client.create_collection(name, dimension=1024)
        index = MilvusVectorIndex(
            milvus_test_config.model_copy(update={"milvus_collection": name})
        )
        with pytest.raises(VectorIndexUnavailable, match="归属"):
            await index.ensure_ready(create=True)
        assert client.has_collection(name) is True
    finally:
        if client.has_collection(name):
            client.drop_collection(name)
        client.close()
