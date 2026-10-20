"""真实 Milvus 上的向量中断恢复与失效清理测试。"""

from __future__ import annotations

import pytest

from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, VectorRecord
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore

pytestmark = pytest.mark.knowledge_integration

REVISION = "fake-encoder-v1"


def _snapshot(identifier: int, *, active: bool = True) -> ChunkSnapshot:
    """构造测试快照。参数：主键与发布资格；返回：快照；异常：校验失败。"""
    text = KnowledgeText(category="分类", questions="问法", answer=f"恢复答案{identifier}。")
    return ChunkSnapshot(
        id=identifier,
        source_key=f"doc:recovery-{identifier}.md",
        ordinal=0,
        text=text,
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
        content_hash=content_fingerprint(text),
        embedding_revision=REVISION,
        is_active=active,
        vectorize_status="pending",
    )


async def test_crash_after_upsert_then_rerun(milvus_index) -> None:
    """真实向量写入后中断，重跑回填 done 并可检索。

    参数:
        milvus_index: 真实索引夹具。
    返回值:
        None
    异常:
        无
    """
    snapshot = _snapshot(8101)
    store = MemoryKnowledgeStore(snapshots=[snapshot])
    synchronizer = VectorSynchronizer(store, FakeEncoder(), milvus_index)

    def checkpoint(label: str) -> None:
        """在真实向量写入后模拟中断。参数：阶段；返回：None；异常：RuntimeError。"""
        if label == "after_upsert":
            raise RuntimeError("模拟真实向量写入后中断")

    async with store.writer() as writer:
        with pytest.raises(RuntimeError):
            await synchronizer.run(writer, checkpoint=checkpoint)
    assert (await store.get_chunks([snapshot.id]))[0].vectorize_status == "pending"
    hits = await milvus_index.search([1.0] + [0.0] * 1023, 1)
    assert [hit.id for hit in hits] == [snapshot.id]

    async with store.writer() as writer:
        report = await synchronizer.run(writer)
    assert report.marked_done == 1
    assert (await store.get_chunks([snapshot.id]))[0].vectorize_status == "done"
    assert [hit.id for hit in await milvus_index.search([1.0] + [0.0] * 1023, 1)] == [
        snapshot.id
    ]


async def test_cleanup_inactive_vector_on_real_milvus(milvus_index) -> None:
    """vector_id 为空的失效向量也能从真实集合删除。

    参数:
        milvus_index: 真实索引夹具。
    返回值:
        None
    异常:
        无
    """
    snapshot = _snapshot(8102, active=False)
    store = MemoryKnowledgeStore(snapshots=[snapshot])
    await milvus_index.upsert(
        [
            VectorRecord(
                id=snapshot.id,
                vector=[1.0] + [0.0] * 1023,
                content_hash=snapshot.content_hash,
                embedding_revision=snapshot.embedding_revision,
            )
        ]
    )
    synchronizer = VectorSynchronizer(store, FakeEncoder(), milvus_index)
    async with store.writer() as writer:
        report = await synchronizer.run(writer)
    assert report.deleted_vectors == 1
    ids = []
    async for batch in milvus_index.iter_ids(100):
        ids.extend(batch)
    assert snapshot.id not in ids
