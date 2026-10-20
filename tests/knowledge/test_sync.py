"""向量同步的可恢复性与失效清理离线测试。"""

from __future__ import annotations

import pytest

from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, VectorRecord
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import VectorIndexUnavailable
from tests.knowledge.fakes import (
    FakeEncoder,
    MemoryKnowledgeStore,
    MemoryVectorIndex,
)

REVISION = "fake-encoder-v1"


def _snapshot(
    identifier: int = 1,
    *,
    active: bool = True,
    status: str = "pending",
    vector_id: str | None = None,
    content_hash: str | None = None,
) -> ChunkSnapshot:
    """构造测试快照。参数：主键、发布/同步状态与指纹；返回：快照；异常：校验失败。"""
    text = KnowledgeText(category="分类", questions="问法", answer=f"答案{identifier}。")
    return ChunkSnapshot(
        id=identifier,
        source_key=f"doc:{identifier}.md",
        ordinal=0,
        text=text,
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
        content_hash=content_hash or content_fingerprint(text),
        embedding_revision=REVISION,
        is_active=active,
        vector_id=vector_id,
        vectorize_status=status,
    )


async def test_ack_lost_reuses_primary_key() -> None:
    """写入确认丢失后保持 pending，重跑复用同一主键。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    snapshot = _snapshot()
    store = MemoryKnowledgeStore(snapshots=[snapshot])
    index = MemoryVectorIndex(fail_after_upsert=True)
    synchronizer = VectorSynchronizer(store, FakeEncoder(), index)
    async with store.writer() as writer:
        with pytest.raises(VectorIndexUnavailable):
            await synchronizer.run(writer)
    assert index.ids == {snapshot.id}
    assert (await store.get_chunks([snapshot.id]))[0].vectorize_status == "pending"

    index.fail_after_upsert = False
    async with store.writer() as writer:
        report = await synchronizer.run(writer)
    assert report.upserted == 1
    assert report.marked_done == 1
    assert report.remaining_pending == 0
    assert index.ids == {snapshot.id}
    assert (await store.get_chunks([snapshot.id]))[0].vectorize_status == "done"


async def test_crash_after_vector_before_mark_done() -> None:
    """向量写入后中断，重跑能回填 done。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    snapshot = _snapshot()
    store = MemoryKnowledgeStore(snapshots=[snapshot])
    index = MemoryVectorIndex()
    synchronizer = VectorSynchronizer(store, FakeEncoder(), index)

    def checkpoint(label: str) -> None:
        """在向量写入后模拟进程中断。参数：阶段；返回：None；异常：RuntimeError。"""
        if label == "after_upsert":
            raise RuntimeError("模拟进程中断")

    async with store.writer() as writer:
        with pytest.raises(RuntimeError):
            await synchronizer.run(writer, checkpoint=checkpoint)
    assert index.ids == {snapshot.id}
    assert (await store.get_chunks([snapshot.id]))[0].vectorize_status == "pending"

    async with store.writer() as writer:
        report = await synchronizer.run(writer)
    assert report.marked_done == 1
    assert (await store.get_chunks([snapshot.id]))[0].vectorize_status == "done"


async def test_changed_or_inactive_snapshot_not_marked_done() -> None:
    """编码后原文变化时条件回填失败，保留 pending。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    snapshot = _snapshot()
    store = MemoryKnowledgeStore(snapshots=[snapshot])
    index = MemoryVectorIndex()
    synchronizer = VectorSynchronizer(store, FakeEncoder(), index)

    def checkpoint(label: str) -> None:
        """在写入后改变当前内容指纹。参数：阶段；返回：None；异常：无。"""
        if label == "after_upsert":
            current = store._snapshots[snapshot.id]
            store._snapshots[snapshot.id] = current.model_copy(
                update={"content_hash": "0" * 64}
            )

    async with store.writer() as writer:
        report = await synchronizer.run(writer, checkpoint=checkpoint)
    assert report.marked_done == 0
    assert report.remaining_pending == 1
    assert (await store.get_chunks([snapshot.id]))[0].vectorize_status == "pending"


async def test_cleanup_does_not_need_saved_vector_id() -> None:
    """vector_id 已清空的 inactive 向量仍能按主键清理。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    snapshot = _snapshot(identifier=2, active=False, vector_id=None)
    store = MemoryKnowledgeStore(snapshots=[snapshot])
    index = MemoryVectorIndex()
    index.records[snapshot.id] = VectorRecord(
        id=snapshot.id,
        vector=[1.0] + [0.0] * 1023,
        content_hash=snapshot.content_hash,
        embedding_revision=snapshot.embedding_revision,
    )
    synchronizer = VectorSynchronizer(store, FakeEncoder(), index)
    async with store.writer() as writer:
        report = await synchronizer.run(writer)
    assert report.deleted_vectors == 1
    assert index.ids == set()
    assert report.remaining_pending == 0
