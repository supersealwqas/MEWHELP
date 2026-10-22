"""向量同步的可恢复性与失效清理离线测试。"""

from __future__ import annotations

import pytest
import asyncio

from app.knowledge.errors import LeaseLost
from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, VectorRecord
from app.knowledge.store import KnowledgeWriter
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint, format_knowledge
from app.knowledge.vector import VectorIndexUnavailable
from tests.knowledge.fakes import (
    FakeEncoder,
    MemoryKnowledgeStore,
    MemoryVectorIndex,
    MemoryKnowledgeWriter,
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


class _Lease:
    """可切换失锁状态的租约边界替身。"""

    def __init__(self) -> None:
        """初始化持锁状态。参数：无；返回：None；异常：无。"""
        self.lost = False
        self.connection = self

    def in_transaction(self) -> bool:
        """模拟没有业务事务。参数：无；返回：False；异常：无。"""
        return False

    async def assert_held(self) -> None:
        """核验租约。参数：无；返回：None；异常：LeaseLost。"""
        if self.lost:
            raise LeaseLost("模拟写锁丢失")


class _Writer(MemoryKnowledgeWriter):
    """带独立租约的内存写入器，仅替代外部 MySQL。"""

    def __init__(self, store, lease) -> None:
        """保存存储与租约。参数：存储、租约；返回：None；异常：无。"""
        super().__init__(store)
        self.lease = lease

    async def assert_held(self) -> None:
        """核验测试租约。参数：无；返回：None；异常：LeaseLost。"""
        await self.lease.assert_held()


def _record(snapshot: ChunkSnapshot) -> VectorRecord:
    """构造完整索引记录。参数：快照；返回：向量记录；异常：校验失败。"""
    return VectorRecord(id=snapshot.id, vector=[1.0] + [0.0] * 1023,
                        text=format_knowledge(snapshot.text),
                        content_hash=snapshot.content_hash,
                        embedding_revision=snapshot.embedding_revision)


async def test_lost_lease_with_zero_pending_cannot_delete() -> None:
    """零 pending 也必须先核锁，不能删除索引。参数：无；返回：None；异常：断言失败。"""
    snapshot = _snapshot(active=False)
    store, index, lease = MemoryKnowledgeStore(snapshots=[snapshot]), MemoryVectorIndex(), _Lease()
    index.records[snapshot.id] = _record(snapshot)
    lease.lost = True
    writer = KnowledgeWriter(None, lease)
    with pytest.raises(LeaseLost):
        await VectorSynchronizer(store, FakeEncoder(), index).run(writer)
    assert index.ids == {snapshot.id}


async def test_lease_lost_during_encode_prevents_upsert() -> None:
    """编码期间失锁时旧向量不能写入。参数：无；返回：None；异常：断言失败。"""
    snapshot, lease = _snapshot(), _Lease()
    store, index = MemoryKnowledgeStore(snapshots=[snapshot]), MemoryVectorIndex()

    class LosingEncoder(FakeEncoder):
        """返回编码结果前模拟失锁。"""

        async def encode(self, texts):
            """编码并失锁。参数：文本；返回：向量；异常：无。"""
            vectors = await super().encode(texts)
            lease.lost = True
            return vectors

    with pytest.raises(LeaseLost):
        await VectorSynchronizer(store, LosingEncoder(), index).run(KnowledgeWriter(None, lease))
    assert index.ids == set()
    assert store._snapshots[1].vectorize_status == "pending"


@pytest.mark.parametrize("damage", ["missing", "content", "revision"])
async def test_done_damage_is_repaired_and_normal_rerun_does_not_encode(damage) -> None:
    """done 缺失或指纹失配会重同步，正常重跑不编码。参数：损坏类型；返回：None；异常：断言失败。"""
    snapshot = _snapshot(status="done", vector_id="1")
    store, index, encoder = MemoryKnowledgeStore(snapshots=[snapshot]), MemoryVectorIndex(), FakeEncoder()
    if damage != "missing":
        field = "content_hash" if damage == "content" else "embedding_revision"
        index.records[1] = _record(snapshot).model_copy(update={field: "stale"})
    synchronizer = VectorSynchronizer(store, encoder, index)
    async with store.writer() as writer:
        report = await synchronizer.run(writer, batch_size=1)
    assert report.upserted == report.marked_done == 1
    assert index.records[1] == _record(snapshot)
    assert store._snapshots[1].vectorize_status == "done"
    encoder.inputs.clear()
    async with store.writer() as writer:
        await synchronizer.run(writer)
    assert encoder.inputs == []


@pytest.mark.parametrize("damage", ["characters", "tokens", "dimension", "nonfinite"])
async def test_invalid_sync_input_cannot_be_published(damage) -> None:
    """共享硬限在同步边界拒绝不合格文本/向量。参数：违规类型；返回：None；异常：断言失败。"""
    snapshot = _snapshot()
    if damage == "characters":
        snapshot = snapshot.model_copy(update={"text": KnowledgeText(
            category="分", questions="问", answer="字" * 1599)})

    class InvalidEncoder(FakeEncoder):
        """返回指定违规 token 数或向量。"""

        def count_tokens(self, text):
            """统计长度。参数：文本；返回：token 数；异常：无。"""
            return 2049 if damage == "tokens" else len(text)

        async def encode(self, texts):
            """编码违规向量。参数：文本；返回：向量；异常：无。"""
            if damage == "dimension":
                return [[1.0]]
            if damage == "nonfinite":
                return [[float("nan")] * 1024]
            return await super().encode(texts)

    store, index = MemoryKnowledgeStore(snapshots=[snapshot]), MemoryVectorIndex()
    async with store.writer() as writer:
        with pytest.raises((ValueError, VectorIndexUnavailable)):
            await VectorSynchronizer(store, InvalidEncoder(), index).run(writer)
    assert index.ids == set()
    assert store._snapshots[1].vectorize_status == "pending"


@pytest.mark.parametrize("operation", ["delete", "upsert"])
async def test_late_old_rpc_is_repaired_by_next_public_sync(operation) -> None:
    """旧请求晚完成不能回填，新 sync 补回缺失/旧指纹。参数：操作；返回：None；异常：断言失败。"""
    snapshot = _snapshot(active=operation == "upsert")
    store, lease = MemoryKnowledgeStore(snapshots=[snapshot]), _Lease()
    entered, finish = asyncio.Event(), asyncio.Event()

    class DelayedIndex(MemoryVectorIndex):
        """仅第一次选定 RPC 延迟，其余照常执行。"""

        delayed = False

        async def _delay(self, name):
            """等待受控旧请求。参数：操作名；返回：None；异常：取消异常。"""
            if name == operation and not self.delayed:
                self.delayed = True
                entered.set()
                await finish.wait()

        async def upsert(self, records):
            """延迟后写入。参数：记录；返回：None；异常：索引异常。"""
            await self._delay("upsert")
            await super().upsert(records)

        async def delete(self, ids):
            """延迟后删除。参数：主键；返回：None；异常：索引异常。"""
            await self._delay("delete")
            await super().delete(ids)

    index = DelayedIndex()
    index.records[1] = _record(snapshot)
    sync = VectorSynchronizer(store, FakeEncoder(), index)
    old_task = asyncio.create_task(sync.run(_Writer(store, lease)))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        lease.lost = True
        updated = _snapshot(content_hash="b" * 64)
        store._snapshots[1] = updated
        async with store.writer() as writer:
            await sync.run(writer)
        assert store._snapshots[1].vectorize_status == "done"
    finally:
        finish.set()
    with pytest.raises(LeaseLost):
        await old_task
    assert 1 not in index.ids or index.records[1].content_hash != updated.content_hash
    async with store.writer() as writer:
        await sync.run(writer)
    assert index.records[1] == _record(updated)
    assert store._snapshots[1].vector_id == "1"


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
