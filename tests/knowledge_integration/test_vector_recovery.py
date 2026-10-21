"""真实 Milvus 上的向量中断恢复与失效清理测试。"""

from __future__ import annotations

import pytest
import asyncio
import json
import threading

from sqlalchemy import text as sql_text

from app.knowledge.schemas import ChunkDraft, ChunkSnapshot, KnowledgeText, Publication, RetrievalPolicy, VectorRecord
from app.knowledge.search import KnowledgeSearch, KnowledgeSearchUnavailable
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import MilvusVectorIndex, VectorIndexUnavailable
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


@pytest.mark.parametrize("operation", ["delete", "upsert"])
async def test_late_rpc_real_databases_recover_current_facts(
    knowledge_store, milvus_index, milvus_test_config, monkeypatch, operation
) -> None:
    """真实双库验证旧请求晚完成后的状态、指纹和查询恢复。

    参数：专用 MySQL/Milvus 夹具、补丁和旧操作类型。返回：None。
    异常：断言失败、实际双库错误；测试只操作 mewhelp_ch03_test。
    """
    from app.knowledge import vector as vector_module

    async def publish(answer):
        """同来源同主键发布新事实。参数：答案；返回：快照；异常：存储异常。"""
        async with knowledge_store.writer() as writer:
            frame = await writer.frame()
            rows = await writer.publish(Publication(
                expected_token=frame.token, embedding_revision=REVISION,
                drafts=[ChunkDraft(source_key="doc:late-rpc.md", ordinal=0,
                                   text=KnowledgeText(category="退款", questions="退款多久到账？", answer=answer),
                                   content_type="faq")],
            ))
            return rows[0]

    encoder = FakeEncoder()
    sync = VectorSynchronizer(knowledge_store, encoder, milvus_index)
    original = await publish("旧版处理说明。")
    async with knowledge_store.writer() as writer:
        await sync.run(writer)
    done = (await knowledge_store.get_chunks([original.id]))[0]
    async with knowledge_store.writer() as writer:
        if operation == "delete":
            frame = await writer.frame()
            await writer.publish(Publication(expected_token=frame.token, drafts=[],
                                             embedding_revision=REVISION, inactive_ids=[original.id]))
        else:
            assert await writer.mark_pending(done)

    entered, finish, finished = threading.Event(), threading.Event(), threading.Event()
    base_client = milvus_index._client
    late_errors = []

    class DelayedClient:
        """在真实 SDK 变更边界延迟旧调用，读取仍直达 Milvus。"""

        def __getattr__(self, name):
            """转发真实客户端。参数：成员名；返回：成员；异常：AttributeError。"""
            return getattr(base_client, name)

        def _mutate(self, name, *args, **kwargs):
            """等新发布完成后执行真实变更。参数：SDK 参数；返回：SDK 结果；异常：实际错误。"""
            if name != operation:
                return getattr(base_client, name)(*args, **kwargs)
            entered.set()
            if not finish.wait(10):
                raise RuntimeError("测试未释放旧请求")
            try:
                return getattr(base_client, name)(*args, **kwargs)
            except Exception as error:
                late_errors.append(error)
                raise
            finally:
                finished.set()

        def delete(self, *args, **kwargs):
            """执行旧删除。参数：SDK 参数；返回：SDK 结果；异常：实际错误。"""
            return self._mutate("delete", *args, **kwargs)

        def upsert(self, *args, **kwargs):
            """执行旧写入。参数：SDK 参数；返回：SDK 结果；异常：实际错误。"""
            return self._mutate("upsert", *args, **kwargs)

    class BoundedOldIndex(MilvusVectorIndex):
        """仅把旧变更的本地等待缩短，真实分页仍使用正常超时。"""

        async def _bounded_mutation(self, name, values):
            """缩短选中请求等待。参数：操作、参数；返回：None；异常：索引错误。"""
            with monkeypatch.context() as scoped:
                scoped.setattr(vector_module, "_RPC_TIMEOUT_SECONDS", 0.5)
                await getattr(super(), name)(values)

        async def upsert(self, records):
            """发送有界旧写入。参数：向量记录；返回：None；异常：索引错误。"""
            await self._bounded_mutation("upsert", records)

        async def delete(self, ids):
            """发送有界旧删除。参数：主键；返回：None；异常：索引错误。"""
            await self._bounded_mutation("delete", ids)

    old_index = BoundedOldIndex(milvus_test_config)
    old_index._client, old_index._ready = DelayedClient(), True
    try:
        async with knowledge_store.writer() as writer:
            old_task = asyncio.create_task(VectorSynchronizer(knowledge_store, encoder, old_index).run(writer))
            assert await asyncio.to_thread(entered.wait, 3), "旧请求必须确实已发出"
            # 真实释放旧连接的命名锁；旧 RPC 尚未结束，模拟跨库未知结果。
            await writer._lease.connection.execute(
                sql_text("SELECT RELEASE_LOCK(:name)"),
                {"name": writer._lease._lock_name},
            )
            await writer._lease.connection.commit()
            with pytest.raises(VectorIndexUnavailable):
                await asyncio.wait_for(old_task, 2)
        before_new = (await knowledge_store.get_chunks([original.id]))[0]
        assert before_new.vectorize_status == "pending"
        assert before_new.is_active is (operation == "upsert")
        updated = await publish("新版说明：退款审核通常为1至3个工作日，不代表到账时间。")
        assert updated.id == original.id
        async with knowledge_store.writer() as writer:
            await sync.run(writer)
        current = (await knowledge_store.get_chunks([updated.id]))[0]
        assert current.is_active and current.vectorize_status == "done"
        assert await milvus_index.get_fingerprints([current.id]) == {
            current.id: (current.content_hash, current.embedding_revision)}
    finally:
        finish.set()
        if entered.is_set():
            assert await asyncio.to_thread(finished.wait, 3)
        if not old_task.done():
            old_task.cancel()
        await asyncio.gather(old_task, return_exceptions=True)
    assert late_errors == []
    damaged = await milvus_index.get_fingerprints([current.id])
    if operation == "delete":
        assert damaged == {}
    else:
        assert damaged == {original.id: (original.content_hash, original.embedding_revision)}

    search = KnowledgeSearch(knowledge_store, encoder, milvus_index, RetrievalPolicy(
        embedding_revision=REVISION, chunking_version="test", min_score=0.5,
        calibration_digest="test",
    ))
    if operation == "delete":
        assert await search.query_faq("退款多久到账？") == []
    else:
        with pytest.raises(KnowledgeSearchUnavailable):
            await search.query_faq("退款多久到账？")
    async with knowledge_store.writer() as writer:
        report = await sync.run(writer)
    restored = (await knowledge_store.get_chunks([current.id]))[0]
    fingerprints = await milvus_index.get_fingerprints([current.id])
    hits = await search.query_faq("退款多久到账？")
    assert report.marked_done == 1
    assert restored.is_active and restored.vectorize_status == "done"
    assert restored.vector_id == str(current.id)
    assert fingerprints == {current.id: (current.content_hash, current.embedding_revision)}
    assert [hit["answer"] for hit in hits] == [updated.text.answer]
    print(json.dumps({"operation": operation, "id": current.id,
                      "old_content_hash": original.content_hash,
                      "current_content_hash": current.content_hash,
                      "embedding_revision": current.embedding_revision,
                      "before_new": {"active": before_new.is_active, "status": before_new.vectorize_status},
                      "damaged_fingerprints": damaged,
                      "recovered": {"active": restored.is_active, "status": restored.vectorize_status,
                                    "vector_id": restored.vector_id, "fingerprints": fingerprints},
                      "hits": hits}, ensure_ascii=True))
