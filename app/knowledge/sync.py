"""知识原文到 Milvus 的可恢复 Dense 向量同步。

先分页对照 active/done 的同步指纹，缺失或失配时条件恢复 pending；再编码
active/pending，以同一主键全量 upsert 并条件回填 done，最后清理失效向量。checkpoint
只用于测试/显式恢复演示，不对外提供 HTTP 故障入口。
"""

from __future__ import annotations

from typing import Callable

from app.knowledge.contracts import KnowledgeEncoder, VectorIndex
from app.knowledge.limits import knowledge_overflow, valid_dense_vector
from app.knowledge.schemas import (
    SyncReport,
    VectorRecord,
)
from app.knowledge.store import KnowledgeStore, KnowledgeWriter
from app.knowledge.text import format_knowledge
from app.knowledge.vector import VectorIndexUnavailable


class VectorSynchronizer:
    """驱动一次可恢复的向量补齐与清理。"""

    def __init__(
        self,
        store: KnowledgeStore,
        encoder: KnowledgeEncoder,
        index: VectorIndex,
    ) -> None:
        """保存同步依赖。

        参数:
            store: 知识原文存储。
            encoder: BGE-M3 编码器。
            index: Milvus 索引 Adapter。
        返回值:
            None
        异常:
            无
        """
        self.store = store
        self.encoder = encoder
        self.index = index

    async def run(
        self,
        writer: KnowledgeWriter,
        *,
        batch_size: int = 16,
        checkpoint: Callable[[str], None] | None = None,
    ) -> SyncReport:
        """执行 done 对照修复、pending 补齐、条件回填与失效向量清理。

        参数:
            writer: 持有写锁的写入器。
            batch_size: 主键分页大小。
            checkpoint: 仅测试/演示使用的故障注入点。
        返回值:
            SyncReport: 扫描、写入、回填、删除与剩余 pending 数量。
        异常:
            VectorIndexUnavailable、EmbeddingUnavailable、存储异常。
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        await writer.assert_held()
        # 旧 RPC 可能在新任务发布之后才落地；每轮都以 MySQL 对照修复 done。
        await self._reconcile_done(writer, batch_size)
        scanned = upserted = marked_done = deleted_vectors = 0
        after_id = 0
        while True:
            await writer.assert_held()
            page = await self.store.pending_page(after_id, batch_size)
            if not page:
                break
            if checkpoint:
                checkpoint("before_encode")
            await writer.assert_held()
            if any(knowledge_overflow(row.text, self.encoder.count_tokens) for row in page):
                raise ValueError("知识三格或格式化输入超过硬限")
            texts = [format_knowledge(snapshot.text) for snapshot in page]
            vectors = await self.encoder.encode(texts)
            await writer.assert_held()
            if len(vectors) != len(page) or not all(
                valid_dense_vector(vector) for vector in vectors
            ):
                raise VectorIndexUnavailable("编码器返回向量数量、维度或数值不合法")
            records = [
                VectorRecord(
                    id=snapshot.id,
                    vector=vector,
                    content_hash=snapshot.content_hash,
                    embedding_revision=snapshot.embedding_revision,
                )
                for snapshot, vector in zip(page, vectors)
            ]
            if checkpoint:
                checkpoint("before_upsert")
            await writer.assert_held()
            await self.index.upsert(records)
            await writer.assert_held()
            scanned += len(page)
            upserted += len(records)
            if checkpoint:
                checkpoint("after_upsert")
            for snapshot in page:
                await writer.assert_held()
                if await writer.mark_vectorized(snapshot):
                    marked_done += 1
            if checkpoint:
                checkpoint("after_mark")
            after_id = page[-1].id

        async for ids in self.index.iter_ids(batch_size):
            await writer.assert_held()
            active = await self.store.active_ids(ids)
            stale = [identifier for identifier in ids if identifier not in active]
            if stale:
                await writer.assert_held()
                await self.index.delete(stale)
                await writer.assert_held()
                deleted_vectors += len(stale)

        await writer.assert_held()
        remaining_pending = await self._remaining_pending(batch_size)
        return SyncReport(
            scanned=scanned,
            upserted=upserted,
            marked_done=marked_done,
            deleted_vectors=deleted_vectors,
            remaining_pending=remaining_pending,
        )

    async def _reconcile_done(self, writer: KnowledgeWriter, batch_size: int) -> None:
        """分页对照 active/done 同步指纹并条件恢复 pending。

        参数：writer 为持锁写入器，batch_size 为正页大小。返回：None。
        异常：LeaseLost、索引或存储异常；未知回读结果绝不当成缺向量。
        """
        after_id = 0
        while True:
            await writer.assert_held()
            page = await self.store.active_page(after_id, batch_size)
            if not page:
                return
            done = [row for row in page if row.vectorize_status == "done"]
            if done:
                fingerprints = await self.index.get_fingerprints([row.id for row in done])
                await writer.assert_held()
                for row in done:
                    if (
                        row.vector_id != str(row.id)
                        or fingerprints.get(row.id)
                        != (row.content_hash, row.embedding_revision)
                    ):
                        await writer.mark_pending(row)
            after_id = page[-1].id

    async def _remaining_pending(self, batch_size: int) -> int:
        """统计 active/pending 行数。参数：页大小；返回：数量；异常：存储异常。"""
        count = 0
        after_id = 0
        while True:
            page = await self.store.pending_page(after_id, batch_size)
            if not page:
                break
            count += len(page)
            after_id = page[-1].id
        return count
