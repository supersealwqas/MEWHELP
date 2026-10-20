"""知识原文到 Milvus 的可恢复 Dense 向量同步。

按主键分页扫描 active/pending，编码后以同一主键全量 upsert，再条件回填
done；随后分页对照 Milvus 主键与 MySQL 发布资格，清理失效向量。checkpoint
只用于测试/显式恢复演示，不对外提供 HTTP 故障入口。
"""

from __future__ import annotations

from typing import Callable

from app.knowledge.schemas import (
    SyncReport,
    VectorRecord,
)
from app.knowledge.store import KnowledgeStore, KnowledgeWriter
from app.knowledge.text import format_knowledge
from app.knowledge.vector import MilvusVectorIndex, VectorIndexUnavailable


class VectorSynchronizer:
    """驱动一次可恢复的向量补齐与清理。"""

    def __init__(
        self,
        store: KnowledgeStore,
        encoder,
        index: MilvusVectorIndex,
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
        """执行 pending 补齐、条件回填与失效向量清理。

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
        scanned = upserted = marked_done = deleted_vectors = 0
        after_id = 0
        while True:
            page = await self.store.pending_page(after_id, batch_size)
            if not page:
                break
            if checkpoint:
                checkpoint("before_encode")
            texts = [format_knowledge(snapshot.text) for snapshot in page]
            vectors = await self.encoder.encode(texts)
            if len(vectors) != len(page):
                raise VectorIndexUnavailable("编码器返回向量数量不一致")
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
            await self.index.upsert(records)
            scanned += len(page)
            upserted += len(records)
            if checkpoint:
                checkpoint("after_upsert")
            for snapshot in page:
                if await writer.mark_vectorized(snapshot):
                    marked_done += 1
            if checkpoint:
                checkpoint("after_mark")
            after_id = page[-1].id

        async for ids in self.index.iter_ids(batch_size):
            active = await self.store.active_ids(ids)
            stale = [identifier for identifier in ids if identifier not in active]
            if stale:
                await self.index.delete(stale)
                deleted_vectors += len(stale)

        remaining_pending = await self._remaining_pending(batch_size)
        return SyncReport(
            scanned=scanned,
            upserted=upserted,
            marked_done=marked_done,
            deleted_vectors=deleted_vectors,
            remaining_pending=remaining_pending,
        )

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
