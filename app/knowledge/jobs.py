"""知识库导入、历史对话挖掘与恢复任务编排。

任务在同一写锁内先恢复 pending 批次，再按会话边界领取新来源；模型抽取后
写入暂存，统一执行生命周期去重/隔离，最后同步向量。失败不推进来源，保留
可恢复批次；本模块不启动 HTTP 长任务，也不直接访问 Milvus 业务数据。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.knowledge.chunking import chunk_markdown
from app.knowledge.llm import KnowledgeModel
from app.knowledge.manifest import load_manifest
from app.knowledge.prompts import EXTRACTION_PROMPT_VERSION
from app.knowledge.schemas import (
    ChunkSnapshot,
    ExtractionBatch,
    JobReport,
    SyncReport,
)
from app.knowledge.store import KnowledgeStore
from app.knowledge.sync import VectorSynchronizer

_MAX_CONVERSATIONS_PER_BATCH = 20
_TURN_SCAN_LIMIT = 200


def _turns_digest(turns) -> str:
    """计算批次的稳定输入指纹。参数：轮次；返回：SHA256；异常：无。"""
    payload = [turn.model_dump(mode="json") for turn in turns]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _batch_from_turns(turns, upper_id: int) -> ExtractionBatch:
    """由同一会话组构造稳定批次。参数：轮次与上界；返回：批次；异常：校验失败。"""
    ordered = sorted(turns, key=lambda turn: turn.message_id)
    digest = _turns_digest(ordered)
    return ExtractionBatch(
        batch_no=f"batch-{digest[:16]}",
        source_ids=[turn.message_id for turn in ordered],
        snapshot_upper_id=upper_id,
        input_hash=digest,
        prompt_version=EXTRACTION_PROMPT_VERSION,
        turns=ordered,
    )


class KnowledgeJobs:
    """一次导入、挖掘或恢复任务的编排器。"""

    def __init__(
        self,
        store: KnowledgeStore,
        encoder,
        index,
        model: KnowledgeModel,
        lifecycle,
        synchronizer: VectorSynchronizer,
    ) -> None:
        """保存任务依赖。

        参数:
            store: 知识库存储。
            encoder: BGE-M3 编码器。
            index: Milvus 索引。
            model: 抽取/关系模型。
            lifecycle: 文档权威与去重生命周期。
            synchronizer: 向量同步器。
        返回值:
            None
        异常:
            无
        """
        self.store = store
        self.encoder = encoder
        self.index = index
        self.model = model
        self.lifecycle = lifecycle
        self.synchronizer = synchronizer

    async def import_documents(self, manifest_path: Path) -> JobReport:
        """导入清单内全部文档并同步向量。参数：清单路径；返回：报告；异常：切分/模型/存储错误。"""
        root = manifest_path.parent
        specs = load_manifest(manifest_path, root=root)
        async with self.store.writer() as writer:
            for spec in specs:
                source = spec.path.read_text(encoding="utf-8")
                drafts = chunk_markdown(source, spec, self.encoder.count_tokens)
                await self.lifecycle.publish_documents(writer, drafts, [spec.source_key])
            sync = await self.synchronizer.run(writer)
        return JobReport(
            batch_count=0,
            extracted=0,
            kept=0,
            discarded=0,
            quarantined=0,
            sync=sync,
        )

    async def _process_pending(self, writer) -> tuple[int, int, list[int]]:
        """恢复 pending 批次并抽取。参数：写入器；返回：批次数/抽取数/暂存 ID；异常：模型/存储错误。"""
        batches = await writer.pending_batches()
        extracted = 0
        staged_ids: list[int] = []
        for batch in batches:
            qas = await self.model.extract(batch)
            staged = await writer.complete_batch(batch.batch_no, qas)
            extracted += len(qas)
            staged_ids.extend(row.id for row in staged)
        return len(batches), extracted, staged_ids

    async def _process_new(self, writer) -> tuple[int, int, list[int]]:
        """领取未分配轮次并按会话边界抽取。参数：写入器；返回：批次数/抽取数/暂存 ID；异常：模型/存储错误。"""
        upper = await writer.source_upper_id()
        turns = await writer.unassigned_turns(upper, 0, _TURN_SCAN_LIMIT)
        grouped: dict[int, list] = {}
        for turn in turns:
            grouped.setdefault(turn.conversation_id, []).append(turn)
        batches = 0
        extracted = 0
        staged_ids: list[int] = []
        conversations = list(grouped.values())
        for start in range(0, len(conversations), _MAX_CONVERSATIONS_PER_BATCH):
            selected = [
                turn
                for group in conversations[start : start + _MAX_CONVERSATIONS_PER_BATCH]
                for turn in group
            ]
            if not selected:
                continue
            batch = _batch_from_turns(selected, upper)
            reserved = await writer.reserve_batch(batch)
            qas = await self.model.extract(reserved)
            staged = await writer.complete_batch(reserved.batch_no, qas)
            batches += 1
            extracted += len(qas)
            staged_ids.extend(row.id for row in staged)
        return batches, extracted, staged_ids

    async def _finish(
        self,
        writer,
        *,
        batch_count: int,
        extracted: int,
        staged_ids: list[int],
    ) -> JobReport:
        """统一去重、隔离与向量同步。参数：写入器与统计；返回：报告；异常：模型/存储错误。"""
        await self.lifecycle.reconcile_staging(writer)
        sync = await self.synchronizer.run(writer)
        frame = await writer.frame()
        wanted = set(staged_ids)
        rows = [row for row in frame.staged if row.id in wanted]
        kept = sum(1 for row in rows if row.status == "kept")
        discarded = sum(1 for row in rows if row.status == "discarded")
        quarantined = sum(
            1 for row in rows if row.conflict_group and row.status == "extracted"
        )
        return JobReport(
            batch_count=batch_count,
            extracted=extracted,
            kept=kept,
            discarded=discarded,
            quarantined=quarantined,
            sync=sync,
        )

    async def mine_once(self) -> JobReport:
        """恢复 pending 并领取一批新来源，随后去重与同步。参数：无；返回：报告；异常：模型/存储错误。"""
        async with self.store.writer() as writer:
            pending_count, pending_extracted, pending_ids = await self._process_pending(writer)
            new_count, new_extracted, new_ids = await self._process_new(writer)
            return await self._finish(
                writer,
                batch_count=pending_count + new_count,
                extracted=pending_extracted + new_extracted,
                staged_ids=pending_ids + new_ids,
            )

    async def recover_pending(self) -> JobReport:
        """只恢复已预留批次，不领取新来源。参数：无；返回：报告；异常：模型/存储错误。"""
        async with self.store.writer() as writer:
            count, extracted, staged_ids = await self._process_pending(writer)
            return await self._finish(
                writer,
                batch_count=count,
                extracted=extracted,
                staged_ids=staged_ids,
            )

    async def sync_pending(self) -> SyncReport:
        """补齐 pending 向量并清理失效向量。参数：无；返回：同步报告；异常：存储/向量错误。"""
        async with self.store.writer() as writer:
            return await self.synchronizer.run(writer)
