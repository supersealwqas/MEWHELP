"""知识库导入、历史对话挖掘与恢复任务编排。

任务在同一写锁内先恢复 pending 批次，再按会话边界领取新来源；模型抽取后
写入暂存，统一执行生命周期去重/隔离，最后同步向量。失败不推进来源，保留
可恢复批次；本模块不启动 HTTP 长任务，也不直接访问 Milvus 业务数据。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.knowledge.chunking import chunk_markdown
from app.knowledge.contracts import KnowledgeModel
from app.knowledge.grounding import turns_digest
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
_BATCH_INPUT_CHAR_BUDGET = 8000


def _turns_digest(turns) -> str:
    """计算批次的稳定输入指纹。参数：轮次；返回：SHA256；异常：无。"""
    return turns_digest(turns)


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


def _estimate_turns(turns) -> int:
    """估算轮次序列的字符量。参数：轮次；返回：字符数；异常：无。"""
    return len(
        json.dumps(
            [turn.model_dump(mode="json") for turn in turns],
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _split_batches(turns) -> list[list]:
    """按会话边界、20 会话和输入预算切分批次。参数：轮次；返回：批次轮次；异常：ValueError。"""
    grouped: dict[int, list] = {}
    for turn in turns:
        grouped.setdefault(turn.conversation_id, []).append(turn)
    batches: list[list] = []
    current: list = []

    def flush() -> None:
        """提交当前批次。参数：无；返回：None；异常：无。"""
        nonlocal current
        if current:
            batches.append(current)
            current = []

    for group in grouped.values():
        current_conversations = {turn.conversation_id for turn in current}
        group_conversations = {turn.conversation_id for turn in group}
        if current and (
            len(current_conversations | group_conversations)
            > _MAX_CONVERSATIONS_PER_BATCH
            or _estimate_turns(current + group) > _BATCH_INPUT_CHAR_BUDGET
        ):
            flush()
        if _estimate_turns(group) > _BATCH_INPUT_CHAR_BUDGET:
            for turn in group:
                if _estimate_turns([turn]) > _BATCH_INPUT_CHAR_BUDGET:
                    raise ValueError(f"单轮超过知识模型输入预算: {turn.message_id}")
                if current and _estimate_turns(current + [turn]) > _BATCH_INPUT_CHAR_BUDGET:
                    flush()
                current.append(turn)
        else:
            current.extend(group)
    flush()
    return batches


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
        # 先完成全部解析/切分；任一文件失败时不发布任何部分文档。
        prepared_drafts = []
        source_keys: list[str] = []
        for spec in specs:
            source = spec.path.read_text(encoding="utf-8")
            prepared_drafts.extend(
                chunk_markdown(source, spec, self.encoder.count_tokens)
            )
            source_keys.append(spec.source_key)
        async with self.store.writer() as writer:
            await self.lifecycle.publish_documents(
                writer, prepared_drafts, source_keys
            )
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

    async def _reserve_snapshot(self, writer, upper: int) -> None:
        """以ID游标分页预留冻结上界内所有来源，完成后才抽取。

        参数：writer持锁写入器；upper固定最大消息ID，不能扩展到新消息。
        返回：None。异常：来源变更、预算超限或存储错误，已预留批次保留恢复。
        """
        cursor = 0
        while True:
            turns = await writer.unassigned_turns(upper, cursor, _TURN_SCAN_LIMIT)
            if not turns:
                return
            for selected in _split_batches(turns):
                await writer.reserve_batch(_batch_from_turns(selected, upper))
            cursor = turns[-1].message_id

    async def _finish(
        self,
        writer,
        *,
        batch_count: int,
        extracted: int,
        staged_ids: list[int],
    ) -> JobReport:
        """统一去重、隔离与向量同步。参数：写入器与统计；返回：报告；异常：模型/存储错误。"""
        await writer.ensure_staging_grounded()
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
        """有中断先恢复固定upper；无中断才冻结新upper并统一去重同步。

        参数：无。返回：JobReport，完整快照处理统计。异常：模型/来源/存储错误。
        """
        async with self.store.writer() as writer:
            # 恢复先补齐旧upper，防止半快照过早统一去重；新任务仍从0补扫迟提交低ID。
            await self._reserve_snapshot(writer, await writer.recovery_upper_id())
            if not await writer.pending_batches():
                await self._reserve_snapshot(writer, await writer.source_upper_id())
            pending_count, pending_extracted, pending_ids = await self._process_pending(writer)
            return await self._finish(
                writer,
                batch_count=pending_count,
                extracted=pending_extracted,
                staged_ids=pending_ids,
            )

    async def recover_pending(self) -> JobReport:
        """补齐持久upper内未预留来源并恢复，不领取upper之后来源。

        参数：无。返回：JobReport，恢复批次统计。异常：模型/来源/存储错误。
        """
        async with self.store.writer() as writer:
            await self._reserve_snapshot(writer, await writer.recovery_upper_id())
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
