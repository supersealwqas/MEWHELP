"""MySQL 知识库原文、批次与原子发布存储。

读取接口不持有写锁；导入、挖掘和同步通过 KnowledgeWriter 在一条持锁物理
连接上执行短事务。发布、撤回与暂存处置同事务提交，向量回填使用内容指纹和
编码规范做条件更新，任何锁丢失或快照变化都失败关闭。
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.knowledge.errors import (
    ConcurrentKnowledgeWrite,
    KnowledgeStoreUnavailable,
    LeaseLost,
    SnapshotChanged,
)
from app.knowledge.lease import MySqlWriteLease
from app.knowledge.schemas import (
    ChunkDraft,
    ChunkSnapshot,
    CompletedTurn,
    ExtractedQA,
    ExtractionBatch,
    KnowledgeText,
    Publication,
    StageChange,
    StagedQA,
    StoreFrame,
)
from app.knowledge.text import content_fingerprint
from app.models.customer_service import Message
from app.models.knowledge import KnowledgeChunk, QAExtractionBatch, QAExtractionStaging

__all__ = [
    "KnowledgeStore",
    "KnowledgeWriter",
    "ConcurrentKnowledgeWrite",
    "KnowledgeStoreUnavailable",
    "LeaseLost",
    "SnapshotChanged",
]


def _snapshot_from_row(row: KnowledgeChunk) -> ChunkSnapshot:
    """把 MySQL 行映射为共享快照。参数：ORM 行；返回：快照；异常：字段校验失败。"""
    return ChunkSnapshot(
        id=row.id,
        source_key=row.source_key,
        ordinal=row.ordinal,
        text=KnowledgeText(
            category=row.category,
            questions=row.questions,
            answer=row.answer,
        ),
        section_path=row.section_path,
        content_type=row.content_type or "unknown",
        is_key_clause=bool(row.is_key_clause),
        content_hash=row.content_hash,
        embedding_revision=row.embedding_revision,
        is_active=bool(row.is_active),
        vector_id=row.vector_id,
        vectorize_status=row.vectorize_status,
        prev_chunk_id=row.prev_chunk_id,
        next_chunk_id=row.next_chunk_id,
    )


def _staged_from_row(row: QAExtractionStaging) -> StagedQA:
    """把 MySQL 暂存行映射为共享模型。参数：ORM 行；返回：StagedQA；异常：字段校验失败。"""
    return StagedQA(
        id=row.id,
        source_ids=list(row.source_ids or []),
        question=row.question,
        answer=row.answer,
        category=row.category,
        status=row.status,
        conflict_group=row.conflict_group,
        knowledge_id=row.knowledge_id,
    )


def _qa_fingerprint(qa: ExtractedQA) -> str:
    """计算来源与问答的规范化指纹。参数：候选 QA；返回：SHA256；异常：无。"""
    payload = json.dumps(
        {
            "source_ids": sorted(qa.source_ids),
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class KnowledgeStore:
    """知识库只读入口与写入器工厂。"""

    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        lease_engine: AsyncEngine,
    ) -> None:
        """保存会话工厂与锁专用引擎。

        参数:
            factory: 普通异步会话工厂。
            lease_engine: 使用 NullPool 的 MySQL 写锁引擎。
        返回值:
            None
        异常:
            无
        """
        self._factory = factory
        self._lease_engine = lease_engine

    @asynccontextmanager
    async def writer(self):
        """获取写锁并返回 KnowledgeWriter。

        参数:
            无
        返回值:
            AsyncContextManager[KnowledgeWriter]: 持锁写入器。
        异常:
            ConcurrentKnowledgeWrite、KnowledgeStoreUnavailable: 获取锁失败。
        """
        lease = await MySqlWriteLease.acquire(self._lease_engine)
        try:
            yield KnowledgeWriter(self._factory, lease)
        finally:
            await lease.release()

    async def get_chunks(self, ids: list[int]) -> list[ChunkSnapshot]:
        """按主键读取知识快照。参数：主键列表；返回：快照；异常：无。"""
        if not ids:
            return []
        async with self._factory() as session:
            rows = (
                await session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)))
            ).all()
            return [_snapshot_from_row(row) for row in rows]

    async def pending_page(self, after_id: int, limit: int) -> list[ChunkSnapshot]:
        """按主键分页读取 active/pending 行。参数：游标与页大小；返回：快照；异常：ValueError。"""
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        async with self._factory() as session:
            rows = (
                await session.scalars(
                    select(KnowledgeChunk)
                    .where(
                        KnowledgeChunk.id > after_id,
                        KnowledgeChunk.is_active.is_(True),
                        KnowledgeChunk.vectorize_status == "pending",
                    )
                    .order_by(KnowledgeChunk.id)
                    .limit(limit)
                )
            ).all()
            return [_snapshot_from_row(row) for row in rows]

    async def active_ids(self, ids: list[int]) -> set[int]:
        """返回仍 active 的主键。参数：主键列表；返回：集合；异常：无。"""
        if not ids:
            return set()
        async with self._factory() as session:
            rows = (
                await session.scalars(
                    select(KnowledgeChunk.id).where(
                        KnowledgeChunk.id.in_(ids),
                        KnowledgeChunk.is_active.is_(True),
                    )
                )
            ).all()
        return set(rows)


class KnowledgeWriter:
    """在持锁物理连接上执行短事务的写入器。"""

    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        lease: MySqlWriteLease,
    ) -> None:
        """保存会话工厂与租约。

        参数:
            factory: 普通异步会话工厂。
            lease: 已持有的 MySQL 写锁。
        返回值:
            None
        异常:
            无
        """
        self._factory = factory
        self._lease = lease

    @asynccontextmanager
    async def _transaction(self):
        """在持锁连接上开启短事务。参数：无；返回：会话；异常：LeaseLost、数据库异常。"""
        await self._lease.assert_held()
        connection = self._lease.connection
        # 锁校验的查询会自动开启事务；先结束它，避免 Session 只提交 SAVEPOINT。
        if connection.in_transaction():
            await connection.commit()
        session = AsyncSession(bind=connection, expire_on_commit=False)
        try:
            yield session
            await session.commit()
            # Session 绑定外部连接时可能只释放 SAVEPOINT；确保外层事务也提交。
            if connection.in_transaction():
                await connection.commit()
        except Exception:
            await session.rollback()
            if connection.in_transaction():
                await connection.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def _read_session(self):
        """在持锁连接上只读。参数：无；返回：会话；异常：LeaseLost、数据库异常。"""
        await self._lease.assert_held()
        connection = self._lease.connection
        if connection.in_transaction():
            await connection.commit()
        session = AsyncSession(bind=connection, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.rollback()
            if connection.in_transaction():
                await connection.rollback()
            await session.close()

    async def _token(self, session: AsyncSession) -> str:
        """计算覆盖原文、发布资格与暂存的令牌。参数：会话；返回：SHA256；异常：数据库异常。"""
        chunks = (
            await session.execute(
                select(
                    KnowledgeChunk.id,
                    KnowledgeChunk.source_key,
                    KnowledgeChunk.content_hash,
                    KnowledgeChunk.embedding_revision,
                    KnowledgeChunk.is_active,
                    KnowledgeChunk.vectorize_status,
                ).order_by(KnowledgeChunk.id)
            )
        ).all()
        staged = (
            await session.execute(
                select(
                    QAExtractionStaging.id,
                    QAExtractionStaging.source_fingerprint,
                    QAExtractionStaging.status,
                    QAExtractionStaging.conflict_group,
                    QAExtractionStaging.knowledge_id,
                ).order_by(QAExtractionStaging.id)
            )
        ).all()
        payload = {
            "chunks": [list(row) for row in chunks],
            "staged": [list(row) for row in staged],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def frame(self) -> StoreFrame:
        """读取当前状态与令牌。参数：无；返回：StoreFrame；异常：存储异常。"""
        async with self._read_session() as session:
            token = await self._token(session)
            chunks = (
                await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id))
            ).all()
            staged = (
                await session.scalars(
                    select(QAExtractionStaging).order_by(QAExtractionStaging.id)
                )
            ).all()
            return StoreFrame(
                token=token,
                chunks=[_snapshot_from_row(row) for row in chunks],
                staged=[_staged_from_row(row) for row in staged],
            )

    async def chunks_by_sources(self, source_keys: list[str]) -> list[ChunkSnapshot]:
        """按来源键读取旧块。参数：来源键；返回：快照；异常：存储异常。"""
        if not source_keys:
            return []
        async with self._read_session() as session:
            rows = (
                await session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.source_key.in_(source_keys))
                    .order_by(KnowledgeChunk.source_key, KnowledgeChunk.ordinal)
                )
            ).all()
            return [_snapshot_from_row(row) for row in rows]

    async def _repair_pointers(self, session: AsyncSession, source_key: str) -> None:
        """修复同来源 active 块的前后指针。参数：会话与来源键；返回：None；异常：数据库异常。"""
        rows = (
            await session.scalars(
                select(KnowledgeChunk)
                .where(
                    KnowledgeChunk.source_key == source_key,
                    KnowledgeChunk.is_active.is_(True),
                )
                .order_by(KnowledgeChunk.ordinal)
                .with_for_update()
            )
        ).all()
        for index, row in enumerate(rows):
            row.prev_chunk_id = rows[index - 1].id if index > 0 else None
            row.next_chunk_id = rows[index + 1].id if index + 1 < len(rows) else None
        await session.flush()

    async def publish(self, change: Publication) -> list[ChunkSnapshot]:
        """原子发布知识、撤回旧资格并处理暂存状态。

        参数:
            change: 含快照令牌、草案、撤回和暂存处置的变更集。
        返回值:
            list[ChunkSnapshot]: 本次草案与撤回行的更新后快照。
        异常:
            SnapshotChanged: 令牌不匹配或目标行不存在。
            LeaseLost、KnowledgeStoreUnavailable: 锁或数据库不可用。
        """
        async with self._transaction() as session:
            if await self._token(session) != change.expected_token:
                raise SnapshotChanged("知识库快照已变化，拒绝发布")

            affected_rows: list[KnowledgeChunk] = []
            affected_sources: set[str] = set(change.replace_sources)

            for draft in change.drafts:
                existing = await session.scalar(
                    select(KnowledgeChunk)
                    .where(
                        KnowledgeChunk.source_key == draft.source_key,
                        KnowledgeChunk.ordinal == draft.ordinal,
                    )
                    .with_for_update()
                )
                content_hash = content_fingerprint(draft.text)
                if existing is None:
                    row = KnowledgeChunk(
                        source_key=draft.source_key,
                        ordinal=draft.ordinal,
                        category=draft.text.category,
                        questions=draft.text.questions,
                        answer=draft.text.answer,
                        section_path=draft.section_path,
                        content_type=draft.content_type,
                        is_key_clause=draft.is_key_clause,
                        content_hash=content_hash,
                        embedding_revision=change.embedding_revision,
                        is_active=True,
                        vectorize_status="pending",
                    )
                    session.add(row)
                else:
                    input_changed = (
                        content_hash != existing.content_hash
                        or change.embedding_revision != existing.embedding_revision
                        or not existing.is_active
                    )
                    existing.category = draft.text.category
                    existing.questions = draft.text.questions
                    existing.answer = draft.text.answer
                    existing.section_path = draft.section_path
                    existing.content_type = draft.content_type
                    existing.is_key_clause = draft.is_key_clause
                    existing.content_hash = content_hash
                    existing.embedding_revision = change.embedding_revision
                    existing.is_active = True
                    if input_changed:
                        existing.vectorize_status = "pending"
                        existing.vector_id = None
                    row = existing
                affected_sources.add(draft.source_key)
                affected_rows.append(row)

            await session.flush()

            for source_key in change.replace_sources:
                incoming = {
                    draft.ordinal for draft in change.drafts if draft.source_key == source_key
                }
                old_rows = (
                    await session.scalars(
                        select(KnowledgeChunk)
                        .where(KnowledgeChunk.source_key == source_key)
                        .with_for_update()
                    )
                ).all()
                for row in old_rows:
                    if row.ordinal not in incoming:
                        row.is_active = False
                        row.vector_id = None
                        row.vectorize_status = "pending"
                affected_sources.add(source_key)

            for identifier in change.inactive_ids:
                row = await session.scalar(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.id == identifier)
                    .with_for_update()
                )
                if row is None:
                    raise SnapshotChanged("待撤回知识不存在")
                row.is_active = False
                row.vector_id = None
                row.vectorize_status = "pending"
                affected_sources.add(row.source_key)
                affected_rows.append(row)

            for stage_change in change.stage_changes:
                staged = await session.scalar(
                    select(QAExtractionStaging)
                    .where(QAExtractionStaging.id == stage_change.id)
                    .with_for_update()
                )
                if staged is None:
                    raise SnapshotChanged("暂存记录不存在")
                staged.status = stage_change.status
                staged.conflict_group = stage_change.conflict_group
                if stage_change.reason:
                    staged.reason = stage_change.reason
                if stage_change.evidence_refs:
                    staged.evidence_refs = list(stage_change.evidence_refs)
                if stage_change.knowledge_source_key:
                    related = await session.scalar(
                        select(KnowledgeChunk)
                        .where(KnowledgeChunk.source_key == stage_change.knowledge_source_key)
                        .order_by(KnowledgeChunk.ordinal)
                        .limit(1)
                    )
                    staged.knowledge_id = related.id if related else None
                    affected_sources.add(stage_change.knowledge_source_key)

            await session.flush()
            for source_key in affected_sources:
                await self._repair_pointers(session, source_key)

            result_rows = {
                row.id: row for row in affected_rows if row.id is not None
            }
            return [_snapshot_from_row(row) for row in result_rows.values()]

    async def mark_vectorized(self, snapshot: ChunkSnapshot) -> bool:
        """条件回填向量完成状态。参数：已编码快照；返回：是否成功；异常：存储异常。"""
        async with self._transaction() as session:
            result = await session.execute(
                update(KnowledgeChunk)
                .where(
                    KnowledgeChunk.id == snapshot.id,
                    KnowledgeChunk.is_active.is_(True),
                    KnowledgeChunk.content_hash == snapshot.content_hash,
                    KnowledgeChunk.embedding_revision == snapshot.embedding_revision,
                )
                .values(vector_id=str(snapshot.id), vectorize_status="done")
                .execution_options(synchronize_session=False)
            )
            return result.rowcount == 1

    def _batch_values(self, batch: ExtractionBatch) -> dict[str, Any]:
        """提取批次持久字段。参数：批次；返回：字段映射；异常：无。"""
        return {
            "batch_no": batch.batch_no,
            "source_ids": list(batch.source_ids),
            "snapshot_upper_id": batch.snapshot_upper_id,
            "input_hash": batch.input_hash,
            "prompt_version": batch.prompt_version,
        }

    async def reserve_batch(self, batch: ExtractionBatch) -> ExtractionBatch:
        """幂等预留批次并拒绝同源双批次。参数：批次；返回：已预留批次；异常：并发冲突。"""
        async with self._transaction() as session:
            existing = await session.scalar(
                select(QAExtractionBatch)
                .where(QAExtractionBatch.batch_no == batch.batch_no)
                .with_for_update()
            )
            if existing is not None:
                if (
                    list(existing.source_ids or []) == list(batch.source_ids)
                    and existing.snapshot_upper_id == batch.snapshot_upper_id
                    and existing.input_hash == batch.input_hash
                    and existing.prompt_version == batch.prompt_version
                ):
                    return batch
                raise ConcurrentKnowledgeWrite("批次号已被其他输入占用")

            assigned = {
                int(identifier)
                for source_ids in (
                    await session.scalars(select(QAExtractionBatch.source_ids))
                ).all()
                for identifier in (source_ids or [])
            }
            if assigned.intersection(batch.source_ids):
                raise ConcurrentKnowledgeWrite("来源已属于其他抽取批次")
            row = QAExtractionBatch(**self._batch_values(batch))
            session.add(row)
            await session.flush()
            return batch

    async def pending_batches(self) -> list[ExtractionBatch]:
        """返回 pending 批次并重建完整轮次。参数：无；返回：批次；异常：来源缺失。"""
        async with self._read_session() as session:
            rows = (
                await session.scalars(
                    select(QAExtractionBatch)
                    .where(QAExtractionBatch.status == "pending")
                    .order_by(QAExtractionBatch.id)
                )
            ).all()
            return [await self._batch_from_row(session, row) for row in rows]

    async def _batch_from_row(
        self, session: AsyncSession, row: QAExtractionBatch
    ) -> ExtractionBatch:
        """从批次行重建固定轮次。参数：会话与行；返回：批次；异常：来源缺失。"""
        source_ids = [int(identifier) for identifier in (row.source_ids or [])]
        turns = await self._load_turns(session, source_ids)
        return ExtractionBatch(
            batch_no=row.batch_no,
            source_ids=source_ids,
            snapshot_upper_id=row.snapshot_upper_id,
            input_hash=row.input_hash,
            prompt_version=row.prompt_version,
            turns=turns,
        )

    async def _load_turns(
        self, session: AsyncSession, source_ids: list[int]
    ) -> list[CompletedTurn]:
        """按最终回答消息 ID 重建来源轮次。参数：会话与来源 ID；返回：轮次；异常：来源缺失。"""
        turns: list[CompletedTurn] = []
        for identifier in sorted(source_ids):
            assistant = await session.get(Message, identifier)
            if (
                assistant is None
                or assistant.role != "assistant"
                or assistant.turn_status != "completed"
                or not assistant.content
                or assistant.tool_calls
            ):
                raise KnowledgeStoreUnavailable(f"来源消息不可用: {identifier}")
            user = await session.scalar(
                select(Message)
                .where(
                    Message.conversation_id == assistant.conversation_id,
                    Message.turn_id == assistant.turn_id,
                    Message.role == "user",
                )
                .order_by(Message.id)
                .limit(1)
            )
            if user is None or not user.content:
                raise KnowledgeStoreUnavailable(f"来源轮次缺少用户消息: {identifier}")
            turns.append(
                CompletedTurn(
                    message_id=assistant.id,
                    conversation_id=assistant.conversation_id,
                    turn_id=assistant.turn_id,
                    question=user.content,
                    answer=assistant.content,
                )
            )
        return turns

    async def complete_batch(
        self, batch_no: str, qas: list[ExtractedQA]
    ) -> list[StagedQA]:
        """写入候选 QA 并把批次标记 done。参数：批次号与 QA；返回：暂存行；异常：快照变化。"""
        async with self._transaction() as session:
            batch = await session.scalar(
                select(QAExtractionBatch)
                .where(QAExtractionBatch.batch_no == batch_no)
                .with_for_update()
            )
            if batch is None:
                raise SnapshotChanged("抽取批次不存在")
            if batch.status != "done":
                for qa in qas:
                    fingerprint = _qa_fingerprint(qa)
                    existing = await session.scalar(
                        select(QAExtractionStaging)
                        .where(QAExtractionStaging.source_fingerprint == fingerprint)
                        .with_for_update()
                    )
                    if existing is None:
                        session.add(
                            QAExtractionStaging(
                                batch_no=batch_no,
                                source_ids=list(qa.source_ids),
                                source_fingerprint=fingerprint,
                                question=qa.question,
                                answer=qa.answer,
                                category=qa.category,
                                status="extracted",
                                evidence_refs=[],
                            )
                        )
                batch.status = "done"
                await session.flush()
            rows = (
                await session.scalars(
                    select(QAExtractionStaging)
                    .where(QAExtractionStaging.batch_no == batch_no)
                    .order_by(QAExtractionStaging.id)
                )
            ).all()
            return [_staged_from_row(row) for row in rows]

    async def unassigned_turns(
        self, snapshot_upper_id: int, after_id: int, limit: int
    ) -> list[CompletedTurn]:
        """扫描未分配批次的完成轮次。参数：快照上界、游标、页大小；返回：轮次；异常：存储异常。"""
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        async with self._read_session() as session:
            assigned = {
                int(identifier)
                for source_ids in (
                    await session.scalars(select(QAExtractionBatch.source_ids))
                ).all()
                for identifier in (source_ids or [])
            }
            assistants = (
                await session.scalars(
                    select(Message)
                    .where(
                        Message.role == "assistant",
                        Message.turn_status == "completed",
                        Message.content.is_not(None),
                        Message.content != "",
                        Message.tool_calls.is_(None),
                        Message.id > after_id,
                        Message.id <= snapshot_upper_id,
                    )
                    .order_by(Message.id)
                )
            ).all()
            turns: list[CompletedTurn] = []
            for assistant in assistants:
                if assistant.id in assigned:
                    continue
                user = await session.scalar(
                    select(Message)
                    .where(
                        Message.conversation_id == assistant.conversation_id,
                        Message.turn_id == assistant.turn_id,
                        Message.role == "user",
                    )
                    .order_by(Message.id)
                    .limit(1)
                )
                if user is None or not user.content:
                    continue
                turns.append(
                    CompletedTurn(
                        message_id=assistant.id,
                        conversation_id=assistant.conversation_id,
                        turn_id=assistant.turn_id,
                        question=user.content,
                        answer=assistant.content,
                    )
                )
                if len(turns) >= limit:
                    break
            return turns

    async def source_upper_id(self) -> int:
        """返回完成最终回答的最大消息 ID。参数：无；返回：ID；异常：存储异常。"""
        async with self._read_session() as session:
            value = await session.scalar(
                select(func.max(Message.id)).where(
                    Message.role == "assistant",
                    Message.turn_status == "completed",
                    Message.content.is_not(None),
                    Message.content != "",
                    Message.tool_calls.is_(None),
                )
            )
        return int(value or 0)
