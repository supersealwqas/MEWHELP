"""MySQL 知识库原文、批次与原子发布存储。

读取接口不持有写锁；导入、挖掘和同步通过 KnowledgeWriter 在一条持锁物理
连接上执行短事务。发布、撤回与暂存处置同事务提交，向量回填使用内容指纹和
编码规范做条件更新，任何锁丢失或快照变化都失败关闭。
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
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
from app.knowledge.grounding import turns_digest, validate_grounding
from app.knowledge.prompts import EXTRACTION_PROMPT_VERSION
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
        source_evidence=[item for item in (row.evidence_refs or []) if isinstance(item, dict) and item.get("kind") == "source"],
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
        """按主键读取知识快照。

        参数:
            ids: 待读取的知识主键列表，允许为空。
        返回值:
            list[ChunkSnapshot]: 已存在的知识快照；空输入返回空列表。
        异常:
            数据库连接或查询异常向调用方传播；快照字段不合法时抛出 pydantic.ValidationError。
        """
        if not ids:
            return []
        async with self._factory() as session:
            rows = (
                await session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)))
            ).all()
            return [_snapshot_from_row(row) for row in rows]

    async def get_raw_chunks(self, ids: list[int]) -> list[dict]:
        """用一次只读select返回原始18字段，不改变共享快照的NULL映射。

        参数：ids为知识主键列表，允许为空。
        返回：list[dict]，所有原始字段；主键/前后指针为字符串，时间保留原值的ISO表示。
        异常：数据库连接、查询或字段错误原样传播，由调试服务安全包装。
        """
        if not ids:
            return []
        columns = list(KnowledgeChunk.__table__.columns)
        async with self._factory() as session:
            rows = (await session.execute(select(*columns).where(KnowledgeChunk.id.in_(sorted(set(ids)))))).mappings().all()
        result = []
        for row in rows:
            item = dict(row)
            for key, value in item.items():
                if value is not None and key in ("id", "prev_chunk_id", "next_chunk_id", "vector_id"):
                    item[key] = str(value)
                elif isinstance(value, (datetime, date)):
                    # MySQL DATETIME无时区，保留原始墙钟时间，不能虚构UTC换算。
                    item[key] = value.isoformat()
            result.append(item)
        return result

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
        """返回仍具备发布资格的知识主键。

        参数:
            ids: 待核对的知识主键列表，允许为空。
        返回值:
            set[int]: 输入中仍为 active 的主键；空输入返回空集合。
        异常:
            数据库连接或查询异常向调用方传播。
        """
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

    async def active_page(self, after_id: int, limit: int) -> list[ChunkSnapshot]:
        """按主键分页读取全部 active 行。参数：游标、正页大小；返回：快照；异常：ValueError/存储异常。"""
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        async with self._factory() as session:
            rows = (
                await session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.id > after_id, KnowledgeChunk.is_active.is_(True))
                    .order_by(KnowledgeChunk.id)
                    .limit(limit)
                )
            ).all()
            return [_snapshot_from_row(row) for row in rows]


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

    async def assert_held(self) -> None:
        """核锁并结束只读事务，供外部向量变更前后使用。参数：无；返回：None；异常：LeaseLost/存储异常。"""
        await self._lease.assert_held()
        connection = self._lease.connection
        if connection.in_transaction():
            await connection.commit()

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
    async def transaction(self):
        """为同租约关联写入提供短事务，不能用于等待模型或索引。

        参数：无，调用方必须仍位于当前writer生命周期内。
        返回：绑定当前持锁物理连接的AsyncSession；异常：失锁、断线或事务错误。
        """
        async with self._transaction() as session:
            yield session

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
                    QAExtractionStaging.question,
                    QAExtractionStaging.answer,
                    QAExtractionStaging.category,
                    QAExtractionStaging.evidence_refs,
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
            # 直接调用生命周期也必须补证，不能绕过jobs发布无来源的旧候选。
            await self._ground_staging(session)

            # 所有步骤复用当前会话；辅助方法不提交，任一步失败都由外层整体回滚。
            affected_rows = await self._save_drafts(
                session, change.drafts, change.embedding_revision
            )
            affected_sources = set(change.replace_sources)
            affected_sources.update(draft.source_key for draft in change.drafts)
            # 保持原有刷新位置：先取得新主键，再撤回旧块和关联暂存记录。
            await session.flush()

            await self._withdraw_missing_chunks(
                session, change.replace_sources, change.drafts
            )
            withdrawn = await self._withdraw_chunks(session, change.inactive_ids)
            affected_rows.extend(withdrawn)
            affected_sources.update(row.source_key for row in withdrawn)
            affected_sources.update(
                await self._apply_stage_changes(session, change.stage_changes)
            )

            await session.flush()
            for source_key in affected_sources:
                await self._repair_pointers(session, source_key)

            result_rows = {row.id: row for row in affected_rows if row.id is not None}
            return [_snapshot_from_row(row) for row in result_rows.values()]

    async def _save_drafts(
        self, session: AsyncSession, drafts: list[ChunkDraft], embedding_revision: str
    ) -> list[KnowledgeChunk]:
        """按来源和序号保存草案，保持知识主键稳定。

        参数:
            session: publish 持有的同一事务会话，本方法不提交。
            drafts: 按原发布顺序排列的草案。
            embedding_revision: 本次发布采用的编码规范标识。
        返回值:
            list[KnowledgeChunk]: 与草案顺序对应的已新增或更新实体。
        异常:
            数据库异常由调用方整体回滚。
        """
        rows: list[KnowledgeChunk] = []
        for draft in drafts:
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
                    embedding_revision=embedding_revision,
                    is_active=True,
                    vectorize_status="pending",
                )
                session.add(row)
            else:
                input_changed = (
                    content_hash != existing.content_hash
                    or embedding_revision != existing.embedding_revision
                    or not existing.is_active
                )
                existing.category = draft.text.category
                existing.questions = draft.text.questions
                existing.answer = draft.text.answer
                existing.section_path = draft.section_path
                existing.content_type = draft.content_type
                existing.is_key_clause = draft.is_key_clause
                existing.content_hash = content_hash
                existing.embedding_revision = embedding_revision
                existing.is_active = True
                # 元数据更新仍保留 done，只有编码输入或发布资格变化才重新同步。
                if input_changed:
                    existing.vectorize_status = "pending"
                    existing.vector_id = None
                row = existing
            rows.append(row)
        return rows

    @staticmethod
    def _deactivate_chunk(row: KnowledgeChunk) -> None:
        """清除知识发布资格、向量关联及旧邻接指针。

        参数:
            row: 当前事务中已经锁定的知识实体。
        返回值:
            None；只修改实体字段，不执行提交。
        异常:
            无。
        """
        row.is_active = False
        row.vector_id = None
        row.vectorize_status = "pending"
        # 两种撤回入口使用同一规则，旧块不能继续指向有效知识。
        row.prev_chunk_id = None
        row.next_chunk_id = None

    async def _withdraw_missing_chunks(
        self, session: AsyncSession, source_keys: list[str], drafts: list[ChunkDraft]
    ) -> None:
        """撤回替换来源中未出现在新草案里的旧块。

        参数:
            session: publish 持有的同一事务会话。
            source_keys: 本次完整替换的来源；空列表不撤回。
            drafts: 本次全部草案，用于确定各来源保留的序号。
        返回值:
            None；保留旧实体作为历史记录。
        异常:
            数据库异常由调用方整体回滚。
        """
        for source_key in source_keys:
            incoming = {draft.ordinal for draft in drafts if draft.source_key == source_key}
            old_rows = (
                await session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.source_key == source_key)
                    .with_for_update()
                )
            ).all()
            for row in old_rows:
                if row.ordinal not in incoming:
                    self._deactivate_chunk(row)

    async def _withdraw_chunks(
        self, session: AsyncSession, identifiers: list[int]
    ) -> list[KnowledgeChunk]:
        """按显式主键撤回知识，并返回需要汇报的实体。

        参数:
            session: publish 持有的同一事务会话。
            identifiers: 按原发布顺序排列的待撤回主键。
        返回值:
            list[KnowledgeChunk]: 已撤回实体，保留传入顺序。
        异常:
            SnapshotChanged: 任一目标不存在；数据库异常由调用方整体回滚。
        """
        rows: list[KnowledgeChunk] = []
        for identifier in identifiers:
            row = await session.scalar(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.id == identifier)
                .with_for_update()
            )
            if row is None:
                raise SnapshotChanged("待撤回知识不存在")
            self._deactivate_chunk(row)
            rows.append(row)
        return rows

    async def _apply_stage_changes(
        self, session: AsyncSession, changes: list[StageChange]
    ) -> set[str]:
        """更新暂存处置、追加历史审计并关联已发布知识。

        参数:
            session: publish 持有的同一事务会话。
            changes: 按原发布顺序排列的暂存处置；空字段沿用原规则。
        返回值:
            set[str]: 需要重建邻接指针的知识来源。
        异常:
            SnapshotChanged: 暂存目标不存在；数据库异常由调用方整体回滚。
        """
        affected_sources: set[str] = set()
        for change in changes:
            staged = await session.scalar(
                select(QAExtractionStaging)
                .where(QAExtractionStaging.id == change.id)
                .with_for_update()
            )
            if staged is None:
                raise SnapshotChanged("暂存记录不存在")
            staged.status = change.status
            staged.conflict_group = change.conflict_group
            if change.reason:
                staged.reason = change.reason
            if change.evidence_refs:
                # 按JSON内容去重并保留首次出现顺序，旧版本原文快照不能被覆盖。
                audit = []
                seen = set()
                for item in list(staged.evidence_refs or []) + list(change.evidence_refs):
                    fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True)
                    if fingerprint not in seen:
                        audit.append(item)
                        seen.add(fingerprint)
                staged.evidence_refs = audit
            if change.knowledge_source_key:
                related = await session.scalar(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.source_key == change.knowledge_source_key)
                    .order_by(KnowledgeChunk.ordinal)
                    .limit(1)
                )
                staged.knowledge_id = related.id if related else None
                affected_sources.add(change.knowledge_source_key)
        return affected_sources

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

    async def mark_pending(self, snapshot: ChunkSnapshot) -> bool:
        """把缺向量或指纹失配的原 done 快照条件恢复为 pending。

        参数：snapshot 为对照时的 active/done 快照。返回：bool，仍为该状态才成功。
        异常：LeaseLost、存储异常；旧快照不覆盖新原文、编码规范或发布状态。
        """
        if not snapshot.is_active or snapshot.vectorize_status != "done":
            return False
        async with self._transaction() as session:
            result = await session.execute(
                update(KnowledgeChunk)
                .where(
                    KnowledgeChunk.id == snapshot.id,
                    KnowledgeChunk.is_active.is_(True),
                    KnowledgeChunk.vectorize_status == "done",
                    KnowledgeChunk.vector_id == snapshot.vector_id,
                    KnowledgeChunk.content_hash == snapshot.content_hash,
                    KnowledgeChunk.embedding_revision == snapshot.embedding_revision,
                )
                .values(vector_id=None, vectorize_status="pending")
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
            loaded = await self._load_turns(session, batch.source_ids)
            if turns_digest(loaded) != batch.input_hash or turns_digest(batch.turns) != batch.input_hash:
                raise SnapshotChanged("预留来源指纹已变化")
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
                ):
                    return await self._batch_from_row(session, existing)
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
        if turns_digest(turns) != row.input_hash:
            raise SnapshotChanged("抽取批次来源指纹已变化")
        return ExtractionBatch(
            batch_no=row.batch_no,
            source_ids=source_ids,
            snapshot_upper_id=row.snapshot_upper_id,
            input_hash=row.input_hash,
            prompt_version=EXTRACTION_PROMPT_VERSION if row.status == "pending" else row.prompt_version,
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
            # done幂等重读也必须回核固定来源；只读验证，不再次抽取或改已完成身份。
            loaded = await self._batch_from_row(session, batch)
            if batch.status != "done":
                # 校验全部候选后才写；候选、代码审计、done同短事务提交。
                verified = [validate_grounding(qa, loaded.turns) for qa in qas]
                for qa, audit in verified:
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
                                evidence_refs=audit,
                            )
                        )
                    else:
                        existing.evidence_refs = audit + [item for item in (existing.evidence_refs or []) if not isinstance(item, dict) or item.get("kind") != "source"]
                batch.status = "done"
                batch.prompt_version = EXTRACTION_PROMPT_VERSION
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
            assigned = select(QAExtractionBatch.id).where(
                func.json_contains(QAExtractionBatch.source_ids, func.json_array(Message.id)) == 1
            ).exists()
            turns: list[CompletedTurn] = []
            cursor = after_id
            while len(turns) < limit:
                assistants = (
                    await session.scalars(
                        select(Message)
                        .where(
                            Message.role == "assistant",
                            Message.turn_status == "completed",
                            Message.content.is_not(None),
                            Message.content != "",
                            Message.tool_calls.is_(None),
                            Message.id > cursor,
                            Message.id <= snapshot_upper_id,
                            ~assigned,
                        )
                        .order_by(Message.id)
                        .limit(limit - len(turns))
                    )
                ).all()
                if not assistants:
                    break
                cursor = assistants[-1].id
                for assistant in assistants:
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
            return turns

    async def recovery_upper_id(self) -> int:
        """获取全部持久批次的恢复上界（含旧done半快照）。

        参数：无。返回：int，无批次为0。异常：存储异常。
        """
        async with self._read_session() as session:
            value = await session.scalar(select(func.max(QAExtractionBatch.snapshot_upper_id)))
        return int(value or 0)

    async def _ground_staging(self, session: AsyncSession) -> None:
        """补证参与发布的旧暂存并保留处置审计。

        参数：session为当前短事务；所有extracted/kept均须来源可回核。
        返回：None。异常：来源缺失、指纹改变或GroundingError，事务整体回滚。
        """
        rows = (await session.scalars(select(QAExtractionStaging).where(
            QAExtractionStaging.status.in_(["extracted", "kept"])
        ).with_for_update())).all()
        for row in rows:
            batch = await session.scalar(select(QAExtractionBatch).where(QAExtractionBatch.batch_no == row.batch_no))
            if batch is None:
                turns = await self._load_turns(session, list(row.source_ids or []))
            else:
                turns = (await self._batch_from_row(session, batch)).turns
            _, audit = validate_grounding(_staged_from_row(row), turns)
            previous = [item for item in (row.evidence_refs or []) if isinstance(item, dict) and item.get("kind") == "source"]
            if previous and previous != audit:
                raise SnapshotChanged("暂存来源证据指纹已变化")
            row.evidence_refs = audit + [item for item in (row.evidence_refs or [])
                                         if not isinstance(item, dict) or item.get("kind") != "source"]

    async def ensure_staging_grounded(self) -> None:
        """在关系判定前补证旧暂存。参数：无；返回：None；异常：来源/摘要校验失败。"""
        async with self._transaction() as session:
            await self._ground_staging(session)

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
