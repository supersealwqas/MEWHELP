"""CH03 知识库测试替身：不加载 torch 的编码器与离线配置构造工具。"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from typing import Any

from app.config import Settings
from app.knowledge.errors import ConcurrentKnowledgeWrite, SnapshotChanged
from app.knowledge.schemas import (
    ChunkDraft,
    ChunkSnapshot,
    CompletedTurn,
    EvidenceItem,
    ExtractedQA,
    ExtractionBatch,
    Publication,
    RelationDecision,
    StagedQA,
    StoreFrame,
    VectorHit,
    VectorRecord,
)
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import VectorIndexUnavailable


def make_settings(**overrides: object) -> Settings:
    """构造离线测试配置。

    参数:
        **overrides: 需要覆盖的 Settings 字段；未指定字段使用固定的离线测试值。
    返回值:
        Settings: 不读取真实 .env 的完整配置实例。
    异常:
        pydantic.ValidationError: 覆盖值与字段约束冲突时抛出。
    """
    values: dict[str, object] = {
        "chat_base_url": "http://test.invalid/v1",
        "chat_model": "test-model",
        "chat_api_key": "test-key",
        "knowledge_model_path": "models/bge-m3",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakeEncoder:
    """实现与 BgeM3Encoder 相同接口的测试编码器。

    记录输入文本，返回固定的 1024 维单位向量，不导入或加载 torch。
    """

    def __init__(self, revision: str = "fake-encoder-v1") -> None:
        """初始化测试编码器。

        参数:
            revision: 测试用编码规范标识，供 Contract 测试断言版本传递。
        返回值:
            None
        异常:
            无
        """
        self.revision = revision
        self.inputs: list[str] = []
        self.closed = False
        self._vector = [1.0] + [0.0] * 1023

    def count_tokens(self, text: str) -> int:
        """按字符数模拟 token 计数。

        参数:
            text: 待计数文本。
        返回值:
            int: 文本字符数。
        异常:
            无
        """
        return len(text)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """记录输入并返回固定向量。

        参数:
            texts: 待编码文本列表。
        返回值:
            list[list[float]]: 与输入等长的 1024 维单位向量。
        异常:
            无
        """
        self.inputs.extend(texts)
        return [list(self._vector) for _ in texts]

    async def aclose(self) -> None:
        """标记测试编码器已关闭。

        参数:
            无
        返回值:
            None
        异常:
            无
        """
        self.closed = True


class MemoryKnowledgeStore:
    """与 MySQL KnowledgeStore 同接口的内存契约替身。

    保存知识快照、暂存行、批次和已完成轮次；支持注入发布失败与快照变化，
    不连接数据库，也不替代真实 MySQL 集成证据。
    """

    def __init__(
        self,
        *,
        snapshots: list[ChunkSnapshot] | None = None,
        staged: list[StagedQA] | None = None,
        batches: list[ExtractionBatch] | None = None,
        turns: list[CompletedTurn] | None = None,
        staged_batches: dict[int, str] | None = None,
        fail_on_publish: bool = False,
        snapshot_changed: bool = False,
    ) -> None:
        """初始化内存状态。

        参数:
            snapshots: 初始知识快照。
            staged: 初始暂存行。
            batches: 初始抽取批次。
            turns: 可分配的已完成轮次。
            staged_batches: 暂存行到批次的映射。
            fail_on_publish: 发布时注入失败。
            snapshot_changed: 强制报告快照变化。
        返回值:
            None
        异常:
            无
        """
        self._snapshots = {row.id: row for row in snapshots or []}
        self._staged = {row.id: row for row in staged or []}
        self._batches = {batch.batch_no: batch for batch in batches or []}
        self._batch_status = {batch.batch_no: "pending" for batch in batches or []}
        self._staged_batch = dict(staged_batches or {})
        self._staged_fingerprints: dict[str, int] = {}
        self._turns = list(turns or [])
        self.fail_on_publish = fail_on_publish
        self.snapshot_changed = snapshot_changed
        self._next_id = max(self._snapshots, default=0) + 1
        self._next_staged_id = max(self._staged, default=0) + 1

    def _token(self) -> str:
        """计算内存状态令牌。参数：无；返回：SHA256；异常：无。"""
        payload = {
            "chunks": [
                row.model_dump(mode="json")
                for row in sorted(self._snapshots.values(), key=lambda item: item.id)
            ],
            "staged": [
                row.model_dump(mode="json")
                for row in sorted(self._staged.values(), key=lambda item: item.id)
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _allocate_id(self) -> int:
        """分配知识主键。参数：无；返回：正整数；异常：无。"""
        identifier = self._next_id
        self._next_id += 1
        return identifier

    def _allocate_staged_id(self) -> int:
        """分配暂存主键。参数：无；返回：正整数；异常：无。"""
        identifier = self._next_staged_id
        self._next_staged_id += 1
        return identifier

    @asynccontextmanager
    async def writer(self):
        """提供内存写上下文。参数：无；返回：写入器；异常：无。"""
        yield MemoryKnowledgeWriter(self)

    async def get_chunks(self, ids: list[int]) -> list[ChunkSnapshot]:
        """按主键读取快照。参数：主键列表；返回：快照；异常：无。"""
        return [self._snapshots[identifier] for identifier in ids if identifier in self._snapshots]

    async def pending_page(self, after_id: int, limit: int) -> list[ChunkSnapshot]:
        """按主键分页读取 active/pending。参数：游标与页大小；返回：快照；异常：无。"""
        rows = sorted(self._snapshots.values(), key=lambda item: item.id)
        return [
            row
            for row in rows
            if row.id > after_id and row.is_active and row.vectorize_status == "pending"
        ][:limit]

    async def active_ids(self, ids: list[int]) -> set[int]:
        """返回仍具备发布资格的主键。参数：主键列表；返回：集合；异常：无。"""
        return {
            identifier
            for identifier in ids
            if identifier in self._snapshots and self._snapshots[identifier].is_active
        }


class MemoryKnowledgeWriter:
    """内存 KnowledgeWriter 实现。"""

    def __init__(self, store: MemoryKnowledgeStore) -> None:
        """保存所属存储。参数：存储；返回：None；异常：无。"""
        self.store = store

    async def frame(self) -> StoreFrame:
        """返回当前快照与令牌。参数：无；返回：StoreFrame；异常：无。"""
        return StoreFrame(
            token=self.store._token(),
            chunks=sorted(self.store._snapshots.values(), key=lambda item: item.id),
            staged=sorted(self.store._staged.values(), key=lambda item: item.id),
        )

    async def chunks_by_sources(self, source_keys: list[str]) -> list[ChunkSnapshot]:
        """按来源键读取旧块。参数：来源键；返回：快照；异常：无。"""
        wanted = set(source_keys)
        return sorted(
            [row for row in self.store._snapshots.values() if row.source_key in wanted],
            key=lambda item: (item.source_key, item.ordinal),
        )

    def _repair_pointers(self, snapshots: dict[int, ChunkSnapshot], source_key: str) -> None:
        """修复同一来源 active 块的前后指针。参数：快照与来源键；返回：None；异常：无。"""
        active = sorted(
            [
                row
                for row in snapshots.values()
                if row.source_key == source_key and row.is_active
            ],
            key=lambda item: item.ordinal,
        )
        for index, row in enumerate(active):
            previous = active[index - 1].id if index > 0 else None
            following = active[index + 1].id if index + 1 < len(active) else None
            if row.prev_chunk_id != previous or row.next_chunk_id != following:
                snapshots[row.id] = row.model_copy(
                    update={"prev_chunk_id": previous, "next_chunk_id": following}
                )

    async def publish(self, change: Publication) -> list[ChunkSnapshot]:
        """原子应用发布、撤回与暂存处置。参数：变更集；返回：快照；异常：SnapshotChanged。"""
        if self.store.snapshot_changed or change.expected_token != self.store._token():
            raise SnapshotChanged("知识库快照已变化，拒绝发布")
        if self.store.fail_on_publish:
            raise RuntimeError("注入的发布失败")

        snapshots = dict(self.store._snapshots)
        staged = dict(self.store._staged)
        affected_sources: set[str] = set(change.replace_sources)
        published_ids: list[int] = []

        for draft in change.drafts:
            content_hash = content_fingerprint(draft.text)
            existing = next(
                (
                    row
                    for row in snapshots.values()
                    if row.source_key == draft.source_key and row.ordinal == draft.ordinal
                ),
                None,
            )
            if existing is None:
                identifier = self.store._allocate_id()
                snapshot = ChunkSnapshot(
                    id=identifier,
                    source_key=draft.source_key,
                    ordinal=draft.ordinal,
                    text=draft.text,
                    section_path=draft.section_path,
                    content_type=draft.content_type,
                    is_key_clause=draft.is_key_clause,
                    content_hash=content_hash,
                    embedding_revision=change.embedding_revision,
                    is_active=True,
                    vectorize_status="pending",
                )
                snapshots[identifier] = snapshot
            else:
                input_changed = (
                    content_hash != existing.content_hash
                    or change.embedding_revision != existing.embedding_revision
                    or not existing.is_active
                )
                update: dict[str, Any] = {
                    "text": draft.text,
                    "section_path": draft.section_path,
                    "content_type": draft.content_type,
                    "is_key_clause": draft.is_key_clause,
                    "content_hash": content_hash,
                    "embedding_revision": change.embedding_revision,
                    "is_active": True,
                }
                if input_changed:
                    update["vectorize_status"] = "pending"
                    update["vector_id"] = None
                snapshot = existing.model_copy(update=update)
                snapshots[existing.id] = snapshot
            published_ids.append(snapshot.id)
            affected_sources.add(draft.source_key)

        for source_key in change.replace_sources:
            incoming = {
                draft.ordinal for draft in change.drafts if draft.source_key == source_key
            }
            for row in list(snapshots.values()):
                if row.source_key == source_key and row.ordinal not in incoming:
                    snapshots[row.id] = row.model_copy(
                        update={
                            "is_active": False,
                            "vector_id": None,
                            "vectorize_status": "pending",
                        }
                    )
            affected_sources.add(source_key)

        for identifier in change.inactive_ids:
            row = snapshots.get(identifier)
            if row is None:
                raise SnapshotChanged("待撤回知识不存在")
            snapshots[identifier] = row.model_copy(
                update={
                    "is_active": False,
                    "vector_id": None,
                    "vectorize_status": "pending",
                }
            )
            affected_sources.add(row.source_key)

        for stage_change in change.stage_changes:
            row = staged.get(stage_change.id)
            if row is None:
                raise SnapshotChanged("暂存记录不存在")
            update: dict[str, Any] = {
                "status": stage_change.status,
                "conflict_group": stage_change.conflict_group,
            }
            if stage_change.knowledge_source_key:
                related = next(
                    (
                        candidate
                        for candidate in sorted(snapshots.values(), key=lambda item: item.id)
                        if candidate.source_key == stage_change.knowledge_source_key
                    ),
                    None,
                )
                update["knowledge_id"] = related.id if related else None
                affected_sources.add(stage_change.knowledge_source_key)
            staged[row.id] = row.model_copy(update=update)

        for source_key in affected_sources:
            self._repair_pointers(snapshots, source_key)

        self.store._snapshots = snapshots
        self.store._staged = staged
        result_ids = list(dict.fromkeys(published_ids + list(change.inactive_ids)))
        return [snapshots[identifier] for identifier in result_ids if identifier in snapshots]

    async def mark_vectorized(self, snapshot: ChunkSnapshot) -> bool:
        """条件回填向量完成状态。参数：快照；返回：是否成功；异常：无。"""
        current = self.store._snapshots.get(snapshot.id)
        if (
            current is None
            or not current.is_active
            or current.content_hash != snapshot.content_hash
            or current.embedding_revision != snapshot.embedding_revision
        ):
            return False
        self.store._snapshots[current.id] = current.model_copy(
            update={"vector_id": str(current.id), "vectorize_status": "done"}
        )
        return True

    async def reserve_batch(self, batch: ExtractionBatch) -> ExtractionBatch:
        """幂等预留批次并拒绝同源双批次。参数：批次；返回：批次；异常：ConcurrentKnowledgeWrite。"""
        existing = self.store._batches.get(batch.batch_no)
        if existing is not None:
            if (
                existing.source_ids == batch.source_ids
                and existing.snapshot_upper_id == batch.snapshot_upper_id
                and existing.input_hash == batch.input_hash
                and existing.prompt_version == batch.prompt_version
            ):
                return existing
            raise ConcurrentKnowledgeWrite("批次号已被其他输入占用")
        assigned = {
            identifier
            for other in self.store._batches.values()
            for identifier in other.source_ids
        }
        if assigned.intersection(batch.source_ids):
            raise ConcurrentKnowledgeWrite("来源已属于其他抽取批次")
        self.store._batches[batch.batch_no] = batch
        self.store._batch_status[batch.batch_no] = "pending"
        return batch

    async def pending_batches(self) -> list[ExtractionBatch]:
        """返回 pending 批次。参数：无；返回：批次；异常：无。"""
        return [
            batch
            for batch_no, batch in self.store._batches.items()
            if self.store._batch_status.get(batch_no) == "pending"
        ]

    async def complete_batch(
        self, batch_no: str, qas: list[ExtractedQA]
    ) -> list[StagedQA]:
        """把批次 QA 写入暂存并标记 done。参数：批次号与 QA；返回：暂存行；异常：SnapshotChanged。"""
        if batch_no not in self.store._batches:
            raise SnapshotChanged("抽取批次不存在")
        if self.store._batch_status.get(batch_no) == "done":
            return [
                row
                for identifier, row in self.store._staged.items()
                if self.store._staged_batch.get(identifier) == batch_no
            ]
        for qa in qas:
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
            fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            identifier = self.store._staged_fingerprints.get(fingerprint)
            if identifier is None:
                identifier = self.store._allocate_staged_id()
                self.store._staged[identifier] = StagedQA(
                    id=identifier,
                    source_ids=qa.source_ids,
                    question=qa.question,
                    answer=qa.answer,
                    category=qa.category,
                    status="extracted",
                )
                self.store._staged_fingerprints[fingerprint] = identifier
            self.store._staged_batch[identifier] = batch_no
        self.store._batch_status[batch_no] = "done"
        return [
            row
            for identifier, row in self.store._staged.items()
            if self.store._staged_batch.get(identifier) == batch_no
        ]

    async def unassigned_turns(
        self, snapshot_upper_id: int, after_id: int, limit: int
    ) -> list[CompletedTurn]:
        """返回未分配批次的完成轮次。参数：快照上界、游标、页大小；返回：轮次；异常：无。"""
        assigned = {
            identifier
            for batch in self.store._batches.values()
            for identifier in batch.source_ids
        }
        rows = [
            turn
            for turn in sorted(self.store._turns, key=lambda item: item.message_id)
            if after_id < turn.message_id <= snapshot_upper_id
            and turn.message_id not in assigned
        ]
        return rows[:limit]

    async def source_upper_id(self) -> int:
        """返回完成轮次最大消息 ID。参数：无；返回：ID；异常：无。"""
        return max((turn.message_id for turn in self.store._turns), default=0)


class MemoryVectorIndex:
    """与 MilvusVectorIndex 同接口的内存向量索引。

    支持写入成功后抛错、删除失败和主键分页；只用于离线契约测试，不替代
    Milvus 真实集合的所有权与持久化证据。
    """

    def __init__(self, *, fail_after_upsert: bool = False, fail_delete: bool = False) -> None:
        """初始化内存索引。

        参数:
            fail_after_upsert: 写入后模拟确认丢失。
            fail_delete: 删除时模拟故障。
        返回值:
            None
        异常:
            无
        """
        self.records: dict[int, VectorRecord] = {}
        self.fail_after_upsert = fail_after_upsert
        self.fail_delete = fail_delete
        self.drop_calls = 0
        self.closed = False

    @property
    def ids(self) -> set[int]:
        """返回当前主键集合。参数：无；返回：集合；异常：无。"""
        return set(self.records)

    async def ensure_ready(self, create: bool = False) -> None:
        """模拟初始化。参数：create；返回：None；异常：无。"""
        del create

    async def upsert(self, records: list[VectorRecord]) -> None:
        """写入或覆盖记录。参数：记录；返回：None；异常：VectorIndexUnavailable。"""
        for record in records:
            self.records[record.id] = record
        if self.fail_after_upsert:
            raise VectorIndexUnavailable("模拟 Milvus 写入确认丢失")

    async def search(self, vector: list[float], limit: int) -> list[VectorHit]:
        """按余弦相似度返回固定主键命中。参数：向量与数量；返回：命中；异常：无。"""
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        query_norm = sum(value * value for value in vector) ** 0.5
        scored: list[VectorHit] = []
        for record in self.records.values():
            record_norm = sum(value * value for value in record.vector) ** 0.5
            score = 0.0
            if query_norm and record_norm:
                score = sum(a * b for a, b in zip(vector, record.vector)) / (
                    query_norm * record_norm
                )
            scored.append(
                VectorHit(
                    id=record.id,
                    score=float(score),
                    content_hash=record.content_hash,
                    embedding_revision=record.embedding_revision,
                )
            )
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]

    async def iter_ids(self, batch_size: int = 1000):
        """分页迭代主键。参数：页大小；返回：异步批次；异常：ValueError。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        identifiers = sorted(self.records)
        for offset in range(0, len(identifiers), batch_size):
            yield identifiers[offset : offset + batch_size]

    async def delete(self, ids: list[int]) -> None:
        """按主键删除记录。参数：主键；返回：None；异常：VectorIndexUnavailable。"""
        if self.fail_delete:
            raise VectorIndexUnavailable("模拟 Milvus 删除失败")
        for identifier in ids:
            self.records.pop(identifier, None)

    async def aclose(self) -> None:
        """标记索引关闭。参数：无；返回：None；异常：无。"""
        self.closed = True


class ScriptedKnowledgeModel:
    """返回预置抽取/关系结果的知识模型替身。"""

    def __init__(
        self,
        *,
        extract_qas: list[ExtractedQA] | None = None,
        relation_decisions: list[RelationDecision] | None = None,
        extract_error: Exception | None = None,
        relation_error: Exception | None = None,
    ) -> None:
        """保存预置结果与异常。

        参数:
            extract_qas: 抽取返回的候选知识。
            relation_decisions: 关系判定返回。
            extract_error: 抽取时抛出的异常。
            relation_error: 关系判定时抛出的异常。
        返回值:
            None
        异常:
            无
        """
        self.extract_qas = list(extract_qas or [])
        self.relation_decisions = list(relation_decisions or [])
        self.extract_error = extract_error
        self.relation_error = relation_error
        self.batches: list[ExtractionBatch] = []
        self.relation_inputs: list[tuple[list[EvidenceItem], list[tuple[str, str]]]] = []

    async def extract(self, batch: ExtractionBatch) -> list[ExtractedQA]:
        """记录批次并返回预置 QA。参数：批次；返回：QA；异常：预置异常。"""
        self.batches.append(batch)
        if self.extract_error is not None:
            raise self.extract_error
        return list(self.extract_qas)

    async def relate(
        self, items: list[EvidenceItem], pairs: list[tuple[str, str]]
    ) -> list[RelationDecision]:
        """记录输入并返回预置关系。参数：证据与 pair；返回：决策；异常：预置异常。"""
        self.relation_inputs.append((items, pairs))
        if self.relation_error is not None:
            raise self.relation_error
        return list(self.relation_decisions)
