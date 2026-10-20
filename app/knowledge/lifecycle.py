"""知识发布生命周期：文档权威、全局去重与持久冲突隔离。

文档重导入以 data/kb 为权威，同事务撤回被推翻的已发布挖掘知识；离线挖掘
先做同义合并，无文档裁定的互斥组整组撤回并写入持久 conflict_group。生命周期
只构造 Publication，不直接写 Milvus；向量同步由调用方在同一 lease 内完成。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any

from app.knowledge.llm import KnowledgeModel, KnowledgeModelError
from app.knowledge.schemas import (
    ChunkDraft,
    ChunkSnapshot,
    EvidenceItem,
    KnowledgeText,
    Publication,
    RelationDecision,
    RelationPolicy,
    StageChange,
    StagedQA,
)
from app.knowledge.text import content_fingerprint, format_knowledge

_DOCUMENT_PREFIX = "doc:"
_MINING_PREFIX = "qa:"


def _cosine(left: list[float], right: list[float]) -> float:
    """计算余弦相似度。参数：向量；返回：分数；异常：无。"""
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class _DisjointSet:
    """候选同义合并的最小并查集。"""

    def __init__(self, values: list[int]) -> None:
        """初始化集合。参数：候选 ID；返回：None；异常：无。"""
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        """查找根。参数：ID；返回：根；异常：无。"""
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        """合并两个集合。参数：ID；返回：None；异常：KeyError。"""
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


class KnowledgeLifecycle:
    """执行文档权威切换与挖掘知识全局去重。"""

    def __init__(
        self,
        encoder,
        model: KnowledgeModel,
        relation_policy: RelationPolicy,
        checkpoint: Callable[[str], None] | None = None,
    ) -> None:
        """保存依赖与故障注入点。

        参数:
            encoder: 具备 revision/count_tokens/encode 的编码器。
            model: 知识关系模型。
            relation_policy: 关系候选阈值与版本。
            checkpoint: 测试/显式恢复演示使用的检查点。
        返回值:
            None
        异常:
            无
        """
        self.encoder = encoder
        self.model = model
        self.relation_policy = relation_policy
        self.checkpoint = checkpoint

    def _checkpoint(self, label: str) -> None:
        """触发检查点。参数：阶段；返回：None；异常：检查点异常。"""
        if self.checkpoint is not None:
            self.checkpoint(label)

    @staticmethod
    def _draft_item(draft: ChunkDraft) -> EvidenceItem:
        """构造文档草案证据。参数：草案；返回：证据；异常：无。"""
        return EvidenceItem(
            ref=f"draft:{draft.source_key}:{draft.ordinal}",
            origin="document",
            text=draft.text,
            source_refs=[draft.source_key],
        )

    @staticmethod
    def _chunk_item(chunk: ChunkSnapshot) -> EvidenceItem:
        """构造已发布知识证据。参数：快照；返回：证据；异常：无。"""
        return EvidenceItem(
            ref=f"chunk:{chunk.id}",
            origin="conversation" if chunk.source_key.startswith(_MINING_PREFIX) else "document",
            text=chunk.text,
            source_refs=[chunk.source_key],
        )

    @staticmethod
    def _staged_item(staged: StagedQA) -> EvidenceItem:
        """构造暂存候选证据。参数：暂存；返回：证据；异常：无。"""
        return EvidenceItem(
            ref=f"staged:{staged.id}",
            origin="conversation",
            text=KnowledgeText(
                category=staged.category,
                questions=staged.question,
                answer=staged.answer,
            ),
            source_refs=[f"qa:{staged.id}"],
        )

    async def _relations(
        self, items: list[EvidenceItem], pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], RelationDecision]:
        """用 Dense 候选阈值筛选后请求关系判定。

        参数:
            items: 证据项。
            pairs: 待筛候选组合。
        返回值:
            dict: pair 到关系结论。
        异常:
            KnowledgeModelError: 模型缺失或多出请求 pair。
        """
        if not pairs:
            return {}
        vectors = await self.encoder.encode([format_knowledge(item.text) for item in items])
        if len(vectors) != len(items):
            raise KnowledgeModelError("关系候选编码数量不一致")
        by_ref = {item.ref: vector for item, vector in zip(items, vectors)}
        candidates = [
            pair
            for pair in pairs
            if _cosine(by_ref[pair[0]], by_ref[pair[1]]) >= self.relation_policy.min_score
        ]
        if not candidates:
            return {}
        mapped: dict[tuple[str, str], RelationDecision] = {}
        # 只把候选对实际引用的证据交给模型，并按小批控制输入预算。
        for start in range(0, len(candidates), 8):
            group = candidates[start : start + 8]
            needed = {ref for pair in group for ref in pair}
            group_items = [item for item in items if item.ref in needed]
            decisions = await self.model.relate(group_items, group)
            group_map = {
                (decision.left_ref, decision.right_ref): decision for decision in decisions
            }
            if set(group_map) != set(group):
                raise KnowledgeModelError("关系判定没有覆盖全部候选 pair")
            mapped.update(group_map)
        return mapped

    def _too_large(self, text: KnowledgeText) -> bool:
        """判断单条候选是否超过三格/token 硬限。参数：三格；返回：是否超限；异常：无。"""
        if len(text.category) + len(text.questions) + len(text.answer) > 1600:
            return True
        count_tokens = getattr(self.encoder, "count_tokens", None)
        if count_tokens is None:
            return False
        return count_tokens(format_knowledge(text)) > 2048

    def _unchanged(self, drafts: list[ChunkDraft], old: list[ChunkSnapshot]) -> list[ChunkSnapshot] | None:
        """判断文档是否完全未变化。参数：草案与旧块；返回：旧快照或 None；异常：无。"""
        if not drafts and not old:
            return []
        old_by_key = {(row.source_key, row.ordinal): row for row in old}
        active_old = [row for row in old if row.is_active]
        if len(active_old) != len(drafts):
            return None
        for draft in drafts:
            row = old_by_key.get((draft.source_key, draft.ordinal))
            if (
                row is None
                or not row.is_active
                or row.text != draft.text
                or row.section_path != draft.section_path
                or row.content_type != draft.content_type
                or row.is_key_clause != draft.is_key_clause
                or row.content_hash != content_fingerprint(draft.text)
                or row.embedding_revision != self.encoder.revision
            ):
                return None
        return sorted(active_old, key=lambda row: (row.source_key, row.ordinal))

    async def publish_documents(
        self,
        writer,
        drafts: list[ChunkDraft],
        source_keys: list[str],
    ) -> list[ChunkSnapshot]:
        """按文档权威原子发布并撤回冲突的已发布挖掘知识。

        参数:
            writer: 持锁 KnowledgeWriter。
            drafts: 本次文档切分草案。
            source_keys: 本次明确更新的文档来源键。
        返回值:
            list[ChunkSnapshot]: 发布或撤回后的快照。
        异常:
            KnowledgeModelError、SnapshotChanged、存储异常。
        """
        frame = await writer.frame()
        old = await writer.chunks_by_sources(source_keys)
        unchanged = self._unchanged(drafts, old)
        if unchanged is not None:
            return unchanged

        active_mined = [
            row
            for row in frame.chunks
            if row.is_active and row.source_key.startswith(_MINING_PREFIX)
        ]
        quarantined = [row for row in frame.staged if row.conflict_group]
        items = (
            [self._draft_item(draft) for draft in drafts]
            + [self._chunk_item(row) for row in active_mined]
            + [self._staged_item(row) for row in quarantined]
        )
        pairs = [
            (item.ref, other.ref)
            for item in items[: len(drafts)]
            for other in items[len(drafts) :]
        ]
        decisions = await self._relations(items, pairs)
        if any(decision.relation == "uncertain" for decision in decisions.values()):
            raise KnowledgeModelError("关系判定不确定，停止文档切换")
        inactive_ids: set[int] = set()
        stage_changes: dict[int, StageChange] = {}
        quarantined_by_id = {row.id: row for row in quarantined}
        for (draft_ref, right_ref), decision in decisions.items():
            if decision.relation not in ("conflict", "same"):
                continue
            reason = "被文档推翻" if decision.relation == "conflict" else "文档重复"
            if right_ref.startswith("chunk:"):
                chunk_id = int(right_ref.removeprefix("chunk:"))
                inactive_ids.add(chunk_id)
                for staged in frame.staged:
                    if staged.knowledge_id == chunk_id and staged.status != "discarded":
                        stage_changes[staged.id] = StageChange(
                            id=staged.id,
                            status="discarded",
                            reason=reason,
                            evidence_refs=[draft_ref, right_ref],
                        )
            elif right_ref.startswith("staged:"):
                staged_id = int(right_ref.removeprefix("staged:"))
                if staged_id in quarantined_by_id:
                    stage_changes[staged_id] = StageChange(
                        id=staged_id,
                        status="discarded",
                        conflict_group=None,
                        reason=reason,
                        evidence_refs=[draft_ref, right_ref],
                    )

        self._checkpoint("before_document_publish")
        published = await writer.publish(
            Publication(
                expected_token=frame.token,
                drafts=list(drafts),
                embedding_revision=self.encoder.revision,
                replace_sources=list(dict.fromkeys(source_keys)),
                inactive_ids=sorted(inactive_ids),
                stage_changes=list(stage_changes.values()),
            )
        )
        self._checkpoint("after_document_publish")
        return published

    @staticmethod
    def _group_id(refs: list[str]) -> str:
        """生成持久冲突组身份。参数：证据引用；返回：组 ID；异常：无。"""
        digest = hashlib.sha256("|".join(sorted(refs)).encode("utf-8")).hexdigest()
        return f"cg:{digest[:16]}"

    def _merged_draft(self, candidates: list[StagedQA]) -> ChunkDraft | None:
        """合并同义候选的真实问法。参数：候选；返回：草案或 None；异常：无。"""
        ordered = sorted(candidates, key=lambda row: row.id)
        questions: list[str] = []
        for candidate in ordered:
            for question in candidate.question.splitlines():
                if question and question not in questions:
                    questions.append(question)
        merged_questions = "\n".join(questions)
        answer, category = ordered[0].answer, ordered[0].category
        if len(category) + len(merged_questions) + len(answer) > 1600:
            return None
        count_tokens = getattr(self.encoder, "count_tokens", None)
        if count_tokens is not None:
            text = KnowledgeText(category=category, questions=merged_questions, answer=answer)
            if count_tokens(format_knowledge(text)) > 2048:
                return None
        return ChunkDraft(
            source_key=f"qa:{ordered[0].id}",
            ordinal=0,
            text=KnowledgeText(category=category, questions=merged_questions, answer=answer),
            section_path=None,
            content_type="faq",
            is_key_clause=False,
        )

    async def reconcile_staging(self, writer) -> list[ChunkSnapshot]:
        """全局去重暂存候选并持久隔离无文档裁定的冲突组。

        参数:
            writer: 持锁 KnowledgeWriter。
        返回值:
            list[ChunkSnapshot]: 本次新发布或撤回的块。
        异常:
            KnowledgeModelError、SnapshotChanged、存储异常。
        """
        frame = await writer.frame()
        raw_candidates = [
            row
            for row in frame.staged
            if row.status == "extracted" and not row.conflict_group
        ]
        if not raw_candidates:
            return []
        oversized_changes: dict[int, StageChange] = {}
        candidates = []
        for row in raw_candidates:
            text = KnowledgeText(
                category=row.category, questions=row.question, answer=row.answer
            )
            if self._too_large(text):
                oversized_changes[row.id] = StageChange(
                    id=row.id,
                    status="extracted",
                    reason="候选超过单块硬限",
                    evidence_refs=[],
                )
            else:
                candidates.append(row)

        active_mined = [
            row
            for row in frame.chunks
            if row.is_active and row.source_key.startswith(_MINING_PREFIX)
        ]
        active_documents = [
            row
            for row in frame.chunks
            if row.is_active and row.source_key.startswith(_DOCUMENT_PREFIX)
        ]
        quarantined = [row for row in frame.staged if row.conflict_group]
        quarantined_by_id = {row.id: row for row in quarantined}
        candidate_by_id = {row.id: row for row in candidates}
        chunk_by_id = {row.id: row for row in active_mined + active_documents}
        items = (
            [self._staged_item(row) for row in candidates]
            + [self._staged_item(row) for row in quarantined]
            + [self._chunk_item(row) for row in active_mined + active_documents]
        )
        pairs: list[tuple[str, str]] = []
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                pairs.append((f"staged:{left.id}", f"staged:{right.id}"))
            for row in quarantined:
                pairs.append((f"staged:{left.id}", f"staged:{row.id}"))
            for chunk in active_mined + active_documents:
                pairs.append((f"staged:{left.id}", f"chunk:{chunk.id}"))
        decisions = await self._relations(items, pairs)

        doc_discard: dict[int, str] = {}
        active_same: dict[int, ChunkSnapshot] = {}
        active_conflict: dict[int, list[ChunkSnapshot]] = {}
        same_edges: list[tuple[int, int]] = []
        conflict_edges: set[tuple[int, int]] = set()
        decision_evidence: dict[int, list[str]] = {}
        quarantine_join: dict[int, str] = {}
        uncertain_candidates: set[int] = set()

        for (left_ref, right_ref), decision in decisions.items():
            if left_ref.startswith("staged:") and right_ref.startswith("staged:"):
                left_id = int(left_ref.removeprefix("staged:"))
                right_id = int(right_ref.removeprefix("staged:"))
                quarantined_id = (
                    right_id if right_id in quarantined_by_id else left_id
                )
                if quarantined_id in quarantined_by_id:
                    if decision.relation in ("same", "conflict"):
                        candidate_id = left_id if quarantined_id == right_id else right_id
                        if candidate_id not in quarantined_by_id:
                            quarantine_join[candidate_id] = quarantined_by_id[
                                quarantined_id
                            ].conflict_group
                    continue
                if decision.relation == "uncertain":
                    uncertain_candidates.update((left_id, right_id))
                elif decision.relation == "same":
                    same_edges.append((left_id, right_id))
                elif decision.relation == "conflict":
                    conflict_edges.add((min(left_id, right_id), max(left_id, right_id)))
            elif left_ref.startswith("staged:") and right_ref.startswith("chunk:"):
                staged_id = int(left_ref.removeprefix("staged:"))
                chunk = chunk_by_id[int(right_ref.removeprefix("chunk:"))]
                if decision.relation == "uncertain":
                    uncertain_candidates.add(staged_id)
                elif chunk.source_key.startswith(_DOCUMENT_PREFIX):
                    if decision.relation in ("conflict", "same"):
                        doc_discard[staged_id] = (
                            "被文档推翻" if decision.relation == "conflict" else "文档重复"
                        )
                elif decision.relation == "conflict":
                    active_conflict.setdefault(staged_id, []).append(chunk)
                elif decision.relation == "same":
                    active_same[staged_id] = chunk
            decision_evidence.setdefault(
                int(left_ref.removeprefix("staged:")), []
            ).extend([left_ref, right_ref])

        excluded = (
            set(doc_discard)
            | set(active_same)
            | set(active_conflict)
            | set(quarantine_join)
            | uncertain_candidates
        )
        dsu = _DisjointSet([row.id for row in candidates if row.id not in excluded])
        for left_id, right_id in same_edges + list(conflict_edges):
            if left_id in dsu.parent and right_id in dsu.parent:
                dsu.union(left_id, right_id)

        drafts: list[ChunkDraft] = []
        stage_changes: dict[int, StageChange] = dict(oversized_changes)
        inactive_ids: set[int] = set()

        for staged_id, reason in doc_discard.items():
            stage_changes[staged_id] = StageChange(
                id=staged_id,
                status="discarded",
                reason=reason,
                evidence_refs=decision_evidence.get(staged_id, []),
            )
        for staged_id, chunk in active_same.items():
            stage_changes[staged_id] = StageChange(
                id=staged_id,
                status="discarded",
                reason="已有同义知识",
                knowledge_source_key=chunk.source_key,
                evidence_refs=decision_evidence.get(staged_id, []),
            )
        for staged_id, group in quarantine_join.items():
            stage_changes[staged_id] = StageChange(
                id=staged_id,
                status="extracted",
                conflict_group=group,
                reason="并入持久隔离组",
                evidence_refs=decision_evidence.get(staged_id, []),
            )
        for staged_id in uncertain_candidates:
            stage_changes[staged_id] = StageChange(
                id=staged_id,
                status="extracted",
                reason="关系不确定，等待权威资料",
                evidence_refs=decision_evidence.get(staged_id, []),
            )
        for staged_id, chunks in active_conflict.items():
            refs = [f"staged:{staged_id}"] + [f"chunk:{chunk.id}" for chunk in chunks]
            group = self._group_id(refs)
            inactive_ids.update(chunk.id for chunk in chunks)
            stage_changes[staged_id] = StageChange(
                id=staged_id,
                status="extracted",
                conflict_group=group,
                reason="无文档裁定的互斥冲突",
                evidence_refs=decision_evidence.get(staged_id, []),
            )
            for chunk in chunks:
                for staged in frame.staged:
                    if (
                        staged.knowledge_id == chunk.id
                        and staged.status in ("kept", "extracted")
                        and not staged.conflict_group
                    ):
                        stage_changes[staged.id] = StageChange(
                            id=staged.id,
                            status="extracted",
                            conflict_group=group,
                            reason="无文档裁定的互斥冲突",
                            evidence_refs=[f"chunk:{chunk.id}"],
                        )

        components: dict[int, list[int]] = {}
        for identifier in dsu.parent:
            components.setdefault(dsu.find(identifier), []).append(identifier)
        for component in components.values():
            component_set = set(component)
            if component_set & uncertain_candidates:
                for identifier in component:
                    stage_changes[identifier] = StageChange(
                        id=identifier,
                        status="extracted",
                        reason="同组关系不确定，等待权威资料",
                        evidence_refs=[f"staged:{identifier}"],
                    )
                continue
            related_active: list[ChunkSnapshot] = []
            for identifier in component:
                related_active.extend(active_conflict.get(identifier, []))
            has_conflict = any(
                left in component_set and right in component_set
                for left, right in conflict_edges
            ) or bool(related_active)
            if has_conflict:
                refs = [f"staged:{identifier}" for identifier in component] + [
                    f"chunk:{chunk.id}" for chunk in related_active
                ]
                group = self._group_id(refs)
                inactive_ids.update(chunk.id for chunk in related_active)
                for identifier in component:
                    stage_changes[identifier] = StageChange(
                        id=identifier,
                        status="extracted",
                        conflict_group=group,
                        reason="无文档裁定的互斥冲突",
                        evidence_refs=[f"staged:{identifier}"]
                        + [f"chunk:{chunk.id}" for chunk in related_active],
                    )
                continue
            if len(component) > 1:
                merged = self._merged_draft([candidate_by_id[identifier] for identifier in component])
                if merged is None:
                    for identifier in component:
                        stage_changes[identifier] = StageChange(
                            id=identifier,
                            status="extracted",
                            reason="合并超过单块硬限",
                            evidence_refs=[f"staged:{identifier}"],
                        )
                    continue
                ordered = sorted(component)
                drafts.append(merged)
                for index, identifier in enumerate(ordered):
                    stage_changes[identifier] = StageChange(
                        id=identifier,
                        status="kept" if index == 0 else "discarded",
                        reason="同义合并",
                        knowledge_source_key=merged.source_key,
                        evidence_refs=[f"staged:{identifier}"],
                    )
            else:
                identifier = component[0]
                candidate = candidate_by_id[identifier]
                draft = ChunkDraft(
                    source_key=f"qa:{identifier}",
                    ordinal=0,
                    text=KnowledgeText(
                        category=candidate.category,
                        questions=candidate.question,
                        answer=candidate.answer,
                    ),
                    section_path=None,
                    content_type="faq",
                    is_key_clause=False,
                )
                drafts.append(draft)
                stage_changes[identifier] = StageChange(
                    id=identifier,
                    status="kept",
                    reason="保留",
                    knowledge_source_key=draft.source_key,
                    evidence_refs=[f"staged:{identifier}"],
                )

        if not drafts and not inactive_ids and not stage_changes:
            return []
        self._checkpoint("before_staging_publish")
        published = await writer.publish(
            Publication(
                expected_token=frame.token,
                drafts=drafts,
                embedding_revision=self.encoder.revision,
                inactive_ids=sorted(inactive_ids),
                stage_changes=list(stage_changes.values()),
            )
        )
        self._checkpoint("after_staging_publish")
        return published
