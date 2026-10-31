"""知识发布生命周期：文档权威、全局去重与持久冲突隔离。

文档重导入以 data/kb 为权威，同事务撤回被推翻的已发布挖掘知识；离线挖掘
先做同义合并，无文档裁定的互斥组整组撤回并写入持久 conflict_group。生命周期
只构造 Publication，不直接写 Milvus；向量同步由调用方在同一 lease 内完成。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.knowledge.contracts import KnowledgeEncoder, KnowledgeModel
from app.knowledge.llm import KnowledgeModelError
from app.knowledge.limits import knowledge_overflow
from app.knowledge.prompts import EXTRACTION_INPUT_BUDGET, relation_input_size
from app.knowledge.relations import relation_components
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
    StoreFrame,
)
from app.knowledge.text import content_fingerprint, format_knowledge

_DOCUMENT_PREFIX = "doc:"
_MINING_PREFIX = "qa:"


@dataclass(frozen=True)
class _StagingRelations:
    """暂存关系判定的完整结果，集中保存组成员、风险与文档审计证据。"""

    components: list[list[str]]  # 包含历史来源约束的稳定引用组。
    risky_refs: set[str]  # 存在冲突或不确定关系的知识引用。
    uncertain_refs: set[str]  # 与权威文档关系仍不确定的知识引用。
    document_coverage: dict[str, list[tuple[ChunkSnapshot, RelationDecision]]]  # 文档对知识的明确裁定。
    evidence_by_ref: dict[str, list[str | dict]]  # 每个引用对应的关系和文档快照审计。


@dataclass(frozen=True)
class _StagingChanges:
    """暂存各组的处置结果，等待发布前检查点通过后组成一次Publication。"""

    drafts: list[ChunkDraft]  # 各组产生的新增或合并草案，保持组处置顺序。
    inactive_ids: set[int]  # 需要撤回的已发布知识主键。
    stage_changes: dict[int, StageChange]  # 暂存主键到最终处置，保持首次加入顺序。


def _cosine(left: list[float], right: list[float]) -> float:
    """计算余弦相似度。参数：向量；返回：分数；异常：无。"""
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class KnowledgeLifecycle:
    """执行文档权威切换与挖掘知识全局去重。"""

    def __init__(
        self,
        encoder: KnowledgeEncoder,
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
        self,
        items: list[EvidenceItem],
        pairs: list[tuple[str, str]],
        *,
        assert_held: Callable[[], Awaitable[None]],
        dense: bool = True,
    ) -> dict[tuple[str, str], RelationDecision]:
        """用 Dense 候选阈值筛选后请求关系判定。

        参数:
            items: 证据项。
            pairs: 待筛候选组合；不得重复或引用不存在的证据。
            assert_held: 当前writer的异步核锁接口；失锁时停止新增外部工作。
            dense: 是否使用Dense阈值；文档相互检查必须为False。
        返回值:
            dict: pair 到关系结论。
        异常:
            KnowledgeModelError: 模型缺失或多出请求 pair。
            LeaseLost、存储或编码异常：原样传播，不能继续后续批次。
        """
        if not pairs:
            return {}
        item_map = {item.ref: item for item in items}
        if len(item_map) != len(items) or len(set(pairs)) != len(pairs):
            raise KnowledgeModelError("关系请求包含重复证据或pair")
        if any(left not in item_map or right not in item_map for left, right in pairs):
            raise KnowledgeModelError("关系请求引用未知证据")
        mapped: dict[tuple[str, str], RelationDecision] = {}
        candidates = []
        for left, right in pairs:
            first, second = item_map[left], item_map[right]
            # 精确QA只规范边界与行尾，不能改变句内数字、空白或Unicode字符。
            if (first.origin == second.origin == "conversation"
                    and self._exact_qa(first.text) == self._exact_qa(second.text)):
                mapped[(left, right)] = RelationDecision(
                    left_ref=left, right_ref=right, relation="same",
                    evidence_refs=[left, right], reason="精确问题与答案一致")
            else:
                candidates.append((left, right))
        if dense and candidates:
            needed = {ref for pair in candidates for ref in pair}
            encoded_items = [item for item in items if item.ref in needed]
            await assert_held()
            vectors = await self.encoder.encode([format_knowledge(item.text) for item in encoded_items])
            # 外部等待期间可能失锁，先保留失锁原因，再校验返回内容。
            await assert_held()
            if len(vectors) != len(encoded_items):
                raise KnowledgeModelError("关系候选编码数量不一致")
            by_ref = {item.ref: vector for item, vector in zip(encoded_items, vectors)}
            candidates = [pair for pair in candidates
                          if _cosine(by_ref[pair[0]], by_ref[pair[1]]) >= self.relation_policy.min_score]

        def group_items(group: list[tuple[str, str]]) -> list[EvidenceItem]:
            """裁剪当前请求证据。参数：group引用对；返回：证据列表；异常：无。"""
            needed = {ref for pair in group for ref in pair}
            return [item for item in items if item.ref in needed]

        groups: list[list[tuple[str, str]]] = []
        current: list[tuple[str, str]] = []
        for pair in candidates:
            trial = current + [pair]
            if len(trial) > 8 or relation_input_size(group_items(trial), trial) > EXTRACTION_INPUT_BUDGET:
                if current:
                    groups.append(current)
                current = [pair]
                if relation_input_size(group_items(current), current) > EXTRACTION_INPUT_BUDGET:
                    raise KnowledgeModelError("单对关系证据超过输入预算")
            else:
                current = trial
        if current:
            groups.append(current)
        # 所有决定收齐且逐条校验后才允许调用者构造一次Publication。
        for group in groups:
            evidence = group_items(group)
            await assert_held()
            decisions = await self.model.relate(evidence, group)
            await assert_held()
            refs = {item.ref for item in evidence}
            group_map = {}
            for decision in decisions:
                pair = (decision.left_ref, decision.right_ref)
                if pair not in group or pair in group_map or not set(decision.evidence_refs).issubset(refs):
                    raise KnowledgeModelError("关系判定包含重复pair或未知引用")
                group_map[pair] = decision
            if len(decisions) != len(group) or set(group_map) != set(group):
                raise KnowledgeModelError("关系判定没有覆盖全部候选 pair")
            mapped.update(group_map)
        return mapped

    @staticmethod
    def _exact_qa(text: KnowledgeText) -> tuple[str, str]:
        """获得精确QA键。参数：text三格；返回：边界规范的问题与答案；异常：无。"""
        return tuple(value.replace("\r\n", "\n").replace("\r", "\n").strip()
                     for value in (text.questions, text.answer))

    def _historical_exact_relations(
        self, candidates: list[StagedQA], staged: list[StagedQA], active: list[ChunkSnapshot]
    ) -> dict[tuple[str, str], RelationDecision]:
        """把历史完整QA的精确重现连接到当前有效知识。

        参数：candidates为本次候选；staged为全部历史暂存审计；active仅含有效挖掘知识块。
        返回：候选与旧知识引用对到same决定的映射，供既有关系组件统一处理。
        异常：Pydantic字段校验失败。
        """
        active_by_id = {row.id: row for row in active}
        exact_chunks: dict[tuple[str, str], set[int]] = {}
        for row in staged:
            chunk = active_by_id.get(row.knowledge_id)
            if chunk is None or row.status not in ("kept", "discarded"):
                continue
            # 历史问法必须取完整原问；旧答案已变化时不能凭历史关联复活过期事实。
            key = self._exact_qa(self._staged_item(row).text)
            if key[1] == self._exact_qa(chunk.text)[1]:
                exact_chunks.setdefault(key, set()).add(chunk.id)
        decisions = {}
        for candidate in candidates:
            key = self._exact_qa(self._staged_item(candidate).text)
            for chunk_id in sorted(exact_chunks.get(key, set())):
                left, right = f"staged:{candidate.id}", f"chunk:{chunk_id}"
                decisions[(left, right)] = RelationDecision(
                    left_ref=left, right_ref=right, relation="same",
                    evidence_refs=[left, right], reason="历史完整问题与当前有效答案精确一致")
        return decisions

    @staticmethod
    def _document_audit(document: ChunkDraft, decision: RelationDecision) -> dict:
        """固定文档裁定时快照。参数：document草案/快照与decision决定；返回：JSON证据；异常：无。"""
        return {"kind": "document", "source_key": document.source_key,
                "ordinal": document.ordinal, "id": getattr(document, "id", None),
                "section_path": document.section_path,
                "content_hash": getattr(document, "content_hash", content_fingerprint(document.text)),
                "text": document.text.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json")}

    def _too_large(self, text: KnowledgeText) -> bool:
        """判断单条候选是否超过三格/token 硬限。参数：三格；返回：是否超限；异常：无。"""
        return knowledge_overflow(text, self.encoder.count_tokens)

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
            KnowledgeModelError、LeaseLost、SnapshotChanged、存储异常。
        """
        frame = await writer.frame()
        old = await writer.chunks_by_sources(source_keys)
        unchanged = self._unchanged(drafts, old)
        if unchanged is not None:
            return unchanged

        # 本次草案替换其来源旧块，其他active文档继续作为权威语料。
        retained_documents = [row for row in frame.chunks
                              if row.is_active and row.source_key.startswith(_DOCUMENT_PREFIX)
                              and row.source_key not in source_keys]
        document_items = ([self._draft_item(draft) for draft in drafts]
                          + [self._chunk_item(row) for row in retained_documents])
        document_pairs = [(left.ref, right.ref)
                          for index, left in enumerate(document_items)
                          for right in document_items[index + 1:]]
        document_decisions = await self._relations(
            document_items, document_pairs, assert_held=writer.assert_held, dense=False
        )
        if any(decision.relation in ("conflict", "uncertain") for decision in document_decisions.values()):
            raise KnowledgeModelError("权威文档存在冲突或不确定，停止文档切换")

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
        decisions = await self._relations(items, pairs, assert_held=writer.assert_held)
        if any(decision.relation == "uncertain" for decision in decisions.values()):
            raise KnowledgeModelError("关系判定不确定，停止文档切换")
        inactive_ids: set[int] = set()
        stage_changes: dict[int, StageChange] = {}
        quarantined_by_id = {row.id: row for row in quarantined}
        documents_by_ref = {self._draft_item(draft).ref: draft for draft in drafts}
        for (draft_ref, right_ref), decision in decisions.items():
            if decision.relation not in ("conflict", "same"):
                continue
            reason = "被文档推翻" if decision.relation == "conflict" else "文档重复"
            if right_ref.startswith("chunk:"):
                chunk_id = int(right_ref.removeprefix("chunk:"))
                inactive_ids.add(chunk_id)
                for staged in frame.staged:
                    if staged.knowledge_id == chunk_id:
                        previous = stage_changes.get(staged.id)
                        stage_changes[staged.id] = StageChange(
                            id=staged.id,
                            status="discarded",
                            reason=reason,
                            evidence_refs=(list(previous.evidence_refs) if previous else [])
                            + [draft_ref, right_ref, self._document_audit(documents_by_ref[draft_ref], decision)],
                        )
            elif right_ref.startswith("staged:"):
                staged_id = int(right_ref.removeprefix("staged:"))
                if staged_id in quarantined_by_id:
                    previous = stage_changes.get(staged_id)
                    stage_changes[staged_id] = StageChange(
                        id=staged_id,
                        status="discarded",
                        conflict_group=None,
                        reason=reason,
                        evidence_refs=(list(previous.evidence_refs) if previous else [])
                        + [draft_ref, right_ref, self._document_audit(documents_by_ref[draft_ref], decision)],
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

    def _merged_draft(
        self, candidates: list[StagedQA], active: list[ChunkSnapshot] | None = None
    ) -> ChunkDraft | None:
        """合并已有知识与新候选的真实问法。

        参数：candidates非空新候选，active按稳定主键排序的旧知识。
        返回：保留旧来源答案的草案；超过三格或token硬限时为None。
        异常：Pydantic字段校验失败。
        """
        ordered = sorted(candidates, key=lambda row: row.id)
        existing = sorted(active or [], key=lambda row: row.id)
        questions: list[str] = []
        texts = [row.text.questions for row in existing] + [row.question for row in ordered]
        for value in texts:
            for question in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                if question and question not in questions:
                    questions.append(question)
        merged_questions = "\n".join(questions)
        answer = existing[0].text.answer if existing else ordered[0].answer
        category = existing[0].text.category if existing else ordered[0].category
        text = KnowledgeText(category=category, questions=merged_questions, answer=answer)
        if self._too_large(text):
            return None
        return ChunkDraft(
            source_key=existing[0].source_key if existing else f"qa:{ordered[0].id}",
            ordinal=existing[0].ordinal if existing else 0,
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
            KnowledgeModelError、LeaseLost、SnapshotChanged、存储异常。
        """
        frame = await writer.frame()
        raw_candidates = [
            row
            for row in frame.staged
            if row.status == "extracted" and not row.conflict_group
        ]
        if not raw_candidates:
            return []

        candidates, oversized_changes = self._prepare_staging_candidates(raw_candidates)
        relations = await self._staging_relations(frame, candidates, assert_held=writer.assert_held)
        changes = self._staging_group_changes(
            frame, candidates, relations, oversized_changes
        )
        if not changes.drafts and not changes.inactive_ids and not changes.stage_changes:
            return []

        self._checkpoint("before_staging_publish")
        published = await writer.publish(
            Publication(
                expected_token=frame.token,
                drafts=changes.drafts,
                embedding_revision=self.encoder.revision,
                inactive_ids=sorted(changes.inactive_ids),
                stage_changes=list(changes.stage_changes.values()),
            )
        )
        self._checkpoint("after_staging_publish")
        return published

    def _prepare_staging_candidates(
        self, raw_candidates: list[StagedQA]
    ) -> tuple[list[StagedQA], dict[int, StageChange]]:
        """筛出可参与关系判定的候选，并保留超限候选的审计处置。

        参数：raw_candidates为尚未隔离、状态为extracted的暂存记录，保持读取顺序。
        返回：合规候选列表，以及按暂存主键索引的超限处置。
        异常：Pydantic字段校验失败或编码器计数异常。
        """
        oversized_changes: dict[int, StageChange] = {}
        candidates: list[StagedQA] = []
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
        return candidates, oversized_changes

    async def _staging_relations(
        self,
        frame: StoreFrame,
        candidates: list[StagedQA],
        *,
        assert_held: Callable[[], Awaitable[None]],
    ) -> _StagingRelations:
        """收齐候选关系，并补判候选触达的历史知识与权威文档的关系。

        参数：frame为固定发布快照；candidates为已通过单块硬限的候选，允许为空；
        assert_held为当前writer的异步核锁接口，覆盖候选判定与历史知识补判。
        返回：完整关系组、风险引用及可用于整组处置的文档审计证据。
        异常：KnowledgeModelError、LeaseLost、存储/编码器异常或关系引用不存在时的KeyError。
        """
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
        decisions = self._historical_exact_relations(candidates, frame.staged, active_mined)
        # 历史精确匹配先于Dense/模型；仍进入同一组件、文档裁定与来源发布门禁。
        decisions.update(await self._relations(
            items, [pair for pair in pairs if pair not in decisions], assert_held=assert_held
        ))

        # 稳定ref图包含所有暂存审计来源，不能先排除存在风险的节点。
        staged_by_id = {row.id: row for row in frame.staged}
        qa_chunks = {row.id: row for row in frame.chunks if row.source_key.startswith(_MINING_PREFIX)}
        qa_decisions = [
            decision for decision in decisions.values()
            if not (
                decision.right_ref.startswith("chunk:")
                and chunk_by_id[int(decision.right_ref.removeprefix("chunk:"))].source_key.startswith(
                    _DOCUMENT_PREFIX
                )
            )
        ]
        components = relation_components(frame.staged, list(qa_chunks.values()), qa_decisions)
        candidate_refs = {f"staged:{row.id}" for row in candidates}
        reached = {
            ref for members in components
            if set(members) & candidate_refs
            for ref in members
        }
        # 只对新候选触达的实际旧知识补文档裁定，关联来源审计由其知识块承接。
        authority_items = []
        for ref in sorted(reached - candidate_refs):
            if ref.startswith("chunk:"):
                authority_items.append(self._chunk_item(qa_chunks[int(ref.removeprefix("chunk:"))]))
            else:
                row = staged_by_id[int(ref.removeprefix("staged:"))]
                if row.knowledge_id not in qa_chunks:
                    authority_items.append(self._staged_item(row))
        document_items = [self._chunk_item(row) for row in active_documents]
        authority_pairs = [
            (item.ref, document.ref)
            for item in authority_items
            for document in document_items
        ]
        decisions.update(await self._relations(
            authority_items + document_items, authority_pairs, assert_held=assert_held
        ))
        return self._summarize_staging_relations(components, decisions, chunk_by_id)

    def _summarize_staging_relations(
        self,
        components: list[list[str]],
        decisions: dict[tuple[str, str], RelationDecision],
        chunk_by_id: dict[int, ChunkSnapshot],
    ) -> _StagingRelations:
        """按知识引用归集风险、文档覆盖和审计证据，不改变关系组。

        参数：components为QA关系组；decisions为收齐且保持判定顺序的关系；
        chunk_by_id为候选判定所使用的有效挖掘和文档知识。
        返回：供组处置使用的关系结果；文档裁定仅形成覆盖证据，不连接组。
        异常：决定引用不存在的知识块时抛出KeyError。
        """
        risky_refs: set[str] = set()
        uncertain_refs: set[str] = set()
        document_coverage: dict[str, list[tuple[ChunkSnapshot, RelationDecision]]] = {}
        evidence_by_ref: dict[str, list[str | dict]] = {}
        for (left, right), decision in decisions.items():
            if (
                right.startswith("chunk:")
                and chunk_by_id[int(right.removeprefix("chunk:"))].source_key.startswith(_DOCUMENT_PREFIX)
            ):
                document = chunk_by_id[int(right.removeprefix("chunk:"))]
                if decision.relation in ("same", "conflict"):
                    document_coverage.setdefault(left, []).append((document, decision))
                    if decision.relation == "conflict":
                        risky_refs.add(left)
                elif decision.relation == "uncertain":
                    risky_refs.add(left)
                    uncertain_refs.add(left)
                evidence_by_ref.setdefault(left, []).extend([left, right, self._document_audit(document, decision)])
                continue
            if decision.relation in ("conflict", "uncertain"):
                risky_refs.update((left, right))
            for ref in (left, right):
                evidence_by_ref.setdefault(ref, []).extend([left, right])

        return _StagingRelations(
            components=components,
            risky_refs=risky_refs,
            uncertain_refs=uncertain_refs,
            document_coverage=document_coverage,
            evidence_by_ref=evidence_by_ref,
        )

    def _staging_group_changes(
        self,
        frame: StoreFrame,
        candidates: list[StagedQA],
        relations: _StagingRelations,
        oversized_changes: dict[int, StageChange],
    ) -> _StagingChanges:
        """依次按文档裁定、风险隔离、同义合并处置各组。

        参数：frame为固定发布快照；candidates为合规候选；relations为完整关系结果；
        oversized_changes为候选筛选时保留的超限审计处置，不在本方法中修改。
        返回：按原有顺序累积的草案、撤回主键与暂存处置，允许全部为空。
        异常：Pydantic字段校验失败、编码器计数异常或组引用不存在时的KeyError。
        """
        candidate_by_id = {row.id: row for row in candidates}
        candidate_refs = {f"staged:{row.id}" for row in candidates}
        staged_by_id = {row.id: row for row in frame.staged}
        qa_chunks = {
            row.id: row for row in frame.chunks
            if row.source_key.startswith(_MINING_PREFIX)
        }
        drafts: list[ChunkDraft] = []
        stage_changes: dict[int, StageChange] = dict(oversized_changes)
        inactive_ids: set[int] = set()
        for component in relations.components:
            member_refs = set(component)
            new_refs = member_refs & candidate_refs
            if not new_refs:
                continue
            member_staged = sorted(
                [staged_by_id[int(ref.removeprefix("staged:"))]
                 for ref in component if ref.startswith("staged:")],
                key=lambda row: row.id,
            )
            related_active = sorted(
                [qa_chunks[int(ref.removeprefix("chunk:"))]
                 for ref in component if ref.startswith("chunk:")
                 and qa_chunks[int(ref.removeprefix("chunk:"))].is_active],
                key=lambda row: row.id,
            )
            existing_groups = {row.conflict_group for row in member_staged if row.conflict_group}
            evidence: list[str | dict] = list(sorted(member_refs))
            for ref in sorted(member_refs):
                evidence.extend(relations.evidence_by_ref.get(ref, []))
            has_risk = bool(member_refs & relations.risky_refs or existing_groups)
            # 文档不作为图边；只有覆盖整组实际知识的决定才能整体处置。
            actual_refs = {ref for ref in member_refs if ref.startswith("chunk:")}
            actual_refs.update(
                f"staged:{row.id}" for row in member_staged
                if row.knowledge_id not in qa_chunks
            )
            covered = actual_refs.issubset(relations.document_coverage) and not bool(
                member_refs & relations.uncertain_refs
            )
            if covered:
                inactive_ids.update(row.id for row in related_active)
                for row in member_staged:
                    ref = (
                        f"chunk:{row.knowledge_id}" if row.knowledge_id in qa_chunks
                        else f"staged:{row.id}"
                    )
                    choices = relations.document_coverage[ref]
                    reason = (
                        "被文档推翻" if any(decision.relation == "conflict" for _, decision in choices)
                        else "文档重复"
                    )
                    stage_changes[row.id] = StageChange(
                        id=row.id, status="discarded", reason=reason, evidence_refs=evidence
                    )
                continue
            if has_risk:
                # 单个历史组保留原身份；桥接多个旧组时全部成员获得统一新身份。
                group = next(iter(existing_groups)) if len(existing_groups) == 1 else self._group_id(component)
                inactive_ids.update(row.id for row in related_active)
                for row in member_staged:
                    stage_changes[row.id] = StageChange(
                        id=row.id, status="extracted", conflict_group=group,
                        reason="关系组存在冲突或不确定，等待权威资料", evidence_refs=evidence)
                continue
            new_candidates = sorted(
                [candidate_by_id[int(ref.removeprefix("staged:"))] for ref in new_refs],
                key=lambda row: row.id,
            )
            merged = self._merged_draft(new_candidates, related_active)
            if merged is None:
                for row in new_candidates:
                    stage_changes[row.id] = StageChange(
                        id=row.id, status="extracted", reason="合并超过单块硬限", evidence_refs=evidence)
                continue
            drafts.append(merged)
            if related_active:
                primary = related_active[0]
                inactive_ids.update(row.id for row in related_active[1:])
                for row in member_staged:
                    status = row.status if row not in new_candidates else "discarded"
                    stage_changes[row.id] = StageChange(
                        id=row.id, status=status, reason="同义合并至已有知识",
                        knowledge_source_key=primary.source_key, evidence_refs=evidence)
            else:
                for row in new_candidates:
                    stage_changes[row.id] = StageChange(
                        id=row.id, status="kept" if row.id == new_candidates[0].id else "discarded",
                        reason="同义合并" if len(new_candidates) > 1 else "保留",
                        knowledge_source_key=merged.source_key, evidence_refs=evidence)

        return _StagingChanges(
            drafts=drafts,
            inactive_ids=inactive_ids,
            stage_changes=stage_changes,
        )
