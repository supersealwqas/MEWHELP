"""无文档冲突隔离、持久组身份与不自动解封测试。"""

from __future__ import annotations

from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.prompts import RELATION_PROMPT_VERSION
from app.knowledge.schemas import (
    ChunkDraft,
    ChunkSnapshot,
    KnowledgeText,
    RelationDecision,
    RelationPolicy,
    StagedQA,
)
from app.knowledge.text import content_fingerprint
from tests.knowledge.fakes import (
    FakeEncoder,
    MemoryKnowledgeStore,
    ScriptedKnowledgeModel,
)

POLICY = RelationPolicy(
    min_score=0.0,
    embedding_revision="fake-encoder-v1",
    prompt_version=RELATION_PROMPT_VERSION,
    calibration_digest="test-only",
)


def _text(answer: str = "知识答案。", questions: str = "问法") -> KnowledgeText:
    """构造三格文本。参数：答案/问法；返回：文本；异常：校验失败。"""
    return KnowledgeText(category="物流", questions=questions, answer=answer)


def _draft(answer: str = "新文档知识。") -> ChunkDraft:
    """构造文档草案。参数：答案；返回：草案；异常：校验失败。"""
    return ChunkDraft(
        source_key="doc:new.md",
        ordinal=0,
        text=_text(answer=answer, questions="文档问法"),
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
    )


def _snapshot(identifier: int, text: KnowledgeText) -> ChunkSnapshot:
    """构造已发布挖掘快照。参数：主键/文本；返回：快照；异常：校验失败。"""
    return ChunkSnapshot(
        id=identifier,
        source_key=f"qa:{identifier}",
        ordinal=0,
        text=text,
        section_path=None,
        content_type="faq",
        is_key_clause=False,
        content_hash=content_fingerprint(text),
        embedding_revision="fake-encoder-v1",
        is_active=True,
        vector_id=str(identifier),
        vectorize_status="done",
    )


def _staged(
    identifier: int,
    question: str,
    *,
    answer: str = "未满99元收6元。",
    status: str = "extracted",
    knowledge_id: int | None = None,
    conflict_group: str | None = None,
) -> StagedQA:
    """构造暂存候选。参数：主键/问题/答案/状态；返回：暂存；异常：校验失败。"""
    return StagedQA(
        id=identifier,
        source_ids=[identifier],
        question=question,
        answer=answer,
        category="物流",
        status=status,
        knowledge_id=knowledge_id,
        conflict_group=conflict_group,
    )


def _decision(left: str, right: str, relation: str) -> RelationDecision:
    """构造关系决策。参数：引用与关系；返回：决策；异常：校验失败。"""
    return RelationDecision(
        left_ref=left,
        right_ref=right,
        relation=relation,
        evidence_refs=[left, right],
        reason="测试冲突",
    )


def _lifecycle(model) -> KnowledgeLifecycle:
    """构造生命周期。参数：模型；返回：生命周期；异常：无。"""
    return KnowledgeLifecycle(FakeEncoder(), model, POLICY)


async def _frame(store):
    """读取帧。参数：存储；返回：StoreFrame；异常：无。"""
    async with store.writer() as writer:
        return await writer.frame()


async def test_undocumented_conflict_withdraws_entire_group() -> None:
    """无文档裁定的互斥候选整组回到 extracted 并写持久组身份。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore(
        staged=[
            _staged(1, "偏远地区包邮吗", answer="偏远地区包邮。"),
            _staged(2, "偏远地区包邮吗", answer="偏远地区不包邮。"),
        ]
    )
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("staged:1", "staged:2", "conflict")]
    )
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.chunks == []
    groups = {row.conflict_group for row in frame.staged}
    assert len(groups) == 1 and next(iter(groups))
    assert all(row.status == "extracted" for row in frame.staged)


async def test_published_member_withdrawn_into_group() -> None:
    """已发布冲突成员被撤回，暂存候选保持隔离。参数：无；返回：None；异常：无。"""
    chunk = _snapshot(10, _text(answer="偏远地区包邮。"))
    store = MemoryKnowledgeStore(
        snapshots=[chunk],
        staged=[
            _staged(
                1,
                "偏远地区包邮吗",
                answer="偏远地区不包邮。",
                knowledge_id=10,
            )
        ],
    )
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("staged:1", "chunk:10", "conflict")]
    )
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    published = next(row for row in frame.chunks if row.id == 10)
    staged = next(row for row in frame.staged if row.id == 1)
    assert published.is_active is False
    assert staged.status == "extracted" and staged.conflict_group


async def test_one_sided_rerun_and_new_alias_cannot_unquarantine() -> None:
    """单侧重跑或新别名不能解封已有冲突组。参数：无；返回：None；异常：无。"""
    store = MemoryKnowledgeStore(
        staged=[
            _staged(1, "偏远地区包邮吗", answer="偏远地区包邮。", conflict_group="cg:test"),
            _staged(2, "偏远地区包邮吗", answer="偏远地区不包邮。", conflict_group="cg:test"),
        ]
    )
    store._staged[3] = _staged(3, "偏远地区免运费吗", answer="偏远地区包邮。")
    model = ScriptedKnowledgeModel(
        relation_decisions=[
            _decision("staged:3", "staged:1", "same"),
            _decision("staged:3", "staged:2", "unrelated"),
        ]
    )
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    alias = next(row for row in frame.staged if row.id == 3)
    assert alias.status == "extracted"
    assert alias.conflict_group == "cg:test"
    assert not frame.chunks

    model_after = ScriptedKnowledgeModel()
    async with store.writer() as writer:
        await _lifecycle(model_after).reconcile_staging(writer)
    assert model_after.relation_inputs == []
    assert (await _frame(store)).staged == frame.staged


async def test_uncertain_candidate_is_not_published() -> None:
    """关系不确定的候选留在 extracted，不发布。参数：无；返回：None；异常：无。"""
    store = MemoryKnowledgeStore(
        staged=[
            _staged(1, "模棱两可的问题", answer="答案一。"),
            _staged(2, "模棱两可的问题", answer="答案二。"),
        ]
    )
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("staged:1", "staged:2", "uncertain")]
    )
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.chunks == []
    assert all(row.status == "extracted" for row in frame.staged)


async def test_published_conflict_requarantines_kept_staging() -> None:
    """已发布 QA 被新候选推翻时，原 kept 暂存同事务回到隔离组。参数：无；返回：None；异常：无。"""
    chunk = _snapshot(10, _text(answer="偏远地区包邮。"))
    store = MemoryKnowledgeStore(
        snapshots=[chunk],
        staged=[
            _staged(1, "偏远地区包邮吗", answer="偏远地区包邮。", status="kept", knowledge_id=10),
            _staged(2, "偏远地区包邮吗", answer="偏远地区不包邮。"),
        ],
    )
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("staged:2", "chunk:10", "conflict")]
    )
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    published = next(row for row in frame.chunks if row.id == 10)
    kept = next(row for row in frame.staged if row.id == 1)
    new = next(row for row in frame.staged if row.id == 2)
    assert published.is_active is False
    assert kept.status == "extracted" and kept.conflict_group
    assert new.status == "extracted" and new.conflict_group == kept.conflict_group


async def test_document_not_covering_group_does_not_release_it() -> None:
    """文档不覆盖隔离组时不能解封。参数：无；返回：None；异常：无。"""
    store = MemoryKnowledgeStore(
        staged=[
            _staged(1, "偏远地区包邮吗", answer="偏远地区包邮。", conflict_group="cg:test"),
            _staged(2, "偏远地区包邮吗", answer="偏远地区不包邮。", conflict_group="cg:test"),
        ]
    )
    model = ScriptedKnowledgeModel(
        relation_decisions=[
            _decision("draft:doc:new.md:0", "staged:1", "unrelated"),
            _decision("draft:doc:new.md:0", "staged:2", "unrelated"),
        ]
    )
    async with store.writer() as writer:
        await _lifecycle(model).publish_documents(writer, [_draft()], ["doc:new.md"])
    frame = await _frame(store)
    assert any(row.source_key == "doc:new.md" and row.is_active for row in frame.chunks)
    assert all(row.conflict_group == "cg:test" for row in frame.staged)
    assert model.relation_inputs


async def test_document_can_discard_covered_quarantined_member() -> None:
    """文档能裁定覆盖到的隔离成员，但未覆盖成员继续隔离。参数：无；返回：None；异常：无。"""
    store = MemoryKnowledgeStore(
        staged=[
            _staged(1, "偏远地区包邮吗", answer="偏远地区包邮。", conflict_group="cg:test"),
            _staged(2, "偏远地区包邮吗", answer="偏远地区不包邮。", conflict_group="cg:test"),
        ]
    )
    model = ScriptedKnowledgeModel(
        relation_decisions=[
            _decision("draft:doc:new.md:0", "staged:1", "conflict"),
            _decision("draft:doc:new.md:0", "staged:2", "unrelated"),
        ]
    )
    async with store.writer() as writer:
        await _lifecycle(model).publish_documents(writer, [_draft()], ["doc:new.md"])
    frame = await _frame(store)
    covered = next(row for row in frame.staged if row.id == 1)
    uncovered = next(row for row in frame.staged if row.id == 2)
    assert covered.status == "discarded" and covered.conflict_group is None
    assert uncovered.status == "extracted" and uncovered.conflict_group == "cg:test"
