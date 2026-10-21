"""知识生命周期：文档权威、同义合并与原子发布测试。"""

from __future__ import annotations

import pytest

from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.llm import KnowledgeModelError
from app.knowledge.prompts import RELATION_PROMPT_VERSION, relation_input_size
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
    MappedKnowledgeModel,
    source_backed_store as MemoryKnowledgeStore,
    ScriptedKnowledgeModel,
)

POLICY = RelationPolicy(
    min_score=0.0,
    embedding_revision="fake-encoder-v1",
    prompt_version=RELATION_PROMPT_VERSION,
    calibration_digest="test-only",
)


def _text(answer: str = "知识答案。", category: str = "分类", questions: str = "问法") -> KnowledgeText:
    """构造三格文本。参数：答案/分类/问法；返回：文本；异常：校验失败。"""
    return KnowledgeText(category=category, questions=questions, answer=answer)


def _draft(
    *,
    source_key: str = "doc:a.md",
    ordinal: int = 0,
    answer: str = "知识答案。",
    section_path: str | None = "甲",
) -> ChunkDraft:
    """构造文档草案。参数：来源/序号/答案/章节；返回：草案；异常：校验失败。"""
    return ChunkDraft(
        source_key=source_key,
        ordinal=ordinal,
        text=_text(answer=answer),
        section_path=section_path,
        content_type="faq",
        is_key_clause=False,
    )


def _snapshot(
    identifier: int,
    source_key: str,
    text: KnowledgeText,
    *,
    active: bool = True,
    status: str = "done",
    vector_id: str | None = None,
) -> ChunkSnapshot:
    """构造已发布快照。参数：主键/来源/文本/状态；返回：快照；异常：校验失败。"""
    return ChunkSnapshot(
        id=identifier,
        source_key=source_key,
        ordinal=0,
        text=text,
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
        content_hash=content_fingerprint(text),
        embedding_revision="fake-encoder-v1",
        is_active=active,
        vector_id=vector_id or (str(identifier) if status == "done" else None),
        vectorize_status=status,
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
        reason="测试关系",
    )


def _lifecycle(model, checkpoint=None) -> KnowledgeLifecycle:
    """构造 FakeEncoder 生命周期。参数：模型/检查点；返回：生命周期；异常：无。"""
    return KnowledgeLifecycle(FakeEncoder(), model, POLICY, checkpoint)


async def _frame(store) -> object:
    """读取内存存储当前帧。参数：存储；返回：StoreFrame；异常：无。"""
    async with store.writer() as writer:
        return await writer.frame()


async def test_unchanged_document_reimport_is_noop() -> None:
    """无变化重导不调用模型也不改变状态。参数：无；返回：None；异常：无。"""
    draft = _draft()
    snapshot = _snapshot(1, "doc:a.md", draft.text)
    store = MemoryKnowledgeStore(snapshots=[snapshot])
    model = ScriptedKnowledgeModel()
    before = await _frame(store)
    async with store.writer() as writer:
        result = await _lifecycle(model).publish_documents(
            writer, [draft], ["doc:a.md"]
        )
    assert [row.id for row in result] == [1]
    assert model.relation_inputs == []
    assert await _frame(store) == before


async def test_document_overrules_already_published_qa_atomically() -> None:
    """文档冲突与旧 QA 撤回、暂存 discarded 同事务提交。参数：无；返回：None；异常：无。"""
    chunk = _snapshot(10, "qa:1", _text(answer="旧知识。", questions="旧问法"))
    staged = _staged(1, "旧问法", answer="旧知识。", status="kept", knowledge_id=10)
    store = MemoryKnowledgeStore(snapshots=[chunk], staged=[staged])
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("draft:doc:a.md:0", "chunk:10", "conflict")]
    )
    async with store.writer() as writer:
        await _lifecycle(model).publish_documents(
            writer, [_draft(answer="新文档知识。")], ["doc:a.md"]
        )
    frame = await _frame(store)
    old = next(row for row in frame.chunks if row.id == 10)
    document = next(row for row in frame.chunks if row.source_key == "doc:a.md")
    rejected = next(row for row in frame.staged if row.id == 1)
    assert old.is_active is False and old.text.answer == "旧知识。"
    assert document.is_active is True
    assert rejected.status == "discarded"


async def test_document_judgment_failure_publishes_nothing() -> None:
    """关系判定失败时没有任何部分发布。参数：无；返回：None；异常：无。"""
    chunk = _snapshot(10, "qa:1", _text(answer="旧知识。", questions="旧问法"))
    store = MemoryKnowledgeStore(snapshots=[chunk])
    model = ScriptedKnowledgeModel(
        relation_error=KnowledgeModelError("模拟关系失败")
    )
    before = await _frame(store)
    with pytest.raises(KnowledgeModelError):
        async with store.writer() as writer:
            await _lifecycle(model).publish_documents(
                writer, [_draft(answer="新文档知识。")], ["doc:a.md"]
            )
    assert await _frame(store) == before


async def test_publish_checkpoint_failure_preserves_recoverable_state() -> None:
    """发布前检查点失败不改变状态。参数：无；返回：None；异常：无。"""
    chunk = _snapshot(10, "qa:1", _text(answer="旧知识。", questions="旧问法"))
    store = MemoryKnowledgeStore(snapshots=[chunk])

    def checkpoint(label: str) -> None:
        """在文档发布前模拟中断。参数：阶段；返回：None；异常：RuntimeError。"""
        if label == "before_document_publish":
            raise RuntimeError("模拟发布前中断")

    before = await _frame(store)
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("draft:doc:a.md:0", "chunk:10", "conflict")]
    )
    with pytest.raises(RuntimeError):
        async with store.writer() as writer:
            await _lifecycle(model, checkpoint).publish_documents(
                writer, [_draft(answer="新文档知识。")], ["doc:a.md"]
            )
    assert await _frame(store) == before


async def test_metadata_not_used_as_embedding_text() -> None:
    """关系候选编码只使用三格，不包含章节元数据。参数：无；返回：None；异常：无。"""
    chunk = _snapshot(10, "qa:1", _text(answer="旧知识。", questions="旧问法"))
    store = MemoryKnowledgeStore(snapshots=[chunk])
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("draft:doc:a.md:0", "chunk:10", "unrelated")]
    )
    encoder = FakeEncoder()
    lifecycle = KnowledgeLifecycle(encoder, model, POLICY)
    async with store.writer() as writer:
        await lifecycle.publish_documents(
            writer,
            [_draft(answer="新文档知识。", section_path="不该出现")],
            ["doc:a.md"],
        )
    assert encoder.inputs
    assert all("不该出现" not in text for text in encoder.inputs)
    assert all(text.startswith("分类：") for text in encoder.inputs)


async def test_cross_batch_synonyms_merge_real_questions_only() -> None:
    """跨批同义候选只合并真实问法，不生成调和问题。参数：无；返回：None；异常：无。"""
    store = MemoryKnowledgeStore(
        staged=[
            _staged(1, "邮费是多少"),
            _staged(2, "运费怎么算"),
        ]
    )
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("staged:1", "staged:2", "same")]
    )
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    merged = next(row for row in frame.chunks if row.source_key == "qa:1")
    assert set(merged.text.questions.splitlines()) == {"邮费是多少", "运费怎么算"}
    assert merged.text.answer == "未满99元收6元。"
    statuses = {row.id: row.status for row in frame.staged}
    assert statuses == {1: "kept", 2: "discarded"}
    assert not any(row.source_key == "qa:2" for row in frame.chunks)


async def test_document_uncertain_stops_switch() -> None:
    """文档关系不确定时不得切换旧知识。参数：无；返回：None；异常：无。"""
    chunk = _snapshot(10, "qa:1", _text(answer="旧知识。", questions="旧问法"))
    store = MemoryKnowledgeStore(snapshots=[chunk])
    model = ScriptedKnowledgeModel(
        relation_decisions=[_decision("draft:doc:a.md:0", "chunk:10", "uncertain")]
    )
    before = await _frame(store)
    with pytest.raises(KnowledgeModelError):
        async with store.writer() as writer:
            await _lifecycle(model).publish_documents(
                writer, [_draft(answer="新文档知识。")], ["doc:a.md"]
            )
    assert await _frame(store) == before


class _PairModel:
    """按请求 pair 返回无关关系并记录输入证据。"""

    def __init__(self) -> None:
        """初始化记录。参数：无；返回：None；异常：无。"""
        self.calls: list[tuple[list, list]] = []

    async def relate(self, items, pairs):
        """按请求 pair 返回 unrelated。参数：证据/pair；返回：决策；异常：无。"""
        self.calls.append((list(items), list(pairs)))
        return [_decision(left, right, "unrelated") for left, right in pairs]


async def test_relation_model_receives_only_pair_items() -> None:
    """大批已发布知识只按候选对裁剪证据，并分批调用模型。参数：无；返回：None；异常：无。"""
    snapshots = [
        _snapshot(identifier, f"qa:{identifier}", _text(answer=f"旧知识{identifier}。"))
        for identifier in range(100, 125)
    ]
    store = MemoryKnowledgeStore(snapshots=snapshots)
    model = _PairModel()
    async with store.writer() as writer:
        await _lifecycle(model).publish_documents(
            writer, [_draft(answer="新文档知识。")], ["doc:a.md"]
        )
    assert model.calls
    assert all(len(items) <= 9 for items, _ in model.calls)
    assert all(len(pairs) <= 8 for _, pairs in model.calls)
    assert sum(len(pairs) for _, pairs in model.calls) == 25


async def test_oversized_candidate_not_published() -> None:
    """单条候选超过硬限时保留 extracted 且不发布。参数：无；返回：None；异常：无。"""
    store = MemoryKnowledgeStore(
        staged=[_staged(1, "超长问法", answer="长" * 1601)]
    )
    async with store.writer() as writer:
        await _lifecycle(ScriptedKnowledgeModel()).reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.chunks == []
    assert next(row for row in frame.staged if row.id == 1).status == "extracted"


@pytest.mark.parametrize("relations", [
    {("staged:1", "staged:2"): "conflict", ("staged:2", "staged:3"): "uncertain"},
    {("staged:1", "staged:2"): "same", ("staged:2", "staged:3"): "conflict"},
])
async def test_constraint_chain_quarantines_all_members(relations) -> None:
    """冲突或不确定通过整条关系链传播。参数：relations链关系；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(i, f"问题{i}", answer=f"答案{i}。") for i in (1, 2, 3)])
    async with store.writer() as writer:
        await _lifecycle(MappedKnowledgeModel(relations)).reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.chunks == []
    assert len({row.conflict_group for row in frame.staged}) == 1
    assert all(row.conflict_group and row.status == "extracted" for row in frame.staged)


async def test_uncertain_alias_joins_old_quarantine() -> None:
    """新不确定成员不能绕过持久隔离组。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(1, "旧一", conflict_group="cg:old"),
                                        _staged(2, "旧二", conflict_group="cg:old"),
                                        _staged(3, "新")])
    async with store.writer() as writer:
        await _lifecycle(MappedKnowledgeModel({("staged:3", "staged:1"): "uncertain"})).reconcile_staging(writer)
    frame = await _frame(store)
    assert {row.conflict_group for row in frame.staged} == {"cg:old"}
    assert frame.chunks == []


async def test_alias_same_active_and_conflict_with_other_withdraws_all() -> None:
    """已有active不能切断same接conflict关系。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(10, "qa:1", _text(answer="旧答", questions="旧问"))],
                                 staged=[_staged(1, "旧问", answer="旧答", status="kept", knowledge_id=10),
                                         _staged(2, "新问", answer="旧答"), _staged(3, "相反", answer="另一答")])
    model = MappedKnowledgeModel({("staged:2", "chunk:10"): "same", ("staged:2", "staged:3"): "conflict"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert all(not row.is_active for row in frame.chunks)
    assert len({row.conflict_group for row in frame.staged}) == 1
    assert all(row.conflict_group and row.status == "extracted" for row in frame.staged)


async def test_new_bridge_unifies_every_old_group_member() -> None:
    """新成员桥接两个旧组时旧组全部身份统一。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(1, "甲一", conflict_group="cg:a"),
                                        _staged(2, "甲二", conflict_group="cg:a"),
                                        _staged(3, "乙一", conflict_group="cg:b"),
                                        _staged(4, "乙二", conflict_group="cg:b"), _staged(5, "新问")])
    async with store.writer() as writer:
        await _lifecycle(MappedKnowledgeModel({("staged:5", "staged:1"): "same", ("staged:5", "staged:3"): "uncertain"})).reconcile_staging(writer)
    frame = await _frame(store)
    assert len({row.conflict_group for row in frame.staged}) == 1
    assert all(row.conflict_group for row in frame.staged)


async def test_same_chain_merges_multiple_active_and_relinks_discarded_sources() -> None:
    """same组保留稳定主键、全部真实问法并撤回重关联重复块。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(10, "qa:1", _text(answer="可信答", questions="旧问一\n旧问二")),
                                          _snapshot(20, "qa:3", _text(answer="可信答", questions="旧问三"))],
                                 staged=[_staged(1, "旧问一", answer="可信答", status="kept", knowledge_id=10),
                                         _staged(2, "旧问二", answer="可信答", status="discarded", knowledge_id=10),
                                         _staged(3, "旧问三", answer="可信答", status="kept", knowledge_id=20),
                                         _staged(4, "新问四", answer="可信答")])
    model = MappedKnowledgeModel({("staged:4", "chunk:10"): "same", ("staged:4", "chunk:20"): "same"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    main = next(row for row in frame.chunks if row.id == 10)
    assert main.source_key == "qa:1" and main.vectorize_status == "pending" and main.vector_id is None
    assert main.text.questions.splitlines() == ["旧问一", "旧问二", "旧问三", "新问四"]
    assert next(row for row in frame.chunks if row.id == 20).is_active is False
    assert all(row.knowledge_id == 10 for row in frame.staged)


async def test_cross_batch_same_overflow_keeps_original_without_truncating() -> None:
    """合并超限不截断旧块或新问法。参数：无；返回：None；异常：断言失败。"""
    old = _snapshot(10, "qa:1", _text(answer="答。", questions="旧" * 900))
    store = MemoryKnowledgeStore(snapshots=[old], staged=[_staged(1, "旧" * 900, answer="答。", status="kept", knowledge_id=10),
                                                         _staged(2, "新" * 900, answer="答。")])
    async with store.writer() as writer:
        await _lifecycle(MappedKnowledgeModel({("staged:2", "chunk:10"): "same"})).reconcile_staging(writer)
    frame = await _frame(store)
    assert next(row for row in frame.chunks if row.id == 10) == old
    assert next(row for row in frame.staged if row.id == 2).status == "extracted"


async def test_exact_duplicate_is_deduplicated_without_model() -> None:
    """精确QA可独立于Dense阈值及模型故障去重。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(1, "问原话", answer="收6 00元。"),
                                        _staged(2, "问原话", answer="收6 00元。")])
    model = ScriptedKnowledgeModel(relation_error=KnowledgeModelError("不可用"))
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert len(frame.chunks) == 1 and model.relation_inputs == []


async def test_exact_duplicate_normalization_preserves_internal_number_spaces() -> None:
    """句内数字空白不同不能误合并。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(1, "问", answer="收6 00元。"), _staged(2, "问", answer="收600元。")])
    model = MappedKnowledgeModel({})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    assert len((await _frame(store)).chunks) == 2
    assert model.calls


@pytest.mark.parametrize("existing", [False, True])
async def test_conflicting_documents_publish_nothing_even_below_dense(existing) -> None:
    """新文档之间及未更新权威文档相反时零发布。参数：existing旧权威；返回：None；异常：断言失败。"""
    drafts = [_draft(answer="包邮。")]
    snapshots = []
    if existing:
        snapshots = [_snapshot(20, "doc:b.md", _text(answer="不包邮。"))]
        pair = ("draft:doc:a.md:0", "chunk:20")
    else:
        drafts.append(_draft(source_key="doc:b.md", answer="不包邮。", section_path="完全不同章节"))
        pair = ("draft:doc:a.md:0", "draft:doc:b.md:0")
    store = MemoryKnowledgeStore(snapshots=snapshots)
    model = MappedKnowledgeModel({pair: "conflict"})
    before = await _frame(store)
    with pytest.raises(KnowledgeModelError):
        async with store.writer() as writer:
            await _lifecycle(model).publish_documents(writer, drafts, [draft.source_key for draft in drafts])
    assert await _frame(store) == before


async def test_relation_batches_respect_actual_system_and_items_budget() -> None:
    """八条长证据按实际system消息预算拆组。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(i, f"qa:{i}", _text(answer="文" * 1500, questions=f"旧问{i}")) for i in range(10, 18)])
    model = MappedKnowledgeModel({})
    async with store.writer() as writer:
        await _lifecycle(model).publish_documents(writer, [_draft()], ["doc:a.md"])
    assert sum(len(pairs) for _, pairs in model.calls) == 8
    assert len(model.calls) > 1
    assert all(relation_input_size(items, pairs) <= 12000 for items, pairs in model.calls)


async def test_relation_duplicate_decisions_and_forged_evidence_stop_publication() -> None:
    """重复决定或未知证据必须在任何发布前拒绝。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(1, "甲"), _staged(2, "乙")])
    relation = _decision("staged:1", "staged:2", "same")
    model = ScriptedKnowledgeModel(relation_decisions=[relation, relation])
    before = await _frame(store)
    with pytest.raises(KnowledgeModelError):
        async with store.writer() as writer:
            await _lifecycle(model).reconcile_staging(writer)
    assert await _frame(store) == before


@pytest.mark.parametrize("full", [False, True])
async def test_document_can_judge_actual_conflict_group_without_joining_unrelated(full) -> None:
    """文档完整覆盖可裁定，局部覆盖不能释放未覆盖成员。参数：full完整覆盖；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(50, "doc:policy.md", _text(answer="权威答。"))],
                                 staged=[_staged(1, "甲问", answer="甲答。", conflict_group="cg:old"),
                                         _staged(2, "乙问", answer="乙答。", conflict_group="cg:old"),
                                         _staged(3, "新甲问", answer="甲答。"), _staged(4, "无关问", answer="无关答。")])
    relations = {("staged:3", "staged:1"): "same", ("staged:3", "chunk:50"): "conflict",
                 ("staged:1", "chunk:50"): "conflict", ("staged:4", "chunk:50"): "same"}
    if full:
        relations[("staged:2", "chunk:50")] = "same"
    async with store.writer() as writer:
        await _lifecycle(MappedKnowledgeModel(relations)).reconcile_staging(writer)
    frame = await _frame(store)
    rows = {row.id: row for row in frame.staged}
    assert rows[4].status == "discarded" and rows[4].conflict_group is None
    assert not any(row.source_key.startswith("qa:") and row.is_active for row in frame.chunks)
    if full:
        assert all(rows[i].status == "discarded" and rows[i].conflict_group is None for i in (1, 2, 3))
    else:
        assert rows[2].status == "extracted" and rows[2].conflict_group == "cg:old"


async def test_document_judges_active_content_without_rechecking_each_source_audit() -> None:
    """active三格被权威覆盖后关联审计不能使整组裁定死锁。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(10, "qa:1", _text(answer="旧答。", questions="旧问一\n旧问二")),
                                          _snapshot(50, "doc:policy.md", _text(answer="权威答。"))],
                                 staged=[_staged(1, "旧问一", answer="旧答。", status="kept", knowledge_id=10),
                                         _staged(2, "旧问二", answer="旧答。", status="discarded", knowledge_id=10),
                                         _staged(3, "新问", answer="新答。")])
    model = MappedKnowledgeModel({("staged:3", "chunk:10"): "conflict",
                          ("staged:3", "chunk:50"): "same", ("chunk:10", "chunk:50"): "conflict"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert all(row.status == "discarded" and row.conflict_group is None for row in frame.staged)
    assert next(row for row in frame.chunks if row.id == 10).is_active is False
    assert not any(left in ("staged:1", "staged:2") for _, pairs in model.calls for left, _ in pairs)


async def test_uncertain_document_pair_cannot_release_other_covered_member() -> None:
    """额外不确定权威证据不能被same裁定覆盖。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(50, "doc:one.md", _text()), _snapshot(60, "doc:two.md", _text())],
                                 staged=[_staged(1, "问", answer="答。")])
    model = MappedKnowledgeModel({("staged:1", "chunk:50"): "same", ("staged:1", "chunk:60"): "uncertain"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.staged[0].status == "extracted" and frame.staged[0].conflict_group


class _LowDenseEncoder(FakeEncoder):
    """为不同证据返回正交向量，保证相似度低于真实候选阈值。"""

    async def encode(self, texts):
        """返回正交向量。参数：texts待编码三格；返回：1024维列表；异常：无。"""
        self.inputs.extend(texts)
        return [[float(index == position) for position in range(1024)] for index, _ in enumerate(texts)]


@pytest.mark.parametrize("encoder_type", [FakeEncoder, _LowDenseEncoder])
@pytest.mark.parametrize("historical_question", ["旧问一", "旧问二\n补充条件"])
async def test_historical_exact_qa_reuses_merged_chunk_without_model(
    encoder_type, historical_question: str
) -> None:
    """历史完整问法重现仍复用主键并保留原文证据。

    参数：encoder_type为正常或低Dense编码器；historical_question覆盖kept/discarded完整问法。
    返回：None。
    异常：断言失败；精确关系错误进入不可用模型时抛出KnowledgeModelError。
    """
    store = MemoryKnowledgeStore(staged=[_staged(1, "旧问一", answer="可信答。"),
                                        _staged(2, "旧问二\n补充条件", answer="可信答。")])
    async with store.writer() as writer:
        await _lifecycle(MappedKnowledgeModel({("staged:1", "staged:2"): "same"})).reconcile_staging(writer)
    merged = await _frame(store)
    assert [(row.id, row.text.questions) for row in merged.chunks] == [(1, "旧问一\n旧问二\n补充条件")]
    assert [row.status for row in merged.staged] == ["kept", "discarded"]
    assert all(row.source_evidence for row in merged.staged)

    store = MemoryKnowledgeStore(snapshots=merged.chunks,
                                 staged=merged.staged + [_staged(3, historical_question, answer="可信答。")])
    model = ScriptedKnowledgeModel(relation_error=KnowledgeModelError("模型不可用"))
    lifecycle = KnowledgeLifecycle(encoder_type(), model, POLICY.model_copy(update={"min_score": 0.8}))
    async with store.writer() as writer:
        await lifecycle.reconcile_staging(writer)
    frame = await _frame(store)
    assert [(row.id, row.source_key, row.text.questions) for row in frame.chunks if row.is_active] == [
        (1, "qa:1", "旧问一\n旧问二\n补充条件")]
    assert [(row.status, row.knowledge_id) for row in frame.staged] == [
        ("kept", 1), ("discarded", 1), ("discarded", 1)]
    for previous, current in zip(merged.staged, frame.staged):
        assert (current.question, current.answer, current.source_ids, current.source_evidence) == (
            previous.question, previous.answer, previous.source_ids, previous.source_evidence)
    assert frame.staged[2].source_evidence and not model.relation_inputs


async def test_historical_multiline_fragment_is_not_an_exact_qa() -> None:
    """多行原问的单行片段不能冒充历史完整问法。参数：无；返回：None；异常：断言失败。"""
    original = _snapshot(10, "qa:1", _text(questions="旧问一\n旧问二\n补充条件", answer="可信答。"))
    store = MemoryKnowledgeStore(snapshots=[original], staged=[
        _staged(1, "旧问一", answer="可信答。", status="kept", knowledge_id=10),
        _staged(2, "旧问二\n补充条件", answer="可信答。", status="discarded", knowledge_id=10),
        _staged(3, "补充条件", answer="可信答。")])
    model = ScriptedKnowledgeModel(relation_error=KnowledgeModelError("模型不可用"))
    lifecycle = KnowledgeLifecycle(_LowDenseEncoder(), model, POLICY.model_copy(update={"min_score": 0.8}))
    async with store.writer() as writer:
        await lifecycle.reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.chunks[0] == original
    assert [(row.id, row.status, row.knowledge_id) for row in frame.staged] == [
        (1, "kept", 10), (2, "discarded", 10), (3, "kept", 11)]
    assert not model.relation_inputs


async def test_historical_expired_answer_does_not_match_current_chunk() -> None:
    """过期历史答案不能把新候选接到当前可信知识。参数：无；返回：None；异常：断言失败。"""
    original = _snapshot(10, "qa:1", _text(questions="旧问一\n旧问二", answer="当前可信答。"))
    store = MemoryKnowledgeStore(snapshots=[original, _snapshot(50, "doc:policy.md", _text(answer="当前可信答。"))],
                                 staged=[_staged(1, "旧问一", answer="当前可信答。", status="kept", knowledge_id=10),
                                         _staged(2, "旧问二", answer="过期答。", status="discarded", knowledge_id=10),
                                         _staged(3, "旧问二", answer="过期答。")])
    model = MappedKnowledgeModel({("staged:3", "chunk:50"): "conflict"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.chunks[0] == original
    assert len(frame.chunks) == 2
    assert [(row.status, row.knowledge_id) for row in frame.staged] == [
        ("kept", 10), ("discarded", 10), ("discarded", None)]


async def test_historical_document_withdrawal_cannot_reactivate_old_chunk() -> None:
    """文档已撤回的历史精确QA不得重新激活旧块。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(10, "qa:1", _text(questions="旧问一\n旧问二", answer="过期答。"))],
                                 staged=[_staged(1, "旧问一", answer="过期答。", status="kept", knowledge_id=10),
                                         _staged(2, "旧问二", answer="过期答。", status="discarded", knowledge_id=10)])
    async with store.writer() as writer:
        await _lifecycle(MappedKnowledgeModel({("draft:doc:a.md:0", "chunk:10"): "conflict"})).publish_documents(
            writer, [_draft(answer="当前可信答。")], ["doc:a.md"])
    withdrawn = await _frame(store)
    assert not withdrawn.chunks[0].is_active
    assert all(row.status == "discarded" for row in withdrawn.staged)

    store = MemoryKnowledgeStore(snapshots=withdrawn.chunks,
                                 staged=withdrawn.staged + [_staged(3, "旧问二", answer="过期答。")])
    model = MappedKnowledgeModel({("staged:3", "chunk:11"): "conflict"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert frame.chunks == withdrawn.chunks
    assert all(row.status == "discarded" and not row.conflict_group for row in frame.staged)
    assert frame.staged[2].knowledge_id is None


async def test_document_conflict_detection_bypasses_low_dense_threshold() -> None:
    """文档全pair检查不受Dense低分影响。参数：无；返回：None；异常：断言失败。"""
    model = MappedKnowledgeModel({("draft:doc:a.md:0", "draft:doc:a.md:1"): "conflict"})
    store = MemoryKnowledgeStore()
    lifecycle = KnowledgeLifecycle(_LowDenseEncoder(), model, POLICY.model_copy(update={"min_score": 0.8}))
    with pytest.raises(KnowledgeModelError):
        async with store.writer() as writer:
            await lifecycle.publish_documents(writer, [_draft(answer="包邮。"), _draft(ordinal=1, answer="不包邮。")], ["doc:a.md"])
    assert model.calls and (await _frame(store)).chunks == []


async def test_exact_qa_matches_before_dense_and_preserves_unicode_distinctions() -> None:
    """边界strip行尾规范可去重，Unicode数字仍逐字区分。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(1, " 问\r\n", answer="答①。\r\n"), _staged(2, "问", answer="答①。")])
    model = ScriptedKnowledgeModel(relation_error=KnowledgeModelError("模型不可用"))
    lifecycle = KnowledgeLifecycle(_LowDenseEncoder(), model, POLICY.model_copy(update={"min_score": 0.8}))
    async with store.writer() as writer:
        await lifecycle.reconcile_staging(writer)
    assert len((await _frame(store)).chunks) == 1 and not model.relation_inputs
    assert lifecycle._exact_qa(_text(answer="答①。")) != lifecycle._exact_qa(_text(answer="答1。"))


async def test_complete_authority_resolves_uncertain_group_with_inactive_old_chunk() -> None:
    """充分权威可裁定不确定组及旧撤回块，不会永久隔离。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(10, "qa:1", _text(answer="旧答。", questions="旧问"), active=False),
                                          _snapshot(50, "doc:policy.md", _text(answer="权威答。"))],
                                 staged=[_staged(1, "旧问", answer="旧答。", conflict_group="cg:old", knowledge_id=10),
                                         _staged(2, "新问", answer="新答。")])
    model = MappedKnowledgeModel({("staged:2", "staged:1"): "uncertain", ("staged:2", "chunk:50"): "same",
                          ("chunk:10", "chunk:50"): "conflict"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert all(row.status == "discarded" and row.conflict_group is None for row in frame.staged)
    assert all(not row.is_active for row in frame.chunks if row.source_key.startswith("qa:"))


async def test_partial_document_conflict_cannot_publish_same_group() -> None:
    """same链只有局部被文档推翻时整组不能发布。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(50, "doc:policy.md", _text())],
                                 staged=[_staged(1, "问一", answer="答一。"), _staged(2, "问二", answer="答二。")])
    model = MappedKnowledgeModel({("staged:1", "staged:2"): "same", ("staged:1", "chunk:50"): "conflict"})
    async with store.writer() as writer:
        await _lifecycle(model).reconcile_staging(writer)
    frame = await _frame(store)
    assert all(not row.source_key.startswith("qa:") for row in frame.chunks)
    assert all(row.status == "extracted" and row.conflict_group for row in frame.staged)


async def test_single_relation_pair_over_budget_fails_before_model_and_publication() -> None:
    """单pair也超限时安全失败且零模型调用。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(snapshots=[_snapshot(10, "qa:old", _text(answer="旧" * 12000))])
    model = MappedKnowledgeModel({})
    before = await _frame(store)
    with pytest.raises(KnowledgeModelError, match="单对"):
        async with store.writer() as writer:
            await _lifecycle(model).publish_documents(writer, [_draft()], ["doc:a.md"])
    assert not model.calls and await _frame(store) == before


async def test_forged_relation_evidence_ref_fails_before_publication() -> None:
    """模型伪造证据引用不能进入发布。参数：无；返回：None；异常：断言失败。"""
    store = MemoryKnowledgeStore(staged=[_staged(1, "问一"), _staged(2, "问二")])
    decision = _decision("staged:1", "staged:2", "same").model_copy(update={"evidence_refs": ["missing:1"]})
    model = ScriptedKnowledgeModel(relation_decisions=[decision])
    before = await _frame(store)
    with pytest.raises(KnowledgeModelError, match="引用"):
        async with store.writer() as writer:
            await _lifecycle(model).reconcile_staging(writer)
    assert await _frame(store) == before
