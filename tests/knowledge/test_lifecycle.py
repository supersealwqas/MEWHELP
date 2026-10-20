"""知识生命周期：文档权威、同义合并与原子发布测试。"""

from __future__ import annotations

import pytest

from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.llm import KnowledgeModelError
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
