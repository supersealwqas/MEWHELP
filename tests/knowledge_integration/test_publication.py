"""真实 MySQL 上的文档权威切换与发布原子性测试。"""

from __future__ import annotations

import hashlib

import pytest

from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.prompts import RELATION_PROMPT_VERSION
from app.knowledge.schemas import (
    ChunkDraft,
    KnowledgeText,
    RelationDecision,
    RelationPolicy,
)
from app.knowledge.text import content_fingerprint
from app.models.knowledge import KnowledgeChunk, QAExtractionStaging
from tests.knowledge.fakes import FakeEncoder, ScriptedKnowledgeModel

pytestmark = pytest.mark.knowledge_integration

POLICY = RelationPolicy(
    min_score=0.0,
    embedding_revision="fake-encoder-v1",
    prompt_version=RELATION_PROMPT_VERSION,
    calibration_digest="integration-test",
)


def _text(answer: str = "旧知识。", questions: str = "旧问法") -> KnowledgeText:
    """构造三格文本。参数：答案/问法；返回：文本；异常：校验失败。"""
    return KnowledgeText(category="物流", questions=questions, answer=answer)


def _draft(answer: str = "新文档知识。") -> ChunkDraft:
    """构造文档草案。参数：答案；返回：草案；异常：校验失败。"""
    return ChunkDraft(
        source_key="doc:a.md",
        ordinal=0,
        text=_text(answer=answer, questions="文档问法"),
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
    )


async def _insert_mined(factory) -> tuple[int, int]:
    """写入一条已发布挖掘知识与对应 kept 暂存。

    参数:
        factory: 真实测试库会话工厂。
    返回值:
        tuple[int, int]: 知识主键与暂存主键。
    异常:
        数据库异常。
    """
    text = _text(answer="旧知识。", questions="旧问法")
    digest = hashlib.sha256(f"staged-{text.answer}".encode("utf-8")).hexdigest()
    async with factory.begin() as session:
        chunk = KnowledgeChunk(
            source_key="qa:1",
            ordinal=0,
            category=text.category,
            questions=text.questions,
            answer=text.answer,
            section_path=None,
            content_type="faq",
            is_key_clause=False,
            content_hash=content_fingerprint(text),
            embedding_revision="fake-encoder-v1",
            is_active=True,
            vectorize_status="done",
        )
        session.add(chunk)
        await session.flush()
        chunk.vector_id = str(chunk.id)
        staged = QAExtractionStaging(
            batch_no="publication-integration",
            source_ids=[1],
            source_fingerprint=digest,
            question=text.questions,
            answer=text.answer,
            category=text.category,
            status="kept",
            evidence_refs=[],
            knowledge_id=chunk.id,
        )
        session.add(staged)
        await session.flush()
        return chunk.id, staged.id


async def _insert_document(factory) -> int:
    """写入一条与草案一致的文档知识。参数：工厂；返回：主键；异常：数据库错误。"""
    draft = _draft()
    async with factory.begin() as session:
        chunk = KnowledgeChunk(
            source_key="doc:a.md",
            ordinal=0,
            category=draft.text.category,
            questions=draft.text.questions,
            answer=draft.text.answer,
            section_path=draft.section_path,
            content_type=draft.content_type,
            is_key_clause=False,
            content_hash=content_fingerprint(draft.text),
            embedding_revision="fake-encoder-v1",
            is_active=True,
            vectorize_status="done",
        )
        session.add(chunk)
        await session.flush()
        chunk.vector_id = str(chunk.id)
        return chunk.id


async def test_document_overrules_published_qa_in_real_mysql(
    knowledge_store, knowledge_database
) -> None:
    """真实事务同时撤回旧 QA 并发布新文档。参数：真实存储与库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    chunk_id, staged_id = await _insert_mined(factory)
    model = ScriptedKnowledgeModel(
        relation_decisions=[
            RelationDecision(
                left_ref="draft:doc:a.md:0",
                right_ref=f"chunk:{chunk_id}",
                relation="conflict",
                evidence_refs=["draft:doc:a.md:0", f"chunk:{chunk_id}"],
                reason="文档规则不同",
            )
        ]
    )
    lifecycle = KnowledgeLifecycle(FakeEncoder(), model, POLICY)
    async with knowledge_store.writer() as writer:
        await lifecycle.publish_documents(writer, [_draft()], ["doc:a.md"])
        frame = await writer.frame()
    old = next(row for row in frame.chunks if row.id == chunk_id)
    document = next(row for row in frame.chunks if row.source_key == "doc:a.md")
    rejected = next(row for row in frame.staged if row.id == staged_id)
    assert old.is_active is False and old.text.answer == "旧知识。"
    assert document.is_active is True
    assert rejected.status == "discarded"


async def test_unchanged_document_reimport_is_noop_in_real_mysql(
    knowledge_store, knowledge_database
) -> None:
    """真实重导相同文档不调用模型、不改变状态。参数：真实存储与库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    existing_id = await _insert_document(factory)
    model = ScriptedKnowledgeModel()
    lifecycle = KnowledgeLifecycle(FakeEncoder(), model, POLICY)
    async with knowledge_store.writer() as writer:
        before = await writer.frame()
        result = await lifecycle.publish_documents(writer, [_draft()], ["doc:a.md"])
        after = await writer.frame()
    assert [row.id for row in result] == [existing_id]
    assert model.relation_inputs == []
    assert before == after


async def test_checkpoint_failure_leaves_no_partial_document(
    knowledge_store, knowledge_database
) -> None:
    """真实发布前 checkpoint 失败不留下新文档或撤回。参数：真实存储与库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    chunk_id, _ = await _insert_mined(factory)

    def checkpoint(label: str) -> None:
        """在文档发布前中断。参数：阶段；返回：None；异常：RuntimeError。"""
        if label == "before_document_publish":
            raise RuntimeError("模拟真实事务前中断")

    model = ScriptedKnowledgeModel(
        relation_decisions=[
            RelationDecision(
                left_ref="draft:doc:a.md:0",
                right_ref=f"chunk:{chunk_id}",
                relation="conflict",
                evidence_refs=["draft:doc:a.md:0", f"chunk:{chunk_id}"],
                reason="文档规则不同",
            )
        ]
    )
    lifecycle = KnowledgeLifecycle(FakeEncoder(), model, POLICY, checkpoint)
    async with knowledge_store.writer() as writer:
        before = await writer.frame()
        with pytest.raises(RuntimeError):
            await lifecycle.publish_documents(writer, [_draft()], ["doc:a.md"])
        after = await writer.frame()
    assert before == after
