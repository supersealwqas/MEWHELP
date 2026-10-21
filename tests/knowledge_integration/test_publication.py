"""真实 MySQL 上的文档权威切换与发布原子性测试。"""

from __future__ import annotations

import hashlib

import pytest

from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.llm import KnowledgeModelError
from app.knowledge.prompts import RELATION_PROMPT_VERSION
from app.knowledge.schemas import (
    ChunkDraft,
    KnowledgeText,
    Publication,
    RelationDecision,
    RelationPolicy,
)
from app.knowledge.text import content_fingerprint
from app.models.knowledge import KnowledgeChunk, QAExtractionStaging
from app.models.customer_service import Conversation, Message
from tests.knowledge.fakes import FakeEncoder, ScriptedKnowledgeModel
from tests.knowledge.test_lifecycle import _MappedModel

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
        conversation = Conversation(session_id="publication-original", user_id="task4-test")
        session.add(conversation)
        await session.flush()
        session.add(Message(conversation_id=conversation.id, role="user", content=text.questions,
                            turn_id="original-turn", turn_status="completed"))
        source = Message(conversation_id=conversation.id, role="assistant", content=text.answer,
                         turn_id="original-turn", turn_status="completed")
        session.add(source)
        await session.flush()
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
            source_ids=[source.id],
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


async def _source_staged(factory, question: str, answer: str, *, knowledge_id=None, status="extracted") -> int:
    """创建真实轮次及其历史暂存。参数：factory工厂/问答/关联知识/状态；返回：暂存ID；异常：数据库错误。"""
    async with factory.begin() as session:
        conversation = Conversation(session_id=f"publication-{question}", user_id="task4-test")
        session.add(conversation)
        await session.flush()
        user = Message(conversation_id=conversation.id, role="user", content=question, turn_id="source-turn", turn_status="completed")
        assistant = Message(conversation_id=conversation.id, role="assistant", content=answer, turn_id="source-turn", turn_status="completed")
        session.add_all([user, assistant])
        await session.flush()
        staged = QAExtractionStaging(batch_no="task4-publication", source_ids=[assistant.id],
                                     source_fingerprint=hashlib.sha256(question.encode()).hexdigest(),
                                     question=question, answer=answer, category="物流", status=status,
                                     knowledge_id=knowledge_id, evidence_refs=[])
        session.add(staged)
        await session.flush()
        return staged.id


async def test_same_active_keeps_stable_id_and_source_audit_in_real_mysql(knowledge_store, knowledge_database) -> None:
    """跨批same真实旧新问法更新原主键且来源审计保存。参数：测试库/存储；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    async with knowledge_store.writer() as writer:
        before = await writer.frame()
        rows = await writer.publish(Publication(expected_token=before.token, embedding_revision="fake-encoder-v1",
                                                drafts=[_draft().model_copy(update={"source_key": "qa:stable", "text": _text("可信答。", "旧问")})]))
        old = rows[0]
        await writer.mark_vectorized(old)
    first = await _source_staged(factory, "旧问", "可信答。", knowledge_id=old.id, status="kept")
    second = await _source_staged(factory, "新问", "可信答。")
    lifecycle = KnowledgeLifecycle(FakeEncoder(), _MappedModel({(f"staged:{second}", f"chunk:{old.id}"): "same"}), POLICY)
    async with knowledge_store.writer() as writer:
        await lifecycle.reconcile_staging(writer)
        frame = await writer.frame()
    main = next(row for row in frame.chunks if row.id == old.id)
    assert main.text.questions == "旧问\n新问" and main.text.answer == "可信答。"
    assert main.vector_id is None and main.vectorize_status == "pending" and main.source_key == "qa:stable"
    assert {row.knowledge_id for row in frame.staged if row.id in (first, second)} == {old.id}
    assert all(row.source_evidence for row in frame.staged)


async def test_document_snapshot_survives_same_ordinal_update_in_real_mysql(knowledge_store, knowledge_database) -> None:
    """文档覆盖暂存后同ordinal更新仍可审计旧路径hash三格与source。参数：测试库/存储；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    identifier = await _source_staged(factory, "旧问", "旧答。")
    model = _MappedModel({})
    lifecycle = KnowledgeLifecycle(FakeEncoder(), model, POLICY)
    first = _draft("旧权威答。").model_copy(update={"section_path": "旧章/旧节"})
    async with knowledge_store.writer() as writer:
        await lifecycle.publish_documents(writer, [first], ["doc:a.md"])
        frame = await writer.frame()
        document = next(row for row in frame.chunks if row.source_key == "doc:a.md")
        model.relations[(f"staged:{identifier}", f"chunk:{document.id}")] = "conflict"
        await lifecycle.reconcile_staging(writer)
        await lifecycle.publish_documents(writer, [_draft("新权威答。")], ["doc:a.md"])
    async with factory() as session:
        staged = await session.get(QAExtractionStaging, identifier)
        audit = list(staged.evidence_refs)
    documents = [item for item in audit if isinstance(item, dict) and item.get("kind") == "document"]
    assert any(item["id"] == document.id and item["section_path"] == "旧章/旧节"
               and item["content_hash"] == content_fingerprint(first.text)
               and item["text"] == first.text.model_dump(mode="json") for item in documents)
    assert any(isinstance(item, dict) and item.get("kind") == "source" for item in audit)


@pytest.mark.parametrize("old_document", [False, True])
async def test_document_conflict_has_zero_real_mysql_publication(knowledge_store, old_document) -> None:
    """新权威相互或与未更新权威冲突时真实事务零发布。参数：存储/old_document；返回：None；异常：断言失败。"""
    model = _MappedModel({})
    lifecycle = KnowledgeLifecycle(FakeEncoder(), model, POLICY)
    one, two = _draft("包邮。"), _draft("不包邮。").model_copy(update={"source_key": "doc:b.md"})
    async with knowledge_store.writer() as writer:
        if old_document:
            published = await lifecycle.publish_documents(writer, [two], [two.source_key])
            pair = ("draft:doc:a.md:0", f"chunk:{published[0].id}")
            drafts = [one]
        else:
            pair = ("draft:doc:a.md:0", "draft:doc:b.md:0")
            drafts = [one, two]
        model.relations[pair] = "conflict"
        before = await writer.frame()
        with pytest.raises(KnowledgeModelError):
            await lifecycle.publish_documents(writer, drafts, [draft.source_key for draft in drafts])
        assert await writer.frame() == before


async def test_all_document_judgment_snapshots_retained_in_one_publication(knowledge_store, knowledge_database) -> None:
    """同一发布多个权威裁定必须全部留在审计。参数：存储/测试库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    identifier = await _source_staged(factory, "旧問", "旧答。")
    async with knowledge_store.writer() as writer:
        frame = await writer.frame()
        published = await writer.publish(Publication(expected_token=frame.token, embedding_revision="fake-encoder-v1",
                                                     drafts=[_draft("旧答。").model_copy(update={"source_key": "qa:old"})],
                                                     stage_changes=[]))
    async with factory.begin() as session:
        row = await session.get(QAExtractionStaging, identifier)
        row.knowledge_id = published[0].id
        row.status = "kept"
    first, second = _draft("权威答。"), _draft("权威答。").model_copy(update={"source_key": "doc:b.md"})
    model = _MappedModel({("draft:doc:a.md:0", f"chunk:{published[0].id}"): "conflict",
                          ("draft:doc:b.md:0", f"chunk:{published[0].id}"): "conflict"})
    async with knowledge_store.writer() as writer:
        await KnowledgeLifecycle(FakeEncoder(), model, POLICY).publish_documents(writer, [first, second], [first.source_key, second.source_key])
    async with factory() as session:
        row = await session.get(QAExtractionStaging, identifier)
        audits = [item for item in row.evidence_refs if isinstance(item, dict) and item.get("kind") == "document"]
    assert {item["source_key"] for item in audits} == {"doc:a.md", "doc:b.md"}


async def test_conflict_uncertain_chain_is_quarantined_in_real_transaction(knowledge_store, knowledge_database) -> None:
    """真实来源链的冲突不确定整组原子隔离。参数：存储/测试库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    identifiers = [await _source_staged(factory, f"问{i}", f"答{i}。") for i in range(3)]
    one, two, three = identifiers
    model = _MappedModel({(f"staged:{one}", f"staged:{two}"): "conflict",
                          (f"staged:{two}", f"staged:{three}"): "uncertain"})
    async with knowledge_store.writer() as writer:
        await KnowledgeLifecycle(FakeEncoder(), model, POLICY).reconcile_staging(writer)
        frame = await writer.frame()
    assert frame.chunks == []
    assert len({row.conflict_group for row in frame.staged}) == 1
    assert all(row.status == "extracted" and row.conflict_group and row.source_evidence for row in frame.staged)
