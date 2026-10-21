"""真实 MySQL 上的知识发布、撤回与指针恢复测试。"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.knowledge.errors import SnapshotChanged
from app.knowledge.grounding import validate_grounding
from app.knowledge.schemas import (
    ChunkDraft,
    CompletedTurn,
    ExtractedQA,
    KnowledgeText,
    Publication,
    StageChange,
)
from app.knowledge.text import content_fingerprint
from app.models.customer_service import Conversation, Message
from app.models.knowledge import QAExtractionStaging

pytestmark = pytest.mark.knowledge_integration

REVISION = "integration-revision-v1"


def _draft(
    *,
    source_key: str = "doc:a.md",
    ordinal: int = 0,
    answer: str = "第一版答案。",
    section_path: str | None = "甲",
) -> ChunkDraft:
    """构造测试草案。参数：来源、序号、答案、章节；返回：草案；异常：校验失败。"""
    return ChunkDraft(
        source_key=source_key,
        ordinal=ordinal,
        text=KnowledgeText(category="分类", questions="问法", answer=answer),
        section_path=section_path,
        content_type="faq",
        is_key_clause=False,
    )


async def _publish(store, drafts: list[ChunkDraft], **kwargs):
    """以当前 frame 令牌发布。参数：存储、草案、附加变更；返回：快照；异常：存储异常。"""
    async with store.writer() as writer:
        frame = await writer.frame()
        return await writer.publish(
            Publication(
                expected_token=frame.token,
                drafts=drafts,
                embedding_revision=REVISION,
                **kwargs,
            )
        )


async def test_publish_repeat_source_ordinal_keeps_id(knowledge_store) -> None:
    """重复来源/序号复用主键。参数：真实存储；返回：None；异常：断言失败。"""
    first = await _publish(knowledge_store, [_draft()])
    second = await _publish(knowledge_store, [_draft()])
    assert [row.id for row in second] == [row.id for row in first]


async def test_metadata_only_change_keeps_done(knowledge_store) -> None:
    """元数据变化保留 done 与 vector_id。参数：真实存储；返回：None；异常：断言失败。"""
    published = await _publish(knowledge_store, [_draft(section_path="甲")])
    async with knowledge_store.writer() as writer:
        assert await writer.mark_vectorized(published[0]) is True
        rows = await writer.chunks_by_sources(["doc:a.md"])
    assert rows[0].vectorize_status == "done"
    await _publish(knowledge_store, [_draft(section_path="甲 / 乙")])
    async with knowledge_store.writer() as writer:
        updated = (await writer.chunks_by_sources(["doc:a.md"]))[0]
    assert updated.section_path == "甲 / 乙"
    assert updated.vectorize_status == "done"
    assert updated.vector_id == str(updated.id)


async def test_publish_writes_embedding_revision(knowledge_store) -> None:
    """真实行记录发布编码规范。参数：真实存储；返回：None；异常：断言失败。"""
    published = await _publish(knowledge_store, [_draft()])
    assert published[0].embedding_revision == REVISION


async def test_chunks_by_sources_scopes_old_rows(knowledge_store) -> None:
    """按来源读取旧块，不返回其他来源。参数：真实存储；返回：None；异常：断言失败。"""
    await _publish(
        knowledge_store,
        [_draft(source_key="doc:a.md"), _draft(source_key="doc:b.md")],
    )
    async with knowledge_store.writer() as writer:
        rows = await writer.chunks_by_sources(["doc:a.md"])
    assert {row.source_key for row in rows} == {"doc:a.md"}


async def test_quarantine_preserves_id_and_raw_text(knowledge_store) -> None:
    """撤回保留主键和原文。参数：真实存储；返回：None；异常：断言失败。"""
    published = await _publish(knowledge_store, [_draft(answer="必须保留的原文。")])
    original = published[0]
    await _publish(knowledge_store, [], inactive_ids=[original.id])
    async with knowledge_store.writer() as writer:
        rows = await writer.chunks_by_sources(["doc:a.md"])
    assert rows[0].id == original.id
    assert rows[0].text == original.text
    assert rows[0].is_active is False
    assert rows[0].vectorize_status == "pending"
    assert await knowledge_store.active_ids([original.id]) == set()


async def test_pointer_repair_and_shortened_source(knowledge_store) -> None:
    """文档变短时旧序号退役，active 指针跳过退役块。参数：真实存储；返回：None；异常：断言失败。"""
    await _publish(
        knowledge_store,
        [_draft(ordinal=0), _draft(ordinal=1), _draft(ordinal=2)],
    )
    await _publish(
        knowledge_store,
        [_draft(ordinal=0), _draft(ordinal=2)],
        replace_sources=["doc:a.md"],
    )
    async with knowledge_store.writer() as writer:
        rows = await writer.chunks_by_sources(["doc:a.md"])
    by_ordinal = {row.ordinal: row for row in rows}
    assert by_ordinal[1].is_active is False
    assert by_ordinal[1].prev_chunk_id is None
    assert by_ordinal[1].next_chunk_id is None
    assert by_ordinal[0].next_chunk_id == by_ordinal[2].id
    assert by_ordinal[2].prev_chunk_id == by_ordinal[0].id


async def test_explicit_withdrawal_clears_both_pointers_and_repairs_neighbors(
    knowledge_store,
) -> None:
    """显式撤回断开旧块两侧指针并重新连接有效块。

    参数：knowledge_store 为独立真实测试库存储。返回：None。异常：断言或数据库错误。
    """
    original = await _publish(knowledge_store, [_draft(ordinal=index) for index in range(3)])
    await _publish(knowledge_store, [], inactive_ids=[original[1].id])
    async with knowledge_store.writer() as writer:
        rows = await writer.chunks_by_sources(["doc:a.md"])
    first, withdrawn, last = sorted(rows, key=lambda row: row.ordinal)
    assert withdrawn.is_active is False
    assert withdrawn.prev_chunk_id is None
    assert withdrawn.next_chunk_id is None
    assert first.prev_chunk_id is None
    assert first.next_chunk_id == last.id
    assert last.prev_chunk_id == first.id
    assert last.next_chunk_id is None


async def test_stage_audit_appends_stable_unique_document_snapshots(
    knowledge_store, knowledge_database,
) -> None:
    """同序号文档更新仍保留来源、旧处置和完整历史文档快照。

    参数：knowledge_store、knowledge_database 均仅指向 mewhelp_ch03_test。
    返回：None。异常：断言、来源校验或数据库错误。
    """
    _, factory, _ = knowledge_database
    async with factory.begin() as session:
        conversation = Conversation(session_id="store-audit", user_id="ch03-test-user")
        session.add(conversation)
        await session.flush()
        user = Message(
            conversation_id=conversation.id, role="user", content="问法",
            turn_id="audit-1", turn_status="completed",
        )
        assistant = Message(
            conversation_id=conversation.id, role="assistant", content="第一版答案。",
            turn_id="audit-1", turn_status="completed",
        )
        session.add_all([user, assistant])
        await session.flush()
        turn = CompletedTurn(
            message_id=assistant.id, conversation_id=conversation.id, turn_id="audit-1",
            question=user.content, answer=assistant.content,
        )
        qa = ExtractedQA(
            source_ids=[assistant.id], question=user.content,
            answer=assistant.content, category="分类",
        )
        _, source_audit = validate_grounding(qa, [turn])
        staged = QAExtractionStaging(
            batch_no="legacy-store-audit", source_ids=[assistant.id],
            source_fingerprint="b" * 64, question=qa.question, answer=qa.answer,
            category=qa.category, status="extracted",
            evidence_refs=source_audit + ["old:relation"],
        )
        session.add(staged)
        await session.flush()
        stage_id = staged.id

    old = (await _publish(knowledge_store, [_draft(section_path="甲 / 旧章节")]))[0]
    old_document = {
        "kind": "document", "source_key": old.source_key, "ordinal": old.ordinal,
        "id": old.id, "section_path": old.section_path, "content_hash": old.content_hash,
        "text": old.text.model_dump(mode="json"), "relation": "same",
    }
    first_change = StageChange(
        id=stage_id, status="kept", reason="旧文档支持",
        evidence_refs=[
            old_document, old_document, source_audit[0],
            "current:relation", "current:relation",
        ],
    )
    await _publish(knowledge_store, [], stage_changes=[first_change])
    async with factory() as session:
        stored = await session.scalar(
            select(QAExtractionStaging).where(QAExtractionStaging.id == stage_id)
        )
        assert stored.evidence_refs == source_audit + ["old:relation", old_document, "current:relation"]

    new_draft = _draft(answer="第二版答案。", section_path="乙 / 新章节")
    new_document = {
        "kind": "document", "source_key": old.source_key, "ordinal": old.ordinal,
        "id": old.id, "section_path": new_draft.section_path,
        "content_hash": content_fingerprint(new_draft.text),
        "text": new_draft.text.model_dump(mode="json"), "relation": "conflict",
    }
    # JSON对象键顺序不同仍是同一证据；首个快照的顺序和内容必须保持稳定。
    reordered_old = dict(reversed(list(old_document.items())))
    reordered_old["text"] = dict(reversed(list(old_document["text"].items())))
    second_change = StageChange(
        id=stage_id, status="kept", reason="新文档判定",
        evidence_refs=[reordered_old, new_document, "current:relation"],
    )
    updated = (await _publish(knowledge_store, [new_draft], stage_changes=[second_change]))[0]
    await _publish(knowledge_store, [], stage_changes=[second_change])
    assert updated.id == old.id
    assert updated.content_hash != old.content_hash
    async with factory() as session:
        stored = await session.scalar(
            select(QAExtractionStaging).where(QAExtractionStaging.id == stage_id)
        )
        assert stored.evidence_refs == source_audit + [
            "old:relation", old_document, "current:relation", new_document,
        ]
        historical = next(
            item for item in stored.evidence_refs
            if isinstance(item, dict) and item.get("kind") == "document"
        )
        assert historical["section_path"] == "甲 / 旧章节"
        assert historical["content_hash"] == old.content_hash
        assert historical["text"] == old.text.model_dump(mode="json")


async def test_mark_vectorized_rejects_stale_snapshot(knowledge_store) -> None:
    """旧快照不能把变化后的新原文回填为 done。参数：真实存储；返回：None；异常：断言失败。"""
    published = await _publish(knowledge_store, [_draft(answer="旧答案。")])
    stale = published[0]
    await _publish(knowledge_store, [_draft(answer="新答案。")])
    async with knowledge_store.writer() as writer:
        assert await writer.mark_vectorized(stale) is False


async def test_publish_atomic_on_missing_stage_change(knowledge_store) -> None:
    """暂存处置失败时整笔发布回滚。参数：真实存储；返回：None；异常：断言失败。"""
    with pytest.raises(SnapshotChanged):
        await _publish(
            knowledge_store,
            [_draft()],
            stage_changes=[
                StageChange(id=999999, status="discarded", reason="不存在的暂存行")
            ],
        )
    async with knowledge_store.writer() as writer:
        assert await writer.chunks_by_sources(["doc:a.md"]) == []


@pytest.mark.parametrize("changed", ["content", "revision", "inactive", "pending"])
async def test_mark_pending_rejects_changed_done_snapshot(knowledge_store, changed) -> None:
    """旧 done 对照结果不能覆盖新的原文或状态。参数：真实存储、变化；返回：None；异常：断言失败。"""
    from sqlalchemy import update
    from app.models.knowledge import KnowledgeChunk

    original = (await _publish(knowledge_store, [_draft()]))[0]
    async with knowledge_store.writer() as writer:
        await writer.mark_vectorized(original)
    snapshot = (await knowledge_store.get_chunks([original.id]))[0]
    changes = {"content": {"content_hash": "f" * 64},
               "revision": {"embedding_revision": "new-revision"},
               "inactive": {"is_active": False},
               "pending": {"vectorize_status": "pending", "vector_id": None}}[changed]
    async with knowledge_store._factory() as session:
        await session.execute(update(KnowledgeChunk).where(KnowledgeChunk.id == original.id).values(**changes))
        await session.commit()
    async with knowledge_store.writer() as writer:
        assert callable(getattr(writer, "mark_pending", None)), "缺少条件恢复 pending 接口"
        assert await writer.mark_pending(snapshot) is False
    current = (await knowledge_store.get_chunks([original.id]))[0]
    for field, value in changes.items():
        assert getattr(current, field) == value


async def test_active_pages_and_conditional_pending_recovery(knowledge_store) -> None:
    """active 分页覆盖 done 并可条件重置。参数：真实存储；返回：None；异常：断言失败。"""
    rows = await _publish(knowledge_store, [_draft(ordinal=0), _draft(ordinal=1), _draft(ordinal=2)])
    await _publish(knowledge_store, [], inactive_ids=[rows[1].id])
    async with knowledge_store.writer() as writer:
        await writer.mark_vectorized(rows[0])
    assert callable(getattr(knowledge_store, "active_page", None)), "缺少 active 主键分页"
    first = await knowledge_store.active_page(0, 1)
    second = await knowledge_store.active_page(first[0].id, 1)
    assert [first[0].id, second[0].id] == [rows[0].id, rows[2].id]
    async with knowledge_store.writer() as writer:
        assert await writer.mark_pending(first[0]) is True
    restored = (await knowledge_store.get_chunks([rows[0].id]))[0]
    assert restored.vectorize_status == "pending" and restored.vector_id is None
