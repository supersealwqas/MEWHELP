"""真实 MySQL 上的知识发布、撤回与指针恢复测试。"""

from __future__ import annotations

import pytest

from app.knowledge.errors import SnapshotChanged
from app.knowledge.schemas import (
    ChunkDraft,
    KnowledgeText,
    Publication,
    StageChange,
)

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
    assert by_ordinal[0].next_chunk_id == by_ordinal[2].id
    assert by_ordinal[2].prev_chunk_id == by_ordinal[0].id


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
