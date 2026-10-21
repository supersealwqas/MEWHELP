"""KnowledgeStore 发布、撤回与批次契约的离线测试。"""

from __future__ import annotations

import pytest

from app.knowledge.schemas import ChunkDraft, KnowledgeText, Publication, StageChange
from app.models import customer_service
from tests.knowledge.fakes import MemoryKnowledgeStore

REVISION = "test-embedding-revision-v1"


def _draft(
    *,
    source_key: str = "doc:a.md",
    ordinal: int = 0,
    answer: str = "第一版答案。",
    section_path: str | None = "甲",
    is_key_clause: bool = False,
) -> ChunkDraft:
    """构造测试用知识块草案。

    参数:
        source_key: 稳定来源键。
        ordinal: 来源内序号。
        answer: 答案文本。
        section_path: 章节路径。
        is_key_clause: 是否关键条款。
    返回值:
        ChunkDraft: 合法草案。
    异常:
        pydantic.ValidationError: 字段不合法时抛出。
    """
    return ChunkDraft(
        source_key=source_key,
        ordinal=ordinal,
        text=KnowledgeText(category="分类", questions="问法", answer=answer),
        section_path=section_path,
        content_type="faq",
        is_key_clause=is_key_clause,
    )


async def _publish(store, drafts: list[ChunkDraft], **kwargs):
    """以当前 frame token 原子发布。参数：存储、草案、附加变更；返回：快照；异常：存储异常。"""
    async with store.writer() as writer:
        frame = await writer.frame()
        return await writer.publish(
            Publication(expected_token=frame.token, drafts=drafts, embedding_revision=REVISION, **kwargs)
        )


async def test_repeat_source_ordinal_keeps_id() -> None:
    """同一来源与序号重复发布复用主键，不新增第二行。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore()
    first = await _publish(store, [_draft()])
    second = await _publish(store, [_draft()])
    assert [snapshot.id for snapshot in second] == [snapshot.id for snapshot in first]


async def test_metadata_only_change_keeps_done() -> None:
    """只改章节元数据时保留已完成向量状态。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore()
    published = await _publish(store, [_draft(section_path="甲")])
    async with store.writer() as writer:
        assert await writer.mark_vectorized(published[0]) is True
        frame = await writer.frame()
        await writer.publish(
            Publication(
                expected_token=frame.token,
                drafts=[_draft(section_path="甲 / 乙")],
                embedding_revision=REVISION,
            )
        )
        updated = (await writer.chunks_by_sources(["doc:a.md"]))[0]
    assert updated.section_path == "甲 / 乙"
    assert updated.vectorize_status == "done"
    assert updated.vector_id == str(updated.id)


async def test_publish_writes_embedding_revision() -> None:
    """新快照记录发布时传入的编码规范标识。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore()
    published = await _publish(store, [_draft()])
    assert published[0].embedding_revision == REVISION


async def test_chunks_by_sources_scopes_old_rows() -> None:
    """按来源键读取旧块，不返回其他文档。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore()
    await _publish(store, [_draft(source_key="doc:a.md"), _draft(source_key="doc:b.md")])
    async with store.writer() as writer:
        rows = await writer.chunks_by_sources(["doc:a.md"])
    assert {row.source_key for row in rows} == {"doc:a.md"}


async def test_publish_and_stage_changes_atomic() -> None:
    """发布失败时不留下部分知识或暂存状态。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore(fail_on_publish=True)
    async with store.writer() as writer:
        before = await writer.frame()
        with pytest.raises(RuntimeError):
            await writer.publish(
                Publication(
                    expected_token=before.token,
                    drafts=[_draft()],
                    embedding_revision=REVISION,
                    stage_changes=[
                        StageChange(id=1, status="discarded", reason="测试并发失败")
                    ],
                )
            )
        after = await writer.frame()
    assert before == after


async def test_quarantine_preserves_id_and_raw_text() -> None:
    """撤回只取消发布资格，保留主键和原文。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore()
    published = await _publish(store, [_draft(answer="需要保留的原文。")])
    original = published[0]
    await _publish(store, [], inactive_ids=[original.id])
    async with store.writer() as writer:
        inactive = (await writer.chunks_by_sources(["doc:a.md"]))[0]
    assert inactive.id == original.id
    assert inactive.text == original.text
    assert inactive.is_active is False
    assert inactive.vectorize_status == "pending"
    assert await store.active_ids([inactive.id]) == set()


def test_ch02_metadata_is_not_extended() -> None:
    """知识库表不进入 CH02 的 Base.metadata。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    assert "knowledge_chunks" not in customer_service.Base.metadata.tables


async def test_snapshot_change_is_rejected() -> None:
    """发布令牌变化时拒绝提交，不能覆盖他人写入。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    store = MemoryKnowledgeStore(snapshot_changed=True)
    with pytest.raises(Exception, match="快照|snapshot|变化"):
        await _publish(store, [_draft()])


def test_stage_change_accepts_document_snapshot_evidence() -> None:
    """处置证据同时接受旧引用和完整文档快照。

    参数：无。返回：None。异常：字段校验或断言失败。
    """
    document = {
        "kind": "document",
        "source_key": "doc:a.md",
        "ordinal": 0,
        "id": 1,
        "section_path": "甲 / 退货",
        "content_hash": "a" * 64,
        "text": {"category": "售后", "questions": "怎么退货", "answer": "七日内可申请。"},
        "relation": "same",
    }
    change = StageChange(
        id=1,
        status="kept",
        reason="文档支持",
        evidence_refs=["chunk:1", document],
    )
    assert change.evidence_refs == ["chunk:1", document]


@pytest.mark.parametrize("withdrawal", ["shortened_source", "inactive_id"])
async def test_withdrawal_clears_both_pointers_and_repairs_active_neighbors(withdrawal) -> None:
    """两种撤回入口均断开旧块指针并接回有效邻居。

    参数：withdrawal 为来源替换或显式主键撤回。返回：None。异常：断言失败。
    """
    store = MemoryKnowledgeStore()
    original = await _publish(store, [_draft(ordinal=index) for index in range(3)])
    if withdrawal == "shortened_source":
        await _publish(
            store,
            [_draft(ordinal=0), _draft(ordinal=2)],
            replace_sources=["doc:a.md"],
        )
    else:
        await _publish(store, [], inactive_ids=[original[1].id])
    async with store.writer() as writer:
        rows = await writer.chunks_by_sources(["doc:a.md"])
    first, withdrawn, last = sorted(rows, key=lambda row: row.ordinal)
    assert withdrawn.is_active is False
    assert withdrawn.prev_chunk_id is None
    assert withdrawn.next_chunk_id is None
    assert first.prev_chunk_id is None
    assert first.next_chunk_id == last.id
    assert last.prev_chunk_id == first.id
    assert last.next_chunk_id is None
