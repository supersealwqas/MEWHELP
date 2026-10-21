"""真实 MySQL 上的来源扫描、批次幂等与迟提交补扫测试。"""

from __future__ import annotations

import pytest

from app.knowledge.errors import ConcurrentKnowledgeWrite
from app.knowledge.grounding import turns_digest
from app.knowledge.schemas import ExtractedQA, ExtractionBatch
from app.models.customer_service import Conversation, Message

pytestmark = pytest.mark.knowledge_integration


async def _create_turn(
    factory,
    *,
    session_id: str,
    question: str,
    answer: str,
    turn_id: str = "turn-1",
) -> None:
    """写入一个 completed 轮次。参数：工厂、会话标识、问答、轮次；返回：None；异常：数据库错误。"""
    async with factory.begin() as session:
        conversation = Conversation(session_id=session_id, user_id="ch03-test-user")
        session.add(conversation)
        await session.flush()
        session.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=question,
                    turn_id=turn_id,
                    turn_status="completed",
                ),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=answer,
                    turn_id=turn_id,
                    turn_status="completed",
                ),
            ]
        )


async def _create_explicit_turn(
    session,
    *,
    conversation_id: int,
    session_id: str,
    user_id: int,
    assistant_id: int,
    question: str,
    answer: str,
    turn_id: str,
) -> None:
    """在调用方事务中写入显式 ID 轮次。参数：会话、ID、问答；返回：None；异常：数据库错误。"""
    session.add(
        Conversation(
            id=conversation_id,
            session_id=session_id,
            user_id="ch03-test-user",
        )
    )
    await session.flush()
    session.add_all(
        [
            Message(
                id=user_id,
                conversation_id=conversation_id,
                role="user",
                content=question,
                turn_id=turn_id,
                turn_status="completed",
            ),
            Message(
                id=assistant_id,
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                turn_id=turn_id,
                turn_status="completed",
            ),
        ]
    )


def _batch(batch_no: str, turns) -> ExtractionBatch:
    """由轮次构造批次。参数：批次号与轮次；返回：批次；异常：校验失败。"""
    return ExtractionBatch(
        batch_no=batch_no,
        source_ids=[turn.message_id for turn in turns],
        snapshot_upper_id=max(turn.message_id for turn in turns),
        input_hash=turns_digest(list(turns)),
        prompt_version="ch03-test-v1",
        turns=list(turns),
    )


async def test_scan_reserve_complete_and_recover(knowledge_store, knowledge_database) -> None:
    """扫描完成轮次、预留批次、零/一 QA 提交与恢复。参数：真实存储与库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    await _create_turn(
        factory,
        session_id="batch-session-1",
        question="运费怎么算",
        answer="未满 99 元收 6 元。",
    )
    await _create_turn(
        factory,
        session_id="batch-session-2",
        question="多久发货",
        answer="现货 48 小时内发货。",
    )
    async with knowledge_store.writer() as writer:
        upper = await writer.source_upper_id()
        turns = await writer.unassigned_turns(upper, 0, 10)
        assert len(turns) == 2
        # 只把第一轮放入批次，第二轮保持未分配，验证完成后仍可继续扫描。
        batch = ExtractionBatch(
            batch_no="batch-1",
            source_ids=[turns[0].message_id],
            snapshot_upper_id=upper,
            input_hash=turns_digest([turns[0]]),
            prompt_version="ch03-test-v1",
            turns=[turns[0]],
        )
        await writer.reserve_batch(batch)
        pending = await writer.pending_batches()
        assert [item.batch_no for item in pending] == ["batch-1"]
        staged = await writer.complete_batch(
            batch.batch_no,
            [
                ExtractedQA(
                    source_ids=[turns[0].message_id],
                    question=turns[0].question,
                    answer=turns[0].answer,
                    category="物流",
                )
            ],
        )
        assert len(staged) == 1 and staged[0].status == "extracted"
        assert staged[0].source_evidence[0]["quote"] == turns[0].answer
        assert await writer.pending_batches() == []
        remaining = await writer.unassigned_turns(upper, 0, 10)
        assert [turn.message_id for turn in remaining] == [turns[1].message_id]


async def test_zero_qa_batch_is_done(knowledge_store, knowledge_database) -> None:
    """零 QA 批次仍标记 done，不再被扫描。参数：真实存储与库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    await _create_turn(
        factory,
        session_id="empty-session",
        question="你好",
        answer="您好，有什么可以帮您？",
    )
    async with knowledge_store.writer() as writer:
        upper = await writer.source_upper_id()
        turns = await writer.unassigned_turns(upper, 0, 10)
        batch = await writer.reserve_batch(_batch("batch-empty", turns))
        assert await writer.complete_batch(batch.batch_no, []) == []
        assert await writer.pending_batches() == []
        assert await writer.unassigned_turns(upper, 0, 10) == []


async def test_same_source_cannot_join_two_batches(
    knowledge_store, knowledge_database
) -> None:
    """同一来源不能同时进入两个批次。参数：真实存储与库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    await _create_turn(
        factory,
        session_id="conflict-session",
        question="退货多久到账",
        answer="到账时效以平台规则为准。",
    )
    async with knowledge_store.writer() as writer:
        upper = await writer.source_upper_id()
        turns = await writer.unassigned_turns(upper, 0, 10)
        await writer.reserve_batch(_batch("batch-a", turns))
        with pytest.raises(ConcurrentKnowledgeWrite):
            await writer.reserve_batch(_batch("batch-b", turns))


async def test_late_committed_smaller_id_is_found_without_duplicate(
    knowledge_store, knowledge_database
) -> None:
    """迟提交的小 ID 在从头补扫时被发现，已分配来源不重复。参数：真实存储与库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    async with factory() as session:
        transaction = await session.begin()
        await _create_explicit_turn(
            session,
            conversation_id=8002,
            session_id="late-high-session",
            user_id=9102,
            assistant_id=9002,
            question="高 ID 问题",
            answer="高 ID 回答。",
            turn_id="high-turn",
        )
        await transaction.commit()

    async with knowledge_store.writer() as writer:
        upper = await writer.source_upper_id()
        first = await writer.unassigned_turns(upper, 0, 10)
        assert [turn.message_id for turn in first] == [9002]
        await writer.reserve_batch(_batch("batch-high", first))

    async with factory() as session:
        transaction = await session.begin()
        await _create_explicit_turn(
            session,
            conversation_id=8001,
            session_id="late-low-session",
            user_id=9101,
            assistant_id=9001,
            question="低 ID 迟到问题",
            answer="低 ID 迟到回答。",
            turn_id="low-turn",
        )
        await transaction.commit()

    async with knowledge_store.writer() as writer:
        assert await writer.source_upper_id() == 9002
        second = await writer.unassigned_turns(9002, 0, 10)
    assert [turn.message_id for turn in second] == [9001]


async def test_real_hash_change_and_atomic_qa_validation(knowledge_store, knowledge_database) -> None:
    """真实库验证原子候选写入与预留后来源变更拒绝。参数：专用存储/库；返回：None；异常：断言失败。"""
    from sqlalchemy import update, select
    from app.knowledge.errors import SnapshotChanged
    from app.knowledge.grounding import GroundingError
    from app.models.knowledge import QAExtractionStaging, QAExtractionBatch
    _, factory, _ = knowledge_database
    await _create_turn(factory, session_id="hash-audit", question="运费多少", answer="运费6元。")
    async with knowledge_store.writer() as writer:
        turns = await writer.unassigned_turns(await writer.source_upper_id(), 0, 10)
        batch = _batch("hash-audit", turns)
        await writer.reserve_batch(batch)
        valid = ExtractedQA(source_ids=batch.source_ids, question="运费多少", answer="运费6元。", category="物流")
        with pytest.raises(GroundingError):
            await writer.complete_batch(batch.batch_no, [valid, valid.model_copy(update={"answer": "运费600元。"})])
        assert (await writer.frame()).staged == []
        assert len(await writer.pending_batches()) == 1
        rows = await writer.complete_batch(batch.batch_no, [valid])
        assert rows[0].source_evidence[0]["message_id"] == turns[0].message_id
        assert len(await writer.complete_batch(batch.batch_no, [valid])) == 1
    async with factory() as session:
        assert (await session.scalar(select(QAExtractionBatch).where(QAExtractionBatch.batch_no == batch.batch_no))).status == "done"
        persisted = await session.scalar(select(QAExtractionStaging).where(QAExtractionStaging.batch_no == batch.batch_no))
        assert persisted.evidence_refs[0]["quote"] == "运费6元。"
    async with factory.begin() as session:
        await session.execute(update(Message).where(Message.id == turns[0].message_id).values(content="运费600元。"))
    async with knowledge_store.writer() as writer:
        with pytest.raises(SnapshotChanged, match="指纹"):
            await writer.ensure_staging_grounded()


async def test_real_assigned_page_does_not_hide_later_unassigned(knowledge_store, knowledge_database) -> None:
    """已分配200轮之后仍能扫描后续来源，恢复上界含done。参数：真实库；返回：None；异常：断言失败。"""
    _, factory, _ = knowledge_database
    for i in range(201):
        await _create_turn(factory, session_id=f"page-{i}", question="你好", answer="您好。")
    async with knowledge_store.writer() as writer:
        upper = await writer.source_upper_id()
        first = await writer.unassigned_turns(upper, 0, 200)
        batch = _batch("first-page", first).model_copy(update={"snapshot_upper_id": upper})
        await writer.reserve_batch(batch)
        await writer.complete_batch(batch.batch_no, [])
        assert await writer.recovery_upper_id() == upper
        remaining = await writer.unassigned_turns(upper, 0, 200)
        assert len(remaining) == 1 and remaining[0].message_id == upper


async def test_real_done_idempotent_read_rechecks_source_hash(knowledge_store, knowledge_database) -> None:
    """真实done重读核hash，源改动拒绝且保留已完成身份及证据。参数：真实库/存储；返回：None；异常：断言失败。"""
    from sqlalchemy import update, select
    from app.knowledge.errors import SnapshotChanged
    from app.models.knowledge import QAExtractionBatch, QAExtractionStaging
    _, factory, _ = knowledge_database
    await _create_turn(factory, session_id="done-hash", question="运费多少", answer="运费6元。")
    async with knowledge_store.writer() as writer:
        turns = await writer.unassigned_turns(await writer.source_upper_id(), 0, 10)
        batch = _batch("done-hash", turns)
        await writer.reserve_batch(batch)
        before = await writer.complete_batch(batch.batch_no, [ExtractedQA(
            source_ids=batch.source_ids, question=turns[0].question, answer=turns[0].answer, category="物流")])
        assert await writer.complete_batch(batch.batch_no, []) == before
    async with factory.begin() as session:
        await session.execute(update(Message).where(Message.id == turns[0].message_id).values(content="运费600元。"))
    async with knowledge_store.writer() as writer:
        with pytest.raises(SnapshotChanged, match="指纹"):
            await writer.complete_batch(batch.batch_no, [])
        assert (await writer.frame()).staged == before
    async with factory() as session:
        done = await session.scalar(select(QAExtractionBatch).where(QAExtractionBatch.batch_no == batch.batch_no))
        assert done.status == "done" and done.input_hash == batch.input_hash
        row = await session.scalar(select(QAExtractionStaging).where(QAExtractionStaging.batch_no == batch.batch_no))
        assert row.answer == "运费6元。"
        assert row.evidence_refs == before[0].source_evidence
