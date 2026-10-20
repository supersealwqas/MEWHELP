"""真实 MySQL 上的来源扫描、批次幂等与迟提交补扫测试。"""

from __future__ import annotations

import pytest

from app.knowledge.errors import ConcurrentKnowledgeWrite
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
        input_hash=f"input-{batch_no}",
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
            input_hash="input-batch-1",
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
                    answer="可泛化的运费知识。",
                    category="物流",
                )
            ],
        )
        assert len(staged) == 1 and staged[0].status == "extracted"
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
