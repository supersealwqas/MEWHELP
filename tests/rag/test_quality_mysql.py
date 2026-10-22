"""仅操作 CH04 专用资源的真实 MySQL 质量表兼容及重复迁移测试。

默认跳过；MEWHELP_RUN_QUALITY_MYSQL=1 显式开启，仅在 mewhelp_ch04_test 创建缺失
业务/质量表，清理本例随机会话和台账行，不删除表或清空任何旧章数据库。
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, inspect, select

from app.config import Settings
from app.db.migrate_rag_quality import migrate_quality_tables, validate_quality_schema
from app.db.session import create_database
from app.models.customer_service import Base, Conversation, Message
from app.models.rag_quality import FaithCase, LowConfidenceQuestion


@pytest.mark.skipif(os.getenv("MEWHELP_RUN_QUALITY_MYSQL") != "1", reason="需要显式 MEWHELP_RUN_QUALITY_MYSQL=1")
async def test_refusal_and_pool_commit_idempotently_and_failure_rolls_back(monkeypatch) -> None:
    """真实 MySQL 验证同轮并发拒答只入池一次，异常时两者均回滚。参数：替换器；返回：None；异常：SQL/断言错误。"""
    from app.repositories import chat as module
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL, SelfCheckFailure
    from app.schemas.tools import ToolContext

    config = Settings(_env_file=os.getenv("MEWHELP_TEST_ENV_FILE", ".env")).model_copy(update={"mysql_database": "mewhelp_ch04_test"})
    engine, factory = create_database(config)
    context = None
    try:
        async with engine.begin() as connection:
            await connection.run_sync(lambda sync: Message.__table__.create(sync, checkfirst=True))
        await migrate_quality_tables(config)
        repository = module.ChatRepository(factory)
        raw = "  烦死了，这个到底怎么退啊？！  "
        context = await repository.start_turn(ToolContext(session_id="pool-" + uuid4().hex, user_id="quality-test",
                                                        turn_id=str(uuid4()), user_message=raw), raw)
        failure = SelfCheckFailure(reason="当前证据不够")
        await asyncio.gather(*(repository.finish_turn(context, FIXED_KNOWLEDGE_REFUSAL, low_confidence=failure) for _ in range(3)))
        async with factory() as session:
            questions = (await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == context.conversation_id))).all()
            replies = (await session.scalars(select(Message).where(Message.conversation_id == context.conversation_id, Message.role == "assistant"))).all()
            assert len(questions) == len(replies) == 1 and questions[0].raw_question == raw
            assert questions[0].source == "self_check" and replies[0].turn_status == "completed"
        second = await repository.start_turn(context.model_copy(update={"turn_id": str(uuid4())}), raw)

        def fail_pool(**kwargs):
            """注入池写入构造故障。参数：数据库字段；返回：不返回；异常：RuntimeError。"""
            raise RuntimeError("pool unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(module, "LowConfidenceQuestion", fail_pool)
            with pytest.raises(RuntimeError):
                await repository.finish_turn(second, FIXED_KNOWLEDGE_REFUSAL, low_confidence=failure)
        async with factory() as session:
            rows = (await session.scalars(select(Message).where(Message.turn_id == second.turn_id))).all()
            assert len(rows) == 1 and rows[0].role == "user" and rows[0].turn_status == "running"
    finally:
        if context is not None:
            async with factory.begin() as session:
                await session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == context.conversation_id))
                await session.execute(delete(Message).where(Message.conversation_id == context.conversation_id))
                await session.execute(delete(Conversation).where(Conversation.id == context.conversation_id))
        await engine.dispose()


@pytest.mark.skipif(os.getenv("MEWHELP_RUN_QUALITY_MYSQL") != "1", reason="需要显式 MEWHELP_RUN_QUALITY_MYSQL=1")
async def test_faith_case_concurrent_recurrence_updates_latest_snapshot_without_losing_history() -> None:
    """真实台账同题并发累计、人工解决后复发，完整快照更新而首次/处置时间保持。参数：无；返回：None；异常：SQL/断言错误。"""
    from app.knowledge.answering import EvidenceCitation
    from app.repositories.faith_cases import FaithCaseDraft, FaithCaseRepository

    config = Settings(_env_file=os.getenv("MEWHELP_TEST_ENV_FILE", ".env")).model_copy(update={"mysql_database": "mewhelp_ch04_test"})
    engine, factory = create_database(config)
    eval_id = "T" + uuid4().hex[:15]
    try:
        await migrate_quality_tables(config)
        repository = FaithCaseRepository(factory)
        citations = [EvidenceCitation(n=index, chunk_id=index, section_path="售后 / 政策", question=f"问{index}", answer=f"原文{index}")
                     for index in range(1, 11)]
        draft = FaithCaseDraft(eval_id=eval_id, bucket="A_policy", query="退款到账几天", strategy="hybrid_rerank",
                               answer="保证两天到账[1]", reason="证据没有到账时限", citations=citations, judge_model="judge-test")
        await repository.record(draft)
        async with factory() as session:
            first = await session.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id))
            first_time = first.first_seen_at
            first_id = first.id
        await repository.resolve(eval_id, "已解决", "已补充证据与拒绝到账承诺")
        async with factory() as session:
            solved = await session.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id))
            resolved = solved.resolved_at
        latest = draft.model_copy(update={"answer": "保证三天到账[2]", "reason": "又编造到账天数"})
        await asyncio.gather(*(repository.record(latest) for _ in range(3)))
        async with factory() as session:
            rows = (await session.scalars(select(FaithCase).where(FaithCase.eval_id == eval_id))).all()
            assert len(rows) == 1
            row = rows[0]
            assert row.id == first_id and row.seen_count == 4
            assert row.first_seen_at == first_time and row.resolved_at == resolved
            assert row.status == "未解决" and row.resolution is None
            assert row.answer == "保证三天到账[2]" and row.reason == "又编造到账天数"
            assert len(row.citations) == 10 and row.citations[-1]["answer"] == "原文10"
    finally:
        async with factory.begin() as session:
            await session.execute(delete(FaithCase).where(FaithCase.eval_id == eval_id))
        await engine.dispose()


@pytest.mark.skipif(os.getenv("MEWHELP_RUN_QUALITY_MYSQL") != "1", reason="需要显式 MEWHELP_RUN_QUALITY_MYSQL=1")
async def test_quality_migration_is_repeatable_and_retains_existing_utf8_rows() -> None:
    """真实 MySQL 迁移保持旧记录与中文枚举，不重建已有表。参数：无；返回：None；异常：SQL/断言错误。"""
    config = Settings(_env_file=os.getenv("MEWHELP_TEST_ENV_FILE", ".env"))
    assert config.knowledge_rag_test_database == "mewhelp_ch04_test"
    config = config.model_copy(update={"mysql_database": "mewhelp_ch04_test"})
    engine, factory = create_database(config)
    session_key = "quality-" + uuid4().hex
    conversation_id = None
    try:
        # 专用新库只补会话表；不调用 Base.metadata.create_all 重建/创建无关业务表。
        async with engine.begin() as connection:
            await connection.run_sync(lambda sync: Conversation.__table__.create(sync, checkfirst=True))
        await migrate_quality_tables(config)
        async with factory.begin() as session:
            conversation = Conversation(session_id=session_key, user_id="quality-test")
            session.add(conversation)
            await session.flush()
            conversation_id = conversation.id
            session.add(LowConfidenceQuestion(conversation_id=conversation_id, raw_question="你们到底能不能退啊？", source="self_check", reason="当前证据不足"))
        assert await migrate_quality_tables(config) == []
        async with engine.connect() as connection:
            await connection.run_sync(lambda sync: validate_quality_schema(inspect(sync)))
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == conversation_id))).all()
            assert len(rows) == 1 and rows[0].source == "self_check" and rows[0].raw_question == "你们到底能不能退啊？"
    finally:
        if conversation_id is not None:
            async with factory.begin() as session:
                await session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == conversation_id))
                await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
        await engine.dispose()
