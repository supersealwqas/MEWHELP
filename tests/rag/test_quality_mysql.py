"""仅操作 CH04 专用资源的真实 MySQL 质量表兼容及重复迁移测试。

默认跳过；MEWHELP_RUN_QUALITY_MYSQL=1 显式开启，仅在 mewhelp_ch04_test 创建缺失
业务/质量表，清理本例随机会话和台账行，不删除表或清空任何旧章数据库。
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, inspect, select

from app.config import Settings
from app.db.migrate_rag_quality import migrate_quality_tables, validate_quality_schema
from app.db.session import create_database
from app.models.customer_service import Base, Conversation
from app.models.rag_quality import FaithCase, LowConfidenceQuestion


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
