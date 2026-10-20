"""CH03 专用 MySQL 测试库夹具，不继承 CH02 的 mewhelp_test 夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db.session import create_database
from app.knowledge.store import KnowledgeStore
from app.models.customer_service import Base as CustomerBase
from app.models.knowledge import KnowledgeBase


@pytest.fixture
async def knowledge_database():
    """创建 mewhelp_ch03_test 连接并清理本章与来源表。

    参数:
        无
    返回值:
        tuple: 普通引擎、会话工厂、锁专用引擎。
    异常:
        AssertionError、数据库异常: 目标库不符或建表/清理失败时抛出。
    """
    config = Settings(_env_file=Path(__file__).resolve().parents[2] / ".env")
    assert config.knowledge_test_database == "mewhelp_ch03_test"
    engine, factory = create_database(config, database="mewhelp_ch03_test")
    lease_engine = create_async_engine(
        engine.url,
        poolclass=NullPool,
        connect_args={"connect_timeout": 5},
    )
    async with engine.begin() as connection:
        assert await connection.scalar(text("SELECT DATABASE()")) == "mewhelp_ch03_test"
        await connection.run_sync(CustomerBase.metadata.create_all)
        await connection.run_sync(KnowledgeBase.metadata.create_all)
    async with engine.begin() as connection:
        # 只清理本章测试库；顺序满足外键依赖，不触碰应用库或 CH02 测试库。
        await connection.execute(text("DELETE FROM qa_extraction_staging"))
        await connection.execute(text("DELETE FROM qa_extraction_batches"))
        await connection.execute(text("UPDATE knowledge_chunks SET prev_chunk_id = NULL, next_chunk_id = NULL"))
        await connection.execute(text("DELETE FROM knowledge_chunks"))
        await connection.execute(text("DELETE FROM tickets"))
        await connection.execute(text("DELETE FROM messages"))
        await connection.execute(text("DELETE FROM conversations"))
    try:
        yield engine, factory, lease_engine
    finally:
        await lease_engine.dispose()
        await engine.dispose()


@pytest.fixture
async def knowledge_store(knowledge_database):
    """提供真实 KnowledgeStore。

    参数:
        knowledge_database: 专用测试库夹具。
    返回值:
        KnowledgeStore: 可读写本章三表的存储。
    异常:
        无
    """
    _, factory, lease_engine = knowledge_database
    yield KnowledgeStore(factory, lease_engine)
