"""真实 MySQL 集成测试夹具：严格隔离 mewhelp_test，禁止清理应用数据库。"""

from pathlib import Path

import pytest
from sqlalchemy import text

from app.config import Settings
from app.db.session import create_database
from app.models.customer_service import Base
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository


@pytest.fixture
async def database():
    """提供独立测试库。参数：无；返回：engine 与 session factory；异常：数据库连接或目标校验失败。"""
    config = Settings(_env_file=Path(__file__).resolve().parents[2] / ".env")
    assert config.mysql_test_database == "mewhelp_test"
    engine, factory = create_database(config, database="mewhelp_test")
    async with engine.begin() as connection:
        assert await connection.scalar(text("SELECT DATABASE()")) == "mewhelp_test"
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine, factory
    finally:
        await engine.dispose()


@pytest.fixture
def repositories(database):
    """提供两个真实存储对象。参数：database 测试库夹具；返回：聊天与业务 Repository；异常：无。"""
    _, factory = database
    return ChatRepository(factory), SupportRepository(factory)
