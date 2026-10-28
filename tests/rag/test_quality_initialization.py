"""业务初始化包含退款申请与会话摘要；质量模型加载后不能扩大业务创建/清理范围。"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from app.models.customer_service import Base
from app.models.rag_quality import FaithCase, LowConfidenceQuestion
from tests.knowledge.fakes import make_settings


def test_original_business_metadata_excludes_quality_tables() -> None:
    """业务及工具审计七表与质量台账保持隔离。参数：无；返回：None；异常：断言失败。"""
    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets", "refund_applications", "conversation_summaries", "tool_audit_logs"}
    assert FaithCase.__table__.metadata is not Base.metadata
    assert LowConfidenceQuestion.__table__.metadata is not Base.metadata


async def test_original_initializer_creates_only_original_business_tables(monkeypatch) -> None:
    """实际初始化发出的DDL不能包含质量表或依赖模块导入改变其范围。参数：替换器；返回：None；异常：断言失败。"""
    from app.db import init_db
    statements = []

    class Connection:
        """拦截SQL执行边界并让真实metadata生成DDL。"""

        async def run_sync(self, function):
            """用模拟engine编译真实建表。参数：同步回调；返回：回调结果；异常：无。"""
            from sqlalchemy import create_mock_engine
            engine = create_mock_engine("mysql+pymysql://", lambda statement, *args, **kwargs: statements.append(str(statement)))
            return function(engine)

    class Engine:
        """替代外部数据库。"""

        @asynccontextmanager
        async def begin(self):
            """提供连接。参数：无；返回：连接；异常：无。"""
            yield Connection()

        async def dispose(self):
            """释放边界。参数：无；返回：None；异常：无。"""

    async def seed(factory):
        """替代外部种子写入。参数：工厂；返回：None；异常：无。"""

    monkeypatch.setattr(init_db, "create_database", lambda **kwargs: (Engine(), None))
    monkeypatch.setattr(init_db, "seed_database", seed)
    await init_db.initialize_database()
    assert not any("faith_cases" in statement or "low_confidence_questions" in statement for statement in statements)
    assert sum("CREATE TABLE" in statement for statement in statements) == 7
