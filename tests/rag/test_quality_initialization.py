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


def test_ch04_first_create_does_not_depend_on_review_queue(monkeypatch) -> None:
    """ORM 新外键不得改变 CH04 首次建表范围。参数：替换器；返回：None；异常：断言失败。"""
    from sqlalchemy import Table, create_mock_engine
    from app.db import migrate_rag_quality
    from tests.rag.test_quality_schema import SchemaInspector

    statements = []
    inspector = SchemaInspector()
    inspector.columns["conversations"] = []
    inspector.columns.pop("low_confidence_questions")
    inspector.columns.pop("faith_cases")

    def execute(statement, *args, **kwargs):
        """编译首次 DDL 并模拟表已创建。参数：DDL/绑定；返回：None；异常：编译失败。"""
        statements.append(str(statement.compile(dialect=engine.dialect)))
        table = getattr(statement, "element", None)
        if isinstance(table, Table):
            inspector.columns[table.name] = SchemaInspector().columns[table.name]

    engine = create_mock_engine("mysql+pymysql://", execute)
    monkeypatch.setattr(migrate_rag_quality, "inspect", lambda connection: inspector)
    assert migrate_rag_quality._migrate(engine) == ["low_confidence_questions", "faith_cases"]
    assert not any("review_queue" in item or "retrieved_chunks" in item or "matched_review_id" in item for item in statements)


async def test_ch09_application_startup_rejects_old_schema_before_search(monkeypatch) -> None:
    """主应用未接上严格核验而提前加载检索时失败。参数：替换器；返回：None；异常：断言失败。"""
    from app import main
    from app.db.migrate_ch09 import validate_ch09_schema
    from app.db.migrate_rag_quality import QualitySchemaError
    from tests.rag.test_quality_schema import SchemaInspector

    disposed = []

    class Engine:
        """启动错误仍必须关闭已创建的业务引擎。"""

        async def dispose(self):
            """记录关闭。参数：无；返回：None；异常：无。"""
            disposed.append(True)

    async def check(config):
        """以真实验证器拒绝旧形状。参数：配置；返回：不返回；异常：QualitySchemaError。"""
        validate_ch09_schema(SchemaInspector())

    monkeypatch.setattr(main, "create_database", lambda: (Engine(), None))
    monkeypatch.setattr(main, "check_ch09_tables", check)
    with pytest.raises(QualitySchemaError):
        async with main.database_runtime():
            pytest.fail("旧结构不应启动")
    assert disposed == [True]
