"""独立手写反射夹具验证 CH09 预检、只读预览和断点续迁。"""

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.dialects import mysql

from tests.rag.test_quality_schema import SchemaInspector


class FlywheelInspector(SchemaInspector):
    """依据用户 DDL 手工录入六步迁移后的反射，不从待测 ORM 推导期望。"""

    def __init__(self, steps=6):
        """构造合法前缀状态。参数：steps 已执行的 0 至 6 步；返回：None；异常：无。"""
        super().__init__()
        self.steps = steps
        self.columns["conversations"] = []
        if steps >= 1:
            self.columns["review_queue"] = [
                self._column("id", mysql.BIGINT(unsigned=True), False, None),
                self._column("normalized_question", mysql.VARCHAR(512), False, None),
                self._column("ai_suggested_answer", mysql.TEXT(), True, None),
                self._column("occurrence_count", mysql.INTEGER(unsigned=True), False, "1"),
                self._column("review_status", mysql.ENUM("待审", "通过", "驳回"), False, "'待审'"),
                self._column("approved_answer", mysql.TEXT(), True, None),
                self._column("created_at", mysql.DATETIME(), False, "CURRENT_TIMESTAMP"),
                self._column("updated_at", mysql.DATETIME(), False, "CURRENT_TIMESTAMP() ON UPDATE CURRENT_TIMESTAMP()"),
            ]
        if steps >= 2:
            self.columns["eval_runs"] = [
                self._column("id", mysql.BIGINT(unsigned=True), False, None),
                self._column("triggered_by", mysql.ENUM("定时", "手动"), False, "'定时'"),
                self._column("dataset_size", mysql.INTEGER(unsigned=True), False, None),
                self._column("metrics", mysql.JSON(), False, None),
                self._column("created_at", mysql.DATETIME(), False, "CURRENT_TIMESTAMP"),
            ]
        if steps >= 3:
            self.columns["low_confidence_questions"].insert(5, self._column("retrieved_chunks", mysql.JSON(), True, None))
        if steps >= 4:
            self.columns["low_confidence_questions"].insert(6, self._column("matched_review_id", mysql.BIGINT(unsigned=True), True, None))

    def get_indexes(self, name):
        """返回已执行步骤拥有的索引。参数：表名；返回：索引列表；异常：无。"""
        if name in {"review_queue", "eval_runs"}:
            column = "review_status" if name == "review_queue" else "created_at"
            return [{"name": "idx_" + column, "column_names": [column], "unique": False}]
        indexes = super().get_indexes(name)
        if name == "low_confidence_questions" and self.steps >= 5:
            indexes.append({"name": "idx_matched_review_id", "column_names": ["matched_review_id"], "unique": False})
        return indexes

    def get_foreign_keys(self, name):
        """返回会话外键及第六步审核外键。参数：表名；返回：外键列表；异常：无。"""
        foreign = super().get_foreign_keys(name)
        if name == "low_confidence_questions" and self.steps >= 6:
            foreign.append({"name": "fk_lcq_review", "constrained_columns": ["matched_review_id"],
                            "referred_table": "review_queue", "referred_columns": ["id"],
                            "options": {"ondelete": "SET NULL"}})
        return foreign

    def get_check_constraints(self, name):
        """固定 DDL 没有额外检查约束。参数：表名；返回：空列表；异常：无。"""
        return []


class PreviewConnection:
    """只读预览不得触发写 SQL；模拟实际 MySQL 方言。"""

    dialect = mysql.dialect()

    def execute(self, statement):
        """写操作立即失败。参数：SQL；返回：无；异常：AssertionError。"""
        raise AssertionError(f"预览发生写入：{statement}")

    def exec_driver_sql(self, statement):
        """预览不得直交驱动写入。参数：SQL；返回：无；异常：AssertionError。"""
        raise AssertionError(f"预览发生驱动写入：{statement}")


@pytest.mark.parametrize("steps", range(7))
def test_preview_resumes_each_ddl_breakpoint_without_writes(monkeypatch, steps):
    """已有步骤被重复创建或遗漏剩余步骤时失败。参数：替换器/断点；返回：None；异常：断言失败。"""
    from app.db import migrate_ch09

    inspector = FlywheelInspector(steps)
    monkeypatch.setattr(migrate_ch09, "inspect", lambda connection: inspector)
    statements = migrate_ch09.migrate_connection(PreviewConnection())
    assert len(statements) == 6 - steps
    assert all("DROP " not in statement and "CREATE TABLE low_confidence_questions" not in statement for statement in statements)
    if steps == 0:
        assert statements[0].startswith("CREATE TABLE review_queue")
        assert statements[1].startswith("CREATE TABLE eval_runs")
        assert "AFTER reason" in statements[2] and "AFTER retrieved_chunks" in statements[3]


@pytest.mark.parametrize("fault", ["enum", "index", "index_prefix", "implicit_index", "foreign", "foreign_schema", "onupdate", "extra_column"])
def test_ch09_rejects_changed_enum_index_or_foreign_key(monkeypatch, fault):
    """不兼容结构被预览或启动误接纳时失败。参数：替换器/故障；返回：None；异常：断言失败。"""
    from app.db import migrate_ch09
    from app.db.migrate_rag_quality import QualitySchemaError

    inspector = FlywheelInspector()
    if fault == "enum":
        inspector.columns["review_queue"][4]["type"] = mysql.ENUM("待审", "通过", "乱码")
    elif fault in {"index", "index_prefix", "implicit_index"}:
        original = inspector.get_indexes

        def indexes(name):
            """注入同名错误索引。参数：表名；返回：索引；异常：无。"""
            rows = original(name)
            if name == "low_confidence_questions" and fault == "implicit_index":
                rows.append({"name": "fk_lcq_conversation", "column_names": ["matched_review_id"], "unique": False})
            if name == "review_queue":
                if fault == "index":
                    rows[0]["column_names"] = ["created_at"]
                elif fault == "index_prefix":
                    rows[0]["dialect_options"] = {"mysql_length": 2}
            return rows

        inspector.get_indexes = indexes
    elif fault in {"foreign", "foreign_schema"}:
        original = inspector.get_foreign_keys

        def foreign(name):
            """注入级联删除或跨库外键。参数：表名；返回：外键；异常：无。"""
            rows = original(name)
            if name == "low_confidence_questions":
                if fault == "foreign":
                    rows[-1]["options"] = {"ondelete": "CASCADE"}
                else:
                    rows[-1]["referred_schema"] = "another_database"
            return rows

        inspector.get_foreign_keys = foreign
    elif fault == "onupdate":
        inspector.columns["review_queue"][-1]["default"] = "CURRENT_TIMESTAMP"
    else:
        inspector.columns["eval_runs"].append(inspector._column("published", mysql.INTEGER(), False, "0"))
    monkeypatch.setattr(migrate_ch09, "inspect", lambda connection: inspector)
    with pytest.raises(QualitySchemaError):
        migrate_ch09.migrate_connection(PreviewConnection(), apply=True)
    with pytest.raises(QualitySchemaError):
        migrate_ch09.validate_ch09_schema(inspector)


@pytest.mark.parametrize("steps", range(6))
def test_startup_requires_complete_ch09_schema(steps):
    """旧表和迁移中间态不能接收新章请求。参数：断点；返回：None；异常：断言失败。"""
    from app.db.migrate_ch09 import validate_ch09_schema
    from app.db.migrate_rag_quality import QualitySchemaError

    with pytest.raises(QualitySchemaError):
        validate_ch09_schema(FlywheelInspector(steps))
    validate_ch09_schema(FlywheelInspector())


async def test_migration_rejects_actual_database_mismatch_and_disposes(monkeypatch):
    """显式目标与实际连接不符仍执行 DDL 时失败。参数：替换器；返回：None；异常：断言失败。"""
    from app.db import migrate_ch09
    from tests.knowledge.fakes import make_settings

    disposed = []

    class Connection:
        """实际连接错误库，禁止进入同步 DDL。"""

        async def scalar(self, statement):
            """读取实际库。参数：SQL；返回：错误库名；异常：无。"""
            return "mewhelp"

    class Engine:
        """隔离外部连接并记录清理。"""

        @asynccontextmanager
        async def begin(self):
            """产生连接。参数：无；返回：连接；异常：调用方错误。"""
            yield Connection()

        async def dispose(self):
            """记录关闭。参数：无；返回：None；异常：无。"""
            disposed.append(True)

    monkeypatch.setattr(migrate_ch09, "create_database", lambda *args, **kwargs: (Engine(), None))
    with pytest.raises(ValueError):
        await migrate_ch09.migrate(make_settings(), "mewhelp_ch09_test", apply=True)
    assert disposed == [True]
    with pytest.raises(ValueError):
        await migrate_ch09.migrate(make_settings(), "unapproved_database", apply=True)
