"""用户 CH04 DDL 的字段/约束和已有表兼容校验测试。"""

from __future__ import annotations

import copy

import pytest
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable


class SchemaInspector:
    """手工核对用户 DDL 的 MySQL 反射边界，不由模型定义生成期望。"""

    def __init__(self) -> None:
        """初始化完整反射夹具。参数：无；返回：None；异常：无。"""
        self.columns = {
            "low_confidence_questions": [
                self._column("id", mysql.BIGINT(unsigned=True), False, None),
                self._column("conversation_id", mysql.BIGINT(unsigned=True), True, None),
                self._column("raw_question", mysql.TEXT(), False, None),
                self._column("source", mysql.ENUM("retrieval_low_conf", "self_check", "user_feedback"), False, None),
                self._column("reason", mysql.TEXT(), True, None),
                self._column("created_at", mysql.DATETIME(), False, "CURRENT_TIMESTAMP"),
            ],
            "faith_cases": [
                self._column("id", mysql.BIGINT(unsigned=True), False, None),
                self._column("eval_id", mysql.VARCHAR(16), False, None),
                self._column("bucket", mysql.VARCHAR(24), False, None),
                self._column("query", mysql.VARCHAR(512), False, None),
                self._column("strategy", mysql.VARCHAR(24), False, "'hybrid_rerank'"),
                self._column("answer", mysql.TEXT(), False, None),
                self._column("reason", mysql.TEXT(), False, None),
                self._column("citations", mysql.JSON(), True, None),
                self._column("judge_model", mysql.VARCHAR(64), True, None),
                self._column("status", mysql.ENUM("未解决", "已解决", "无需解决"), False, "'未解决'"),
                self._column("seen_count", mysql.INTEGER(unsigned=True), False, "1"),
                self._column("first_seen_at", mysql.DATETIME(), False, "CURRENT_TIMESTAMP"),
                self._column("last_seen_at", mysql.DATETIME(), False, "CURRENT_TIMESTAMP"),
                self._column("resolution", mysql.VARCHAR(300), True, None),
                self._column("resolved_at", mysql.DATETIME(), True, None),
            ],
        }

    @staticmethod
    def _column(name, type_, nullable, default):
        """返回一列反射。参数：名称/类型/空值/默认；返回：字典；异常：无。"""
        return {"name": name, "type": type_, "nullable": nullable, "default": default,
                "autoincrement": name == "id"}

    def has_table(self, name):
        """判断表存在。参数：表名；返回：bool；异常：无。"""
        return name in self.columns

    def get_columns(self, name):
        """读取列。参数：表名；返回：完整副本；异常：无。"""
        return copy.deepcopy(self.columns[name])

    def get_pk_constraint(self, name):
        """读取主键。参数：表名；返回：单 id；异常：无。"""
        return {"constrained_columns": ["id"]}

    def get_indexes(self, name):
        """读取用户规定索引。参数：表名；返回：索引；异常：无。"""
        if name == "low_confidence_questions":
            return [{"name": "idx_source", "column_names": ["source"], "unique": False},
                    {"name": "idx_created_at", "column_names": ["created_at"], "unique": False}]
        return [{"name": "uk_eval_id", "column_names": ["eval_id"], "unique": True},
                {"name": "idx_status", "column_names": ["status"], "unique": False},
                {"name": "idx_last_seen_at", "column_names": ["last_seen_at"], "unique": False}]

    def get_unique_constraints(self, name):
        """读取唯一约束。参数：表名；返回：约束；异常：无。"""
        return [{"name": "uk_eval_id", "column_names": ["eval_id"]}] if name == "faith_cases" else []

    def get_foreign_keys(self, name):
        """读取会话外键。参数：表名；返回：外键；异常：无。"""
        return [{"name": "fk_lcq_conversation", "constrained_columns": ["conversation_id"],
                 "referred_table": "conversations", "referred_columns": ["id"]}] if name == "low_confidence_questions" else []

    def get_table_options(self, name):
        """读取存储编码。参数：表名；返回：InnoDB/utf8mb4；异常：无。"""
        return {"mysql_engine": "InnoDB", "mysql_default charset": "utf8mb4"}


def test_low_confidence_orm_exact_fields_no_useful_or_turn_id() -> None:
    """问题池不能偷偷加 useful 或轮次列，不改变用户原始 DDL。参数：无；返回：None；异常：断言失败。"""
    from app.models.rag_quality import LowConfidenceQuestion

    table = LowConfidenceQuestion.__table__
    assert set(table.columns.keys()) == {"id", "conversation_id", "raw_question", "source", "reason", "created_at"}
    assert table.c.conversation_id.nullable
    assert table.c.source.type.enums == ["retrieval_low_conf", "self_check", "user_feedback"]
    ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
    assert "CONSTRAINT fk_lcq_conversation FOREIGN KEY(conversation_id) REFERENCES conversations (id)" in ddl
    assert "CHARSET=utf8mb4" in ddl


def test_faith_case_orm_retains_all_snapshot_and_recurrence_fields() -> None:
    """台账完整证据、唯一题号及复发时间不能丢掉。参数：无；返回：None；异常：断言失败。"""
    from app.models.rag_quality import FaithCase

    table = FaithCase.__table__
    assert set(table.columns.keys()) == {"id", "eval_id", "bucket", "query", "strategy", "answer", "reason", "citations",
                                         "judge_model", "status", "seen_count", "first_seen_at", "last_seen_at", "resolution", "resolved_at"}
    assert table.c.eval_id.type.length == 16 and table.c.resolution.type.length == 300
    assert table.c.status.type.enums == ["未解决", "已解决", "无需解决"]
    assert table.c.seen_count.type.unsigned
    assert isinstance(table.c.citations.type, __import__("sqlalchemy").JSON)
    assert any(constraint.name == "uk_eval_id" for constraint in table.constraints)


def test_valid_existing_quality_schema_is_idempotently_accepted() -> None:
    """已有用户 DDL 可重复核验，不要求删除或重建。参数：无；返回：None；异常：断言失败。"""
    from app.db.migrate_rag_quality import validate_quality_schema

    validate_quality_schema(SchemaInspector())


@pytest.mark.parametrize("fault", ["column", "nullable", "enum", "length", "unsigned", "default", "index", "foreign", "charset"])
def test_incompatible_quality_schema_is_rejected_without_mutation(fault: str) -> None:
    """结构不兼容应停止，不能 create_all 后直接声称迁移成功。参数：故障类型；返回：None；异常：断言失败。"""
    from app.db.migrate_rag_quality import QualitySchemaError, validate_quality_schema

    inspector = SchemaInspector()
    if fault == "column":
        inspector.columns["low_confidence_questions"].append(inspector._column("useful", mysql.BOOLEAN(), False, None))
    elif fault == "nullable":
        inspector.columns["low_confidence_questions"][1]["nullable"] = False
    elif fault == "enum":
        inspector.columns["faith_cases"][9]["type"] = mysql.ENUM("garbled", "已解决", "无需解决")
    elif fault == "length":
        inspector.columns["faith_cases"][1]["type"] = mysql.VARCHAR(8)
    elif fault == "unsigned":
        inspector.columns["faith_cases"][10]["type"] = mysql.INTEGER(unsigned=False)
    elif fault == "default":
        inspector.columns["faith_cases"][9]["default"] = "'已解决'"
    elif fault == "index":
        inspector.get_indexes = lambda name: []
        inspector.get_unique_constraints = lambda name: []
    elif fault == "foreign":
        inspector.get_foreign_keys = lambda name: []
    else:
        inspector.get_table_options = lambda name: {"mysql_engine": "InnoDB", "mysql_default charset": "latin1"}
    with pytest.raises(QualitySchemaError):
        validate_quality_schema(inspector)
