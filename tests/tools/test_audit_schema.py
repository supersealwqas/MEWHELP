"""核对CH08审计映射的MySQL字段和约束，防止默认建表偏离既有DDL。"""

from sqlalchemy.dialects import mysql


def test_audit_schema_matches_supplied_mysql_ddl():
    """字段、类型、索引及默认值必须匹配用户DDL。参数：无；返回：None；异常：断言失败。"""
    from app.models.tool_audit import ToolAuditLog
    table = ToolAuditLog.__table__
    assert list(table.columns.keys()) == ["id", "conversation_id", "tool_call_id", "tool_name", "tool_source",
        "mcp_server", "arguments", "result_summary", "status", "error_message", "retry_count", "duration_ms", "created_at"]
    assert {name: str(table.c[name].type.compile(dialect=mysql.dialect())) for name in
            ("id", "conversation_id", "retry_count", "duration_ms")} == {
        "id": "BIGINT UNSIGNED", "conversation_id": "BIGINT UNSIGNED", "retry_count": "TINYINT UNSIGNED", "duration_ms": "INTEGER UNSIGNED"}
    assert table.c.status.type.enums == ["成功", "失败", "超时", "校验拦下", "权限拒绝"]
    assert table.c.tool_source.type.enums == ["builtin", "mcp"]
    assert table.c.tool_name.type.length == 128 and table.c.error_message.type.length == 512
    assert {column.name for column in table.c if not column.nullable} == {
        "id", "tool_name", "tool_source", "status", "retry_count", "created_at"}
    assert str(table.c.retry_count.server_default.arg) == "0"
    assert str(table.c.created_at.server_default.arg) == "CURRENT_TIMESTAMP"
    assert list(table.primary_key.columns.keys()) == ["id"] and not table.foreign_keys
    assert {(index.name, tuple(index.columns.keys()), index.unique) for index in table.indexes} == {
        ("idx_conversation_id", ("conversation_id",), False), ("idx_tool_name", ("tool_name",), False), ("idx_status", ("status",), False)}
    assert table.dialect_options["mysql"]["engine"] == "InnoDB"
    assert table.dialect_options["mysql"]["charset"] == "utf8mb4"
