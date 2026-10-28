"""只在mewhelp_test验证审计迁移、中文枚举和独立事务，不接应用业务库。"""

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.mysql


async def test_migration_preview_apply_repeat(database):
    """DDL文件建表与ORM相容，预览不执行且应用后幂等。参数：测试库；返回：None；异常：数据库/断言错误。"""
    from app.db.migrate_ch08 import migrate_connection
    engine, _ = database
    async with engine.begin() as connection:
        assert await connection.scalar(text("SELECT DATABASE()")) == "mewhelp_test"
        await connection.execute(text("DROP TABLE IF EXISTS tool_audit_logs"))
        preview = await connection.run_sync(migrate_connection)
        assert len(preview) == 1 and "CREATE TABLE tool_audit_logs" in preview[0]
        assert await connection.run_sync(migrate_connection) == preview
        assert await connection.run_sync(lambda conn: migrate_connection(conn, apply=True)) == preview
        assert await connection.run_sync(lambda conn: migrate_connection(conn, apply=True)) == []
        ddl = (await connection.execute(text("SHOW CREATE TABLE tool_audit_logs"))).one()[1]
        assert "FOREIGN KEY" not in ddl and "UNIQUE KEY" not in ddl and "权限拒绝" in ddl


@pytest.mark.parametrize("alter", [
    "MODIFY retry_count TINYINT NOT NULL DEFAULT 0",
    "MODIFY status ENUM('成功','失败') NOT NULL",
    "MODIFY tool_name VARCHAR(128) NULL",
    "MODIFY retry_count TINYINT UNSIGNED NOT NULL DEFAULT 1",
    "DROP INDEX idx_status",
    "ADD UNIQUE INDEX unexpected_unique (tool_call_id)",
    "DROP INDEX idx_tool_name, ADD INDEX idx_tool_name (tool_name(4))",
    "DROP INDEX idx_tool_name, ADD FULLTEXT INDEX idx_tool_name (tool_name)",
    "ADD CONSTRAINT success_only CHECK (status = '成功')",
    "MODIFY duration_ms INT UNSIGNED GENERATED ALWAYS AS (1) STORED",
])
async def test_incompatible_existing_schema_rejected(database, alter):
    """列名一样但类型、默认或索引变化也必须拒绝。参数：测试库/固定变更；返回：None；异常：断言失败。"""
    from app.db.migrate_ch08 import migrate_connection
    engine, _ = database
    async with engine.begin() as connection:
        await connection.execute(text("ALTER TABLE tool_audit_logs " + alter))
        with pytest.raises(ValueError, match="不兼容"):
            await connection.run_sync(lambda conn: migrate_connection(conn, apply=True))


async def test_chinese_enum_round_trip(database):
    """全部中文状态及Unicode参数真实回读。参数：测试库；返回：None；异常：断言失败。"""
    from app.repositories.tool_audit import ToolAuditEntry, ToolAuditRepository
    engine, factory = database
    repository = ToolAuditRepository(factory)
    for status in ("成功", "失败", "超时", "校验拦下", "权限拒绝"):
        await repository.record(ToolAuditEntry(tool_name="查询", tool_source="mcp", mcp_server="售后",
                                               status=status, arguments={"说明": "中文"}))
    async with engine.connect() as connection:
        rows = (await connection.execute(text("SELECT status, arguments FROM tool_audit_logs ORDER BY id"))).all()
        assert [row[0] for row in rows] == ["成功", "失败", "超时", "校验拦下", "权限拒绝"]
        assert all("中文" in row[1] for row in rows)


async def test_null_conversation_and_no_foreign_key(database):
    """空会话和不存在的会话均不受审计外键限制。参数：测试库；返回：None；异常：断言失败。"""
    from app.repositories.tool_audit import ToolAuditEntry, ToolAuditRepository
    engine, factory = database
    for conversation in (None, 987654321):
        await ToolAuditRepository(factory).record(ToolAuditEntry(conversation_id=conversation,
            tool_name="query_order", tool_source="builtin", status="成功"))
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT COUNT(*) FROM tool_audit_logs")) == 2


async def test_audit_uses_independent_session(database):
    """审计提交不提交业务事务，业务回滚不丢审计。参数：测试库；返回：None；异常：断言失败。"""
    from app.repositories.tool_audit import ToolAuditEntry, ToolAuditRepository
    from app.models.customer_service import Conversation
    engine, factory = database
    async with factory() as business:
        business.add(Conversation(session_id="ch08-isolation", user_id="demo-user"))
        await business.flush()
        await ToolAuditRepository(factory).record(ToolAuditEntry(tool_name="query_order", tool_source="builtin", status="成功"))
        assert business.in_transaction()
        await business.rollback()
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT COUNT(*) FROM conversations")) == 0
        assert await connection.scalar(text("SELECT COUNT(*) FROM tool_audit_logs")) == 1
