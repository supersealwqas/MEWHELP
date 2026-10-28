"""CH08显式增量迁移，复用既有SQL文件，拒绝覆盖不兼容表，默认只预览。"""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import inspect, text

from app.config import Settings
from app.db.session import create_database
from app.models.tool_audit import ToolAuditLog


def _default(value) -> str | None:
    """规范MySQL反射的等价默认值。参数：DDL默认值；返回：标准文本或空；异常：无。"""
    return str(value).strip("'").lower().replace("()", "") if value is not None else None


def _verify(connection) -> None:
    """逐列及约束核验既有表。参数：MySQL连接；返回：None；异常：ValueError表示不兼容。"""
    inspector, expected = inspect(connection), ToolAuditLog.__table__
    actual = {row["name"]: row for row in inspector.get_columns(expected.name)}
    if set(actual) != set(expected.columns.keys()):
        raise ValueError("已有tool_audit_logs字段不兼容")
    for column in expected.c:
        row = actual[column.name]
        wanted = column.type.compile(dialect=connection.dialect)
        found = row["type"].compile(dialect=connection.dialect)
        default = str(column.server_default.arg) if column.server_default is not None else None
        if (wanted != found or column.nullable != row["nullable"] or _default(default) != _default(row["default"])
                or row.get("computed")
                or (column.name == "id" and not row.get("autoincrement"))):
            raise ValueError(f"已有tool_audit_logs.{column.name}结构不兼容")
    index_rows = inspector.get_indexes(expected.name)
    # 同名同列的前缀、全文或空间索引不等于用户DDL要求的普通完整索引。
    for row in index_rows:
        index_options = row.get("dialect_options", {})
        if (index_options.get("mysql_length") or index_options.get("mysql_prefix")
                or index_options.get("mysql_using", "BTREE").upper() != "BTREE"):
            raise ValueError("已有tool_audit_logs索引属性不兼容")
    indexes = {(row["name"], tuple(row["column_names"]), bool(row["unique"])) for row in index_rows}
    wanted_indexes = {(row.name, tuple(row.columns.keys()), row.unique) for row in expected.indexes}
    options = inspector.get_table_options(expected.name)
    if (indexes != wanted_indexes or inspector.get_foreign_keys(expected.name)
            or inspector.get_check_constraints(expected.name)
            or inspector.get_unique_constraints(expected.name)
            or inspector.get_pk_constraint(expected.name)["constrained_columns"] != ["id"]
            or options.get("mysql_engine", "").lower() != "innodb"
            or not options.get("mysql_default charset", options.get("mysql_charset", "")).startswith("utf8mb4")):
        raise ValueError("已有tool_audit_logs索引或表选项不兼容")


def migrate_connection(connection, *, apply: bool = False) -> list[str]:
    """从用户DDL创建缺失审计表，存在则严格核验。参数：已核实的连接/应用开关；返回：SQL列表；异常：类型/结构/DDL错误。"""
    if connection.dialect.name != "mysql":
        raise ValueError("CH08迁移只支持MySQL")
    if inspect(connection).has_table("tool_audit_logs"):
        _verify(connection)
        return []
    source = (Path(__file__).resolve().parents[2] / "sql/ch08-ddl.sql").read_text(encoding="utf-8")
    statement = "CREATE TABLE " + source.split("CREATE TABLE", 1)[1].strip().rstrip(";")
    if apply:
        connection.execute(text("SET NAMES utf8mb4"))
        connection.execute(text(statement))
        _verify(connection)
    return [statement]


async def migrate(config: Settings, database: str, *, apply: bool = False) -> list[str]:
    """核验目标库再执行迁移。参数：配置/显式库/应用开关；返回：SQL列表；异常：连接或结构错误。"""
    if database not in {config.mysql_database, "mewhelp_test"}:
        raise ValueError("目标必须是配置应用库或mewhelp_test")
    engine, _ = create_database(config, database=database)
    try:
        async with engine.begin() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != database:
                raise ValueError("实际数据库与显式目标不一致")
            return await connection.run_sync(lambda conn: migrate_connection(conn, apply=apply))
    finally:
        await engine.dispose()


def main() -> None:
    """迁移CLI默认预览。参数：--env-file/--database/--apply；返回：None；异常：安全报错并非零退出。"""
    parser = argparse.ArgumentParser(description="CH08审计增量迁移，默认预览")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        statements = asyncio.run(migrate(Settings(_env_file=args.env_file), args.database, apply=args.apply))
    except Exception as error:
        parser.exit(1, f"CH08迁移失败：{type(error).__name__}\n")
    print({"database": args.database, "applied": args.apply, "statements": statements})


if __name__ == "__main__":
    main()
