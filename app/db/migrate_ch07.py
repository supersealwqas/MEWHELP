"""CH07显式增量迁移：补齐会话锚点和只追加摘要表，默认预览，重复执行不修改旧行。"""

import argparse
import asyncio

from sqlalchemy import inspect, text
from sqlalchemy.schema import CreateTable

from app.config import Settings
from app.db.session import create_database
from app.models.customer_service import ConversationSummary


CONVERSATION_COLUMNS = {
    "summary": "TEXT NULL COMMENT '最近完整摘要段拼成的有界投影' AFTER status",
    "summary_upto_msg_id": "BIGINT UNSIGNED NULL COMMENT '摘要已覆盖到的消息主键' AFTER summary",
    "layer1_from_msg_id": "BIGINT UNSIGNED NULL COMMENT '此消息之后为原文层' AFTER summary_upto_msg_id",
}


def migrate_connection(connection, *, apply: bool = False) -> list[str]:
    """检查并补齐CH07结构，不重建表或改业务数据。

    参数：connection已核实目标的MySQL连接；apply为True才执行DDL。
    返回：待执行或已执行语句；再次执行返回空列表。
    异常：ValueError表示缺基础表或已有结构不兼容；数据库DDL错误透传。
    """
    if connection.dialect.name != "mysql":
        raise ValueError("CH07生产迁移只支持MySQL")
    inspector = inspect(connection)
    if not inspector.has_table("conversations") or not inspector.has_table("messages"):
        raise ValueError("请先准备conversations和messages基础表")
    columns = {column["name"]: column for column in inspector.get_columns("conversations")}
    statements = []
    for name, definition in CONVERSATION_COLUMNS.items():
        if name not in columns:
            statements.append(f"ALTER TABLE conversations ADD COLUMN {name} {definition}")
            continue
        kind = str(columns[name]["type"]).upper()
        if not columns[name]["nullable"] or (name == "summary" and kind != "TEXT") or (
                name != "summary" and (kind != "BIGINT" or not getattr(columns[name]["type"], "unsigned", False))):
            raise ValueError(f"已有conversations.{name}结构不兼容")
    table = ConversationSummary.__table__
    if not inspector.has_table(table.name):
        statements.append(str(CreateTable(table).compile(dialect=connection.dialect)))
        # CreateTable只生成表及约束，独立索引按用户DDL补齐。
        statements.append("CREATE INDEX idx_conv_upto ON conversation_summaries (conversation_id, upto_msg_id)")
    else:
        actual = {column["name"] for column in inspector.get_columns(table.name)}
        unique = [item["column_names"] for item in inspector.get_unique_constraints(table.name)]
        indices = [item["column_names"] for item in inspector.get_indexes(table.name)]
        if (actual != set(table.columns.keys()) or ["conversation_id", "seq"] not in unique
                or inspector.get_pk_constraint(table.name)["constrained_columns"] != ["id"]):
            raise ValueError("已有conversation_summaries结构不兼容")
        if ["conversation_id", "upto_msg_id"] not in indices:
            statements.append("CREATE INDEX idx_conv_upto ON conversation_summaries (conversation_id, upto_msg_id)")
    if apply:
        connection.execute(text("SET NAMES utf8mb4"))
        for statement in statements:
            connection.execute(text(statement))
        if migrate_connection(connection):
            raise ValueError("CH07迁移未完整生效")
    return statements


async def migrate(config: Settings, database: str, *, apply: bool = False) -> list[str]:
    """核对目标后执行增量迁移。参数：config配置、database显式库、apply应用开关；返回：语句；异常：目标/连接/DDL错误。"""
    if database not in {config.mysql_database, "mewhelp_test", "mewhelp_ch04_test"}:
        raise ValueError("目标必须是配置应用库或指定测试库")
    engine, _ = create_database(config, database=database)
    try:
        async with engine.begin() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != database:
                raise ValueError("实际数据库与显式目标不一致")
            return await connection.run_sync(lambda conn: migrate_connection(conn, apply=apply))
    finally:
        await engine.dispose()


def main() -> None:
    """执行迁移CLI。参数：--env-file/--database/--apply；返回：None并打印语句；异常：只打印安全类型并非零退出。"""
    parser = argparse.ArgumentParser(description="CH07增量迁移，默认预览，不删除数据")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        statements = asyncio.run(migrate(Settings(_env_file=args.env_file), args.database, apply=args.apply))
    except Exception as error:
        parser.exit(1, f"CH07迁移失败：{type(error).__name__}\n")
    print({"database": args.database, "applied": args.apply, "statements": statements})


if __name__ == "__main__":
    main()
