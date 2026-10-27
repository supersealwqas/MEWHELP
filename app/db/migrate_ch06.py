"""CH06显式增量迁移：保留业务行，扩充等待状态并新增独立退款申请表；默认只预览。"""

import argparse
import asyncio
from sqlalchemy import inspect, text
from sqlalchemy.schema import CreateTable
from app.config import Settings
from app.db.session import create_database
from app.models.customer_service import RefundApplication

WAIT_DDL = ("ALTER TABLE messages DROP CHECK ck_turn_status, ADD CONSTRAINT ck_turn_status "
            "CHECK (turn_status IN ('running','awaiting_input','completed','failed','cancelled'))")


def migrate_connection(connection, *, apply: bool = False) -> list[str]:
    """检查并按需扩充状态，不删除业务数据。

    参数：connection已核实目标的同步连接；apply为True才执行DDL。
    返回：待执行或已执行的语句列表；已应用返回空列表。
    异常：ValueError，缺表/未知约束；数据库错误原样抛出。
    """
    inspector = inspect(connection)
    if not inspector.has_table("messages"):
        raise ValueError("缺少messages，请先初始化业务表")
    checks = inspector.get_check_constraints("messages")
    current = next((item["sqltext"] for item in checks if item["name"] == "ck_turn_status"), None)
    if current is None:
        raise ValueError("缺少ck_turn_status，拒绝猜测数据库结构")
    statements = [] if "awaiting_input" in current else [WAIT_DDL]
    refund_table = RefundApplication.__table__
    if not inspector.has_table(refund_table.name):
        statements.append(str(CreateTable(refund_table).compile(dialect=connection.dialect)))
    else:
        columns = {item["name"] for item in inspector.get_columns(refund_table.name)}
        unique_columns = [set(item["column_names"]) for item in inspector.get_unique_constraints(refund_table.name)]
        if (columns != set(refund_table.columns.keys()) or {"action_id"} not in unique_columns
                or inspector.get_pk_constraint(refund_table.name)["constrained_columns"] != ["refund_no"]):
            raise ValueError("已有refund_applications结构不兼容，请核对后迁移")
    if apply:
        for statement in statements:
            connection.execute(text(statement))
        current = next(item["sqltext"] for item in inspect(connection).get_check_constraints("messages") if item["name"] == "ck_turn_status")
        if "awaiting_input" not in current:
            raise ValueError("等待状态迁移未生效")
    return statements


async def migrate(config: Settings, database: str, *, apply: bool = False) -> list[str]:
    """只访问配置应用库或明确的测试库。参数：config配置、database目标、apply应用开关；返回：迁移语句；异常：目标/连接/结构错误。"""
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
    """解析显式迁移命令。参数：--env-file/--database/--apply；返回：None，打印安全结果；异常：错误仅打印类型，退出非零。"""
    parser = argparse.ArgumentParser(description="CH06迁移，默认预览，不清理数据")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        statements = asyncio.run(migrate(Settings(_env_file=args.env_file), args.database, apply=args.apply))
    except Exception as error:
        # 驱动异常可能带连接信息，命令行不输出其原始消息。
        parser.exit(1, f"CH06迁移失败：{type(error).__name__}\n")
    print({"database": args.database, "applied": args.apply, "statements": statements})


if __name__ == "__main__":
    main()
