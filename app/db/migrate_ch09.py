"""CH09 固定 SQL 的只读预览、严格启动核验与六步可恢复增量迁移。"""

import argparse
import asyncio
from pathlib import Path
import re

import anyio
from sqlalchemy import inspect, text

from app.config import Settings
from app.db.migrate_rag_quality import QualitySchemaError, _validate_table
from app.db.session import create_database
from app.models.flywheel import EvalRun, ReviewQueue
from app.models.rag_quality import FaithCase, LowConfidenceQuestion


CH09_TABLES = (ReviewQueue.__table__, EvalRun.__table__)
LCQ = LowConfidenceQuestion.__table__
ADDED_COLUMNS = ("retrieved_chunks", "matched_review_id")


def _preflight(inspector, *, complete: bool) -> None:
    """预检所有已有结构，允许续迁缺项但绝不容忍错误形状。

    参数：inspector 为 MySQL 反射；complete 为 True 时禁止任何缺表/缺列/缺约束。
    返回：None；异常：QualitySchemaError，依赖缺失或现存结构与固定 DDL 不符。
    """
    for required in ("conversations", LCQ.name, FaithCase.__tablename__):
        if not inspector.has_table(required):
            raise QualitySchemaError(f"缺少 {required}，请先完成业务初始化和 CH04 迁移")
    _validate_table(inspector, FaithCase.__table__)
    for table in CH09_TABLES:
        if inspector.has_table(table.name):
            _validate_table(inspector, table)
        elif complete:
            raise QualitySchemaError(f"缺少 {table.name}，请显式完成 CH09 迁移")
    _validate_table(inspector, LCQ,
        optional_columns=() if complete else ADDED_COLUMNS,
        optional_indexes=() if complete else ("idx_matched_review_id",),
        optional_foreign_keys=() if complete else ("fk_lcq_review",))


def validate_ch09_schema(inspector) -> None:
    """只读验证启动需要的完整 CH09 结构。参数：同步 Inspector；返回：None；异常：QualitySchemaError/反射错误。"""
    _preflight(inspector, complete=True)


def _ddl_steps() -> list[str]:
    """从结构权威 SQL 提取两次 CREATE 和四项独立 ALTER，保留注释/中文/默认值。

    参数：无，路径固定为仓库 sql/ch09-ddl.sql。
    返回：list[str]，按依赖顺序排列的六个 DDL。
    异常：ValueError，固定 SQL 结构改变；OSError，文件不可读取。
    """
    source = (Path(__file__).resolve().parents[2] / "sql/ch09-ddl.sql").read_text(encoding="utf-8")
    # 只去除整行注释；列 COMMENT 内的分号和逗号必须保留。
    source = re.sub(r"(?m)^\s*--[^\n]*", "", source)
    creates = re.findall(r"CREATE TABLE (?:review_queue|eval_runs)\s*\(.*?\) ENGINE=.*?;", source, re.S)
    altered = source.split("ALTER TABLE low_confidence_questions", 1)[1].strip().removesuffix(";")
    clauses = re.split(r",\s*\n\s*(?=ADD )", altered)
    if len(creates) != 2 or len(clauses) != 4:
        raise ValueError("CH09 固定 DDL 步骤已改变，需人工核对")
    return [statement.rstrip(";") for statement in creates] + [
        "ALTER TABLE low_confidence_questions " + clause.strip() for clause in clauses]


def migrate_connection(connection, *, apply: bool = False) -> list[str]:
    """先完整预检，再只执行尚缺的固定 DDL；断点重跑不破坏旧数据。

    参数：connection 为已核实数据库的同步 MySQL 连接；apply 默认 False 只预览。
    返回：list[str]，本轮待执行或已执行 SQL；完整结构返回空列表。
    异常：ValueError，方言错误；QualitySchemaError，预检不兼容；DDL/反射错误原样传播。
    """
    if connection.dialect.name != "mysql":
        raise ValueError("CH09 迁移只支持 MySQL")
    inspector = inspect(connection)
    _preflight(inspector, complete=False)
    columns = {item["name"] for item in inspector.get_columns(LCQ.name)}
    indexes = {item["name"] for item in inspector.get_indexes(LCQ.name)}
    foreign = {item["name"] for item in inspector.get_foreign_keys(LCQ.name)}
    exists = [*(inspector.has_table(table.name) for table in CH09_TABLES),
              *(name in columns for name in ADDED_COLUMNS),
              "idx_matched_review_id" in indexes, "fk_lcq_review" in foreign]
    pending = [statement for statement, present in zip(_ddl_steps(), exists, strict=True) if not present]
    if apply and pending:
        connection.execute(text("SET NAMES utf8mb4"))
        for statement in pending:
            # DDL 可能已在服务器提交后中断；下次重新反射全部结构再选择剩余步骤。
            # 固定 DDL 的 COMMENT 含 JSON 冒号；直交驱动，避免 text() 误识别为绑定参数。
            connection.exec_driver_sql(statement)
        validate_ch09_schema(inspect(connection))
    return pending


async def check_ch09_tables(config: Settings) -> None:
    """主应用启动时只读核验完整迁移。参数：配置目标库；返回：None；异常：结构/连接错误。"""
    engine, _ = create_database(config)
    try:
        async with engine.connect() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != config.mysql_database:
                raise ValueError("CH09 核验实际数据库与配置不一致")
            await connection.run_sync(lambda sync: validate_ch09_schema(inspect(sync)))
    finally:
        with anyio.CancelScope(shield=True):
            await engine.dispose()


async def migrate(config: Settings, database: str, *, apply: bool = False) -> list[str]:
    """显式指定目标且核对实际数据库，默认不写入。

    参数：config 为连接配置；database 为配置应用库或 mewhelp_ch09_test；apply 为明确执行开关。
    返回：list[str]，待执行或已执行的 SQL。
    异常：ValueError，目标不符；QualitySchemaError，结构不兼容；数据库错误；取消在清理后传播。
    """
    if not database or database not in {config.mysql_database, "mewhelp_ch09_test"}:
        raise ValueError("CH09 目标必须为配置应用库或 mewhelp_ch09_test")
    engine, _ = create_database(config, database=database)
    try:
        async with engine.begin() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != database:
                raise ValueError("实际数据库与显式目标不一致")
            return await connection.run_sync(lambda sync: migrate_connection(sync, apply=apply))
    finally:
        with anyio.CancelScope(shield=True):
            await engine.dispose()


def main() -> None:
    """运行默认只读的具名迁移命令。参数：--env-file/--database/--apply；返回：None；异常：失败非零退出。"""
    parser = argparse.ArgumentParser(description="CH09 固定 DDL 增量迁移，默认预览")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        statements = asyncio.run(migrate(Settings(_env_file=args.env_file), args.database, apply=args.apply))
    except Exception as error:
        # 不将驱动异常中的连接凭据或原始用户数据打印到终端。
        parser.exit(1, f"CH09 迁移失败：{type(error).__name__}\n")
    print({"database": args.database, "applied": args.apply, "statements": statements})


if __name__ == "__main__":
    main()
