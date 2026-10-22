"""显式创建/校验用户 CH04 两张质量表的非破坏性迁移。

先检查已有表的字段、索引、枚举、默认和外键，兼容才创建缺失表；MySQL DDL
不是整体可回滚事务，允许部分成功后重跑，但不擅自 ALTER/drop/recreate。
命令入口可显式使用 mewhelp_ch04_test，两类旧测试库不复用。
"""

from __future__ import annotations

import argparse
import asyncio

import anyio
from sqlalchemy import inspect, text
from sqlalchemy.dialects import mysql

from app.config import Settings
from app.db.session import create_database
from app.models.rag_quality import QUALITY_TABLES


class QualitySchemaError(RuntimeError):
    """质量表已有结构不兼容或依赖缺失，需要操作人核查，不能自动破坏性修复。"""


def _default(value: object) -> str | None:
    """统一 MySQL 反射默认值写法。参数：value 默认表达式；返回：规范字符串或 None；异常：无。"""
    if value is None:
        return None
    return str(value).strip().strip("'").lower().removesuffix("()")


def _validate_table(inspector, table) -> None:
    """检查一张已有表与用户定义逐字段兼容。

    参数：inspector 同步 SQLAlchemy Inspector；table 为两张质量表之一。
    返回：None，结构满足约束。
    异常：QualitySchemaError，列/类型/索引/枚举/默认/编码/外键不一致；底层反射错误。
    """
    name = table.name
    columns = {column["name"]: column for column in inspector.get_columns(name)}
    if set(columns) != set(table.columns.keys()):
        raise QualitySchemaError(f"{name} 字段集合不兼容，不自动修改已有表")
    for expected in table.columns:
        actual = columns[expected.name]
        actual_type, wanted = actual["type"], expected.type
        actual_compiled = str(actual_type.compile(dialect=mysql.dialect())).upper()
        wanted_compiled = str(wanted.compile(dialect=mysql.dialect())).upper()
        if actual_compiled != wanted_compiled or actual["nullable"] != expected.nullable:
            raise QualitySchemaError(f"{name}.{expected.name} 类型/长度/空值约束不兼容")
        if hasattr(wanted, "enums") and actual_type.enums != wanted.enums:
            raise QualitySchemaError(f"{name}.{expected.name} 中文枚举不兼容")
        default = None if expected.server_default is None else expected.server_default.arg
        if _default(actual.get("default")) != _default(default):
            raise QualitySchemaError(f"{name}.{expected.name} 默认值不兼容")
        if expected.primary_key and not actual.get("autoincrement"):
            raise QualitySchemaError(f"{name} 主键不是自增")
    if inspector.get_pk_constraint(name).get("constrained_columns") != ["id"]:
        raise QualitySchemaError(f"{name} 主键不兼容")
    indexes = {index["name"]: index for index in inspector.get_indexes(name)}
    for expected in table.indexes:
        actual = indexes.get(expected.name)
        if (actual is None or actual.get("column_names") != [column.name for column in expected.columns]
                or bool(actual.get("unique")) != expected.unique):
            raise QualitySchemaError(f"{name} 缺少兼容索引 {expected.name}")
    if name == "faith_cases":
        unique = [*inspector.get_unique_constraints(name), *[index for index in indexes.values() if index.get("unique")]]
        if not any(item.get("name") == "uk_eval_id" and item.get("column_names") == ["eval_id"] for item in unique):
            raise QualitySchemaError("faith_cases 缺少 uk_eval_id 唯一约束")
    if name == "low_confidence_questions":
        foreign = inspector.get_foreign_keys(name)
        if not any(item.get("name") == "fk_lcq_conversation" and item.get("constrained_columns") == ["conversation_id"]
                   and item.get("referred_table") == "conversations" and item.get("referred_columns") == ["id"] for item in foreign):
            raise QualitySchemaError("low_confidence_questions 会话外键不兼容")
    options = inspector.get_table_options(name)
    charset = options.get("mysql_default charset") or options.get("mysql_charset")
    if str(options.get("mysql_engine", "")).lower() != "innodb" or str(charset).lower() != "utf8mb4":
        raise QualitySchemaError(f"{name} 必须使用 InnoDB/utf8mb4")


def validate_quality_schema(inspector, *, require_all: bool = True) -> None:
    """校验两张质量表，不写表或修复数据。

    参数：inspector 同步反射对象；require_all 默认 True 要求均存在，False 仅预检已有表。
    返回：None，所有已有表通过校验。
    异常：QualitySchemaError，缺表或任一结构不兼容；反射依赖错误。
    """
    for table in QUALITY_TABLES:
        if inspector.has_table(table.name):
            _validate_table(inspector, table)
        elif require_all:
            raise QualitySchemaError(f"缺少 {table.name}，请显式运行质量表迁移")


def _migrate(connection) -> list[str]:
    """在同步桥内预检并只创建缺失两表，再全量核验。

    参数：connection 为 AsyncConnection.run_sync 传入的同步连接。
    返回：list[str]，本次创建的表名；已有兼容表不在列表中。
    异常：QualitySchemaError，缺会话表或已有表不兼容；DDL/反射错误。
    """
    inspector = inspect(connection)
    if not inspector.has_table("conversations"):
        raise QualitySchemaError("缺少 conversations，请先初始化业务表；本迁移不创建无关表")
    validate_quality_schema(inspector, require_all=False)
    created = []
    for table in QUALITY_TABLES:
        if not inspector.has_table(table.name):
            table.create(connection, checkfirst=True)
            created.append(table.name)
    validate_quality_schema(inspect(connection))
    return created


async def check_quality_tables(config: Settings) -> None:
    """只读检查目标库已完成质量表迁移。参数：config 明确目标；返回：None；异常：结构/数据库错误。"""
    engine, _ = create_database(config)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(lambda sync: validate_quality_schema(inspect(sync)))
    finally:
        with anyio.CancelScope(shield=True):
            await engine.dispose()


async def migrate_quality_tables(config: Settings) -> list[str]:
    """显式迁移两张质量表，UTF-8 会话解析中文枚举，退出保证关闭。

    参数：config 目标 MySQL 连接，调用人明确指定应用/专用测试库。
    返回：list[str]，本次新建表名，重复执行为空。
    异常：QualitySchemaError/数据库连接/DDL 错误，不自动删除或 ALTER；取消清理后传播。
    """
    engine, _ = create_database(config)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET NAMES utf8mb4"))
            return await connection.run_sync(_migrate)
    finally:
        with anyio.CancelScope(shield=True):
            await engine.dispose()


def main() -> None:
    """显式迁移命令。参数：命令行 --test；返回：None；异常：配置/数据库/结构错误。"""
    parser = argparse.ArgumentParser(description="显式创建/校验 CH04 两张质量表，不破坏已有表")
    parser.add_argument("--test", action="store_true", help="只使用 mewhelp_ch04_test")
    args = parser.parse_args()
    config = Settings()
    if args.test:
        if config.knowledge_rag_test_database != "mewhelp_ch04_test":
            raise SystemExit("CH04 测试库必须为 mewhelp_ch04_test")
        config = config.model_copy(update={"mysql_database": config.knowledge_rag_test_database})
    created = asyncio.run(migrate_quality_tables(config))
    print({"database": config.mysql_database, "created": created, "validated": [table.name for table in QUALITY_TABLES]})


if __name__ == "__main__":
    main()
