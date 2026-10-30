"""显式创建/校验用户 CH04 两张质量表的非破坏性迁移。

先检查已有表的字段、索引、枚举、默认和外键，兼容才创建缺失表；MySQL DDL
不是整体可回滚事务，允许部分成功后重跑，但不擅自 ALTER/drop/recreate。
命令入口可显式使用 mewhelp_ch04_test，两类旧测试库不复用。
"""

from __future__ import annotations

import argparse
import asyncio

import anyio
from sqlalchemy import Column, Enum, ForeignKey, Index, MetaData, Table, Text, UniqueConstraint, inspect, text
from sqlalchemy.dialects import mysql

from app.config import Settings
from app.db.session import create_database
from app.models.rag_quality import QUALITY_TABLES
from app.models.customer_service import Conversation, TABLE_OPTIONS


# CH04 的首次建表定义固定在本章版本，不随后续 ORM 新字段或跨元数据外键演进。
CH04_LOW_CONFIDENCE = Table(
    "low_confidence_questions", MetaData(),
    Column("id", mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键"),
    Column("conversation_id", mysql.BIGINT(unsigned=True), ForeignKey(Conversation.__table__.c.id, name="fk_lcq_conversation"), nullable=True, comment="来源会话"),
    Column("raw_question", Text, nullable=False, comment="用户原话,带情绪口语"),
    Column("source", Enum("retrieval_low_conf", "self_check", "user_feedback"), nullable=False, comment="入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决"),
    Column("reason", Text, nullable=True, comment="判不能的原因,留作复盘"),
    Column("created_at", mysql.DATETIME(), nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="入池时间"),
    Index("idx_source", "source"), Index("idx_created_at", "created_at"),
    **TABLE_OPTIONS, comment="低置信度问题池",
)
CH04_QUALITY_TABLES = (CH04_LOW_CONFIDENCE, QUALITY_TABLES[1])


class QualitySchemaError(RuntimeError):
    """质量表已有结构不兼容或依赖缺失，需要操作人核查，不能自动破坏性修复。"""


def _default(value: object) -> str | None:
    """统一 MySQL 反射默认值写法。参数：value 默认表达式；返回：规范字符串或 None；异常：无。"""
    if value is None:
        return None
    return " ".join(str(value).strip().strip("'").lower().replace("()", "").split())


def _validate_table(inspector, table, *, optional_columns=(), optional_indexes=(), optional_foreign_keys=()) -> None:
    """检查一张已有表与用户定义逐字段兼容。

    参数：inspector 同步反射对象；table 期望表；optional_* 为 CH09 续迁时可暂缺的新增约束名称。
    返回：None，结构满足约束。
    异常：QualitySchemaError，列/类型/索引/枚举/默认/编码/外键不一致；底层反射错误。
    """
    name = table.name
    columns = {column["name"]: column for column in inspector.get_columns(name)}
    if not set(table.columns.keys()) - set(optional_columns) <= set(columns) <= set(table.columns.keys()):
        raise QualitySchemaError(f"{name} 字段集合不兼容，不自动修改已有表")
    for expected in table.columns:
        if expected.name not in columns:
            continue
        actual = columns[expected.name]
        actual_type, wanted = actual["type"], expected.type
        actual_compiled = str(actual_type.compile(dialect=mysql.dialect())).upper()
        wanted_compiled = str(wanted.compile(dialect=mysql.dialect())).upper()
        if actual_compiled != wanted_compiled or actual["nullable"] != expected.nullable or actual.get("computed"):
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
    expected_indexes = {index.name: (list(index.columns.keys()), index.unique) for index in table.indexes}
    expected_unique = {item.name: list(item.columns.keys()) for item in table.constraints if isinstance(item, UniqueConstraint)}
    expected_indexes.update({key: (value, True) for key, value in expected_unique.items()})
    implicit_indexes = set()
    for foreign_key in table.foreign_key_constraints:
        foreign_columns = list(foreign_key.columns.keys())
        if not any(columns[:len(foreign_columns)] == foreign_columns for columns, _ in expected_indexes.values()):
            # InnoDB 会为未被显式索引覆盖的外键生成同名普通索引；仍逐项核对列与属性。
            expected_indexes[foreign_key.name] = (foreign_columns, False)
            implicit_indexes.add(foreign_key.name)
    if not set(expected_indexes) - set(optional_indexes) - implicit_indexes <= set(indexes) <= set(expected_indexes):
        raise QualitySchemaError(f"{name} 索引集合不兼容")
    for key, actual in indexes.items():
        wanted_columns, unique = expected_indexes[key]
        options = actual.get("dialect_options", {})
        if (actual.get("column_names") != wanted_columns or bool(actual.get("unique")) != unique
                or options.get("mysql_length") or options.get("mysql_prefix")
                or str(options.get("mysql_using", "BTREE")).upper() != "BTREE"):
            raise QualitySchemaError(f"{name} 索引 {key} 不兼容")
    actual_unique = {item["name"]: item["column_names"] for item in inspector.get_unique_constraints(name)}
    if actual_unique != expected_unique:
        raise QualitySchemaError(f"{name} 唯一约束不兼容")
    foreign = {item["name"]: item for item in inspector.get_foreign_keys(name)}
    wanted_foreign = {item.name: item for item in table.foreign_key_constraints}
    if not set(wanted_foreign) - set(optional_foreign_keys) <= set(foreign) <= set(wanted_foreign):
        raise QualitySchemaError(f"{name} 外键集合不兼容")
    for key, actual in foreign.items():
        expected = wanted_foreign[key]
        actions = actual.get("options", {})
        # MySQL 未指定动作与 RESTRICT/NO ACTION 等价；SET NULL 必须明确保留。
        ondelete = str(actions.get("ondelete", "RESTRICT")).upper()
        onupdate = str(actions.get("onupdate", "RESTRICT")).upper()
        if (actual.get("constrained_columns") != list(expected.columns.keys())
                or actual.get("referred_table") != expected.referred_table.name
                or actual.get("referred_columns") != [item.column.name for item in expected.elements]
                or actual.get("referred_schema") != expected.referred_table.schema
                or ondelete not in ({expected.ondelete} if expected.ondelete else {"RESTRICT", "NO ACTION"})
                or onupdate not in {"RESTRICT", "NO ACTION"}):
            raise QualitySchemaError(f"{name} 外键 {key} 不兼容")
    if hasattr(inspector, "get_check_constraints") and inspector.get_check_constraints(name):
        raise QualitySchemaError(f"{name} 含固定 DDL 未规定的检查约束")
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
            if (table.name == "low_confidence_questions"
                    and {item["name"] for item in inspector.get_columns(table.name)} == set(CH04_LOW_CONFIDENCE.c.keys())):
                table = CH04_LOW_CONFIDENCE
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
    for table in CH04_QUALITY_TABLES:
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
