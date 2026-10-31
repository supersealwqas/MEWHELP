"""CH10具名固定迁移：默认只读预览，已有结构不兼容即拒绝。"""

import argparse
import asyncio
from pathlib import Path
import re

from sqlalchemy import inspect, text

from app.config import Settings
from app.db.session import create_database
from app.db.migrate_rag_quality import QualitySchemaError, _validate_table
from app.knowledge.lease import finish_cleanup
from app.models.topics import TopicClassification


def validate_ch10_schema(inspector, *, require_all=True):
    """核对固定字段、约束和中文注释。参数：同步反射/是否要求表存在；返回：None；异常：结构不兼容或数据库错误。"""
    if not inspector.has_table("low_confidence_questions"):
        raise QualitySchemaError("缺少 low_confidence_questions，请先完成问题池迁移")
    columns = {r["name"]: r for r in inspector.get_columns("low_confidence_questions")}
    primary = inspector.get_pk_constraint("low_confidence_questions").get("constrained_columns")
    wanted = str(TopicClassification.__table__.c.question_id.type.compile(dialect=inspector.bind.dialect)).upper()
    if ("id" not in columns or primary != ["id"] or columns["id"]["nullable"]
            or str(columns["id"]["type"].compile(dialect=inspector.bind.dialect)).upper() != wanted):
        raise QualitySchemaError("问题池主键必须为 BIGINT UNSIGNED NOT NULL")
    table = TopicClassification.__table__
    if not inspector.has_table(table.name):
        if require_all:
            raise QualitySchemaError("缺少 topic_classifications，请显式运行 CH10 迁移")
        return
    _validate_table(inspector, table)
    actual = {row["name"]: row for row in inspector.get_columns(table.name)}
    if any(actual[column.name].get("comment") != column.comment for column in table.columns):
        raise QualitySchemaError("主题表字段中文 COMMENT 不兼容")
    if inspector.get_table_comment(table.name).get("text") != table.comment:
        raise QualitySchemaError("主题表中文 COMMENT 不兼容")


def migrate_connection(connection, *, apply=False):
    """预检后仅创建缺失主题表。参数：同步MySQL连接/显式apply；返回：待执行SQL列表；异常：结构、方言或DDL错误。"""
    if connection.dialect.name != "mysql":
        raise ValueError("CH10 迁移只支持 MySQL")
    inspector = inspect(connection)
    validate_ch10_schema(inspector, require_all=False)
    if inspector.has_table("topic_classifications"):
        return []
    source = (Path(__file__).resolve().parents[2] / "sql/ch10-ddl.sql").read_text(encoding="utf-8")
    statements = re.findall(r"CREATE TABLE topic_classifications\s*\(.*?\) ENGINE=.*?;", source, re.S)
    if len(statements) != 1:
        raise ValueError("CH10 固定 DDL 已改变，需核查")
    if apply:
        connection.execute(text("SET NAMES utf8mb4"))
        # COMMENT中标点不得被SQLAlchemy解析成绑定参数。
        connection.exec_driver_sql(statements[0].rstrip(";"))
        validate_ch10_schema(inspect(connection))
    return statements


async def check_ch10_tables(config):
    """开启旁路时只读检查结构。参数：应用Settings；返回：None；异常：结构/库名/连接错误。"""
    engine, _ = create_database(config)
    try:
        async with engine.connect() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != config.mysql_database:
                raise ValueError("CH10 实际数据库与配置不一致")
            await connection.run_sync(lambda sync: validate_ch10_schema(inspect(sync)))
    finally:
        await finish_cleanup(engine.dispose())


async def migrate(config, database, *, apply=False):
    """只允许应用库或专库，核对连接目标后预览/执行。参数：配置/明确库名/apply；返回：SQL列表；异常：输入/结构/数据库错误。"""
    if not database or database not in {config.mysql_database, "mewhelp_ch10_test"}:
        raise ValueError("CH10 目标必须为配置应用库或 mewhelp_ch10_test")
    engine, _ = create_database(config, database=database)
    try:
        async with engine.begin() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != database:
                raise ValueError("CH10 实际数据库与显式目标不一致")
            return await connection.run_sync(lambda sync: migrate_connection(sync, apply=apply))
    finally:
        await finish_cleanup(engine.dispose())


def main():
    """运行具名默认只读CLI。参数：--env-file/--database/--apply；返回：None；异常：失败非零退出。"""
    parser = argparse.ArgumentParser(description="CH10 固定主题表迁移，默认预览")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        statements = asyncio.run(migrate(Settings(_env_file=args.env_file), args.database, apply=args.apply))
    except Exception as error:
        # 只打印类型，不泄露凭据或原始问题。
        parser.exit(1, f"CH10 迁移失败：{type(error).__name__}\n")
    print(dict(database=args.database, applied=args.apply, statements=statements))


if __name__ == "__main__":
    main()
