"""显式、可重复地为messages增加引用JSON；不修改两张质量表或历史正文。"""

import argparse
import asyncio

import anyio
from sqlalchemy import JSON, inspect, text

from app.config import Settings
from app.db.session import create_database


class CitationSchemaError(ValueError):
    """messages不存在或已有引用列不兼容，需要人工核对部署结构。"""


def validate_citation_schema(inspector) -> None:
    """只读验证引用列，禁止自动覆盖错误类型。

    参数：inspector SQLAlchemy结构反射器。
    返回：None，成功表示可空JSON存在。
    异常：CitationSchemaError，缺表/缺列/类型或可空性不符。
    """
    if not inspector.has_table("messages"):
        raise CitationSchemaError("缺少messages，请先显式初始化业务表")
    column = next((item for item in inspector.get_columns("messages") if item["name"] == "rag_metadata"), None)
    if column is None or not isinstance(column["type"], JSON) or column.get("nullable") is not True:
        raise CitationSchemaError("messages.rag_metadata必须为可空JSON，请显式运行或核对引用迁移")


def migrate_connection(connection) -> list[str]:
    """为已有业务库增加唯一引用列，并重新反射验证。

    参数：connection 同步数据库连接，由异步run_sync传入。
    返回：新建列名列表；已兼容时为空。
    异常：CitationSchemaError或DDL错误；不删除表、不清行、不覆盖已有列。
    """
    inspector = inspect(connection)
    if not inspector.has_table("messages"):
        raise CitationSchemaError("缺少messages，请先显式初始化业务表")
    existing = any(column["name"] == "rag_metadata" for column in inspector.get_columns("messages"))
    if not existing:
        connection.execute(text("ALTER TABLE messages ADD COLUMN rag_metadata JSON NULL"))
    validate_citation_schema(inspect(connection))
    return [] if existing else ["messages.rag_metadata"]


async def migrate_citation_metadata(config: Settings) -> list[str]:
    """显式执行指定数据库的增量迁移并保证释放连接。

    参数：config 明确目标MySQL配置。
    返回：本次新增列列表。
    异常：连接、DDL和结构异常透传；取消后清理资源。
    """
    engine, _ = create_database(config)
    try:
        async with engine.begin() as connection:
            return await connection.run_sync(migrate_connection)
    finally:
        with anyio.CancelScope(shield=True):
            await engine.dispose()


def main() -> None:
    """提供显式目标的命令入口。

    参数：命令行--database目标库，或--test固定mewhelp_ch04_test。
    返回：None，打印目标库与新增列。
    异常：配置、数据库或结构不符时退出失败。
    """
    parser = argparse.ArgumentParser(description="显式增加messages引用JSON，不修改质量表")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--database", help="明确指定待迁移数据库")
    target.add_argument("--test", action="store_true", help="仅迁移mewhelp_ch04_test")
    args = parser.parse_args()
    config = Settings().model_copy(update={"mysql_database": "mewhelp_ch04_test" if args.test else args.database})
    print({"database": config.mysql_database, "added": asyncio.run(migrate_citation_metadata(config))})


if __name__ == "__main__":
    main()
