"""显式建表及种子初始化命令，应用启动不会自动清库或重灌数据。"""

import argparse
import asyncio
from sqlalchemy import text
from app.config import settings
from app.db.seed import seed_database
from app.db.session import create_database
from app.models.customer_service import Base


async def initialize_database(test: bool = False, *, reset: bool = False) -> None:
    """建立表并补种子。参数：test 选择测试库，reset 仅允许测试库重建；返回：None；异常：错误目标 ValueError 或数据库错误。"""
    if reset and not test:
        raise ValueError("--reset 必须同时指定 --test，禁止重建应用库")
    if test and settings.mysql_test_database != "mewhelp_test":
        raise ValueError("测试库必须为 mewhelp_test")
    engine, factory = create_database(database="mewhelp_test" if test else None)
    try:
        async with engine.begin() as connection:
            # 不只相信配置；删表前必须在同一连接上核实数据库实际身份。
            if test and await connection.scalar(text("SELECT DATABASE()")) != "mewhelp_test":
                raise ValueError("实际连接不是 mewhelp_test，已拒绝初始化")
            if reset:
                await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
        await seed_database(factory)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="显式创建四张表并补充演示数据")
    parser.add_argument("--test", action="store_true", help="仅初始化 mewhelp_test")
    parser.add_argument("--reset", action="store_true", help="仅在 --test 下重建测试表；请先停止测试库服务")
    arguments = parser.parse_args()
    asyncio.run(initialize_database(arguments.test, reset=arguments.reset))
    print("四张表和演示数据初始化完成。")
