"""构造异步 MySQL 连接和会话工厂，不在导入时连接数据库。"""

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, settings


def create_database(config: Settings | None = None, *, database: str | None = None) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """创建异步数据库连接资源，不在此处连接数据库。

    参数:
        config: 显式配置；为 None 时使用调用时的模块配置。
        database: 可选数据库名；为空时使用配置中的业务数据库名。
    返回值:
        tuple[AsyncEngine, async_sessionmaker[AsyncSession]]: 引擎及会话工厂。
    异常:
        ValueError: MySQL 密码缺失时抛出。
        sqlalchemy.exc.ArgumentError、ImportError: 引擎配置或驱动不可用时抛出。
    """
    # 运行时取值，避免函数默认参数永久保留导入时的旧配置。
    if config is None:
        config = settings
    password = config.mysql_password.get_secret_value()
    if not password:
        raise ValueError("请先在 .env 配置 MYSQL_PASSWORD 并初始化 MySQL")
    url = URL.create("mysql+asyncmy", username=config.mysql_user, password=password,
                     host=config.mysql_host, port=config.mysql_port,
                     database=database or config.mysql_database, query={"charset": "utf8mb4"})
    engine = create_async_engine(url, pool_pre_ping=True, pool_recycle=1800,
                                 connect_args={"connect_timeout": 5, "init_command": "SET time_zone = '+00:00'"})
    return engine, async_sessionmaker(engine, expire_on_commit=False)
