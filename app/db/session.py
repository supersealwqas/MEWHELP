"""构造异步 MySQL 连接和会话工厂，不在导入时连接数据库。"""

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, settings


def create_database(config: Settings = settings, *, database: str | None = None) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """创建连接资源。参数：config 配置，database 可覆盖库名；返回：引擎及会话工厂；异常：密码缺失时 ValueError。"""
    password = config.mysql_password.get_secret_value()
    if not password:
        raise ValueError("请先在 .env 配置 MYSQL_PASSWORD 并初始化 MySQL")
    url = URL.create("mysql+asyncmy", username=config.mysql_user, password=password,
                     host=config.mysql_host, port=config.mysql_port,
                     database=database or config.mysql_database, query={"charset": "utf8mb4"})
    engine = create_async_engine(url, pool_pre_ping=True, pool_recycle=1800,
                                 connect_args={"connect_timeout": 5, "init_command": "SET time_zone = '+00:00'"})
    return engine, async_sessionmaker(engine, expire_on_commit=False)
