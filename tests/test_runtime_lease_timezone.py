"""验证主应用公共持锁引擎的 MySQL 会话时区，构造真实引擎但不连接数据库。"""

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool


async def test_application_lease_engine_initializes_utc(monkeypatch, offline_settings):
    """公共 lease 引擎遗漏 UTC 初始化或失去连接超时时失败。

    参数：monkeypatch 隔离外部依赖；offline_settings 独立应用配置。
    返回：None。
    异常：预期 RuntimeError 在任何真实数据库或向量调用前停止启动。
    """
    from app import main
    engine = create_async_engine("mysql+asyncmy://offline:offline@127.0.0.1/unused")
    constructed = []

    def lease_engine(url, **options):
        """记录生产实际连接选项。参数：URL 与引擎选项；返回：真实引擎；异常：配置错误。"""
        created = create_async_engine(url, **options)
        constructed.append((created, options))
        return created

    async def checked_schema(settings):
        """隔离数据库结构核验。参数：设置；返回：None；异常：无。"""

    @asynccontextmanager
    async def stopped_search(settings, store):
        """在外部搜索启动前终止。参数：配置和仓储；返回：不产出；异常：RuntimeError。"""
        raise RuntimeError("offline assembly boundary")
        yield

    monkeypatch.setattr(main, "create_database", lambda: (engine, None))
    monkeypatch.setattr(main, "check_ch09_tables", checked_schema)
    monkeypatch.setattr(main, "create_async_engine", lease_engine)
    monkeypatch.setattr(main, "online_search_runtime", stopped_search)
    with pytest.raises(RuntimeError, match="offline assembly boundary"):
        async with main.database_runtime():
            pytest.fail("测试不能启动外部服务")
    assert len(constructed) == 1
    created, options = constructed[0]
    assert isinstance(created.pool, NullPool)
    assert options["connect_args"] == {"connect_timeout": 5, "init_command": "SET time_zone = '+00:00'"}
