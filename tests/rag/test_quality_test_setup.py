"""真实质量测试必须独立准备空库，初始化失败不能被清理错误掩盖。"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


async def test_setup_creates_parent_before_messages_and_quality_tables(monkeypatch):
    """每例能从空库准备父子表，不依赖前一测试。参数：替换器；返回：None；异常：断言错误。"""
    from tests.rag import test_quality_mysql as live
    observed = []

    class Connection:
        """替换DDL外部连接，保留准备函数调用顺序。"""

        async def run_sync(self, callback):
            """执行准备回调。参数：同步回调；返回：结果；异常：回调异常。"""
            return callback(None)

    class Engine:
        """替换建表连接边界。"""

        @asynccontextmanager
        async def begin(self):
            """提供DDL连接。参数：无；返回：连接；异常：无。"""
            yield Connection()

    async def migrate(config):
        """记录质量迁移。参数：配置；返回：None；异常：无。"""
        observed.append("quality")

    async def citations(config):
        """记录引用迁移。参数：配置；返回：None；异常：无。"""
        observed.append("citations")

    monkeypatch.setattr(live.Conversation.__table__, "create", lambda *args, **kwargs: observed.append("conversation"))
    monkeypatch.setattr(live.Message.__table__, "create", lambda *args, **kwargs: observed.append("message"))
    monkeypatch.setattr(live, "migrate_quality_tables", migrate)
    monkeypatch.setattr(live, "migrate_citation_metadata", citations)
    await live.prepare_quality_test_schema(Engine(), None, include_messages=True)
    assert observed == ["conversation", "message", "quality", "citations"]
    observed.clear()
    await live.prepare_quality_test_schema(Engine(), None)
    assert observed == ["conversation", "quality"]


async def test_failed_setup_preserves_original_error_and_disposes(monkeypatch):
    """准备失败时不得对不存在的台账执行DELETE。参数：替换器；返回：None；异常：断言错误。"""
    from tests.rag import test_quality_mysql as live
    from app.config import settings
    disposed = []

    class Engine:
        """仅允许关闭的失败连接边界。"""

        async def dispose(self):
            """记录释放。参数：无；返回：None；异常：无。"""
            disposed.append(True)

    class Factory:
        """初始化失败时绝不应进入数据清理。"""

        def begin(self):
            """阻止掩盖首个错误。参数：无；返回：不返回；异常：AssertionError。"""
            raise AssertionError("不应清理尚未写入的记录")

    async def fail(*args, **kwargs):
        """注入准备错误。参数：任意；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("original setup failure")

    monkeypatch.setattr(live, "Settings", lambda **kwargs: settings)
    monkeypatch.setattr(live, "create_database", lambda config: (Engine(), Factory()))
    monkeypatch.setattr(live, "migrate_quality_tables", fail)
    monkeypatch.setattr(live, "prepare_quality_test_schema", fail, raising=False)
    with pytest.raises(RuntimeError, match="original setup failure"):
        await live.test_faith_case_concurrent_recurrence_updates_latest_snapshot_without_losing_history()
    assert disposed == [True]
