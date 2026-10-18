"""真实应用 lifespan 的恢复门禁与资源释放验证。"""

from contextlib import asynccontextmanager
import pytest
from fastapi.testclient import TestClient


def test_lifespan_recovery_failure_blocks_startup(monkeypatch):
    """恢复失败不能接受聊天，仍须释放连接。参数：替换器；返回：None；异常：断言失败。"""
    from app import main
    from app.repositories.chat import ChatRepository
    disposed = []
    class Engine:
        """记录资源生命周期的外部边界。"""
        async def dispose(self):
            """记录关闭。参数：无；返回：None；异常：无。"""
            disposed.append(True)
    async def broken_recovery(self):
        """模拟恢复事务不可用。参数：仓库自身；返回：不返回；异常：ConnectionError。"""
        raise ConnectionError("recovery unavailable")
    monkeypatch.setattr(main, "create_database", lambda: (Engine(), None), raising=False)
    monkeypatch.setattr(ChatRepository, "recover_interrupted_turns", broken_recovery)
    app = main.create_app(runtime_factory=main.database_runtime)
    with pytest.raises(ConnectionError):
        with TestClient(app):
            pytest.fail("恢复失败后不应进入服务状态")
    assert disposed == [True] and not hasattr(app.state, "chat_service")


def test_runtime_available_only_within_lifespan():
    """依赖需由 lifespan 提供并关闭。参数：无；返回：None；异常：断言失败。"""
    from app.main import create_app
    states = []
    @asynccontextmanager
    async def runtime():
        """提供可追踪运行时。参数：无；返回：服务；异常：无。"""
        states.append("open")
        try:
            yield object()
        finally:
            states.append("closed")
    app = create_app(runtime_factory=runtime)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert hasattr(app.state, "chat_service")
    assert states == ["open", "closed"] and not hasattr(app.state, "chat_service")
