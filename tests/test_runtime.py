"""真实应用 lifespan 的恢复门禁与资源释放验证。"""

from contextlib import asynccontextmanager
import pytest
from fastapi.testclient import TestClient


def test_lifespan_recovery_failure_blocks_startup(monkeypatch, tmp_path, offline_settings):
    """新对账路径读取失败不能接受聊天，仍关闭保存器及外部资源。参数：替换器/隔离目录/配置；返回：None；异常：断言失败。"""
    from app import main
    from app.repositories.chat import ChatRepository
    disposed = []
    observed = []
    original_workflow_runtime, original_service = main.workflow_runtime, main.ChatService

    @asynccontextmanager
    async def checked_workflow_runtime(deps, path):
        """保留真实保存器并核对生产依赖共享。参数：实例依赖/路径；返回：图；异常：原运行时故障。"""
        assert deps.registry is deps.executor.registry is deps.mcp.registry
        assert deps.permissions is deps.executor.permissions is deps.mcp.permissions
        observed.append(deps)
        async with original_workflow_runtime(deps, path) as graph:
            yield graph

    def checked_service(repository, graph, executor, config, **options):
        """保留真实CH05服务并核对统一执行器。参数：服务资源；返回：服务；异常：原初始化错误。"""
        service = original_service(repository, graph, executor, config, **options)
        assert service.executor is observed[0].executor
        return service
    class Engine:
        """记录资源生命周期的外部边界。"""
        url = "mysql+asyncmy://test/isolated"

        def __init__(self, name):
            """标记被关闭的资源。参数：name测试资源名；返回：None；异常：无。"""
            self.name = name

        async def dispose(self):
            """记录关闭。参数：无；返回：None；异常：无。"""
            disposed.append(self.name)
    async def broken_recovery(self):
        """模拟恢复事务不可用。参数：仓库自身；返回：不返回；异常：ConnectionError。"""
        raise ConnectionError("recovery unavailable")
    @asynccontextmanager
    async def search(config, store):
        """替换模型/索引外部资源，保留真实启动与检查点。参数：配置/仓储；返回：占位资源；异常：原启动错误。"""
        try:
            from types import SimpleNamespace
            from tests.knowledge.fakes import FakeEncoder, MemoryVectorIndex
            yield SimpleNamespace(encoder=FakeEncoder(), index=MemoryVectorIndex())
        finally:
            disposed.append("search")

    offline_settings.workflow_checkpoint_path = str(tmp_path / "startup.sqlite")
    async def checked_schema(config):
        """隔离新增 MySQL 核验边界，保留恢复流程。参数：配置；返回：None；异常：无。"""

    monkeypatch.setattr(main, "check_ch09_tables", checked_schema)
    monkeypatch.setattr(main, "create_database", lambda: (Engine("business"), None))
    monkeypatch.setattr(main, "create_async_engine", lambda *args, **kwargs: Engine("lease"))
    monkeypatch.setattr(main, "online_search_runtime", search)
    monkeypatch.setattr(main, "workflow_runtime", checked_workflow_runtime)
    monkeypatch.setattr(main, "ChatService", checked_service)
    monkeypatch.setattr(ChatRepository, "list_pending_workflow_turns", broken_recovery)
    app = main.create_app(runtime_factory=main.database_runtime)
    with pytest.raises(ConnectionError):
        with TestClient(app):
            pytest.fail("恢复失败后不应进入服务状态")
    assert disposed == ["search", "lease", "business"] and not hasattr(app.state, "chat_service")
    assert len(observed) == 1 and observed[0].mcp._closed


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


def test_evaluation_owner_lives_and_closes_with_application(monkeypatch, tmp_path):
    """应用没有拥有或关闭评估管理器时失败。参数：隔离报告目录；返回：None；异常：断言失败。"""
    from app.api import rageval
    from app.main import create_app

    @asynccontextmanager
    async def runtime():
        """隔离聊天外部资源。参数：无；返回：服务占位；异常：无。"""
        yield object()

    monkeypatch.setattr(rageval, "REPORTS_ROOT", tmp_path)
    monkeypatch.setattr(rageval, "JOB_STATE", tmp_path / "state.json")
    monkeypatch.setattr(rageval, "JOB_LOG", tmp_path / "job.log")
    application = create_app(runtime_factory=runtime)
    with TestClient(application):
        assert hasattr(application.state, "evaluation_jobs")
        manager = application.state.evaluation_jobs
        assert manager.status()["status"] == "idle"
    assert not hasattr(application.state, "evaluation_jobs")
    with pytest.raises(RuntimeError, match="关闭"):
        manager.start(tmp_path / "dataset.json", ["dense"])


async def test_ch05_runtime_rejects_explicit_dense_before_opening_resources(monkeypatch, offline_settings):
    """显式dense配置悄悄换检索器或继续启动时失败。参数：隔离配置/替换器；返回：None；异常：预期ValueError。"""
    from app import main
    offline_settings.knowledge_retrieval_mode = "dense"

    def forbidden_database():
        """配置拒绝前不应接数据库。参数：无；返回：无；异常：AssertionError。"""
        raise AssertionError("不应创建数据库")

    monkeypatch.setattr(main, "create_database", forbidden_database)
    with pytest.raises(ValueError, match="hybrid"):
        async with main.database_runtime():
            pytest.fail("不应启动")


async def test_workflow_file_failure_blocks_runtime_without_memory_fallback(tmp_path):
    """checkpoint路径失败悄悄降级内存时失败。参数：tmp_path目录；返回：None；异常：预期OSError。"""
    from app.workflow.runtime import workflow_runtime
    from tests.workflow.graph_fakes import dependencies
    invalid = tmp_path / "occupied"
    invalid.write_text("文件不是目录", encoding="utf-8")
    with pytest.raises(OSError):
        async with workflow_runtime(dependencies(), invalid / "workflow.sqlite3"):
            pytest.fail("不应启动")
