"""全局 Pytest 测试配置文件与 Fixture 夹具定义。"""

import os
from contextlib import asynccontextmanager
import pytest


# 离线测试进程使用虚拟凭据，模块导入期间也不会读取真实账号进行模型初始化。
os.environ.update({
    "CHAT_BASE_URL": "http://test.invalid/v1",
    "CHAT_MODEL": "test-model",
    "CHAT_API_KEY": "test-key",
})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """在每个测试用例运行前，隔离关键环境变量，避免开发环境的 .env 文件影响单测默认值判定。

    参数:
        monkeypatch: pytest 提供的环境修改夹具。
    """
    keys_to_clean = [
        "CHAT_BASE_URL",
        "CHAT_MODEL",
        "CHAT_API_KEY",
        "TOKEN_BUDGET",
        "CHAT_TOKEN_BUDGET",
        "EMBED_BASE_URL",
        "EMBED_MODEL",
        "EMBED_API_KEY",
        "RERANK_BASE_URL",
        "RERANK_MODEL",
        "RERANK_API_KEY",
    ]
    # 仅备份并临时清理可能影响配置单测的变量
    for key in keys_to_clean:
        if key in os.environ:
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def offline_chat_runtime(monkeypatch, request):
    """离线测试不连接外部数据库。参数：替换器、测试请求；返回：服务依赖；异常：无。"""
    if request.node.get_closest_marker("mysql"):
        yield None
        return
    from app import main
    from app.config import settings
    from app.services.chat import ChatService
    from app.tools.registry import build_registry
    from tests.chat_fakes import FAQRepository, MemoryRepository, ScriptedModel
    repository, support, model = MemoryRepository(), FAQRepository(), ScriptedModel()
    service = ChatService(repository, support, model, build_registry(), settings)

    @asynccontextmanager
    async def runtime():
        """提供测试专用边界。参数：无；返回：真实聊天服务；异常：无。"""
        yield service
    monkeypatch.setattr(main, "create_runtime", runtime)
    yield service, repository, support, model
