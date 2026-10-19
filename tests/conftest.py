"""全局 Pytest 夹具：隔离离线配置与聊天运行时，保留显式集成配置来源。"""

import os
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from pydantic_settings import DotEnvSettingsSource


# 收集测试之前保护配置模块的第一次导入；环境与 dotenv 补丁只在此代码块内有效。
# 显式传入 _env_file 的配置专项测试和 MySQL 夹具随后仍使用真实加载规则。
with patch.dict(os.environ, {
    "CHAT_BASE_URL": "http://test.invalid/v1",
    "CHAT_MODEL": "test-model",
    "CHAT_API_KEY": "test-key",
}, clear=True), patch.object(DotEnvSettingsSource, "_read_env_files", return_value={}):
    from app import config

    _OFFLINE_SETTINGS = config.Settings(
        _env_file=None, chat_base_url="http://test.invalid/v1",
        chat_model="test-model", chat_api_key="test-key",
    )
    config.settings = _OFFLINE_SETTINGS


@pytest.fixture
def offline_settings(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """为离线消费者注入完整虚拟配置，覆盖模块已保存的旧引用。

    参数:
        monkeypatch: pytest 提供的属性替换夹具，测试结束后自动还原。
        request: 当前测试请求；带 mysql 标记时保留显式集成配置。
    返回值:
        Settings | None: 每个离线测试独立的配置；MySQL 测试返回 None。
    异常:
        pydantic.ValidationError: 测试配置与生产字段约束不一致时抛出。
    """
    if request.node.get_closest_marker("mysql"):
        return None
    from app import main
    from app.core import llm
    from app.db import session

    # 所有字段显式传入，防止主机环境覆盖预算、工具限制或数据库默认值。
    settings = config.Settings(_env_file=None, **_OFFLINE_SETTINGS.model_dump())
    for module in (config, main, llm, session):
        monkeypatch.setattr(module, "settings", settings)
    return settings


@pytest.fixture(autouse=True)
def offline_chat_runtime(monkeypatch, request, offline_settings):
    """提供离线聊天边界。参数：替换器、测试请求、隔离配置；返回：服务依赖；异常：无。"""
    if request.node.get_closest_marker("mysql"):
        yield None
        return
    from app import main
    from app.services.chat import ChatService
    from app.tools.registry import build_registry
    from tests.chat_fakes import FAQRepository, MemoryRepository, ScriptedModel
    repository, support, model = MemoryRepository(), FAQRepository(), ScriptedModel()
    service = ChatService(repository, support, model, build_registry(), offline_settings)

    @asynccontextmanager
    async def runtime():
        """提供测试专用边界。参数：无；返回：真实聊天服务；异常：无。"""
        yield service
    monkeypatch.setattr(main, "create_runtime", runtime)
    yield service, repository, support, model
