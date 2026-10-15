"""应用全局配置模块单元测试。

验证基础默认配置、环境变量覆盖生效逻辑、未知环境变量忽略以及单例导出正确性。
"""

import pytest
from app.config import Settings, settings


def test_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试在无环境变量传入时，Settings 默认配置项符合任务规范约束。

    参数:
        monkeypatch: pytest 环境清理夹具。

    返回值:
        None: 断言各字段默认值。
    """
    # 清理所有相关的环境变向
    for env_var in [
        "CHAT_BASE_URL",
        "CHAT_MODEL",
        "CHAT_API_KEY",
        "CHAT_TOKEN_BUDGET",
        "EMBED_BASE_URL",
        "EMBED_MODEL",
        "EMBED_API_KEY",
        "RERANK_BASE_URL",
        "RERANK_MODEL",
        "RERANK_API_KEY",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    conf = Settings(_env_file=None)
    assert conf.chat_base_url == "http://8.219.192.36:9000"
    assert conf.chat_model == "claude-opus-4-8"
    assert conf.chat_api_key == "test-token"
    assert conf.chat_token_budget == 2000

    assert conf.embed_base_url is None
    assert conf.embed_model is None
    assert conf.embed_api_key is None

    assert conf.rerank_base_url is None
    assert conf.rerank_model is None
    assert conf.rerank_api_key is None


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试通过外部环境变量覆盖配置时，各字段与类型转换均正常生效。

    参数:
        monkeypatch: pytest 环境变量修改夹具。

    返回值:
        None: 断言覆盖后的配置值与类型。
    """
    monkeypatch.setenv("CHAT_BASE_URL", "http://custom-host:9999")
    monkeypatch.setenv("CHAT_MODEL", "custom-model")
    monkeypatch.setenv("CHAT_API_KEY", "custom-token-123")
    monkeypatch.setenv("CHAT_TOKEN_BUDGET", "4000")
    monkeypatch.setenv("EMBED_BASE_URL", "http://embed-host:9001")
    monkeypatch.setenv("EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("EMBED_API_KEY", "embed-key-456")
    monkeypatch.setenv("RERANK_BASE_URL", "http://rerank-host:9002")
    monkeypatch.setenv("RERANK_MODEL", "bge-reranker-large")
    monkeypatch.setenv("RERANK_API_KEY", "rerank-key-789")

    conf = Settings(_env_file=None)
    assert conf.chat_base_url == "http://custom-host:9999"
    assert conf.chat_model == "custom-model"
    assert conf.chat_api_key == "custom-token-123"
    assert conf.chat_token_budget == 4000
    assert isinstance(conf.chat_token_budget, int)

    assert conf.embed_base_url == "http://embed-host:9001"
    assert conf.embed_model == "bge-m3"
    assert conf.embed_api_key == "embed-key-456"

    assert conf.rerank_base_url == "http://rerank-host:9002"
    assert conf.rerank_model == "bge-reranker-large"
    assert conf.rerank_api_key == "rerank-key-789"


def test_extra_env_vars_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试传入未定义的冗余环境变量时，模型静默忽略且不引发报错。

    参数:
        monkeypatch: pytest 环境变量修改夹具。

    返回值:
        None: 断言未识别字段不作为属性注入。
    """
    monkeypatch.setenv("UNRECOGNIZED_CUSTOM_ENV_VAR", "value")
    conf = Settings(_env_file=None)
    assert not hasattr(conf, "unrecognized_custom_env_var")


def test_settings_singleton_instance() -> None:
    """测试导出的全局 settings 实例为 Settings 的有效单例。

    返回值:
        None: 断言单例类型。
    """
    assert isinstance(settings, Settings)
