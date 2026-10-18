"""应用全局配置模块单元测试。

验证基础默认配置、环境变量覆盖生效逻辑、未知环境变量忽略以及单例导出正确性。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError
from app.config import Settings, settings


@pytest.fixture(autouse=True)
def configured_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """为配置测试提供完整的虚拟账号信息。

    参数:
        monkeypatch: 隔离进程环境变量的测试夹具。
    返回值:
        None: 将三个必填项写入测试环境。
    异常:
        无业务异常。
    """
    monkeypatch.setenv("CHAT_BASE_URL", "http://test.invalid/v1")
    monkeypatch.setenv("CHAT_MODEL", "test-model")
    monkeypatch.setenv("CHAT_API_KEY", "test-key")


@pytest.mark.parametrize("field", ["CHAT_BASE_URL", "CHAT_MODEL", "CHAT_API_KEY"])
def test_missing_chat_configuration_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """验证任何账号必填项缺失时启动即报错。

    参数:
        monkeypatch: 环境变量夹具。
        field: 本次移除的必填环境变量名称。
    返回值:
        None: 断言错误指向缺失配置。
    异常:
        AssertionError: 缺失配置被示例值替代时抛出。
    """
    monkeypatch.delenv(field)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    assert any(item["loc"] == (field.lower(),) for item in error.value.errors())


@pytest.mark.parametrize("budget", [0, -1])
def test_token_budget_must_be_positive(budget: int) -> None:
    """验证上下文预算不能配置为零或负数。

    参数:
        budget: 不合法的预算值。
    返回值:
        None: 断言配置校验拒绝该值。
    异常:
        AssertionError: 非法预算被接受时抛出。
    """
    with pytest.raises(ValidationError):
        Settings(_env_file=None, token_budget=budget)


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
        "TOKEN_BUDGET",
        "CHAT_TOKEN_BUDGET",
        "EMBED_BASE_URL",
        "EMBED_MODEL",
        "EMBED_API_KEY",
        "RERANK_BASE_URL",
        "RERANK_MODEL",
        "RERANK_API_KEY",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    conf = Settings(
        _env_file=None, chat_base_url="http://test.invalid/v1",
        chat_model="test-model", chat_api_key="test-key",
    )
    assert conf.chat_base_url == "http://test.invalid/v1"
    assert conf.chat_model == "test-model"
    assert conf.chat_api_key == "test-key"
    assert conf.token_budget == 2000

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
    monkeypatch.setenv("TOKEN_BUDGET", "4000")
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
    assert conf.token_budget == 4000
    assert isinstance(conf.token_budget, int)

    assert conf.embed_base_url == "http://embed-host:9001"
    assert conf.embed_model == "bge-m3"
    assert conf.embed_api_key == "embed-key-456"

    assert conf.rerank_base_url == "http://rerank-host:9002"
    assert conf.rerank_model == "bge-reranker-large"
    assert conf.rerank_api_key == "rerank-key-789"


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_legacy_token_budget_compatible(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """验证旧预算配置仍可读取，且新旧配置同时出现时优先使用新名称。

    参数:
        monkeypatch: pytest 环境变量修改夹具。
        source: 配置来源，取 environment 或 dotenv。

    返回值:
        None: 断言兼容旧配置与新名称的优先级。

    异常:
        AssertionError: 预算配置读取结果不符时抛出。
        OSError: 临时配置文件无法写入时抛出。
    """
    if source == "environment":
        monkeypatch.setenv("CHAT_TOKEN_BUDGET", "3500")
        assert Settings(_env_file=None).token_budget == 3500
        monkeypatch.setenv("TOKEN_BUDGET", "4500")
        assert Settings(_env_file=None).token_budget == 4500
    else:
        # 使用独立临时目录，避免本机公共 pytest 临时目录的权限影响验证。
        with TemporaryDirectory(prefix="mewhelp-config-") as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("CHAT_TOKEN_BUDGET=3500\n", encoding="utf-8")
            assert Settings(_env_file=env_file).token_budget == 3500
            env_file.write_text(
                "CHAT_TOKEN_BUDGET=3500\nTOKEN_BUDGET=4500\n", encoding="utf-8"
            )
            assert Settings(_env_file=env_file).token_budget == 4500


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


@pytest.mark.parametrize("values", [
    {"mysql_port": 0}, {"mysql_port": 65536}, {"tool_timeout_seconds": 0},
    {"tool_max_retries": 2}, {"tool_max_retries": -1}, {"tool_retry_delay_seconds": -1},
])
def test_invalid_database_and_tool_limits(values: dict) -> None:
    """拒绝无效端口、无限执行或额外重试配置。参数：非法配置组合；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)
