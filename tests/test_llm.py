"""大语言模型客户端工厂测试，验证连接配置与显式参数透传。"""

from langchain_openai import ChatOpenAI

from app.config import settings
from app.core import get_chat_model
from app.core.llm import get_chat_model as direct_get_chat_model


def test_get_chat_model_default_configuration() -> None:
    """验证默认模型连接配置及包导出一致性。

    参数:
        无。

    返回值:
        None: 通过断言验证全局配置及默认采样参数。

    异常:
        AssertionError: 工厂返回值与预期不符时抛出。
    """
    model = get_chat_model()
    assert get_chat_model is direct_get_chat_model
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == settings.chat_model
    assert str(model.openai_api_base).rstrip("/") == settings.chat_base_url.rstrip("/")
    assert model.openai_api_key.get_secret_value() == settings.chat_api_key
    assert model.temperature == 0.0
    assert model.streaming is False
    assert model.request_timeout == 60.0
    assert model.max_retries == 1


def test_get_chat_model_custom_parameters() -> None:
    """验证统一工厂支持显式模型、流式标志、温度与额外客户端参数。

    参数:
        无。

    返回值:
        None: 通过断言验证各参数被正确透传。

    异常:
        AssertionError: 任一参数未被正确透传时抛出。
    """
    model = get_chat_model(
        streaming=True,
        model="test-model",
        temperature=0.7,
        max_tokens=512,
        request_timeout=15.0,
    )
    assert model.model_name == "test-model"
    assert model.streaming is True
    assert model.temperature == 0.7
    assert model.max_tokens == 512
    assert model.request_timeout == 15.0
