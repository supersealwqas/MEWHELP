"""大语言模型 (LLM) 客户端工厂模块。

基于全局配置统一创建并返回配置完备的 LangChain ChatOpenAI 客户端实例，
隔离上游模型服务提供方的连接凭证与端点细节。
"""

from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings


def get_chat_model(
    streaming: bool = False,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ChatOpenAI:
    """获取配置统一的大语言模型 ChatOpenAI 客户端实例工厂函数。

    依据应用全局配置 settings 中的对话模型端点、模型名称以及 API 密钥，
    初始化并返回用于客服对话生成或售后字段结构化提取的 ChatOpenAI 实例。

    参数:
        streaming: 是否启用流式响应，默认为 False。
        model: 指定模型名称；为 None 或空字符串时沿用全局配置。
        temperature: 模型采样的多样性温度系数，取值通常在 0.0 到 2.0 之间。
            默认值为 0.0，以保证提取与客服问答的确定性与稳定性。
        **kwargs: 额外透传给 ChatOpenAI 构造器的关键字参数；默认请求超时为 60 秒、
            最多重试 1 次，显式传入的 request_timeout、max_retries 可覆盖默认值。

    返回值:
        ChatOpenAI: 已注入 base_url、model、api_key 及指定 temperature 的对话模型客户端实例。

    异常:
        pydantic.ValidationError: 若传入的关键字参数与 ChatOpenAI 字段定义冲突时可能抛出。
    """
    # 为上游请求设置有限等待和重试次数；显式传入的调用参数仍优先。
    kwargs.setdefault("request_timeout", 60.0)
    kwargs.setdefault("max_retries", 1)
    # 实例化 ChatOpenAI，统一绑定全局配置中的端点、模型标识与鉴权密钥
    return ChatOpenAI(
        base_url=settings.chat_base_url,
        model=model or settings.chat_model,
        api_key=settings.chat_api_key,
        temperature=temperature,
        streaming=streaming,
        **kwargs,
    )
