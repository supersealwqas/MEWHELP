"""大语言模型 (LLM) 客户端工厂模块。

基于全局配置统一创建并返回配置完备的 LangChain ChatOpenAI 客户端实例，
隔离上游模型服务提供方的连接凭证与端点细节。
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

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


class ChatModelAdapter:
    """隔离两段模型请求：流式汇总决策，随后以无工具模型流式作答。"""

    async def decide(self, messages, tools) -> AIMessage:
        """汇总完整申请。参数：完整消息与工具列表；返回：AIMessage；异常：网络错误、空响应或坏 JSON。"""
        model = get_chat_model(streaming=True).bind_tools(tools, tool_choice="auto")
        gathered = None
        async for chunk in model.astream(messages):
            gathered = chunk if gathered is None else gathered + chunk
        if gathered is None:
            raise ValueError("模型未返回决策")
        # LangChain 的分片解析允许自动补全 JSON；业务执行前必须严格解析原始全文。
        if isinstance(gathered, AIMessageChunk) and gathered.tool_call_chunks:
            calls = []
            for chunk in gathered.tool_call_chunks:
                args = json.loads(chunk.get("args") or "")
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是对象")
                calls.append({"name": chunk.get("name"), "id": chunk.get("id"), "args": args})
            return AIMessage(content="", tool_calls=calls)
        return AIMessage(content=gathered.content, tool_calls=gathered.tool_calls,
                         invalid_tool_calls=gathered.invalid_tool_calls)

    async def stream_answer(self, messages):
        """生成最终文本。参数：含工具事实的消息；返回：异步字符串流；异常：网络错误或意外工具申请。"""
        model = get_chat_model(streaming=True, temperature=0.3)
        async for chunk in model.astream(messages):
            if chunk.tool_calls or chunk.invalid_tool_calls or getattr(chunk, "tool_call_chunks", None):
                raise ValueError("回答阶段不允许再次调用工具")
            if isinstance(chunk.content, str):
                delta = chunk.content
            else:
                delta = "".join(block if isinstance(block, str) else block["text"]
                    for block in chunk.content if isinstance(block, str) or
                    (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)))
            if delta:
                yield delta
