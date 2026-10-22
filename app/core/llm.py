"""大语言模型 (LLM) 客户端工厂模块。

基于全局配置统一创建并返回配置完备的 LangChain ChatOpenAI 客户端实例，
隔离上游模型服务提供方的连接凭证与端点细节。
"""

import json
from contextlib import aclosing
from typing import Any
import anyio

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.config import settings
from app.core.prompts import ANSWER_NUDGE
from app.schemas.chat import InvalidToolDecision


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
    """隔离可重复的工具决策与最终回答：完整汇总申请，最终以无工具模型流式作答。"""

    # 回答阶段允许的尝试次数；仅在模型没产出任何正文时启用，网络异常不在此重试。
    ANSWER_ATTEMPTS = 2

    # 首轮没有正文时的纠正指令。追加在消息末尾而非改 system，实测对当前网关有效。
    ANSWER_NUDGE = ANSWER_NUDGE

    async def decide(self, messages, tools) -> AIMessage:
        """汇总完整申请。参数：完整消息与工具列表；返回：AIMessage；异常：网络错误、空响应或携带审计的 InvalidToolDecision。"""
        model = get_chat_model(streaming=True).bind_tools(tools, tool_choice="auto")
        gathered = None
        stream = model.astream(messages)
        try:
            async for chunk in stream:
                gathered = chunk if gathered is None else gathered + chunk
        finally:
            with anyio.CancelScope(shield=True):
                await stream.aclose()
        if gathered is None:
            raise ValueError("模型未返回决策")
        # LangChain 的分片解析允许自动补全 JSON；业务执行前必须严格解析原始全文。
        if isinstance(gathered, AIMessageChunk) and gathered.tool_call_chunks:
            try:
                calls = []
                for chunk in gathered.tool_call_chunks:
                    args = json.loads(chunk.get("args") or "")
                    if not isinstance(args, dict):
                        raise ValueError("工具参数必须是对象")
                    # Python 容许 NaN/Infinity，指数也可能溢出成 inf；MySQL JSON 不接受这些值。
                    # 在落库前严格校验整棵参数树，失败时保留原始字符串审计而非写入坏 JSON。
                    json.dumps(args, allow_nan=False)
                    calls.append({"name": chunk.get("name"), "id": chunk.get("id"), "args": args})
                return AIMessage(content="", tool_calls=calls)
            except (ValueError, TypeError, ValidationError) as error:
                # 空参数与坏 JSON 不能修补成合法申请；保留原文供服务保存，日志只记安全异常类型。
                raise InvalidToolDecision(tool_call_chunks=gathered.tool_call_chunks) from error
        return AIMessage(content=gathered.content, tool_calls=gathered.tool_calls,
                         invalid_tool_calls=gathered.invalid_tool_calls)

    async def stream_answer(self, messages):
        """生成最终文本。

        参数:
            messages: 已组装好的回答阶段消息，不含工具定义。
        返回值:
            异步字符串流：仅包含可展示的正文增量，不含推理块或工具申请。
        异常:
            网络错误向上传播；连续 ANSWER_ATTEMPTS 次都没有正文时抛 ValueError。
        """
        # 上游（实测 OpenAI 兼容网关）在未绑定工具的请求里仍可能只回 tool_call，
        # tool_choice=none 对它无效；已验证唯一可靠的补救是在消息末尾追加一条
        # 明确“只输出文字”的纠正指令。首次失败后才启用，不改变正常请求的输入。
        for attempt in range(self.ANSWER_ATTEMPTS):
            produced = False
            attempt_messages = messages if attempt == 0 else [*messages, HumanMessage(content=self.ANSWER_NUDGE)]
            # 用 aclosing 保证外层提前关闭（停止回复）时立即关闭本次上游流，不等垃圾回收。
            async with aclosing(self._answer_once(attempt_messages)) as once:
                async for delta in once:
                    produced = True
                    yield delta
            if produced:
                return
        raise ValueError("模型未返回有效文本")

    async def _answer_once(self, messages):
        """执行一次回答流，丢弃违规的工具申请分片。

        参数:
            messages: 回答阶段消息。
        返回值:
            异步字符串流：本次尝试中可展示的正文增量。
        异常:
            网络错误向上传播；流在本层关闭，不依赖垃圾回收。
        """
        model = get_chat_model(streaming=True, temperature=0.3)
        stream = model.astream(messages)
        try:
            async for chunk in stream:
                # 回答阶段没有绑定工具；模型若仍输出工具申请，丢弃该分片继续等正文，
                # 不能因为协议外分片把整轮回答判为失败（否则用户只看到“本轮暂时未能完成”）。
                if chunk.tool_calls or chunk.invalid_tool_calls or getattr(chunk, "tool_call_chunks", None):
                    continue
                if isinstance(chunk.content, str):
                    delta = chunk.content
                else:
                    delta = "".join(block if isinstance(block, str) else block["text"]
                        for block in chunk.content if isinstance(block, str) or
                        (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)))
                if delta:
                    yield delta
        finally:
            # 直接关闭内部模型流，不能等异步生成器被垃圾回收后才释放网络连接。
            with anyio.CancelScope(shield=True):
                await stream.aclose()
