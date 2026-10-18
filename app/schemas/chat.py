"""对话相关的 Pydantic 请求与响应数据契约模块。

定义用于前后端 HTTP 与 SSE 通信的数据模型，包括客户端发起的对话请求 (ChatRequest)
以及服务端向客户端逐 token 推送的流式增量文本块 (ChatChunk)。
"""

from typing import Literal
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ChatRequest(BaseModel):
    """对话请求模型。

    用于接收前端发起的单次问答交互请求，严格要求携带会话标识及非空问题内容。
    """

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="会话唯一标识符，用于多轮对话历史上下文关联与隔离",
    )
    user_id: str = Field(default="demo-user", min_length=1, max_length=64, description="演示用户标识，不作为身份认证")
    message: str = Field(
        ...,
        min_length=1,
        description="用户发送的当前轮次提问内容，不可为空或纯空白字符",
    )

    @field_validator("session_id", "message", "user_id")
    @classmethod
    def validate_not_whitespace(cls, value: str, info: ValidationInfo) -> str:
        """校验字符串字段不能为纯空白字符。

        参数:
            value (str): 待校验的原始字符串输入
            info (ValidationInfo): 校验元数据，提供正在校验的字段名称信息

        返回值:
            str: 校验通过的原始字符串

        异常:
            ValueError: 当输入值仅包含空白字符或为空时抛出
        """
        # 拦截首尾去空后长度为0的纯空白字符串，避免无效调用
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} 不能为空或纯空白字符")
        return value


class ChatChunk(BaseModel):
    """对话流式响应增量数据块模型。

    用于 Server-Sent Events (SSE) 逐 token 向前端推送大模型生成的增量文本。
    """

    delta: str = Field(
        ...,
        description="大模型生成的单次增量文本内容，结束时可为空字符串",
    )


class ChatEvent(BaseModel):
    """服务层的语义事件，由 API 层编码为 SSE。"""
    event: Literal["delta", "tool_status", "error", "done"] = Field(description="事件类型")
    data: dict = Field(default_factory=dict, description="安全事件载荷；done 为空")
