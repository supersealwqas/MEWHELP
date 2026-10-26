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
        max_length=4000,
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
            str: 消息保持原文；会话和用户标识去除首尾空白。

        异常:
            ValueError: 当输入值仅包含空白字符或为空时抛出
        """
        # 拦截首尾去空后长度为0的纯空白字符串，避免无效调用
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} 不能为空或纯空白字符")
        # MySQL 的 utf8mb4_bin 比较忽略末尾空格；统一标识，避免同会话绕过内存互斥。
        return value if info.field_name == "message" else value.strip()


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
    event: Literal["delta", "tool_status", "answer_metadata", "error", "done"] = Field(description="事件类型")
    data: dict = Field(default_factory=dict, description="安全事件载荷；error 可包含 retry_safe 重试提示，done 为空")


class InvalidToolDecision(ValueError):
    """携带非法模型申请的审计数据；异常说明不含原始参数，不能直接回灌模型。"""

    def __init__(self, *, tool_calls=None, invalid_tool_calls=None, tool_call_chunks=None,
                 raw_output: str | None = None, validation_category: str | None = None):
        """保存诊断所需的模型申请字段。

        参数:
            tool_calls: 已解析申请，可包含缺失或重复 ID；没有时为 None。
            invalid_tool_calls: 模型库识别的非法申请；没有时为 None。
            tool_call_chunks: 聚合后的原始申请片段，包含尚未成功解析的参数字符串。
            raw_output: 无工具终止决定的原文；仅审计最多1000字符，不能作为正文或重试输入。
            validation_category: 终止决定失败类别，json为JSON格式、shape为字段形状、schema为字段取值。
        返回值:
            None；audit_record 为可保存于 JSON 列的诊断记录，不是可执行申请。
        异常:
            无主动抛出的业务异常。
        """
        super().__init__("模型工具申请不完整")
        # 诊断对象没有顶层 id/name/args，历史组装校验会排除它，避免重放损坏协议。
        self.audit_record = {"type": "invalid_tool_decision", "tool_calls": tool_calls or [],
                             "invalid_tool_calls": invalid_tool_calls or [],
                             "tool_call_chunks": tool_call_chunks or []}
        if raw_output is not None:
            # 只扩展终止决定诊断，保持既有工具错误记录形状；长输出不能无限膨胀审计。
            self.audit_record["terminal_decision"] = {"raw_output": raw_output[:1000],
                "raw_length": len(raw_output), "truncated": len(raw_output) > 1000,
                "validation_category": validation_category}
