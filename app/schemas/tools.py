"""工具结果和可信服务端上下文；业务参数 Schema 在工具阶段扩展。"""

from pydantic import BaseModel, ConfigDict, Field


class ToolContext(BaseModel):
    """不可变调用上下文，由服务端生成，永不交由模型指定。"""
    model_config = ConfigDict(frozen=True)
    conversation_id: int = Field(default=0, ge=0, description="数据库会话主键，首次保存前为 0")
    session_id: str = Field(description="客户端会话标识")
    user_id: str = Field(description="会话所属演示用户")
    turn_id: str = Field(description="当前轮次 UUID")
    tool_call_id: str = Field(default="", description="原工具调用 ID；决策前可为空")
    user_message: str = Field(description="当前用户原话，用于关键词校验")


class ToolError(BaseModel):
    """可安全回灌给模型的稳定错误，不包含原始异常。"""
    code: str = Field(description="稳定错误代码")
    message: str = Field(description="安全的中文错误提示")


class ToolResult(BaseModel):
    """统一工具结果；未命中仍属于成功查询。"""
    ok: bool = Field(description="工具执行是否成功")
    data: dict | list | None = Field(default=None, description="成功的业务数据")
    error: ToolError | None = Field(default=None, description="失败时的安全错误")
    is_mock: bool = Field(default=False, description="是否为随机演示数据")
