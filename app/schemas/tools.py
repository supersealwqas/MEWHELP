"""工具业务参数、隐藏的运行时身份、统一工具结果及服务端调用上下文。

本模块只描述数据结构，不含执行逻辑与业务规则。工具层、执行器、仓储层共用这里的
Schema，避免各处自行约定字段名导致协议漂移。
"""

from typing import Annotated, Literal
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, ConfigDict, Field

# 业务拒绝的统一字段名。工具返回业务失败时必须携带这两个键，执行器据此识别并转为失败事实。
# 集中定义避免工具层与执行器各写一份字符串字面量。
ERROR_FIELD = "error"
CODE_FIELD = "code"


class BusinessArgs(BaseModel):
    """严格校验工具参数，禁止类型强转及未声明字段，统一去除字符串首尾空白。"""
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class OwnedOrderArgs(BusinessArgs):
    """订单查询的完整运行参数，身份由系统注入且不交给模型填写。"""
    order_id: str = Field(min_length=1, max_length=64, description="用户提供或历史中已明确的订单号，不得猜测")
    user_id: Annotated[str, InjectedToolArg] = Field(min_length=1, description="由系统注入的当前演示用户标识")


class LogisticsArgs(BusinessArgs):
    """物流查询参数，单号来自用户明确提供或本会话成功的工具事实。"""
    tracking_no: str = Field(min_length=1, max_length=64, description="用户提供、本轮或本会话历史query_order成功结果中的物流单号，不得填写订单号或猜测")
    user_id: Annotated[str, InjectedToolArg] = Field(min_length=1, description="由系统注入的当前演示用户标识")


class KeywordArgs(BusinessArgs):
    """商品或 FAQ 的单个关键词。"""
    keyword: str = Field(min_length=1, max_length=100, description="用户原话中用于检索的关键词")


class RefundPolicyArgs(BusinessArgs):
    """退款政策查询参数，原话身份仍由可信运行上下文提供。"""
    query: str = Field(min_length=1, max_length=2000, description="当前用户完整退款问题，不得新增型号、否定条件或承诺")


class TicketArgs(BusinessArgs):
    """人工工单业务参数，不包含会话身份。"""
    description: str = Field(min_length=1, max_length=2000, description="用户明确要求提交处理的问题描述")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单分类，仅售后、投诉、咨询")


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
    """统一工具结果；未命中仍属于成功查询。

    这是执行器与仓储层之间的事实载体；is_mock 仅供后台审计及状态协议使用。
    模型消息组装时移除来源标记，仅把 ok、data、error 作为业务结果交给模型。
    """

    ok: bool = Field(description="工具执行是否成功")
    data: dict | None = Field(default=None, description="成功的业务数据")
    error: ToolError | None = Field(default=None, description="失败时的安全错误")
    is_mock: bool = Field(default=False, description="是否为演示数据")

    @classmethod
    def success(cls, data: dict, *, is_mock: bool = False) -> "ToolResult":
        """构造成功结果。

        参数:
            data: 业务数据字典，已确认不是错误载荷。
            is_mock: 数据是否来自演示工具。
        返回值:
            ToolResult: ok=True 的结果。
        异常:
            无主动抛出的业务异常。
        """
        return cls(ok=True, data=data, is_mock=is_mock)

    @classmethod
    def failed(cls, code: str, message: str, *, is_mock: bool = False) -> "ToolResult":
        """构造失败结果。

        参数:
            code: 稳定错误码，供前端与固定回复表匹配。
            message: 可安全回灌给模型的中文说明。
            is_mock: 失败发生在演示工具上。
        返回值:
            ToolResult: ok=False 且带 error 的结果。
        异常:
            无主动抛出的业务异常。
        """
        return cls(ok=False, error=ToolError(code=code, message=message), is_mock=is_mock)


def is_business_rejection(payload: dict) -> bool:
    """判断工具返回的字典是否为业务拒绝载荷。

    参数:
        payload: 工具返回的原始业务字典。
    返回值:
        bool: 同时携带 error 与 code 两个字符串字段时为 True。
    异常:
        无主动抛出的业务异常。
    """
    return isinstance(payload.get(ERROR_FIELD), str) and isinstance(payload.get(CODE_FIELD), str)
