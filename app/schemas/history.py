"""数据库历史的纯数据契约，隔离 ORM 对象与模型消息。"""

from typing import Literal
from pydantic import BaseModel, Field


class StoredMessage(BaseModel):
    """一条已提交的消息及原始工具关联。"""
    id: int | None = Field(default=None, gt=0, strict=True, description="消息数据库主键；旧测试可空，在线分层必须提供真实ID")
    role: Literal["user", "assistant", "tool"] = Field(description="持久化消息角色")
    content: str = Field(description="消息内容或工具结果 JSON")
    tool_calls: list[dict] | None = Field(default=None, description="模型申请列表")
    tool_call_id: str | None = Field(default=None, description="结果对应的工具申请 ID")


class StoredTurn(BaseModel):
    """按轮次分组的历史，用于完整配对与裁剪。"""
    turn_id: str = Field(description="轮次唯一编号")
    turn_status: str = Field(description="轮次完成状态")
    messages: list[StoredMessage] = Field(default_factory=list, description="按消息主键顺序排列的记录")
    order_selection: dict | None = Field(default=None, description="来源用户消息中已保存的选单状态，包含首次接受订单")
    ticket_intent: dict | None = Field(default=None, description="来源用户行保存的有界建单诉求，旧历史可缺省")
    ticket_confirmation: dict | None = Field(default=None, description="聊天工具的冻结预览及已保存决定，旧历史可缺省")
    receipt_pending: bool = Field(default=False, description="独立工单仍可补写回执；即使轮已标中断也不能跨摘要边界")
