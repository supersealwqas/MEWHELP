"""多会话只读接口契约：数据库BIGINT用十进制字符串，原始消息和状态不裁剪。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConversationItem(BaseModel):
    """本人会话侧栏条目，排序只依赖稳定创建主键。"""

    id: str = Field(description="会话BIGINT主键的十进制字符串")
    session_id: str = Field(description="继续聊天时沿用的会话标识")
    preview: str = Field(description="首条用户原问预览，最多80字符")
    created_at: datetime = Field(description="会话创建时间")
    has_summary: bool = Field(description="是否已经提交过摘要段，含显式无事实段")


class ConversationPage(BaseModel):
    """可遍历所有本人会话的倒序分页。"""

    items: list[ConversationItem] = Field(description="本页会话，新建在前")
    next_before_id: str | None = Field(description="下一页开游标，无更多时为null")


class ConversationMessage(BaseModel):
    """完整原始消息及其轮次和引用元数据，读取不能触发业务操作。"""

    id: str = Field(description="消息主键字符串")
    conversation_id: str = Field(description="所属会话主键字符串")
    role: Literal["user", "assistant", "tool"] = Field(description="原始角色")
    content: str | None = Field(description="原始完整正文，历史空正文保留null")
    tool_calls: list[dict] | None = Field(description="完整原始工具申请")
    tool_call_id: str | None = Field(description="原始工具关联键")
    rag_metadata: dict | None = Field(description="原始引用、完成和等待状态；可执行动作须另查当前状态")
    turn_id: str = Field(description="原始轮次标识")
    turn_status: str = Field(description="该轮当前持久化状态")
    created_at: datetime = Field(description="原始消息生成时间")


class ConversationMessagePage(BaseModel):
    """按消息主键正序回载原文，每页重新核对会话归属。"""

    session_id: str = Field(description="会话标识，用于核对回载目标")
    items: list[ConversationMessage] = Field(description="本页完整消息")
    next_after_id: str | None = Field(description="下一页消息开游标，无更多时为null")
