"""已完成原轮的未解决反馈接口，客户端只能指定身份和否定反馈。"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.chat import get_chat_service
from app.repositories.chat import FeedbackNotFound, FeedbackNotReady
from app.schemas.flywheel import FeedbackReceipt
from app.services.chat import ChatService

router = APIRouter(prefix="/api/chat", tags=["feedback"])


class UnresolvedFeedback(BaseModel):
    """严格否定反馈载荷，禁止客户端注入原问、证据或池主键。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    user_id: str = Field(min_length=1, max_length=64, description="现有演示用户身份")
    resolved: Literal[False] = Field(description="固定false表示原轮问题未解决")

    @field_validator("resolved", mode="before")
    @classmethod
    def require_false(cls, value):
        """拒绝0等与False相等的非布尔值。参数：原始值；返回：False；异常：ValueError。"""
        if value is not False:
            raise ValueError("resolved必须为false")
        return value

    @field_validator("user_id")
    @classmethod
    def normalize_user(cls, value):
        """按现有用户身份规则去空白。参数：身份字符串；返回：规范身份；异常：ValueError空白。"""
        if not value.strip():
            raise ValueError("user_id不能为空")
        return value.strip()


@router.post("/{session_id}/turns/{turn_id}/feedback", response_model=FeedbackReceipt)
async def record_feedback(session_id: str, turn_id: str, request: UnresolvedFeedback,
                          service: ChatService = Depends(get_chat_service)) -> FeedbackReceipt:
    """保存原轮否定反馈。参数：路径身份、严格载荷、共享服务；返回：幂等回执；异常：HTTP404/409及数据库错误。"""
    try:
        return await service.chat.record_unresolved_feedback(session_id, request.user_id, turn_id)
    except FeedbackNotFound as error:
        raise HTTPException(404, "原轮不存在或不可访问") from error
    except FeedbackNotReady as error:
        raise HTTPException(409, "原轮尚未完成，暂不能反馈") from error
