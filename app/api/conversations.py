"""本人会话和消息的只读分页入口，沿用现有演示身份边界，不创建会话或触发摘要。"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.chat import get_chat_service
from app.repositories.context import ContextNotFound, ContextRepository
from app.schemas.conversations import ConversationPage, ConversationMessagePage
from app.services.chat import ChatService

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _id(value: str | None, *, zero: bool = False) -> int | None:
    """精确解析BIGINT游标。参数：十进制字符串/None及是否允许0；返回：int/None；异常：HTTPException422。"""
    if value is None:
        return None
    if not (1 <= len(value) <= 20 and value.isascii() and value.isdigit()
            and (0 if zero else 1) <= int(value) <= 2**64-1):
        raise HTTPException(422, "会话或消息主键格式不合法")
    return int(value)


@router.get("", response_model=ConversationPage)
async def list_conversations(user_id: str = Query(default="demo-user", min_length=1, max_length=64),
                             before_id: str | None = None, limit: int = Query(default=50, ge=1, le=100),
                             service: ChatService = Depends(get_chat_service)) -> dict:
    """返回本人会话。参数：演示身份、倒序游标、分页数量、共享服务；返回：字符串ID分页；异常：422或数据库错误。"""
    page = await ContextRepository(service.chat.factory).list_conversations(user_id, before_id=_id(before_id), limit=limit)
    return {"items":[{**row, "id":str(row["id"])} for row in page["items"]],
            "next_before_id":str(page["next_before_id"]) if page["next_before_id"] is not None else None}


@router.get("/{conversation_id}/messages", response_model=ConversationMessagePage)
async def list_messages(conversation_id: str,
                        user_id: str = Query(default="demo-user", min_length=1, max_length=64),
                        after_id: str = "0", limit: int = Query(default=100, ge=1, le=200),
                        service: ChatService = Depends(get_chat_service)) -> dict:
    """回载本人会话原文，未知与越权统一404。

    参数：会话主键、演示身份、正序消息游标、分页数量、共享服务；返回：未裁剪消息及引用/状态。
    异常：HTTPException404/422或数据库错误，不执行创建、模型、摘要和业务动作。
    """
    try:
        page = await ContextRepository(service.chat.factory).list_messages(user_id, _id(conversation_id),
            after_id=_id(after_id, zero=True), limit=limit)
    except ContextNotFound as error:
        raise HTTPException(404, "会话不存在或不可访问") from error
    return {**page, "items":[{**row,"id":str(row["id"]),"conversation_id":str(row["conversation_id"])} for row in page["items"]],
            "next_after_id":str(page["next_after_id"]) if page["next_after_id"] is not None else None}
