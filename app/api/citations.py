"""按会话归属读取已完成回答的证据快照及白名单文档。"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.chat import get_chat_service
from app.knowledge.sources import read_source
from app.services.chat import ChatService

router = APIRouter(prefix="/api/chat", tags=["citations"])


@router.get("/{session_id}/turns/{turn_id}/citations/{n}")
async def get_citation(session_id: str, turn_id: str, n: int,
                       user_id: str = Query(default="demo-user", min_length=1, max_length=64),
                       service: ChatService = Depends(get_chat_service)) -> dict:
    """返回保存的完整证据快照，未知和无权访问统一404。

    参数：session_id 会话；turn_id 轮次；n 正引用编号；user_id 现有演示归属标识；service 共享服务。
    返回：dict，已完成回答的一条快照。
    异常：HTTPException 404，不泄露会话或引用是否存在；数据库错误透传。
    """
    citation = await service.chat.get_citation(session_id, user_id, turn_id, n)
    if citation is None:
        raise HTTPException(404, "引用不存在或不可访问")
    return citation


@router.get("/{session_id}/turns/{turn_id}/citations/{n}/source")
async def get_source(citation: dict = Depends(get_citation)) -> dict:
    """在相同归属检查后读取已登记全文，历史问答不暴露私人对话。

    参数：citation 依赖返回的可信快照。
    返回：dict，当前全文与旧证据版本区别。
    异常：HTTPException 404，来源不可公开读取。
    """
    source = read_source(citation)
    if source is None:
        raise HTTPException(404, "引用不存在或不可访问")
    return source
