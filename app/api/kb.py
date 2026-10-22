"""知识库只读调试接口，显式 Top-K 覆盖聊天默认条数。

只调用现有应用共享检索，不启动导入/后台任务；后端故障返回安全 503，
不能伪装成成功空命中。接口默认 top_k=5，正文始终来自 MySQL 权威回读。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.chat import get_chat_service
from app.knowledge.schemas import KnowledgeUnavailable
from app.services.chat import ChatService

router = APIRouter(prefix="/kb", tags=["knowledge"])


@router.get("/search")
async def search_knowledge(
    query: Annotated[str, Query(min_length=1, max_length=2000, description="当前检索问题原话")],
    top_k: Annotated[int, Query(ge=1, le=50, description="调试结果条数，覆盖聊天默认值")] = 5,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    """只读搜索权威知识，调试默认 5 条。

    参数：query 为 1–2000 字符非空问题；top_k 为 1–50，默认 5；service 为已就绪共享服务。
    返回：dict，query/top_k/hits，不把模型或索引故障当正常零命中。
    异常：HTTPException 422，query/条数非法；503，知识依赖不可用；FastAPI 参数校验错误。
    """
    try:
        rows = await service.executor.support.query_faq(query, limit=top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="检索问题或条数不合法，请缩小问题范围") from error
    except KnowledgeUnavailable as error:
        raise HTTPException(status_code=503, detail="知识检索暂时不可用，请稍后再试") from error
    return {"query": query, "top_k": top_k, "hits": rows}
