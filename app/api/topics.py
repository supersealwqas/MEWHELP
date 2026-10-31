"""主题旁路HTTP接口：完整分布、共同锁后台触发与真实状态。"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.chat import get_chat_service
from app.knowledge.errors import ConcurrentKnowledgeWrite

router = APIRouter(prefix="/api/flywheel/topics", tags=["topics"])


class ClassifyRequest(BaseModel):
    """只接受显式重跑布尔值，不允许操纵模型或数据库。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    reclassify: bool = Field(default=False, description="True时覆盖全池全部分页，未确定结果保留旧分类")


def get_topics(service=Depends(get_chat_service)):
    """取得生命周期共享主题组件。参数：聊天服务；返回：主题服务；异常：关闭或未启用HTTP503。"""
    topics = getattr(service, "topics", None)
    if topics is None:
        raise HTTPException(503, "主题分类旁路未启用，请配置TOPIC_ENABLED并完成显式迁移")
    return topics


@router.get("")
async def distribution(scope: Annotated[Literal["pending", "all"], Query()] = "pending", topics=Depends(get_topics)):
    """读取固定17类全池分布。参数：scope/共享主题服务；返回：覆盖、来源与版本事实；异常：依赖HTTP503。"""
    try:
        return await topics.distribution(scope)
    except Exception as error:
        raise HTTPException(503, "主题数据库或批次原件不可用，请稍后重试") from error


@router.post("/classify", status_code=202)
async def classify(body: ClassifyRequest, topics=Depends(get_topics)):
    """取得共同锁后后台启动，不阻塞长HTTP。参数：请求/共享服务；返回：202运行中；异常：占用409/依赖503。"""
    try:
        return await topics.trigger(reclassify=body.reclassify)
    except ConcurrentKnowledgeWrite as error:
        raise HTTPException(409, "已有主题批次正在运行，请刷新状态") from error
    except Exception as error:
        raise HTTPException(503, "主题批次暂时无法启动，请稍后重试") from error


@router.get("/status")
async def status(topics=Depends(get_topics)):
    """刷新实际批次和全表版本。参数：共享服务；返回：状态字典；异常：服务依赖负责明确失败状态。"""
    return await topics.status()
