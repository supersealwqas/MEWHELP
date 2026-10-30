"""人工审核与飞轮处理HTTP接口；发布失败保留已冻结答案和可重试状态。"""

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.chat import get_chat_service
from app.knowledge.schemas import KnowledgeUnavailable
from app.repositories.flywheel import ReviewConflict, ReviewNotFound
from app.schemas.flywheel import ApprovalRequest
from app.services.chat import ChatService

router = APIRouter(prefix="/api/flywheel", tags=["flywheel"])
ReviewId = Annotated[int, Path(gt=0, le=18446744073709551615)]


async def _call(operation, review_id=None, repository=None):
    """统一转译可修正输入、状态和依赖错误。参数：待执行协程/可选审核主键/权威仓储；返回：操作结果；异常：HTTP404/409/422/503及数据库错误。"""
    try:
        return await operation
    except ReviewNotFound as error:
        raise HTTPException(404, "审核缺口不存在") from error
    except ReviewConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (KnowledgeUnavailable, TimeoutError) as error:
        detail = {"message": "请求未完整完成，请查看审核状态后重试",
            "review_id": str(review_id) if review_id is not None else None}
        if repository is not None:
            try:
                # 错误可能发生在冻结前或完成提交后，只由持久字段派生状态。
                async with asyncio.timeout(5):
                    row = await repository.review_detail(review_id)
                detail["publication_status"] = row["publication_status"]
            except Exception:
                # 状态暂时不可读时省略，不能猜成pending或not_started。
                pass
        raise HTTPException(503, detail) from error


@router.get("/reviews")
async def list_reviews(status: Literal["待审", "通过", "驳回"] | None = None,
                       after_id: int = Query(0, ge=0, le=18446744073709551615),
                       limit: int = Query(50, ge=1, le=200), service: ChatService = Depends(get_chat_service)) -> dict:
    """分页列出审核队列。参数：中文状态/游标/页长/共享服务；返回：条目和下一游标；异常：HTTP422或数据库错误。"""
    return await _call(service.review_publisher.repository.list_reviews(status, after_id, limit))


@router.get("/reviews/{review_id}")
async def review_detail(review_id: ReviewId, service: ChatService = Depends(get_chat_service)) -> dict:
    """展示全部原话和历史快照。参数：主键/共享服务；返回：审核详情；异常：HTTP404/422或数据库错误。"""
    return await _call(service.review_publisher.repository.review_detail(review_id))


@router.post("/reviews/{review_id}/approve")
async def approve(review_id: ReviewId, request: ApprovalRequest, service: ChatService = Depends(get_chat_service)) -> dict:
    """接收明确人工正文并发布。参数：主键/严格人工载荷/共享服务；返回：审核状态；异常：HTTP404/409/422/503。"""
    return await _call(service.review_publisher.approve(review_id, request.answer), review_id, service.review_publisher.repository)


@router.post("/reviews/{review_id}/reject")
async def reject(review_id: ReviewId, service: ChatService = Depends(get_chat_service)) -> dict:
    """驳回尚未冻结条目。参数：主键/共享服务；返回：审核状态；异常：HTTP404/409/422或数据库错误。"""
    return await _call(service.review_publisher.repository.reject(review_id))


@router.post("/reviews/{review_id}/retry")
async def retry(review_id: ReviewId, service: ChatService = Depends(get_chat_service)) -> dict:
    """只重试已冻结人工答案。参数：主键/共享服务；返回：审核状态；异常：HTTP404/409/422/503。"""
    return await _call(service.review_publisher.publish(review_id), review_id, service.review_publisher.repository)


@router.post("/process")
async def process(service: ChatService = Depends(get_chat_service)) -> dict:
    """人工触发当前同实例归并worker。参数：共享服务；返回：处理/合并/失败计数；异常：依赖或数据库错误。"""
    return await _call(service.flywheel.process_pending())
