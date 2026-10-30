"""本机观测用量查询接口，返回完整性、原始用量覆盖率及意图分组。"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.observability.costs import get_intent_usage

router = APIRouter(prefix="/api/observability", tags=["observability"])


@router.get("/costs")
async def intent_costs(request: Request, start: datetime = Query(alias="from"), end: datetime = Query(alias="to")):
    """查询意图用量。参数：HTTP请求/带时区ISO起止时间；返回：200汇总或503不完整状态；异常：HTTP422。"""
    if set(request.query_params) - {"from", "to"}:
        raise HTTPException(422, "仅支持from和to时间参数")
    try:
        result = await get_intent_usage(start, end)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    return JSONResponse(result, status_code=200 if result["coverage"]["complete"] else 503)
