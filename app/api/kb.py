"""知识库只读调试接口，显式 Top-K 覆盖聊天默认条数。

只调用现有应用共享检索，不启动导入/后台任务；后端故障返回安全 503，
不能伪装成成功空命中。接口默认 top_k=5，正文始终来自 MySQL 权威回读。
"""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from app.api.chat import ClosingStreamingResponse, get_chat_service
from app.knowledge.inspection import KnowledgeInspector, RetrievalDebugRequest
from app.knowledge.schemas import KnowledgeUnavailable
from app.services.chat import ChatService
from app.knowledge.vector import MAX_SIGNED_INT64

router = APIRouter(prefix="/kb", tags=["knowledge"])


@router.get("/search")
async def search_knowledge(
    query: Annotated[str, Query(min_length=1, max_length=2000, description="当前检索问题原话")],
    top_k: Annotated[int, Query(ge=1, le=50, description="调试结果条数，覆盖聊天默认值")] = 5,
    category: Annotated[str | None, Query(min_length=1, max_length=200, description="知识品类精确筛选")] = None,
    content_type: Annotated[str | None, Query(min_length=1, max_length=64, description="知识内容类型精确筛选")] = None,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    """只读搜索权威知识，调试默认 5 条。

    参数：query 为 1–2000 字符；top_k 为 1–50，默认 5；category/content_type 为可选筛选；service 为共享服务。
    返回：dict，query/top_k/hits，不把模型或索引故障当正常零命中。
    异常：HTTPException 422，query/条数非法；503，知识依赖不可用；FastAPI 参数校验错误。
    """
    try:
        filters = {key: value for key, value in {"category": category, "content_type": content_type}.items()
                   if value is not None}
        options = {"filters": filters} if filters else {}
        rows = await service.executor.support.query_faq(query, limit=top_k, **options)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="检索问题或条数不合法，请缩小问题范围") from error
    except KnowledgeUnavailable as error:
        raise HTTPException(status_code=503, detail="知识检索暂时不可用，请稍后再试") from error
    return {"query": query, "top_k": top_k, "hits": rows}


def get_knowledge_inspector(request: Request) -> KnowledgeInspector:
    """从当前在线服务取得共享混合检索并预检，公开依赖允许测试替换。

    参数：request HTTP请求；返回：请求局部只读包装器，不创建资源。
    异常：HTTPException 503，服务缺失、非hybrid或共享依赖未就绪。
    """
    try:
        service = getattr(request.app.state, "chat_service", None)
        search = service.executor.support.knowledge_search if service is not None else None
        inspector = KnowledgeInspector(search)
        inspector.preflight()
        return inspector
    except Exception as error:
        raise HTTPException(status_code=503, detail="当前在线混合知识检索暂时不可用") from error


@router.get("/collection")
async def inspect_collection(inspector: KnowledgeInspector = Depends(get_knowledge_inspector)) -> dict:
    """参数：共享只读包装器；返回：实时count与公开结构；异常：HTTPException503，Milvus失败。"""
    try:
        return await inspector.collection()
    except Exception as error:
        raise HTTPException(status_code=503, detail="知识集合读取暂时不可用") from error


@router.get("/records")
async def inspect_records(
    after_id: Annotated[int, Query(ge=0, le=MAX_SIGNED_INT64, description="上一页末条主键，signed INT64，默认0")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="实时记录页大小，1–100，默认20")] = 20,
    inspector: KnowledgeInspector = Depends(get_knowledge_inspector),
) -> dict:
    """参数：非负游标/页大小/共享包装器；返回：原始双库记录页；异常：422参数错误，503索引失败。"""
    try:
        return await inspector.records(after_id=after_id, limit=limit)
    except Exception as error:
        raise HTTPException(status_code=503, detail="知识记录读取暂时不可用") from error


@router.get("/records/{id}")
async def inspect_record(
    id: Annotated[int, Path(gt=0, le=MAX_SIGNED_INT64, description="知识正signed INT64主键")],
    inspector: KnowledgeInspector = Depends(get_knowledge_inspector),
) -> dict:
    """参数：正主键/共享包装器；返回：含Dense向量的详情；异常：404缺记录，422参数，503索引失败。"""
    try:
        result = await inspector.record(id)
    except Exception as error:
        raise HTTPException(status_code=503, detail="知识记录读取暂时不可用") from error
    if result is None:
        raise HTTPException(status_code=404, detail="该知识记录不存在")
    return result


class _SafeDebugRoute(APIRoute):
    """仅为调试POST安全编码参数错误，避免非法NaN输入在框架422处理中再次编码失败。"""

    def get_route_handler(self):
        """参数：无；返回：标准路由的安全校验包装；异常：原运行时错误/取消传播。"""
        handler = super().get_route_handler()

        async def safe_handler(request: Request):
            """参数：HTTP请求；返回：标准响应；异常：HTTPException422，不回显原非法输入。"""
            try:
                return await handler(request)
            except RequestValidationError as error:
                # 不把非有限浮点或任意客户端输入写入422响应，保持OpenAPI模型不变。
                raise HTTPException(status_code=422, detail="检索参数不合法，请检查问题、权重和结果条数") from error

        return safe_handler


async def retrieval_debug(request: RetrievalDebugRequest,
                          inspector: KnowledgeInspector = Depends(get_knowledge_inspector)) -> ClosingStreamingResponse:
    """参数：严格调试请求/预检完成的共享包装器；返回：拥有上游关闭责任的SSE；异常：422/503预检或取消。"""
    try:
        inspector.preflight()
    except Exception as error:
        raise HTTPException(status_code=503, detail="当前在线混合知识检索暂时不可用") from error
    stream = inspector.debug(request)

    async def event_stream():
        """参数：无；返回：编码SSE流；异常：取消/网络错误传播，finally释放本次迭代器。"""
        try:
            async for event in stream:
                yield f"event: {event['event']}\ndata: " + json.dumps(event["data"], ensure_ascii=False, allow_nan=False) + "\n\n"
        finally:
            await stream.aclose()

    return ClosingStreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


router.add_api_route("/retrieval-debug", retrieval_debug, methods=["POST"], route_class_override=_SafeDebugRoute)
