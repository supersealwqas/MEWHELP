"""原客服聊天入口的 SSE 编码层，业务编排和数据持久化由 ChatService 负责。"""

import json
import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest
from app.services.chat import ChatService

router = APIRouter(prefix="/api", tags=["chat"])


class ClosingStreamingResponse(StreamingResponse):
    """响应拥有正文迭代器，即使网络send失败也立即关闭服务和上游流。"""

    async def __call__(self, scope, receive, send):
        """发送后保证关闭。参数：ASGI请求/接收/发送；返回：None；异常：原网络或取消错误透传。"""
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()


def get_chat_service(request: Request) -> ChatService:
    """获取已完成启动恢复的服务。参数：HTTP 请求；返回：共享聊天服务；异常：启动未就绪时 RuntimeError。"""
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise RuntimeError("聊天服务尚未就绪")
    return service


@router.post("/chat")
async def chat(req: ChatRequest, service: ChatService = Depends(get_chat_service)) -> StreamingResponse:
    """把语义事件编码为 SSE。参数：请求和依赖注入服务；返回：流式响应；异常：客户端断开导致取消。"""
    async def event_stream():
        """编码并关闭本轮生成器。参数：无；返回：SSE 字符串流；异常：取消向上传播。"""
        stream = service.stream_turn(req)
        try:
            async for event in stream:
                if event.event == "done":
                    yield "data: [DONE]\n\n"
                else:
                    prefix = "" if event.event == "delta" else f"event: {event.event}\n"
                    yield prefix + "data: " + json.dumps(event.data, ensure_ascii=False) + "\n\n"
        finally:
            await stream.aclose()
    return ClosingStreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
