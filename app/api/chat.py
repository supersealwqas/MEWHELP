"""客服多轮对话 SSE 流式输出 API 路由模块。

本模块提供基于 Server-Sent Events (SSE) 协议的多轮客服问答流式输出端点，
集成内存会话管理、Token 预算裁剪、智能客服角色人设 Prompt 编排及大模型流式调用。
主要包含接口：
- POST /api/chat: 接收用户消息并逐 token 推送增量回答，会话结束时回写完整 AI 历史。
"""

import json
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.llm import get_llm
from app.core.memory import memory, trim_history
from app.core.prompts import CUSTOMER_SERVICE_PROMPT
from app.schemas.chat import ChatRequest

# 注册客服对话 API 路由器，统一前缀为 /api，分组标签为 chat
router = APIRouter(prefix="/api", tags=["chat"])

# 暴露别名以提升导入语义清晰度
chat_router = router


@router.post("/chat", summary="客服多轮对话 SSE 流式输出")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """客服多轮对话 SSE 流式响应端点。

    接收客户端提问，自动关联并裁剪会话历史，调用大语言模型进行逐 token 流式输出，
    并在输出完成时将完整的 AI 回复回写至内存会话中。若发生上游模型故障，
    向客户端推送特定格式的错误帧并安全关闭连接。

    参数:
        request (ChatRequest): 包含 session_id (会话标识符) 与 message (用户提问) 的请求体对象。

    返回值:
        fastapi.responses.StreamingResponse: media_type 为 text/event-stream 的流式 HTTP 响应，
        包含逐帧增量数据、完成标志或异常帧。

    异常:
        fastapi.exceptions.RequestValidationError: 当入参格式不合法时由 FastAPI 框架自动抛出 HTTP 422。
    """
    # 1. 立即将当前用户提问记录至会话历史中
    memory.add_user_message(request.session_id, request.message)

    # 2. 读取该会话的完整历史记录，并执行 Token 预算裁剪
    history = memory.get_messages(request.session_id)
    trimmed = trim_history(history, max_tokens=settings.chat_token_budget)

    # 3. 编排客服对话调用链：提示词模板绑定指定多样性温度 (0.7) 的 LLM 实例
    chain = CUSTOMER_SERVICE_PROMPT | get_llm(temperature=0.7)

    async def event_generator() -> AsyncGenerator[str, None]:
        """异步生成 Server-Sent Events 协议格式的数据帧。

        逐 chunk 消费大模型流式输出，序列化为 SSE data 帧；流式正常完成时输出 [DONE] 标记
        并将全量拼接回答回写至内存；若流式中途或开始时抛出异常，捕获并输出 event: error 帧。

        返回值:
            AsyncGenerator[str, None]: 格式化后的 SSE 字符串数据流

        异常:
            内部捕获所有 Exception，转换为 SSE error 帧安全输出，对外不抛出未捕获异常。
        """
        try:
            full_text: list[str] = []
            # 异步消费上游模型的 chunk 增量
            async for chunk in chain.astream({"history": trimmed}):
                # 兼容处理 chunk.content 的不同类型
                if isinstance(chunk.content, str):
                    delta = chunk.content
                elif chunk.content is None:
                    delta = ""
                else:
                    delta = str(chunk.content)

                # 过滤空增量，避免发送无效的数据帧
                if delta:
                    full_text.append(delta)
                    chunk_data = json.dumps({"delta": delta}, ensure_ascii=False)
                    yield f"data: {chunk_data}\n\n"

            # 模型输出完毕，推送标准结束标志帧
            yield "data: [DONE]\n\n"

            # 正常完成时，将合并的 AI 回复完整追加进会话存储
            memory.add_ai_message(request.session_id, "".join(full_text))

        except Exception as exc:
            # 捕获上游网络中断、鉴权失败或超时等异常，推送标准错误事件帧
            error_data = json.dumps(
                {"error": f"上游模型调用失败: {str(exc)}"},
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {error_data}\n\n"

    # 4. 设置标准 SSE 流式响应头并返回
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )
