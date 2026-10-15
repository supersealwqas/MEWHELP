"""客服多轮对话 SSE 流式输出 API 路由模块。

本模块提供基于 Server-Sent Events (SSE) 协议的多轮客服问答流式输出端点，
集成内存会话管理、Token 预算裁剪、智能客服角色人设 Prompt 编排及大模型流式调用。
主要包含接口：
- POST /api/chat: 接收用户消息并逐 token 推送增量回答，会话结束时回写完整 AI 历史。
"""

import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.core.memory import SessionStore, trim_history
from app.core.prompts import CUSTOMER_SERVICE_PROMPT
from app.schemas.chat import ChatRequest

# 注册客服对话 API 路由器，统一前缀为 /api，分组标签为 chat
router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

# 与参考项目第一章一致，由对话入口持有当前进程的会话存储。
store = SessionStore()


@router.post("/chat", summary="客服多轮对话 SSE 流式输出")
async def chat(req: ChatRequest) -> StreamingResponse:
    """客服多轮对话 SSE 流式响应端点。

    接收客户端提问，自动关联并裁剪会话历史，调用大语言模型进行逐 token 流式输出，
    并在输出完成时将完整的 AI 回复回写至内存会话中。若发生上游模型故障，
    向客户端推送特定格式的错误帧并安全关闭连接。

    参数:
        req (ChatRequest): 包含 session_id (会话标识符) 与 message (用户提问) 的请求体对象。

    返回值:
        fastapi.responses.StreamingResponse: media_type 为 text/event-stream 的流式 HTTP 响应，
        包含逐帧增量数据、完成标志或异常帧。

    异常:
        fastapi.exceptions.RequestValidationError: 当入参格式不合法时由 FastAPI 框架自动抛出 HTTP 422。
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        """异步生成 Server-Sent Events 协议格式的数据帧。

        逐 chunk 消费大模型流式输出，先保存完整问答再发出 [DONE]。
        初始化或流式过程中发生异常时，输出 error 帧，失败轮次不写入历史。

        参数:
            无；使用当前请求及共享的会话存储。

        返回值:
            AsyncGenerator[str, None]: 格式化后的 SSE 字符串数据流

        异常:
            内部捕获所有 Exception，转换为 SSE error 帧安全输出，对外不抛出未捕获异常。
        """
        try:
            # 在局部副本中准备本轮消息，只有完整成功后才提交到共享历史。
            user_message = HumanMessage(content=req.message)
            history = [*store.get(req.session_id), user_message]
            trimmed = trim_history(history, max_tokens=settings.token_budget)
            if not trimmed:
                # 预算无法容纳当前消息时明确提示，避免模型在缺少用户问题时作答。
                error_data = json.dumps(
                    {"error": "当前消息过长，请缩短后重试"}, ensure_ascii=False
                )
                yield f"event: error\ndata: {error_data}\n\n"
                return

            # 工厂初始化也纳入错误处理，避免创建客户端失败时直接中断 HTTP 请求。
            chain = CUSTOMER_SERVICE_PROMPT | get_chat_model(streaming=True, temperature=0.3)
            full_text: list[str] = []
            # 异步消费上游模型的 chunk 增量
            async for chunk in chain.astream({"history": trimmed}):
                # 兼容处理 chunk.content 的不同类型
                if isinstance(chunk.content, str):
                    delta = chunk.content
                elif isinstance(chunk.content, list):
                    # 只提取可展示文本，忽略推理、工具及其他内容块，避免把字典直接展示。
                    delta = "".join(
                        block if isinstance(block, str) else block["text"]
                        for block in chunk.content
                        if isinstance(block, str) or (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                        )
                    )
                else:
                    delta = ""

                # 过滤空增量，避免发送无效的数据帧
                if delta:
                    full_text.append(delta)
                    chunk_data = json.dumps({"delta": delta}, ensure_ascii=False)
                    yield f"data: {chunk_data}\n\n"

            answer = "".join(full_text)
            if not answer.strip():
                raise ValueError("模型未返回有效文本")
            # 先一次性保存完整问答，再发结束帧，客户端立即断开也不会丢历史。
            store.append(req.session_id, user_message, AIMessage(content=answer))
            yield "data: [DONE]\n\n"

        except Exception as exc:
            # 仅记录错误类型，不将上游原始响应、连接凭据等内容暴露给客户端或日志。
            logger.warning("客服模型调用失败：%s", type(exc).__name__)
            error_data = json.dumps(
                {"error": "上游模型暂时不可用，请稍后重试"},
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {error_data}\n\n"

    # 设置标准 SSE 流式响应头并返回
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )
