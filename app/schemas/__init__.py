"""数据契约与 Schema 统一导出模块。

集中对外导出对话与售后相关的 Pydantic 数据模型：
- ChatRequest: 对话请求入参模型
- ChatChunk: 对话流式响应增量模型
- ExtractRequest: 售后提取请求入参模型
- AfterSalesTicket: 售后工单结构化数据模型
"""

from app.schemas.after_sales import AfterSalesTicket, ExtractRequest
from app.schemas.chat import ChatChunk, ChatRequest

__all__ = [
    "AfterSalesTicket",
    "ChatChunk",
    "ChatRequest",
    "ExtractRequest",
]
