"""API 路由模块包。

集中暴露与管理 MewHelp 电商客服系统的各类 API 路由端点，包括：
- chat_router: 客服多轮对话 SSE 流式输出路由 (POST /api/chat)
- extract_router: 售后信息结构化提取路由 (POST /api/extract)
"""

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

__all__ = ["chat_router", "extract_router"]
