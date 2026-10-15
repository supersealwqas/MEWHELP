"""MewHelp 核心底层业务组件包。

导出内存会话存储管理器 (SessionStore)、
对话历史 Token 裁剪函数 (trim_history)、客服与提取 Prompt 模板
(CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT) 以及大模型客户端工厂 (get_chat_model)。
"""

from app.core.llm import get_chat_model
from app.core.memory import SessionStore, trim_history
from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT

__all__ = [
    "CUSTOMER_SERVICE_PROMPT",
    "EXTRACT_PROMPT",
    "SessionStore",
    "get_chat_model",
    "trim_history",
]
