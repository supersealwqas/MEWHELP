"""MewHelp 核心底层业务组件包。

导出内存会话存储管理器 (InMemoryChatHistory)、全局会话存储单例 (memory)、
对话历史 Token 裁剪函数 (trim_history)、客服与提取 Prompt 模板
(CUSTOMER_SERVICE_PROMPT, EXTRACTION_PROMPT) 以及大模型客户端工厂 (get_llm)。
"""

from app.core.llm import get_llm
from app.core.memory import InMemoryChatHistory, memory, trim_history
from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACTION_PROMPT

__all__ = [
    "CUSTOMER_SERVICE_PROMPT",
    "EXTRACTION_PROMPT",
    "InMemoryChatHistory",
    "get_llm",
    "memory",
    "trim_history",
]
