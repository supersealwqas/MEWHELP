"""知识三格、格式化编码输入与 Dense 向量的共享硬限和校验。"""

from __future__ import annotations

import math

from app.knowledge.contracts import TokenCounter
from app.knowledge.schemas import KnowledgeText
from app.knowledge.text import format_knowledge

MAX_KNOWLEDGE_CHARACTERS = 1600
MAX_INPUT_TOKENS = 2048
VECTOR_DIM = 1024
DEFAULT_TARGET_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64


def knowledge_overflow(
    text: KnowledgeText,
    count_tokens: TokenCounter,
    *,
    max_characters: int = MAX_KNOWLEDGE_CHARACTERS,
    max_tokens: int = MAX_INPUT_TOKENS,
) -> bool:
    """检查三格字符和格式化输入 token 硬限。

    参数：text 为三格原文；count_tokens 为实际 tokenizer；max_characters/max_tokens 为硬限。
    返回：bool，任一超限为 True。异常：tokenizer 依赖异常。
    """
    return (
        len(text.category) + len(text.questions) + len(text.answer) > max_characters
        or count_tokens(format_knowledge(text)) > max_tokens
    )


def valid_dense_vector(vector: list[float]) -> bool:
    """校验固定维度及有限浮点。参数：向量；返回：是否合法；异常：无。"""
    return len(vector) == VECTOR_DIM and all(math.isfinite(value) for value in vector)
