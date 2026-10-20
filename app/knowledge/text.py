"""知识文本格式化与内容指纹。

向量化输入固定为三格带标签文本；内容指纹覆盖三格与格式规范版本，
用于判断原文或编码输入是否变化，不把章节路径等元数据混入指纹。
"""

from __future__ import annotations

import hashlib

from app.knowledge.schemas import KnowledgeText

FORMAT_VERSION = "ch03-v1"


def format_knowledge(text: KnowledgeText) -> str:
    """把三格知识格式化为唯一向量化文本。

    参数:
        text: 分类、问法、答案三格知识。
    返回值:
        str: 固定标签顺序的格式化文本，不包含来源路径或技术元数据。
    异常:
        无
    """
    return f"分类：{text.category}\n问法：{text.questions}\n答案：{text.answer}"


def content_fingerprint(text: KnowledgeText) -> str:
    """计算三格知识内容指纹。

    参数:
        text: 分类、问法、答案三格知识。
    返回值:
        str: 覆盖格式规范版本与三格文本的 64 位十六进制 SHA256。
    异常:
        无
    """
    payload = f"{FORMAT_VERSION}\0{format_knowledge(text)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
