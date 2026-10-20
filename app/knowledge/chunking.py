"""基于 LangChain splitter 的 Markdown 结构感知切分。

使用 `MarkdownHeaderTextSplitter` 按 h1–h4 分节并保留 header metadata，
`RecursiveCharacterTextSplitter` 按中文段落/句末标点递归切分；再以完整句
尾部做重叠，表格按完整数据行切分并复制表头。单个完整句子/代码块/表格行
超过硬限时拒绝整份文档，不静默截断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.knowledge.manifest import DocumentSpec
from app.knowledge.schemas import ChunkDraft, KnowledgeText
from app.knowledge.text import format_knowledge

CHUNKING_VERSION = "ch03-v2"

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# 中文无词边界：优先段落/换行，再句末标点，最后逐字兜底。
CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]
_SENTENCE_RE = re.compile(r"[^。！？!?…\n]*[。！？!?…\n]|[^。！？!?…\n]+$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

_DEFAULT_TARGET_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 64
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_MAX_CHARACTERS = 1600


class ChunkingError(ValueError):
    """切分失败，携带文件、章节和行范围。"""

    def __init__(
        self,
        reason: str,
        *,
        path: Path,
        section_path: str | None,
        line_start: int,
        line_end: int,
    ) -> None:
        """构造带位置的切分错误。

        参数:
            reason: 安全的原因说明。
            path: 出错文档路径。
            section_path: 当前章节路径；根内容为 None。
            line_start: 0 基起始行。
            line_end: 0 基结束行（不含）。
        返回值:
            None
        异常:
            无
        """
        self.path = path
        self.section_path = section_path
        self.line_start = line_start
        self.line_end = line_end
        location = f"{path}:{section_path or '<root>'}:{line_start + 1}-{line_end}"
        super().__init__(f"{location}: {reason}")


@dataclass(frozen=True)
class _Context:
    """当前 section 的三格与元数据上下文。"""

    category: str
    questions: str
    section_path: str | None
    is_key_clause: bool


def split_sections(markdown: str) -> list[Document]:
    """按 Markdown 标题分节并保留 header metadata。

    参数:
        markdown: Markdown 原文。
    返回值:
        list[Document]: 带 h1–h4 metadata 的 section 文档。
    异常:
        无
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=True,
    )
    return splitter.split_text(markdown)


def recursive_split(
    text: str, chunk_size: int, count_tokens: Callable[[str], int]
) -> list[str]:
    """按中文分隔符递归切分 section 文本。

    参数:
        text: section 正文。
        chunk_size: 以 count_tokens 计的最大块大小。
        count_tokens: token 计数函数。
    返回值:
        list[str]: 递归切分结果。
    异常:
        ValueError: chunk_size 非正时抛出。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=CJK_SEPARATORS,
        is_separator_regex=False,
        length_function=count_tokens,
        # 分隔符归入前一块，避免句号被推到下一块开头而丢失完整句。
        keep_separator="end",
    )
    return splitter.split_text(text)


def _split_sentences(text: str) -> list[str]:
    """按句末标点/换行拆分完整句。参数：文本；返回：句子；异常：无。"""
    return [match for match in _SENTENCE_RE.findall(text) if match]


def _trailing_sentences(
    text: str, max_tokens: int, count_tokens: Callable[[str], int]
) -> str:
    """取尾部完整句作为重叠，超预算时仍保留最后一个完整句。参数：文本/预算/计数；返回：重叠文本；异常：无。"""
    if max_tokens <= 0:
        return ""
    selected: list[str] = []
    total = 0
    for sentence in reversed(_split_sentences(text)):
        weight = count_tokens(sentence)
        if selected and total + weight > max_tokens:
            break
        selected.insert(0, sentence)
        total += weight
    return "".join(selected)


def apply_sentence_overlap(
    chunks: list[str],
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """为后续块添加上一块尾部完整句重叠。参数：块/预算/计数；返回：重叠后块；异常：无。"""
    if not chunks:
        return []
    result = [chunks[0]]
    for index in range(1, len(chunks)):
        overlap = _trailing_sentences(chunks[index - 1], overlap_tokens, count_tokens)
        result.append(overlap + chunks[index] if overlap else chunks[index])
    return result


def _find_table_header(lines: list[str]) -> int:
    """查找表头行下标。参数：非空行；返回：表头下标或 -1；异常：无。"""
    for index in range(len(lines) - 1):
        if (
            lines[index].lstrip().startswith("|")
            and _TABLE_SEP_RE.match(lines[index + 1])
            and "-" in lines[index + 1]
        ):
            return index
    return -1


def is_table_block(text: str) -> bool:
    """判断文本是否含 Markdown 表格。参数：文本；返回：是否表格；异常：无。"""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return _find_table_header(lines) != -1


def split_table_rows(table_markdown: str, max_rows: int) -> list[str]:
    """按完整数据行切表格并复制表头。参数：表格文本/最大行数；返回：表格块；异常：无。"""
    if max_rows <= 0:
        raise ValueError("max_rows 必须为正数")
    lines = [line for line in table_markdown.strip().splitlines() if line.strip()]
    header_index = _find_table_header(lines)
    if header_index == -1:
        return [table_markdown.strip()]
    preamble = lines[:header_index]
    header, separator = lines[header_index], lines[header_index + 1]
    rows = lines[header_index + 2 :]
    if len(rows) <= max_rows:
        return [table_markdown.strip()]
    blocks: list[str] = []
    for block_index, start in enumerate(range(0, len(rows), max_rows)):
        group = rows[start : start + max_rows]
        block = (
            [*preamble, header, separator, *group]
            if block_index == 0
            else [header, separator, *group]
        )
        blocks.append("\n".join(block))
    return blocks


def _context(metadata: dict, document: DocumentSpec) -> _Context:
    """从 header metadata 推导三格与章节元数据。参数：metadata/文档；返回：上下文；异常：无。"""
    headings = [
        str(metadata[key])
        for key in ("h1", "h2", "h3", "h4")
        if metadata.get(key)
    ]
    if headings:
        questions = headings[-1]
        category = " / ".join(headings[:-1]) if len(headings) > 1 else headings[-1]
        return _Context(
            category=category,
            questions=questions,
            section_path=" / ".join(headings),
            is_key_clause=headings[-1] in document.key_sections,
        )
    fallback = document.path.stem or document.source_key
    return _Context(
        category=fallback,
        questions=fallback,
        section_path=None,
        is_key_clause=False,
    )


def _line_range(source: str, content: str) -> tuple[int, int]:
    """按 section 内容定位源文件行范围。参数：原文/内容；返回：起止行；异常：无。"""
    normalized_source = source.replace("\r\n", "\n").replace("\r", "\n")
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    index = normalized_source.find(normalized_content)
    if index < 0:
        return 0, max(1, normalized_source.count("\n") + 1)
    start = normalized_source[:index].count("\n")
    end = start + normalized_content.count("\n") + 1
    return start, end


def _hard_overflow(
    category: str,
    questions: str,
    answer: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    max_characters: int,
) -> bool:
    """判断三格文本是否超过字符/token 硬限。参数：三格与限额；返回：是否超限；异常：无。"""
    if len(category) + len(questions) + len(answer) > max_characters:
        return True
    return (
        count_tokens(
            format_knowledge(
                KnowledgeText(category=category, questions=questions, answer=answer)
            )
        )
        > max_tokens
    )


def _raise_overflow(
    *,
    document: DocumentSpec,
    context: _Context,
    source: str,
    content: str,
    reason: str,
) -> None:
    """抛出带章节/行范围的切分错误。参数：文档/上下文/原文/内容/原因；返回：不返回；异常：ChunkingError。"""
    line_start, line_end = _line_range(source, content)
    raise ChunkingError(
        reason,
        path=document.path,
        section_path=context.section_path,
        line_start=line_start,
        line_end=line_end,
    )


def _validate_atomic(
    *,
    content: str,
    document: DocumentSpec,
    context: _Context,
    source: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    max_characters: int,
) -> None:
    """拒绝超硬限的完整句子或代码块。参数：内容/上下文/限额；返回：None；异常：ChunkingError。"""
    for fence in _FENCE_RE.findall(content):
        if _hard_overflow(
            context.category,
            context.questions,
            fence,
            count_tokens,
            max_tokens,
            max_characters,
        ):
            _raise_overflow(
                document=document,
                context=context,
                source=source,
                content=content,
                reason="完整代码块超过硬限",
            )
    for sentence in _split_sentences(content):
        if _hard_overflow(
            context.category,
            context.questions,
            sentence,
            count_tokens,
            max_tokens,
            max_characters,
        ):
            _raise_overflow(
                document=document,
                context=context,
                source=source,
                content=content,
                reason="完整句子超过硬限",
            )


def _max_table_rows(
    *,
    content: str,
    context: _Context,
    target_tokens: int,
    count_tokens: Callable[[str], int],
) -> int:
    """按目标预算计算表格每块数据行数。参数：表格/上下文/目标/计数；返回：行数；异常：无。"""
    lines = [line for line in content.strip().splitlines() if line.strip()]
    header_index = _find_table_header(lines)
    if header_index == -1:
        return 1
    header_block = "\n".join(lines[header_index : header_index + 2])
    rows = lines[header_index + 2 :]
    if not rows:
        return 1
    row_size = max(count_tokens(row) for row in rows)
    overhead = count_tokens(
        f"分类：{context.category}\n问法：{context.questions}\n答案："
    )
    available = target_tokens - overhead - count_tokens(header_block)
    return max(1, available // max(1, row_size))


def chunk_markdown(
    source: str,
    document: DocumentSpec,
    count_tokens: Callable[[str], int],
    *,
    target_tokens: int = _DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_characters: int = _DEFAULT_MAX_CHARACTERS,
) -> list[ChunkDraft]:
    """把 Markdown 切分为可导入知识块。

    参数:
        source: Markdown 原文。
        document: 文档描述与清单元数据。
        count_tokens: 与编码器一致的 token 计数函数。
        target_tokens: 软目标 token 数。
        overlap_tokens: 完整句重叠预算。
        max_tokens: 格式化后 token 硬限。
        max_characters: 三格字符硬限。
    返回值:
        list[ChunkDraft]: 序号从 0 连续的知识块。
    异常:
        ChunkingError: 完整句子/代码块/表格行超过硬限。
    """
    if target_tokens <= 0 or overlap_tokens < 0 or max_tokens <= 0 or max_characters <= 0:
        raise ValueError("切分限额必须为正数，overlap_tokens 不得为负")

    chunks: list[ChunkDraft] = []

    def append(answer: str, context: _Context, content: str) -> None:
        """追加一个知识块。参数：答案/上下文/源 section；返回：None；异常：ChunkingError。"""
        cleaned = answer.strip()
        if not cleaned:
            return
        if _hard_overflow(
            context.category,
            context.questions,
            cleaned,
            count_tokens,
            max_tokens,
            max_characters,
        ):
            _raise_overflow(
                document=document,
                context=context,
                source=source,
                content=content,
                reason="切分块超过硬限",
            )
        chunks.append(
            ChunkDraft(
                source_key=document.source_key,
                ordinal=len(chunks),
                text=KnowledgeText(
                    category=context.category,
                    questions=context.questions,
                    answer=cleaned,
                ),
                section_path=context.section_path,
                content_type=document.content_type,
                is_key_clause=context.is_key_clause,
            )
        )

    for section in split_sections(source):
        content = section.page_content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not content:
            continue
        context = _context(section.metadata, document)
        if is_table_block(content):
            max_rows = _max_table_rows(
                content=content,
                context=context,
                target_tokens=target_tokens,
                count_tokens=count_tokens,
            )
            for table_block in split_table_rows(content, max_rows):
                append(table_block, context, content)
            continue
        _validate_atomic(
            content=content,
            document=document,
            context=context,
            source=source,
            count_tokens=count_tokens,
            max_tokens=max_tokens,
            max_characters=max_characters,
        )
        overhead = count_tokens(
            f"分类：{context.category}\n问法：{context.questions}\n答案："
        )
        parts = recursive_split(
            content,
            max(1, target_tokens - overhead),
            count_tokens,
        )
        for part in apply_sentence_overlap(parts, overlap_tokens, count_tokens):
            append(part, context, content)
    return chunks
