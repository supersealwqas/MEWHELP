"""基于 LangChain 标题与递归 splitter 的 Markdown 结构感知切分。

标题 splitter 负责 h1–h6 元数据；回映原文避免其空白规范化破坏代码/列表。
完整原子由 markdown_atoms 识别，再按最终三格预算组合；软目标可放宽，
硬限不可截断，重叠只使用完整原子且不能重复整个块。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.knowledge.contracts import TokenCounter
from app.knowledge.limits import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    MAX_INPUT_TOKENS,
    MAX_KNOWLEDGE_CHARACTERS,
    knowledge_overflow,
)
from app.knowledge.manifest import DocumentSpec
from app.knowledge.markdown_atoms import Atom, AtomError, fence_end, list_item_end, parse_atoms
from app.knowledge.schemas import ChunkDraft, KnowledgeText
from app.knowledge.text import format_knowledge

CHUNKING_VERSION = "ch03-v3"
HEADERS = [("#" * level, f"h{level}") for level in range(1, 7)]


class ChunkingError(ValueError):
    """切分失败，携带文件、章节和原文行范围。"""

    def __init__(self, reason: str, *, path: Path, section_path: str | None,
                 line_start: int, line_end: int) -> None:
        """构造位置错误。参数：原因/路径/章节/0 基起止行；返回：None；异常：无。"""
        self.path = path
        self.section_path = section_path
        self.line_start = line_start
        self.line_end = line_end
        location = f"{path}:{section_path or '<root>'}:{line_start + 1}-{line_end}"
        super().__init__(f"{location}: {reason}")


@dataclass(frozen=True)
class _Context:
    """当前章节的三格与元数据上下文。"""

    category: str
    questions: str
    section_path: str | None
    is_key_clause: bool


def split_sections(
    markdown: str, *, headers: list[tuple[str, str]] | None = None,
) -> list[Document]:
    """由 LangChain 维护标题树，同时从原文恢复完整正文。

    参数：markdown 为原文；headers 为需要识别的标题列表，None 沿用 CH03 的 h1–h6。
    返回：含指定标题层级和 line_start 的 Document。
    异常：ValueError，依赖输出无法回映原文时明确拒绝。
    """
    selected_headers = HEADERS if headers is None else headers
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    normalized = ["".join(filter(str.isprintable, line.strip())) for line in lines]
    prefix = "MEWHELP_SOURCE_LINE_"
    while prefix in markdown:
        prefix += "_"
    # 保护围栏、完整列表及缩进行；标题识别/标题栈仍交给 LangChain。
    index = 0
    while index < len(lines):
        try:
            end = fence_end(lines, index)
            if end is None:
                end = list_item_end(lines, index)
        except AtomError:
            # 将未闭合代码保留到章节解析阶段，届时可报告正确的标题路径。
            end = len(lines)
        if end is not None:
            for protected_index in range(index, end):
                normalized[protected_index] = f"{prefix}{protected_index}"
            index = end
        else:
            if lines[index].startswith(("    ", "\t")) and lines[index].strip():
                normalized[index] = f"{prefix}{index}"
            index += 1

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=selected_headers, strip_headers=True,
                                          return_each_line=True)
    sections: list[Document] = []
    cursor = 0
    previous_end = 0
    for part in splitter.split_text("\n".join(normalized)):
        positions: list[int] = []
        for line in part.page_content.splitlines():
            try:
                cursor = normalized.index(line, cursor)
            except ValueError as error:
                raise ValueError("LangChain 章节输出无法回映原文") from error
            positions.append(cursor)
            cursor += 1
        start, end = positions[0], positions[-1] + 1
        # 同标题的相邻段落恢复中间空行；遇到真实标题则不能跨章节合并/重叠。
        if (sections and {key: sections[-1].metadata[key] for _, key in selected_headers
                          if key in sections[-1].metadata} == part.metadata
                and not "".join(lines[previous_end:start]).strip()):
            sections[-1].page_content += "".join(lines[previous_end:end])
        else:
            sections.append(Document(page_content="".join(lines[start:end]),
                                     metadata={**part.metadata, "line_start": start}))
        previous_end = end
    return sections


def _context(metadata: dict, document: DocumentSpec) -> _Context:
    """推导三格与章节。参数：header metadata/清单；返回：上下文；异常：无。"""
    headings = [str(metadata[key]) for _, key in HEADERS if metadata.get(key)]
    if headings:
        return _Context(
            category=" / ".join(headings[:-1]) if len(headings) > 1 else headings[-1],
            questions=headings[-1], section_path=" / ".join(headings),
            is_key_clause=headings[-1] in document.key_sections,
        )
    fallback = document.path.stem or document.source_key
    return _Context(fallback, fallback, None, False)


def _render(atoms: list[Atom]) -> str:
    """原序拼接正文或同表行。参数：非空原子组；返回：答案；异常：无。"""
    return (atoms[0].header + "".join(atom.text for atom in atoms)).strip()


def _same_group(left: Atom, right: Atom) -> bool:
    """判断能否同块。参数：相邻原子；返回：bool；异常：无。"""
    if left.kind == "code" or right.kind == "code":
        return False
    if left.kind == "table" or right.kind == "table":
        return left.kind == right.kind and left.table_id == right.table_id
    return True


def _overlap(atoms: list[Atom], budget: int, count_tokens: TokenCounter) -> list[Atom]:
    """取上一块尾部完整句/条目，禁止整块重复。参数：原子/预算/计数；返回：尾部；异常：计数错误。"""
    selected: list[Atom] = []
    if budget <= 0:
        return selected
    for atom in reversed(atoms[1:]):
        if atom.kind not in {"sentence", "list"}:
            break
        candidate = [atom, *selected]
        if selected and count_tokens(_render(candidate)) > budget:
            break
        selected = candidate
    return selected


def chunk_markdown(
    source: str,
    document: DocumentSpec,
    count_tokens: TokenCounter,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    max_tokens: int = MAX_INPUT_TOKENS,
    max_characters: int = MAX_KNOWLEDGE_CHARACTERS,
) -> list[ChunkDraft]:
    """将 Markdown 完整原子按三格软目标与硬限组合为知识块。

    参数：source 原文；document 清单；count_tokens 真实计数函数；target_tokens
        默认软目标 512；overlap_tokens 默认 64；max_tokens 格式化硬限 2048；
        max_characters 三格合计硬限 1600。所有限额为正，重叠可为零。
    返回：序号从零连续、原序完整的 ChunkDraft 列表。
    异常：ChunkingError，完整原子超硬限或结构不支持；ValueError，非法限额。
    """
    if target_tokens <= 0 or overlap_tokens < 0 or max_tokens <= 0 or max_characters <= 0:
        raise ValueError("切分限额必须为正数，overlap_tokens 不得为负")
    chunks: list[ChunkDraft] = []
    for section in split_sections(source):
        context = _context(section.metadata, document)
        line_start = section.metadata["line_start"]

        def error(reason: str, start: int, end: int) -> ChunkingError:
            """定位到源文件。参数：原因/章节内起止行；返回：业务错误；异常：无。"""
            return ChunkingError(reason, path=document.path, section_path=context.section_path,
                                 line_start=line_start + start, line_end=line_start + end)

        try:
            atoms = parse_atoms(section.page_content)
        except AtomError as failure:
            raise error(str(failure), failure.line_start, failure.line_end) from failure

        def text(group: list[Atom]) -> KnowledgeText:
            """构造真正送编码的三格。参数：原子组；返回：KnowledgeText；异常：无。"""
            return KnowledgeText(category=context.category, questions=context.questions, answer=_render(group))

        def overflow(group: list[Atom]) -> bool:
            """按公共约束检查完整输入。参数：原子组；返回：bool；异常：计数错误。"""
            return knowledge_overflow(text(group), count_tokens, max_tokens=max_tokens,
                                      max_characters=max_characters)

        def append(group: list[Atom]) -> None:
            """追加已校验的原子组。参数：非空原子组；返回：None；异常：无。"""
            chunks.append(ChunkDraft(
                source_key=document.source_key, ordinal=len(chunks), text=text(group),
                section_path=context.section_path, content_type=document.content_type,
                is_key_clause=context.is_key_clause,
            ))

        current: list[Atom] = []
        for atom in atoms:
            if overflow([atom]):
                raise error("完整原子（含三格及本表表头）超过硬限", atom.line_start, atom.line_end)
            if not current:
                current = [atom]
                continue
            candidate = [*current, atom]
            same_group = _same_group(current[-1], atom)
            if (same_group and not overflow(candidate)
                    and count_tokens(format_knowledge(text(candidate))) <= target_tokens):
                current = candidate
                continue
            append(current)
            overlap = _overlap(current, overlap_tokens, count_tokens) if same_group else []
            # 软目标不截断原子；重叠挤占硬限时只剔除整原子，保留本次的新内容。
            while overlap and overflow([*overlap, atom]):
                overlap.pop(0)
            current = [*overlap, atom]
        if current:
            append(current)
    return chunks
