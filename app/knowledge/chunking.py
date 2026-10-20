"""Markdown 结构感知切分。

从 markdown-it 的结构 token 与原始行恢复标题层级、段落、列表、代码围栏和
表格；按完整句子/完整列表项/完整表格行聚合，保留父节引言和重复标题，
不把围栏内伪标题当章节，不在硬限处截断句子或表格行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from markdown_it import MarkdownIt

from app.knowledge.manifest import DocumentSpec
from app.knowledge.schemas import ChunkDraft, KnowledgeText
from app.knowledge.text import format_knowledge

CHUNKING_VERSION = "ch03-v1"

_DEFAULT_TARGET_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 64
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_MAX_CHARACTERS = 1600
_SENTENCE_ENDS = "。！？!?"
_CLOSING_CHARS = "”’」』）)】》"


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


@dataclass
class _Unit:
    """一个按原文顺序排列的结构原子集合。"""

    kind: Literal["text", "list_item", "code", "table"]
    line_start: int
    line_end: int
    text: str = ""
    table_header: str = ""
    table_separator: str = ""
    table_rows: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class _Section:
    """同一标题路径下按原文顺序排列的内容单元。"""

    stack: list[str]
    units: list[_Unit] = field(default_factory=list)


@dataclass
class _Atom:
    """切分时可独立保留的完整句子或结构项。"""

    text: str
    line_start: int
    line_end: int
    newline_before: bool


def _split_sentences(text: str) -> list[str]:
    """按中文句末标点拆分完整句子并保留闭合引号。

    参数:
        text: 段落或列表项的原始文本。
    返回值:
        list[str]: 非空完整句子；没有句末标点时返回整段。
    异常:
        无
    """
    sentences: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in _SENTENCE_ENDS:
            continue
        end = index + 1
        while end < len(text) and text[end] in _CLOSING_CHARS:
            end += 1
        sentence = text[start:end]
        if sentence.strip():
            sentences.append(sentence)
        start = end
    tail = text[start:]
    if tail.strip():
        sentences.append(tail)
    return sentences


def _map_of(token) -> tuple[int, int]:
    """读取 token 的 0 基行范围。

    参数:
        token: markdown-it token。
    返回值:
        tuple[int, int]: 起止行；缺失时返回 (0, 0)。
    异常:
        无
    """
    if not token.map:
        return 0, 0
    return int(token.map[0]), int(token.map[1])


def _table_unit(lines: list[str], start: int, end: int) -> _Unit | None:
    """把表格原始行整理为表头、分隔行和数据行。

    参数:
        lines: 原文按行拆分结果。
        start: 表格起始行。
        end: 表格结束行（不含）。
    返回值:
        _Unit | None: 表格单元；不足表头与分隔行时返回 None。
    异常:
        无
    """
    raw = [
        (index, lines[index].strip())
        for index in range(start, min(end, len(lines)))
        if lines[index].strip()
    ]
    if len(raw) < 2:
        return None
    return _Unit(
        kind="table",
        line_start=start,
        line_end=end,
        table_header=raw[0][1],
        table_separator=raw[1][1],
        table_rows=raw[2:],
    )


def _collect_sections(source: str, document: DocumentSpec) -> list[_Section]:
    """按 Markdown 结构收集带标题上下文的原子单元。

    参数:
        source: Markdown 原文。
        document: 文档描述，用于章节层级和关键条款判断。
    返回值:
        list[_Section]: 仅含实际内容的章节。
    异常:
        无；结构异常由后续硬限或空内容处理。
    """
    del document
    lines = source.splitlines()
    tokens = MarkdownIt("commonmark").enable("table").parse(source)
    sections: list[_Section] = []
    stack: list[str] = []
    current: _Section | None = None

    def ensure_section() -> _Section:
        """取得当前标题下的内容容器。参数：无；返回：章节；异常：无。"""
        nonlocal current
        if current is None:
            current = _Section(stack=list(stack))
            sections.append(current)
        return current

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "heading_open":
            level = int(token.tag[1])
            title = tokens[index + 1].content.strip() if index + 1 < len(tokens) else ""
            if title:
                del stack[level - 1 :]
                stack.append(title)
                current = None
            index += 1
        elif token.type == "paragraph_open":
            start, end = _map_of(token)
            text = "\n".join(lines[start:end]).strip()
            if text:
                ensure_section().units.append(_Unit("text", start, end, text=text))
            index += 1
        elif token.type in ("fence", "code_block"):
            start, end = _map_of(token)
            text = token.content.rstrip("\n") if token.type == "fence" else "\n".join(lines[start:end]).strip()
            if text.strip():
                ensure_section().units.append(_Unit("code", start, end, text=text))
            index += 1
        elif token.type == "table_open":
            start, end = _map_of(token)
            unit = _table_unit(lines, start, end)
            if unit is not None:
                ensure_section().units.append(unit)
            cursor = index + 1
            while cursor < len(tokens) and tokens[cursor].type != "table_close":
                cursor += 1
            index = cursor
        elif token.type in ("bullet_list_open", "ordered_list_open"):
            depth = 1
            cursor = index + 1
            while cursor < len(tokens) and depth > 0:
                inner = tokens[cursor]
                if inner.type in ("bullet_list_open", "ordered_list_open"):
                    depth += 1
                elif inner.type in ("bullet_list_close", "ordered_list_close"):
                    depth -= 1
                    if depth == 0:
                        break
                elif inner.type == "list_item_open" and depth == 1:
                    start, end = _map_of(inner)
                    text = "\n".join(lines[start:end]).strip()
                    if text:
                        ensure_section().units.append(_Unit("list_item", start, end, text=text))
                cursor += 1
            index = cursor
        else:
            index += 1
    return [section for section in sections if section.units]


def _context(stack: list[str], document: DocumentSpec) -> tuple[str, str, str | None, bool]:
    """从标题栈推导三格文本的 category/questions 与元数据。

    参数:
        stack: 当前完整标题路径。
        document: 文档描述，用于根分类和关键条款。
    返回值:
        tuple: category、questions、section_path、is_key_clause。
    异常:
        无
    """
    if stack:
        questions = stack[-1]
        category = " / ".join(stack[:-1]) if len(stack) > 1 else stack[-1]
        section_path = " / ".join(stack)
        return category, questions, section_path, stack[-1] in document.key_sections
    fallback = document.path.stem or document.source_key
    return fallback, fallback, None, False


def _atoms_for_unit(unit: _Unit) -> list[_Atom]:
    """把文本/列表/代码单元转换为完整原子。

    参数:
        unit: 结构单元。
    返回值:
        list[_Atom]: 句子或原子项。
    异常:
        无
    """
    if unit.kind == "text":
        return [
            _Atom(sentence, unit.line_start, unit.line_end, index == 0)
            for index, sentence in enumerate(_split_sentences(unit.text))
        ]
    return [_Atom(unit.text, unit.line_start, unit.line_end, True)]


class _ChunkBuilder:
    """在固定章节上下文中生成受限额约束的 ChunkDraft。"""

    def __init__(
        self,
        *,
        document: DocumentSpec,
        category: str,
        questions: str,
        section_path: str | None,
        is_key_clause: bool,
        count_tokens: Callable[[str], int],
        target_tokens: int,
        overlap_tokens: int,
        max_tokens: int,
        max_characters: int,
        chunks: list[ChunkDraft],
    ) -> None:
        """保存上下文与限额。

        参数:
            document: 当前文档。
            category: 三格分类。
            questions: 三格问法。
            section_path: 完整章节路径。
            is_key_clause: 是否关键条款。
            count_tokens: token 计数函数。
            target_tokens: 软目标。
            overlap_tokens: 重叠预算。
            max_tokens: 格式化后 token 硬限。
            max_characters: 三格字符硬限。
            chunks: 全文档结果容器。
        返回值:
            None
        异常:
            无
        """
        self.document = document
        self.category = category
        self.questions = questions
        self.section_path = section_path
        self.is_key_clause = is_key_clause
        self.count_tokens = count_tokens
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens
        self.max_characters = max_characters
        self.chunks = chunks

    def _join(self, atoms: list[_Atom]) -> str:
        """按原子边界拼接答案文本。参数：原子列表；返回：答案；异常：无。"""
        parts: list[str] = []
        for atom in atoms:
            if parts and atom.newline_before:
                parts.append("\n")
            parts.append(atom.text)
        return "".join(parts)

    def _tokens(self, answer: str) -> int:
        """计算格式化后的 token 数。参数：答案；返回：token 数；异常：计数函数异常。"""
        return self.count_tokens(
            format_knowledge(
                KnowledgeText(category=self.category, questions=self.questions, answer=answer)
            )
        )

    def _overflow(self, atoms: list[_Atom]) -> bool:
        """判断原子集合是否超过硬限。参数：原子列表；返回：是否超限；异常：无。"""
        answer = self._join(atoms)
        if len(self.category) + len(self.questions) + len(answer) > self.max_characters:
            return True
        return self._tokens(answer) > self.max_tokens

    def _error(self, reason: str, line_start: int, line_end: int) -> None:
        """抛出带位置的切分错误。参数：原因与行范围；返回：无；异常：ChunkingError。"""
        raise ChunkingError(
            reason,
            path=self.document.path,
            section_path=self.section_path,
            line_start=line_start,
            line_end=line_end,
        )

    def _emit(self, atoms: list[_Atom]) -> None:
        """提交一个知识块。参数：原子列表；返回：无；异常：无。"""
        answer = self._join(atoms)
        if not answer.strip():
            return
        self.chunks.append(
            ChunkDraft(
                source_key=self.document.source_key,
                ordinal=len(self.chunks),
                text=KnowledgeText(
                    category=self.category,
                    questions=self.questions,
                    answer=answer,
                ),
                section_path=self.section_path,
                content_type=self.document.content_type,
                is_key_clause=self.is_key_clause,
            )
        )

    def _overlap_tail(self, atoms: list[_Atom]) -> list[_Atom]:
        """取满足重叠预算的完整尾部原子。参数：当前原子；返回：重叠原子；异常：无。"""
        tail: list[_Atom] = []
        used = 0
        for atom in reversed(atoms):
            weight = self.count_tokens(atom.text)
            if weight > self.overlap_tokens - used:
                break
            tail.insert(0, atom)
            used += weight
        return tail

    def add_text(self, atoms: list[_Atom]) -> None:
        """把完整句子/列表项聚合为受目标与硬限约束的知识块。

        参数:
            atoms: 按原文顺序排列的原子。
        返回值:
            None
        异常:
            ChunkingError: 单个完整原子超过硬限。
        """
        current: list[_Atom] = []
        for atom in atoms:
            if not current:
                if self._overflow([atom]):
                    self._error("单个完整原子超过硬限", atom.line_start, atom.line_end)
                current = [atom]
                continue
            candidate = current + [atom]
            if self._overflow(candidate):
                self._emit(current)
                current = self._overlap_tail(current)
                if self._overflow(current + [atom]):
                    current = []
                    if self._overflow([atom]):
                        self._error("单个完整原子超过硬限", atom.line_start, atom.line_end)
            elif self._tokens(self._join(candidate)) > self.target_tokens:
                self._emit(current)
                current = self._overlap_tail(current)
                if self._tokens(self._join(current + [atom])) > self.target_tokens:
                    current = []
            current.append(atom)
        if current:
            self._emit(current)

    def add_table(self, unit: _Unit) -> None:
        """把表格按完整数据行分组并复制表头。

        参数:
            unit: 表格单元。
        返回值:
            None
        异常:
            ChunkingError: 表头加单行超过硬限。
        """
        rows = unit.table_rows
        if not rows:
            return

        def render(selected: list[tuple[int, str]]) -> str:
            """组装含表头与分隔行的表格文本。参数：数据行；返回：表格文本；异常：无。"""
            return "\n".join(
                [unit.table_header, unit.table_separator] + [line for _, line in selected]
            )

        current: list[tuple[int, str]] = []
        for row in rows:
            candidate = current + [row]
            answer = render(candidate)
            hard = (
                len(self.category) + len(self.questions) + len(answer) > self.max_characters
                or self._tokens(answer) > self.max_tokens
            )
            if hard:
                if not current:
                    self._error("表格表头与单行超过硬限", row[0], row[0] + 1)
                self._emit(
                    [
                        _Atom(render(current), current[0][0], current[-1][0] + 1, True)
                    ]
                )
                current = [row]
                single = render(current)
                if (
                    len(self.category) + len(self.questions) + len(single) > self.max_characters
                    or self._tokens(single) > self.max_tokens
                ):
                    self._error("表格表头与单行超过硬限", row[0], row[0] + 1)
            elif current and self._tokens(render(candidate)) > self.target_tokens:
                self._emit(
                    [
                        _Atom(render(current), current[0][0], current[-1][0] + 1, True)
                    ]
                )
                current = [row]
            else:
                current = candidate
        if current:
            self._emit(
                [_Atom(render(current), current[0][0], current[-1][0] + 1, True)]
            )


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
    """把一份 Markdown 切分为可导入的知识块。

    参数:
        source: Markdown 原文。
        document: 文档描述与清单元数据。
        count_tokens: 与编码器一致的 token 计数函数。
        target_tokens: 软目标 token 数，不用于硬截断。
        overlap_tokens: 完整句子重叠预算。
        max_tokens: 格式化后 token 硬限。
        max_characters: category/questions/answer 三格字符硬限。
    返回值:
        list[ChunkDraft]: 序号从 0 连续、按原文顺序排列的知识块。
    异常:
        ValueError: 限额参数不合法。
        ChunkingError: 完整原子超过硬限。
    """
    if target_tokens <= 0 or overlap_tokens < 0 or max_tokens <= 0 or max_characters <= 0:
        raise ValueError("切分限额必须为正数，overlap_tokens 不得为负")

    chunks: list[ChunkDraft] = []
    for section in _collect_sections(source, document):
        category, questions, section_path, is_key_clause = _context(section.stack, document)
        builder = _ChunkBuilder(
            document=document,
            category=category,
            questions=questions,
            section_path=section_path,
            is_key_clause=is_key_clause,
            count_tokens=count_tokens,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            max_characters=max_characters,
            chunks=chunks,
        )
        pending: list[_Atom] = []
        for unit in section.units:
            if unit.kind == "table":
                if pending:
                    builder.add_text(pending)
                    pending = []
                builder.add_table(unit)
            else:
                pending.extend(_atoms_for_unit(unit))
        if pending:
            builder.add_text(pending)
    return chunks
