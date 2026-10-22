"""保留 Markdown 原文的完整句、列表条目、代码围栏及表格行解析。

这里只识别不可截断的内容边界；标题元数据和三格预算由 chunking 负责。
正文使用 LangChain 递归 splitter，最后一级只允许完整句边界，不逐字兜底。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 句末后的闭合引号、括号及原始空白归入同一句，防止下一块以闭引号开头。
SENTENCE_END = r"[。！？!?…]+[”’\"'）)\]】》」』]*[ \t\n]*"
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_LIST = re.compile(r"^( *)(?:[-+*]|\d+[.)])[ \t]+")
_ATX_HEADER = re.compile(r"^[ \t]*#{1,6}(?:[ \t]|$)")
_TABLE_SEPARATOR_CELL = re.compile(r":?-+:?")
_HORIZONTAL_RULE = re.compile(r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
_BLOCK_QUOTE = re.compile(r"^ {0,3}>")


@dataclass(frozen=True)
class Atom:
    """保留原文的不可拆原子；表格行另携带本表表头与表身份。"""

    text: str  # 原始内容，包含相邻分隔空白。
    kind: str  # sentence/text/list/code/table。
    line_start: int  # 当前章节内 0 基起始行。
    line_end: int  # 当前章节内 0 基结束行，不含。
    header: str = ""  # 仅表格行有本表表头及分隔行。
    table_id: int | None = None  # 以表格起始行区别同章节多张表。


class AtomError(ValueError):
    """无法保持完整结构的原子解析错误，携带章节内行范围。"""

    def __init__(self, reason: str, start: int, end: int) -> None:
        """保存错误位置。参数：原因/起止行；返回：None；异常：无。"""
        super().__init__(reason)
        self.line_start = start
        self.line_end = end


def fence_end(lines: list[str], start: int) -> int | None:
    """识别围栏的排他结束位置。参数：原始行/起点；返回：结束或非围栏 None；异常：AtomError 未闭合。"""
    opening = _FENCE.match(lines[start].rstrip("\r\n"))
    if not opening:
        return None
    marker = opening[1]
    closing = re.compile(r"^[ \t]*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}[ \t]*$")
    for index in range(start + 1, len(lines)):
        if closing.fullmatch(lines[index].rstrip("\r\n")):
            return index + 1
    raise AtomError("代码围栏未闭合，无法保持完整代码块", start, len(lines))


def _table_cells(line: str) -> list[str] | None:
    """按有效竖线识别单元格，仅供结构校验，不改写原文。

    参数：line 为单行 Markdown，可带行尾空白与换行。
    返回：list[str] 为去除外侧边框后的单元格；原行没有竖线时返回 None。
    异常：无。
    """
    text = line.strip()
    if "|" not in text:
        return None
    # 与聊天端 markdown-it 一致：先拆表格列，再解析行内代码，反引号不会保护竖线。
    # 表格规则只检查竖线紧前方的反斜杠，不按正文转义规则计算反斜杠奇偶数。
    boundaries = [index for index, char in enumerate(text)
                  if char == "|" and (index == 0 or text[index - 1] != "\\")]
    if not boundaries:
        return [text]
    starts = [0, *(boundary + 1 for boundary in boundaries)]
    ends = [*boundaries, len(text)]
    cells = [text[start:end].strip() for start, end in zip(starts, ends)]
    # 首尾竖线是可选边框，中间的空单元格仍须计入列数。
    if boundaries[-1] == len(text) - 1:
        cells.pop()
    if boundaries[0] == 0:
        cells.pop(0)
    return cells


def _table_start(lines: list[str], index: int) -> bool:
    """识别本表表头与分隔行，并校验二者列数。

    参数：lines 为原始行；index 为待识别的表头行索引。
    返回：bool，是否为含至少一列的表格起点。
    异常：AtomError，表头与合法分隔行的列数不一致。
    """
    if index + 1 >= len(lines):
        return False
    header = _table_cells(lines[index])
    separator = _table_cells(lines[index + 1])
    # markdown-it 接受单列无边框分隔行；仍要求上一行有竖线，避免误收普通水平线。
    if separator is None:
        separator = [lines[index + 1].strip()]
    if not header or not separator or not all(_TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in separator):
        return False
    if len(header) != len(separator):
        raise AtomError(
            f"表格列数不一致：表头 {len(header)} 列，分隔行 {len(separator)} 列", index, index + 2,
        )
    return True


def list_item_end(lines: list[str], start: int) -> int | None:
    """取完整条目边界，供标题保护与原子解析复用。

    参数：lines 原始行；start 待识别行索引，须满足 0 <= start < len(lines)。
    返回：int | None，整数为排他结束行，非列表返回 None。
    异常：AtomError，条目内围栏未闭合。
    """
    match = _LIST.match(lines[start])
    if match is None:
        return None
    indent = len(match[1])
    # 内容缩进包含标记和其后空白，例如 '- ' 为 2，'10. ' 为 4，不能固定写成四空格。
    content_indent = len(match[0].expandtabs(4))
    index = start + 1
    after_blank = False
    while index < len(lines):
        line = lines[index]
        sibling = _LIST.match(line)
        if sibling and len(sibling[1]) <= indent:
            break
        if line.strip():
            expanded = line.expandtabs(4)
            leading = len(expanded) - len(expanded.lstrip(" "))
            # 标题/围栏可无空行中断条目；仅达到内容缩进的结构归本条目。
            if leading < content_indent and (
                after_blank or _ATX_HEADER.match(line) or _FENCE.match(line)
            ):
                break
            # 缩进围栏属于本条目，内部的列表符号不是新条目。
            end = fence_end(lines, index)
            if end is not None:
                index = end
                after_blank = False
                continue
        after_blank = not line.strip()
        index += 1
    return index


def parse_atoms(content: str) -> list[Atom]:
    """按原序识别结构，再递归拆分正文完整句。

    参数：content 为保留缩进和空白的章节原文。
    返回：有行范围的 Atom 列表；异常：AtomError，未闭合围栏、无数据表格或表格列数不一致。
    """
    lines = content.splitlines(keepends=True)
    atoms: list[Atom] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            # 分隔空白属于前一个原子；不制造独立的空知识块。
            if atoms:
                previous = atoms[-1]
                atoms[-1] = Atom(previous.text + lines[index], previous.kind, previous.line_start,
                                 index + 1, previous.header, previous.table_id)
            index += 1
            continue
        end = fence_end(lines, index)
        if end is not None:
            atoms.append(Atom("".join(lines[index:end]), "code", index, end))
        elif _table_start(lines, index):
            header = "".join(lines[index:index + 2]).rstrip("\n") + "\n"
            columns = len(_table_cells(lines[index]) or [])
            end = index + 2
            while end < len(lines) and lines[end].strip():
                # 与页面一致，块级结构结束当前表格；普通无竖线行仍属于表内数据。
                if (_FENCE.match(lines[end]) or _LIST.match(lines[end]) or _ATX_HEADER.match(lines[end])
                        or _HORIZONTAL_RULE.fullmatch(lines[end].rstrip("\r\n"))
                        or _BLOCK_QUOTE.match(lines[end]) or lines[end].expandtabs(4).startswith("    ")):
                    break
                # 表内裸分隔线属于表后的水平线，不能把前一数据行改认成新表头。
                if end + 1 < len(lines) and "|" in lines[end + 1] and _table_start(lines, end):
                    break
                cells = _table_cells(lines[end])
                if cells is None:
                    # 无外框单列行保留原文；多列表则在下方明确报告缺列。
                    cells = [lines[end].strip()]
                # 不补齐或丢弃单元格，避免表格展示时静默丢失业务条件。
                if len(cells) != columns:
                    raise AtomError(
                        f"表格列数不一致：表头 {columns} 列，数据行 {len(cells)} 列", end, end + 1,
                    )
                atoms.append(Atom(lines[end], "table", end, end + 1, header, index))
                end += 1
            if end == index + 2:
                raise AtomError("表格缺少数据行，不能只发布表头", index, end)
        elif _LIST.match(lines[index]):
            end = list_item_end(lines, index)
            assert end is not None
            atoms.append(Atom("".join(lines[index:end]), "list", index, end))
        else:
            end = index + 1
            while end < len(lines) and lines[end].strip():
                if _FENCE.match(lines[end]) or _LIST.match(lines[end]) or _table_start(lines, end):
                    break
                end += 1
            paragraph = "".join(lines[index:end])
            # size=1 强制发现所有完整边界；没有句末时整段保留，稍后按业务硬限判断。
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1, chunk_overlap=0, length_function=len,
                separators=[r"\n[ \t]*\n", SENTENCE_END],
                keep_separator="end", is_separator_regex=True, strip_whitespace=False,
            )
            offset = 0
            for text in splitter.split_text(paragraph):
                if not text.strip():
                    continue
                start_line = index + paragraph[:offset].count("\n")
                kind = "sentence" if re.search(SENTENCE_END + r"$", text) else "text"
                atoms.append(Atom(text, kind, start_line, start_line + text.rstrip("\n").count("\n") + 1))
                offset += len(text)
        index = end
    return atoms
