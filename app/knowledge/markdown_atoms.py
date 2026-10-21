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
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


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


def _table_start(lines: list[str], index: int) -> bool:
    """识别本表表头与分隔行。参数：原始行/索引；返回：bool；异常：无。"""
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and bool(_TABLE_SEPARATOR.fullmatch(lines[index + 1].rstrip("\r\n")))
    )


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
    返回：有行范围的 Atom 列表；异常：AtomError，未闭合围栏或无数据表格。
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
            end = index + 2
            while end < len(lines) and lines[end].strip() and "|" in lines[end] and not _table_start(lines, end):
                # 表格遇到围栏/列表即结束，不能把其后的正文归为数据行。
                if _FENCE.match(lines[end]) or _LIST.match(lines[end]):
                    break
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
