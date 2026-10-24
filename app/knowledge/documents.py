"""参考参数的 Markdown 文档切分入口。

按 h1–h4 保存章节路径，节内正文以 400 字符递归切分，再追加最多 60 字的
完整句尾部重叠；表格按 10 数据行拆分并复制表头，表前说明放入首块。
旧 CH03 token 切分仍在 chunking.py，本模块只在新导入入口生效。
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.knowledge.chunking import ChunkingError, split_sections
from app.knowledge.contracts import TokenCounter
from app.knowledge.limits import knowledge_overflow
from app.knowledge.manifest import DocumentSpec
from app.knowledge.markdown_atoms import Atom, AtomError, parse_atoms
from app.knowledge.schemas import ChunkDraft, KnowledgeText

CHUNKING_VERSION = "ch04-char400-sentence60-table10-v1"
HEADERS = [("#" * level, f"h{level}") for level in range(1, 5)]
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]
_SENTENCE_END = re.compile(r"[。！？；!?;]+[”’\"'）)\]】》」』]*")


def _positive_integer(value: int, name: str, *, zero_allowed: bool = False) -> None:
    """验证切分限额。参数：value 整数、name 参数名、zero_allowed 是否允许零；返回：None；异常：ValueError。"""
    if type(value) is not int or value < (0 if zero_allowed else 1):
        raise ValueError(f"{name} 必须为{'非负' if zero_allowed else '正'}整数")


def recursive_split(body: str, chunk_size: int = 400) -> list[str]:
    """按中文优先分隔符切出无重叠的基础文本块。

    参数：body 为节内原文；chunk_size 为正整数字符上限，默认 400，不是 token。
    返回：list[str]，保留分隔符和原始空白的非空基础块。
    异常：ValueError，chunk_size 不合法。
    """
    _positive_integer(chunk_size, "chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, length_function=len,
        separators=SEPARATORS, keep_separator="end", strip_whitespace=False,
    )
    return [chunk for chunk in splitter.split_text(body) if chunk.strip()]


def apply_sentence_overlap(chunks: list[str], overlap: int = 60) -> list[str]:
    """向每块前面添加上一基础块尾部的完整句，不将已添加重叠级联。

    参数：chunks 为按原序、无重叠的基础块；overlap 为非负字符预算，默认 60。
    返回：list[str]，首块不变；首个尾句即使超过预算也完整保留。
    异常：ValueError，overlap 非法。
    """
    _positive_integer(overlap, "overlap", zero_allowed=True)
    if overlap == 0 or len(chunks) < 2:
        return list(chunks)
    source = "".join(chunks)
    sentences: list[tuple[int, int]] = []
    start = 0
    for ending in _SENTENCE_END.finditer(source):
        # 利用原基础块拼接后的句边界，排除刚好从半句开头开始的伪完整句。
        while start < ending.end() and source[start].isspace():
            start += 1
        sentences.append((start, ending.end()))
        start = ending.end()
    result = [chunks[0]]
    previous_start = 0
    for previous, current in zip(chunks, chunks[1:]):
        previous_end = previous_start + len(previous)
        content_end = previous_end - (len(previous) - len(previous.rstrip()))
        # 最后一句可能从更早基础块开始；结束于上一块尾部时回取原文整句，不能只复制尾片段。
        complete = [(left, right) for left, right in sentences
                    if right > previous_start and right <= content_end]
        tail_start = content_end
        # 基础块的末尾仍是半句时不越过半句去复制前面句子。
        if complete and complete[-1][1] == content_end:
            for left, _ in reversed(complete):
                if tail_start < content_end and content_end - left > overlap:
                    break
                tail_start = left
        result.append(source[tail_start:content_end] + current)
        previous_start = previous_end
    return result


def _body_groups(
    atoms: list[Atom], *, chunk_size: int, overlap: int, table_max_rows: int,
) -> list[str]:
    """按正文、代码和同表行组合节内内容，保留表前说明。

    参数：atoms 原序结构原子；chunk_size/overlap 为字符预算；table_max_rows 为数据行上限。
    返回：list[str]，有序正文块，表格各块均自带本表表头。
    异常：ValueError，递归拆分限额不合法。
    """
    result: list[str] = []
    prose: list[str] = []

    def flush_prose() -> None:
        """切分累计正文。参数：无；返回：None，追加到结果；异常：ValueError，非法字符限额。"""
        if prose:
            result.extend(apply_sentence_overlap(recursive_split("".join(prose), chunk_size), overlap))
            prose.clear()

    index = 0
    while index < len(atoms):
        atom = atoms[index]
        if atom.kind == "code":
            flush_prose()
            result.append(atom.text)
            index += 1
        elif atom.kind == "table":
            # 表前散文是该表首块的前言，不丢弃，也不重复到后续表格块。
            intro = "".join(prose).strip()
            prose.clear()
            rows: list[Atom] = []
            while index < len(atoms) and atoms[index].kind == "table" and atoms[index].table_id == atom.table_id:
                rows.append(atoms[index])
                index += 1
            for offset in range(0, len(rows), table_max_rows):
                body = atom.header + "".join(row.text for row in rows[offset:offset + table_max_rows])
                result.append((intro + "\n\n" if offset == 0 and intro else "") + body)
        else:
            prose.append(atom.text)
            index += 1
    flush_prose()
    return result


def build_document_chunks(
    source: str,
    document: DocumentSpec,
    count_tokens: TokenCounter,
    *,
    chunk_size: int = 400,
    overlap: int = 60,
    table_max_rows: int = 10,
) -> list[ChunkDraft]:
    """按参考字符参数切分 Markdown，硬限只校验而不用于选择基础切分单位。

    参数：source 原文；document 清单描述；count_tokens 编码器真实计数，仅验证硬限；
        chunk_size 正字符目标默认 400；overlap 非负完整句重叠默认 60；table_max_rows 正数据行上限默认 10。
    返回：list[ChunkDraft]，含连续序号、父分类、h1–h4 路径和完整正文。
    异常：ValueError，限额非法；ChunkingError，表格/代码不合法或最终三格超存储/编码硬限；编码依赖错误。
    """
    _positive_integer(chunk_size, "chunk_size")
    _positive_integer(overlap, "overlap", zero_allowed=True)
    _positive_integer(table_max_rows, "table_max_rows")
    chunks: list[ChunkDraft] = []
    for section in split_sections(source, headers=HEADERS):
        headings = [str(section.metadata[key]) for _, key in HEADERS if section.metadata.get(key)]
        fallback = document.path.stem or document.source_key
        question = headings[-1] if headings else fallback
        category = " / ".join(headings[:-1]) if len(headings) > 1 else question
        section_path = " / ".join(headings) if headings else None
        line_start = section.metadata["line_start"]
        try:
            atoms = parse_atoms(section.page_content)
        except AtomError as error:
            raise ChunkingError(str(error), path=document.path, section_path=section_path,
                                line_start=line_start + error.line_start,
                                line_end=line_start + error.line_end) from error
        for answer in _body_groups(atoms, chunk_size=chunk_size, overlap=overlap, table_max_rows=table_max_rows):
            if not answer.strip():
                continue
            text = KnowledgeText(category=category, questions=question, answer=answer.strip())
            # 字符目标不是最终硬限：重叠、表头与长前言都必须计入存储及编码预算。
            if knowledge_overflow(text, count_tokens):
                raise ChunkingError("完整知识块（含重叠、表头及三格）超过硬限", path=document.path,
                                    section_path=section_path, line_start=line_start,
                                    line_end=line_start + len(section.page_content.splitlines()))
            chunks.append(ChunkDraft(source_key=document.source_key, ordinal=len(chunks), text=text,
                                     section_path=section_path, content_type=document.content_type,
                                     is_key_clause=question in document.key_sections))
    return chunks
