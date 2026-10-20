"""Markdown 结构切分的离线契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.knowledge.chunking import CHUNKING_VERSION, ChunkingError, chunk_markdown
from app.knowledge.manifest import DocumentSpec
from app.knowledge.text import format_knowledge

CASES_PATH = Path("tests/fixtures/ch03/chunk_cases.json")


def _cases() -> dict:
    """读取切分夹具。

    参数:
        无
    返回值:
        dict: 测试样例。
    异常:
        OSError、JSONDecodeError: 夹具不可读或格式错误时抛出。
    """
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _spec(
    path: str = "sample.md",
    content_type: str = "policy",
    key_sections: list[str] | None = None,
) -> DocumentSpec:
    """构造切分测试文档描述。

    参数:
        path: 文档相对路径。
        content_type: 内容类型。
        key_sections: 关键条款章节标题。
    返回值:
        DocumentSpec: 测试文档。
    异常:
        无
    """
    return DocumentSpec(
        path=Path(path),
        source_key=f"doc:{path}",
        content_type=content_type,
        key_sections=list(key_sections or []),
    )


def _chunk(
    source: str,
    *,
    spec: DocumentSpec | None = None,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list:
    """用字符长度模拟 tokenizer 执行切分。

    参数:
        source: Markdown 原文。
        spec: 文档描述；默认 sample.md。
        target_tokens: 软目标 token 数。
        overlap_tokens: 重叠 token 预算。
    返回值:
        list: ChunkDraft 列表。
    异常:
        ChunkingError: 原子超过硬限时抛出。
    """
    return chunk_markdown(
        source,
        spec or _spec(),
        len,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )


def test_parent_intro_and_duplicate_heading_paths() -> None:
    """父节引言保留，重复标题不丢内容且路径一致。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    source = (
        "# 甲\n\n引言。\n\n"
        "## 乙\n\n第一句。第二句。第三句。\n\n"
        "## 乙\n\n重复标题内容。\n"
    )
    chunks = _chunk(source, spec=_spec(key_sections=["乙"]))
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))

    intro = chunks[0]
    assert intro.section_path == "甲"
    assert intro.text.category == "甲"
    assert intro.text.questions == "甲"
    assert intro.text.answer == "引言。"
    assert intro.is_key_clause is False

    policy_chunks = [chunk for chunk in chunks if chunk.text.questions == "乙"]
    assert policy_chunks
    assert all(chunk.text.category == "甲" for chunk in policy_chunks)
    assert all(chunk.section_path == "甲 / 乙" for chunk in policy_chunks)
    assert all(chunk.is_key_clause is True for chunk in policy_chunks)
    combined = "\n".join(chunk.text.answer for chunk in policy_chunks)
    assert "第一句。第二句。第三句。" in combined
    assert "重复标题内容。" in combined


def test_sentence_overlap_keeps_punctuation() -> None:
    """按完整句子重叠，第二块以第二句开头并保留问号/句号。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    case = _cases()["sentence_overlap"]
    chunks = _chunk(
        case["source"],
        target_tokens=case["target_tokens"],
        overlap_tokens=case["overlap_tokens"],
    )
    assert [chunk.text.answer for chunk in chunks] == case["expected_answers"]
    assert all(chunk.text.category == "甲" and chunk.text.questions == "乙" for chunk in chunks)
    assert all(chunk.section_path == "甲 / 乙" for chunk in chunks)


def test_table_headers_repeated_without_lost_rows() -> None:
    """大表格逐行分组，每块复制表头且不丢失任何数据行。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    row_count = _cases()["large_table_rows"]
    table_lines = ["| 行号 | 说明 |", "| --- | --- |"]
    table_lines.extend(f"| {index} | 内容{index} |" for index in range(1, row_count + 1))
    source = "# 售后手册\n\n## 常见问题处理时限\n\n" + "\n".join(table_lines) + "\n"
    chunks = _chunk(source, target_tokens=120, overlap_tokens=0)
    assert len(chunks) > 1

    row_ids: list[int] = []
    for chunk in chunks:
        assert "| 行号 | 说明 |" in chunk.text.answer
        assert "| --- | --- |" in chunk.text.answer
        assert chunk.text.category == "售后手册"
        assert chunk.text.questions == "常见问题处理时限"
        assert chunk.section_path == "售后手册 / 常见问题处理时限"
        assert len(chunk.text.category + chunk.text.questions + chunk.text.answer) <= 1600
        assert len(format_knowledge(chunk.text)) <= 2048
        for line in chunk.text.answer.splitlines():
            if line.startswith("| ") and "内容" in line:
                row_ids.append(int(line.split("|")[1].strip()))

    unique_ids = list(dict.fromkeys(row_ids))
    assert unique_ids == list(range(1, row_count + 1))


def test_code_fence_heading_is_not_section() -> None:
    """代码围栏内的 # 行不产生新章节，代码内容完整保留。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    source = "# 甲\n\n## 乙\n\n```python\n# 不是标题\nprint('x')\n```\n"
    chunks = _chunk(source)
    assert all(chunk.text.questions != "不是标题" for chunk in chunks)
    assert all(chunk.section_path == "甲 / 乙" for chunk in chunks)
    assert any("print('x')" in chunk.text.answer for chunk in chunks)


def test_inline_code_pipe_and_escaped_pipe() -> None:
    """表格行中的转义竖线与行内代码竖线不被当成列分隔。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    source = (
        "# 表\n\n## 数据\n\n"
        "| 名称 | 值 |\n"
        "| --- | --- |\n"
        "| a\\|b | `c|d` |\n"
        "| e | f |\n"
    )
    chunks = _chunk(source)
    answer = "\n".join(chunk.text.answer for chunk in chunks)
    assert "| a\\|b | `c|d` |" in answer
    assert "| e | f |" in answer
    assert chunks[0].section_path == "表 / 数据"


def test_crlf_source_maps() -> None:
    """CRLF 原文与 LF 原文得到相同的块顺序和内容。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    cases = _cases()
    lf_chunks = _chunk(
        cases["sentence_overlap"]["source"],
        target_tokens=cases["sentence_overlap"]["target_tokens"],
        overlap_tokens=cases["sentence_overlap"]["overlap_tokens"],
    )
    crlf_chunks = _chunk(
        cases["crlf_source"],
        target_tokens=cases["sentence_overlap"]["target_tokens"],
        overlap_tokens=cases["sentence_overlap"]["overlap_tokens"],
    )
    assert [
        (chunk.text.category, chunk.text.questions, chunk.text.answer, chunk.section_path)
        for chunk in crlf_chunks
    ] == [
        (chunk.text.category, chunk.text.questions, chunk.text.answer, chunk.section_path)
        for chunk in lf_chunks
    ]


def test_atomic_oversize_rejects_whole_document() -> None:
    """单个无句末原子超过 1600 字符时拒绝整份文档并报告位置。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    source = "# 甲\n\n## 乙\n\n" + "长" * _cases()["oversize_characters"] + "\n"
    with pytest.raises(ChunkingError) as error:
        _chunk(source)
    message = str(error.value)
    assert "sample.md" in message
    assert "甲 / 乙" in message
    assert "5" in message


def test_chunking_version_is_stable() -> None:
    """切分规范版本固定，供检索策略与校准匹配。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    assert CHUNKING_VERSION == "ch03-v2"
