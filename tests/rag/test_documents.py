"""参考字符切分的行为测试，锁定字符预算、完整句重叠及表格原文保留。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.knowledge.chunking import ChunkingError
from app.knowledge.manifest import DocumentSpec, load_manifest


def _spec() -> DocumentSpec:
    """构造文档描述。参数：无；返回：测试清单项；异常：校验失败。"""
    return DocumentSpec(path=Path("sample.md"), source_key="doc:sample.md",
                        content_type="policy", key_sections=["退款"])


def test_unpunctuated_text_uses_400_characters_without_base_overlap() -> None:
    """防止 token 计数或内置 overlap 改变 400 字基础切分。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import recursive_split

    chunks = recursive_split("甲" * 801)
    assert [len(chunk) for chunk in chunks] == [400, 400, 1]
    assert "".join(chunks) == "甲" * 801


def test_document_boundaries_do_not_depend_on_token_counter() -> None:
    """tokenizer 仅验证硬限，不能决定节内块边界。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    body = "甲" * 199 + "。" + "乙" * 199 + "。" + "丙" * 99 + "。"
    first = build_document_chunks("# 政策\n## 退款\n" + body, _spec(), len)
    second = build_document_chunks("# 政策\n## 退款\n" + body, _spec(), lambda text: len(text) // 4 + 2)
    assert [chunk.text.answer for chunk in first] == [chunk.text.answer for chunk in second]
    assert first[0].text.answer == body[:400]
    # 最后一个完整句长 200，超过 overlap=60 也不能截成半句。
    assert first[1].text.answer == "乙" * 199 + "。" + "丙" * 99 + "。"


def test_short_section_has_no_overlap_and_keeps_heading_context() -> None:
    """短节不重复正文，标题路径与父分类完整。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    chunks = build_document_chunks("# 售后政策\n## 退款\n请保留凭证。", _spec(), len)
    assert len(chunks) == 1
    assert chunks[0].text.answer == "请保留凭证。"
    assert chunks[0].text.category == "售后政策"
    assert chunks[0].text.questions == "退款"
    assert chunks[0].section_path == "售后政策 / 退款"
    assert chunks[0].is_key_clause


def test_h5_and_h6_remain_body_under_h4() -> None:
    """五六级标题不提升章节，仍保留为当前正文。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    source = "# 一\n## 二\n### 三\n#### 四\n正文。\n##### 五\n五级内容。\n###### 六\n六级内容。"
    chunks = build_document_chunks(source, _spec(), len)
    assert {chunk.section_path for chunk in chunks} == {"一 / 二 / 三 / 四"}
    assert chunks[0].text.category == "一 / 二 / 三"
    assert "##### 五" in chunks[0].text.answer and "###### 六" in chunks[0].text.answer


@pytest.mark.parametrize("previous,overlap,expected", [
    ("第一句。第二句。第三句。", 10, "第二句。第三句。后文。"),
    ("甲" * 59 + "。", 60, "甲" * 59 + "。后文。"),
    ("甲" * 70 + "。", 60, "甲" * 70 + "。后文。"),
    ("先验货。客服说：“可以退货。”", 5, "客服说：“可以退货。”后文。"),
    ("完整句。末尾半句", 60, "后文。"),
    ("没有句末的正文", 60, "后文。"),
])
def test_overlap_keeps_only_complete_tail_sentences(previous: str, overlap: int, expected: str) -> None:
    """防止截句、闭引号分离和越过半句尾部取重叠。参数：原块/预算/预期；返回：None；异常：断言失败。"""
    from app.knowledge.documents import apply_sentence_overlap

    assert apply_sentence_overlap([previous, "后文。"], overlap) == [previous, expected]


def test_overlap_uses_base_chunks_without_cascading() -> None:
    """上一块的后加重叠不能再次扩散到下一块。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import apply_sentence_overlap

    assert apply_sentence_overlap(["第一句。", "第二句。", "第三句。"], 60) == [
        "第一句。", "第一句。第二句。", "第二句。第三句。",
    ]


def test_table_23_rows_splits_10_10_3_with_intro_and_trailing_prose() -> None:
    """表格数据不能按字符切断，前言保留首块且表后正文不丢。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    header = "| 行号 | 条件 |\n| --- | --- |\n"
    rows = [f"| {number} | 必须验货{number} |" for number in range(1, 24)]
    source = "# 售后\n## 退款\n表前说明必须保留。\n\n" + header + "\n".join(rows) + "\n\n表后说明也须保留。"
    chunks = build_document_chunks(source, _spec(), len)
    tables = [chunk.text.answer for chunk in chunks if "| 行号 |" in chunk.text.answer]
    assert len(tables) == 3
    assert [sum(line.startswith("| ") and "必须验货" in line for line in table.splitlines())
            for table in tables] == [10, 10, 3]
    assert tables[0].startswith("表前说明必须保留。\n\n" + header)
    assert all(table.count("| 行号 | 条件 |") == 1 for table in tables)
    actual = [line for table in tables for line in table.splitlines() if "必须验货" in line]
    assert actual == rows
    assert chunks[-1].text.answer.endswith("表后说明也须保留。")


def test_custom_table_row_limit_preserves_every_row() -> None:
    """显式测试覆盖行数生效，不误数表头为数据行。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    body = "| 条件 |\n| --- |\n" + "\n".join(f"| 条件{index} |" for index in range(12))
    chunks = build_document_chunks("# 政策\n" + body, _spec(), len, table_max_rows=5)
    assert [len(chunk.text.answer.splitlines()) - 2 for chunk in chunks] == [5, 5, 2]


def test_multiple_tables_and_code_do_not_share_headers_or_lose_text() -> None:
    """多表、转义竖线、代码伪标题须按原序保留。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    first = "| 条件 | 值 |\n| --- | --- |\n| a\\|b | 必须验货 |"
    second = "| 型号 | 能力 |\n| --- | --- |\n| Lite | 不联网 |"
    code = "```text\n# 不是章节\n| 也不是 | 表格 |\n```"
    source = "# 政策\n## 退款\n前言。\n\n" + first + "\n\n表间。\n\n" + code + "\n\n" + second + "\n\n结尾。"
    chunks = build_document_chunks(source, _spec(), len)
    answers = [chunk.text.answer for chunk in chunks]
    assert answers == ["前言。\n\n" + first, "表间。", code, second, "结尾。"]
    assert all(chunk.section_path == "政策 / 退款" for chunk in chunks)


def test_malformed_table_reports_source_location() -> None:
    """额外单元格不能静默丢弃，错误定位到原章节行。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    with pytest.raises(ChunkingError, match="列数") as failure:
        build_document_chunks("# 政策\n## 退款\n| 条件 | 值 |\n| --- | --- |\n| 商品 | 7天 | 必须验货 |", _spec(), len)
    assert failure.value.section_path == "政策 / 退款"
    assert (failure.value.line_start, failure.value.line_end) == (4, 5)


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"overlap": -1}, {"table_max_rows": 0}, {"chunk_size": True}])
def test_invalid_chunk_parameters_are_rejected(kwargs: dict) -> None:
    """非法限额必须显式拒绝，不能被默认值吞掉。参数：限额字典；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    with pytest.raises(ValueError):
        build_document_chunks("# 政策\n知识。", _spec(), len, **kwargs)


def test_document_hard_limit_still_rejects_oversized_table_row() -> None:
    """完整表格行超存储硬限时拒绝，不能截断条件。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    with pytest.raises(ChunkingError, match="硬限"):
        build_document_chunks("# 政策\n| 条件 |\n| --- |\n| " + "长" * 1601 + " |", _spec(), len)


def test_reference_chunks_keep_all_authoritative_fixture_facts() -> None:
    """换切分后六份资料的关键事实仍在原章节，不能因表前言丢失。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.documents import build_document_chunks

    root = Path("data/kb")
    chunks = []
    for spec in load_manifest(root / "manifest.json", root=root):
        chunks.extend(build_document_chunks(spec.path.read_text(encoding="utf-8"), spec, len))
    assert len({chunk.source_key for chunk in chunks}) == 6
    cases = json.loads(Path("tests/fixtures/ch03/facts.json").read_text(encoding="utf-8"))["cases"]
    for case in cases:
        answer = "\n".join(chunk.text.answer for chunk in chunks
                           if chunk.source_key == f"doc:{case['source']}" and chunk.section_path == case["section"])
        for expected in case["must_include"]:
            assert expected in answer, case["id"]
        for forbidden in case["must_not_include"]:
            assert forbidden not in answer, case["id"]
