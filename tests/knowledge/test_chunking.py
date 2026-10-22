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


@pytest.mark.parametrize(
    "header,separator,row_template",
    [
        ("| 条件 |", "| --- |", "| 条件{index}：必须验货并保留凭证 |"),
        ("| 条件 |", "---", "| 条件{index}：必须验货并保留凭证 |"),
        ("| 条件 |", "| - |", "| 条件{index}：必须验货并保留凭证 |"),
        ("| 条件 |", "| -- |", "| 条件{index}：必须验货并保留凭证 |"),
        ("| 条件 |", "| --- |", "条件{index}：必须验货并保留凭证"),
        ("| 条件 |", "---", "条件{index}：必须验货并保留凭证"),
        ("条件 |", "---: |", "条件{index}：必须验货并保留凭证 |"),
        ("| 条件", "| :---", "| 条件{index}：必须验货并保留凭证"),
        ("条件 | 处理", ":--- | ---:", "条件{index}：必须验货并保留凭证 | 受理退货"),
    ],
)
def test_long_single_and_double_column_tables_keep_every_row(
    header: str, separator: str, row_template: str,
) -> None:
    """单列和双列长表都逐行切分，并完整重复本表表头。

    参数：header 表头；separator 分隔行；row_template 带 index 占位符的数据行。
    返回：None。异常：无。
    """
    rows = [row_template.format(index=index) for index in range(120)]
    table = "\n".join([header, separator, *rows])
    assert len(table) > 1600
    chunks = _chunk("# 甲\n## 乙\n" + table, target_tokens=120, overlap_tokens=0)
    assert len(chunks) > 1
    restored_rows: list[str] = []
    for chunk in chunks:
        lines = chunk.text.answer.splitlines()
        assert lines[:2] == [header, separator]
        assert chunk.section_path == "甲 / 乙"
        assert len(chunk.text.category + chunk.text.questions + chunk.text.answer) <= 1600
        assert len(format_knowledge(chunk.text)) <= 2048
        restored_rows.extend(lines[2:])
    assert restored_rows == rows


@pytest.mark.parametrize(
    "row",
    [
        "| 退货 | 7天 | 必须验货 |",
        "| 退货 |",
        "退货 | 7天 | 必须验货",
        "退货 |",
        "| 退货 | `7天|必须验货 |",
        "| 退货 | `7天|必须验货` |",
        "| 退货 | ``7天`|必须验货`` |",
        r"正文 a\|b。",
        "必须验货",
    ],
)
def test_table_rejects_extra_or_missing_cells_with_source_location(row: str) -> None:
    """列数错误整表拒绝，避免额外条件在展示时被丢弃。

    参数：row 相对两列表头多列或少列的数据行。
    返回：None。异常：无。
    """
    with pytest.raises(ChunkingError, match="列数") as failure:
        _chunk("# 甲\n## 乙\n| 项 | 值 |\n| --- | --- |\n" + row)
    assert failure.value.path == Path("sample.md")
    assert failure.value.section_path == "甲 / 乙"
    assert (failure.value.line_start, failure.value.line_end) == (4, 5)


@pytest.mark.parametrize(
    "table",
    [
        "| 项 | 值 |\n| --- |\n| 商品 |",
        "| 项 |\n| --- | --- |\n| 商品 | 必须验货 |",
    ],
)
def test_table_rejects_header_and_separator_width_mismatch(table: str) -> None:
    """表头与分隔行列数必须一致。

    参数：table 含不一致表头和分隔行的 Markdown 表格。
    返回：None。异常：无。
    """
    with pytest.raises(ChunkingError, match="列数") as failure:
        _chunk("# 甲\n## 乙\n" + table)
    assert (failure.value.line_start, failure.value.line_end) == (2, 4)


@pytest.mark.parametrize("separator", ["| - | - |", "| -- | -- |"])
def test_short_table_separators_cannot_bypass_extra_cell_validation(separator: str) -> None:
    """页面接受的短分隔行也必须拒绝含额外条件的第三列。

    参数：separator 每列含一或两个连字符的两列表格分隔行。
    返回：None。异常：无。
    """
    source = "# 甲\n## 乙\n| 项 | 值 |\n" + separator + "\n| 退货 | 7天 | 必须验货 |"
    with pytest.raises(ChunkingError, match="列数") as failure:
        _chunk(source)
    assert (failure.value.line_start, failure.value.line_end) == (4, 5)


def test_horizontal_rule_without_pipe_header_stays_prose() -> None:
    """没有竖线表头的正文分隔线不能被当作单列表格。

    参数：无。返回：None。异常：无。
    """
    body = "退货条件\n---\n必须验货"
    assert [chunk.text.answer for chunk in _chunk("# 甲\n## 乙\n" + body)] == [body]


@pytest.mark.parametrize("marker", ["---", "***", "___", "- - -"])
def test_single_column_table_before_horizontal_rule_keeps_its_data_row(marker: str) -> None:
    """表后水平线不能让最后一行数据被误认成新表头。

    参数：marker 为页面接受的水平线标记。返回：None。异常：无。
    """
    table = "| 条件 |\n| --- |\n| 必须验货 |"
    following = marker + "\n后续说明。"
    chunks = _chunk("# 甲\n## 乙\n" + table + "\n" + following, overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == [table, following]


def test_adjacent_tables_with_explicit_separators_keep_their_own_headers() -> None:
    """相邻表格带明确分隔行时，仍各自复制表头并保留数据。

    参数：无。返回：None。异常：无。
    """
    first = "| 凭证 |\n| --- |\n| 原包装 |"
    second = "| 条件 |\n| --- |\n| 必须验货 |"
    chunks = _chunk("# 甲\n## 乙\n" + first + "\n" + second, overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == [first, second]


@pytest.mark.parametrize("row", [
    r"| a\|b | `c\|d` |", r"| a\\|b | `c\|d` |",
    r"| a\\\|b | `c\|d` |", r"| a | ``c`\|d`` |",
])
def test_table_width_ignores_escaped_pipes_in_text_and_code(row: str) -> None:
    """与页面渲染一致，竖线前紧邻反斜杠时不制造额外列。

    参数：row 含转义竖线或不同长度反引号的数据行。
    返回：None。异常：无。
    """
    table = "| 名\\|称 | 值 |\n| --- | --- |\n" + row
    assert [chunk.text.answer for chunk in _chunk("# 甲\n## 乙\n" + table)] == [table]


@pytest.mark.parametrize("gap,following", [
    ("\n", "后续必须保留凭证。"), ("", "- 先核对 a|b。"), ("", "```text\n| 表外代码 |\n```"),
])
def test_table_does_not_consume_following_prose_list_or_fence(gap: str, following: str) -> None:
    """表外正文、列表与围栏不能误作待校验的数据行。

    参数：gap 分隔空行；following 表后正文、列表或代码围栏；正文须由空行结束表格。
    返回：None。异常：无。
    """
    table = "| 项 | 值 |\n| --- | --- |\n| 退货 | 必须验货 |"
    chunks = _chunk("# 甲\n## 乙\n" + table + "\n" + gap + following, overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == [table, following]


def test_single_column_table_keeps_plain_following_line_as_data_without_blank() -> None:
    """无空行时单列普通文字仍是一行表格数据，并复制表头。

    参数：无。返回：None。异常：无。
    """
    header = "| 条件 |\n| --- |\n"
    chunks = _chunk("# 甲\n## 乙\n" + header + "必须验货\n请保留凭证。", target_tokens=20)
    assert [chunk.text.answer for chunk in chunks] == [header + "必须验货", header + "请保留凭证。"]


def test_single_column_table_ends_before_plain_following_line_after_blank() -> None:
    """空行结束无外框单列数据，表后普通说明不复制表头。

    参数：无。返回：None。异常：无。
    """
    table = "| 条件 |\n| --- |\n必须验货"
    chunks = _chunk("# 甲\n## 乙\n" + table + "\n\n请保留凭证。", overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == [table, "请保留凭证。"]


def test_single_column_table_ends_at_following_heading_without_blank() -> None:
    """无空行的新标题仍结束单列表格，后续正文沿用新章节。

    参数：无。返回：None。异常：无。
    """
    table = "| 条件 |\n| --- |\n必须验货"
    chunks = _chunk("# 甲\n## 乙\n" + table + "\n### 凭证\n请保留凭证。", overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == [table, "请保留凭证。"]
    assert [chunk.section_path for chunk in chunks] == ["甲 / 乙", "甲 / 乙 / 凭证"]


@pytest.mark.parametrize("following,expected_body", [("> 说明", "> 说明"), ("    缩进说明", "缩进说明")])
def test_single_column_table_does_not_consume_quote_or_indented_block(
    following: str, expected_body: str,
) -> None:
    """表后的引用或缩进内容仍交给原有正文流程，不能成为表格数据。

    参数：following 紧跟表格的块级内容；expected_body 现有正文流程输出的答案。
    返回：None。异常：无。
    """
    table = "| 条件 |\n| --- |\n必须验货"
    chunks = _chunk("# 甲\n## 乙\n" + table + "\n" + following, overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == [table, expected_body]


def test_single_column_table_header_can_contain_only_an_escaped_pipe() -> None:
    """单列表头中的竖线可作为转义内容，仍完整复制表头。

    参数：无。返回：None。异常：无。
    """
    header = "退货\\|退款\n| --- |\n"
    chunks = _chunk("# 甲\n## 乙\n" + header + "| 必须验货 |\n| 保留凭证 |", target_tokens=20)
    assert [chunk.text.answer for chunk in chunks] == [header + "| 必须验货 |", header + "| 保留凭证 |"]


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


def test_escaped_pipe_inside_inline_code_is_preserved() -> None:
    """表格行内代码的竖线正确转义后，不会被页面误分列或丢弃。

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
        "| a\\|b | `c\\|d` |\n"
        "| e | f |\n"
    )
    chunks = _chunk(source)
    answer = "\n".join(chunk.text.answer for chunk in chunks)
    assert "| a\\|b | `c\\|d` |" in answer
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
    assert CHUNKING_VERSION == "ch03-v3"


def test_long_sentence_below_hard_limit_stays_whole() -> None:
    """长句允许超过软目标。参数：无；返回：None；异常：无。"""
    sentence = "退货前必须确保商品包装与配件齐全，" * 70 + "才可申请。"
    chunks = _chunk("# 甲\n## 乙\n\n" + sentence, target_tokens=80)
    assert [chunk.text.answer for chunk in chunks] == [sentence]


def test_closing_quotes_belong_to_the_complete_sentence() -> None:
    """闭合引号必须跟随句末。参数：无；返回：None；异常：无。"""
    chunks = _chunk("# 甲\n## 乙\n客服说：“可以退货。”另一个问题？", target_tokens=20)
    assert [chunk.text.answer for chunk in chunks] == ["客服说：“可以退货。”", "另一个问题？"]


def test_complete_list_items_keep_continuations_and_nested_items() -> None:
    """列表无句号也不切开，缩进与后续说明保留。参数：无；返回：None；异常：无。"""
    first = "- " + "必须保留包装与配件" * 10 + "\n  另附订单截图\n  - 内层凭证"
    second = "- " + "确认商品型号" * 10
    chunks = _chunk("# 甲\n## 乙\n\n" + first + "\n" + second, target_tokens=35, overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == [first, second]


@pytest.mark.parametrize("marker", ["```", "~~~", "````"])
def test_fences_are_atomic_and_preserve_original_indentation(marker: str) -> None:
    """代码含伪标题和标点仍完整保留。参数：marker 围栏；返回：None；异常：无。"""
    code = marker + "python\n# 不是标题\nif True:\n    print('句号。')\n" + marker
    chunks = _chunk("# 甲\n## 乙\n\n前文。\n\n" + code + "\n\n后文。", target_tokens=28, overlap_tokens=0)
    assert [chunk.text.answer for chunk in chunks] == ["前文。", code, "后文。"]
    assert all(chunk.section_path == "甲 / 乙" for chunk in chunks)


def test_multiple_tables_only_repeat_their_own_headers_in_original_order() -> None:
    """多表、正文、代码按原序，表头不串表。参数：无；返回：None；异常：无。"""
    first = "| 项 | 值 |\n| --- | --- |\n| 重量 | 8kg |\n| 功率 | 30W |"
    second = "| 型号 | 是否联网 |\n| --- | --- |\n| Lite | 不支持 |\n| Pro | 支持 |"
    code = "```text\n| 不是 | 表格 |\n```"
    source = "# 甲\n## 乙\n\n表前。\n\n" + first + "\n\n表间。\n\n" + code + "\n\n" + second + "\n\n表后。"
    answers = [chunk.text.answer for chunk in _chunk(source, target_tokens=42, overlap_tokens=0)]
    assert answers == [
        "表前。", "| 项 | 值 |\n| --- | --- |\n| 重量 | 8kg |",
        "| 项 | 值 |\n| --- | --- |\n| 功率 | 30W |", "表间。", code,
        "| 型号 | 是否联网 |\n| --- | --- |\n| Lite | 不支持 |",
        "| 型号 | 是否联网 |\n| --- | --- |\n| Pro | 支持 |", "表后。",
    ]


def test_long_body_after_table_uses_sentence_boundaries() -> None:
    """表后长正文可以独立拆分。参数：无；返回：None；异常：无。"""
    table = "| 项 | 值 |\n| --- | --- |\n| 重量 | 8kg |"
    body = "请保留商品包装。" * 260
    answers = [chunk.text.answer for chunk in _chunk("# 甲\n## 乙\n" + table + "\n\n" + body, overlap_tokens=0)]
    assert answers[0] == table
    assert "".join(answers[1:]) == body
    assert all(answer.endswith("。") for answer in answers[1:])


def test_six_level_headers_supply_complete_context_and_reset() -> None:
    """六级标题进入问法/路径并在上级重置。参数：无；返回：None；异常：无。"""
    source = "# 一\n## 二\n### 三\n#### 四\n##### 五\n五级正文。\n###### 六\n六级正文。\n## 新二\n新正文。"
    chunks = _chunk(source, spec=_spec(key_sections=["六"]))
    assert [(c.text.questions, c.text.category, c.section_path, c.is_key_clause) for c in chunks] == [
        ("五", "一 / 二 / 三 / 四", "一 / 二 / 三 / 四 / 五", False),
        ("六", "一 / 二 / 三 / 四 / 五", "一 / 二 / 三 / 四 / 五 / 六", True),
        ("新二", "一", "一 / 新二", False),
    ]


@pytest.mark.parametrize("limit_name,limit", [("max_tokens", 34), ("max_characters", 23)])
def test_overlap_drops_whole_atoms_to_respect_formatted_hard_limit(limit_name: str, limit: int) -> None:
    """重叠挤占标签/三格预算时丢整句而不拒绝可拆正文。参数：限额；返回：None；异常：无。"""
    source = "# 甲\n## 乙\n" + "一" * 10 + "。" + "二" * 10 + "。" + "三" * 10 + "。"
    chunks = chunk_markdown(source, _spec(), len, target_tokens=28, overlap_tokens=64, **{limit_name: limit})
    assert [c.text.answer for c in chunks] == ["一" * 10 + "。", "二" * 10 + "。", "三" * 10 + "。"]


def test_character_hard_limit_can_split_many_short_sentences_below_token_target() -> None:
    """字符先达上限时仍按完整句切块。参数：无；返回：None；异常：无。"""
    body = "商品必须保留原包装。" * 200
    chunks = chunk_markdown("# 甲\n## 乙\n" + body, _spec(), lambda text: len(text) // 10 + 2, overlap_tokens=0)
    assert "".join(c.text.answer for c in chunks) == body
    assert len(chunks) > 1
    assert all(len(c.text.category + c.text.questions + c.text.answer) <= 1600 for c in chunks)


def test_quote_ending_counts_toward_atomic_hard_limit() -> None:
    """闭合引号计入原子硬限。参数：无；返回：None；异常：无。"""
    with pytest.raises(ChunkingError):
        chunk_markdown("# 甲\n## 乙\n“" + "长" * 10 + "。”", _spec(), len, max_characters=14)


def test_long_list_with_short_sentences_rejects_as_one_item() -> None:
    """完整列表条目超硬限不能借句号拆开。参数：无；返回：None；异常：无。"""
    with pytest.raises(ChunkingError):
        _chunk("# 甲\n## 乙\n- " + "先验货。" * 500)


def test_atomic_table_row_fits_soft_target_but_not_formatted_hard_limit() -> None:
    """复制表头和格式标签都计入硬限。参数：无；返回：None；异常：无。"""
    table = "| 项 | 值 |\n| --- | --- |\n| 重量 | 8kg |"
    with pytest.raises(ChunkingError):
        chunk_markdown("# 甲\n## 乙\n" + table, _spec(), len, target_tokens=100, max_tokens=len(table) + 12)


def test_overlap_with_multiple_previous_atoms_is_removed_when_new_sentence_fills_hard_limit() -> None:
    """真实多句尾部重叠必须让位于新完整句。参数：无；返回：None；异常：无。"""
    chunks = chunk_markdown("# 甲\n## 乙\n一一一一一。二二二二二。" + "三" * 15 + "。",
                            _spec(), len, target_tokens=27, overlap_tokens=64, max_tokens=31)
    assert [c.text.answer for c in chunks] == ["一一一一一。二二二二二。", "三" * 15 + "。"]


def test_overlap_keeps_closed_quoted_sentence_and_never_half_quote() -> None:
    """多句重叠保持完整中文引语。参数：无；返回：None；异常：无。"""
    chunks = _chunk("# 甲\n## 乙\n先验货。客服说：“可以退货。”最后再退款。", target_tokens=29, overlap_tokens=5)
    assert [c.text.answer for c in chunks] == ["先验货。客服说：“可以退货。”", "客服说：“可以退货。”最后再退款。"]


def test_unclosed_fence_reports_actual_section_and_original_lines() -> None:
    """未闭合围栏不能丢失原文。参数：无；返回：None；异常：无。"""
    with pytest.raises(ChunkingError) as failure:
        _chunk("# 甲\n\n## 乙\n\n前文。\n\n```python\n# 伪标题\nprint(1)")
    assert failure.value.section_path == "甲 / 乙"
    assert (failure.value.line_start, failure.value.line_end) == (6, 9)


def test_short_inner_fence_does_not_close_a_longer_fence() -> None:
    """更短围栏和伪标题都是外层代码内容。参数：无；返回：None；异常：无。"""
    code = "````markdown\n```\n# 伪标题\n```\n````"
    chunks = _chunk("# 甲\n## 乙\n" + code, target_tokens=25)
    assert [c.text.answer for c in chunks] == [code]
    assert chunks[0].section_path == "甲 / 乙"


def test_plain_line_break_does_not_create_half_sentence() -> None:
    """段内折行不等于句末。参数：无；返回：None；异常：无。"""
    sentence = "退货前必须确保\n全部配件和包装齐全，\n才可受理。"
    assert [c.text.answer for c in _chunk("# 甲\n## 乙\n" + sentence, target_tokens=24)] == [sentence]


def test_new_section_with_repeated_title_does_not_overlap_prior_content() -> None:
    """同名章节也是独立边界。参数：无；返回：None；异常：无。"""
    chunks = _chunk("# 甲\n## 乙\n第一句。第二句。\n## 乙\n第三句。", target_tokens=24)
    assert [c.text.answer for c in chunks] == ["第一句。第二句。", "第三句。"]


def test_table_without_data_is_rejected_instead_of_publishing_only_header() -> None:
    """表头无数据不能生成知识块。参数：无；返回：None；异常：无。"""
    with pytest.raises(ChunkingError):
        _chunk("# 甲\n## 乙\n| 项 | 值 |\n| --- | --- |")


def test_compact_table_without_outer_pipes_keeps_separate_header() -> None:
    """无外侧竖线的 Markdown 表格仍逐行保留。参数：无；返回：None；异常：无。"""
    chunks = _chunk("# 甲\n## 乙\n型号 | 联网\n--- | ---\nLite | 否\nPro | 是", target_tokens=22)
    assert [c.text.answer for c in chunks] == [
        "型号 | 联网\n--- | ---\nLite | 否", "型号 | 联网\n--- | ---\nPro | 是",
    ]


@pytest.mark.parametrize("marker,padding", [("- ", "  "), ("10. ", "    ")])
def test_list_internal_heading_preserves_item_and_following_sibling_context(marker: str, padding: str) -> None:
    """条目内部标题不污染同级条目。参数：marker 列表标记；padding 达到条目内容缩进的空白；返回：None；异常：无。"""
    item = marker + "先确认订单\n" + padding + "### 检查附件\n" + padding + "请保留完整附件"
    sibling = marker + "申请退货"
    chunks = _chunk("# 甲\n## 乙\n" + item + "\n" + sibling, target_tokens=20, overlap_tokens=0)
    assert [c.text.answer for c in chunks] == [item, sibling]
    assert [c.section_path for c in chunks] == ["甲 / 乙", "甲 / 乙"]
    assert [c.text.questions for c in chunks] == ["乙", "乙"]


@pytest.mark.parametrize("padding", ["", " ", "  ", "   "])
def test_heading_below_list_content_indent_is_an_external_section(padding: str) -> None:
    """未达到条目内容缩进的标题仍建立章节。参数：padding 为 0–3 个空格；返回：None；异常：无。"""
    source = "# 甲\n## 乙\n10. 先确认订单\n" + padding + "### 外部条款\n后续规则。\n- 新条目"
    chunks = _chunk(source, target_tokens=20, overlap_tokens=0)
    assert [c.text.answer for c in chunks] == ["10. 先确认订单", "后续规则。", "- 新条目"]
    assert [c.section_path for c in chunks] == ["甲 / 乙", "甲 / 乙 / 外部条款", "甲 / 乙 / 外部条款"]


def test_unindented_heading_after_internal_list_heading_changes_real_section() -> None:
    """离开含内部标题的条目后正常更新全局标题。参数：无；返回：None；异常：无。"""
    item = "- 先确认订单\n  ### 检查附件\n  请保留完整附件"
    source = "# 甲\n## 乙\n" + item + "\n- 申请退货\n### 退款规则\n请先验货。"
    chunks = _chunk(source, target_tokens=20, overlap_tokens=0)
    assert [c.text.answer for c in chunks] == [item, "- 申请退货", "请先验货。"]
    assert [c.section_path for c in chunks] == ["甲 / 乙", "甲 / 乙", "甲 / 乙 / 退款规则"]


@pytest.mark.parametrize("marker,fence_indent", [("- ", ""), ("10. ", "  ")])
def test_fence_below_list_content_indent_ends_item_without_blank_line(marker: str, fence_indent: str) -> None:
    """列表外围栏无空行也独立成块。参数：marker 列表标记；fence_indent 为不足条目内容缩进的空白；返回：None；异常：无。"""
    item = marker + "先确认订单"
    fence = fence_indent + "```text\n" + "a" * 25 + "\n" + fence_indent + "```"
    body = "后续正文必须保持完整。" * 3
    chunks = chunk_markdown("# 甲\n## 乙\n" + item + "\n" + fence + "\n" + body,
                            _spec(), len, target_tokens=40, max_characters=60, overlap_tokens=0)
    assert [c.text.answer for c in chunks[:2]] == [item, fence.strip()]
    # 三句正文加三格标签超过软目标，允许按完整句分块，但不能丢失或并入围栏。
    assert "".join(c.text.answer for c in chunks[2:]) == body
    assert all(c.text.answer.endswith("。") for c in chunks[2:])
    assert all(c.section_path == "甲 / 乙" for c in chunks)


@pytest.mark.parametrize("marker,padding", [("- ", "  "), ("10. ", "    ")])
def test_fence_at_list_content_indent_remains_inside_complete_item(marker: str, padding: str) -> None:
    """内部围栏与完整条目一起保留。参数：marker 列表标记；padding 达到条目内容缩进的空白；返回：None；异常：无。"""
    item = (marker + "先确认订单\n" + padding + "```text\n" + padding + "# 不是全局标题\n"
            + padding + "凭证\n" + padding + "```\n" + padding + "请保留附件")
    sibling = marker + "申请退货"
    chunks = _chunk("# 甲\n## 乙\n" + item + "\n" + sibling, target_tokens=20, overlap_tokens=0)
    assert [c.text.answer for c in chunks] == [item, sibling]
    assert all(c.section_path == "甲 / 乙" for c in chunks)
