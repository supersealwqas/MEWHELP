"""知识库资料事实基线评估：原文覆盖、关键事实与旧运费清理。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.knowledge.chunking import chunk_markdown
from app.knowledge.manifest import load_manifest
from app.knowledge.text import format_knowledge

DOCUMENT_ROOT = Path("data/kb")
FACTS_PATH = Path("tests/fixtures/ch03/facts.json")


def _load_facts() -> dict:
    """读取事实基线。

    参数:
        无
    返回值:
        dict: 事实样例与种子文件列表。
    异常:
        OSError、JSONDecodeError: 基线不可读或格式错误时抛出。
    """
    return json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def _chunks() -> list:
    """切分清单内全部文档。

    参数:
        无
    返回值:
        list: 所有 ChunkDraft。
    异常:
        ChunkingError、OSError: 切分或读取失败时抛出。
    """
    chunks = []
    for spec in load_manifest(DOCUMENT_ROOT / "manifest.json", root=DOCUMENT_ROOT):
        source = spec.path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(source, spec, len))
    return chunks


def test_manifest_documents_produce_hard_limit_chunks() -> None:
    """六份资料都能切分，且每块满足三格长度与格式化 token 硬限。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    chunks = _chunks()
    assert chunks
    assert {chunk.source_key for chunk in chunks} == {
        "doc:product-faq.md",
        "doc:returns-policy.md",
        "doc:billing-shipping.md",
        "doc:member-benefits.md",
        "doc:after-sales-manual.md",
        "doc:product-specs.md",
    }
    for chunk in chunks:
        assert len(chunk.text.category + chunk.text.questions + chunk.text.answer) <= 1600
        assert len(format_knowledge(chunk.text)) <= 2048


FACTS = _load_facts()["cases"]


@pytest.mark.parametrize("case", FACTS, ids=[case["id"] for case in FACTS])
def test_fact_case_is_covered_by_authoritative_section(case: dict) -> None:
    """每个事实样例都能在指定章节找到，并排除错误结论。

    参数:
        case: 事实样例，含来源、章节、必须出现与禁止出现文本。
    返回值:
        None
    异常:
        无
    """
    source_key = f"doc:{case['source']}"
    matches = [
        chunk
        for chunk in _chunks()
        if chunk.source_key == source_key and chunk.section_path == case["section"]
    ]
    assert matches, f"缺少章节: {source_key} / {case['section']}"
    answer = "\n".join(chunk.text.answer for chunk in matches)
    for expected in case["must_include"]:
        assert expected in answer, f"{case['id']} 缺少事实: {expected}"
    for forbidden in case["must_not_include"]:
        assert forbidden not in answer, f"{case['id']} 仍含错误结论: {forbidden}"


def test_current_seed_and_docs_follow_resolved_shipping_fact() -> None:
    """当前种子、SQL 说明与 FAQ 不再保留 10 元旧运费。

    参数:
        无
    返回值:
        None
    异常:
        OSError: 文件不可读时抛出。
    """
    for relative in _load_facts()["seed_files"]:
        text = Path(relative).read_text(encoding="utf-8")
        assert "10 元" not in text, f"{relative} 仍含 10 元"
        assert "10元" not in text, f"{relative} 仍含 10元"
        assert ("6 元" in text) or ("6元" in text), f"{relative} 缺少 6 元事实"
