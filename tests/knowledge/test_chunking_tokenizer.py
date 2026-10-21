"""用真实本地 BGE-M3 tokenizer 验证切分覆盖、完整原子及三格硬限。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.knowledge.chunking import CHUNKING_VERSION, chunk_markdown
from app.knowledge.embedding import BgeM3Encoder
from app.knowledge.manifest import DocumentSpec, load_manifest
from app.knowledge.text import format_knowledge


@pytest.fixture(scope="module")
def encoder() -> BgeM3Encoder:
    """加载真实本地 tokenizer，不加载 GPU 模型。参数：无；返回：编码器；异常：本地加载错误。"""
    path = Path("models/bge-m3")
    if not (path / "tokenizer.json").exists():
        pytest.skip("真实 BGE-M3 tokenizer 未安装")
    return BgeM3Encoder(Settings(knowledge_model_path=str(path.resolve())))


def test_six_documents_have_full_original_line_coverage_and_safe_real_token_lengths(encoder: BgeM3Encoder) -> None:
    """逐原文行核验六份业务资料，避免解析器自证覆盖。参数：真实编码器；返回：None；异常：断言失败。"""
    reports = []
    for spec in load_manifest(Path("data/kb/manifest.json"), root=Path("data/kb")):
        source = spec.path.read_text(encoding="utf-8")
        chunks = chunk_markdown(source, spec, encoder.count_tokens)
        answers = [chunk.text.answer for chunk in chunks]
        # 六份现有资料没有围栏；仅剔除原始 ATX 标题和空行，正文逐行独立作为覆盖标注。
        body_lines = [line.strip() for line in source.splitlines()
                      if line.strip() and not re.match(r"^#{1,6}\s", line)]
        missing = [line for line in body_lines if not any(line in answer for answer in answers)]
        assert not missing, (spec.path.name, missing)
        # 六份当前资料每个原始正文行均能容纳在一块里，按游标验证原序而非仅做集合比较。
        cursor = 0
        combined = "\n".join(answers)
        for line in body_lines:
            found = combined.find(line, cursor)
            assert found >= 0, (spec.path.name, line)
            cursor = found + len(line)
        tokens = [encoder.count_tokens(format_knowledge(chunk.text)) for chunk in chunks]
        characters = [len(c.text.category + c.text.questions + c.text.answer) for c in chunks]
        assert max(tokens) <= 2048
        assert max(characters) <= 1600
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        assert chunks == chunk_markdown(source, spec, encoder.count_tokens)
        reports.append({"file": spec.path.name, "chunks": len(chunks), "body_lines": len(body_lines),
                        "covered_lines": len(body_lines) - len(missing),
                        "max_tokens": max(tokens), "max_characters": max(characters)})
    print(json.dumps({"chunking_version": CHUNKING_VERSION, "documents": reports}, ensure_ascii=False))


def test_real_tokenizer_keeps_review_long_sentence_and_code_whole(encoder: BgeM3Encoder) -> None:
    """独立复审 R1 的长句/代码用实际 tokenizer 复验。参数：编码器；返回：None；异常：断言失败。"""
    spec = DocumentSpec(Path("review-r1.md"), "doc:review-r1.md", "policy", [])
    sentence = "退货前必须确保商品包装与配件齐全，" * 80 + "才可申请。"
    code = "```python\n# 不是章节\nif True:\n" + "    print('包装与配件必须齐全。')\n" * 42 + "```"
    for content in (sentence, code):
        chunks = chunk_markdown("# 甲\n## 乙\n" + content, spec, encoder.count_tokens)
        assert [c.text.answer for c in chunks] == [content]
        formatted_tokens = encoder.count_tokens(format_knowledge(chunks[0].text))
        assert 512 < formatted_tokens <= 2048
        assert len(chunks[0].text.category + chunks[0].text.questions + content) <= 1600
        print(json.dumps({"case": "sentence" if content == sentence else "code",
                          "chunks": len(chunks), "formatted_tokens": formatted_tokens,
                          "characters": len(chunks[0].text.category + chunks[0].text.questions + content)}))


def test_real_tokenizer_separates_two_tables_and_long_following_body(encoder: BgeM3Encoder) -> None:
    """复审 R2 两表与表后长正文不串表、不截断。参数：编码器；返回：None；异常：断言失败。"""
    spec = DocumentSpec(Path("review-r2.md"), "doc:review-r2.md", "policy", [])
    first = "| 项 | 值 |\n| --- | --- |\n| 重量 | 8kg |"
    second = "| 型号 | 是否联网 |\n| --- | --- |\n| Lite | 不支持 |\n| Pro | 支持 |"
    sentence = "退货前必须确保商品包装与配件齐全。"
    source = "# 甲\n## 乙\n前文。\n\n" + first + "\n\n中间说明。\n\n" + second + "\n\n" + sentence * 260
    chunks = chunk_markdown(source, spec, encoder.count_tokens, overlap_tokens=0)
    assert [c.text.answer for c in chunks[:4]] == ["前文。", first, "中间说明。", second]
    assert "".join(c.text.answer for c in chunks[4:]) == sentence * 260
    assert all(c.text.answer.endswith("。") for c in chunks[4:])
    assert all(encoder.count_tokens(format_knowledge(c.text)) <= 2048 for c in chunks)
    assert all(len(c.text.category + c.text.questions + c.text.answer) <= 1600 for c in chunks)
