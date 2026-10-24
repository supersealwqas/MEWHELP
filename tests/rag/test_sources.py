"""引用来源白名单、完整快照及文件版本变化的离线行为测试。"""

import hashlib
import json

import pytest


def test_source_reads_registered_document_and_detects_change(tmp_path):
    """旧引用保持原文，全文接口报告版本变化。参数：临时目录；返回：None；异常：断言错误。"""
    from app.knowledge.sources import read_source
    body = "# 售后\n\n## 退货\n\n原始条件。\n"
    document = tmp_path / "policy.md"
    document.write_bytes(body.encode("utf-8"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [{"path": "policy.md", "content_type": "policy"}]}), encoding="utf-8")
    citation = {"source_key": "doc:policy.md", "source_revision": hashlib.sha256(body.encode()).hexdigest(),
                "section_path": "售后 / 退货", "answer": "原始条件。"}
    original = read_source(citation, manifest_path=manifest)
    assert original["content"] == body and original["source_changed"] is False
    assert original["section_line"] == 3
    document.write_text(body + "新增条件。", encoding="utf-8")
    changed = read_source(citation, manifest_path=manifest)
    assert changed["source_changed"] is True and citation["answer"] == "原始条件。"


@pytest.mark.parametrize("source_key", ["doc:../secret.md", "doc:/secret.md", "doc:unknown.md", "qa:123", None])
def test_unregistered_or_private_source_is_not_read(tmp_path, source_key):
    """路径穿越、未知文档和历史对话来源不得读取文件。参数：目录/来源；返回：None；异常：断言错误。"""
    from app.knowledge.sources import read_source
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"documents": []}', encoding="utf-8")
    assert read_source({"source_key": source_key}, manifest_path=manifest) is None


def test_unknown_revision_is_never_presented_as_current(tmp_path):
    """历史证据没有文件版本时明确无法核对。参数：临时目录；返回：None；异常：断言错误。"""
    from app.knowledge.sources import read_source
    (tmp_path / "p.md").write_text("# 条款\n旧正文", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"documents": [{"path":"p.md","content_type":"policy"}]}', encoding="utf-8")
    result = read_source({"source_key": "doc:p.md", "source_revision": None}, manifest_path=manifest)
    assert result["source_changed"] is None


def test_version_requires_all_authoritative_chunks_to_match(tmp_path):
    """未同步的新版文件不能冒充旧知识版本。参数：临时目录；返回：None；异常：断言错误。"""
    from types import SimpleNamespace
    from app.knowledge.sources import verified_source_revision
    document = tmp_path / "p.md"
    document.write_bytes("# 政策\n\n## 条件\n\n旧正文。\n".encode())
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"documents": [{"path":"p.md","content_type":"policy"}]}', encoding="utf-8")
    rows = [SimpleNamespace(ordinal=0, questions="条件", answer="旧正文。", category="政策", section_path="政策 / 条件")]
    assert verified_source_revision("doc:p.md", rows, manifest_path=manifest) == hashlib.sha256(document.read_bytes()).hexdigest()
    document.write_bytes("# 政策\n\n## 条件\n\n新正文。\n".encode())
    assert verified_source_revision("doc:p.md", rows, manifest_path=manifest) is None
