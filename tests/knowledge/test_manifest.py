"""资料清单加载与路径安全的离线契约测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.knowledge.manifest import DocumentSpec, ManifestError, load_manifest

DOCUMENT_ROOT = Path("data/kb")
MANIFEST_PATH = DOCUMENT_ROOT / "manifest.json"
_TEMP_BASE = Path(".pytest-tmp")


def test_load_manifest_returns_six_documents() -> None:
    """真实清单只包含六份文档，来源键与内容类型固定。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    specs = load_manifest(MANIFEST_PATH, root=DOCUMENT_ROOT)
    assert [spec.source_key for spec in specs] == [
        "doc:product-faq.md",
        "doc:returns-policy.md",
        "doc:billing-shipping.md",
        "doc:member-benefits.md",
        "doc:after-sales-manual.md",
        "doc:product-specs.md",
    ]
    assert [spec.content_type for spec in specs] == [
        "faq",
        "policy",
        "policy",
        "policy",
        "manual",
        "manual",
    ]
    assert all(isinstance(spec, DocumentSpec) and spec.path.is_file() for spec in specs)
    assert specs[2].key_sections == ["运费与包邮"]


def _write_manifest(root: Path, entries: list[dict]) -> Path:
    """写入临时清单并返回路径。

    参数:
        root: 资料根目录。
        entries: 清单文档条目。
    返回值:
        Path: 临时 manifest.json 路径。
    异常:
        无
    """
    path = root / "manifest.json"
    path.write_text(json.dumps({"documents": entries}, ensure_ascii=False), encoding="utf-8")
    return path


def test_manifest_rejects_escape_and_duplicate_source() -> None:
    """清单拒绝越过根目录的路径、绝对路径与大小写重复来源。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    _TEMP_BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ch03-manifest-", dir=_TEMP_BASE) as temp:
        base = Path(temp)
        root = base / "kb"
        root.mkdir()
        (root / "a.md").write_text("# A\n", encoding="utf-8")
        (base / "outside.md").write_text("# Outside\n", encoding="utf-8")

        escape_manifest = _write_manifest(
            root,
            [{"path": "../outside.md", "content_type": "faq", "key_sections": []}],
        )
        with pytest.raises(ManifestError, match="越界|root|根目录"):
            load_manifest(escape_manifest, root=root)

        absolute_manifest = _write_manifest(
            root,
            [{"path": str((base / "outside.md").resolve()), "content_type": "faq", "key_sections": []}],
        )
        with pytest.raises(ManifestError, match="绝对|absolute"):
            load_manifest(absolute_manifest, root=root)

        duplicate_manifest = _write_manifest(
            root,
            [
                {"path": "a.md", "content_type": "faq", "key_sections": []},
                {"path": "A.md", "content_type": "faq", "key_sections": []},
            ],
        )
        with pytest.raises(ManifestError, match="重复|duplicate"):
            load_manifest(duplicate_manifest, root=root)
