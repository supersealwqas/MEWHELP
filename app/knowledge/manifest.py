"""资料清单加载与路径安全校验。

清单显式列出本章允许导入的六份 Markdown，来源键由规范化相对路径生成；
加载阶段拒绝绝对路径、越界路径、大小写重复来源和缺失文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ManifestError(ValueError):
    """资料清单不合法。"""


@dataclass
class DocumentSpec:
    """一份可导入资料的稳定描述。"""

    path: Path
    source_key: str
    content_type: str
    key_sections: list[str]


def load_manifest(path: Path, *, root: Path) -> list[DocumentSpec]:
    """加载并校验显式资料清单。

    参数:
        path: manifest.json 路径。
        root: 资料根目录；所有文档必须位于其内。
    返回值:
        list[DocumentSpec]: 按清单顺序排列的文档描述。
    异常:
        ManifestError: JSON 格式、路径安全、文件存在性或来源唯一性不合法。
    """
    root_resolved = root.resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"资料清单不可读或格式错误: {path}") from error
    entries = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ManifestError("资料清单缺少 documents 数组")

    seen: set[str] = set()
    documents: list[DocumentSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ManifestError("资料条目必须是对象")
        raw_path = entry.get("path")
        content_type = entry.get("content_type")
        key_sections = entry.get("key_sections", [])
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ManifestError("资料条目缺少 path")
        if Path(raw_path).is_absolute():
            raise ManifestError(f"资料路径不允许使用绝对路径: {raw_path}")
        candidate = Path(raw_path)
        resolved = (root_resolved / candidate).resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ManifestError(f"资料路径越界: {raw_path}")
        if not resolved.is_file():
            raise ManifestError(f"资料文件不存在: {raw_path}")
        if not isinstance(content_type, str) or not content_type.strip():
            raise ManifestError(f"资料条目缺少 content_type: {raw_path}")
        if not isinstance(key_sections, list) or not all(
            isinstance(section, str) and section.strip() for section in key_sections
        ):
            raise ManifestError(f"资料条目 key_sections 不合法: {raw_path}")

        relative = resolved.relative_to(root_resolved).as_posix()
        source_key = f"doc:{relative}"
        folded = source_key.casefold()
        if folded in seen:
            raise ManifestError(f"资料来源重复: {source_key}")
        seen.add(folded)
        documents.append(
            DocumentSpec(
                path=resolved,
                source_key=source_key,
                content_type=content_type,
                key_sections=list(key_sections),
            )
        )
    return documents
