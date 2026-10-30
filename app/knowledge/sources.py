"""按已登记清单读取引用原文，保留旧证据与当前文件的版本区别。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.knowledge.manifest import ManifestError, load_manifest

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "kb" / "manifest.json"


def registered_document(source_key: str | None, manifest_path: Path):
    """从服务端白名单查找来源，不拼接客户端路径。

    参数：source_key 已保存来源键；manifest_path 服务端清单路径。
    返回：DocumentSpec 或 None；私有问答、未登记、文件缺失均为空。
    异常：无；清单不可读时按来源不可用处理。
    """
    if not source_key or not source_key.startswith("doc:"):
        return None
    try:
        return next((item for item in load_manifest(manifest_path, root=manifest_path.parent)
                     if item.source_key == source_key), None)
    except (OSError, ManifestError):
        return None


def section_line(content: str, section_path: str | None) -> int | None:
    """在全文中定位完整章节路径，忽略代码围栏中的伪标题。

    参数：content 文档全文；section_path 保存的完整章节路径。
    返回：唯一匹配的从1开始行号，否则None，避免重复标题误跳。
    异常：无。
    """
    headings, matches = {}, []
    fence = None
    for number, line in enumerate(content.splitlines(), 1):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence:
            continue
        match = re.match(r"^\s{0,3}(#{1,4})\s+(.+?)\s*#*\s*$", line)
        if match:
            level, title = len(match.group(1)), match.group(2)
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = title
            if " / ".join(headings.values()) == section_path:
                matches.append(number)
    return matches[0] if len(matches) == 1 else None


def read_source(citation: dict, *, manifest_path: Path = DEFAULT_MANIFEST) -> dict | None:
    """读取来源全文及版本差异；不会用新正文替换回答时快照。

    参数：citation 已通过会话归属校验的数据库快照；manifest_path 服务器白名单。
    返回：全文、标题、章节行号及source_changed；无合法公开文档时None，未知旧版本时差异为None。
    异常：无；文件删除、不可读或编码非法时返回None。
    """
    document = registered_document(citation.get("source_key"), manifest_path)
    if document is None:
        return None
    try:
        # 同一次读取同时用于正文和版本计算，避免文件更新时二者属于不同版本。
        raw = document.path.read_bytes()
        content = raw.decode("utf-8")
    except (OSError, UnicodeError):
        return None
    revision = hashlib.sha256(raw).hexdigest()
    old_revision = citation.get("source_revision")
    return {"title": document.path.name, "content": content, "section_path": citation.get("section_path"),
            "section_line": section_line(content, citation.get("section_path")),
            "source_revision": revision, "source_changed": old_revision != revision if old_revision else None}


def answer_metadata(context, citations: list) -> dict:
    """生成回答快照及可反馈的完成标识。

    参数：context 服务端会话上下文；citations 实际模型证据全集。
    返回：JSON兼容dict，含完整正文、正文指纹、来源类型及完成状态。
    异常：引用模型或字符串非法时由调用方输入校验处理。
    """
    snapshots = []
    for item in citations:
        snapshot = item.model_dump()
        key = item.source_key or ""
        snapshot["source_type"] = "document" if key.startswith("doc:") else "published_qa" if key.startswith(("qa:", "ch09/review/")) else "unknown"
        snapshot["content_hash"] = hashlib.sha256(item.answer.encode("utf-8")).hexdigest()
        snapshots.append(snapshot)
    return {"session_id": context.session_id, "turn_id": context.turn_id, "status": "completed", "citations": snapshots}


def verified_source_revision(source_key: str, rows: list, *, manifest_path: Path = DEFAULT_MANIFEST) -> str | None:
    """仅在整份已发布知识与当前文档切分完全一致时记录文件指纹。

    参数：source_key 权威来源；rows 同来源全部已发布块；manifest_path 服务端白名单。
    返回：SHA256或None；未同步文件、缺失块、不同切分版本都保守标为无法核对。
    异常：无；来源、编码或切分失败不妨碍保存已有证据。
    """
    from app.knowledge.documents import build_document_chunks

    document = registered_document(source_key, manifest_path)
    if document is None or not rows:
        return None
    try:
        raw = document.path.read_bytes()
        # 此处不调用模型，只验证默认400/60/10切分与仓储完整三格的一致性。
        chunks = build_document_chunks(raw.decode("utf-8"), document=document, count_tokens=len)
    except (OSError, UnicodeError, ValueError):
        return None
    expected = [(item.ordinal, item.text.category, item.text.questions, item.text.answer, item.section_path) for item in chunks]
    actual = [(item.ordinal, item.category, item.questions, item.answer, item.section_path) for item in sorted(rows, key=lambda row: row.ordinal)]
    return hashlib.sha256(raw).hexdigest() if actual == expected else None
