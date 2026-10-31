"""读取主题唯一权威表并验证标签名称、排序和摘要。"""

import hashlib
import json
from pathlib import Path


def load_taxonomy(path: Path | None = None) -> dict:
    """读取权威标签。参数：path可选JSON路径；返回：版本和17类定义；异常：IO、JSON或结构错误。"""
    path = path or Path(__file__).resolve().parents[2] / "data/topics/taxonomy.json"
    taxonomy = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = taxonomy.get("labels", [])
    if not isinstance(taxonomy.get("version"), str) or not taxonomy["version"] or len(entries) != 17:
        raise ValueError("主题表必须有版本及17类")
    if any(not isinstance(row, dict) or not all(row.get(key) for key in ("name", "description", "positive", "negative")) for row in entries):
        raise ValueError("主题定义缺少名称、边界或正反例")
    names = label_names(taxonomy)
    if len(set(names)) != 17 or names[-1] != "其他":
        raise ValueError("主题名称重复或兜底顺序非法")
    return taxonomy


def label_names(taxonomy: dict) -> list[str]:
    """取权威顺序。参数：taxonomy主题表；返回：标签名称列表；异常：结构缺失KeyError。"""
    return [row["name"] for row in taxonomy["labels"]]


def validate_labels(labels: list[str], taxonomy: dict) -> list[str]:
    """验证并排序标签。参数：labels非空不重复列表、taxonomy主题表；返回：规范标签；异常：ValueError。"""
    names = label_names(taxonomy)
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or label not in names for label in labels):
        raise ValueError("主题必须为非空合法名称数组")
    if len(labels) != len(set(labels)) or ("其他" in labels and len(labels) != 1):
        raise ValueError("标签重复或其他与具体标签混用")
    return [name for name in names if name in labels]


def taxonomy_digest(taxonomy: dict) -> str:
    """计算稳定摘要。参数：taxonomy完整表；返回：SHA256十六进制；异常：JSON序列化错误。"""
    return hashlib.sha256(json.dumps(taxonomy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
