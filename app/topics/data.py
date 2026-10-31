"""脱敏、全局家族切分、训练增强和可复核数据制品，不改业务原话。"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from app.topics.taxonomy import label_names, load_taxonomy, taxonomy_digest, validate_labels

SOURCES = {"application_demo", "application_history", "synthetic", "augmentation"}
SPLITS = {"train": .8, "validation": .1, "test": .1}


def clean_question(text: str) -> str:
    """清理明确格式和个人信息。参数：text原话；返回：脱敏文本；异常：非法类型或空文本ValueError。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("问题文本不能为空")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[邮箱]", text)
    text = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[电话]", text)
    text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[证件号]", text)
    # 只清理明确标注的个人字段或长编号，不清理尺码、型号和业务时间。
    text = re.sub(r"((?:订单号|订单编号|身份证号|证件号)\s*[:：]?\s*)[A-Za-z0-9-]{8,}", r"\1[编号]", text)
    text = re.sub(r"((?:姓名|收件人|联系人|收货地址|联系地址|地址)\s*[:：]\s*)[^，,；;。\n]+", r"\1[个人信息]", text)
    return re.sub(r"\s+", " ", text).strip()


def _validate_row(row: dict, taxonomy: dict) -> None:
    """验证统一样本。参数：row样本、taxonomy主题表；返回：无；异常：ValueError。"""
    required = {"id", "text", "labels", "source", "group_id", "pool_ids", "parent_id", "annotation_status"}
    if not required <= row.keys() or any(not isinstance(row[k], str) or not row[k] for k in ("id", "text", "group_id")):
        raise ValueError("样本字段缺失或标识非法")
    if row["source"] not in SOURCES or row["annotation_status"] not in {"model_prelabel", "human_reviewed"}:
        raise ValueError("样本来源或审核状态非法")
    if not isinstance(row["pool_ids"], list) or any(not isinstance(i, str) or not re.fullmatch(r"[1-9]\d*", i) for i in row["pool_ids"]):
        raise ValueError("pool_ids必须为正十进制字符串数组")
    if row["parent_id"] is not None and (not isinstance(row["parent_id"], str) or not row["parent_id"]):
        raise ValueError("父样本标识非法")
    if clean_question(row["text"]) != row["text"]:
        raise ValueError("样本未脱敏或格式未规范")
    if row["labels"] != validate_labels(row["labels"], taxonomy):
        raise ValueError("标签没有按权威顺序排列")
    if row["annotation_status"] == "human_reviewed" and not row.get("human_review"):
        raise ValueError("人工审核必须保留真实确认来源")


def _families(rows: list[dict]) -> list[list[dict]]:
    """合并全局连通家族。参数：rows样本；返回：不可拆分组；异常：父样本缺失或ID重复ValueError。"""
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("样本ID重复")
    parents = list(range(len(rows)))

    def find(i):
        """寻找并压缩组根。参数：i样本索引；返回：根索引；异常：无。"""
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        """合并组。参数：a/b样本索引；返回：无；异常：无。"""
        parents[find(a)] = find(b)

    seen = {}
    for i, row in enumerate(rows):
        # 纠错前脱敏正文也参与并组，防止同原话被模型修成不同文本后泄漏。
        keys = [("group", row["group_id"]), ("text", row["text"])]
        if row.get("original_text"):
            keys.append(("text", row["original_text"]))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
        if row["parent_id"]:
            if row["parent_id"] not in by_id:
                raise ValueError("父样本不在数据集中")
            union(i, by_id[row["parent_id"]])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def split_examples(rows: list[dict], taxonomy: dict, seed: int = 42) -> dict[str, list[dict]]:
    """按全局家族作确定性比例切分。参数：rows规范样本、taxonomy主题表、seed种子；返回：三份；异常：ValueError。"""
    for row in rows:
        _validate_row(row, taxonomy)
    groups = _families(deepcopy(rows))
    rng = random.Random(seed)
    groups.sort(key=lambda group: min(row["id"] for row in group))
    rng.shuffle(groups)
    totals = Counter()
    features = []
    for group in groups:
        count = Counter({"size": len(group)})
        for row in group:
            count["source:" + row["source"]] += 1
            count.update("label:" + label for label in row["labels"])
        totals.update(count)
        features.append((group, count))
    # 大组和稀有标签先安排；统一目标同时考虑样本、来源及多标签覆盖。
    features.sort(key=lambda item: (-len(item[0]), -sum(v / totals[k] for k, v in item[1].items())))
    result = {key: [] for key in SPLITS}
    counts = {key: Counter() for key in SPLITS}
    for group, feature in features:
        scores = {}
        for split, ratio in SPLITS.items():
            scores[split] = sum(((counts[split][key] + value - totals[key] * ratio) ** 2
                                 - (counts[split][key] - totals[key] * ratio) ** 2) / max(totals[key] * ratio, 1)
                                for key, value in feature.items())
        selected = min(scores, key=scores.get)
        counts[selected].update(feature)
        result[selected].extend(group)
    for part in result.values():
        part.sort(key=lambda row: row["id"])
    return result


def augment_training(rows: list[dict], taxonomy: dict, *, reserved_texts: set[str] | None = None) -> list[dict]:
    """仅为训练集作安全增强。参数：rows训练样本、taxonomy主题表、reserved_texts全部留出正文；返回：原样本及派生；异常：ValueError。"""
    result = deepcopy(rows)
    # 留出当前正文和纠错前正文均禁止成为训练派生，避免增强后才发现跨集连通。
    texts = {row["text"] for row in rows} | (reserved_texts or set())
    for row in rows:
        _validate_row(row, taxonomy)
        if row["source"] == "augmentation" or not row["text"].startswith("请问"):
            continue
        text = "想问一下" + row["text"][2:]
        if text in texts:
            continue
        derived = {**deepcopy(row), "id": row["id"] + ":aug", "text": text, "source": "augmentation", "parent_id": row["id"]}
        result.append(derived)
        texts.add(text)
    return result


def _digest(path: Path) -> str:
    """计算文件摘要。参数：path文件；返回：SHA256；异常：IO错误。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset_contents(directory: Path, parts: dict[str, list[dict]], taxonomy: dict, *, seed: int, metadata: dict | None = None,
                            pending_rows: list[dict] | None = None) -> dict:
    """写临时制品并验证全部约束。参数：临时目录、三份、主题表、种子、元信息、待复核题；返回：清单；异常：IO/校验错误。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    all_rows = [row for split in SPLITS for row in parts[split]]
    review_ids = set()
    # 各来源、各标签、近邻、多诉求、否定和全部原池题都进入可核对抽审包。
    for name in label_names(taxonomy):
        review_ids.update(row["id"] for row in all_rows if name in row["labels"] and row["source"] != "augmentation")
        candidates = [row for row in all_rows if name in row["labels"] and row["source"] != "augmentation"]
        review_ids.difference_update(row["id"] for row in candidates[3:] if row["source"] == "synthetic" and len(row["labels"]) == 1
                                    and not re.search(r"不|没|取消|撤回", row["text"]))
    review_rows = [{"split": split, **row, "human_decision": None, "human_name": None} for split, group in parts.items()
                   for row in group if row["id"] in review_ids or row["source"].startswith("application") or row.get("needs_review")]
    review_rows.extend({"split": "pending_not_in_dataset", **row, "human_decision": None, "human_name": None} for row in (pending_rows or []))
    for split, rows in parts.items():
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    review = dict(status="pending_human_review", sampling="每类前三例、全部原池、多诉求、否定和争议；近邻边界按taxonomy核对", taxonomy=taxonomy, items=review_rows)
    (directory / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# CH10 人工抽审（尚未完成）", "", "这是模型预标，不能当作人工金标准。请在review.json填写human_decision、human_name和修改理由。", "", "|ID|切分|来源|题目|模型标签|待确认|", "|---|---|---|---|---|---|"]
    lines += ["|" + "|".join(str(row.get(k, "")).replace("|", "\\|").replace("\n", " ") for k in ("id", "split", "source", "text", "labels", "review_reason")) + "|" for row in review_rows]
    (directory / "review.md").write_text("\n".join(lines), encoding="utf-8")
    files = {name: _digest(directory / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "review.json", "review.md")}
    manifest = dict(version="ch10-dataset-v1", taxonomy_version=taxonomy["version"], taxonomy_sha256=taxonomy_digest(taxonomy),
                    seed=seed, split_ratios=SPLITS, files=files, total=len(all_rows), human_review_status="pending", metadata=metadata or {},
                    counts={split: {"samples": len(group), "sources": dict(Counter(r["source"] for r in group)),
                                    "labels": dict(Counter(label for r in group for label in r["labels"]))} for split, group in parts.items()})
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validate_dataset(directory)
    return manifest


def write_dataset(directory: Path, parts: dict[str, list[dict]], taxonomy: dict, *, seed: int, metadata: dict | None = None,
                  pending_rows: list[dict] | None = None) -> dict:
    """校验通过后原子发布不可覆盖制品。参数：目录、三份、主题表、种子、元信息、待复核题；返回：清单；异常：IO/校验错误，失败保留临时诊断。"""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("制品目录已存在")
    temporary = directory.with_name(f".{directory.name}.building-{uuid4().hex}")
    try:
        manifest = _write_dataset_contents(temporary, parts, taxonomy, seed=seed, metadata=metadata, pending_rows=pending_rows)
        if directory.exists():
            raise FileExistsError("制品目录已由另一个任务创建")
        # 同一父目录改名使读者只看到完整且已校验的制品，失败不阻止原入口恢复。
        temporary.rename(directory)
        return manifest
    except Exception as error:
        if temporary.exists():
            (temporary / "failure.json").write_text(json.dumps({"status": "failed", "error_type": type(error).__name__}, ensure_ascii=False), encoding="utf-8")
        raise


def validate_dataset(directory: Path) -> dict:
    """校验文件摘要、标签、来源及家族隔离。参数：directory制品目录；返回：清单；异常：IO/ValueError。"""
    directory = Path(directory)
    taxonomy = load_taxonomy()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["taxonomy_sha256"] != taxonomy_digest(taxonomy) or manifest["taxonomy_version"] != taxonomy["version"]:
        raise ValueError("主题表摘要或版本不一致")
    required = {"train.jsonl", "validation.jsonl", "test.jsonl", "review.json", "review.md"}
    if set(manifest["files"]) != required:
        raise ValueError("数据文件清单不完整")
    for name, digest in manifest["files"].items():
        if _digest(directory / name) != digest:
            raise ValueError("数据文件摘要不匹配")
    rows, owners = [], {}
    for split in SPLITS:
        part = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for row in part:
            _validate_row(row, taxonomy)
            if split != "train" and row["source"] == "augmentation":
                raise ValueError("留出集不能包含增强")
            owners[row["id"]] = split
        rows.extend(part)
        if len(part) != manifest["counts"][split]["samples"]:
            raise ValueError("样本数量不匹配")
        if (dict(Counter(row["source"] for row in part)) != manifest["counts"][split]["sources"]
                or dict(Counter(label for row in part for label in row["labels"])) != manifest["counts"][split]["labels"]):
            raise ValueError("来源或标签支持数不匹配")
    for group in _families(rows):
        if len({owners[row["id"]] for row in group}) != 1:
            raise ValueError("家族、同文或父子跨集合泄漏")
    if len(rows) != manifest["total"]:
        raise ValueError("总数不一致")
    return manifest
