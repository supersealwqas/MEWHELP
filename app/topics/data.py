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

# 正反情态词和候选连接词只表达语法，不据此放行后续任意正文。
_FIELD_MODAL = r"(?:可不可以|需不需要|应不应该|(?P<modal>可以|需要|应该|必须|支持|允许|能|可|该|应|得|须|要|想|需|会)(?:不(?P=modal))?)"
_FIELD_CHOICE = r"(?:而不是|并不是|而非|并非|不是|不为|还是|或者|或是|或|和)"


def _is_field_business_phrase(text: str) -> bool:
    """完整识别不含具体个人值的字段动作或状态，禁止任意尾部正文。

    参数：text为字段关系词之后或疑问词之后的完整短语，可含语气词。
    返回：bool；仅完整匹配已定义业务语法时为True。异常：无业务异常。
    """
    relation = r"(?:(?:不是|不为|并非|是|为)\s*[:：]?\s*)?"
    particles = r"[的了吗呢呀吧么过]*"
    negative = r"(?:不用|无需|不必|没有|不|没|别)?"
    action = re.fullmatch(relation + negative + r"(?:" + _FIELD_MODAL + r"\s*)?" + negative +
                         r"(?:再|重新|再次|直接|继续|正常)?(?:修改|更改|变更|更换|填写|提供|确认|保留|删除|使用|配送|改|换|填|写|选|加|删)"
                         r"(?:一下|个人|本人|姓名|名字|地址|电话|错|漏|好|掉|过|"
                         r"(?:成|为)(?:公司|家庭|收货|联系|新)?(?:姓名|名字|地址|电话))?" + particles, text.strip())
    description = re.fullmatch(relation + negative + r"(?:不变|没变|未变|未定|一样|不同|有误|错误|"
                               r"正确|有效|无效|必填(?:项)?|选填(?:项)?|姓名|名字|电话|信息|方式|名单|意思|含义|"
                               r"(?:公司|家庭|收货|联系|新)?地址)" + particles, text.strip())
    return bool(action or description)


def _mask_field_candidates(text: str) -> str:
    """遮盖字段的全部候选值，保留否定/选择关系、标点和尾部语气词。

    参数：text为已移除疑问及比较前缀的字段正文，可含多个候选和逗号续接。
    返回：str；不增长长度且可重复清洗的正文。异常：无业务异常。
    """
    parts = re.split(r"([，,]?\s*" + _FIELD_CHOICE + r"\s*)", text)
    for index in range(0, len(parts), 2):
        part = parts[index]
        if not part.strip() or _is_field_business_phrase(part):
            continue
        # 只保留确定的语气词；未知字段值整体遮盖，不按姓名首字或长度放行。
        value = re.fullmatch(r"(\s*)(.*?)([吗呢呀吧么]*\s*)", part, flags=re.DOTALL)
        leading, private, ending = value.groups()
        placeholder = "[个人信息]"
        masked = placeholder if len(private) >= len(placeholder) else "*" * len(private)
        parts[index] = leading + masked + ending
    return "".join(parts)


def _mask_personal_field(match: re.Match) -> str:
    """分离字段的业务语法与具体值，避免确认问句携带个人资料。

    参数：match为第一组含字段前缀、后续正文含字段值的本模块正则匹配。
    返回：str；完整业务短语保持原样，其余候选遮盖且不增加长度。
    异常：仅接受本模块匹配结构，不产生业务异常。
    """
    original, prefix = match.group(0), match.group(1)
    content = original[len(prefix):].lstrip()
    if prefix.rstrip().endswith("联系人") and re.fullmatch(r"工(?:客服|服务|处理|办理|审核|咨询)", content):
        # “联系人工客服”中的“联系人”只是动词和角色名重叠，不是姓名字段。
        return original
    syntax = re.sub(r"^(?:姓名|收件人|联系人|收货地址|联系地址|地址)", "", original).lstrip().lstrip(":：").lstrip()
    question = re.match(r"(?:[是为]\s*[:：]?\s*)?(?:是不是|是否|能否|可否|谁|哪里|在哪|哪个|哪些|什么|怎么|如何)\s*", syntax)
    remainder = syntax[question.end():] if question else syntax
    if _is_field_business_phrase(syntax) or (question and (not remainder or _is_field_business_phrase(remainder))):
        return original
    # 疑问前缀仅界定值的位置；“是否是张三”仍需遮盖张三，不能整段返回。
    predicate = re.match(r"(?:" + _FIELD_MODAL + r"\s*)?(?:不是|不为|并非|就是|还是|是|为|"
                         r"(?:修改|更改|更换|填写|设置|改|换|填|写)(?:成|为))\s*[:：]?\s*", remainder)
    if predicate:
        remainder = remainder[predicate.end():]
    retained = original[:len(original) - len(remainder)]
    return retained + _mask_field_candidates(remainder)


def clean_question(text: str) -> str:
    """清理明确格式和个人信息。参数：text原话；返回：脱敏文本；异常：非法类型或空文本ValueError。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("问题文本不能为空")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[邮箱]", text)
    text = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[电话]", text)
    # 固话必须有完整区号和7至8位号码；字母相连的型号与日期不作电话号码。
    text = re.sub(r"(?<![A-Za-z0-9])(?:\+?86[- ]?)?\(?0\d{2,3}\)?[- ]?\d{7,8}(?:\s*[-转]\s*\d{1,6})?(?!\d)", "[电话]", text)
    text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[证件号]", text)
    # 只清理明确标注的个人字段或长编号，不清理尺码、型号和业务时间。
    text = re.sub(r"((?:订单号|订单编号|身份证号|证件号)\s*(?:[:：]|是|为)?\s*)[A-Za-z0-9-]{8,}", r"\1[编号]", text)
    # 省略“号”时只识别长订单标识，避免把订单日期2026-11-01当作编号。
    text = re.sub(r"(订单\s*(?:[:：]|是|为)?\s*)[A-Za-z0-9-]{12,}", r"\1[编号]", text)
    # 确认/否定问句也进入字段解析；逗号后明确续接的候选不能遗漏第二个姓名。
    text = re.sub(r"((?:姓名|收件人|联系人|收货地址|联系地址|地址)\s*"
                  r"(?:[:：]|[是为]\s*[:：]?|(?=(?:不是|不为|并非|能否|可否|"
                  r"(?:能|可|该|应|得|须|要|想|需|会)不|可不可以|需不需要|应不应该)))\s*)"
                  r"[^，,；;。！？?!\n]+(?:[，,]\s*" + _FIELD_CHOICE + r"\s*[^，,；;。！？?!\n]+)*",
                  _mask_personal_field, text)
    # 无分隔符的显式联系人仅接受紧邻标点的姓名，不把“联系人怎么修改”视作姓名。
    text = re.sub(r"((?:姓名|收件人|联系人)\s*)[\u4e00-\u9fffA-Za-z·]{2,4}(?=[，,；;。！？?!\n]|$)",
                  _mask_personal_field, text)
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
    """写入隔离的临时数据制品并验证分组、标签、来源和文件摘要。

    参数：directory为尚不存在的临时目录；parts包含train、validation、test三份样本列表，
    各行须符合统一数据格式，跨份家族不得重复；taxonomy为已校验的权威主题表。
    seed为实际切分所用的整数种子，只记录、不在此重新切分；metadata为可JSON序列化的
    附加来源/审核说明，None按空字典记录；pending_rows为待复核行或None，只加入抽审包，
    不进入训练/验证/测试文件及其样本计数。
    返回：dict，包含制品版本、类目版本与摘要、seed、切分比例、文件摘要、总量、
    各切分来源/标签计数、审核状态和metadata的清单。
    异常：目录已存在抛FileExistsError，文件操作可抛OSError；结构/JSON数据非法可抛
    KeyError或TypeError，分组、标签或摘要验证失败抛ValueError；调用者负责保留临时诊断。
    """
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
    """经临时目录完整校验后原子发布新制品，拒绝覆盖已有版本。

    参数：directory为尚不存在的最终目录；parts为train、validation、test三份规范样本，
    须满足标签、来源及跨份家族隔离约束；taxonomy为已校验的权威主题表。
    seed为实际切分的整数种子，此处只记录；metadata为可JSON序列化的附加说明或None；
    pending_rows为待复核行或None，仅写入抽审包，不计入三份数据及训练样本统计。
    返回：dict，由临时制品校验产生的清单，包含版本/类目/文件摘要、seed、切分比例、
    总量、来源/标签计数、审核状态及metadata；成功后全部文件位于directory。
    异常：目标已存在或并发创建抛FileExistsError；IO、序列化和制品校验错误传播，
    包括OSError、TypeError、KeyError、ValueError；失败保留临时目录及failure.json，
    诊断写入本身也可能抛OSError。
    """
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
