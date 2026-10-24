"""评估语料指纹与仅由校准集产生的分策略门槛，不复用应用默认阈值。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from app.knowledge.schemas import ChunkSnapshot

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
POLICY_VERSION = "ch04-eval-thresholds-v1"
POLICY_BINDINGS = ("corpus_digest", "dataset_digest", "model", "judge_model", "model_endpoint_digest",
                   "embedding_revision", "reranker_revision", "embedding_artifacts_digest",
                   "reranker_artifacts_digest", "prompt_digest", "source_digest", "chunking_version",
                   "query_rules_version", "request_budget", "evidence_budget", "generation_max_tokens", "temperature")


def build_corpus(manifest_path: Path, *, embedding_revision: str, count_tokens=len) -> list[ChunkSnapshot]:
    """无连接配置地按真实字符规则切分本地清单，供离线校验与真实准备共用。

    参数：manifest_path 清单；embedding_revision 明确编码版本；count_tokens 真实准备传本地token计数。
    返回：临时编号快照，跨库摘要不依赖编号。
    异常：清单/切分/文件错误，不加载聊天配置或模型。
    """
    from app.knowledge.documents import build_document_chunks
    from app.knowledge.manifest import load_manifest
    from app.knowledge.text import content_fingerprint

    result = []
    for document in load_manifest(manifest_path, root=manifest_path.parent):
        for draft in build_document_chunks(document.path.read_text(encoding="utf-8"), document, count_tokens=count_tokens):
            result.append(ChunkSnapshot(**draft.model_dump(), id=len(result) + 1,
                                        content_hash=content_fingerprint(draft.text), embedding_revision=embedding_revision))
    return result


def digest_json(value: object) -> str:
    """计算规范JSON摘要。参数：value 可序列化内容；返回：SHA256；异常：TypeError/ValueError。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def corpus_digest(chunks: list[ChunkSnapshot]) -> str:
    """按稳定来源和正文摘要，不包含数据库ID、同步状态或时间戳。

    参数：chunks 权威快照，来源/序号不能重复。
    返回：str，跨数据库可比的SHA256，正文、过滤元数据及编码规范变化均改变摘要。
    异常：ValueError，稳定来源重复。
    """
    keys = [(chunk.source_key, chunk.ordinal) for chunk in chunks]
    if len(keys) != len(set(keys)):
        raise ValueError("语料稳定来源和序号重复")
    return digest_json([{key: value for key, value in chunk.model_dump().items()
                         if key not in {"id", "vector_id", "vectorize_status", "prev_chunk_id", "next_chunk_id"}}
                        for chunk in sorted(chunks, key=lambda item: (item.source_key, item.ordinal))])


def calibrate_threshold(rows: list[dict], *, strategy: str) -> dict:
    """仅扫描本策略校准题的已观测分界，最大化正负样本平衡准确率。

    参数：rows 含split/strategy/answerable/top_score/error记录；strategy 四策略之一。
    返回：dict，minimum=None 表示没有可用区分力；失败不参与选阈值但单独计数。
    异常：ValueError，混入验收题、其他策略或非有限分数。
    """
    if strategy not in STRATEGIES or any(row["strategy"] != strategy for row in rows):
        raise ValueError("校准不得混用检索策略")
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("阈值只能来自非空校准题集")
    valid = [row for row in rows if row.get("error") is None]
    scores = sorted({row["top_score"] for row in valid if row.get("top_score") is not None})
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("校准分数必须有限")
    positives = sum(row["answerable"] for row in valid)
    negatives = len(valid) - positives
    base = {"strategy": strategy, "minimum": None, "balanced_accuracy": None,
            "sample_count": len(rows), "valid_count": len(valid), "error_count": len(rows) - len(valid)}
    if not positives or not negatives or len(scores) < 2:
        return {**base, "reason": "缺少正负样本或分数恒定，依靠生成自评，不宣称阈值已有效校准"}
    candidates = [*scores, math.nextafter(scores[-1], math.inf)]
    choices = []
    for minimum in candidates:
        accepted = [row for row in valid if row.get("top_score") is not None and row["top_score"] >= minimum]
        true_positive = sum(row["answerable"] for row in accepted)
        false_positive = len(accepted) - true_positive
        balanced = (true_positive / positives + (negatives - false_positive) / negatives) / 2
        choices.append((balanced, -false_positive, minimum))
    balanced, negative_false_positive, minimum = max(choices)
    if balanced <= 0.5:
        return {**base, "balanced_accuracy": balanced, "reason": "观察分数无优于随机的区分力，依靠生成自评"}
    return {**base, "minimum": minimum, "balanced_accuracy": balanced,
            "false_accept_count": -negative_false_positive,
            "reason": "校准集平衡准确率最优；同分优先减少错误放行，再选择更严格分界"}


def validate_policy(policy: dict, metadata: dict, strategies: list[str],
                    calibration_case_count: int | None = None) -> dict[str, float | None]:
    """核验题集、语料、模型、Prompt、源码和预算绑定，禁止错用策略分数。

    参数：policy 校准JSON；metadata 当前运行版本；strategies 需比较的策略；
        calibration_case_count 本次数据集实际校准题数，None 表示不校验题数
        （题量由数据集决定，不再写死 24）。
    返回：dict，策略到门槛，null保持为不启用分数门禁。
    异常：ValueError，版本/数量/绑定缺失或不匹配、门槛非法。
    """
    if policy.get("version") != POLICY_VERSION:
        raise ValueError("校准策略版本不合法")
    if (calibration_case_count is not None
            and policy.get("calibration_case_count") != calibration_case_count):
        raise ValueError(f"校准策略的题数 {policy.get('calibration_case_count')} "
                         f"与本次数据集 {calibration_case_count} 不一致")
    saved = policy.get("metadata", {})
    if any(key not in metadata or key not in saved or metadata[key] != saved[key] for key in POLICY_BINDINGS):
        raise ValueError("校准策略的题集、语料、模型、Prompt、源码或预算已变化")
    values = {}
    for strategy in strategies:
        record = policy.get("thresholds", {}).get(strategy)
        if not isinstance(record, dict) or "minimum" not in record:
            raise ValueError("校准策略缺少本次策略门槛")
        minimum = record["minimum"]
        if minimum is not None and (type(minimum) not in (int, float) or not math.isfinite(minimum)):
            raise ValueError("校准门槛必须是有限数或null")
        values[strategy] = minimum
    return values
