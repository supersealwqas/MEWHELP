"""按实际selected充分性拟合有限证据策略，保存基线、误判与收益不足结论。"""

from bisect import bisect_left
from hashlib import sha256
import json
import math
import random

from app.knowledge.confidence import EvidencePolicy, EvidenceFeatures, FEATURE_PROTOCOL, evidence_policy_digest


def observed_thresholds(values: list[float]) -> list[float]:
    """只产生实际观测值及相邻中点。参数：有限分数；返回：升序候选；异常：ValueError坏数据。"""
    if any(type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("校准分数必须是0–1有限值")
    unique = sorted(set(values))
    return sorted(set(unique + [(left + right) / 2 for left, right in zip(unique, unique[1:])]))


def decision_metrics(rows: list[dict], predictions: list[bool]) -> dict:
    """统计真实充分性的两类错误，分母保留。参数：观测行与判定；返回：指标；异常：标签非法。"""
    if len(rows) != len(predictions) or any(type(row["sufficient"]) is not bool for row in rows):
        raise ValueError("校准行必须有严格实际充分性标签")
    positive = sum(row["sufficient"] for row in rows)
    negative = len(rows) - positive
    false_pass = [row["id"] for row, passed in zip(rows, predictions) if passed and not row["sufficient"]]
    false_block = [row["id"] for row, passed in zip(rows, predictions) if not passed and row["sufficient"]]
    return {"positive_count": positive, "negative_count": negative,
            "false_pass": false_pass, "false_block": false_block,
            "balanced_accuracy": ((positive - len(false_block)) / positive + (negative - len(false_pass)) / negative) / 2
            if positive and negative else None}


def _best_threshold(scores, rows):
    """同一参数形状选最优通过阈值，不重复逐条扫描。

    参数：综合分与实际标签行；返回：阈值、选择键；异常：无合法分数时ValueError。
    """
    ordered = sorted((score, row["sufficient"]) for score, row in zip(scores, rows) if row["features"]["candidate_count"] > 0)
    values, prefix = [], [0]
    for score, label in ordered:
        values.append(score)
        prefix.append(prefix[-1] + label)
    p = sum(row["sufficient"] for row in rows)
    n = len(rows) - p
    empty_negative = sum(not row["sufficient"] and row["features"]["candidate_count"] == 0 for row in rows)
    best = None
    for threshold in observed_thresholds(scores):
        cut = bisect_left(values, threshold)
        tp, tn = prefix[-1] - prefix[cut], cut - prefix[cut] + empty_negative
        # 平衡准确率使用整数共同分母比较；同分先少误放行，再较严格阈值。
        key = (tp * n + tn * p, -(n - tn), threshold)
        if best is None or key > best[1]:
            best = threshold, key
    return best


def _paired_stability(rows, predictions, baseline):
    """固定种子分层重采样检查观测改善的不确定性，不拿验收集调参。

    参数：校准行与两组冻结判定；返回：区间及说明；异常：缺正负样本时调用方须先停止。
    """
    groups = [[i for i, row in enumerate(rows) if row["sufficient"] is label] for label in (True, False)]
    rng, differences = random.Random(9), []
    for _ in range(2000):
        difference = 0.0
        for group in groups:
            sampled = [rng.choice(group) for _ in group]
            difference += sum((predictions[i] == rows[i]["sufficient"]) - (baseline[i] == rows[i]["sufficient"])
                              for i in sampled) / len(group) / 2
        differences.append(difference)
    differences.sort()
    return {"method": "paired-stratified-bootstrap:seed9:n2000", "lower_95": differences[49],
            "upper_95": differences[1949], "stable_improvement": differences[49] > 0,
            "limitation": "同一24题上选择与重采样存在乐观偏差；区间仅作保守发布阻挡，不能替代独立验收。"}


def calibrate_evidence_policy(rows: list[dict], metadata: dict) -> dict:
    """按有限网格选择正式候选并与最高分基线比较，收益不足时不给发布制品。

    参数：rows仅calibration实际ranked/selected与充分性标签；metadata本轮协议及来源摘要。
    返回：policy、candidate_policy、基线/误判/稳定性报告；不足时policy为None。
    异常：ValueError，混入验收题、坏分数、标签或候选协议错误。
    """
    if any(row.get("split") != "calibration" for row in rows):
        raise ValueError("仅校准题可参与策略拟合，禁止混入独立验收")
    valid = [row for row in rows if row.get("error") is None]
    base = {"row_count": len(rows), "usable_count": len(valid),
            "excluded": [{"id": row["id"], "error": row["error"]} for row in rows if row.get("error")],
            "metadata": metadata, "policy": None}
    if not valid or {row["sufficient"] for row in valid} != {True, False}:
        return {**base, "status": "insufficient_labels"}
    if any(type(row["sufficient"]) is not bool for row in valid):
        raise ValueError("实际充分性标签必须为严格布尔")
    for row in valid:
        EvidenceFeatures.model_validate(row["features"])
        observed_thresholds(row["selected_scores"])
    observed = [score for row in valid for score in row["selected_scores"]]
    minimums = observed_thresholds(observed)
    if not minimums:
        return {**base, "status": "insufficient_labels"}
    best, shapes, evaluated = None, {}, 0
    for minimum in minimums:
        counts = [sum(score >= minimum for score in row["selected_scores"]) for row in valid]
        for cap in sorted(set(count for count in counts if count > 0)):
            for top_units in range(1, 5):
                for count_units in range(5 - top_units):
                    gap_units = 4 - top_units - count_units
                    weights = {"top1": top_units / 4, "count": count_units / 4, "gap": gap_units / 4}
                    scores = [weights["top1"] * (row["features"]["top1_score"] or 0.0)
                              + weights["count"] * min(count / cap, 1.0)
                              + weights["gap"] * (row["features"]["score_gap"] or 0.0) for row, count in zip(valid, counts)]
                    threshold, key = _best_threshold(scores, valid)
                    # 参数越少越简单；其次小计数上限、低权重字典序、较严格有效分。
                    simplicity = (-sum(value > 0 for value in weights.values()), -cap,
                                  weights["top1"], -weights["count"], minimum)
                    candidate = {"minimum_score": minimum, "count_cap": cap, "weights": weights,
                                 "threshold": threshold, "scores": scores}
                    selection = (*key[:2], *simplicity, threshold)
                    evaluated += 1
                    shape = "+".join(name for name, value in weights.items() if value > 0)
                    if shape not in shapes or selection > shapes[shape][0]:
                        shapes[shape] = (selection, candidate)
                    if best is None or selection > best[0]:
                        best = selection, candidate
    _, chosen = best
    baseline_scores = [row["features"]["top1_score"] or 0.0 for row in valid]
    baseline_threshold, _ = _best_threshold(baseline_scores, valid)
    predictions = [row["features"]["candidate_count"] > 0 and score >= chosen["threshold"] for row, score in zip(valid, chosen["scores"])]
    baseline_predictions = [row["features"]["candidate_count"] > 0 and score >= baseline_threshold for row, score in zip(valid, baseline_scores)]
    parameters = {key: value for key, value in chosen.items() if key != "scores"}
    source = {key: value for key, value in metadata.items() if key != "protocols"}
    version = sha256(json.dumps({"parameters": parameters, "metadata": metadata}, sort_keys=True).encode()).hexdigest()
    policy = EvidencePolicy(version=version, feature_protocol=FEATURE_PROTOCOL, protocols=metadata["protocols"],
                            training_source=source, **parameters).model_dump(mode="json")
    stability = _paired_stability(valid, predictions, baseline_predictions)
    comparison = [{"shape": shape, **{key: value for key, value in item.items() if key != "scores"},
                   "metrics": decision_metrics(valid, [row["features"]["candidate_count"] > 0 and score >= item["threshold"]
                               for row, score in zip(valid, item["scores"])])} for shape, (_, item) in shapes.items()]
    supported = stability["stable_improvement"] and len(valid) == 24 and not base["excluded"]
    policy_digest = evidence_policy_digest(policy)
    assessed = [{"id": row["id"], "query": row["query"], "sufficient": row["sufficient"],
                 "features": {**row["features"], "effective_count": sum(score >= chosen["minimum_score"] for score in row["selected_scores"])},
                 "evidence_confidence": score, "candidate_passed": passed, "baseline_passed": baseline,
                 "policy_sha256": policy_digest}
                for row, score, passed, baseline in zip(valid, chosen["scores"], predictions, baseline_predictions)]
    return {**base, "status": "supported" if supported else "needs_human_decision",
            "policy": policy if supported else None, "candidate_policy": policy,
            "policy_sha256": policy_digest, "assessed_rows": assessed,
            "selection_rule": "balanced_accuracy > fewer_false_pass > fewer_nonzero_weights > smaller_cap > fixed_lexical_tie",
            "parameter_shapes_evaluated": evaluated, "comparisons": comparison,
            "chosen": decision_metrics(valid, predictions),
            "baseline": {"threshold": baseline_threshold, **decision_metrics(valid, baseline_predictions)},
            "stability": stability,
            "disagreements": [{"id": row["id"], "query": row["query"], "sufficient": row["sufficient"],
                               "candidate_passed": passed, "baseline_passed": baseline}
                              for row, passed, baseline in zip(valid, predictions, baseline_predictions) if passed != baseline]}
