"""纯评估指标：门禁前召回、明确分母和不把拒答算满分的忠实度聚合。"""

from __future__ import annotations

from collections import Counter


def ratio(numerator: float, denominator: int) -> dict:
    """保存比率和可审计分母。参数：numerator 总和、denominator 数量；返回：dict；异常：无。"""
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator}


def retrieval_metrics(ranked_ids: list[int], evidence_groups: list[set[int]],
                      ks: tuple[int, ...] = (1, 3, 5, 10, 50)) -> dict:
    """按去重排名计算事实组覆盖，组内或、所有组命中才算完整证据。

    参数：ranked_ids 门禁前排名；evidence_groups 每个必需事实的等价证据集合；ks 正整数截断位。
    返回：dict，recall 为命中组比例，all_evidence 为所有组命中，MRR 为首个相关证据倒数；无答案均为空。
    异常：ValueError，空证据组或无效截断位。
    """
    if any(not group for group in evidence_groups) or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("证据组不得为空，排名截断必须为正整数")
    ranking = list(dict.fromkeys(ranked_ids))
    output = {}
    for k in ks:
        hits = sum(bool(set(ranking[:k]) & group) for group in evidence_groups)
        output[f"recall@{k}"] = hits / len(evidence_groups) if evidence_groups else None
        output[f"all_evidence@{k}"] = int(hits == len(evidence_groups)) if evidence_groups else None
    relevant = set().union(*evidence_groups) if evidence_groups else set()
    output["mrr@10"] = (next((1 / rank for rank, identifier in enumerate(ranking[:10], 1)
                                    if identifier in relevant), 0.0) if evidence_groups else None)
    return output


def aggregate_results(rows: list[dict]) -> dict:
    """汇总完整样本且将系统失败与拒答分开计数。

    参数：rows 含 answerable/refused/error 和可选 retrieval/faithfulness 的逐题记录。
    返回：dict，所有比率带分子分母，检索失败仍留在可答题分母且按零覆盖计；另报成功检索数。
    异常：无主动异常，错误记录结构由调用者保证。
    """
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    answered = [row for row in rows if row.get("refused") is False]
    faithful = [row["faithfulness"] for row in rows if row.get("faithfulness") is not None]
    total_facts = sum(item.get("total_count", 0) for item in rows)
    supported = sum(item.get("supported_count", 0) for item in rows)
    result = {
        "sample_count": len(rows), "answerable_count": len(answerable), "unanswerable_count": len(unanswerable),
        "error_count": sum(row.get("error") is not None for row in rows),
        "error_stages": dict(Counter(row["error"]["stage"] for row in rows if row.get("error"))),
        "coverage": ratio(sum(row.get("citation_valid") is True for row in answered), len(rows)),
        "false_refusal_rate": ratio(sum(row.get("refused") is True for row in answerable), len(answerable)),
        "unanswerable_correct_refusal": ratio(sum(row.get("refused") is True for row in unanswerable), len(unanswerable)),
        "citation_validity": ratio(sum(row.get("citation_valid") is True for row in answered), len(answered)),
        "faithfulness_macro": ratio(sum(faithful), len(faithful)),
        "faithfulness_micro": ratio(supported, total_facts),
        "retrieval_success_count": sum(row.get("retrieval") is not None for row in rows),
    }
    for name in [*(f"recall@{k}" for k in (1, 3, 5, 10, 50)),
                 *(f"all_evidence@{k}" for k in (1, 3, 5, 10, 50)), "mrr@10"]:
        result[name] = ratio(sum((row.get("retrieval") or {}).get(name) or 0 for row in answerable), len(answerable))
    return result
