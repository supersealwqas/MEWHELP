"""评估计分与阈值选择回归，防止漏分母、重复命中和固定集泄漏。"""

import pytest


def test_recall_uses_distinct_evidence_groups_and_rank() -> None:
    """核对排名、组内或与组间覆盖。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.metrics import retrieval_metrics

    result = retrieval_metrics([9, 9, 2, 3], [{2, 4}, {8}])
    assert result["recall@1"] == 0
    assert result["recall@3"] == 0.5
    assert result["mrr@10"] == 0.5
    assert result["all_evidence@3"] == 0
    assert retrieval_metrics([2, 8], [{2, 4}, {8}])["all_evidence@3"] == 1
    assert retrieval_metrics([1], [])["recall@50"] is None


def test_failures_and_refusals_preserve_denominators() -> None:
    """全拒答忠实度为空且失败计入总数。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.metrics import aggregate_results

    rows = [
        {"answerable": True, "refused": True, "error": None, "faithfulness": None},
        {"answerable": False, "refused": True, "error": None, "faithfulness": None},
        {"answerable": True, "refused": None, "error": {"stage": "retrieval"}, "faithfulness": None},
    ]
    result = aggregate_results(rows)
    assert result["sample_count"] == 3
    assert result["error_count"] == 1
    assert result["coverage"] == {"value": 0, "numerator": 0, "denominator": 3}
    assert result["false_refusal_rate"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert result["unanswerable_correct_refusal"]["value"] == 1
    assert result["faithfulness_macro"]["value"] is None


def test_calibration_separates_score_spaces_and_rejects_evaluation_rows() -> None:
    """阈值只读校准集且恒定分数明确不设门槛。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.policy import calibrate_threshold

    rows = [
        {"split": "calibration", "strategy": "bm25", "answerable": True, "top_score": 12.0, "error": None},
        {"split": "calibration", "strategy": "bm25", "answerable": False, "top_score": 3.0, "error": None},
    ]
    policy = calibrate_threshold(rows, strategy="bm25")
    assert 3 < policy["minimum"] <= 12
    assert policy["balanced_accuracy"] == 1
    assert calibrate_threshold([{**row, "top_score": 1.0} for row in rows], strategy="bm25")["minimum"] is None
    with pytest.raises(ValueError, match="校准"):
        calibrate_threshold([{**rows[0], "split": "evaluation"}], strategy="bm25")
    with pytest.raises(ValueError, match="策略"):
        calibrate_threshold(rows, strategy="dense")


def test_corpus_digest_ignores_ids_but_tracks_metadata() -> None:
    """相同语料跨库摘要相同、筛选元数据变化不同。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.policy import corpus_digest
    from tests.rag.test_retrieval import _chunk

    original = _chunk(1)
    moved = original.model_copy(update={"id": 500, "vector_id": "500", "next_chunk_id": 501})
    assert corpus_digest([original]) == corpus_digest([moved])
    assert corpus_digest([original]) != corpus_digest([moved.model_copy(update={"content_type": "faq"})])
