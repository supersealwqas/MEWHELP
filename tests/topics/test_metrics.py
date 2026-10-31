"""验证多标签手算指标、验证集阈值选优及不确定解码边界。"""

import math

import pytest

from app.topics.taxonomy import load_taxonomy


def test_multilabel_metrics_count_extra_and_missing_labels():
    """捕获把多标签多打漏打判为完全匹配的错误。参数：无；返回：无；异常：断言失败。"""
    from app.topics.metrics import compute_metrics
    report = compute_metrics([["退换货", "物流"], ["物流"], ["退换货"], ["尺码"]],
                             [["退换货"], ["物流", "尺码"], [], ["尺码"]], load_taxonomy())
    assert report["per_label"]["退换货"]["confusion_matrix"] == [[2, 0], [1, 1]]
    assert report["per_label"]["物流"]["confusion_matrix"] == [[2, 0], [1, 1]]
    assert report["per_label"]["尺码"]["confusion_matrix"] == [[2, 1], [0, 1]]
    assert report["per_label"]["尺码"]["support"] == 1
    assert report["per_label"]["尺码"]["precision"] == .5
    assert report["micro_f1"] == pytest.approx(2 / 3)
    assert report["macro_f1"] == pytest.approx(2 / 3)
    assert report["exact_match"] == .25
    assert report["per_label"]["发票"]["f1"] is None
    assert "发票" in report["unsupported"]


def test_no_supported_labels_do_not_claim_perfect_score():
    """捕获无支持类别被冒充满分的错误。参数：无；返回：无；异常：断言失败。"""
    from app.topics.metrics import compute_metrics
    report = compute_metrics([[]], [[]], load_taxonomy())
    assert report["macro_f1"] is None and report["micro_f1"] == 0
    assert report["exact_match"] == 1 and len(report["unsupported"]) == 17


def test_threshold_uses_only_given_validation_and_stable_tiebreak():
    """捕获阈值取错集合、范围或平局不稳定的错误。参数：无；返回：无；异常：断言失败。"""
    from app.topics.metrics import select_threshold
    taxonomy = load_taxonomy()
    probabilities = [[.7] + [.02] * 16, [.4] + [.02] * 16]
    targets = [[1.] + [0.] * 16, [0.] * 17]
    assert select_threshold(probabilities, targets, taxonomy) == .5
    assert select_threshold([[.2] + [.02] * 16], [[1.] + [0.] * 16], taxonomy) == .2


@pytest.mark.parametrize("probabilities,threshold,want", [
    ([.6] + [.1] * 15 + [.9], .5, ["退换货"]),
    ([.1] * 16 + [.9], .5, ["其他"]),
    ([.1] * 17, .5, []),
])
def test_decode_other_is_exclusive_and_uncertain_stays_empty(probabilities, threshold, want):
    """捕获其他混用和强制最大概率标签的错误。参数：概率、阈值、预期；返回：无；异常：断言失败。"""
    from app.topics.metrics import decode_labels
    assert decode_labels(probabilities, threshold, load_taxonomy()) == want


@pytest.mark.parametrize("probabilities", [[.1] * 16, [.1] * 18, [math.nan] * 17, [1.1] * 17])
def test_decode_rejects_wrong_dimensions_and_nonfinite_probability(probabilities):
    """捕获模型输出维数与非有限概率被默许的错误。参数：非法概率；返回：无；异常：断言失败。"""
    from app.topics.metrics import decode_labels
    with pytest.raises(ValueError):
        decode_labels(probabilities, .5, load_taxonomy())


def test_metrics_reject_unknown_and_mixed_other_labels():
    """捕获错误标签进入统计分母的错误。参数：无；返回：无；异常：断言失败。"""
    from app.topics.metrics import compute_metrics, select_threshold
    for labels in (["未知"], ["其他", "尺码"], ["尺码", "尺码"]):
        with pytest.raises(ValueError):
            compute_metrics([["尺码"]], [labels], load_taxonomy())
    with pytest.raises(ValueError):
        compute_metrics([["尺码"]], [], load_taxonomy())
    with pytest.raises(ValueError):
        select_threshold([[.2] * 17], [[.5] * 17], load_taxonomy())
