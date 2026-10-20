"""真实验收工具的阈值与事实校验逻辑测试。"""

from __future__ import annotations

import pytest

from app.knowledge.schemas import VectorHit
from tests.verify_ch03_live import ScoredQuery, choose_threshold, validate_answer


def _hit(identifier: int, score: float) -> VectorHit:
    """构造命中。参数：主键/分数；返回：命中；异常：校验失败。"""
    return VectorHit(
        id=identifier,
        score=score,
        content_hash="a" * 64,
        embedding_revision="rev",
    )


def test_choose_threshold_retains_positive_and_rejects_negative() -> None:
    """正例保留且负例低于阈值。参数：无；返回：None；异常：无。"""
    scores = [
        ScoredQuery("positive", [1], False, [_hit(1, 0.91), _hit(2, 0.3)]),
        ScoredQuery("negative", [], True, [_hit(9, 0.4)]),
    ]
    assert choose_threshold(scores) == 0.91


def test_choose_threshold_rejects_overlap() -> None:
    """正负例分数重叠时不能生成策略。参数：无；返回：None；异常：无。"""
    scores = [
        ScoredQuery("positive", [1], False, [_hit(1, 0.5)]),
        ScoredQuery("negative", [], True, [_hit(9, 0.7)]),
    ]
    with pytest.raises(ValueError):
        choose_threshold(scores)


def test_choose_threshold_requires_positive_recall() -> None:
    """正例未进入 Top3 时停止校准。参数：无；返回：None；异常：无。"""
    scores = [ScoredQuery("positive", [1], False, [_hit(2, 0.9)])]
    with pytest.raises(ValueError):
        choose_threshold(scores)


def test_validate_answer_rejects_missing_or_forbidden_fact() -> None:
    """缺事实或含禁止结论均失败。参数：无；返回：None；异常：无。"""
    case = {
        "must_include": ["6 元"],
        "must_not_include": ["10 元"],
    }
    assert validate_answer(case, {"answer": "未满99元收6元。", "hits": [{"id": 1}]}) == []
    assert validate_answer(case, {"answer": "未满99元收10元。", "hits": [{"id": 1}]})


def test_validate_empty_case_requires_no_hits() -> None:
    """负例必须空命中。参数：无；返回：None；异常：无。"""
    case = {"expect_empty": True, "must_not_include": ["6 元"]}
    assert validate_answer(case, {"answer": "没有查到相关政策。", "hits": []}) == []
    assert validate_answer(case, {"answer": "没有查到。", "hits": [{"id": 1}]})


def test_validate_accepts_safe_paraphrase_and_markdown() -> None:
    """同义事实与 Markdown 强调不应被误判。参数：无；返回：None；异常：无。"""
    case = {
        "must_include": ["不承诺"],
        "must_include_any": [["到账时效", "到账时间"]],
    }
    observed = {
        "answer": "内部垫芯**不可机洗**；平台不承诺具体的到账时间。",
        "hits": [{"id": 1}],
    }
    assert validate_answer(case, observed) == []
