"""验证生成前最高分证据闸、故障隔离与完整权威引用。"""

import math

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("scores,threshold,passed,max_score", [
    ([0.3], 0.3, True, 0.3), ([0.299], 0.3, False, 0.299),
    ([], 0.3, False, None), ([0.0], 0.0, True, 0.0), ([1.0], 1.0, True, 1.0),
    ([0.01] * 29 + [0.8], 0.3, True, 0.8),
])
def test_gate_uses_maximum_score_and_only_weak_success_creates_pool_draft(scores, threshold, passed, max_score):
    """误用首条、严格大于阈值或空命中放行时失败。参数：分数、阈值、预期通过及最高分；返回：None；异常：断言失败。"""
    from app.schemas.workflow import RetrievalLowConfidence
    from app.workflow.routing import assess_evidence

    decision = assess_evidence(scores, threshold)
    assert decision.passed is passed
    assert decision.max_score == max_score and decision.threshold == threshold
    if passed:
        assert decision.low_confidence is None
    else:
        assert isinstance(decision.low_confidence, RetrievalLowConfidence)
        assert decision.low_confidence.source == "retrieval_low_conf"
        assert decision.low_confidence.reason


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf, -0.001, 1.001, "0.3", True, None,
                                      TimeoutError("timeout"), RuntimeError("database failure")])
def test_bad_scores_and_dependency_errors_cannot_become_weak_evidence(bad_score):
    """坏数据或依赖失败被吞成低分入池草案时失败。参数：bad_score 非法值；返回：None；异常：断言失败。"""
    from app.workflow.routing import assess_evidence

    # 有一条合格分也不能掩盖另一条坏数据。
    with pytest.raises(ValueError):
        assess_evidence([0.9, bad_score], 0.3)


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -0.001, 1.001, "0.3", True, None])
def test_invalid_threshold_is_a_fault_even_with_no_results(threshold):
    """坏阈值借空结果绕过校验并误入低证据池时失败。参数：threshold 非法门槛；返回：None；异常：断言失败。"""
    from app.workflow.routing import assess_evidence

    with pytest.raises(ValueError):
        assess_evidence([], threshold)


def test_gate_models_reject_extra_fields_coercion_and_inconsistent_pool_drafts():
    """闸快照接受矛盾通过状态或非本入口草案时失败。参数：无；返回：None；异常：断言失败。"""
    from app.schemas.workflow import GateDecision, RetrievalLowConfidence

    good = {"passed": True, "max_score": 0.8, "threshold": 0.3, "low_confidence": None}
    for change in ({"passed": 1}, {"threshold": "0.3"}, {"max_score": math.inf},
                   {"passed": False}, {"max_score": None}, {"unexpected": True},
                   {"low_confidence": {"source": "retrieval_low_conf", "reason": "低分"}}):
        with pytest.raises(ValidationError):
            GateDecision.model_validate({**good, **change})
    with pytest.raises(ValidationError):
        GateDecision(passed=False, max_score=0.1, threshold=0.3, low_confidence=None)
    for change in ({"source": "self_check"}, {"reason": 1}, {"reason": ""}, {"question": "伪造原话"}):
        with pytest.raises(ValidationError):
            RetrievalLowConfidence.model_validate({"reason": "无命中", **change})


def _ranked_chunk(number, answer):
    """构造完整权威重排记录。参数：number 正整数主键、answer 完整正文；返回：RankedChunk；异常：模型校验失败。"""
    from app.knowledge.rerank import RankedChunk
    from app.knowledge.schemas import ChunkSnapshot, KnowledgeText

    return RankedChunk(score=0.1 if number == 1 else 0.8, chunk=ChunkSnapshot(
        id=number, source_key="doc:refund-policy.md", ordinal=number - 1,
        section_path=f"退款条款/第{number}条", content_type="policy", is_key_clause=True,
        text=KnowledgeText(category="退款退货", questions=f"退款条件{number}？", answer=answer),
        content_hash="a" * 64, embedding_revision="fixture", vector_id=str(number),
        vectorize_status="done", is_active=True,
    ))


def test_citations_keep_all_thirty_original_chunks_and_stable_numbers_after_edge_order():
    """退款证据被截成10块、正文裁短或显示时重新编号时失败。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import edge_order
    from app.workflow.routing import citations_from_ranked

    answers = [f"  第{number}条：" + "完整条款。" * 2400 + "不得拆封。\n " for number in range(1, 31)]
    ranked = [_ranked_chunk(number, answers[number - 1]) for number in range(1, 31)]
    citations = citations_from_ranked(ranked)
    assert len(citations) == 30
    assert [item.n for item in citations] == list(range(1, 31))
    assert [item.chunk_id for item in citations] == list(range(1, 31))
    assert [item.answer for item in citations] == answers
    assert citations[0].question == "退款条件1？"
    assert citations[0].section_path == "退款条款/第1条"
    assert citations[0].source_key == "doc:refund-policy.md"
    assert citations[0].source_revision is None
    displayed = edge_order(citations)
    assert [item.n for item in displayed[:4]] == [1, 3, 5, 7]
    assert [item.n for item in displayed[-4:]] == [8, 6, 4, 2]
    assert displayed[-1].answer == answers[1]
    assert citations_from_ranked([]) == []
