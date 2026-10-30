"""正式证据策略边界：全候选、缺失分差、稳定身份及历史兼容。"""

import importlib

import pytest

from app.knowledge.rerank import RankedChunk
from app.knowledge.schemas import ChunkSnapshot, KnowledgeText
from app.schemas.workflow import GateDecision
from app.workflow.routing import citations_from_ranked


def confidence():
    """加载待实现算法。参数：无；返回：模块；异常：缺功能时测试失败。"""
    try:
        return importlib.import_module("app.knowledge.confidence")
    except ModuleNotFoundError:
        pytest.fail("正式证据策略尚未实现")


def ranked(identifier=1, score=0.8, answer="完整原文"):
    """构造独立权威候选。参数：主键/分数/正文；返回：RankedChunk；异常：字段校验。"""
    return RankedChunk(chunk=ChunkSnapshot(id=identifier, source_key="doc:test", ordinal=identifier,
        text=KnowledgeText(category="测试", questions="测试问题", answer=answer), content_type="faq",
        content_hash="a" * 64, embedding_revision="test"), score=score)


def policy(**updates):
    """构造测试校准制品。参数：字段覆盖；返回：策略；异常：参数非法。"""
    return confidence().EvidencePolicy(version="test-v2", feature_protocol="evidence-v2",
        minimum_score=0.5, count_cap=2, weights={"top1": 0.5, "count": 0.25, "gap": 0.25},
        threshold=0.5, protocols={"query": "test", "rerank": "test", "budget": "test"},
        training_source={"run": "test"}, **updates)


def test_single_candidate_has_no_fabricated_gap():
    """缺第二条不能造差值。参数：无；返回：None；异常：断言失败。"""
    rows = [ranked()]
    features = confidence().extract_features(rows, citations_from_ranked(rows), minimum_score=0.5)
    assert features.top2_score is None and features.score_gap is None
    assert confidence().assess_evidence(rows, citations_from_ranked(rows), policy()).evidence_confidence == 0.525


def test_gate_uses_all_ranked_candidates_not_top5_snapshot():
    """第六条最高分必须进入全候选统计，次序无影响。参数：无；返回：None；异常：断言失败。"""
    rows = [ranked(i, 0.1) for i in range(1, 6)] + [ranked(6, 0.9)]
    selected = citations_from_ranked(rows)
    result = confidence().extract_features(rows, selected, minimum_score=0.5)
    assert (result.top1_score, result.top2_score, result.candidate_count, result.effective_count) == (0.9, 0.1, 6, 1)
    assert result == confidence().extract_features(list(reversed(rows)), selected, minimum_score=0.5)


def test_duplicate_identity_uses_highest_score_and_rejects_conflict():
    """重复块只计一次、实际最高分保留、原文冲突报故障。参数：无；返回：None；异常：断言失败。"""
    rows = [ranked(score=0.1), ranked(score=0.9)]
    result = confidence().extract_features(rows, citations_from_ranked(rows[:1]), minimum_score=0.5)
    assert (result.candidate_count, result.effective_count, result.top1_score) == (1, 1, 0.9)
    with pytest.raises(ValueError, match="原文|身份"):
        confidence().extract_features([*rows, ranked(answer="冲突")], [], minimum_score=0.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_feature_threshold_is_rejected(value):
    """坏参数不能降级为低置信度。参数：非法值；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        confidence().extract_features([], [], minimum_score=value)


def test_v1_checkpoint_reads_but_new_search_requires_v2_policy(tmp_path):
    """旧检查点可解释，新请求不能用旧制品。参数：临时目录；返回：None；异常：断言失败。"""
    old = GateDecision.model_validate({"passed": True, "max_score": 0.8, "threshold": 0.3})
    assert getattr(old, "schema_version", None) == 1
    path = tmp_path / "old.json"
    path.write_text('{"threshold":0.3}', encoding="utf-8")
    with pytest.raises(ValueError):
        confidence().load_evidence_policy(path, protocols={"query": "test"})


def test_budget_rejects_all_evidence_without_pool_signal():
    """非空候选全被预算排除必须抛预算错误。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import ContextBudgetExceeded
    with pytest.raises(ContextBudgetExceeded):
        confidence().assess_evidence([ranked()], [], policy())


@pytest.mark.parametrize("update", [{"weights": {"top1": float("nan"), "count": 0.0, "gap": 0.0}},
    {"weights": {"top1": 1.25, "count": -0.25, "gap": 0.0}}, {"count_cap": 0}, {"threshold": float("nan")}])
def test_corrupt_policy_cannot_be_used_as_gate(update):
    """坏权重/上限/门槛必须拒绝，不能生成低分。参数：非法覆盖；返回：None；异常：断言失败。"""
    payload = policy().model_dump()
    payload.update(update)
    with pytest.raises(ValueError):
        confidence().EvidencePolicy.model_validate(payload)


def test_invalid_raw_ranked_score_is_rejected():
    """外部伪造布尔分数不能被数值转换掩盖。参数：无；返回：None；异常：断言失败。"""
    bad = ranked().model_copy(update={"score": True})
    with pytest.raises(ValueError):
        confidence().extract_features([bad], [], minimum_score=0.5)
