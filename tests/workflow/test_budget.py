"""验证每轮保守预算、真实用量未知值与最终正文预留的可观察边界。"""

import pytest


def test_reserve_retains_retry_allowance_and_missing_usage():
    """成功后退还重试占额或把未知写零时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger

    ledger = BudgetLedger(total_token_budget=50000)
    ledger.reserve("answer", 16000, 4096)
    ledger.record_usage("answer", None, "stop")
    assert ledger.snapshot()["reserved_units"] == 40192
    assert ledger.snapshot()["actual_tokens"] is None


def test_exhausted_budget_never_counts_a_rejected_request():
    """拒绝请求后仍改账或超额发起请求时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded

    ledger = BudgetLedger(total_token_budget=40000)
    with pytest.raises(WorkflowBudgetExceeded):
        ledger.reserve("answer", 16000, 4096)
    assert ledger.snapshot()["model_calls"] == 0
    assert ledger.snapshot()["actual_tokens"] is None


def test_usage_and_snapshot_round_trip_do_not_share_mutable_state():
    """跨节点恢复丢账、缺用量写零或共享可变记录时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger

    ledger = BudgetLedger()
    ledger.reserve("classify", 100, 128)
    ledger.record_usage("classify", {"input_tokens": 80, "output_tokens": 10, "total_tokens": 90}, "stop")
    snapshot = ledger.snapshot()
    assert snapshot["actual_tokens"] == 90
    recovered = BudgetLedger.from_snapshot(snapshot)
    recovered.reserve("rewrite", 100, 2048)
    recovered.record_usage("rewrite", None, "stop")
    assert recovered.snapshot()["actual_tokens"] is None
    assert ledger.snapshot()["model_calls"] == 1
    assert snapshot["records"][0]["finish_reason"] == "stop"


def test_decision_headroom_is_checked_without_charging_future_answers():
    """每步重复扣未来正文或忘记补救调用额度时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded

    ledger = BudgetLedger(max_model_calls=5, total_token_budget=40000)
    for _ in range(3):
        ledger.ensure_capacity([("decide", 100, 2048), ("answer", 100, 4096), ("answer", 150, 4096)])
        ledger.reserve("decide", 100, 2048)
    assert ledger.snapshot()["reserved_units"] == 12888
    with pytest.raises(WorkflowBudgetExceeded):
        ledger.ensure_capacity([("decide", 100, 2048), ("answer", 100, 4096), ("answer", 150, 4096)])
    ledger.reserve("answer", 100, 4096)
    ledger.reserve("answer", 150, 4096)
    assert ledger.snapshot()["model_calls"] == 5


def test_fact_growth_stops_before_decision_and_keeps_existing_ledger():
    """工具结果变大后仍在无答案预算下决策时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded

    ledger = BudgetLedger(total_token_budget=30000)
    ledger.reserve("decide", 100, 2048)
    with pytest.raises(WorkflowBudgetExceeded):
        ledger.ensure_capacity([("decide", 4000, 2048), ("answer", 4000, 4096), ("answer", 4050, 4096)])
    assert ledger.snapshot()["model_calls"] == 1


@pytest.mark.parametrize("stage,limit", [("understand",2048),("classify",4096),("expand",2048),
    ("rewrite",2048),("decide",2048),("answer",4096)])
def test_request_and_output_limits_are_enforced(stage, limit):
    """超完整输入或单次输出上限未停止时失败。参数：stage 阶段、limit 输出上限；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded

    ledger = BudgetLedger()
    with pytest.raises(WorkflowBudgetExceeded):
        ledger.reserve(stage, 16001, limit)
    with pytest.raises(WorkflowBudgetExceeded):
        ledger.reserve(stage, 100, limit + 1)
    assert ledger.snapshot()["model_calls"] == 0


def test_answer_attempts_and_decision_limits_survive_restore():
    """多次调用适配器绕过单轮正文/决策上限时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded

    ledger = BudgetLedger()
    for _ in range(5):
        ledger.reserve("decide", 20, 2048)
    with pytest.raises(WorkflowBudgetExceeded):
        ledger.reserve("decide", 20, 2048)
    for _ in range(2):
        ledger.reserve("answer", 20, 4096)
    with pytest.raises(WorkflowBudgetExceeded):
        BudgetLedger.from_snapshot(ledger.snapshot()).reserve("answer", 20, 4096)


@pytest.mark.parametrize("stage,limit", [("understand", 2048), ("classify", 4096), ("expand", 2048)])
def test_understanding_classification_expansion_once_after_restore(stage, limit):
    """恢复账本后重复优化/分类调用绕过单轮上限时失败。参数：阶段和输出上限；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded

    ledger = BudgetLedger()
    ledger.reserve(stage, 100, limit)
    ledger.record_usage(stage, None, "tool_calls")
    ledger.record_failure(stage, {"type": "protocol_failure"})
    snapshot = ledger.snapshot()
    restored = BudgetLedger.from_snapshot(snapshot)
    with pytest.raises(WorkflowBudgetExceeded):
        restored.reserve(stage, 100, limit)
    assert restored.snapshot() == snapshot
