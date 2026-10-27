"""CH07窗口预算行为测试：覆盖固定扣减、峰值、中文计量与非法配置，避免用条数假装token。"""

import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.config import Settings


def _api():
    """延迟载入待实现预算模块。参数：无；返回：模块；异常：功能缺失时ModuleNotFoundError。"""
    return importlib.import_module("app.core.context_budget")


def _calculate(**changes):
    """提供独立手算基准。参数：changes覆盖预算项；返回：预算；异常：被测预算错误。"""
    values = dict(window=18000, fixed_tokens=1000, evidence_tokens=1000,
                  summary_tokens=500, output_tokens=2000, safety_tokens=500,
                  current_peak_tokens=5000, target_turns=20, steady_turn_tokens=1000)
    return _api().derive_history_budget(**(values | changes))


def test_window_deducts_all_fixed_costs_and_current_peak():
    """漏扣任何开销均会把可用历史算大。参数：无；返回：None；异常：断言失败。"""
    budget = _calculate()
    assert (budget.history, budget.layer1, budget.layer2) == (8000, 5600, 2400)
    assert budget.available_history == 8000
    assert budget.request_limit == 15500


def test_desired_turns_cap_history_even_when_window_has_room():
    """窗口富余不能无视期望历史上限。参数：无；返回：None；异常：断言失败。"""
    budget = _calculate(target_turns=2)
    assert (budget.history, budget.layer1, budget.layer2) == (2000, 1400, 600)


def test_rounding_never_creates_or_loses_the_last_token():
    """两层合计必须等于预算，不能浮点向上超限。参数：无；返回：None；异常：断言失败。"""
    budget = _calculate(window=18001)
    assert (budget.history, budget.layer1, budget.layer2) == (8001, 5600, 2401)


@pytest.mark.parametrize("window", [10000, 11000])
def test_too_small_window_reports_inability_to_keep_one_turn(window):
    """原文层连一稳态轮都放不下时不能返回可用假预算。参数：window窗口；返回：None；异常：断言失败。"""
    with pytest.raises(_api().ContextBudgetExceeded, match="上下文预算不足"):
        _calculate(window=window)


@pytest.mark.parametrize("field", ["window", "fixed_tokens", "evidence_tokens", "summary_tokens",
    "output_tokens", "safety_tokens", "current_peak_tokens", "target_turns", "steady_turn_tokens"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_budget_components_are_not_coerced(field, value):
    """负数、布尔和浮点不能被当成合法整数预算。参数：field字段、value坏值；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        _calculate(**{field: value})


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan"), True])
def test_invalid_chinese_factor_is_rejected(factor):
    """非法系数不能把中文计量缩成零或无限。参数：factor系数；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        _api().TokenEstimator(non_ascii_tokens=factor)


def test_counting_uses_unicode_without_json_ascii_escape():
    """中文不应先转六字节转义再用ASCII价格计算。参数：无；返回：None；异常：断言失败。"""
    counter = _api().TokenEstimator()
    assert counter.value_tokens("中文") == 5  # 两个汉字4单位，两个JSON引号0.5，向上取整。
    assert counter.value_tokens("abcd") == 2
    assert counter.message_tokens([HumanMessage(content="猫")]) == 10


def test_changed_chinese_factor_changes_the_same_budget_calculation():
    """计量口径变更必须传到固定开销扣减。参数：无；返回：None；异常：断言失败。"""
    message = [HumanMessage(content="猫" * 100)]
    conservative = _api().TokenEstimator(non_ascii_tokens=2).message_tokens(message)
    calibrated = _api().TokenEstimator(non_ascii_tokens=1).message_tokens(message)
    assert (conservative, calibrated) == (208, 108)
    assert _calculate(fixed_tokens=conservative).history == 8792
    assert _calculate(fixed_tokens=calibrated).history == 8892


def test_message_counter_includes_calls_names_and_tool_ids_but_not_checkpoint_metadata():
    """漏算申请/结果标识会低估峰值，状态元数据则不能重复收费。参数：无；返回：None；异常：断言失败。"""
    counter = _api().TokenEstimator()
    ordinary = AIMessage(content="")
    call = AIMessage(content="", tool_calls=[{"name": "query_product", "args": {"keyword": "饮水机" * 20}, "id": "call-1"}])
    assert counter.message_tokens([call]) > counter.message_tokens([ordinary]) + 100
    assert counter.message_tokens([ToolMessage(content="结果", tool_call_id="x" * 64)]) > counter.message_tokens([ToolMessage(content="结果", tool_call_id="x")])
    named = HumanMessage(content="猫", name="n" * 40)
    assert counter.message_tokens([named]) > 20
    source = HumanMessage(content="猫", id="persistent-id", additional_kwargs={"db_message_id": 987})
    assert counter.message_tokens([source]) == 10


def test_demo_env_changes_effective_limits_and_keeps_multi_call_peak(monkeypatch):
    """三步仍可有四个工具结果，峰值不能按步数少算一个。参数：monkeypatch环境夹具；返回：None；异常：断言失败。"""
    for key, value in {"MODEL_CONTEXT_WINDOW": "18000", "MAX_OUTPUT_TOKENS": "2000",
        "MAX_USER_INPUT_TOKENS": "2000", "MAX_AGENT_STEPS": "3",
        "TOOL_RESULT_MAX_TOKENS": "1200", "RERANK_TOP_K": "5"}.items():
        monkeypatch.setenv(key, value)
    settings = Settings(_env_file=None, chat_base_url="http://test.invalid/v1", chat_model="test-model", chat_api_key="test-key")
    assert settings.workflow_max_decisions == 3
    budget = _api().context_budget(settings, fixed_tokens=2634)
    assert budget.current_peak_tokens == 8336  # 2000 + 4 * (1200 + 384)，不误取3个工具。
    assert (budget.evidence_tokens, budget.summary_tokens, budget.safety_tokens) == (1125, 750, 900)
    assert (budget.history, budget.layer1, budget.layer2) == (2255, 1578, 677)


@pytest.mark.parametrize("changes", [{"model_context_window": 0}, {"max_output_tokens": -1},
    {"max_user_input_tokens": 0}, {"tool_result_max_tokens": True}, {"context_non_ascii_tokens": float("nan")},
    {"context_non_ascii_tokens": True}])
def test_new_limit_configuration_validates_before_use(changes):
    """配置入口不能接受会让预算失真的值。参数：changes坏配置；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chat_base_url="http://test.invalid/v1", chat_model="test-model", chat_api_key="test-key", **changes)
