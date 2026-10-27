"""验证 CH06 理解失败透传、可信来源校验及失败后的原始账本。"""

import asyncio
import json
from copy import deepcopy

import pytest

from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
from app.workflow.model import WorkflowModelAdapter
from tests.workflow.test_model import call_response, sdk_runtime, text_response, understanding_response


def source_messages(history, raw):
    """显式提供原问和可信源，避免用尾随资料猜原问。参数：history历史、raw原问；返回：模型输入；异常：无。"""
    from app.workflow.context import ModelContext
    return ModelContext([*history, {"role":"user", "content":raw}], raw_query=raw,
        current_message_id="current", source_turns=[{"messages":history}], summary_sources=[], details={})


@pytest.mark.parametrize("response", [
    text_response('{"resolved_query":"坏正文"}'),
    call_response("submit_understanding", '{"resolved_query":'),
    call_response("query_order", '{"order_id":"1001"}'),
    text_response("被截断", "length"),
    RuntimeError("上游不可用"),
], ids=["text_json", "broken_arguments", "wrong_tool", "length", "network"])
async def test_understanding_failure_preserves_original_and_budget(sdk_runtime, response):
    """理解优化失败若改原问、吞账本或重复调用则失败。参数：SDK边界、失败响应；返回：None；异常：断言失败。"""
    raw = "  如果我是会员，实付39块9，订单1001包邮吗？\n"
    messages = [{"role": "user", "content": raw}]
    original = deepcopy(messages)
    sdk_runtime.responses.append(response)
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(messages, ledger)
    assert result.model_dump() == {"resolved_query": raw, "standard_query": raw,
                                   "needs_resolution": False, "order_id": None}
    assert messages == original and len(sdk_runtime.requests) == 1
    snapshot = ledger.snapshot()
    assert snapshot["model_calls"] == 1 and snapshot["reserved_units"] > 0
    assert snapshot["records"][0]["failure"]["query_fallback"] == "original_query"
    assert snapshot["records"][0]["stage"] == "understand"


@pytest.mark.parametrize("size", [2000, 2001, 4000])
async def test_understanding_fallback_preserves_long_original_for_input_guard(sdk_runtime, size):
    """回退不能裁短长原问以绕过后续检索上限。参数：SDK边界、size长度；返回：None；异常：断言失败。"""
    from app.workflow.routing import knowledge_input_stop_reason

    raw = "问" * size
    sdk_runtime.responses.append(text_response("坏输出"))
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
        [{"role": "user", "content": raw}], ledger)
    assert result.resolved_query == result.standard_query == raw
    assert knowledge_input_stop_reason(result.resolved_query) == ("input_limit" if size > 2000 else None)
    assert ledger.snapshot()["model_calls"] == 1


@pytest.mark.parametrize("raw,history,derived", [
    ("那能退吗", [], "订单1001能退吗"),
    ("订单1002能退吗", [{"role": "user", "content": "订单1001"}], "订单1001能退吗"),
    ("那能退吗", [{"role": "assistant", "content": "订单1001金额99元"}], "订单1001满99元能退吗"),
    ("满99元包邮吗", [], "满199元包邮吗"),
], ids=["invented_order", "changed_order", "assistant_only_facts", "changed_amount"])
async def test_invalid_understanding_source_falls_back_once(sdk_runtime, raw, history, derived):
    """编造编号或助手散文被当事实来源时失败。参数：SDK脚本与问法；返回：None；异常：断言失败。"""
    sdk_runtime.responses.append(understanding_response(derived, needs_resolution=True, usage=True))
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
        source_messages(history, raw), ledger)
    assert result.resolved_query == result.standard_query == raw
    assert result.order_id is None and not result.needs_resolution
    assert ledger.snapshot()["actual_tokens"] == 112
    assert ledger.snapshot()["records"][0]["failure"]["terminal_decision"]["validation_category"] == "understanding_reference"
    assert len(sdk_runtime.requests) == 1


async def test_valid_understanding_uses_successful_tool_facts(sdk_runtime):
    """成功工具事实不能被丢弃，失败工具和助手编造不能替代可信来源。参数：SDK边界；返回：None；异常：断言失败。"""
    raw, resolved = "那它还能退吗？", "订单1001的猫砂盆还能退吗？"
    history = [{"role": "user", "content": "帮我查一下上一笔订单"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "order-call", "name": "query_order", "args": {"order_id": "1001"}, "type": "tool_call"}]},
        {"role": "tool", "tool_call_id": "order-call",
         "content": json.dumps({"ok": True, "data": {"order_id": "1001", "product_name": "猫砂盆"}, "error": None})}]
    sdk_runtime.responses.append(understanding_response(resolved, needs_resolution=True, order_id="1001", usage=True))
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
        source_messages(history, raw), ledger)
    assert result.resolved_query == resolved and result.order_id == "1001"
    assert "failure" not in ledger.snapshot()["records"][0]


async def test_understanding_budget_signal_is_not_downgraded(sdk_runtime):
    """总预算停止不能伪装为理解失败继续下游。参数：SDK边界；返回：None；异常：断言失败。"""
    ledger = BudgetLedger(total_token_budget=1)
    with pytest.raises(WorkflowBudgetExceeded):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
            [{"role": "user", "content": "能退吗"}], ledger)
    assert sdk_runtime.requests == [] and ledger.snapshot()["model_calls"] == 0


async def test_understanding_cancellation_propagates(sdk_runtime):
    """取消信号不能变成成功回退结果。参数：SDK边界；返回：None；异常：断言失败。"""
    sdk_runtime.responses.append(asyncio.CancelledError())
    ledger = BudgetLedger()
    with pytest.raises(asyncio.CancelledError):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
            [{"role": "user", "content": "能退吗"}], ledger)
    assert ledger.snapshot()["model_calls"] == 1
    assert "failure" not in ledger.snapshot()["records"][0]


async def test_understanding_requires_real_current_user_message(sdk_runtime):
    """无当前用户消息时不能拿助手话作回退原句。参数：SDK边界；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
            [{"role": "assistant", "content": "订单1001"}], BudgetLedger())
    assert sdk_runtime.requests == []


async def test_classification_network_failure_remains_service_error(sdk_runtime):
    """分类网络故障不能被协议回退掩盖。参数：SDK边界；返回：None；异常：断言失败。"""
    sdk_runtime.responses.append(RuntimeError("上游不可用"))
    ledger = BudgetLedger()
    with pytest.raises(RuntimeError):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify([], ledger)
    assert ledger.snapshot()["model_calls"] == 1


async def test_invalid_resolved_query_alone_keeps_failure_diagnostic(sdk_runtime):
    """完整问失真被强制透传修正后也不能漏记失败。参数：SDK边界；返回：None；异常：断言失败。"""
    sdk_runtime.responses.append(understanding_response("订单1001的运费怎么算", standard="运费怎么算"))
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
        source_messages([{"role": "user", "content": "订单1001"}], "运费怎么算"), ledger)
    assert result.resolved_query == result.standard_query == "运费怎么算"
    assert ledger.snapshot()["records"][0]["failure"]["terminal_decision"]["validation_category"] == "understanding_reference"


async def test_complete_query_whitespace_restoration_is_not_source_failure(sdk_runtime):
    """仅按约定恢复原问空白不能误报来源失真。参数：SDK边界；返回：None；异常：断言失败。"""
    raw = "  运费怎么算？  "
    sdk_runtime.responses.append(understanding_response(raw.strip(), standard=raw))
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
        [{"role": "user", "content": raw}], ledger)
    assert result.resolved_query == raw
    assert "failure" not in ledger.snapshot()["records"][0]
