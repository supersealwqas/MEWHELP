"""验证分类派生问法失真时保留可信原话、原始诊断和既有知识输入上限。"""

import json
from copy import deepcopy

import pytest

from app.core.model_budget import BudgetLedger
from app.schemas.chat import InvalidToolDecision
from app.workflow.graph import build_workflow
from app.workflow.model import WorkflowModelAdapter
from tests.workflow.graph_fakes import NodeTrace, dependencies, ranked, turn
from tests.workflow.test_model import intent_response, sdk_runtime, terminal_response, text_response


@pytest.mark.parametrize("query,history,derived", [
    ("包邮吗", [], "1001订单包邮吗"),
    ("1002包邮吗", [{"role": "user", "content": "1001订单"}], "1001订单包邮吗"),
    ("包邮吗", [{"role": "assistant", "content": "您的订单是1001，金额99元。"}], "1001订单满99元包邮吗"),
], ids=["invented_order", "changed_order", "assistant_only_facts"])
async def test_invalid_query_reference_falls_back_without_another_model_call(sdk_runtime, query, history, derived):
    """换编号或采用助手散文时，不能终止分类、放行派生问法或额外请求模型。

    参数：sdk_runtime真实SDK的HTTP边界；query完整原问；history历史；derived失真问法。
    返回：None，核验原话、仅一次请求、真实用量和原始工具诊断。
    异常：AssertionError，回退、审计或输入完整性被破坏。
    """
    messages = [*history, {"role": "user", "content": query}]
    original = deepcopy(messages)
    sdk_runtime.responses.extend([intent_response("商品咨询", knowledge_query=derived, usage=True) for _ in range(2)])
    ledger = BudgetLedger()

    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify(messages, ledger)

    assert result.intent == "商品咨询" and result.knowledge_query == query
    assert messages == original and sdk_runtime.requests[0]["messages"][1:] == original
    assert len(sdk_runtime.requests) == ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["actual_tokens"] == 112
    failure = ledger.snapshot()["records"][0]["failure"]
    assert failure["terminal_decision"]["validation_category"] == "intent_reference"
    assert failure["query_fallback"] == "original_query"
    assert failure["tool_call_chunks"][0]["name"] == "submit_intent"
    assert json.loads(failure["tool_call_chunks"][0]["args"]) == {
        "intent": "商品咨询", "knowledge_query": derived}


@pytest.mark.parametrize("query,derived", [
    ("  如果我是会员，实付金额就是39块9，好像是金卡会员包邮吗\n", "金卡会员是否有免邮或包邮权益"),
    ("39块9" + "问" * 1996, "会员包邮政策"),
], ids=["dropped_amount", "2000_characters"])
async def test_amount_only_difference_keeps_derived_query_without_fallback(sdk_runtime, query, derived):
    """金额被概括成检索问法时，不应再当作失真问法回退（回归：曾导致 conv112/116 整轮 failed）。

    参数：sdk_runtime真实SDK的HTTP边界；query含金额原问；derived省略金额的派生问法。
    返回：None，核验派生问法被接受、仅一次请求且无失败诊断。
    异常：AssertionError，金额仍触发回退或发生额外请求。
    """
    messages = [{"role": "user", "content": query}]
    sdk_runtime.responses.append(intent_response("商品咨询", knowledge_query=derived, usage=True))
    ledger = BudgetLedger()

    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify(messages, ledger)

    assert result.knowledge_query == derived
    assert len(sdk_runtime.requests) == 1
    assert "failure" not in ledger.snapshot()["records"][0]


@pytest.mark.parametrize("query,derived", [
    ("满99元包邮吗", "满199元包邮吗"),
    ("39块9包邮吗", "99元包邮吗"),
], ids=["changed_threshold", "changed_amount"])
async def test_tampered_amount_still_falls_back_to_original(sdk_runtime, query, derived):
    """放宽金额保留后，改写金额本身仍必须被拦下并回退到原问。

    参数：sdk_runtime真实SDK的HTTP边界；query原问；derived被篡改金额的问法。
    返回：None，核验仍回退到原问。
    异常：AssertionError，篡改金额被放行。
    """
    sdk_runtime.responses.append(intent_response("商品咨询", knowledge_query=derived, usage=True))
    ledger = BudgetLedger()

    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify(
        [{"role": "user", "content": query}], ledger)

    assert result.knowledge_query == query
    failure = ledger.snapshot()["records"][0]["failure"]
    assert failure["query_fallback"] == "original_query"


async def test_valid_query_keeps_user_history_reference_without_fallback(sdk_runtime):
    """合法用户历史补全被一律丢弃或留下误报诊断时失败。

    参数：sdk_runtime真实SDK的HTTP边界。返回：None。
    异常：AssertionError，已核验问法被替换、输入改变或产生额外请求。
    """
    messages = [{"role": "user", "content": "1001订单"},
                {"role": "assistant", "content": "好的。"}, {"role": "user", "content": "包邮吗"}]
    sdk_runtime.responses.append(intent_response("商品咨询", knowledge_query="1001订单的包邮规则"))
    ledger = BudgetLedger()

    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify(messages, ledger)

    assert result.knowledge_query == "1001订单的包邮规则"
    assert len(sdk_runtime.requests) == 1 and "failure" not in ledger.snapshot()["records"][0]
    assert sdk_runtime.requests[0]["messages"][1:] == messages


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": "用户想问金卡会员包邮。"}]])
async def test_query_without_any_original_user_message_fails_explicitly(sdk_runtime, messages):
    """缺少用户原问时不能拿助手内容或模型问法冒充可信回退来源。

    参数：sdk_runtime真实SDK边界；messages无用户原问的异常输入。返回：None。
    异常：AssertionError，异常输入被接受、诊断丢失或发生无意义重试。
    """
    sdk_runtime.responses.append(intent_response("商品咨询", knowledge_query="金卡会员包邮政策"))
    ledger = BudgetLedger()

    with pytest.raises(InvalidToolDecision) as failure:
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify(messages, ledger)

    assert len(sdk_runtime.requests) == 1
    assert failure.value.audit_record["terminal_decision"]["validation_category"] == "intent_reference"
    assert failure.value.audit_record["classification_budget"]["model_calls"] == 1
    assert "query_fallback" not in ledger.snapshot()["records"][0]["failure"]


async def test_membership_amount_fallback_reaches_real_graph_retrieval(sdk_runtime):
    """真实分类边界丢金额后应按完整原问检索，不能停在分类重试或篡改持久化原话。

    参数：sdk_runtime真实SDK的HTTP边界。返回：None，核对图完成及检索输入。
    异常：AssertionError，未进入知识检索、原话丢失或模型请求超脚本。
    """
    query = "如果我是会员，实付金额就是39块9，好像是金卡会员包邮吗"
    sdk_runtime.responses.extend([
        intent_response("商品咨询", knowledge_query="金卡会员是否有免邮或包邮权益"),
        terminal_response("answer"), text_response("若您是金卡会员，每月可享3次免基础运费权益[1]。")])
    deps = dependencies(ranked=[ranked(answer="金卡会员每月享3次免基础运费权益。")])
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)

    result = await build_workflow(deps, checkpointer=None).ainvoke(turn(query))

    assert deps.search.calls == [("retrieve", "金卡会员是否有免邮或包邮权益")]
    assert result["raw_query"] == result["context"]["user_message"] == query
    assert result["messages"][0] == {"role": "user", "content": query}
    assert result["answer"] == "若您是金卡会员，每月可享3次免基础运费权益[1]。"
    assert len(sdk_runtime.requests) == 3
    # 金额被概括成检索问法属于合法归一，不再产生失真回退诊断。
    assert "failure" not in result["budget"]["records"][0]


@pytest.mark.parametrize("intent", ["商品咨询", "退款退货", "订单"])
async def test_overlong_original_keeps_real_graph_input_limit(sdk_runtime, intent):
    """超长原问不能构造非法分类结果、绕过输入上限或在到达上限前因失真问法失败。

    参数：sdk_runtime真实SDK边界；intent知识或混合业务意图。返回：None。
    异常：AssertionError，原话被截断、发生检索/决策或未按input_limit完成。
    """
    query = "39块9" + "问" * 1997
    sdk_runtime.responses.extend([intent_response(intent, knowledge_query="会员包邮政策") for _ in range(2)])
    deps, trace = dependencies(), NodeTrace()
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)

    result = await build_workflow(deps, checkpointer=None).ainvoke(turn(query), {"callbacks": [trace]})

    assert result["stop_reason"] == "input_limit" and result["route"] == "knowledge"
    assert result["raw_query"] == result["context"]["user_message"] == query
    assert deps.search.calls == [] and deps.executor.calls == []
    assert not {"confidence_check", "main_agent"}.intersection(trace.nodes)
    assert len(sdk_runtime.requests) == 1
    # 超长原问由检索前长度检查拦住；分类本身合法，不再留失真回退诊断。
    assert "failure" not in result["budget"]["records"][0]
    assert deps.chat.finished[0]["low_confidence"] is None
