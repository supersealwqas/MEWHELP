"""整轮裁剪与中断工具事实恢复的测试，防止消息协议被历史裁剪破坏。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.schemas.history import StoredMessage, StoredTurn


def stored_turn(status="cancelled", paired=True):
    """构造已提交工单历史。参数：状态、是否有结果；返回：StoredTurn；异常：无。"""
    messages = [StoredMessage(role="user", content="请建工单"), StoredMessage(role="assistant", content="",
        tool_calls=[{"name": "create_ticket", "args": {"description": "延误", "ticket_type": "咨询"}, "id": "call1"}])]
    if paired:
        messages.append(StoredMessage(role="tool", content='{"ok":true,"data":{"ticket_no":"T-original"}}', tool_call_id="call1"))
    return StoredTurn(turn_id="t1", turn_status=status, messages=messages)


def test_interrupted_history_retains_paired_facts_with_marker():
    """丢掉中断轮的已建工单会使此测试失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import assemble_turns
    messages = assemble_turns([stored_turn()])[0]
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert "T-original" in messages[2].content and "未完成" in messages[-1].content


def test_unpaired_and_running_history_never_leak_tool_requests():
    """未配对申请与运行中轮次不能进入模型上下文。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import assemble_turns
    history = assemble_turns([stored_turn(paired=False), stored_turn("running")])
    assert len(history) == 1 and [m.type for m in history[0]] == ["human", "ai"]
    assert not any(getattr(m, "tool_calls", None) for m in history[0])


def test_trim_whole_turn_without_orphaning_tool_result():
    """预算不足只能丢完整旧轮，不能留下孤立工具结果。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import trim_turns
    old = [HumanMessage(content="很长旧问题" * 100), AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "old"}]), ToolMessage(content="答案", tool_call_id="old")]
    current = [HumanMessage(content="你好")]
    assert trim_turns([old], current, 30) == current


def test_current_question_and_tool_result_cannot_be_silently_trimmed():
    """当前工具事实超预算必须显式失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import ContextBudgetExceeded, trim_turns
    with pytest.raises(ContextBudgetExceeded):
        trim_turns([], [HumanMessage(content="问题"), ToolMessage(content="长结果" * 100, tool_call_id="x")], 15)


def test_chinese_text_uses_larger_budget_than_equal_length_ascii():
    """把中文按英文字符低估会漏过超限。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import ContextBudgetExceeded, trim_turns
    ascii_message = [HumanMessage(content="a" * 100)]
    assert trim_turns([], ascii_message, 100) == ascii_message
    with pytest.raises(ContextBudgetExceeded):
        trim_turns([], [HumanMessage(content="中" * 100)], 100)


def test_tool_arguments_and_protocol_metadata_consume_dialog_budget():
    """忽略工具参数或调用标识会错误保留超限申请。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import ContextBudgetExceeded, trim_turns
    current = [HumanMessage(content="请建单"), AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "中" * 100}, "id": "call-1"}]),
        ToolMessage(content="已保存", tool_call_id="call-1")]
    with pytest.raises(ContextBudgetExceeded):
        trim_turns([], current, 150)


def test_message_name_and_call_id_consume_dialog_budget():
    """只算正文会漏掉长协议标识占用。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import ContextBudgetExceeded, trim_turns
    with pytest.raises(ContextBudgetExceeded):
        trim_turns([], [HumanMessage(content="你好", name="x" * 400)], 100)
    with pytest.raises(ContextBudgetExceeded):
        trim_turns([], [ToolMessage(content="结果", tool_call_id="x" * 400)], 100)


def test_prepare_counts_system_without_charging_it_to_dialog_budget():
    """把系统说明算入对话预算或漏掉完整请求预算都会失败。参数：无；返回：None；异常：断言失败。"""
    from app.core import memory
    current = [HumanMessage(content="你好")]
    prepared = memory.prepare_messages([], current, 30, system_text="中" * 100, request_max_tokens=300)
    assert prepared == [SystemMessage(content="中" * 100), *current]
    with pytest.raises(memory.ContextBudgetExceeded):
        memory.prepare_messages([], current, 30, system_text="中" * 100, request_max_tokens=200)


def test_prepare_counts_tool_schema_only_when_supplied():
    """遗漏工具定义开销或把工具占用算入对话预算都会失败。参数：无；返回：None；异常：断言失败。"""
    from app.core import memory
    current = [HumanMessage(content="你好")]
    tools = [{"type": "function", "function": {
        "name": "query_test", "description": "a" * 4000, "parameters": {"type": "object"}}}]
    assert memory.prepare_messages([], current, 30, system_text="规则", request_max_tokens=100) == [
        SystemMessage(content="规则"), *current]
    with pytest.raises(memory.ContextBudgetExceeded):
        memory.prepare_messages([], current, 30, system_text="规则", tools=tools, request_max_tokens=1000)
    assert memory.prepare_messages([], current, 30, system_text="规则", tools=tools,
                                   request_max_tokens=1200)[1:] == current


def test_prepare_tool_schema_counts_unicode_without_ascii_escaping():
    """工具说明被转为ASCII转义后会低估中文权重。参数：无；返回：None；异常：断言失败。"""
    from app.core import memory
    tools = [{"type": "function", "function": {
        "name": "query_test", "description": "中" * 1000, "parameters": {"type": "object"}}}]
    with pytest.raises(memory.ContextBudgetExceeded):
        memory.prepare_messages([], [HumanMessage(content="你好")], 100, system_text="规则", tools=tools,
                                request_max_tokens=1900)
    assert len(memory.prepare_messages([], [HumanMessage(content="你好")], 100,
                                       system_text="规则", tools=tools, request_max_tokens=2200)) == 2


@pytest.mark.parametrize(("dialog_budget", "request_budget"), [(100, 500), (500, 130)])
def test_prepare_discards_old_complete_turn_for_either_budget(dialog_budget, request_budget):
    """任一预算不足时必须丢整轮，不能拆开调用和结果。参数：两层预算；返回：None；异常：断言失败。"""
    from app.core import memory
    old = [HumanMessage(content="中" * 100), AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "old"}]),
        ToolMessage(content="七天内可申请", tool_call_id="old")]
    recent = [HumanMessage(content="上一问"), AIMessage(content="上一答")]
    current = [HumanMessage(content="当前问题")]
    prepared = memory.prepare_messages([old, recent], current, dialog_budget,
                                       system_text="固定规则", request_max_tokens=request_budget)
    assert prepared == [SystemMessage(content="固定规则"), *recent, *current]
    assert old[0].content == "中" * 100


def test_prepare_retains_current_tool_facts_or_explicitly_rejects():
    """完整请求不足时静默丢当前已建单事实会失败。参数：无；返回：None；异常：断言失败。"""
    from app.core import memory
    current = [HumanMessage(content="请建单"), AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "延误"}, "id": "call-1"}]),
        ToolMessage(content='{"ticket_no":"T-saved","status":"待处理"}', tool_call_id="call-1")]
    prepared = memory.prepare_messages([], current, 500, system_text="系统规则", request_max_tokens=500)
    assert prepared[1:] == current
    with pytest.raises(memory.ContextBudgetExceeded):
        memory.prepare_messages([], current, 500, system_text="中" * 100, request_max_tokens=230)
    assert "T-saved" in current[-1].content


@pytest.mark.parametrize("scenario", ["greeting", "order", "faq"])
def test_default_budgets_fit_real_prompts_and_business_tools(scenario):
    """默认预算拒绝正常中文问候或业务查询会失败。参数：业务场景；返回：None；异常：断言失败。"""
    from app.core import memory
    from app.config import Settings
    from app.core.prompts import CUSTOMER_SERVICE_SYSTEM, FINAL_ANSWER_SYSTEM, TOOL_DECISION_SYSTEM
    from app.tools.registry import build_registry
    settings = Settings(_env_file=None, chat_base_url="http://test.invalid/v1",
                        chat_model="test-model", chat_api_key="test-key")
    assert settings.request_token_budget == 16000
    questions = {"greeting": "你好", "order": "帮我查订单 ORD-001", "faq": "退货政策是什么"}
    current = [HumanMessage(content=questions[scenario])]
    decision = memory.prepare_messages([], current, settings.token_budget,
        system_text=CUSTOMER_SERVICE_SYSTEM + TOOL_DECISION_SYSTEM, tools=build_registry().all(),
        request_max_tokens=settings.request_token_budget)
    assert decision[1:] == current
    if scenario != "greeting":
        tool_name = "query_order" if scenario == "order" else "query_faq"
        args = {"order_id": "ORD-001"} if scenario == "order" else {"keyword": "退货政策"}
        result = ('{"ok":true,"data":{"order_id":"ORD-001","status":"已发货"}}' if scenario == "order"
                  else '{"ok":true,"data":{"hits":[{"answer":"签收七天内可申请退货，具体条件以规则为准"}]}}')
        current.extend([AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": "call-1"}]),
                        ToolMessage(content=result, tool_call_id="call-1")])
    answer = memory.prepare_messages([], current, settings.token_budget,
        system_text=CUSTOMER_SERVICE_SYSTEM + FINAL_ANSWER_SYSTEM,
        request_max_tokens=settings.request_token_budget)
    assert answer[1:] == current
