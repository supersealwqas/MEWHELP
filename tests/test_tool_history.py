"""整轮裁剪与中断工具事实恢复的测试，防止消息协议被历史裁剪破坏。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
