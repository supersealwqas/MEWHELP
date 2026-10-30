"""验证新轮State完整初始化、历史隔离和全部运行字段重置。"""

import json

import pytest

from app.schemas.tools import ToolContext


def _context(query="新的问题", turn_id="new-turn"):
    """构造可信运行上下文。参数：query 原话、turn_id 轮次；返回：ToolContext；异常：模型校验失败。"""
    return ToolContext(conversation_id=42, session_id="session", user_id="user", turn_id=turn_id,
                       user_message=query)


def test_new_turn_replaces_every_runtime_field_and_retains_history():
    """上一轮意图、证据、调用或建议进入下一轮时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.state import ConversationState, new_turn_state

    history = [{"turn_id": "prior", "turn_status": "completed", "messages": [
        {"role": "user", "content": "查物流", "tool_calls": None, "tool_call_id": None},
        {"role": "assistant", "content": "请给订单号", "tool_calls": None, "tool_call_id": None},
    ]}]
    previous = new_turn_state(_context("查物流", "prior"), history, 11)
    previous.update(resolved_query="旧问题", messages=[{"role": "tool", "content": "旧事实"}],
                    trace_identity={"trace_id": "旧轮追踪", "root_observation_id": "旧根"},
                    retrieval_snapshot={"status": "fault", "attempts": []}, pool_signals=[{"source": "self_check", "reason": "旧理由"}],
                    intent="物流", standard_query="旧检索问题", intent_confidence=0.9, order_id="1001",
                    order_data={"order_id": "1001"}, expanded_queries=["旧扩写"], selection_action={"action_id": "old"}, route="business", evidence=[{"chunk_id": 9}], evidence_scores=[0.9],
                    gate={"passed": True}, retrieval_attempted=True, optional_knowledge_probe=True,
                    knowledge_faults={"old": "OSError"}, degradations=[{"stage": "answer", "error_type": "ConnectionError"}],
                    pending_calls=[{"id": "old-call"}], seen_call_ids=["old-call"],
                    pending_bindings={"old-call": {"identity": "old"}}, pending_tool_bindings={"old-call": {}},
                    pending_ticket={"action": "old"}, ticket_intent={"status": "confirmed"},
                    ticket_write_state={"status": "completed"}, knowledge_cache={"old": {}}, route_ready=True,
                    seen_call_signatures=["old-signature"], tool_count=4, decision_count=5,
                    suggested_kinds=["handoff"], suggested_actions=[{"action_id": "old-action"}],
                    refund_action={"action_id": "old-refund"}, answer="旧答案", stop_reason="answer", completion_metadata={"old": True})
    prior_ledger = BudgetLedger.from_snapshot(previous["budget"])
    prior_ledger.reserve("classify", 10, 128)
    previous["budget"] = prior_ledger.snapshot()

    # 模拟checkpoint替换更新：即便旧State中每个运行字段非空，新轮也全部覆盖。
    merged = {**previous, **new_turn_state(_context("1001"), previous["history_turns"], 13)}
    expected = {
        "context": {"conversation_id": 42, "session_id": "session", "user_id": "user",
                    "turn_id": "new-turn", "tool_call_id": "", "user_message": "1001"},
        "trace_identity": None,
        "retrieval_snapshot": None, "pool_signals": [],
        "raw_query": "1001", "resolved_query": "", "history_turns": history, "audit_cursor": 13,
        "messages": [
            {"id": "turn:prior:user", "role": "user", "content": "查物流", "additional_kwargs": {"turn_id": "prior"}},
            {"id": "turn:prior:answer", "role": "assistant", "content": "请给订单号", "additional_kwargs": {"turn_id": "prior"}},
            {"id": "turn:new-turn:user", "role": "user", "content": "1001", "additional_kwargs": {"turn_id": "new-turn"}}],
        "context_version": 7, "current_message_id": "turn:new-turn:user", "history_context": None, "context_notes": [],
        "intent": None, "standard_query": "", "intent_confidence": None,
        "order_id": None, "order_data": None, "expanded_queries": [], "selection_action": None, "route": None, "evidence": [], "evidence_scores": [],
        "gate": None, "retrieval_attempted": False, "optional_knowledge_probe": False, "knowledge_faults": {}, "degradations": [], "pending_calls": [], "seen_call_ids": [], "seen_call_signatures": [],
        "pending_bindings": {}, "pending_tool_bindings": {}, "tool_call_facts": {}, "pending_ticket": None, "ticket_intent": None,
        "ticket_write_state": None, "knowledge_cache": {}, "route_ready": False,
        "tool_count": 0, "decision_count": 0, "suggested_kinds": [], "suggested_actions": [],
        "refund_action": None, "answer": "", "stop_reason": None, "completion_metadata": {},
    }
    budget = merged.pop("budget")
    assert merged == expected
    assert set(ConversationState.__annotations__) == set(expected) | {"budget"}
    assert budget["model_calls"] == budget["reserved_units"] == 0
    assert budget["actual_tokens"] is None and budget["records"] == []
    assert previous["budget"]["model_calls"] == 1


def test_state_copies_nested_history_and_owns_a_serializable_budget():
    """外部历史或另一轮账本可原地污染当前State时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.state import new_turn_state

    history = [{"turn_id": "old", "turn_status": "completed", "messages": [{"role": "user", "content": "原话"}]}]
    first = new_turn_state(_context(), history, 1)
    second = new_turn_state(_context(), history, 1)
    history[0]["messages"][0]["content"] = "外部修改"
    first["history_turns"][0]["messages"][0]["content"] = "第一轮修改"
    ledger = BudgetLedger.from_snapshot(first["budget"])
    ledger.reserve("classify", 10, 128)
    first["budget"] = ledger.snapshot()
    assert second["history_turns"][0]["messages"][0]["content"] == "原话"
    assert second["budget"]["records"] == []
    assert json.loads(json.dumps(first, ensure_ascii=False)) == first


def test_state_keeps_all_4000_characters_in_raw_query_and_context():
    """合法长原话在State入口被截断或去空白时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.state import new_turn_state

    query = "  " + "问" * 3996 + "\n "
    state = new_turn_state(_context(query), [], 0)
    assert len(query) == 4000
    assert state["raw_query"] == state["context"]["user_message"] == query


@pytest.mark.parametrize("query", ["", " \n ", "问" * 4001], ids=["empty", "whitespace", "4001-chars"])
def test_state_rejects_invalid_query_without_truncating(query):
    """空问题或超4000字原话悄悄进入工作流时失败。参数：query 非法原话；返回：None；异常：断言失败。"""
    from app.workflow.state import new_turn_state

    with pytest.raises(ValueError):
        new_turn_state(_context(query), [], 0)


@pytest.mark.parametrize("cursor", [-1, True, "1"])
def test_state_rejects_invalid_audit_cursor(cursor):
    """历史游标错误导致遗漏或重放历史时失败。参数：cursor 非法游标；返回：None；异常：断言失败。"""
    from app.workflow.state import new_turn_state

    with pytest.raises(ValueError):
        new_turn_state(_context(), [], cursor)


def test_new_turn_clears_selection_and_order():
    """新轮继承旧选单、订单或置信度时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.state import new_turn_state

    old = {"order_id": "1001", "order_data": {"order_id": "1001"}, "selection_action": {"action_id": "old"},
           "expanded_queries": ["旧查询"], "standard_query": "旧问", "intent_confidence": 0.99}
    merged = {**old, **new_turn_state(_context("新的问题"), [], 0)}
    assert merged["order_id"] is merged["order_data"] is merged["selection_action"] is None
    assert merged["intent_confidence"] is None and merged["standard_query"] == ""
    assert merged["expanded_queries"] == []
def test_optional_probe_is_reset_and_missing_marker_is_conservative():
    """新轮不能沿用旧探测资格，旧checkpoint缺字段不能猜成探测。参数：无；返回：None；异常：断言失败。"""
    from tests.workflow.graph_fakes import turn
    from app.workflow.policy_context import optional_probe_miss
    current = turn("查订单")
    assert current["optional_knowledge_probe"] is False
    current.update(route="business", gate={"passed": False})
    current.pop("optional_knowledge_probe")
    assert optional_probe_miss(current) is False
