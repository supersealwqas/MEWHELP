"""真实验收证据校验的离线反例，防止HTTP200或局部成功冒充整轮成功。"""

import copy
import json
import pytest

from tests.ch03_live_runtime import inspect_exchange


def _exchange() -> dict:
    """构造手工核验的完整一轮证据。参数：无；返回：证据；异常：无。"""
    call = {"id": "call1", "name": "query_faq", "args": {"keyword": " 邮费 "}}
    return {
        "http_status": 200, "content_type": "text/event-stream; charset=utf-8",
        "events": [
            {"event": "tool_status", "data": {"tool_call_id": "call1", "name": "query_faq", "status": "succeeded"}},
            {"event": "delta", "data": {"delta": "答案"}},
            {"event": "done", "data": "[DONE]"},
        ],
        "messages": [
            {"id": 1, "turn_id": "turn1", "role": "user", "content": "邮费", "tool_calls": None, "tool_call_id": None, "turn_status": "completed"},
            {"id": 2, "turn_id": "turn1", "role": "assistant", "content": "", "tool_calls": [call], "tool_call_id": None, "turn_status": "completed"},
            {"id": 3, "turn_id": "turn1", "role": "tool", "content": json.dumps({"ok": True, "data": {"hits": [{"id": 1, "answer": "答案", "question": "邮费", "category": "配送"}]}, "error": None, "is_mock": False}), "tool_calls": None, "tool_call_id": "call1", "turn_status": "completed"},
            {"id": 4, "turn_id": "turn1", "role": "assistant", "content": "答案", "tool_calls": None, "tool_call_id": None, "turn_status": "completed"},
        ],
    }


@pytest.mark.parametrize("fault", ["no_messages", "no_done", "duplicate_done", "error", "empty_delta", "wrong_final", "failed_turn", "missing_result", "bad_result", "missing_hits", "no_tool", "keyword_rewrite", "late_status"])
def test_incomplete_exchange_fails(fault: str) -> None:
    """任一证据断链均拒绝通过。参数：故障名；返回：None；异常：断言失败。"""
    observed = _exchange()
    if fault == "no_messages":
        observed["messages"] = []
    elif fault == "no_done":
        observed["events"].pop()
    elif fault == "duplicate_done":
        observed["events"].append(observed["events"][-1])
    elif fault == "error":
        observed["events"].insert(0, {"event": "error", "data": {"error": "失败"}})
    elif fault == "empty_delta":
        observed["events"][1]["data"]["delta"] = ""
    elif fault == "wrong_final":
        observed["messages"][-1]["content"] = "另一回答"
    elif fault == "failed_turn":
        observed["messages"][0]["turn_status"] = "failed"
    elif fault == "missing_result":
        observed["messages"].pop(2)
    elif fault in ("bad_result", "missing_hits"):
        observed["messages"][2]["content"] = "invalid" if fault == "bad_result" else '{"ok":true,"data":{}}'
    elif fault == "no_tool":
        observed["messages"] = [observed["messages"][0], observed["messages"][-1]]
    elif fault == "keyword_rewrite":
        observed["messages"][1]["tool_calls"][0]["args"]["keyword"] = "快递费"
    else:
        observed["events"][0], observed["events"][1] = observed["events"][1], observed["events"][0]
    assert inspect_exchange({"query": "邮费", "expect_empty": False}, observed)["failures"]


def test_success_uses_effective_keyword_and_paired_hits() -> None:
    """原申请去首尾空白后才是编码输入。参数：无；返回：None；异常：断言失败。"""
    result = inspect_exchange({"query": "邮费", "expect_empty": False}, copy.deepcopy(_exchange()))
    assert result["failures"] == []
    assert result["effective_keyword"] == "邮费"
    assert result["executed_keyword"] == "邮费"
    assert result["raw_keyword"] == " 邮费 "
    assert result["hits"][0]["id"] == 1


def test_negative_without_tool_must_not_have_orphan_status() -> None:
    """无工具负例不能带未配对的成功徽章。参数：无；返回：None；异常：断言失败。"""
    observed = _exchange()
    observed["messages"] = [observed["messages"][0], observed["messages"][-1]]
    assert inspect_exchange({"query": "邮费", "expect_empty": True}, observed)["failures"]


def test_complete_negative_without_tool_is_allowed() -> None:
    """无关问题可以不调用工具但仍须完整落库。参数：无；返回：None；异常：断言失败。"""
    observed = _exchange()
    observed["messages"] = [observed["messages"][0], observed["messages"][-1]]
    observed["events"].pop(0)
    assert inspect_exchange({"query": "邮费", "expect_empty": True}, observed)["failures"] == []
