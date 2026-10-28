"""模型结果视图测试：保留事实、枚举可读、预算有界且原始持久化数据不变。"""

from copy import deepcopy
import json

import pytest
from langchain_core.messages import ToolMessage

from app.core.context_budget import ContextBudgetExceeded, TokenEstimator
from app.schemas.tools import ToolResult
from app.tools.contracts import ResultPolicy


def project(result, policy=None, budget=300):
    """调用真实投影入口。参数：完整结果/本地规则/预算；返回：模型字典；异常：预算不足。"""
    from app.tools.presentation import model_tool_result
    return model_tool_result(result, policy or ResultPolicy(), max_tokens=budget, estimator=TokenEstimator())


def test_selected_fields_keep_identifier_and_local_enum_labels():
    """字段投影丢失关键物流号或显示技术枚举会失败。参数：无；返回：None；异常：断言失败。"""
    result = ToolResult.success({"tracking_no": "SF-123", "status": "delivered", "carrier": "顺丰",
                                 "debug_payload": "无须展示"})
    view = project(result, ResultPolicy(fields=("status", "carrier"), enum_labels={"status": {"delivered": "已签收"}}))
    assert view["data"] == {"status": "已签收", "carrier": "顺丰", "tracking_no": "SF-123"}
    assert "is_mock" not in view and result.data["status"] == "delivered"


def test_unknown_enum_keeps_raw_value_and_unknown_marker():
    """未知枚举被套成已知成功状态会失败。参数：无；返回：None；异常：断言失败。"""
    view = project(ToolResult.success({"status": "alien_status"}),
                   ResultPolicy(enum_labels={"status": {"delivered": "已签收"}}))
    assert view["data"]["status"]["value"] == "alien_status"
    assert "未知" in view["data"]["status"]["label"]


def test_large_result_is_bounded_and_original_data_unchanged():
    """预算按字符硬切、丢失工单号或改写完整事实会失败。参数：无；返回：None；异常：断言失败。"""
    result = ToolResult.success({"ticket_no": "T-real", "status": "待处理", "amount": 29.5,
                                 "description": "中文描述" * 1000, "events": ["记录" * 100] * 50})
    original = deepcopy(result.model_dump())
    view = project(result, budget=200)
    assert TokenEstimator().value_tokens(view) <= 200 and view["truncated"] is True
    assert view["data"]["ticket_no"] == "T-real" and view["data"]["status"] == "待处理"
    assert view["data"]["amount"] == 29.5 and result.model_dump() == original


def test_budget_too_small_for_core_receipt_is_explicit():
    """把放不下的核心事实悄悄删掉会失败。参数：无；返回：None；异常：断言失败。"""
    with pytest.raises(ContextBudgetExceeded):
        project(ToolResult.success({"ticket_no": "T-real", "status": "待处理"}), budget=1)


@pytest.mark.parametrize("facts", [
    {"receipt": {"ticket_no": "T-real", "status": "待处理"}},
    {"payload": {"items": [{"tracking_no": "SF-1", "status": "已签收"},
                            {"tracking_no": "SF-2", "status": "运输中"}]}}
])
def test_field_selection_keeps_nested_fact_paths(facts):
    """字段规则删除父容器并丢失嵌套回执或列表ID会失败。参数：事实结构；返回：None；异常：断言失败。"""
    result = ToolResult.success({"message": "处理完成", **deepcopy(facts), "debug": "内部调试"})
    original = deepcopy(result.model_dump())
    expected = {"message": "处理完成", **facts}
    budget = TokenEstimator().value_tokens({"ok": True, "data": expected, "error": None})
    view = project(result, ResultPolicy(fields=("message",)), budget=budget)
    assert view["data"] == expected and result.model_dump() == original
    assert TokenEstimator().value_tokens(view) <= budget


@pytest.mark.parametrize("facts", [
    {"receipt": {"ticket_no": "T-real", "status": "待处理"}},
    {"payload": {"items": [{"tracking_no": "SF-1", "status": "已签收"},
                            {"tracking_no": "SF-2", "status": "运输中"}]}}
])
def test_nested_core_facts_over_budget_raise_instead_of_empty_success(facts):
    """嵌套核心事实放不下却删除父容器或列表项返回空成功会失败。参数：事实结构；返回：None；异常：断言失败。"""
    result = ToolResult.success(deepcopy(facts))
    original = deepcopy(result.model_dump())
    budget = TokenEstimator().value_tokens({"ok": True, "data": facts, "error": None}) - 1
    with pytest.raises(ContextBudgetExceeded):
        project(result, budget=budget)
    assert result.model_dump() == original


def test_nested_list_keeps_all_identifiers_while_trimming_descriptions():
    """列表整体减半丢回执，或为保护父容器而不能裁剪内部描述都会失败。参数：无；返回：None；异常：断言失败。"""
    facts = {"payload": {"items": [{"ticket_no": "T-1", "status": "待处理"},
                                   {"ticket_no": "T-2", "status": "已受理"}]}}
    data = deepcopy(facts)
    for row in data["payload"]["items"]:
        row["description"] = "这是一段可裁剪的详细描述" * 50
        row["debug"] = "短说明"
    result = ToolResult.success(data)
    original = deepcopy(result.model_dump())
    expected = {"ok": True, "data": facts, "error": None, "truncated": True}
    budget = TokenEstimator().value_tokens(expected)
    view = project(result, budget=budget)
    assert view == expected and result.model_dump() == original
    assert TokenEstimator().value_tokens(view) <= budget


def test_failure_category_and_safe_explanation_are_preserved():
    """失败投影伪装成功或抹掉可行动说明会失败。参数：无；返回：None；异常：断言失败。"""
    view = project(ToolResult.failed("ORDER_NOT_OWNED", "没有找到您的这笔订单"))
    assert view["ok"] is False and view["error"]["category"] == "permission_denied"
    assert view["error"]["message"] == "没有找到您的这笔订单" and view["data"] is None


def test_memory_projection_preserves_tool_message_pair_and_unicode():
    """投影丢失调用配对、中文转义或修改原消息会失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import _model_message
    result = ToolResult.success({"ticket_no": "T-real", "status": "pending", "description": "问题" * 500})
    message = ToolMessage(content=result.model_dump_json(), tool_call_id="call-1", id="db:2")
    original = message.content
    projected = _model_message(message, policy=ResultPolicy(enum_labels={"status": {"pending": "待处理"}}),
                               max_tokens=150, estimator=TokenEstimator())
    payload = json.loads(projected.content)
    assert projected.tool_call_id == "call-1" and projected.id == "db:2"
    assert payload["data"]["ticket_no"] == "T-real" and "待处理" in projected.content
    assert "\\u" not in projected.content and message.content == original
