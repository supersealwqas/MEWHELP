"""CH08 独审回归：先按真实历史层筛选，再投影实际需要发送的工具结果。"""

from copy import deepcopy
import json

import pytest

from app.core.context_budget import ContextBudgetExceeded
from app.schemas.history import StoredMessage, StoredTurn
from app.schemas.tools import ToolResult
from app.tools.contracts import ResultPolicy, ToolDefinition, ToolIdentity
from app.tools.registry import ToolRegistry
from app.workflow.context import assemble_context, history_snapshot
from app.workflow.prompts import UNDERSTANDING_PROMPT
from tests.workflow.graph_fakes import dependencies, turn


def large_tool_turn(*, count=1000, status="completed"):
    """建立关键事实无法按单工具上限缩短的真实配对旧轮。

    参数：count 商品数量、status 轮次状态；返回：StoredTurn；异常：输入数据校验错误。
    """
    result = ToolResult.success({"items": [{"product_id": f"product-{index}", "price": index}
                                          for index in range(count)]})
    return StoredTurn(turn_id="old-turn", turn_status=status, messages=[
        StoredMessage(id=1, role="user", content="列出商品"),
        StoredMessage(id=2, role="assistant", content="", tool_calls=[
            {"id": "old-call", "name": "query_product", "args": {"keyword": "猫粮"}}]),
        StoredMessage(id=3, role="tool", tool_call_id="old-call", content=result.model_dump_json()),
        StoredMessage(id=4, role="assistant", content="商品信息已经查询。")])


def followup_state(old):
    """建立保留完整旧回执的新问题状态。

    参数：old 持久化旧轮；返回：新轮 State；异常：输入校验错误。
    """
    state = turn("现在查询会员规则", turn_id="new-turn")
    state["history_turns"] = [old.model_dump(mode="json")]
    return state


def test_summarized_large_tool_does_not_block_next_question():
    """已被摘要覆盖的旧关键事实仍被全量投影而阻止后续提问时失败。

    参数：无；返回：None；异常：上下文预算或断言失败。
    """
    state = followup_state(large_tool_turn())
    original = deepcopy(state["history_turns"])
    settings = dependencies().settings
    stored = {"summary_upto_msg_id": 4, "layer1_from_msg_id": 4,
              "summaries": [{"seq": 1, "content": "曾查询商品信息。", "from_msg_id": 1, "upto_msg_id": 4}]}
    state["history_context"] = history_snapshot(state, settings, stored)
    messages = assemble_context(state, settings, UNDERSTANDING_PROMPT)
    assert state["history_context"]["summary"] == "曾查询商品信息。"
    assert state["history_context"]["layer1"] == state["history_context"]["layer2"] == []
    assert messages[0]["content"] == "现在查询会员规则"
    assert "曾查询商品信息。" in messages[1]["content"]
    assert state["history_turns"] == original


@pytest.mark.parametrize("count,anchor", [(1000, 0), (4000, 0), (1000, 4)])
def test_old_large_tool_demotes_as_whole_turn(count, anchor):
    """旧完整轮的关键事实过大时应走半压标识路径，不能在分层前永久拒绝。

    参数：count 结果规模、anchor 原半压边界；返回：None；异常：上下文或断言失败。
    """
    state = followup_state(large_tool_turn(count=count))
    original = deepcopy(state["history_turns"])
    snapshot = history_snapshot(state, dependencies().settings,
        {"summary_upto_msg_id": 0, "layer1_from_msg_id": anchor, "summaries": []})
    assert snapshot["layer1"] == [] and snapshot["layer1_from_msg_id"] == 4
    assert snapshot["previous_layer1_from_msg_id"] == anchor
    assert snapshot["summary_upto_msg_id"] == 0 and snapshot["summary"] == ""
    assert snapshot["demoted_turns"] == (1 if anchor == 0 else 0)
    receipt = next(row for row in snapshot["layer2"] if row["role"] == "tool")
    assert "工具结果已省略" in receipt["content"] and "query_product" in receipt["content"]
    assert "成功" in receipt["content"] and "原始消息ID=3" in receipt["content"]
    assert receipt["tool_call_id"] == snapshot["layer2"][1]["tool_calls"][0]["id"] == "old-call"
    assert state["history_turns"] == original


def test_summary_anchor_still_cannot_split_large_tool_turn():
    """为绕开大结果而提前删历史，导致非法摘要锚点漏检时失败。

    参数：无；返回：None；异常：预期的 ValueError。
    """
    with pytest.raises(ValueError, match="锚点不能切开"):
        history_snapshot(followup_state(large_tool_turn()), dependencies().settings,
            {"summary_upto_msg_id": 3, "layer1_from_msg_id": 4, "summaries": []})


def test_small_half_layer_result_keeps_local_enum_labels():
    """只投影原文层而让半压层小回执泄漏技术枚举时失败。

    参数：无；返回：None；异常：上下文或断言失败。
    """
    async def unused(arguments, context, scope):
        """提供合法定义但不执行工具。

        参数：公开参数、身份、范围；返回：空成功结果；异常：无。
        """
        return ToolResult.success({})

    registry = ToolRegistry()
    registry.register_definition(ToolDefinition(ToolIdentity("builtin", None, "query_product"), "查询商品",
        {"type": "object", "properties": {}}, unused, "read",
        ResultPolicy(enum_labels={"status": {"available": "可购买"}})))
    old = large_tool_turn(count=1)
    old.messages[2].content = ToolResult.success({"product_id": "product-1", "status": "available"}).model_dump_json()
    state = followup_state(old)
    snapshot = history_snapshot(state, dependencies().settings,
        {"summary_upto_msg_id": 0, "layer1_from_msg_id": 4, "summaries": []}, registry=registry)
    receipt = next(row for row in snapshot["layer2"] if row["role"] == "tool")
    assert json.loads(receipt["content"])["data"] == {"product_id": "product-1", "status": "可购买"}
    assert "available" in old.messages[2].content


def test_unfinished_large_tool_turn_cannot_be_demoted():
    """为了通过预算而把未完成旧轮强行降级时失败。

    参数：无；返回：None；异常：预期的 ContextBudgetExceeded。
    """
    with pytest.raises(ContextBudgetExceeded):
        history_snapshot(followup_state(large_tool_turn(status="awaiting_input")), dependencies().settings)


def test_current_large_tool_facts_still_reject_before_model_input():
    """历史降级修复误把当前轮关键事实也截短或省略时失败。

    参数：无；返回：None；异常：预期的 ContextBudgetExceeded。
    """
    state = turn("列出商品", turn_id="old-turn")
    state["history_turns"] = [large_tool_turn().model_dump(mode="json")]
    from app.workflow.state import full_history_messages
    state["messages"] = full_history_messages(state["history_turns"])
    with pytest.raises(ContextBudgetExceeded, match="关键事实"):
        assemble_context(state, dependencies().settings, UNDERSTANDING_PROMPT)
