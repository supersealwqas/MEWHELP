"""真实ReAct子图的顺序、停止边界、只读范围和即时正文流测试。"""
import asyncio
import json
import pytest
from app.schemas.workflow import AgentDecision
from app.schemas.tools import ToolResult
from app.workflow.graph import build_workflow
from tests.workflow.graph_fakes import dependencies, turn, call

async def test_order_then_logistics_uses_complete_pairs_and_read_only_registry():
    """跳过订单事实、乱序执行或暴露写工具时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call("a", order_id="1001")]),
        AgentDecision(kind="tools", tool_calls=[call("b", "query_logistics", tracking_no="SF1")]),
        AgentDecision(kind="answer")], results=[ToolResult.success({"tracking_no": "SF1"}), ToolResult.success({"status": "运输中"})], answers=[["运输中"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    from app.tools.registry import build_agent_registry, build_ticket_registry
    assert [c["name"] for c in deps.executor.calls] == ["query_order", "query_logistics"]
    assert [t.name for t in build_agent_registry().all()] == ["query_order", "query_logistics", "query_product"]
    assert [t.name for t in build_ticket_registry().all()] == ["create_ticket"]
    assert all({t.name for t in item["tools"]} == {"query_order", "query_logistics", "query_product"} for item in deps.model.inputs if item["stage"] == "decide")
    for item in deps.model.inputs[2:]:
        assert "SF1" in json.dumps(item["messages"], ensure_ascii=False)
    assert [m["tool_call_id"] for m in state["messages"] if m["role"] == "tool"] == ["a", "b"]
    assert deps.chat.order == ["calls", "result", "calls", "result", "finish"]

async def test_first_delta_arrives_before_answer_generator_finishes():
    """将完整答案缓存后伪流式发送时失败。参数：无；返回：None；异常：断言或超时。"""
    deps = dependencies(decisions=[AgentDecision(kind="answer")])
    first_delta_received, answer_completed = asyncio.Event(), asyncio.Event()
    async def stream(messages, ledger):
        """参数：messages和ledger；返回：受控正文流；异常：等待超时。"""
        ledger.reserve("answer", 10, 4096)
        yield "第一段"
        await asyncio.wait_for(first_delta_received.wait(), 2)
        yield "第二段"
        answer_completed.set()
    deps.model.stream_answer = stream
    events = []
    async for namespace, event in build_workflow(deps, checkpointer=None).astream(turn(), stream_mode="custom", subgraphs=True):
        events.append(event)
        if event["event"] == "delta":
            assert namespace[0].startswith("main_agent:")
            if not first_delta_received.is_set():
                assert not answer_completed.is_set()
                first_delta_received.set()
    assert first_delta_received.is_set() and answer_completed.is_set()
    assert "".join(e["data"]["delta"] for e in events if e["event"] == "delta") == "第一段第二段"

async def test_order_rejection_skips_remaining_logistics_and_never_calls_model_again():
    """订单归属拒绝后继续查物流或改写固定拒绝时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call("a"), call("b", "query_logistics", tracking_no="SF2")])],
        results=[ToolResult.failed("order_not_owned", "不可查")])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert len(deps.executor.calls) == 1
    assert state["answer"] == "没有找到您的这笔订单，请确认订单号或物流单号后重试。"
    assert [r[0] for r in deps.chat.results] == ["a", "b"]
    assert state["pending_calls"] == [] and len(deps.model.inputs) == 2

@pytest.mark.parametrize("result", [ToolResult.failed("TOOL_TIMEOUT", "超时"), ConnectionError("故障")])
async def test_tool_failure_pairs_unexecuted_remainder(result):
    """工具故障后状态仍查询中、剩余申请悬空或继续调用时失败。参数：result受控错误；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call("a"), call("b")])], results=[result])
    frames, state = [], None
    async for namespace, mode, payload in build_workflow(deps, checkpointer=None).astream(
            turn(), stream_mode=["custom", "values"], subgraphs=True):
        if mode == "values" and not namespace:
            state = payload
        elif mode == "custom" and payload["event"] == "tool_status":
            frames.append(payload["data"])
    assert len(deps.executor.calls) == 1
    assert [r[0] for r in deps.chat.results] == ["a", "b"]
    assert all(not result.ok for _, result in deps.chat.results)
    assert state["stop_reason"] == "tool_failed" and state["pending_calls"] == []
    assert [(frame["tool_call_id"], frame["status"]) for frame in frames] == [
        ("a", "running"), ("a", "failed"), ("b", "failed")]
    assert frames[1]["label"] == "未完成" and frames[2]["label"] == "未执行"

@pytest.mark.parametrize("second", [call("a"), call("b", order_id=" a ")], ids=["same-id", "same-arguments"])
async def test_duplicate_id_or_signature_never_executes_twice(second):
    """跨决策重复ID或规范化同参数再次执行时失败。参数：second重复申请；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call("a")]), AgentDecision(kind="tools", tool_calls=[second])],
        results=[ToolResult.success({"status": "已发货"})])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert len(deps.executor.calls) == 1 and state["stop_reason"] in {"duplicate_tool_call", "invalid_decision"}
    assert state["pending_calls"] == []

async def test_four_tool_limit_pairs_remaining_without_fifth_execution():
    """同批申请绕过四工具上限时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call(n) for n in range(5)])],
        results=[ToolResult.success({"status": "已发货"}) for _ in range(4)])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert len(deps.executor.calls) == 4 and len(deps.chat.results) == 5
    assert state["tool_count"] == 4 and state["stop_reason"] == "tool_call_limit"

async def test_decision_limit_stops_before_next_model_request():
    """非默认决策上限失效或多一次决策时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(workflow_max_decisions=1, decisions=[AgentDecision(kind="tools", tool_calls=[call("a")])],
        results=[ToolResult.success({"status": "已发货"})])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert state["decision_count"] == 1 and len(deps.model.inputs) == 2
    assert state["stop_reason"] == "decision_limit"

async def test_unknown_or_write_tool_is_not_executed():
    """绕过适配器的写申请进入执行器时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call("a", "create_ticket", description="投诉", ticket_type="投诉")])])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert deps.executor.calls == [] and state["stop_reason"] == "invalid_decision"

async def test_large_tool_fact_stops_without_truncation_or_answer_call():
    """大工具事实被截短并继续模型请求时失败。参数：无；返回：None；异常：断言失败。"""
    fact = "事实" * 10000
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call("a")])], results=[ToolResult.success({"details": fact})])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert state["stop_reason"] == "budget_exceeded" and len(deps.model.inputs) == 2
    assert fact in state["messages"][-1]["content"]

async def test_fifth_decision_can_answer_after_four_tools():
    """把四工具上限错当四次决策，提前阻止最终回答时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[*[AgentDecision(kind="tools", tool_calls=[call(n)]) for n in range(4)], AgentDecision(kind="answer")],
        results=[ToolResult.success({"status": "已发货"}) for _ in range(4)], answers=[["已查询完成"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert state["decision_count"] == 5 and state["tool_count"] == 4 and state["answer"] == "已查询完成"
    assert state["budget"]["model_calls"] == 7


@pytest.mark.parametrize("actions", [["create_ticket"], ["handoff"], ["handoff", "create_ticket"], []])
async def test_answer_receives_current_confirmed_suggestion_context(actions):
    """正文只接当前按钮的中文操作语义，不接可被复述的内部字段。参数：actions当前建议；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["售后"], decisions=[AgentDecision(kind="answer", suggested_actions=actions)], answers=[["请按页面提示确认"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("设备故障，请创建售后工单"))
    actual = next(item["messages"] for item in deps.model.inputs if item["stage"] == "answer")
    description = actual[-1]["content"]
    assert actual[-1]["role"] == "system"
    assert {action["kind"] for action in state["suggested_actions"]} == set(actions)
    assert ("创建工单" in description) == ("create_ticket" in actions)
    assert ("转人工" in description) == ("handoff" in actions)
    if "create_ticket" in actions:
        assert "弹窗" in description and "确认" in description
    if not actions:
        assert "没有" in description and "按钮" in description
    all_input = json.dumps(actual, ensure_ascii=False)
    assert not any(field in all_input for field in
        ("reply_kind", "available_actions", "action_instructions", "create_ticket", "handoff", "user_id", "action_id"))


async def test_answer_action_instructions_cannot_bypass_message_budget():
    """新增按钮说明也必须过现有输入预算，不能预算外加给模型。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(token_budget=80, intents=["售后"],
        decisions=[AgentDecision(kind="answer", suggested_actions=["create_ticket"])], answers=[["旧路径会越过新增说明预算"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("请建单"))
    assert state["stop_reason"] == "budget_exceeded"
    assert not any(item["stage"] == "answer" for item in deps.model.inputs)
