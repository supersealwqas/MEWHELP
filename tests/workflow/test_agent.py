"""真实ReAct子图的顺序、停止边界、只读范围和即时正文流测试。"""
import asyncio
import json
import pytest
from app.schemas.workflow import AgentDecision
from app.schemas.tools import ToolResult
from app.workflow.graph import build_workflow
from tests.workflow.graph_fakes import dependencies, turn, call
from tests.workflow.test_model import sdk_runtime, text_response, call_response, intent_response, understanding_response

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
    for item in deps.model.inputs[3:]:
        assert "SF1" in json.dumps(item["messages"], ensure_ascii=False)
    assert [m.tool_call_id for m in state["messages"] if m.type == "tool"] == ["a", "b"]
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
    assert state["pending_calls"] == [] and len(deps.model.inputs) == 3

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
    assert state["decision_count"] == 1 and len(deps.model.inputs) == 3
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
    assert state["stop_reason"] == "budget_exceeded" and len(deps.model.inputs) == 3
    assert any(fact in message.content for message in state["messages"] if message.type == "tool")

async def test_fifth_decision_can_answer_after_four_tools():
    """把四工具上限错当四次决策，提前阻止最终回答时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[*[AgentDecision(kind="tools", tool_calls=[call(n)]) for n in range(4)], AgentDecision(kind="answer")],
        results=[ToolResult.success({"status": "已发货"}) for _ in range(4)], answers=[["已查询完成"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert state["decision_count"] == 5 and state["tool_count"] == 4 and state["answer"] == "已查询完成"
    assert state["budget"]["model_calls"] == 8


@pytest.mark.parametrize("actions", [["create_ticket"], ["handoff"], ["handoff", "create_ticket"], []])
async def test_answer_receives_current_confirmed_suggestion_context(actions):
    """正文只接当前按钮的中文操作语义，不接可被复述的内部字段。参数：actions当前建议；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["其他"], decisions=[AgentDecision(kind="answer", suggested_actions=actions)], answers=[["请按页面提示确认"]])
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
    deps = dependencies(token_budget=80, intents=["其他"],
        decisions=[AgentDecision(kind="answer", suggested_actions=["create_ticket"])], answers=[["旧路径会越过新增说明预算"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("请建单"))
    assert state["stop_reason"] == "budget_exceeded"
    assert not any(item["stage"] == "answer" for item in deps.model.inputs)


@pytest.mark.parametrize("order_id,amount,query", [
    ("1001", 39.9, "包邮吗"), ("1005", 89.0, "多少钱，需要付邮费吗"),
])
@pytest.mark.parametrize("needs_recovery", [False, True], ids=["direct-control", "recover-missing-control"])
async def test_followup_recovers_decision_format_without_losing_order_or_policy(
        sdk_runtime, order_id, amount, query, needs_recovery):
    """正常订单追问因一次内部格式错误被打断、丢历史或泄漏失败正文时失败。

    参数：sdk_runtime仅替换外部HTTP；order_id、amount、query来自历史失败场景；needs_recovery是否先模拟网关违约。
    返回：None，核对真实适配器、根图、检索注入、正文流及完成状态。
    异常：AssertionError，恢复、事实保留或调用边界不符合预期。
    """
    from app.workflow.model import WorkflowModelAdapter
    from tests.workflow.graph_fakes import ranked

    policy = "单笔实付满99元包邮，未满收6元基础运费；会员免运费权益可能减免。"
    order_result = ToolResult.success({"order_id": order_id, "product": "猫粮",
        "quantity": 1, "unit_price": amount, "total_amount": amount,
        "status": "已发货" if order_id == "1001" else "待发货"})
    state = turn(query)
    state["history_turns"] = [{"turn_id": "previous-order", "turn_status": "completed", "messages": [
        {"role": "user", "content": f"{order_id}订单到哪里了"},
        {"role": "assistant", "content": "", "tool_calls": [call("previous-call", order_id=order_id)]},
        {"role": "tool", "tool_call_id": "previous-call", "content": order_result.model_dump_json()},
        {"role": "assistant", "content": f"已查询订单{order_id}的状态。"},
    ]}]
    invalid_raw = "内部格式失败标记：订单已有金额，请直接作答。"
    answer = f"订单{order_id}的商品金额为{amount}元；普通运费规则见[1]，实际运费还需核对减免。"
    sdk_runtime.responses.extend([
        understanding_response(query), intent_response("商品咨询"),
        *([text_response(invalid_raw)] if needs_recovery else []),
        call_response("submit_decision", '{"kind":"answer","suggested_actions":[]}'), text_response(answer),
    ])
    deps = dependencies(ranked=[ranked(answer=policy)])
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    events, completed = [], None
    async for namespace, mode, value in build_workflow(deps, checkpointer=None).astream(
            state, stream_mode=["custom", "values"], subgraphs=True):
        if mode == "custom":
            events.append(value)
        elif not namespace:
            completed = value

    assert completed["stop_reason"] == "answer"
    assert completed["answer"] == answer and completed["gate"]["passed"]
    assert completed["raw_query"] == query and completed["tool_count"] == 0
    assert deps.executor.calls == [] and deps.chat.calls == []
    assert not any(event["event"] == "tool_status" for event in events)
    assert len(sdk_runtime.requests) == 4 + needs_recovery
    # 检查真实SDK发出的两次决策与正文输入，而非替身自行生成的期望。
    for request in sdk_runtime.requests[2:]:
        messages = request["messages"]
        assert any(m["role"] == "user" and m["content"] == query for m in messages)
        assert any(m["role"] == "tool" and json.loads(m["content"])["data"]["order_id"] == order_id
                   and json.loads(m["content"])["data"]["total_amount"] == amount for m in messages)
        assert policy in json.dumps(messages, ensure_ascii=False)
        assert invalid_raw not in json.dumps(messages, ensure_ascii=False)
    assert "".join(e["data"]["delta"] for e in events if e["event"] == "delta") == answer
    assert deps.chat.finished[0]["workflow_metadata"]["stop_reason"] == "answer"
    assert [r["stage"] for r in completed["budget"]["records"]] == [
        "understand", "classify", "decide", *(["decide"] if needs_recovery else []), "answer"]


@pytest.mark.parametrize("needs_recovery", [False, True], ids=["direct-control", "recover-missing-control"])
async def test_followup_recovery_after_order_and_logistics_never_reexecutes_tools(sdk_runtime, needs_recovery):
    """会话89的1002追问已查到工具事实后，格式恢复不得重跑工具或丢结果。

    参数：sdk_runtime只替换外部HTTP；needs_recovery是否先模拟网关违约；图和适配器使用真实实现。
    返回：None，核对两次工具只执行一次、恢复后完成正文及独立预算记录。
    异常：AssertionError，发生额外执行、事实丢失或恢复失败。
    """
    from app.workflow.model import WorkflowModelAdapter

    state = turn("1002呢")
    state["history_turns"] = [{"turn_id": "previous-route", "turn_status": "completed", "messages": [
        {"role": "user", "content": "1001是什么物品？从哪里到哪里"},
        {"role": "assistant", "content": "1001的物品是猫粮，从杭州发往宁波。"},
    ]}]
    order_call = call_response(raw='{"order_id":"1002"}')
    order_call[0]["choices"][0]["delta"]["tool_calls"][0]["id"] = "lookup-order"
    logistics_call = call_response("query_logistics", '{"tracking_no":"SF2"}')
    logistics_call[0]["choices"][0]["delta"]["tool_calls"][0]["id"] = "lookup-logistics"
    answer = "1002的商品是猫粮，从杭州发往乌鲁木齐，目前派送中。"
    sdk_runtime.responses.extend([
        understanding_response("1002呢"), intent_response("订单"), order_call, logistics_call,
        *([text_response("内部尚未遵循格式，已有事实足以回答。")] if needs_recovery else []),
        call_response("submit_decision", '{"kind":"answer","suggested_actions":[]}'), text_response(answer),
    ])
    deps = dependencies(results=[
        ToolResult.success({"order_id": "1002", "product": "猫粮", "tracking_no": "SF2",
            "shipping_origin": "浙江省杭州市", "shipping_destination": "新疆维吾尔自治区乌鲁木齐市"}),
        ToolResult.success({"order_id": "1002", "carrier": "顺丰速运", "tracking_no": "SF2",
            "status": "派送中", "latest_node": "包裹由目的地配送站安排派送"}),
    ])
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    completed = await build_workflow(deps, checkpointer=None).ainvoke(state)

    assert completed["stop_reason"] == "answer" and completed["answer"] == answer
    assert [(c["name"], c["args"]) for c in deps.executor.calls] == [
        ("query_order", {"order_id": "1002"}), ("query_logistics", {"tracking_no": "SF2"})]
    assert completed["tool_count"] == 2 and len(deps.chat.results) == 2
    assert len(sdk_runtime.requests) == 6 + needs_recovery
    assert [c["name"] for c in deps.chat.calls] == ["query_order", "query_logistics"]
    for request in sdk_runtime.requests[4:]:
        facts = [json.loads(m["content"])["data"] for m in request["messages"] if m["role"] == "tool"]
        assert len(facts) == 2
        assert facts[0]["shipping_destination"] == "新疆维吾尔自治区乌鲁木齐市"
        assert facts[1]["status"] == "派送中"
    assert sum(r["stage"] == "decide" for r in completed["budget"]["records"]) == 3 + needs_recovery


@pytest.mark.parametrize("query,kind,actions,answer", [
    ("帮我查订单", "clarify", [], "请提供订单号。"),
    ("请创建咨询工单", "answer", ["create_ticket"], "请点击创建工单按钮，在弹窗中确认。"),
])
async def test_control_decision_preserves_clarification_and_confirmation_actions(
        sdk_runtime, query, kind, actions, answer):
    """终止工具必须保留追问和建单确认入口，不能变成实际查询或写操作。

    参数：sdk_runtime外部HTTP夹具；query用户请求；kind结束类别；actions建议动作；answer预期正文。
    返回：None，核对真实根图结束记录、建议快照、正文输入和执行次数。
    异常：AssertionError，建议丢失、控制调用进入工具历史或发生额外调用。
    """
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.extend([
        understanding_response(query), intent_response("订单"),
        call_response("submit_decision", json.dumps({"kind": kind, "suggested_actions": actions})),
        text_response(answer),
    ])
    deps = dependencies()
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    completed = await build_workflow(deps, checkpointer=None).ainvoke(turn(query))

    assert completed["stop_reason"] == kind and completed["answer"] == answer
    assert [item["kind"] for item in completed["suggested_actions"]] == actions
    assert all(item["question"] == query for item in completed["suggested_actions"])
    assert deps.executor.calls == deps.chat.calls == deps.chat.results == []
    assert completed["tool_count"] == 0 and len(sdk_runtime.requests) == 4
    assert "submit_decision" not in json.dumps(sdk_runtime.requests[-1]["messages"])
    assert ("创建工单" in sdk_runtime.requests[-1]["messages"][-1]["content"]) == bool(actions)


async def test_control_schema_cannot_bypass_current_message_budget():
    """当前问题只够容纳旧业务schema时，新增控制schema必须在调用模型前触发额度检查。

    参数：无，按当前提示和旧公开工具定义计算临界预算。
    返回：None，核对图在完整输入超额时停止且没有调用决策模型。
    异常：AssertionError，遗漏控制schema导致预算外请求。
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from app.core.memory import _estimate_message_tokens, _estimate_token_units
    from app.tools.registry import build_agent_registry
    from app.workflow.prompts import DECISION_PROMPT
    from app.workflow.agent import build_agent_graph

    query = "查询订单1001"
    old_schemas = [convert_to_openai_tool(tool) for tool in build_agent_registry().all()]
    old_input_units = (_estimate_message_tokens([SystemMessage(content=DECISION_PROMPT), HumanMessage(content=query)])
                       + _estimate_token_units(old_schemas))
    deps = dependencies(request_token_budget=old_input_units + 1,
        decisions=[AgentDecision(kind="answer")], answers=[["不应在预算外生成的正文"]])
    state = turn(query)
    state["messages"] = [{"role": "user", "content": query}]
    # 此用例检查Agent的工具schema预算，不让独立分类提示的长度抢先触发边界。
    completed = await build_agent_graph(deps).ainvoke(state)

    assert completed["stop_reason"] == "budget_exceeded"
    assert not any(item["stage"] == "decide" for item in deps.model.inputs)
