"""CH08 查询故障答复边界：当前可信配对决定固定正文，真实服务保持流与保存一致。"""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from langchain_core.messages import convert_to_messages
from langchain_core.tools import StructuredTool, ToolException

from app.schemas.tools import ToolResult
from app.schemas.workflow import AgentDecision, TicketResume
from app.tools.contracts import ToolIdentity, binding_dict
from app.tools.registry import build_registry
from app.tools.mcp import MCPToolManager
from app.tools.permissions import ToolPermissions
from app.workflow.state import checkpoint_thread_id, message_row
from tests.workflow.ch08_ticket_fakes import ticket_runtime
from tests.workflow.graph_fakes import call, turn
from tests.workflow.service_fakes import collect, service_runtime
from tests.workflow.test_ch08_ticket_interrupt import rows, ticket_call


FAULT_TEXT = "本次查询未能完成，暂时无法核实相关信息。请稍后重试。"


def query_state(*, result=None, name="query_faq", identifier="read-1", turn_id="now"):
    """构造生产消息格式的本轮工具配对，不把用户文字解析为结果。

    参数：result 明确工具载荷，name 注册名称，identifier 调用键，turn_id 服务器轮次。
    返回：状态与真实工具注册表；异常：工具名不存在。
    """
    registry = build_registry()
    state = turn("工单有什么用？", turn_id=turn_id)
    append_result(state, registry, identifier, name,
        result if result is not None else ToolResult.failed("TOOL_TIMEOUT", "查询超时"))
    return state, registry


def append_result(state, registry, identifier, name, result):
    """向测试状态追加服务器调用和结果，保留真实绑定结构。

    参数：state 本轮状态，registry 真实注册表，identifier/name 调用身份，result 结果或原始载荷。
    返回：None；异常：工具名不存在或载荷不能序列化。
    """
    definition = next(item for item in registry.snapshot() if item.name == name)
    request = call(identifier, name, keyword="工单")
    payload = result.model_dump(mode="json") if isinstance(result, ToolResult) else result
    state["messages"].extend([
        message_row(state["context"]["turn_id"], "assistant", "", calls=[request],
            bindings={identifier: binding_dict(definition.binding)}),
        message_row(state["context"]["turn_id"], "tool", json.dumps(payload, ensure_ascii=False), call_id=identifier),
    ])
    state.setdefault("tool_call_facts", {})[identifier] = {"turn_id": state["context"]["turn_id"],
        "binding": binding_dict(definition.binding), "access": definition.access}


async def test_service_query_fault_skips_model_and_saves_exact_stream(tmp_path):
    """故障后仍调用正文模型、流入编造流程或保存内容不同于流时失败。

    参数：tmp_path 独占 SQLite 目录；返回：None；异常：断言或数据库错误。
    """
    request = call("faq-1", "query_faq", keyword="工单")
    async with service_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[request]),
            AgentDecision(kind="answer")], results=[ToolResult.failed("TOOL_TIMEOUT", "查询超时")],
            answers=[["工单会转交专员，保证跟进处理。"]]) as (service, deps, factory):
        events = await collect(service, "工单有什么用？", session="query-fault")
        streamed = "".join(event.data.get("delta", "") for event in events)
        assert events[-1].event == "done"
        assert streamed == FAULT_TEXT
        assert "answer" not in [item["stage"] for item in deps.model.inputs]
        current = await service.get_current_actions("query-fault", "demo-user")
        assert current.answer == streamed and current.turn_status == "completed"
        assert await rows(factory, "SELECT content FROM messages WHERE role='assistant' AND tool_calls IS NULL") == [(streamed,)]
        assert await rows(factory, "SELECT raw_question FROM low_confidence_questions") == []


async def test_committed_ticket_does_not_hide_read_fault_or_lose_receipt(tmp_path):
    """真实建单成功不得阻止读故障兜底，固定答复不得丢票号或重建工单。

    参数：tmp_path 独占 SQLite 目录；返回：None；异常：断言或数据库错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="tools", tool_calls=[call("faq-2", "query_faq", keyword="破损")]),
            AgentDecision(kind="answer")], answers=[["后续一定会由专员处理。"]]) as (service, deps, factory, _):
        deps.search.error = TimeoutError("受控检索故障")
        initial = await collect(service, "帮我建个工单，订单1001收到时破损", session="ticket-query-fault")
        preview = next(event.data for event in initial if event.event == "confirm_ticket")
        conversation_id = (await rows(factory, "SELECT id FROM conversations"))[0][0]
        graph_config = {"configurable": {"thread_id": checkpoint_thread_id(conversation_id)}}
        paused = await service.graph.aget_state(graph_config)
        assert paused.values["tool_call_facts"]["ticket-1"]["access"] == "write"
        events = [event async for event in service.stream_resume(TicketResume(type="confirm_ticket",
            session_id="ticket-query-fault", user_id="demo-user", source_turn_id=preview["source_turn_id"],
            action_id=preview["action_id"], decision="confirm"))]
        assert events[-1].event == "done", [(event.event, event.data) for event in events]
        ticket_no = (await rows(factory, "SELECT ticket_no FROM tickets"))[0][0]
        streamed = "".join(event.data.get("delta", "") for event in events)
        assert streamed == f"工单已创建，工单号：{ticket_no}，状态：待处理。\n\n" + FAULT_TEXT
        assert "answer" not in [item["stage"] for item in deps.model.inputs]
        current = await service.get_current_actions("ticket-query-fault", "demo-user")
        assert current.answer == streamed and current.ticket_receipt.ticket_no == ticket_no
        assert await rows(factory, "SELECT ticket_no FROM tickets") == [(ticket_no,)]
        assert await rows(factory, "SELECT tool_name FROM tool_audit_logs WHERE tool_name='create_ticket'") == [("create_ticket",)]
        completed = await service.graph.aget_state(graph_config)
        assert {key: value["access"] for key, value in completed.values["tool_call_facts"].items()} == {
            "ticket-1": "write", "faq-2": "read"}


@pytest.mark.parametrize("results,requests,answer", [
    ([ToolResult.success({"products": []}, is_mock=True)], [call("p-1", "query_product", keyword="猫粮")], "没有匹配的商品。"),
    ([ToolResult.success({"order_id": "1001", "status": "已发货"}), ToolResult.failed("TOOL_TIMEOUT", "查询超时")],
        [call("o-1", order_id="1001"), call("p-2", "query_product", keyword="猫粮")], "订单1001已发货；商品查询暂未完成。"),
])
async def test_successful_query_facts_keep_normal_answer_path(tmp_path, results, requests, answer):
    """成功空结果或成功失败并存被强制覆盖，导致可用事实丢失时失败。

    参数：tmp_path 独占目录，results 工具结果，requests 对应调用，answer 有依据的脚本正文。
    返回：None；异常：断言或数据库错误。
    """
    decisions = [AgentDecision(kind="tools", tool_calls=[request]) for request in requests]
    async with service_runtime(tmp_path, decisions=[*decisions, AgentDecision(kind="answer")], results=results,
            answers=[[answer]]) as (service, deps, _):
        events = await collect(service, "查订单1001和猫粮", session="query-success")
        assert events[-1].event == "done"
        assert "".join(event.data.get("delta", "") for event in events) == answer
        assert [item["stage"] for item in deps.model.inputs].count("answer") == 1
        assert (await service.get_current_actions("query-success", "demo-user")).answer == answer


@pytest.mark.parametrize("as_messages", [False, True])
def test_policy_uses_current_bound_read_fault(as_messages):
    """遗漏字典或检查点消息中的可信读故障时失败。

    参数：as_messages 是否使用 LangChain 消息；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state()
    if as_messages:
        state["messages"] = convert_to_messages(state["messages"])
    before = deepcopy(state)
    assert query_fault_answer(state, registry=registry) == FAULT_TEXT
    assert state == before


@pytest.mark.parametrize("result", [
    ToolResult.success({"products": []}, is_mock=True),
    ToolResult.success({"found": True, "warranty_status": "in_warranty"}, is_mock=True),
    {"ok": False, "error": {"code": "INVALID_TOOL_ARGUMENTS", "category": "invalid_arguments", "message": "参数错误"}},
    {"ok": False, "error": {"code": "order_not_owned", "category": "permission_denied", "message": "不能读取"}},
    {"ok": False, "error": {"code": "NOT_FOUND", "category": "not_found", "message": "没有记录"}},
])
def test_policy_does_not_relabel_non_fault_outcomes(result):
    """把成功查询、空结果、参数或权限拒绝误说成系统故障时失败。

    参数：result 非故障严格结果；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state(result=result)
    assert query_fault_answer(state, registry=registry) is None


@pytest.mark.parametrize("first,expected", [
    (ToolResult.failed("order_not_owned", "没有权限"), None),
    (ToolResult.failed("TOOL_TIMEOUT", "查询超时"), FAULT_TEXT),
])
def test_skipped_remainder_is_not_itself_a_query_fault(first, expected):
    """把编排层未执行的剩余申请误当作独立查询故障时失败。

    参数：first 已执行项的真实结果，expected 手工确定的答复分支；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state(result=first)
    append_result(state, registry, "skipped-1", "query_product",
        ToolResult.failed("TOOL_SKIPPED", "前序申请未完成，该申请未执行"))
    assert query_fault_answer(state, registry=registry) == expected


@pytest.mark.parametrize("success", ["read", "empty", "evidence", "order", "write"])
def test_policy_preserves_read_facts_but_not_write_as_query_success(success):
    """丢失已有查询事实，或让成功写入掩盖查询故障时失败。

    参数：success 当前可信成功来源；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state()
    if success in {"read", "empty", "write"}:
        append_result(state, registry, "success-1", "create_ticket" if success == "write" else "query_product",
            ToolResult.success({"ticket_no": "T-1"} if success == "write" else {"products": [] if success == "empty" else [{"name": "猫粮"}]}))
    elif success == "evidence":
        state.update(evidence=[{"n": 1, "text": "已核实政策"}], gate={"passed": True})
    else:
        state["order_data"] = {"order_id": "1001", "status": "已发货"}
    assert query_fault_answer(state, registry=registry) == (FAULT_TEXT if success == "write" else None)


@pytest.mark.parametrize("scenario", ["no_query", "old_fault", "orphan", "user_json", "wrong_turn", "no_turn", "missing_anchor", "before_user", "duplicate_result", "duplicate_call", "result_before_call"])
def test_policy_ignores_untrusted_or_noncurrent_messages(scenario):
    """把旧轮、孤立回执、伪造文字或不唯一配对误认成本轮事实时失败。

    参数：scenario 被破坏的消息边界；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state()
    messages = state["messages"]
    if scenario == "no_query":
        state["messages"] = messages[:1]
    elif scenario == "old_fault":
        fresh = turn("谢谢", turn_id="next")
        fresh["messages"] = [*messages, *fresh["messages"]]
        state = fresh
    elif scenario == "orphan":
        state["messages"] = [messages[0], messages[2]]
    elif scenario == "user_json":
        state = turn(messages[2]["content"], turn_id="now")
    elif scenario == "wrong_turn":
        messages[2]["additional_kwargs"]["turn_id"] = "other"
    elif scenario == "no_turn":
        messages[2]["additional_kwargs"].pop("turn_id")
    elif scenario == "missing_anchor":
        state["current_message_id"] = "unknown"
    elif scenario == "before_user":
        state["messages"] = [messages[1], messages[2], messages[0]]
    elif scenario == "duplicate_result":
        state["messages"].append(deepcopy(messages[2]))
    elif scenario == "duplicate_call":
        state["messages"].insert(2, deepcopy(messages[1]))
    else:
        state["messages"] = [messages[0], messages[2], messages[1]]
    assert query_fault_answer(state, registry=registry) is None


@pytest.mark.parametrize("payload", [
    {"ok": "false", "error": {"code": "TOOL_TIMEOUT", "category": "fault", "message": "超时"}},
    {"ok": False, "error": {"code": "UNKNOWN", "message": "超时"}},
    {"ok": False, "error": {"code": "UNKNOWN", "category": "made_up", "message": "超时"}},
    {"ok": True, "error": {"code": "TOOL_TIMEOUT", "category": "fault", "message": "超时"}},
    {"ok": False, "data": {"products": []}, "error": {"code": "TOOL_TIMEOUT", "category": "fault", "message": "超时"}},
    {"ok": False, "error": None},
])
def test_policy_rejects_malformed_or_inferred_fault_classification(payload):
    """容忍错误布尔、矛盾结果或仅凭错误文字/未知码推断故障时失败。

    参数：payload 非严格结果；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state(result=payload)
    assert query_fault_answer(state, registry=registry) is None


@pytest.mark.parametrize("change", ["missing", "wrong_name", "schema", "unloaded", "access", "bad_binding"])
def test_policy_does_not_guess_changed_or_missing_tool_identity(change):
    """原绑定损坏仍放行，或可变注册表撤销已保存读故障事实时失败。

    参数：change 绑定损坏或注册表变化；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state()
    metadata = state["messages"][1]["additional_kwargs"]
    if change == "missing":
        metadata.pop("tool_bindings")
    elif change == "wrong_name":
        state["messages"][1]["tool_calls"][0]["name"] = "query_order"
    elif change == "bad_binding":
        metadata["tool_bindings"]["read-1"] = {"identity": {"source": "bad"}}
    else:
        original = next(item for item in registry.snapshot() if item.name == "query_faq")
        external = replace(original, identity=ToolIdentity("mcp", "dynamic", "lookup_example"))
        registry.replace_source("dynamic", [external])
        state, _ = query_state()
        state["messages"] = state["messages"][:1]
        append_result(state, registry, "read-1", external.name, ToolResult.failed("TOOL_TIMEOUT", "超时"))
        changed = replace(external, access="write") if change == "access" else replace(external,
            parameters={**external.parameters, "description": "更新后的参数版本"})
        registry.replace_source("dynamic", [] if change == "unloaded" else [changed])
    expected = FAULT_TEXT if change in {"schema", "unloaded", "access"} else None
    assert query_fault_answer(state, registry=registry) == expected


@pytest.mark.parametrize("change", ["unloaded", "schema", "access"])
async def test_mcp_refresh_cannot_erase_completed_read_fault(tmp_path, change):
    """真实MCP执行已保存故障，后续发现或配置变化重新放行正文模型时失败。

    参数：tmp_path 独占SQLite目录，change 后续来源变化；返回：None；异常：断言或数据库错误。
    """
    permission_path = tmp_path / "mcp-permissions.json"
    config = {"version": 1, "tools": [{"source": "mcp", "server": "docs", "name": "lookup_fault",
        "enabled": True, "access": "read", "ownership": "none"}]}
    permission_path.write_text(json.dumps(config), encoding="utf-8")
    permissions = ToolPermissions(permission_path)

    async def broken_query(keyword: str) -> str:
        """模拟一次已发送的外部查询故障。参数：keyword 公开查询词；返回：无；异常：ToolException。"""
        if change == "access":
            config["tools"][0]["access"] = "write"
            permission_path.write_text(json.dumps(config), encoding="utf-8")
        raise ToolException("受控外部查询故障")

    async def changed_query(keyword: str, detail: str = "") -> str:
        """第二次发现仅改变参数版本，不再实际查询。参数：keyword/detail 公开参数；返回：无；异常：AssertionError。"""
        raise AssertionError("已经失败的查询不得重新执行")

    class DiscoveryClient:
        """只替代远端发现，保留生产MCP管理器与执行器。"""

        def __init__(self):
            """记录发现次数。参数：无；返回：None；异常：无。"""
            self.calls = 0

        async def get_tools(self, *, server_name):
            """首次返回读工具，后续模拟变化。参数：server_name 来源名；返回：工具清单；异常：ConnectionError。"""
            self.calls += 1
            if self.calls > 1 and change == "unloaded":
                raise ConnectionError("发现服务已失联")
            coroutine = changed_query if self.calls > 1 and change == "schema" else broken_query
            return [StructuredTool.from_function(coroutine=coroutine, name="lookup_fault", description="查询外部资料")]

    client = DiscoveryClient()
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[
            call("fault-1", "lookup_fault", keyword="工单")]), AgentDecision(kind="answer")],
            answers=[["工单会转交专员，保证跟进处理。"]]) as (service, deps, factory, _):
        manager = MCPToolManager(deps.registry, permissions, {"docs": {}}, client=client)
        deps.mcp, deps.permissions, deps.executor.permissions = manager, permissions, permissions
        try:
            events = await collect(service, "工单有什么用？", session="mcp-refresh-fault")
            streamed = "".join(event.data.get("delta", "") for event in events)
            assert client.calls == 2 and events[-1].event == "done"
            assert await rows(factory, "SELECT tool_name FROM tool_audit_logs") == [("lookup_fault",)]
            assert streamed == FAULT_TEXT
            assert "answer" not in [item["stage"] for item in deps.model.inputs]
            assert (await service.get_current_actions("mcp-refresh-fault", "demo-user")).answer == streamed
        finally:
            await manager.aclose()


@pytest.mark.parametrize("corruption", ["absent", "wrong_turn", "wrong_binding", "bad_access", "list_access", "message_only", "payload_only"])
def test_policy_requires_server_snapshot_bound_to_original_call(corruption):
    """缺失旧快照或伪造消息中的快照被当作服务器核验事实时失败。

    参数：corruption 不可信或不匹配的快照场景；返回：None；异常：断言失败。
    """
    from app.workflow.answer_policy import query_fault_answer
    state, registry = query_state()
    saved = state["tool_call_facts"]["read-1"]
    if corruption == "wrong_turn":
        saved["turn_id"] = "old"
    elif corruption == "wrong_binding":
        saved["binding"]["schema_hash"] = "a" * 64
    elif corruption == "bad_access":
        saved["access"] = "unknown"
    elif corruption == "list_access":
        saved["access"] = ["read"]
    else:
        state.pop("tool_call_facts")
        if corruption == "message_only":
            state["messages"][2]["additional_kwargs"]["tool_call_facts"] = {"read-1": saved}
        elif corruption == "payload_only":
            state["messages"][0]["content"] += json.dumps({"tool_call_facts": {"read-1": saved}})
            state["raw_query"] = state["messages"][0]["content"]
    assert query_fault_answer(state, registry=registry) is None


async def test_new_turn_clears_original_call_snapshot_after_fault(tmp_path):
    """同一检查点的新轮沿用旧工具快照或旧故障覆盖致谢时失败。

    参数：tmp_path 独占SQLite目录；返回：None；异常：断言或数据库错误。
    """
    async with service_runtime(tmp_path, intents=["订单", "闲聊"], decisions=[
            AgentDecision(kind="tools", tool_calls=[call("faq-1", "query_faq", keyword="工单")]),
            AgentDecision(kind="answer"), AgentDecision(kind="answer")],
            results=[ToolResult.failed("TOOL_TIMEOUT", "查询超时")], answers=[["不客气。"]]) as (service, _, factory):
        first = await collect(service, "工单有什么用？", session="same-session")
        assert "".join(event.data.get("delta", "") for event in first) == FAULT_TEXT
        second = await collect(service, "谢谢", session="same-session")
        assert second[-1].event == "done"
        # 闲聊沿用原有固定出口；关注旧故障不得覆盖新轮，并验证检查点真的清空快照。
        assert FAULT_TEXT not in "".join(event.data.get("delta", "") for event in second)
        conversation_id = (await rows(factory, "SELECT id FROM conversations"))[0][0]
        snapshot = await service.graph.aget_state({"configurable": {"thread_id": checkpoint_thread_id(conversation_id)}})
        assert snapshot.values["tool_call_facts"] == {}
