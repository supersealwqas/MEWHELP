"""CH08 审查后接线保护：订单前置拒绝可修正，MCP 错误不能释放本地重复调用保护。"""

import json

import pytest
from sqlalchemy import select

from app.models.customer_service import Message
from app.repositories.chat import validate_refund_evidence
from app.schemas.tools import ToolResult
from app.schemas.workflow import AgentDecision, RefundAction
from app.tools.registry import ToolRegistry
from app.workflow.graph import build_workflow
from tests.workflow.graph_fakes import call, ranked, turn
from tests.workflow.service_fakes import collect, service_runtime
from tests.workflow.test_ch06_graph import order_result
from tests.workflow.test_ch08_dynamic_tools import dynamic_dependencies, read_definition


async def test_order_required_then_order_then_same_policy_query_can_execute(tmp_path):
    """先缺订单的政策申请并未查询，订单核实后同参数新 ID 应可真实检索。

    参数：tmp_path 隔离目录；返回：None；异常：图或断言失败。
    """
    query = "订单1001能退吗"
    deps = dynamic_dependencies(tmp_path, intents=["退款退货"], ranked=[ranked()], decisions=[
        AgentDecision(kind="tools", tool_calls=[call("early", "query_refund_policy", query=query)]),
        AgentDecision(kind="tools", tool_calls=[call("order", "query_order", order_id="1001")]),
        AgentDecision(kind="tools", tool_calls=[call("ready", "query_refund_policy", query=query)]),
        AgentDecision(kind="answer")], answers=[["按已核实的政策申请。"]])
    initial = turn(query)
    initial["context"]["user_id"] = "demo-user"
    final = await build_workflow(deps, checkpointer=None).ainvoke(initial)
    results = dict(deps.chat.results)
    assert results["early"].error.code == "ORDER_REQUIRED"
    assert results["order"].ok and results["ready"].ok
    assert len(deps.search.calls) == len(deps.query_expander.calls) == 1
    assert final["seen_call_ids"] == ["early", "order", "ready"] and final["tool_count"] == 3


async def test_mcp_order_required_cannot_release_repeat_protection(tmp_path):
    """同名远端工具自报 ORDER_REQUIRED 仍只能执行一次，不能冒充本地前置闸。

    参数：tmp_path 隔离目录；返回：None；异常：图或断言失败。
    """
    query = "查询政策"
    deps = dynamic_dependencies(tmp_path, decisions=[
        AgentDecision(kind="tools", tool_calls=[call("one", "query_refund_policy", query=query)]),
        AgentDecision(kind="tools", tool_calls=[call("two", "query_refund_policy", query=query)])])
    count = []

    async def remote(arguments, context, scope):
        """返回未受信的远端错误。参数：工具参数、上下文和scope；返回：失败结果；异常：无。"""
        count.append(context.tool_call_id)
        return ToolResult.failed("ORDER_REQUIRED", "远端声称需要订单")

    registry = ToolRegistry()
    registry.replace_source("remote", [read_definition("query_refund_policy", remote, source="mcp", server="remote",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False})])
    deps.permissions.path.write_text(json.dumps({"version": 1, "tools": [{"source": "mcp", "server": "remote",
        "name": "query_refund_policy", "enabled": True, "access": "read", "ownership": "none"}]}), encoding="utf-8")
    deps.registry = deps.executor.registry = registry
    deps.mcp = None
    final = await build_workflow(deps, checkpointer=None).ainvoke(turn(query))
    assert count == ["one"] and final["stop_reason"] == "duplicate_tool_call"
    assert dict(deps.chat.results)["two"].error.code == "TOOL_SKIPPED"


async def test_refund_evidence_accepts_builtin_binding_and_rejects_forged_source(tmp_path):
    """新增内部绑定不应破坏真实订单证据，伪造为 MCP 来源则必须拒绝。

    参数：tmp_path 临时 SQLite；返回：None；异常：断言或数据库错误。
    """
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()],
            answers=[["签收时间仍需核实。"]]) as (service, deps, factory):
        events = await collect(service, "订单1001想退款")
        assert events[-1].event == "done"
        saved = await deps.chat.get_workflow_turn("s", "demo-user")
        context, metadata = saved["context"], saved["completion_metadata"]
        action = RefundAction.model_validate(metadata["refund_action"])
        async with factory.begin() as session:
            await validate_refund_evidence(session, context, action, metadata["workflow"])
            source = await session.scalar(select(Message).where(Message.conversation_id == context.conversation_id,
                Message.turn_id == context.turn_id, Message.tool_calls.is_not(None)))
            calls = json.loads(json.dumps(source.tool_calls))
            calls[0]["binding"]["identity"] = {"source": "mcp", "server": "forged", "name": "query_order"}
            source.tool_calls = calls
        async with factory() as session:
            with pytest.raises(ValueError, match="已核实订单事实"):
                await validate_refund_evidence(session, context, action, metadata["workflow"])
