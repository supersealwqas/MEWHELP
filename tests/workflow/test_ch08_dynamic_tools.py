"""CH08 真实图的动态工具可达性、已保存决定和实际 schema 预算回归测试。"""

from copy import deepcopy
import json

import pytest

from app.core.model_budget import BudgetLedger
from app.schemas.tools import ToolResult
from app.schemas.workflow import AgentDecision
from app.tools.contracts import ToolDefinition, ToolIdentity
from app.tools.executor import ToolExecutor
from app.tools.mcp import MCPToolManager
from app.tools.permissions import ToolPermissions
from app.tools.registry import build_agent_registry
from app.workflow.agent import build_agent_graph
from app.workflow.graph import build_workflow
from app.workflow.model import WorkflowModelAdapter
from tests.workflow.graph_fakes import StrictChat, call, dependencies, turn
from tests.workflow.test_model import sdk_runtime, terminal_response


class DynamicChat(StrictChat):
    """仅替换外部消息数据库，保留真实执行器要求的结果重放和申请绑定契约。"""

    async def get_tool_result(self, context):
        """读取已保存原回执。参数：可信调用身份；返回：ToolResult 或 None；异常：无。"""
        return next((result for identifier, result in self.results if identifier == context.tool_call_id), None)

    async def save_tool_calls(self, context, calls, *, deduplicate=False, bindings=None):
        """保存申请与内部绑定。参数：身份、申请、去重开关、可选绑定；返回：None；异常：断言失败。"""
        self.saved_bindings = deepcopy(bindings)
        await super().save_tool_calls(context, calls, deduplicate=deduplicate, bindings=bindings)


def dynamic_dependencies(tmp_path, *, decisions, intents=("订单",), answers=(("已核实",),), ranked=()):
    """接入真实清单/权限/执行器，只隔离外部模型和仓储。

    参数：tmp_path 临时授权文件目录、decisions/intents/answers 有限模型脚本、ranked 检索结果。
    返回：WorkflowDependencies；异常：配置或测试夹具断言错误。
    """
    deps = dependencies(decisions=decisions, intents=intents, answers=answers, ranked=ranked)
    registry = build_agent_registry()
    path = tmp_path / "permissions.json"
    path.write_text('{"version":1,"tools":[]}', encoding="utf-8")
    permissions = ToolPermissions(path)
    permissions.reload()
    deps.registry, deps.permissions = registry, permissions
    deps.mcp = MCPToolManager(registry, permissions, {})
    deps.chat = DynamicChat()
    deps.executor = ToolExecutor(registry, deps.chat, None, permissions=permissions, retry_delay_seconds=0)
    return deps


def read_definition(name, handler, *, source="builtin", server=None, parameters=None):
    """建立此前不存在的新用途。参数：名称、真实处理器、可信来源、参数；返回：ToolDefinition；异常：非法定义。"""
    return ToolDefinition(ToolIdentity(source, server, name), "查询门店营业时间或当前办理进度",
        parameters or {"type": "object", "properties": {}, "additionalProperties": False}, handler, "read")


async def test_runtime_registered_tool_reaches_existing_graph(tmp_path):
    """构图时复制清单导致后注册的新用途不可见或不可执行时失败。

    参数：tmp_path 隔离权限文件；返回：None；异常：图或断言失败。
    """
    deps = dynamic_dependencies(tmp_path, decisions=[
        AgentDecision(kind="tools", tool_calls=[call("hours", "query_store_hours", branch="西湖店")]),
        AgentDecision(kind="answer")])
    graph = build_workflow(deps, checkpointer=None)

    async def hours(arguments, context, scope):
        """返回本地登记的新业务事实。参数：公开参数、身份、资源；返回：营业时间；异常：无。"""
        return ToolResult.success({"branch": arguments["branch"], "hours": "09:00–20:00"})

    deps.registry.register_definition(read_definition("query_store_hours", hours, parameters={
        "type": "object", "properties": {"branch": {"type": "string"}},
        "required": ["branch"], "additionalProperties": False}))
    state = await graph.ainvoke(turn("西湖店几点营业？"))
    assert [identifier for identifier, _ in deps.chat.results] == ["hours"]
    assert deps.chat.results[0][1].data["hours"] == "09:00–20:00"
    assert all("query_store_hours" in {tool.name for tool in item["tools"]}
               for item in deps.model.inputs if item["stage"] == "decide")
    assert state["tool_count"] == 1 and state["decision_count"] == 2


@pytest.mark.parametrize("intent", ["售后", "退款退货"])
async def test_warranty_and_return_progress_are_not_trapped_by_route(tmp_path, intent):
    """固定售后/退款分支抢先结束，导致 MCP 进度查询不可达时失败。

    参数：tmp_path 本地规则、intent 原分类结果；返回：None；异常：图或断言失败。
    """
    deps = dynamic_dependencies(tmp_path, intents=[intent], decisions=[
        AgentDecision(kind="tools", tool_calls=[{"id": "progress", "name": "query_return_progress", "args": {}}]),
        AgentDecision(kind="answer")])
    deps.permissions.path.write_text(json.dumps({"version": 1, "tools": [{
        "source": "mcp", "server": "returns", "name": "query_return_progress", "enabled": True,
        "access": "read", "ownership": "none"}]}), encoding="utf-8")

    async def progress(arguments, context, scope):
        """模拟唯一外部业务回执。参数：公开参数、身份、资源；返回：进度；异常：无。"""
        return ToolResult.success({"progress": "仓库已收件"})

    deps.registry.replace_source("returns", [read_definition("query_return_progress", progress,
        source="mcp", server="returns")])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("之前那笔申请现在处理到哪了？"))
    assert any(identifier == "progress" and result.data == {"progress": "仓库已收件"}
               for identifier, result in deps.chat.results)
    assert state["tool_count"] >= 1


async def test_first_decision_is_not_repeated(tmp_path):
    """子图 START 重做前置决定而丢失已保存申请时失败。

    参数：tmp_path 权限文件；返回：None；异常：图或断言失败。
    """
    deps = dynamic_dependencies(tmp_path, decisions=[AgentDecision(kind="answer")])
    initial = turn("查一下猫粮价格")
    initial.update(route="business", decision_count=1,
        pending_calls=[call("product", "query_product", keyword="猫粮")])
    state = await build_agent_graph(deps).ainvoke(initial)
    assert [identifier for identifier, _ in deps.chat.results] == ["product"]
    assert state["tool_count"] == 1
    assert len([item for item in deps.model.inputs if item["stage"] == "decide"]) == 1
    assert state["decision_count"] == 2


async def test_schema_budget_matches_bound_tools(sdk_runtime):
    """动态可选参数被强制 strict 或 SDK 实际 schema 未计入预算时失败。

    参数：sdk_runtime 仅替换外部 HTTP；返回：None；异常：协议或断言失败。
    """
    async def hours(arguments, context, scope):
        """只提供合法工具定义，本例不执行。参数：参数、身份、资源；返回：业务结果；异常：无。"""
        return ToolResult.success({"hours": "09:00–20:00"})

    definition = read_definition("query_store_hours", hours, parameters={"type": "object",
        "properties": {"branch": {"type": "string"}}, "additionalProperties": False})
    sdk_runtime.responses.append(terminal_response())
    ledger = BudgetLedger()
    await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [definition], ledger)
    request = sdk_runtime.requests[0]
    schemas = request["tools"]
    assert [item["function"]["name"] for item in schemas] == ["query_store_hours", "submit_decision"]
    assert "strict" not in schemas[0]["function"]
    assert "required" not in schemas[0]["function"]["parameters"]
    assert schemas[1]["function"]["strict"] is True
    from langchain_core.messages import convert_to_messages
    expected = ledger.estimator.message_tokens(convert_to_messages(request["messages"])) + ledger.estimator.value_tokens(schemas)
    assert ledger.snapshot()["records"][0]["input_units"] == expected


async def test_unconfirmed_write_is_paired_and_never_reaches_business(tmp_path):
    """共享清单暴露写工具后，缺确认仍发生业务写入或遗漏拒绝事实时失败。

    参数：tmp_path 权限文件；返回：None；异常：图或断言失败。
    """
    deps = dynamic_dependencies(tmp_path, intents=["其他"], decisions=[
        AgentDecision(kind="tools", tool_calls=[call("ticket", "create_ticket", description="帮我建单", ticket_type="咨询")]),
        AgentDecision(kind="answer")], answers=[["创建工单仍需确认。"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("帮我建单"))
    assert len(deps.chat.results) == 1 and deps.chat.results[0][1].error.code == "NOT_CONFIRMED"
    assert state["tool_count"] == 1 and "NOT_CONFIRMED" in json.dumps(deps.model.inputs[-2]["messages"])


@pytest.mark.parametrize("intent", ["投诉", "退款退货"])
async def test_complaint_and_order_selection_keep_original_exits(tmp_path, intent):
    """新前置决定覆盖投诉固定回复或把歧义订单直接查询时失败。

    参数：tmp_path 权限文件、intent 旧固定出口；返回：None；异常：图或断言失败。
    """
    deps = dynamic_dependencies(tmp_path, intents=[intent], decisions=[AgentDecision(kind="answer")], answers=())
    initial = turn("我要投诉" if intent == "投诉" else "其中一笔订单想退货")
    initial["context"]["user_id"] = "demo-user"
    state = await build_workflow(deps, checkpointer=None).ainvoke(initial)
    assert not deps.chat.results and [item["stage"] for item in deps.model.inputs] == ["understand", "classify", "decide"]
    if intent == "投诉":
        from app.workflow.routing import fixed_reply
        assert state["answer"] == fixed_reply("complaint")["answer"]
    else:
        assert state["__interrupt__"][0].value["type"] == "select_order"


async def test_clarify_without_tools_does_not_retrieve_again(tmp_path):
    """已决定缺少信息时又检索知识，改变追问或多耗资源时失败。

    参数：tmp_path 权限文件；返回：None；异常：图或断言失败。
    """
    deps = dynamic_dependencies(tmp_path, intents=["其他"], decisions=[AgentDecision(kind="clarify")],
        answers=[["请补充门店名称。"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("营业情况呢"))
    assert state["answer"] == "请补充门店名称。" and deps.search.calls == []


def test_production_logistics_uses_mcp_and_never_falls_back_to_builtin():
    """旧内置物流撞名隔离MCP，或服务下线后回退本地演示查询时失败。

    参数：无；返回：None；异常：断言失败。
    """
    from app.tools.registry import build_agent_registry

    async def logistics(arguments, context, scope):
        """提供发现后的远端业务定义。参数：参数、身份、资源；返回：结果；异常：无。"""
        return ToolResult.success({"status": "运输中"})

    registry = build_agent_registry()
    rejected = registry.replace_source("logistics", [read_definition("query_logistics", logistics,
        source="mcp", server="logistics")])
    assert rejected == []
    actual = next(row for row in registry.snapshot() if row.name == "query_logistics")
    assert actual.identity.source == "mcp"
    registry.replace_source("logistics", [])
    assert all(row.name != "query_logistics" for row in registry.snapshot())
