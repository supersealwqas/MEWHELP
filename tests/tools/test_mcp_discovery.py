"""验证按Server独立发现、故障撤下、合并刷新及真实LangChain返回适配。"""

import asyncio

from langchain_core.tools import StructuredTool
import pytest

from app.tools.contracts import ToolExecutionScope
from app.tools.registry import ToolRegistry
from tests.tools.test_permissions import context, identity, rule, write_policy


def remote_tool(name="query_warranty", *, schema=None, artifact=None):
    """建立替代远程I/O的LangChain工具。参数：名称/schema/结构化结果；返回：StructuredTool；异常：构造错误。"""
    async def invoke(**arguments):
        """模拟一次远端返回，保留真实内容/附件包装。参数：公开参数；返回：二元组；异常：无。"""
        return [{"type": "text", "text": "演示信息"}], {"structured_content": artifact or {"order_id": arguments.get("order_id"), "status": "in_warranty"}}
    return StructuredTool.from_function(coroutine=invoke, name=name, description="查询演示业务状态",
        args_schema=schema or {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        response_format="content_and_artifact")


class Client:
    """只替换远程发现边界，注册与并发生命周期使用真实实现。"""

    def __init__(self, responses):
        """初始化逐来源响应。参数：来源到工具/异常映射；返回：None；异常：无。"""
        self.responses, self.calls = responses, []
        self.gate = None
        self.entered = asyncio.Event()

    async def get_tools(self, *, server_name):
        """模拟网络发现。参数：来源名；返回：工具表；异常：预设异常。"""
        self.calls.append(server_name)
        self.entered.set()
        if self.gate:
            await self.gate.wait()
        response = self.responses[server_name]
        if isinstance(response, Exception):
            raise response
        if response is None:
            await asyncio.Event().wait()
        return response


def manager(tmp_path, responses):
    """组合真实注册/权限与可控远端。参数：目录和发现响应；返回：管理器/注册表/客户端；异常：配置错误。"""
    from app.tools.permissions import ToolPermissions
    from app.tools.mcp import MCPToolManager
    path = tmp_path / "policy.json"
    write_policy(path, rule(), rule("query_logistics", "logistics", ownership="tracking_no"))
    permissions, registry, client = ToolPermissions(path), ToolRegistry(), Client(responses)
    return MCPToolManager(registry, permissions, {key: {"transport": "http", "url": "http://127.0.0.1:1/mcp"} for key in responses},
                          timeout_seconds=0.05, client=client), registry, client


async def test_slow_server_does_not_remove_other_source(tmp_path):
    """单来源超时不吞掉另一个Server的结果。参数：隔离目录；返回：None；异常：断言失败。"""
    runtime, registry, _ = manager(tmp_path, {"logistics": None, "after_sales": [remote_tool()]})
    await asyncio.wait_for(runtime.refresh(), 1)
    assert [row.name for row in registry.snapshot()] == ["query_warranty"]
    await runtime.aclose()


async def test_failed_discovery_withdraws_only_its_tools(tmp_path):
    """失联来源撤下且重连可恢复。参数：隔离目录；返回：None；异常：断言失败。"""
    runtime, registry, client = manager(tmp_path, {"after_sales": [remote_tool()]})
    await runtime.refresh()
    assert registry.snapshot()
    client.responses["after_sales"] = ConnectionError("offline")
    await runtime.refresh()
    assert registry.snapshot() == ()
    client.responses["after_sales"] = [remote_tool()]
    await runtime.refresh()
    assert registry.snapshot()[0].name == "query_warranty"
    await runtime.aclose()


async def test_refresh_is_coalesced(tmp_path):
    """同时查询共用在途发现，取消一个等待者不伤其他会话。参数：隔离目录；返回：None；异常：断言失败。"""
    runtime, registry, client = manager(tmp_path, {"after_sales": [remote_tool()]})
    client.gate = asyncio.Event()
    first = asyncio.create_task(runtime.refresh())
    await client.entered.wait()
    second = asyncio.create_task(runtime.refresh())
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    client.gate.set()
    await second
    assert client.calls == ["after_sales"]
    assert registry.snapshot()[0].name == "query_warranty"
    await runtime.aclose()


async def test_restarted_server_replaces_schema(tmp_path):
    """每轮重新发现后拿到新schema而非旧工具缓存。参数：隔离目录；返回：None；异常：断言失败。"""
    runtime, registry, client = manager(tmp_path, {"after_sales": [remote_tool()]})
    await runtime.refresh()
    old = registry.snapshot()[0].binding
    client.responses["after_sales"] = [remote_tool(schema={"type": "object", "properties": {"order_id": {"type": "string", "maxLength": 4}}, "required": ["order_id"]})]
    await runtime.refresh()
    assert registry.snapshot()[0].binding.schema_hash != old.schema_hash
    await runtime.aclose()


async def test_structured_artifact_keeps_business_data_and_local_mock_flag(tmp_path):
    """适配器读取真实ToolMessage附件，不把整个包装当业务数据。参数：隔离目录；返回：None；异常：断言失败。"""
    runtime, registry, _ = manager(tmp_path, {"after_sales": [remote_tool()]})
    await runtime.refresh()
    result = await registry.get_definition(identity()).invoke({"order_id": "1001"}, context(), ToolExecutionScope())
    assert result.ok and result.data == {"order_id": "1001", "status": "in_warranty"}
    assert result.is_mock is True
    await runtime.aclose()


async def test_unlisted_tool_annotations_do_not_grant_permission(tmp_path):
    """远端声明只读也不能被新工具拿来授权。参数：隔离目录；返回：None；异常：断言失败。"""
    unknown = remote_tool("unlisted")
    unknown.metadata = {"annotations": {"readOnlyHint": True}}
    runtime, registry, _ = manager(tmp_path, {"after_sales": [unknown]})
    await runtime.refresh()
    assert registry.snapshot() == ()
    await runtime.aclose()


async def test_cross_owner_never_reaches_mcp(tmp_path):
    """已发现工具也必须在网络边界前核验归属。参数：隔离目录；返回：None；异常：断言失败。"""
    calls = []

    async def forbidden(**arguments):
        """记录不应发生的网络调用。参数：公开参数；返回：测试结果；异常：无。"""
        calls.append(arguments)
        return [], {"structured_content": {"order_id": "2003"}}

    tool = remote_tool()
    tool.coroutine = forbidden
    runtime, registry, _ = manager(tmp_path, {"after_sales": [tool]})
    await runtime.refresh()
    result = await registry.get_definition(identity()).invoke({"order_id": "2003"}, context(), ToolExecutionScope())
    assert result.error.category == "permission_denied" and calls == []
    await runtime.aclose()


@pytest.mark.parametrize("invalid", ["schema", "description"])
async def test_bad_schema_is_isolated_without_losing_good_tools(tmp_path, invalid):
    """远端坏定义只隔离自身，不能使整轮发现抛异常。参数：隔离目录/坏字段；返回：None；异常：断言失败。"""
    bad = remote_tool("query_logistics", schema={"type": "object", "properties": {
        "tracking_no": {"type": "string", "format": "unsupported-remote-format"}}})
    if invalid == "description":
        bad = remote_tool("query_logistics")
        bad.description = " "
    runtime, registry, _ = manager(tmp_path, {"logistics": [bad], "after_sales": [remote_tool()]})
    await runtime.refresh()
    assert [row.name for row in registry.snapshot()] == ["query_warranty"]
    await runtime.aclose()


@pytest.mark.parametrize("sources", [("logistics", "after_sales"), ("after_sales", "logistics")])
async def test_duplicate_names_have_no_race_winner(tmp_path, sources):
    """两来源同名时全部隔离，与来源发现顺序无关。参数：目录/来源顺序；返回：None；异常：断言失败。"""
    runtime, registry, _ = manager(tmp_path, {name: [remote_tool()] for name in sources})
    write_policy(runtime.permissions.path, *(rule(server=name) for name in sources))
    await runtime.refresh()
    assert registry.snapshot() == ()
    await runtime.aclose()


@pytest.mark.parametrize("handle_error", [False, True])
async def test_remote_tool_error_is_never_success(tmp_path, handle_error):
    """远端错误无论抛异常还是返回错误消息都不能成为事实。参数：目录/错误包装；返回：None；异常：断言失败。"""
    from langchain_core.tools import ToolException

    async def fail(**arguments):
        """模拟远端业务错误。参数：公开参数；返回：无；异常：ToolException。"""
        raise ToolException("模拟远端错误文本")

    tool = remote_tool()
    tool.coroutine, tool.handle_tool_error = fail, handle_error
    runtime, registry, _ = manager(tmp_path, {"after_sales": [tool]})
    await runtime.refresh()
    result = await registry.get_definition(identity()).invoke({"order_id": "1001"}, context(), ToolExecutionScope())
    assert not result.ok and result.data is None and result.error.category == "fault"
    await runtime.aclose()


async def test_close_cancels_owned_discovery(tmp_path):
    """关闭回收正在发现的任务，不让晚到结果重新注册。参数：隔离目录；返回：None；异常：断言失败。"""
    runtime, registry, client = manager(tmp_path, {"after_sales": [remote_tool()]})
    client.gate = asyncio.Event()
    waiter = asyncio.create_task(runtime.refresh())
    await client.entered.wait()
    await runtime.aclose()
    assert isinstance((await asyncio.gather(waiter, return_exceptions=True))[0], asyncio.CancelledError)
    client.gate.set()
    await runtime.refresh()
    assert registry.snapshot() == ()
