"""验证动态注册的可见性、来源隔离和快照不可被调用者篡改。"""

from dataclasses import replace

import pytest

from app.schemas.tools import ToolResult
from app.tools.registry import ToolRegistry, build_registry


async def echo(arguments, context, scope):
    """返回受控结果。参数：公开参数、可信身份及运行域；返回：成功结果；异常：无。"""
    return ToolResult.success(arguments)


def definition(name="query_hours", *, server=None, schema=None):
    """创建真实注册输入。参数：工具名/可选Server/schema；返回：工具定义；异常：定义非法。"""
    from app.tools.contracts import ResultPolicy, ToolDefinition, ToolIdentity
    return ToolDefinition(
        identity=ToolIdentity(source="mcp" if server else "builtin", server=server, name=name),
        description="查询服务时间", parameters=schema or {"type": "object", "properties": {}},
        invoke=echo, access="read", result_policy=ResultPolicy(),
    )


def test_register_then_snapshot_without_rebuild():
    """运行中新增对下一快照可见。参数：无；返回：None；异常：断言失败。"""
    registry = ToolRegistry()
    previous = registry.snapshot()
    registry.register_definition(definition())
    assert previous == ()
    assert [row.identity.name for row in registry.snapshot()] == ["query_hours"]


def test_duplicate_and_control_names_are_rejected():
    """防止业务工具覆盖控制协议。参数：无；返回：None；异常：断言失败。"""
    registry = ToolRegistry()
    registry.register_definition(definition())
    for name in ("query_hours", "submit_decision", "submit_intent", "submit_understanding"):
        with pytest.raises(ValueError):
            registry.register_definition(definition(name))


def test_snapshots_and_original_schema_cannot_change_registry():
    """调用者修改对象不能扩大参数范围。参数：无；返回：None；异常：断言失败。"""
    original = definition(schema={"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}})
    registry = ToolRegistry()
    registry.register_definition(original)
    original.parameters["properties"]["age"]["minimum"] = -10
    snapshot = registry.snapshot()
    snapshot[0].parameters["properties"]["age"]["minimum"] = -20
    assert registry.get_definition(original.identity).parameters["properties"]["age"]["minimum"] == 0


def test_source_replacement_is_atomic_and_binding_stays_with_source():
    """更新来源不影响其他工具，旧来源不能匹配同名新来源。参数：无；返回：None；异常：断言失败。"""
    registry = ToolRegistry()
    one = definition("query_warranty", server="after_sales")
    registry.register_definition(definition())
    registry.replace_source("after_sales", [one])
    binding = registry.get_definition(one.identity).binding
    registry.replace_source("after_sales", [])
    registry.replace_source("different", [definition("query_warranty", server="different")])
    with pytest.raises(KeyError):
        registry.get_definition(binding.identity)
    assert {row.identity.name for row in registry.snapshot()} == {"query_hours", "query_warranty"}


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_duplicate_names_have_no_race_winner(order):
    """跨Server撞名双方隔离，冲突消失后恢复。参数：登记顺序；返回：None；异常：断言失败。"""
    registry = ToolRegistry()
    for server in order:
        registry.replace_source(server, [definition("query_shared", server=server)])
    assert registry.snapshot() == ()
    registry.replace_source("a", [])
    assert registry.snapshot()[0].identity.server == "b"


def test_mcp_never_overrides_builtin_or_control():
    """拒绝远端抢占本地名称。参数：无；返回：None；异常：断言失败。"""
    registry = ToolRegistry()
    registry.register_definition(definition())
    rejected = registry.replace_source("a", [definition(server="a"), definition("submit_decision", server="a")])
    assert len(rejected) == 2
    assert [(row.identity.source, row.identity.name) for row in registry.snapshot()] == [("builtin", "query_hours")]


def test_builtin_public_schema_and_local_access_ignore_tags():
    """内部身份不公开，篡改tags不改变本地写权限。参数：无；返回：None；异常：断言失败。"""
    registry = build_registry()
    snapshots = registry.snapshot()
    order = next(row for row in snapshots if row.identity.name == "query_order")
    ticket = next(row for row in snapshots if row.identity.name == "create_ticket")
    assert set(order.parameters["properties"]) == {"order_id"}
    assert order.parameters["additionalProperties"] is False
    assert order.access == "read" and ticket.access == "write"
    assert {tool.name for tool in registry.all()} == {row.identity.name for row in snapshots}


def test_invalid_registration_does_not_change_existing_snapshot():
    """无描述、坏schema或过长名称拒绝登记。参数：无；返回：None；异常：断言失败。"""
    registry = ToolRegistry()
    registry.register_definition(definition())
    for candidate in (replace(definition("bad"), description=" "), definition("bad", schema={"type": "not-a-type"})):
        with pytest.raises(ValueError):
            registry.register_definition(candidate)
    with pytest.raises(ValueError):
        registry.register_definition(definition("x" * 65))
    assert len(registry.snapshot()) == 1
