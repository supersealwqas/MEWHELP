"""验证订单和物流工具隐藏身份、强制运行时身份与统一归属拒绝。"""

from langchain_core.tools import InjectedToolArg
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError
import pytest

from app.tools.registry import build_registry


@pytest.mark.parametrize("name,public_field", [
    ("query_order", "order_id"), ("query_logistics", "tracking_no"),
])
def test_owned_tool_schema_hides_required_injected_identity(name, public_field):
    """遗漏注入标记会把身份暴露给模型。参数：工具名及公开字段；返回：None；异常：断言失败。"""
    tool = build_registry().get(name)
    runtime_fields = tool.args_schema.model_fields
    assert "user_id" in runtime_fields
    assert runtime_fields["user_id"].is_required()
    assert InjectedToolArg in runtime_fields["user_id"].metadata
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]
    assert set(parameters["properties"]) == {public_field}
    assert parameters["required"] == [public_field]


@pytest.mark.parametrize("name,args", [
    ("query_order", {"order_id": "1001"}),
    ("query_logistics", {"tracking_no": "MOCK213502378238"}),
])
def test_runtime_schema_normalizes_injected_identity(name, args):
    """执行器使用完整模型规范化注入身份。参数：工具名及公开参数；返回：None；异常：断言失败。"""
    schema = build_registry().get(name).args_schema
    normalized = schema.model_validate({**args, "user_id": " demo-user "}).model_dump()
    assert normalized == {**args, "user_id": "demo-user"}


@pytest.mark.parametrize("name,args", [
    ("query_order", {"order_id": "1001"}),
    ("query_logistics", {"tracking_no": "MOCK213502378238"}),
])
async def test_owned_tools_require_runtime_identity(name, args):
    """运行时缺少身份必须拒绝。参数：工具名及缺少身份的参数；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError) as error:
        await build_registry().get(name).ainvoke(args)
    assert any(item["loc"] == ("user_id",) and item["type"] == "missing" for item in error.value.errors())


@pytest.mark.parametrize("name,args", [
    ("query_order", {"order_id": "1001"}),
    ("query_logistics", {"tracking_no": "MOCK213502378238"}),
])
@pytest.mark.parametrize("invalid_identity", ["", "   ", 1, None])
async def test_owned_tools_reject_invalid_runtime_identity(name, args, invalid_identity):
    """身份不能省略、转型或为空白。参数：工具、公开参数和非法身份；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError):
        await build_registry().get(name).ainvoke({**args, "user_id": invalid_identity})


@pytest.mark.parametrize("order_id,user_id", [
    ("2003", "demo-user"), ("missing-order", "demo-user"),
    ("1001", "demo-other-user"), ("1001", "unknown-user"),
])
async def test_unknown_and_foreign_orders_return_same_safe_error(order_id, user_id):
    """归属拒绝不能泄露订单字段或存在性。参数：订单号与不匹配身份；返回：None；异常：断言失败。"""
    result = await build_registry().get("query_order").ainvoke({"order_id": order_id, "user_id": user_id})
    assert result == {"error": "没有找到您的这笔订单", "code": "order_not_owned"}


@pytest.mark.parametrize("tracking_no,user_id", [
    ("MOCK500290570519", "demo-user"), ("missing-tracking", "demo-user"),
    ("MOCK213502378238", "demo-other-user"), ("MOCK213502378238", "unknown-user"),
    ("1001", "demo-user"),
])
async def test_unknown_and_foreign_logistics_return_same_safe_error(tracking_no, user_id):
    """物流入口同样校验归属且不接受订单号。参数：物流号与身份；返回：None；异常：断言失败。"""
    result = await build_registry().get("query_logistics").ainvoke({"tracking_no": tracking_no, "user_id": user_id})
    assert result == {"error": "没有找到您的这笔订单", "code": "order_not_owned"}


async def test_other_demo_user_can_read_own_order_and_logistics():
    """归属规则不能硬编码成只允许一个身份。参数：无；返回：None；异常：断言失败。"""
    registry = build_registry()
    order = await registry.get("query_order").ainvoke({"order_id": "2003", "user_id": "demo-other-user"})
    assert order["order_id"] == "2003"
    assert order["tracking_no"] == "SF500290570519"
    logistics = await registry.get("query_logistics").ainvoke({
        "tracking_no": order["tracking_no"], "user_id": "demo-other-user",
    })
    assert logistics["order_id"] == "2003"
    assert logistics["status"] == "已签收"


@pytest.mark.parametrize("name,args", [
    ("query_order", {"order_id": "missing-order"}),
    ("query_logistics", {"tracking_no": "missing-tracking"}),
])
async def test_denial_results_are_independent_copies(name, args):
    """调用方修改拒绝结果不能污染后续调用。参数：工具名及未知标识；返回：None；异常：断言失败。"""
    tool = build_registry().get(name)
    first = await tool.ainvoke({**args, "user_id": "demo-user"})
    first["error"] = "调用方修改"
    first["order_id"] = "注入字段"
    second = await tool.ainvoke({**args, "user_id": "demo-user"})
    assert second == {"error": "没有找到您的这笔订单", "code": "order_not_owned"}


async def test_mcp_ownership_denial_happens_before_remote_handler(tmp_path):
    """跨用户参数若发送远端再拒绝会失败。参数：临时规则目录；返回：None；异常：断言失败。"""
    from tests.tools.test_execution import AuditSink, mcp_setup
    from tests.test_tool_executor import ResultStore, call, context, ignore_status
    from app.tools.contracts import ToolExecutionScope
    from app.tools.executor import ToolExecutor

    schema = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
    probe, registry, row, permissions, _, _ = mcp_setup(tmp_path, ownership="order_id", schema=schema)
    audit = AuditSink()
    ctx = context().model_copy(update={"user_id": "demo-user"})
    result = await ToolExecutor(registry, ResultStore(), None, permissions=permissions, audit_repository=audit).execute(
        call("probe", {"order_id": "2003"}), ctx, ignore_status, scope=ToolExecutionScope(binding=row.binding))
    assert not probe.calls and result.error.category == "permission_denied"
    assert audit.entries[0].status == "权限拒绝"
