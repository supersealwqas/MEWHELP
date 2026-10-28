"""验证工具执行器的限时、有限重试、持久化顺序及取消。"""

import asyncio
from typing import Annotated
import pytest
from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field
from app.repositories.chat import PersistenceUncertain
from app.schemas.tools import BusinessArgs, ToolContext, ToolResult
from app.tools.registry import ToolRegistry, build_registry
from app.tools.executor import ToolExecutor
from app.tools.contracts import ToolDefinition, ToolExecutionScope, ToolIdentity, WriteAuthorization, json_hash


class ResultStore:
    """执行器单测的存储边界替身，不用于应用运行。"""
    def __init__(self, result=None):
        """初始化已提交结果。参数：可选原结果；返回：None；异常：无。"""
        self.result, self.writes = result, 0

    async def get_tool_result(self, context):
        """读取已提交事实。参数：调用身份；返回：结果或空；异常：无。"""
        return self.result

    async def save_tool_result(self, context, result):
        """模拟唯一结果持久化。参数：身份和结果；返回：原结果；异常：无。"""
        if self.result is None:
            self.result, self.writes = result, self.writes + 1
        return self.result


def context():
    """生成服务端身份。参数：无；返回：工具上下文；异常：无。"""
    return ToolContext(session_id="s", conversation_id=1, user_id="u", turn_id="t",
                       tool_call_id="call-1", user_message="邮费是多少")


def call(name="slow", args=None):
    """生成匹配调用。参数：名称和参数；返回：申请字典；异常：无。"""
    return {"id": "call-1", "name": name, "args": args or {}}


async def ignore_status(status):
    """测试无需展示状态。参数：状态字典；返回：None；异常：无。"""


async def confirmed_authorization(binding, arguments, ctx):
    """测试显式提供已核验的保存动作凭据。参数：来源/冻结参数/可信身份；返回：凭据；异常：无。"""
    return WriteAuthorization(ctx.conversation_id, ctx.turn_id, "saved-action", ctx.tool_call_id,
                              json_hash(arguments), "confirm", "action")


class IdentityArgs(BusinessArgs):
    """执行器身份注入探针的完整运行参数，保留真实严格校验。"""
    order_id: str = Field(min_length=1, max_length=64, description="订单号")
    user_id: Annotated[str, InjectedToolArg] = Field(min_length=1, description="系统注入用户身份")


@tool(args_schema=IdentityArgs)
async def identity_probe(order_id: str, user_id: Annotated[str, InjectedToolArg]) -> dict:
    """返回实际接收的参数。参数：订单号和系统身份；返回：业务字典；异常：无。"""
    return {"order_id": order_id, "user_id": user_id}


async def test_executor_injects_context_identity_without_mutating_model_call():
    """遗漏注入或直接修改模型申请会失败。参数：无；返回：None；异常：断言失败。"""
    registry, store = ToolRegistry(), ResultStore()
    registry.register(identity_probe)
    model_call = call("identity_probe", {"order_id": " 1001 "})
    result = await ToolExecutor(registry, store, None).execute(model_call, context(), ignore_status)
    assert result.ok and result.data == {"order_id": "1001", "user_id": "u"}
    assert model_call["args"] == {"order_id": " 1001 "}


@pytest.mark.parametrize("args", [
    {"order_id": "1001", "user_id": "victim"},
    {"order_id": "1001", "user_id": "u"},
    {"order_id": "1001", "extra": "value"},
    {"order_id": 1001},
    {"order_id": " "},
])
async def test_executor_rejects_injected_or_invalid_model_arguments(args):
    """模型夹带身份和不合法参数不能进入工具。参数：申请参数；返回：None；异常：断言失败。"""
    registry, store = ToolRegistry(), ResultStore()
    registry.register(identity_probe)
    result = await ToolExecutor(registry, store, None).execute(call("identity_probe", args), context(), ignore_status)
    assert not result.ok and result.error.code == "INVALID_ARGUMENTS" and result.data is None


@pytest.mark.parametrize("payload,ok", [
    ({"order_id": "1001", "tracking_no": "TRACK-1001"}, True),
    ({"error": "没有找到您的这笔订单", "code": "order_not_owned"}, False),
])
async def test_executor_normalizes_business_dictionary(payload, ok):
    """业务字典必须转换为统一结果，拒绝不是成功数据。参数：字典和成功标识；返回：None；异常：断言失败。"""
    @tool
    async def dictionary_probe() -> dict:
        """提供真实工具的字典返回。参数：无；返回：业务字典；异常：无。"""
        return dict(payload)
    registry, store = ToolRegistry(), ResultStore()
    registry.register(dictionary_probe)
    result = await ToolExecutor(registry, store, None).execute(call("dictionary_probe"), context(), ignore_status)
    assert result.ok is ok and store.result == result and store.writes == 1
    if ok:
        assert result.data == payload and result.error is None
    else:
        assert result.data is None
        assert result.error.model_dump() == {
            "message": payload["error"], "code": payload["code"], "category": "permission_denied"}


async def test_timeout_retry_once():
    """超时只尝试两次，最终仅存一个失败。参数：无；返回：None；异常：断言失败。"""
    attempts, states = [], []
    @tool
    async def slow() -> dict:
        """模拟超时边界。参数：无；返回：不会返回；异常：取消。"""
        attempts.append(1)
        await asyncio.Event().wait()
    registry, store = ToolRegistry(), ResultStore()
    registry.register(slow)
    async def status(frame):
        """记录执行顺序。参数：状态帧；返回：None；异常：断言失败。"""
        if frame["status"] == "failed":
            assert store.result is not None
        states.append(frame["status"])
    result = await ToolExecutor(registry, store, None, timeout_seconds=0.01, retry_delay_seconds=0).execute(call(), context(), status)
    assert len(attempts) == 2 and store.writes == 1
    assert not result.ok and result.error.code == "TOOL_TIMEOUT"
    assert states == ["running", "retrying", "failed"]


async def test_validation_and_not_found_not_retried():
    """参数错误和零命中不重试。参数：无；返回：None；异常：断言失败。"""
    class Support:
        """受控 FAQ 数据库边界。"""
        count = 0
        async def query_faq(self, keyword):
            """返回空查询。参数：原文关键词；返回：列表；异常：无。"""
            self.count += 1
            return []
    support = Support()
    for args, expected in [({"keyword": "邮费"}, True), ({"keyword": "运费"}, True), ({"keyword": " "}, False)]:
        store = ResultStore()
        result = await ToolExecutor(build_registry(), store, support).execute(call("query_faq", args), context(), ignore_status)
        assert result.ok is expected and store.writes == 1
    assert support.count == 2


async def test_cancel_during_backoff():
    """取消退避后不再执行或保存失败。参数：无；返回：None；异常：断言失败。"""
    attempts, retrying = [], asyncio.Event()
    @tool
    async def slow() -> dict:
        """模拟可重试超时。参数：无；返回：不返回；异常：TimeoutError。"""
        attempts.append(1)
        raise TimeoutError()
    registry, store = ToolRegistry(), ResultStore()
    registry.register(slow)
    async def status(frame):
        """通知已进入退避。参数：状态帧；返回：None；异常：无。"""
        if frame["status"] == "retrying":
            retrying.set()
    task = asyncio.create_task(ToolExecutor(registry, store, None, retry_delay_seconds=1).execute(call(), context(), status))
    await asyncio.wait_for(retrying.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(attempts) == 1 and store.writes == 0


async def test_error_redacted():
    """原始异常不能回灌或写入结果。参数：无；返回：None；异常：断言失败。"""
    @tool
    async def slow() -> dict:
        """模拟不可重试异常。参数：无；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("SECRET mysql://password@host SELECT * FROM tickets")
    registry, store = ToolRegistry(), ResultStore()
    registry.register(slow)
    result = await ToolExecutor(registry, store, None).execute(call(), context(), ignore_status)
    assert not result.ok
    assert "SECRET" not in result.model_dump_json() and "SELECT" not in result.model_dump_json()


async def test_reuse_persisted_result():
    """已提交结果不再调用业务。参数：无；返回：None；异常：断言失败。"""
    original = ToolResult(ok=True, data={"ticket_no": "T-original"})
    store = ResultStore(original)
    result = await ToolExecutor(build_registry(), store, None).execute(
        call("create_ticket", {"description": "投诉", "ticket_type": "投诉"}), context(), ignore_status)
    assert result == original and store.writes == 0


async def test_unavailable_verification_never_executes():
    """无法核实提交状态就安全结束。参数：无；返回：None；异常：断言失败。"""
    class UnavailableStore(ResultStore):
        """数据库故障边界。"""
        async def get_tool_result(self, context):
            """模拟读取失败。参数：上下文；返回：不返回；异常：ConnectionError。"""
            raise ConnectionError("secret")
    store = UnavailableStore()
    with pytest.raises(PersistenceUncertain):
        await ToolExecutor(build_registry(), store, None).execute(call("create_ticket"), context(), ignore_status)
    assert store.writes == 0


def test_is_mock_tool_reads_tags_only():
    """演示标记只能由工具自身 tags 决定，不能靠工具名猜测。参数：无；返回：None；异常：断言失败。"""
    from langchain_core.tools import tool
    from app.tools.executor import is_mock_tool

    @tool(description="带演示标记")
    def tagged() -> dict:
        """带标记工具。"""
        return {}

    @tool(description="真实工具")
    def real() -> dict:
        """真实工具。"""
        return {}

    tagged.tags = ["mock"]
    assert is_mock_tool(tagged) is True
    assert is_mock_tool(real) is False
    # 未知工具没有对象可查，必须降级为真实语义，避免把未知工具说成演示数据。
    assert is_mock_tool(None) is False


def test_registry_mock_flags_match_tool_tags():
    """注册表内每个工具的演示语义必须与其 tags 一致。参数：无；返回：None；异常：断言失败。"""
    from app.tools.executor import is_mock_tool
    expected = {"query_order": True, "query_product": True, "query_logistics": True,
                "query_faq": False, "create_ticket": False, "query_refund_policy": False}
    actual = {tool.name: is_mock_tool(tool) for tool in build_registry().all()}
    assert actual == expected


async def test_tool_returning_only_data_key_is_not_unpacked():
    """单键 data 字典必须原样保留，不能把业务字段名当协议拆包。参数：无；返回：None；异常：断言失败。"""
    @tool
    async def data_only() -> dict:
        """模拟业务数据恰好只有 data 键。参数：无；返回：业务字典；异常：无。"""
        return {"data": "这是订单摘要文本"}

    registry, store = ToolRegistry(), ResultStore()
    registry.register(data_only)
    result = await ToolExecutor(registry, store, None).execute(call("data_only"), context(), ignore_status)
    assert result.ok and result.data == {"data": "这是订单摘要文本"}


async def test_create_ticket_failure_keeps_business_error_code():
    """工单写入失败必须保留业务错误码，不能被 hasattr 兜底吞成 TOOL_FAILED。参数：无；返回：None；异常：断言失败。"""
    from app.schemas.tools import ToolError, ToolResult

    class FailingSupport:
        """受控工单服务，返回业务失败结果。"""
        async def create_ticket(self, context, description, ticket_type):
            """返回失败结果。参数：上下文、描述、类型；返回：失败 ToolResult；异常：无。"""
            return ToolResult(ok=False, error=ToolError(code="TICKET_CONFLICT", message="工单号冲突"))

    registry, store = ToolRegistry(), ResultStore()
    result = await ToolExecutor(build_registry(), store, FailingSupport()).execute(
        call("create_ticket", {"description": "x", "ticket_type": "咨询"}), context(), ignore_status,
        scope=ToolExecutionScope(authorize_write=confirmed_authorization))
    assert result.ok is False and result.error.code == "TICKET_CONFLICT"
    assert result.error.message == "工单号冲突"


async def test_write_tool_never_retries_even_after_confirmed_rollback():
    """写操作即使查无已提交结果也不得自动再次调用。参数：无；返回：None；异常：断言失败。"""
    attempts = []

    async def create_ticket(arguments, ctx, scope):
        """模拟写操作瞬时故障。参数：参数/身份/资源；返回：不返回；异常：TimeoutError。"""
        attempts.append(1)
        raise TimeoutError()

    registry, store = ToolRegistry(), ResultStore()
    registry.register_definition(ToolDefinition(ToolIdentity("builtin", None, "create_ticket"), "测试工单写入",
        {"type": "object", "properties": {"description": {"type": "string"}}}, create_ticket, "write"))
    result = await ToolExecutor(registry, store, None, max_retries=1, retry_delay_seconds=0).execute(
        call("create_ticket", {"description": "x"}), context(), ignore_status,
        scope=ToolExecutionScope(authorize_write=confirmed_authorization))
    assert len(attempts) == 1 and not result.ok and result.error.code == "TOOL_TIMEOUT"


async def test_write_tool_timeout_message_warns_about_uncertain_state():
    """写操作最终超时时提示状态未确认，不能暗示可以安全重试。参数：无；返回：None；异常：断言失败。"""
    async def create_ticket(arguments, ctx, scope):
        """模拟持续超时。参数：参数/身份/资源；返回：不返回；异常：TimeoutError。"""
        raise TimeoutError()

    registry, store = ToolRegistry(), ResultStore()
    registry.register_definition(ToolDefinition(ToolIdentity("builtin", None, "create_ticket"), "测试工单写入",
        {"type": "object", "properties": {"description": {"type": "string"}}}, create_ticket, "write"))
    result = await ToolExecutor(registry, store, None, max_retries=0).execute(
        call("create_ticket", {"description": "x"}), context(), ignore_status,
        scope=ToolExecutionScope(authorize_write=confirmed_authorization))
    assert result.ok is False and result.error.code == "TOOL_TIMEOUT"
    assert "状态未确认" in result.error.message


async def test_read_tool_still_retries_on_transient_error():
    """只读工具保留一次重试，修复写操作时不能连带砍掉读重试。参数：无；返回：None；异常：断言失败。"""
    attempts = []

    @tool
    async def query_order(order_id: str) -> dict:
        """模拟只读瞬时故障后成功。参数：订单号；返回：业务字典；异常：首次 TimeoutError。"""
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError()
        return {"order_id": order_id}

    registry, store = ToolRegistry(), ResultStore()
    registry.register(query_order)
    result = await ToolExecutor(registry, store, None, max_retries=1, retry_delay_seconds=0).execute(
        call("query_order", {"order_id": "1001"}), context(), ignore_status)
    assert len(attempts) == 2 and result.ok and result.data == {"order_id": "1001"}
