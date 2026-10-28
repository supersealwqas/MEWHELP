"""统一执行器的业务边界测试：严格参数、可信授权、有限读重试和独立审计。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import anyio
import httpx
import pytest
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt

from app.repositories.chat import PersistenceUncertain
from app.schemas.tools import ToolResult
from app.tools.contracts import ToolDefinition, ToolExecutionScope, ToolIdentity, WriteAuthorization, json_hash
from app.tools.executor import ToolExecutor
from app.tools.permissions import ToolPermissions
from app.tools.registry import ToolRegistry, build_registry
from tests.test_tool_executor import ResultStore, call, context, ignore_status


class AuditSink:
    """只替换外部审计数据库，保留执行器生成的完整审计输入。"""

    def __init__(self):
        """初始化记录。参数：无；返回：None；异常：无。"""
        self.entries = []

    async def record(self, entry):
        """接收真实执行器的审计。参数：完整条目；返回：None；异常：无。"""
        self.entries.append(entry)


async def authorize(binding, arguments, ctx):
    """模拟已通过仓储核验的确认边界。参数：来源、冻结参数、身份；返回：凭据；异常：无。"""
    return WriteAuthorization(ctx.conversation_id, ctx.turn_id, "saved-action", ctx.tool_call_id,
                              json_hash(arguments), "confirm", "chat")


def definition(handler, *, name="probe", source="builtin", server=None, access="read", schema=None):
    """构建明确来源的测试业务边界。参数：处理器及定义字段；返回：定义；异常：非法身份。"""
    return ToolDefinition(ToolIdentity(source, server, name), "测试业务操作",
                          schema or {"type": "object", "additionalProperties": False}, handler, access)


async def test_invalid_arguments_returns_paired_error():
    """缺必填字段若进handler、漏保存配对事实或漏校验审计均失败。参数：无；返回：None；异常：断言失败。"""
    calls = []

    class Support:
        """检测实际业务写边界是否被调用。"""

        async def create_ticket(self, ctx, description, ticket_type):
            """记录业务调用。参数：身份/描述/类型；返回：结果；异常：无。"""
            calls.append(description)
            return ToolResult.success({"ticket_no": "T-1"})

    store, audit = ResultStore(), AuditSink()
    result = await ToolExecutor(build_registry(), store, Support(), audit_repository=audit).execute(
        call("create_ticket", {"ticket_type": "咨询"}), context(), ignore_status,
        scope=ToolExecutionScope(authorize_write=authorize))
    assert not calls and store.result == result and store.writes == 1
    assert result.error.category == "invalid_arguments"
    assert result.error.details == [{"path": "/description", "rule": "required", "message": "缺少必填字段，请补充该信息"}]
    assert audit.entries[0].status == "校验拦下" and audit.entries[0].tool_call_id == "call-1"


async def test_read_timeout_retries_once():
    """误删读重试或将两次尝试记成两条审计会失败。参数：无；返回：None；异常：断言失败。"""
    attempts = []

    async def handler(arguments, ctx, scope):
        """模拟外部超时。参数：公开参数/身份/资源；返回：不返回；异常：TimeoutError。"""
        attempts.append(1)
        raise TimeoutError()

    registry, audit, store = ToolRegistry(), AuditSink(), ResultStore()
    registry.register_definition(definition(handler))
    result = await ToolExecutor(registry, store, None, retry_delay_seconds=0, audit_repository=audit).execute(
        call("probe"), context(), ignore_status)
    assert len(attempts) == 2 and result.error.code == "TOOL_TIMEOUT"
    assert len(audit.entries) == 1 and audit.entries[0].retry_count == 1 and audit.entries[0].status == "超时"


async def test_write_timeout_never_retries():
    """未查到提交结果仍重复写入会失败。参数：无；返回：None；异常：断言失败。"""
    attempts = []

    async def handler(arguments, ctx, scope):
        """模拟写入超时。参数：公开参数/身份/资源；返回：不返回；异常：TimeoutError。"""
        attempts.append(1)
        raise TimeoutError()

    registry, audit, store = ToolRegistry(), AuditSink(), ResultStore()
    registry.register_definition(definition(handler, name="create_ticket", access="write"))
    result = await ToolExecutor(registry, store, None, retry_delay_seconds=0, audit_repository=audit).execute(
        call("create_ticket"), context(), ignore_status, scope=ToolExecutionScope(authorize_write=authorize))
    assert len(attempts) == 1 and result.error.code == "TOOL_TIMEOUT"
    assert audit.entries[0].retry_count == 0 and audit.entries[0].status == "超时"


async def test_internal_value_error_is_fault():
    """把内部程序错误误判为用户参数错误会失败。参数：无；返回：None；异常：断言失败。"""
    @tool
    async def probe() -> dict:
        """触发内部异常。参数：无；返回：不返回；异常：ValueError。"""
        raise ValueError("SECRET implementation bug")

    registry, store = ToolRegistry(), ResultStore()
    registry.register(probe)
    result = await ToolExecutor(registry, store, None).execute(call("probe"), context(), ignore_status)
    assert result.error.category == "fault" and result.error.code == "TOOL_FAILED"
    assert "SECRET" not in result.model_dump_json()


class Probe:
    """跟踪真实执行边界，可选模拟业务结果或外部故障。"""

    def __init__(self, outcome=None):
        """保存控制结果。参数：结果或异常；返回：None；异常：无。"""
        self.calls = []
        self.outcome = outcome

    async def invoke(self, arguments, ctx, scope):
        """记录副作用。参数：参数/身份/资源；返回：业务结果；异常：注入异常。"""
        self.calls.append(deepcopy(arguments))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome or ToolResult.success({"receipt": "R-1"})


def mcp_setup(tmp_path, *, enabled=True, ownership="none", schema=None):
    """准备真实权限文件与注册表。参数：临时目录及规则；返回：边界对象；异常：文件错误。"""
    path = tmp_path / "permissions.json"
    config = {"version": 1, "tools": [{"source": "mcp", "server": "original", "name": "probe",
        "enabled": enabled, "access": "read", "ownership": ownership}]}
    path.write_text(json.dumps(config), encoding="utf-8")
    probe, registry = Probe(), ToolRegistry()
    row = definition(probe.invoke, source="mcp", server="original", schema=schema)
    registry.replace_source("original", [row])
    return probe, registry, row, ToolPermissions(path), path, config


async def test_revoked_permission_blocks_saved_pending_call(tmp_path):
    """已保存来源不能永久绕过撤权。参数：临时目录；返回：None；异常：断言失败。"""
    probe, registry, row, permissions, path, config = mcp_setup(tmp_path)
    assert permissions.check(row.identity, {}, context()) is None
    config["tools"][0]["enabled"] = False
    path.write_text(json.dumps(config), encoding="utf-8")
    audit, store = AuditSink(), ResultStore()
    result = await ToolExecutor(registry, store, None, permissions=permissions, audit_repository=audit).execute(
        call("probe"), context(), ignore_status, scope=ToolExecutionScope(binding=row.binding))
    assert not probe.calls and result.error.category == "permission_denied"
    assert audit.entries[0].status == "权限拒绝" and audit.entries[0].mcp_server == "original"


async def test_new_source_cannot_take_old_call(tmp_path):
    """原来源消失后同名新服务不能接管旧申请。参数：临时目录；返回：None；异常：断言失败。"""
    probe, registry, row, permissions, _, _ = mcp_setup(tmp_path)
    registry.replace_source("original", [])
    registry.replace_source("replacement", [replace(row, identity=ToolIdentity("mcp", "replacement", "probe"))])
    audit = AuditSink()
    result = await ToolExecutor(registry, ResultStore(), None, permissions=permissions, audit_repository=audit).execute(
        call("probe"), context(), ignore_status, scope=ToolExecutionScope(binding=row.binding))
    assert not probe.calls and result.error.code == "TOOL_UNAVAILABLE"
    assert audit.entries[0].mcp_server == "original"


async def test_schema_change_revalidates_saved_arguments(tmp_path):
    """执行使用旧schema而漏掉新必填字段会失败。参数：临时目录；返回：None；异常：断言失败。"""
    probe, registry, row, permissions, _, _ = mcp_setup(tmp_path)
    registry.replace_source("original", [replace(row, parameters={"type": "object", "required": ["order_id"],
        "properties": {"order_id": {"type": "string"}}, "additionalProperties": False})])
    result = await ToolExecutor(registry, ResultStore(), None, permissions=permissions).execute(
        call("probe"), context(), ignore_status, scope=ToolExecutionScope(binding=row.binding))
    assert not probe.calls and result.error.category == "invalid_arguments"


async def test_mcp_requires_saved_binding_and_local_permission(tmp_path):
    """仅模型名称不能授权远端调用，缺本地策略也不得执行。参数：临时目录；返回：None；异常：断言失败。"""
    probe, registry, row, permissions, _, _ = mcp_setup(tmp_path)
    audit = AuditSink()
    for scope, policy in [(None, permissions), (ToolExecutionScope(binding=row.binding), None)]:
        result = await ToolExecutor(registry, ResultStore(), None, permissions=policy, audit_repository=audit).execute(
            call("probe"), context(), ignore_status, scope=scope)
        assert not result.ok
    assert not probe.calls
    assert len(audit.entries) == 1 and audit.entries[0].status == "权限拒绝"


async def test_unknown_name_does_not_fabricate_builtin_audit():
    """无法核实来源的名称不能伪造内置业务审计。参数：无；返回：None；异常：断言失败。"""
    audit = AuditSink()
    result = await ToolExecutor(ToolRegistry(), ResultStore(), None, audit_repository=audit).execute(
        call("unknown"), context(), ignore_status)
    assert not result.ok and not audit.entries


@pytest.mark.parametrize("change", [
    {"conversation_id": 2}, {"turn_id": "other"}, {"tool_call_id": "other"},
    {"arguments_hash": "fake"}, {"decision": "cancel"}, {"origin": "model"}, {"action_id": ""},
])
async def test_authorization_is_bound_to_frozen_call(change):
    """跨会话、跨轮、跨调用、改参或取消的凭据不能执行。参数：错误凭据字段；返回：None；异常：断言失败。"""
    probe, registry, audit = Probe(), ToolRegistry(), AuditSink()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write"))

    async def incorrect(binding, arguments, ctx):
        """模拟不匹配的保存动作。参数：来源/参数/身份；返回：不匹配凭据；异常：无。"""
        return replace(await authorize(binding, arguments, ctx), **change)

    result = await ToolExecutor(registry, ResultStore(), None, audit_repository=audit).execute(
        call("create_ticket"), context(), ignore_status, scope=ToolExecutionScope(authorize_write=incorrect))
    assert not probe.calls and result.error.category == "permission_denied"
    assert audit.entries[0].status == "权限拒绝"


async def test_write_without_authorizer_is_denied():
    """默认写允许会失败。参数：无；返回：None；异常：断言失败。"""
    probe, registry = Probe(), ToolRegistry()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write"))
    result = await ToolExecutor(registry, ResultStore(), None).execute(call("create_ticket"), context(), ignore_status)
    assert not probe.calls and result.error.code == "NOT_CONFIRMED"


async def test_authorizer_cannot_mutate_frozen_arguments():
    """确认回调改写参数不能改变保存申请或业务执行参数。参数：无；返回：None；异常：断言失败。"""
    probe, registry = Probe(), ToolRegistry()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write",
        schema={"type": "object", "properties": {"description": {"type": "string"}}}))
    request = call("create_ticket", {"description": "原文"})

    async def mutate(binding, arguments, ctx):
        """尝试改写确认参数副本。参数：来源/参数/身份；返回：改参凭据；异常：无。"""
        arguments["description"] = "changed"
        return await authorize(binding, arguments, ctx)

    result = await ToolExecutor(registry, ResultStore(), None).execute(request, context(), ignore_status,
        scope=ToolExecutionScope(authorize_write=mutate))
    assert not result.ok and not probe.calls and request["args"] == {"description": "原文"}


async def test_confirmation_rechecks_latest_definition():
    """等待期间工具定义变更后仍执行旧schema会失败。参数：无；返回：None；异常：断言失败。"""
    probe, registry = Probe(), ToolRegistry()
    row = definition(probe.invoke, name="create_ticket", access="write")
    registry.register_definition(row)

    async def changed(binding, arguments, ctx):
        """在等待边界替换最新读取快照。参数：来源/参数/身份；返回：凭据；异常：无。"""
        registry._builtin[row.name] = replace(row, parameters={"type": "object", "required": ["description"]})
        registry._rebuild()
        return await authorize(binding, arguments, ctx)

    result = await ToolExecutor(registry, ResultStore(), None).execute(call("create_ticket"), context(), ignore_status,
        scope=ToolExecutionScope(authorize_write=changed))
    assert not probe.calls and result.error.category == "invalid_arguments"


async def test_commit_then_timeout_preserves_receipt():
    """超时表象覆盖真实回执或重复业务写入会失败。参数：无；返回：None；异常：断言失败。"""
    store, audit, registry, attempts = ResultStore(), AuditSink(), ToolRegistry(), []

    async def committed(arguments, ctx, scope):
        """模拟原子提交后失去确认。参数：参数/身份/资源；返回：不返回；异常：TimeoutError。"""
        attempts.append(1)
        await store.save_tool_result(ctx, ToolResult.success({"ticket_no": "T-real"}))
        raise TimeoutError()

    registry.register_definition(definition(committed, name="create_ticket", access="write"))
    result = await ToolExecutor(registry, store, None, audit_repository=audit).execute(call("create_ticket"),
        context(), ignore_status, scope=ToolExecutionScope(authorize_write=authorize))
    assert result.data == {"ticket_no": "T-real"} and len(attempts) == store.writes == 1
    assert audit.entries[0].status == "成功" and "超时" in audit.entries[0].result_summary


@pytest.mark.parametrize("outcome", [ConnectionError("SECRET audit"), "timeout"])
async def test_audit_failure_does_not_change_business_result(outcome):
    """审计失败或阻塞改变已提交业务结果会失败。参数：审计故障；返回：None；异常：断言失败。"""
    class FailedAudit:
        """只替换外部数据库审计边界。"""

        async def record(self, entry):
            """注入审计故障。参数：审计条目；返回：不返回；异常：网络错误或超时取消。"""
            if isinstance(outcome, Exception):
                raise outcome
            await asyncio.Event().wait()

    probe, registry, store = Probe(), ToolRegistry(), ResultStore()
    registry.register_definition(definition(probe.invoke))
    result = await asyncio.wait_for(ToolExecutor(registry, store, None, audit_repository=FailedAudit(),
        audit_timeout_seconds=0.01).execute(call("probe"), context(), ignore_status), 1)
    assert result.data == {"receipt": "R-1"} and store.result == result and len(probe.calls) == 1


async def test_replayed_result_does_not_reaudit(tmp_path):
    """撤权后回放抹掉已提交事实、再次调用或补插审计会失败。参数：临时目录；返回：None；异常：断言失败。"""
    probe, registry, row, permissions, path, config = mcp_setup(tmp_path)
    store, audit = ResultStore(), AuditSink()
    executor = ToolExecutor(registry, store, None, permissions=permissions, audit_repository=audit)
    scope = ToolExecutionScope(binding=row.binding)
    first = await executor.execute(call("probe"), context(), ignore_status, scope=scope)
    config["tools"][0]["enabled"] = False
    path.write_text(json.dumps(config), encoding="utf-8")
    registry.replace_source("original", [])
    second = await executor.execute(call("probe"), context(), ignore_status, scope=scope)
    assert first == second and second.ok and len(probe.calls) == len(audit.entries) == 1


@pytest.mark.parametrize("boundary", ["authorization", "handler", "persistence"])
@pytest.mark.parametrize("signal", ["interrupt", "grouped_interrupt", "cancel", "grouped_cancel"])
async def test_interrupt_has_no_terminal_audit(boundary, signal):
    """暂停或取消被吞成错误、保存假终态或改变异常组会失败。参数：边界/控制信号；返回：None；异常：断言失败。"""
    error = asyncio.CancelledError() if "cancel" in signal else GraphInterrupt(())
    if signal.startswith("grouped_"):
        error = BaseExceptionGroup("transport", [error])

    class InterruptedStore(ResultStore):
        """在可靠结果保存入口注入控制信号，检测执行器是否误写终态审计。"""

        async def save_tool_result(self, ctx, result):
            """模拟保存前暂停或取消。参数：身份/结果；返回：不返回；异常：指定控制信号。"""
            raise error

    probe = Probe(error if boundary == "handler" else None)
    registry, audit, states = ToolRegistry(), AuditSink(), []
    store = InterruptedStore() if boundary == "persistence" else ResultStore()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write"))

    async def interrupted(binding, arguments, ctx):
        """发出图暂停信号。参数：来源/参数/身份；返回：不返回；异常：GraphInterrupt。"""
        raise error

    async def status(frame):
        """记录可见终态。参数：帧；返回：None；异常：无。"""
        states.append(frame["status"])

    with pytest.raises(type(error)) as raised:
        await ToolExecutor(registry, store, None, audit_repository=audit).execute(call("create_ticket"),
            context(), status, scope=ToolExecutionScope(authorize_write=interrupted if boundary == "authorization" else authorize))
    assert raised.value is error
    assert not audit.entries and store.writes == 0 and not set(states) & {"failed", "succeeded"}


@pytest.mark.parametrize("errors,retries", [([TimeoutError(), ConnectionError()], 1),
                                          ([TimeoutError(), ValueError("SECRET")], 0)])
async def test_group_retries_only_when_all_leaves_are_transient(errors, retries):
    """MCP异常组中夹杂非瞬时错误仍重试会失败。参数：叶子错误/期望重试；返回：None；异常：断言失败。"""
    probe, registry, audit = Probe(ExceptionGroup("MCP", errors)), ToolRegistry(), AuditSink()
    registry.register_definition(definition(probe.invoke))
    result = await ToolExecutor(registry, ResultStore(), None, audit_repository=audit, retry_delay_seconds=0).execute(
        call("probe"), context(), ignore_status)
    assert not result.ok and len(probe.calls) == retries + 1 and audit.entries[0].retry_count == retries


async def test_uncertain_write_is_audited_without_saving_failure():
    """无法核实提交时保存假失败或再次写入会失败。参数：无；返回：None；异常：断言失败。"""
    probe, registry, audit = Probe(PersistenceUncertain("SECRET")), ToolRegistry(), AuditSink()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write"))
    store = ResultStore()
    with pytest.raises(PersistenceUncertain):
        await ToolExecutor(registry, store, None, audit_repository=audit).execute(call("create_ticket"), context(),
            ignore_status, scope=ToolExecutionScope(authorize_write=authorize))
    assert len(probe.calls) == 1 and store.writes == 0 and audit.entries[0].status == "失败"
    assert "SECRET" not in audit.entries[0].model_dump_json()


async def test_rejection_passes_through_persistence_and_audit():
    """预算/重复跳过绕开统一留痕会失败。参数：无；返回：None；异常：断言失败。"""
    probe, registry, store, audit = Probe(), ToolRegistry(), ResultStore(), AuditSink()
    registry.register_definition(definition(probe.invoke))
    rejection = ToolResult.failed("TOOL_BUDGET_EXCEEDED", "本轮调用已达上限")
    result = await ToolExecutor(registry, store, None, audit_repository=audit).execute(call("probe"), context(),
        ignore_status, scope=ToolExecutionScope(rejection=rejection))
    assert result == rejection and store.result == rejection and not probe.calls
    assert len(audit.entries) == 1 and audit.entries[0].status == "失败"


async def test_duration_excludes_confirmation_and_audit(monkeypatch):
    """把用户等待或审计耗时计入工具时长会失败。参数：替换器；返回：None；异常：断言失败。"""
    from app.tools import executor as module
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    audit = AuditSink()

    async def delayed_confirmation(binding, arguments, ctx):
        """模拟等待三分钟。参数：来源/参数/身份；返回：凭据；异常：无。"""
        clock[0] += 180
        return await authorize(binding, arguments, ctx)

    async def handler(arguments, ctx, scope):
        """模拟执行125毫秒。参数：参数/身份/资源；返回：结果；异常：无。"""
        clock[0] += 0.125
        return ToolResult.success({"ticket_no": "T-real"})

    registry = ToolRegistry()
    registry.register_definition(definition(handler, name="create_ticket", access="write"))
    await ToolExecutor(registry, ResultStore(), None, audit_repository=audit).execute(call("create_ticket"), context(),
        ignore_status, scope=ToolExecutionScope(authorize_write=delayed_confirmation))
    assert audit.entries[0].duration_ms == 125


@pytest.mark.parametrize("nested", [False, True])
async def test_wrapped_persistence_uncertain_never_saves_failure(nested):
    """嵌套混合异常组的未知提交状态不能落成相反的失败事实。参数：是否嵌套混合；返回：None；异常：断言失败。"""
    error = ExceptionGroup("transport", [PersistenceUncertain("SECRET")])
    if nested:
        error = ExceptionGroup("mixed", [TimeoutError(), error, ValueError("SECRET")])
    probe = Probe(error)
    registry, store, audit = ToolRegistry(), ResultStore(), AuditSink()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write"))
    with pytest.raises(PersistenceUncertain):
        await ToolExecutor(registry, store, None, audit_repository=audit).execute(call("create_ticket"), context(),
            ignore_status, scope=ToolExecutionScope(authorize_write=authorize))
    assert len(probe.calls) == 1 and store.writes == 0 and len(audit.entries) == 1


async def test_authorizer_fault_has_bounded_audit_without_business_write():
    """确认仓储故障跳过异常收尾审计会失败。参数：无；返回：None；异常：断言失败。"""
    probe, registry, store, audit = Probe(), ToolRegistry(), ResultStore(), AuditSink()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write"))

    async def failed_authorization(binding, arguments, ctx):
        """模拟确认仓储程序错误。参数：来源/参数/身份；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("SECRET callback failure")

    with pytest.raises(RuntimeError):
        await ToolExecutor(registry, store, None, audit_repository=audit).execute(call("create_ticket"), context(),
            ignore_status, scope=ToolExecutionScope(authorize_write=failed_authorization))
    assert not probe.calls and store.writes == 0 and len(audit.entries) == 1
    assert audit.entries[0].status == "失败" and "SECRET" not in audit.entries[0].model_dump_json()


async def test_authorizer_fault_audit_timeout_preserves_original_error():
    """确认故障的收尾审计卡死或遮蔽原错误会失败。参数：无；返回：None；异常：断言失败。"""
    class SlowAudit(AuditSink):
        """模拟审计数据库已收请求却一直不响应。"""

        async def record(self, entry):
            """记录安全审计输入后阻塞。参数：条目；返回：不返回；异常：超时取消。"""
            await super().record(entry)
            await asyncio.Event().wait()

    probe, registry, store, audit = Probe(), ToolRegistry(), ResultStore(), SlowAudit()
    registry.register_definition(definition(probe.invoke, name="create_ticket", access="write"))
    error = RuntimeError("SECRET callback failure")

    async def failed_authorization(binding, arguments, ctx):
        """模拟确认仓储故障。参数：来源/参数/身份；返回：不返回；异常：RuntimeError。"""
        raise error

    with pytest.raises(RuntimeError) as raised:
        await asyncio.wait_for(ToolExecutor(registry, store, None, audit_repository=audit,
            audit_timeout_seconds=0.01).execute(call("create_ticket"), context(), ignore_status,
            scope=ToolExecutionScope(authorize_write=failed_authorization)), timeout=1)
    assert raised.value is error and not probe.calls and store.writes == 0
    assert len(audit.entries) == 1 and "SECRET" not in audit.entries[0].model_dump_json()


async def test_persistence_timeout_recovery_is_noted_in_audit():
    """可靠结果保存提交后超时，回读成功却漏记核实说明会失败。参数：无；返回：None；异常：断言失败。"""
    class AckLostStore(ResultStore):
        """模拟结果提交成功但确认丢失的数据库边界。"""

        async def save_tool_result(self, ctx, result):
            """保存事实再丢失确认。参数：身份/结果；返回：不返回；异常：TimeoutError。"""
            await super().save_tool_result(ctx, result)
            raise TimeoutError()

    probe, registry, store, audit = Probe(), ToolRegistry(), AckLostStore(), AuditSink()
    registry.register_definition(definition(probe.invoke))
    result = await ToolExecutor(registry, store, None, audit_repository=audit).execute(call("probe"), context(), ignore_status)
    assert result.ok and store.writes == len(probe.calls) == 1
    assert audit.entries[0].status == "成功" and "超时" in audit.entries[0].result_summary


@pytest.mark.parametrize("error", [httpx.LocalProtocolError("bad request"), anyio.ClosedResourceError()])
async def test_non_network_protocol_and_local_closed_errors_never_retry(error):
    """把所有TransportError或本地关闭误当远端瞬断会失败。参数：非重试异常；返回：None；异常：断言失败。"""
    probe, registry, audit = Probe(error), ToolRegistry(), AuditSink()
    registry.register_definition(definition(probe.invoke))
    result = await ToolExecutor(registry, ResultStore(), None, audit_repository=audit).execute(call("probe"), context(), ignore_status)
    assert not result.ok and len(probe.calls) == 1 and audit.entries[0].retry_count == 0


async def test_empty_query_is_successful_without_retry():
    """查询落空被误判故障或反复查询会失败。参数：无；返回：None；异常：断言失败。"""
    probe, registry, audit = Probe(ToolResult.success({"hits": [], "message": "没有匹配记录"})), ToolRegistry(), AuditSink()
    registry.register_definition(definition(probe.invoke))
    result = await ToolExecutor(registry, ResultStore(), None, audit_repository=audit).execute(call("probe"), context(), ignore_status)
    assert result.ok and result.data["hits"] == [] and len(probe.calls) == 1
    assert audit.entries[0].status == "成功" and audit.entries[0].retry_count == 0
