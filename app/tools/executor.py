"""统一工具生命周期：可信来源、严格参数、写确认、只读重试及独立终态审计。

可靠工具消息保存完整业务事实；审计尽力写入，无法核实提交时不保存相反结果。
确认等待与LangGraph控制信号独立于业务尝试期限，写入任何故障均不会自动再次执行。
"""

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, replace
import logging
import math
import re
import time

import anyio
import httpx
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError

from app.repositories.chat import PersistenceUncertain
from app.repositories.tool_audit import ToolAuditEntry
from app.schemas.tools import ToolContext, ToolError, ToolResult
from app.tools.contracts import ToolBinding, ToolDefinition, ToolExecutionScope, WriteAuthorization, json_hash
from app.tools.validation import validate_arguments

logger = logging.getLogger(__name__)
_LABELS = {"running": "正在查询", "retrying": "正在重试", "succeeded": "已完成", "failed": "未完成"}
_TRANSIENT_DB_CODES = frozenset({1205, 1213, 2003, 2006, 2013})


def is_mock_tool(tool: BaseTool | None) -> bool:
    """保留旧BaseTool演示标记查询。参数：工具或空；返回：是否有mock标签；异常：无。"""
    return tool is not None and "mock" in (getattr(tool, "tags", None) or [])


def _contains_control(error: BaseException) -> bool:
    """识别组内暂停或取消信号。参数：异常树；返回：是否包含控制信号；异常：无。"""
    if isinstance(error, (GraphInterrupt, asyncio.CancelledError)):
        return True
    return isinstance(error, BaseExceptionGroup) and any(_contains_control(child) for child in error.exceptions)


def _contains_uncertain(error: BaseException) -> bool:
    """识别组内任一未知提交状态。参数：异常树；返回：是否不能确认提交；异常：无。"""
    if isinstance(error, PersistenceUncertain):
        return True
    return isinstance(error, BaseExceptionGroup) and any(_contains_uncertain(child) for child in error.exceptions)


def _is_transient(error: Exception) -> bool:
    """严格识别可重试的传输叶子。参数：异常树；返回：全部为瞬时故障才True；异常：无。"""
    if isinstance(error, BaseExceptionGroup):
        # AnyIO的异常组可能混入协议或业务错误，绝不能用any放行整个组。
        return all(_is_transient(child) for child in error.exceptions)
    if isinstance(error, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError,
                          httpx.RemoteProtocolError, anyio.EndOfStream, anyio.BrokenResourceError)):
        return True
    if isinstance(error, DBAPIError):
        codes = getattr(error.orig, "args", ())
        return error.connection_invalidated or bool(codes and codes[0] in _TRANSIENT_DB_CODES)
    # McpError数字码也可能来自远端业务错误，不能仅凭-32000或408重试。
    return False


def _is_timeout(error: BaseException) -> bool:
    """识别纯超时异常或异常组。参数：异常树；返回：是否全为超时；异常：无。"""
    if isinstance(error, BaseExceptionGroup):
        return all(_is_timeout(child) for child in error.exceptions)
    return isinstance(error, (TimeoutError, httpx.TimeoutException))


@dataclass
class _Execution:
    """单调用可变计量，与注册表和检查点隔离。"""

    retries: int = 0  # 已实际开始的额外尝试。
    started_at: float | None = None  # 确认结束后的首次尝试时刻。
    duration_ms: int = 0  # 全部尝试及退避耗时。
    recovery_note: str = ""  # 异常后核实提交的审计说明。


class ToolExecutor:
    """执行已保存的单个业务申请，来源、授权和可靠事实均不由模型决定。"""

    def __init__(self, registry, chat_repository, support_repository, timeout_seconds=5,
                 max_retries=1, retry_delay_seconds=0.2, *, knowledge_timeout_seconds=None,
                 permissions=None, audit_repository=None, audit_timeout_seconds=1):
        """注入依赖及限制。

        参数：registry唯一注册表；chat_repository可靠结果仓储；support_repository业务资源；
            timeout_seconds每次尝试正秒数；max_retries只允许0或1且仅用于读取；retry_delay_seconds非负；
            knowledge_timeout_seconds可选知识工具独立期限；permissions本地权限热加载服务；
            audit_repository可选独立审计仓储；audit_timeout_seconds审计正期限。
        返回：None；异常：ValueError，非有限或越界限制。
        """
        durations = [timeout_seconds, audit_timeout_seconds]
        if knowledge_timeout_seconds is not None:
            durations.append(knowledge_timeout_seconds)
        if (any(not math.isfinite(value) or value <= 0 for value in durations)
                or max_retries not in {0, 1} or not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0):
            raise ValueError("非法工具执行限制")
        self.registry, self.chat, self.support = registry, chat_repository, support_repository
        self.timeout, self.retries, self.delay = timeout_seconds, max_retries, retry_delay_seconds
        self.knowledge_timeout = knowledge_timeout_seconds
        self.permissions, self.audit = permissions, audit_repository
        self.audit_timeout = audit_timeout_seconds

    async def execute(self, call: dict, context: ToolContext,
                      on_status: Callable[[dict], Awaitable[None]], *,
                      scope: ToolExecutionScope | None = None) -> ToolResult:
        """校验、回放或执行一次申请，再保存事实与独立审计。

        参数：call含匹配id/name/args；context服务端可信身份；on_status异步状态回调；scope可信来源与授权资源。
        返回：ToolResult，已可靠保存的原始业务结果。
        异常：ValueError，调用协议/身份非法；PersistenceUncertain，不能核实提交；GraphInterrupt及取消原样透传。
        """
        scope = replace(scope or ToolExecutionScope(), support=self.support)
        self._validate_identity(call, context, scope.binding)
        name = call["name"]
        binding = scope.binding or self._builtin_binding(name)
        scope = replace(scope, binding=binding)
        raw_args = deepcopy(call.get("args"))
        metrics = _Execution()
        mock, write = False, False

        async def emit(status: str) -> None:
            """发布安全状态。参数：状态枚举；返回：None；异常：调用方回调异常。"""
            label = "正在创建工单" if write and status == "running" else _LABELS[status]
            await on_status({"tool_call_id": context.tool_call_id, "name": name,
                             "status": status, "label": label, "is_mock": mock})

        try:
            # 回读优先于最新权限：已提交事实不能因撤权、撤下来源或schema变化被改写。
            existing = await self._read_result(context)
            if existing is not None:
                mock = existing.is_mock
                await emit("succeeded" if existing.ok else "failed")
                return existing
            if binding is None:
                logger.warning("工具申请缺少可信来源 name=%s call=%s", name, context.tool_call_id)
                result = ToolResult.failed("UNKNOWN_TOOL", "该申请没有可核实的工具来源，本次没有执行")
            else:
                row, arguments, result = self._preflight(binding, raw_args, context)
                if row is not None:
                    write, mock = row.access == "write", self._mock(row)
                if result is None and scope.rejection is not None:
                    result = scope.rejection.model_copy(deep=True)
                if result is None and write:
                    result = await self._authorize(raw_args, context, scope)
                    if result is None:
                        # interrupt恢复以及异步确认返回后必须重新读取最新定义/权限/参数。
                        row, arguments, result = self._preflight(binding, raw_args, context)
                        if row is not None:
                            write, mock = row.access == "write", self._mock(row)
                await emit("running")
                if result is None:
                    result = await self._invoke(row, arguments, context, scope, metrics, emit)
                result = result.model_copy(update={"is_mock": mock})
            result = await self._persist(context, result, metrics)
        except (GraphInterrupt, asyncio.CancelledError):
            # 暂停与取消是控制流，不能伪造一条终态结果或审计。
            raise
        except PersistenceUncertain:
            await self._audit(binding, raw_args, context, None, metrics)
            raise
        except Exception as error:
            if _contains_control(error):
                raise
            # 确认仓储等执行前故障仍需有界收尾，但不能保存一条未执行的业务失败事实。
            await self._audit(binding, raw_args, context, self._classify_failure(error, is_write=write), metrics)
            raise
        await self._audit(binding, raw_args, context, result, metrics)
        await emit("succeeded" if result.ok else "failed")
        return result

    @staticmethod
    def _validate_identity(call: dict, context: ToolContext, binding: ToolBinding | None) -> None:
        """阻止协议身份串用。参数：原申请/可信上下文/绑定；返回：None；异常：ValueError，协议非法。"""
        identifier = context.tool_call_id
        if (not isinstance(call, dict) or not isinstance(identifier, str) or not identifier
                or identifier != identifier.strip() or len(identifier) > 64 or call.get("id") != identifier
                or not all(isinstance(value, str) and value.strip() for value in
                           (context.session_id, context.user_id, context.turn_id))
                or not isinstance(call.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", call["name"])):
            raise ValueError("工具调用身份或协议不匹配")
        if binding is not None and (not isinstance(binding, ToolBinding) or binding.identity.name != call["name"]):
            raise ValueError("工具申请与保存来源不匹配")

    def _builtin_binding(self, name: str) -> ToolBinding | None:
        """仅为程序内置调用推导来源。参数：公开名；返回：内置绑定或空；异常：注册快照错误。"""
        return next((row.binding for row in self.registry.snapshot()
                     if row.name == name and row.identity.source == "builtin"), None)

    def _preflight(self, binding: ToolBinding, raw_args: object, context: ToolContext) -> tuple:
        """按原来源读取最新定义并完成校验/权限检查。

        参数：binding保存来源；raw_args原始公开参数；context可信身份。
        返回：(最新定义或空，规范化公开参数或空，拒绝结果或空)；异常：程序错误透传。
        """
        try:
            row = self.registry.get_definition(binding.identity)
        except KeyError:
            return None, None, ToolResult.failed("TOOL_UNAVAILABLE", "原工具来源当前不可用，本次没有执行")
        errors = validate_arguments(row.parameters, raw_args)
        if not isinstance(raw_args, dict) and not errors:
            errors = [{"path": "", "rule": "type", "message": "工具参数必须为对象"}]
        if errors:
            return row, None, ToolResult(ok=False, error=ToolError(code="INVALID_ARGUMENTS",
                message="工具参数不正确，请补充或修正对应字段", details=errors))
        arguments = deepcopy(raw_args)
        # 内置Pydantic业务规则补充JSON Schema无法表达的约束，例如纯空白工单描述。
        if row.legacy_tool is not None:
            fields = getattr(row.legacy_tool.args_schema, "model_fields", {})
            if fields:
                runtime = dict(arguments)
                if "user_id" in fields and "user_id" not in row.parameters.get("properties", {}):
                    runtime["user_id"] = context.user_id
                try:
                    values = row.legacy_tool.args_schema.model_validate(runtime).model_dump()
                except ValidationError:
                    return row, None, ToolResult.failed("INVALID_ARGUMENTS", "工具参数不正确，请补充有效信息")
                arguments = {key: value for key, value in values.items()
                             if key in row.parameters.get("properties", {})}
        if self.permissions is not None:
            denied = self.permissions.check(row.identity, arguments, context)
            if denied is not None:
                return row, None, denied
        elif row.identity.source == "mcp":
            return row, None, ToolResult.failed("PERMISSION_DENIED", "本地尚未授权此工具，本次没有执行")
        # 外部写入以及其他内置写入即使被错误登记，也不能借确认回调放行。
        if row.access == "write" and (row.identity.source != "builtin" or row.name != "create_ticket"):
            return row, None, ToolResult.failed("PERMISSION_DENIED", "本次不允许执行此写入操作")
        return row, arguments, None

    def _mock(self, row: ToolDefinition) -> bool:
        """从当前本地策略读取演示标记。参数：完整定义；返回：标记；异常：策略接口错误。"""
        if row.identity.source == "mcp" and self.permissions is not None:
            policy = self.permissions.policy(row.identity)
            if policy is not None:
                return bool(policy["result_policy"]["is_mock"])
        return row.result_policy.is_mock

    async def _authorize(self, arguments: dict, context: ToolContext, scope: ToolExecutionScope) -> ToolResult | None:
        """在尝试计时前检查冻结动作确认。

        参数：arguments冻结原参数；context可信身份；scope保存动作回调。
        返回：拒绝结果或None；异常：GraphInterrupt、取消或确认仓储故障透传。
        """
        if scope.authorize_write is None:
            return ToolResult.failed("NOT_CONFIRMED", "此写入尚未得到有效确认，本次没有执行")
        authorization = await scope.authorize_write(scope.binding, deepcopy(arguments), context)
        if (not isinstance(authorization, WriteAuthorization)
                or authorization.conversation_id != context.conversation_id
                or authorization.turn_id != context.turn_id or authorization.tool_call_id != context.tool_call_id
                or authorization.arguments_hash != json_hash(arguments)
                or authorization.origin not in {"chat", "action"}
                or not isinstance(authorization.action_id, str) or not authorization.action_id.strip()
                or len(authorization.action_id) > 64 or authorization.action_id != authorization.action_id.strip()
                or authorization.decision not in {"confirm", "cancel"}):
            return ToolResult.failed("NOT_CONFIRMED", "确认凭据与本次申请不匹配，本次没有执行")
        if authorization.decision == "cancel":
            return ToolResult.failed("USER_CANCELLED", "用户已取消，本次没有执行写入")
        return None

    async def _invoke(self, row: ToolDefinition, arguments: dict, context: ToolContext, scope: ToolExecutionScope,
                      metrics: _Execution, emit: Callable[[str], Awaitable[None]]) -> ToolResult:
        """执行有界业务尝试，写入故障仅核实原键。

        参数：最新定义/规范化参数/身份/资源/调用计量/状态回调。
        返回：完整ToolResult；异常：PersistenceUncertain、GraphInterrupt或取消；所有程序错误转安全故障。
        """
        write = row.access == "write"
        timeout = (self.knowledge_timeout if row.name in {"query_faq", "query_refund_policy"}
                   and self.knowledge_timeout is not None else self.timeout)
        metrics.started_at = time.monotonic()
        try:
            for attempt in range((0 if write else self.retries) + 1):
                metrics.retries = attempt
                try:
                    async with asyncio.timeout(timeout):
                        result = await row.invoke(deepcopy(arguments), context, scope)
                    if not isinstance(result, ToolResult):
                        raise TypeError("注册处理器必须返回ToolResult")
                    return result
                except (GraphInterrupt, asyncio.CancelledError):
                    raise
                except Exception as error:
                    if _contains_control(error):
                        raise
                    # 即使错误表示提交未知，也允许只读核实已提交回执；绝不再次调用写handler。
                    committed = await self._read_result(context)
                    if committed is not None:
                        metrics.recovery_note = "超时后回读核实已提交结果" if _is_timeout(error) else "异常后回读核实已提交结果"
                        return committed
                    # 并发传输可能用异常组包装提交未知；任一叶子未知都不能保存相反的失败事实。
                    if _contains_uncertain(error):
                        raise PersistenceUncertain("工具提交状态未确认，请稍后核实，勿重复提交") from None
                    if not write and _is_transient(error) and attempt < self.retries:
                        await emit("retrying")
                        await asyncio.sleep(self.delay)
                        continue
                    return self._classify_failure(error, is_write=write)
        finally:
            metrics.duration_ms = max(0, round((time.monotonic() - metrics.started_at) * 1000))
        raise AssertionError("尝试循环必须返回")

    @staticmethod
    def _classify_failure(error: Exception, *, is_write: bool) -> ToolResult:
        """隐藏异常细节且不把内部ValueError当用户参数错。参数：异常/写标记；返回：安全结果；异常：无。"""
        if _is_timeout(error):
            message = "操作超时，状态未确认，请稍后核实后再决定是否重试" if is_write else "查询超时，请稍后再试"
            return ToolResult.failed("TOOL_TIMEOUT", message)
        return ToolResult.failed("TOOL_FAILED", "工具暂时无法完成，请稍后再试")

    async def _read_result(self, context: ToolContext) -> ToolResult | None:
        """用仓储新会话核实原三键。参数：可信身份；返回：原结果或空；异常：PersistenceUncertain、控制信号。"""
        try:
            return await self.chat.get_tool_result(context)
        except GraphInterrupt:
            raise
        except Exception as error:
            if _contains_control(error):
                raise
            raise PersistenceUncertain("暂时无法核实工具处理状态，请稍后查看，勿重复提交") from None

    async def _persist(self, context: ToolContext, result: ToolResult, metrics: _Execution) -> ToolResult:
        """可靠保存，失败后只读核实真实结果。参数：身份/完整结果/计量；返回：已保存事实；异常：PersistenceUncertain。"""
        try:
            return await self.chat.save_tool_result(context, result)
        except GraphInterrupt:
            raise
        except Exception as error:
            if _contains_control(error):
                raise
            stored = await self._read_result(context)
            if stored is None:
                raise PersistenceUncertain("工具结果保存状态未能确认，请稍后核实") from None
            metrics.recovery_note = "超时后回读核实已提交结果" if _is_timeout(error) else "异常后回读核实已提交结果"
            return stored

    async def _audit(self, binding: ToolBinding | None, arguments: object, context: ToolContext,
                     result: ToolResult | None, metrics: _Execution) -> None:
        """独立有界尝试终态审计，失败只记日志。

        参数：可信绑定/原始参数/身份/可靠结果或未知/计量。
        返回：None；异常：外部取消或图控制信号透传，不泄露审计数据库错误。
        """
        if binding is None or self.audit is None:
            return
        try:
            status = "失败"
            if result is not None:
                if result.ok:
                    status = "成功"
                elif result.error is not None:
                    status = ({"invalid_arguments": "校验拦下", "permission_denied": "权限拒绝"}.get(
                        result.error.category, "超时" if result.error.code == "TOOL_TIMEOUT" else "失败"))
            summary = result.model_dump_json() if result is not None else "提交状态未确认，未保存失败结果"
            if metrics.recovery_note:
                summary = metrics.recovery_note + "；" + summary
            error_message = (result.error.message if result is not None and result.error is not None
                             else "提交状态未确认" if result is None else None)
            entry = ToolAuditEntry(conversation_id=context.conversation_id or None, tool_call_id=context.tool_call_id,
                tool_name=binding.identity.name, tool_source=binding.identity.source, mcp_server=binding.identity.server,
                arguments=deepcopy(arguments) if isinstance(arguments, dict) else None, result_summary=summary,
                status=status, error_message=error_message, retry_count=metrics.retries,
                duration_ms=min(metrics.duration_ms, 4294967295))
            async with asyncio.timeout(self.audit_timeout):
                await self.audit.record(entry)
        except GraphInterrupt:
            raise
        except Exception as error:
            if _contains_control(error):
                raise
            logger.warning("工具审计保存失败 name=%s call=%s error=%s", binding.identity.name,
                           context.tool_call_id, type(error).__name__)
