"""单工具执行、参数校验、有限重试及结果核实；不进行模型决策。"""

import asyncio
from collections.abc import Awaitable, Callable
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError
from app.repositories.chat import PersistenceUncertain
from app.schemas.tools import ToolContext, ToolError, ToolResult

MOCK_TOOLS = {"query_order", "query_product", "query_logistics"}
LABELS = {"running": "正在查询", "retrying": "正在重试", "succeeded": "已完成", "failed": "未完成"}


def failure(code: str, message: str, *, is_mock: bool = False) -> ToolResult:
    """构造安全错误。参数：固定代码、说明和演示标识；返回：结果；异常：无。"""
    return ToolResult(ok=False, error=ToolError(code=code, message=message), is_mock=is_mock)


def is_transient(error: Exception) -> bool:
    """识别允许重试的网络或数据库错误。参数：捕获异常；返回：布尔值；异常：无。"""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if isinstance(error, DBAPIError):
        return error.connection_invalidated or getattr(error.orig, "args", (None,))[0] in {1205, 1213, 2003, 2006, 2013}
    return False


class ToolExecutor:
    """执行一次决策选中的工具，并保证成功事实优先于异常表象。"""
    def __init__(self, registry, chat_repository, support_repository, timeout_seconds=5,
                 max_retries=1, retry_delay_seconds=0.2):
        """注入依赖及限制。参数：注册表、存储、超时秒数、0–1 次重试和退避；返回：None；异常：ValueError。"""
        if timeout_seconds <= 0 or max_retries not in {0, 1} or retry_delay_seconds < 0:
            raise ValueError("非法工具执行限制")
        self.registry, self.chat, self.support = registry, chat_repository, support_repository
        self.timeout, self.retries, self.delay = timeout_seconds, max_retries, retry_delay_seconds

    async def _read_result(self, context):
        """用新会话核实结果。参数：调用身份；返回：结果或空；异常：PersistenceUncertain。"""
        try:
            return await self.chat.get_tool_result(context)
        except Exception:
            raise PersistenceUncertain("暂时无法核实工具处理状态，请稍后查看，勿重复提交") from None

    async def _persist(self, context, result):
        """保存或重新核实结果。参数：身份和候选结果；返回：真实持久化结果；异常：PersistenceUncertain。"""
        try:
            return await self.chat.save_tool_result(context, result)
        except Exception:
            stored = await self._read_result(context)
            if stored is None:
                raise PersistenceUncertain("工具结果保存状态未能确认，请稍后核实") from None
            return stored

    async def execute(self, call: dict, context: ToolContext,
                      on_status: Callable[[dict], Awaitable[None]]) -> ToolResult:
        """执行并发布安全状态。参数：单个申请、可信身份、异步状态回调；返回：落库结果；异常：取消或持久化不确定。"""
        if not context.tool_call_id or call.get("id") != context.tool_call_id:
            raise ValueError("工具调用身份不匹配")
        name, mock = call.get("name", ""), call.get("name") in MOCK_TOOLS

        async def emit(status):
            """发布当前工具状态。参数：固定状态名；返回：None；异常：回调异常。"""
            label = "正在创建工单" if name == "create_ticket" and status == "running" else LABELS[status]
            await on_status({"tool_call_id": context.tool_call_id, "name": name[:100],
                             "status": status, "label": label, "is_mock": mock})

        await emit("running")
        result = await self._read_result(context)
        if result is not None:
            await emit("succeeded" if result.ok else "failed")
            return result
        try:
            business_tool = self.registry.get(name)
        except KeyError:
            result = failure("UNKNOWN_TOOL", "该工具不在可用范围内")
        else:
            try:
                arguments = business_tool.args_schema.model_validate(call.get("args")).model_dump()
            except ValidationError:
                result = failure("INVALID_ARGUMENTS", "工具参数不正确，请补充有效信息", is_mock=mock)
            else:
                config = {"configurable": {"context": context, "support_repository": self.support}}
                for attempt in range(self.retries + 1):
                    try:
                        async with asyncio.timeout(self.timeout):
                            result = await business_tool.ainvoke(arguments, config=config)
                        if not isinstance(result, ToolResult):
                            raise TypeError("工具返回格式不合法")
                        break
                    except PersistenceUncertain:
                        raise
                    except Exception as error:
                        # 异常不等于回滚：先核实原结果，再决定是否允许同键重试。
                        result = await self._read_result(context)
                        if result is not None:
                            break
                        if is_transient(error) and attempt < self.retries:
                            await emit("retrying")
                            await asyncio.sleep(self.delay)
                            continue
                        if isinstance(error, (ValidationError, ValueError)):
                            result = failure("INVALID_ARGUMENTS", "工具参数不正确，请使用原话中的有效信息", is_mock=mock)
                        elif isinstance(error, TimeoutError):
                            result = failure("TOOL_TIMEOUT", "查询超时，请稍后再试", is_mock=mock)
                        else:
                            result = failure("TOOL_FAILED", "工具暂时无法完成，请稍后再试", is_mock=mock)
                        break
        result = await self._persist(context, result)
        await emit("succeeded" if result.ok else "failed")
        return result
