"""单工具执行、参数校验、有限重试及结果核实；不进行模型决策。

职责边界：
- 本模块只负责「一次工具调用」的生命周期，不关心聊天编排、SSE 编码或模型选择。
- 所有工具必须返回 dict；业务拒绝通过 error/code 两字段表达，由本模块转为失败事实。
- 工具执行成功与否一律落库，异常不得覆盖已提交的成功事实。
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError
from app.repositories.chat import PersistenceUncertain
from app.schemas.tools import CODE_FIELD, ERROR_FIELD, ToolContext, ToolResult, is_business_rejection

# 状态标签。running 阶段可按工具覆写，见 _status_label。
_LABELS = {"running": "正在查询", "retrying": "正在重试", "succeeded": "已完成", "failed": "未完成"}

# 写操作的 running 文案与只读查询不同：用户需要知道系统正在建单而非查询。
_WRITE_LABELS = {"running": "正在创建工单"}

# 允许重试的数据库瞬时错误码：锁等待超时、死锁、连接类故障。
_TRANSIENT_DB_CODES = frozenset({1205, 1213, 2003, 2006, 2013})

# 需要在执行前由服务端注入、模型不可见的参数名。执行器按此判定是否注入身份。
_INJECTED_IDENTITY_FIELD = "user_id"

# 写操作标记，与工具定义处的 tags 对应。写操作只在核实到未提交后才重试。
_WRITE_TAG = "write"


def is_mock_tool(tool: BaseTool | None) -> bool:
    """判定是否为演示工具。

    演示语义的唯一真相来源是工具自身的 tags：标记了 'mock' 即为演示数据。
    未标记的工具（含未知工具）一律按真实数据处理，避免把真实数据误标为演示。

    参数:
        tool: 待检查的 LangChain 工具对象；未知工具传入 None。
    返回值:
        bool: tags 含 'mock' 时为 True，否则为 False。
    异常:
        无主动抛出的业务异常。
    """
    if tool is None:
        return False
    return "mock" in (getattr(tool, "tags", None) or [])


def _is_write_tool(tool: BaseTool | None) -> bool:
    """判定是否为写操作工具。

    写操作会改变外部状态，因此重试策略与超时文案都需区别对待。

    参数:
        tool: 待检查的 LangChain 工具对象；未知工具传入 None。
    返回值:
        bool: tags 含 'write' 时为 True，否则为 False。
    异常:
        无主动抛出的业务异常。
    """
    if tool is None:
        return False
    return _WRITE_TAG in (getattr(tool, "tags", None) or [])


def _is_transient(error: Exception) -> bool:
    """识别允许重试的网络或数据库瞬时故障。

    参数:
        error: 工具执行过程中捕获的异常。
    返回值:
        bool: 属于瞬时故障且值得重试时为 True。
    异常:
        无主动抛出的业务异常。
    """
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if isinstance(error, DBAPIError):
        code = getattr(error.orig, "args", (None,))[0]
        return error.connection_invalidated or code in _TRANSIENT_DB_CODES
    return False


class ToolExecutor:
    """执行一次决策选中的工具，并保证成功事实优先于异常表象。"""

    def __init__(self, registry, chat_repository, support_repository, timeout_seconds=5,
                 max_retries=1, retry_delay_seconds=0.2):
        """注入依赖及执行限制。

        参数:
            registry: 工具注册表，提供按名查找。
            chat_repository: 工具结果读写仓储。
            support_repository: 业务仓储，通过 RunnableConfig 传给工具。
            timeout_seconds: 单次尝试超时秒数，必须为正。
            max_retries: 允许的重试次数，只支持 0 或 1。
            retry_delay_seconds: 重试前的退避秒数，不得为负。
        返回值:
            None。
        异常:
            ValueError: 执行限制参数非法。
        """
        if timeout_seconds <= 0 or max_retries not in {0, 1} or retry_delay_seconds < 0:
            raise ValueError("非法工具执行限制")
        self.registry, self.chat, self.support = registry, chat_repository, support_repository
        self.timeout, self.retries, self.delay = timeout_seconds, max_retries, retry_delay_seconds

    async def execute(self, call: dict, context: ToolContext,
                      on_status: Callable[[dict], Awaitable[None]]) -> ToolResult:
        """执行并发布安全状态。

        参数:
            call: 模型的单条工具申请，必须含 id/name/args。
            context: 可信调用身份，其 tool_call_id 必须与申请一致。
            on_status: 异步状态回调，接收 running/retrying/succeeded/failed 帧。
        返回值:
            ToolResult: 已落库的工具结果；重复调用返回首次结果。
        异常:
            ValueError: 调用身份不匹配。
            PersistenceUncertain: 结果保存状态无法确认。
        """
        if not context.tool_call_id or call.get("id") != context.tool_call_id:
            raise ValueError("工具调用身份不匹配")

        name = call.get("name", "")
        tool = self._lookup(name)
        mock = is_mock_tool(tool)

        async def emit(status: str) -> None:
            """发布一帧工具状态。

            参数:
                status: running/retrying/succeeded/failed 之一。
            返回值:
                None。
            异常:
                回调异常向外传播。
            """
            await on_status({"tool_call_id": context.tool_call_id, "name": name[:100],
                             "status": status, "label": self._label(tool, status), "is_mock": mock})

        await emit("running")

        # 幂等：同键结果已存在时直接复用，绝不重复执行写操作。
        existing = await self._read_result(context)
        if existing is not None:
            await emit("succeeded" if existing.ok else "failed")
            return existing

        result = await self._run(name, tool, call.get("args"), context, mock, emit)
        result = await self._persist(context, result)
        await emit("succeeded" if result.ok else "failed")
        return result

    def _lookup(self, name: str) -> BaseTool | None:
        """按名查找工具，未注册返回 None 而不是抛异常。

        参数:
            name: 模型给出的工具名。
        返回值:
            BaseTool | None: 已注册工具，未知工具为 None。
        异常:
            无主动抛出的业务异常。
        """
        try:
            return self.registry.get(name)
        except KeyError:
            return None

    def _label(self, tool: BaseTool | None, status: str) -> str:
        """解析状态文案，写操作使用独立的 running 措辞。

        参数:
            tool: 目标工具，用于判断是否写操作；未知工具为 None。
            status: 状态名。
        返回值:
            str: 面向用户的中文状态文案。
        异常:
            无主动抛出的业务异常。
        """
        if status == "running" and _is_write_tool(tool):
            return _WRITE_LABELS["running"]
        return _LABELS[status]

    async def _run(self, name: str, tool: BaseTool | None, raw_args: Any, context: ToolContext,
                   mock: bool, emit: Callable[[str], Awaitable[None]]) -> ToolResult:
        """执行工具并归一化所有可能的结局。

        参数:
            name: 工具名。
            tool: 已解析的工具对象，未知工具为 None。
            raw_args: 模型提交的原始参数，形状不可信。
            context: 可信调用身份，提供注入的身份字段。
            mock: 该工具是否为演示工具。
            emit: 状态回调，用于发布重试帧。
        返回值:
            ToolResult: 成功、业务拒绝或执行失败的统一结果。
        异常:
            PersistenceUncertain: 结果核实过程中数据库不可用。
        """
        if tool is None:
            return ToolResult.failed("UNKNOWN_TOOL", "该工具不在可用范围内")
        try:
            arguments = self._prepare_arguments(tool, raw_args, context)
        except (ValidationError, ValueError):
            return ToolResult.failed("INVALID_ARGUMENTS", "工具参数不正确，请补充有效信息", is_mock=mock)
        return await self._invoke(tool, arguments, context, mock, emit)

    def _prepare_arguments(self, tool: BaseTool, raw_args: Any, context: ToolContext) -> dict:
        """校验模型参数并注入服务端身份。

        两件事必须同时成立：模型只能提交公开字段；身份只能由服务端注入。
        完整 args_schema 而非可见子模型参与校验，因为可见子模型会丢失严格校验配置。

        参数:
            tool: 目标工具。
            raw_args: 模型提交的原始参数。
            context: 可信调用身份。
        返回值:
            dict: 已规范化、可直接传给工具的参数字典。
        异常:
            ValueError: 参数不是字典，或夹带了模型不可见的字段。
            ValidationError: 参数不符合工具 Schema。
        """
        visible = tool.tool_call_schema.model_fields
        if not isinstance(raw_args, dict) or raw_args.keys() - visible.keys():
            raise ValueError("模型只能提交公开的业务参数")
        arguments = dict(raw_args)
        if _INJECTED_IDENTITY_FIELD in tool.args_schema.model_fields and _INJECTED_IDENTITY_FIELD not in visible:
            arguments[_INJECTED_IDENTITY_FIELD] = context.user_id
        return tool.args_schema.model_validate(arguments).model_dump()

    async def _invoke(self, tool: BaseTool, arguments: dict, context: ToolContext,
                      mock: bool, emit: Callable[[str], Awaitable[None]]) -> ToolResult:
        """带超时与有限重试地调用工具，并把返回字典归一化。

        重试策略按工具语义区分，两者都只重试瞬时故障：
        - 只读工具：超时/网络抖动重试一次。
        - 写操作：仅在核实到「本次未提交」后才重试；提交状态未知时绝不重试，
          因为重复提交会产生第二张工单，比直接失败更糟。

        参数:
            tool: 目标工具。
            arguments: 已校验的完整参数。
            context: 可信调用身份，供工具读取。
            mock: 是否为演示工具。
            emit: 状态回调，重试前发布 retrying。
        返回值:
            ToolResult: 归一化后的结果。
        异常:
            PersistenceUncertain: 结果核实过程中数据库不可用。
        """
        config = {"configurable": {"context": context, "support_repository": self.support}}
        is_write = _is_write_tool(tool)
        for attempt in range(self.retries + 1):
            try:
                async with asyncio.timeout(self.timeout):
                    payload = await tool.ainvoke(arguments, config=config)
                return self._normalize(payload, mock)
            except PersistenceUncertain:
                raise
            except Exception as error:
                # 异常不等于回滚：先核实是否已提交，再决定能否重试。
                # 核实结果为空说明本次确定未提交，重试是安全的；核实失败会抛 PersistenceUncertain。
                committed = await self._read_result(context)
                if committed is not None:
                    return committed
                if _is_transient(error) and attempt < self.retries:
                    await emit("retrying")
                    await asyncio.sleep(self.delay)
                    continue
                return self._classify_failure(error, mock, is_write=is_write)
        raise AssertionError("重试循环必须返回结果")

    @staticmethod
    def _normalize(payload: Any, mock: bool) -> ToolResult:
        """把工具返回的业务字典归一化为统一结果。

        参数:
            payload: 工具返回值，协议要求必须是 dict。
            mock: 是否为演示工具。
        返回值:
            ToolResult: 业务拒绝转为失败事实，其余视为成功数据。
        异常:
            TypeError: 返回类型或业务错误格式不符合协议。
        """
        if not isinstance(payload, dict):
            raise TypeError("工具返回格式不合法，所有业务工具必须返回 dict")
        if is_business_rejection(payload):
            return ToolResult.failed(payload[CODE_FIELD], payload[ERROR_FIELD], is_mock=mock)
        return ToolResult.success(payload, is_mock=mock)

    @staticmethod
    def _classify_failure(error: Exception, mock: bool, *, is_write: bool = False) -> ToolResult:
        """把异常映射为安全的失败结果，不泄露异常原文。

        参数:
            error: 捕获的异常。
            mock: 是否为演示工具。
            is_write: 失败发生在写操作上，超时需要提示用户先核实而非直接重试。
        返回值:
            ToolResult: 带稳定错误码的失败结果。
        异常:
            无主动抛出的业务异常。
        """
        if isinstance(error, (ValidationError, ValueError)):
            return ToolResult.failed("INVALID_ARGUMENTS", "工具参数不正确，请使用原话中的有效信息", is_mock=mock)
        if isinstance(error, TimeoutError):
            message = "操作超时，状态未确认，请稍后核实后再决定是否重试" if is_write else "查询超时，请稍后再试"
            return ToolResult.failed("TOOL_TIMEOUT", message, is_mock=mock)
        return ToolResult.failed("TOOL_FAILED", "工具暂时无法完成，请稍后再试", is_mock=mock)

    async def _read_result(self, context: ToolContext) -> ToolResult | None:
        """用新会话核实已提交结果。

        参数:
            context: 调用身份。
        返回值:
            ToolResult | None: 已提交结果，或 None 表示尚未提交。
        异常:
            PersistenceUncertain: 数据库不可用，无法确认状态。
        """
        try:
            return await self.chat.get_tool_result(context)
        except Exception:
            raise PersistenceUncertain("暂时无法核实工具处理状态，请稍后查看，勿重复提交") from None

    async def _persist(self, context: ToolContext, result: ToolResult) -> ToolResult:
        """保存结果，保存失败时改用核实路径。

        参数:
            context: 调用身份。
            result: 待保存的结果。
        返回值:
            ToolResult: 真实落库的结果。
        异常:
            PersistenceUncertain: 保存与核实都无法确认。
        """
        try:
            return await self.chat.save_tool_result(context, result)
        except Exception:
            stored = await self._read_result(context)
            if stored is None:
                raise PersistenceUncertain("工具结果保存状态未能确认，请稍后核实") from None
            return stored
