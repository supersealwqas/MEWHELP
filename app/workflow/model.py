"""受预算限制的工作流模型边界：严格业务/终止控制工具与无工具最终正文真流式分离。"""

import json
from contextlib import aclosing
from copy import deepcopy
from typing import AsyncIterator

import anyio
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from pydantic import ValidationError

from app.config import settings
from app.core.llm import decision_message, get_chat_model
from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
from app.core.prompts import ANSWER_NUDGE
from app.core.structured import StructuredOutputError, bind_output_tool, output_chunk_size, output_tool, parse_tool_output
from app.schemas.chat import InvalidToolDecision
from app.schemas.workflow import AgentDecision, IntentResult, UnderstandingResult
from app.workflow.prompts import ANSWER_PROMPT, DECISION_NUDGE, DECISION_PROMPT, INTENT_PROMPT, UNDERSTANDING_PROMPT
from app.workflow.routing import _understanding_outcome, original_understanding
from app.workflow.context import ModelContext, log_model_context
from app.tools.contracts import ToolDefinition

_CONTROL_NAME = "submit_decision"
_CONTROL_SCHEMA = {"type": "function", "function": {
    "name": _CONTROL_NAME,
    "description": "提交内部终止决定：已有事实足够时回答，缺必要信息时追问；不执行业务操作。",
    "strict": True,
    "parameters": {"type": "object", "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": ["answer", "clarify"], "description": "回答或追问"},
            "suggested_actions": {"type": "array", "maxItems": 2,
                "items": {"type": "string", "enum": ["handoff", "create_ticket"]},
                "description": "最多两个不重复的建议动作；没有建议时为空数组"}},
        "required": ["kind", "suggested_actions"]}}}


class WorkflowOutputError(ValueError):
    """模型输出截断、拒绝、超长或无有效正文，不能作为正常完成发布。"""


def understanding_tool_schema() -> dict:
    """生成理解绑定与预算共用的严格工具。参数：无；返回：submit_understanding定义；异常：schema转换错误。"""
    return output_tool(UnderstandingResult, "submit_understanding")


def intent_tool_schema() -> dict:
    """生成分类绑定与完整输入预算共用的结果工具。参数：无；返回：严格submit_intent定义；异常：schema转换错误。"""
    return output_tool(IntentResult, "submit_intent")


def decision_tool_schemas(tools: list[ToolDefinition]) -> list[dict]:
    """按本次真实快照生成绑定与预算共用的公开定义。

    参数：tools当前授权快照，名称必须唯一且不能占用控制名。
    返回：list[dict]，业务schema后追加独立的submit_decision控制schema。
    异常：InvalidToolDecision，名称重复或占用控制协议。
    """
    if len({tool.name for tool in tools}) != len(tools) or any(tool.name == _CONTROL_NAME for tool in tools):
        raise InvalidToolDecision()
    # 不向业务schema强加strict，避免改变外部工具可选字段；权限与参数仍由执行器重验。
    return [*[{"type": "function", "function": {"name": tool.name, "description": tool.description,
              "parameters": deepcopy(tool.parameters)}} for tool in tools], deepcopy(_CONTROL_SCHEMA)]


def _parse_terminal_decision(payload: dict) -> AgentDecision:
    """严格校验唯一控制调用的终止参数。

    参数：payload已完成JSON解析的submit_decision参数，只允许kind和suggested_actions。
    返回：AgentDecision，回答或追问且无业务工具申请。
    异常：InvalidToolDecision，字段或取值非法，携带有界的参数JSON诊断。
    """
    category = "shape"
    try:
        if set(payload) != {"kind", "suggested_actions"} or payload["kind"] not in ("answer", "clarify"):
            raise ValueError("终止决定必须精确包含 kind 和 suggested_actions")
        category = "schema"
        return AgentDecision.model_validate(payload)
    except (ValueError, TypeError, ValidationError) as error:
        raise InvalidToolDecision(raw_output=json.dumps(payload, ensure_ascii=False), validation_category=category) from error


def _text(chunk) -> str:
    """提取正文并排除推理块。参数：chunk 消息或分片；返回：str；异常：无。"""
    return chunk.content if isinstance(chunk.content, str) else "".join(
        block if isinstance(block, str) else block["text"] for block in chunk.content
        if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text"
                                     and isinstance(block.get("text"), str)))


def _has_calls(chunk) -> bool:
    """识别所有工具申请形态。参数：chunk 消息或分片；返回：bool；异常：无。"""
    return bool(chunk.tool_calls or chunk.invalid_tool_calls or getattr(chunk, "tool_call_chunks", None))


def _check_finish(reason: str | None, *, tools: bool = False) -> None:
    """拒绝截断和非正常终止。参数：reason 上游原因、tools 是否有申请；返回：None；异常：WorkflowOutputError。"""
    if reason not in ("tool_calls" if tools else "stop",):
        raise WorkflowOutputError("模型输出未正常结束")


class WorkflowModelAdapter:
    """不保存单轮账本的模型适配器；每个调用显式接收请求独占 ledger。"""

    def __init__(self, *, model_factory=None, stream_usage: bool | None = None,
                 intent_model: str | None = None) -> None:
        """参数：model_factory外部模型边界、stream_usage计量开关、intent_model分类模型覆盖；返回：None；异常：无。"""
        self._factory = model_factory or get_chat_model
        self._stream_usage = settings.workflow_stream_usage if stream_usage is None else stream_usage
        self._intent_model = settings.intent_model if intent_model is None else intent_model

    def _model(self, output_limit: int, *, model_name: str | None = None):
        """创建有界请求模型。参数：output_limit输出上限、model_name可选分类模型；返回：兼容聊天模型；异常：配置错误。"""
        options = {"model": model_name} if model_name is not None else {}
        return self._factory(streaming=True, temperature=0.0, max_tokens=output_limit,
                             max_retries=1, request_timeout=60.0, stream_usage=self._stream_usage, **options)

    async def _gather(self, messages: list, model, ledger: BudgetLedger, stage: str, *, source=None, tools=None):
        """内部聚合并记录usage/终止原因。参数：messages/model输入、ledger/stage账本、source原问来源、tools绑定schema；返回：完整分片；异常：模型/取消/超长错误。"""
        gathered, size = None, 0
        log_model_context(source, messages, tools or [], ledger, stage)
        stream = model.astream(messages)
        try:
            async for chunk in stream:
                gathered = chunk if gathered is None else gathered + chunk
                size += output_chunk_size(chunk)
                if size > 20000:
                    raise WorkflowOutputError("内部模型输出超过长度上限")
        finally:
            # 取消和协议失败也保留已收到的计量；缺少尾块只能标未知。
            ledger.record_usage(stage, gathered.usage_metadata if gathered is not None else None,
                                gathered.response_metadata.get("finish_reason") if gathered is not None else None)
            with anyio.CancelScope(shield=True):
                await stream.aclose()
        if gathered is None:
            raise WorkflowOutputError("模型未返回有效决策")
        return gathered

    @staticmethod
    def _failure(gathered, category: str, error: Exception | None = None) -> dict:
        """构造有界内部诊断，避免把网络异常原文带到日志或用户。

        参数：gathered可为空的原始分片、category校验类别、error原异常。
        返回：独立审计字典，协议缺字段额外列missing_fields。
        异常：无主动业务异常。
        """
        invalid = InvalidToolDecision(
            tool_call_chunks=getattr(gathered, "tool_call_chunks", None),
            raw_output=_text(gathered) if gathered is not None else "",
            validation_category=category)
        diagnostic = invalid.audit_record
        if error is not None:
            diagnostic["error_type"] = type(error).__name__
            cause = error
            while cause is not None:
                if isinstance(cause, ValidationError):
                    missing = [str(row["loc"][0]) for row in cause.errors()
                               if row["type"] == "missing" and row["loc"]]
                    if missing:
                        diagnostic["missing_fields"] = sorted(set(missing))
                    break
                cause = cause.__cause__
        return diagnostic

    async def understand(self, messages: list, ledger: BudgetLedger) -> UnderstandingResult:
        """一次工具调用完成指代与标准问法；优化故障整体透传，控制信号保持原义。

        参数：messages带显式原问身份的ModelContext（兼容单条原问）；ledger本轮独占账本。
        返回：严格理解结果；任何调用/解析/来源失败返回完整原问且记录已消耗预算。
        异常：ValueError，无合法当前用户原问；WorkflowBudgetExceeded或取消原样传播。
        """
        converted = convert_to_messages(messages)
        if isinstance(messages, ModelContext):
            raw_query = messages.raw_query
            history_turns, summary_sources = messages.source_turns, messages.summary_sources
        else:
            # 旧单问调用没有资料消息；多条human必须由统一组装器显式携带原问身份。
            users = [message for message in converted if message.type == "human"]
            if len(users) != 1:
                raise ValueError("多轮理解必须显式标识当前用户原问")
            raw_query, history_turns, summary_sources = _text(users[0]), [], []
        fallback = original_understanding(raw_query)
        prepared = [SystemMessage(content=UNDERSTANDING_PROMPT), *converted]
        tool, output_limit = understanding_tool_schema(), ledger.output_limit("understand")
        ledger.reserve("understand", ledger.estimator.message_tokens(prepared) + ledger.estimator.value_tokens([tool]), output_limit)
        gathered = None
        try:
            gathered = await self._gather(prepared, bind_output_tool(self._model(output_limit), tool), ledger, "understand", source=messages, tools=[tool])
            result = parse_tool_output(gathered, UnderstandingResult, "submit_understanding")
            validated, source_failed = _understanding_outcome(raw_query, result, history_turns, summary_sources=summary_sources)
            if source_failed:
                diagnostic = self._failure(gathered, "understanding_reference")
                diagnostic["query_fallback"] = "original_query"
                ledger.record_failure("understand", diagnostic)
            return validated
        except WorkflowBudgetExceeded:
            # 预算是控制状态，不能被理解优化回退吞掉；取消继承BaseException也自然传播。
            raise
        except Exception as error:
            diagnostic = self._failure(gathered, "understanding_structure", error)
            diagnostic["query_fallback"] = "original_query"
            ledger.record_failure("understand", diagnostic)
            return fallback

    async def classify(self, messages: list, ledger: BudgetLedger) -> IntentResult:
        """用独立单模型分类一次，协议失败归其他并保留失败账本，不发起补救。

        参数：messages完整分类输入与历史；ledger独占账本。
        返回：严格intent/confidence；解析、枚举、数值或输出协议失败为其他/0。
        异常：WorkflowBudgetExceeded、网络/配置错误及取消原样传播，不能伪装为模型分类。
        """
        prepared = [SystemMessage(content=INTENT_PROMPT), *convert_to_messages(messages)]
        tool, output_limit = intent_tool_schema(), ledger.output_limit("classify")
        ledger.reserve("classify", ledger.estimator.message_tokens(prepared) + ledger.estimator.value_tokens([tool]), output_limit)
        gathered = None
        try:
            gathered = await self._gather(prepared,
                bind_output_tool(self._model(output_limit, model_name=self._intent_model), tool), ledger, "classify", source=messages, tools=[tool])
            return parse_tool_output(gathered, IntentResult, "submit_intent")
        except (StructuredOutputError, WorkflowOutputError) as error:
            ledger.record_failure("classify", self._failure(gathered, "intent_structure", error))
            return IntentResult(intent="其他", confidence=0.0)

    async def decide(self, messages: list, tools: list[ToolDefinition], ledger: BudgetLedger) -> AgentDecision:
        """仅接受业务工具或唯一终止控制工具，缺控制调用或其参数非法时有限纠正。

        参数：messages完整事实、tools当前授权快照、ledger独占账本。
        返回：AgentDecision，纠正轮只能回答或追问，不能再申请工具。
        异常：预算/协议/网络错误及取消；坏业务批次、非法ID和异常结束不触发纠正。
        """
        schemas = decision_tool_schemas(tools)
        available = {tool.name: tool for tool in tools}
        converted = convert_to_messages(messages)
        prepared = [SystemMessage(content=DECISION_PROMPT), *converted]
        answer_messages = [SystemMessage(content=ANSWER_PROMPT), *converted]
        retry_messages = [*answer_messages, HumanMessage(content=ANSWER_NUDGE)]
        for attempt in range(2):
            # 原失败文本只留诊断；固定system提示不会取代最近一条用户诉求或裁掉已知事实。
            current = prepared if attempt == 0 else [*prepared, HumanMessage(content=DECISION_NUDGE)]
            bound_schemas = schemas
            input_units = ledger.estimator.message_tokens(current) + ledger.estimator.value_tokens(bound_schemas)
            # 每次真实请求都保留两次正文余量；纠正提示也计入完整输入，不能预算外重试。
            ledger.ensure_capacity([("decide", input_units, ledger.output_limit("decide")),
                ("answer", ledger.estimator.message_tokens(answer_messages), ledger.output_limit("answer")),
                ("answer", ledger.estimator.message_tokens(retry_messages), ledger.output_limit("answer"))])
            ledger.reserve("decide", input_units, ledger.output_limit("decide"))
            model = self._model(ledger.output_limit("decide"))
            choice = "required" if attempt == 0 else {"type": "function", "function": {"name": _CONTROL_NAME}}
            bound = model.bind_tools(bound_schemas, tool_choice=choice, parallel_tool_calls=False)
            gathered = await self._gather(current, bound, ledger, "decide", source=messages, tools=bound_schemas)
            _check_finish(gathered.response_metadata.get("finish_reason"), tools=_has_calls(gathered))
            try:
                message = decision_message(gathered)
            except InvalidToolDecision:
                chunks = getattr(gathered, "tool_call_chunks", None) or []
                control = chunks[0] if len(chunks) == 1 else {}
                # 只重试身份明确的单个控制分片；不修补JSON，也不把业务或混合坏批次带入纠正。
                if (attempt == 0 and control.get("name") == _CONTROL_NAME
                        and isinstance(control.get("id"), str) and control["id"].strip()):
                    continue
                raise
            if message.invalid_tool_calls:
                raise InvalidToolDecision(invalid_tool_calls=message.invalid_tool_calls)
            if not message.tool_calls:
                # 自由文本即使恰好是旧版合法JSON也不能成为终止决定或最终正文。
                if attempt:
                    raise InvalidToolDecision(raw_output=_text(message), validation_category="missing_control_tool")
                continue
            calls, seen = message.tool_calls, set()
            try:
                for call in calls:
                    call_id, name = call.get("id"), call.get("name")
                    if (not isinstance(call_id, str) or not call_id.strip() or call_id in seen
                            or name not in {*available, _CONTROL_NAME}):
                        raise ValueError("工具名称或 ID 非法")
                    seen.add(call_id)
                terminal = any(call["name"] == _CONTROL_NAME for call in calls)
                if terminal and len(calls) != 1:
                    raise ValueError("终止控制调用不能混入业务批次或重复提交")
                if attempt and not terminal:
                    raise ValueError("纠正轮只能提交终止控制调用")
            except (ValueError, TypeError) as error:
                # 先核验整个批次，任何混合、非法ID或未知工具都不能部分执行或进入纠正。
                raise InvalidToolDecision(tool_calls=calls) from error
            if terminal:
                try:
                    return _parse_terminal_decision(calls[0]["args"])
                except InvalidToolDecision:
                    if attempt:
                        raise
                    continue
            break
        validated_calls = []
        try:
            for call in message.tool_calls:
                call_id, name, args = call.get("id"), call.get("name"), call.get("args")
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是JSON对象")
                # 这里只验证协议；缺字段或类型错误形成真实配对回执后，模型才有机会纠正。
                validated_calls.append({"id":call_id, "name":name, "args":deepcopy(args)})
            return AgentDecision(kind="tools", tool_calls=validated_calls)
        except (ValueError, TypeError, ValidationError) as error:
            raise InvalidToolDecision(tool_calls=message.tool_calls) from error

    async def stream_answer(self, messages: list, ledger: BudgetLedger) -> AsyncIterator[str]:
        """独立无工具正文流，最多一次有计费的空正文补救。参数：messages 完整事实、ledger 独占账本；返回：正文增量流；异常：预算/输出/网络错误及取消。"""
        prepared = [SystemMessage(content=ANSWER_PROMPT), *convert_to_messages(messages)]
        for attempt in range(2):
            current = prepared if attempt == 0 else [*prepared, HumanMessage(content=ANSWER_NUDGE)]
            ledger.reserve("answer", ledger.estimator.message_tokens(current), ledger.output_limit("answer"))
            log_model_context(messages, current, [], ledger, "answer")
            produced = False
            async with aclosing(self._answer_once(current, ledger)) as stream:
                async for delta in stream:
                    produced = True
                    yield delta
            if produced:
                return
        raise WorkflowOutputError("模型两次均未返回有效正文")

    async def _answer_once(self, messages: list, ledger: BudgetLedger) -> AsyncIterator[str]:
        """直接转发正文，过滤协议外工具/推理并及时关闭。参数：messages 完整输入、ledger 已占额账本；返回：正文增量流；异常：超长/截断/网络/取消。"""
        stream = self._model(ledger.output_limit("answer")).astream(messages)
        usage, reason, size = None, None, 0
        has_text, prefix = False, ""
        try:
            async for chunk in stream:
                if chunk.usage_metadata is not None:
                    usage = chunk.usage_metadata
                reason = chunk.response_metadata.get("finish_reason", reason)
                if _has_calls(chunk):
                    continue
                delta = _text(chunk)
                size += len(delta)
                if size > 12000:
                    raise WorkflowOutputError("正文超过12000字符上限")
                if delta:
                    if has_text:
                        # 正文开始后，单独空格/换行也是正文，不能逐块 strip 后丢弃。
                        yield delta
                    elif delta.strip():
                        has_text = True
                        yield prefix + delta
                        prefix = ""
                    else:
                        # 只暂存尚未出现正文的空白，使整流纯空白仍可触发一次补救。
                        prefix += delta
            # 仅工具输出可触发一次空正文补救；length/过滤等异常不能当空响应重试。
            if not has_text and reason == "tool_calls":
                return
            _check_finish(reason)
        finally:
            ledger.record_usage("answer", usage, reason)
            with anyio.CancelScope(shield=True):
                await stream.aclose()
