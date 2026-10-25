"""受预算限制的工作流模型边界：内部 JSON/完整工具申请与无工具最终正文真流式分离。"""

from contextlib import aclosing
from typing import AsyncIterator

import anyio
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from pydantic import ValidationError

from app.config import settings
from app.core.llm import decision_message, get_chat_model, strict_json_object
from app.core.memory import _estimate_message_tokens, _estimate_token_units
from app.core.model_budget import BudgetLedger
from app.core.prompts import ANSWER_NUDGE
from app.schemas.chat import InvalidToolDecision
from app.schemas.workflow import AgentDecision, IntentResult
from app.workflow.prompts import ANSWER_PROMPT, DECISION_PROMPT, INTENT_PROMPT

READ_ONLY_NAMES = frozenset({"query_order", "query_logistics", "query_product"})


class WorkflowOutputError(ValueError):
    """模型输出截断、拒绝、超长或无有效正文，不能作为正常完成发布。"""


def parse_intent(raw: str) -> IntentResult:
    """严格解析七意图 JSON。参数：raw 完整原文；返回：IntentResult；异常：InvalidToolDecision，不修复或猜测。"""
    try:
        return IntentResult.model_validate(strict_json_object(raw))
    except (ValueError, TypeError, ValidationError) as error:
        raise InvalidToolDecision() from error


def parse_decision(raw: str) -> AgentDecision:
    """严格解析无工具终止 JSON。参数：raw 完整原文；返回：AgentDecision；异常：InvalidToolDecision，结构不精确。"""
    try:
        payload = strict_json_object(raw)
        if set(payload) != {"kind", "suggested_actions"} or payload["kind"] not in ("answer", "clarify"):
            raise ValueError("终止决定必须精确包含 kind 和 suggested_actions")
        return AgentDecision.model_validate(payload)
    except (ValueError, TypeError, ValidationError) as error:
        raise InvalidToolDecision() from error


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

    def __init__(self, *, model_factory=None, stream_usage: bool | None = None) -> None:
        """参数：model_factory 外部模型边界，stream_usage 显式覆盖计量开关；返回：None；异常：无。"""
        self._factory = model_factory or get_chat_model
        self._stream_usage = settings.workflow_stream_usage if stream_usage is None else stream_usage

    def _model(self, output_limit: int):
        """创建有界请求模型。参数：output_limit 单次输出上限；返回：ChatOpenAI 兼容对象；异常：配置错误。"""
        return self._factory(streaming=True, temperature=0.0, max_tokens=output_limit,
                             max_retries=1, request_timeout=60.0, stream_usage=self._stream_usage)

    async def _gather(self, messages: list, model, ledger: BudgetLedger, stage: str):
        """内部聚合并记录全部 usage/终止原因，不向用户输出。参数：messages/model 输入依赖、ledger/stage 本次账本；返回：完整分片；异常：模型/取消/超长错误。"""
        gathered, size = None, 0
        stream = model.astream(messages)
        try:
            async for chunk in stream:
                gathered = chunk if gathered is None else gathered + chunk
                size += len(_text(chunk))
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

    async def classify(self, messages: list, ledger: BudgetLedger) -> IntentResult:
        """分类当前主诉求并保留历史上下文。参数：messages 完整分类输入、ledger 独占账本；返回：IntentResult；异常：预算/协议/网络错误。"""
        prepared = [SystemMessage(content=INTENT_PROMPT), *convert_to_messages(messages)]
        ledger.reserve("classify", _estimate_message_tokens(prepared), 128)
        gathered = await self._gather(prepared, self._model(128), ledger, "classify")
        if _has_calls(gathered):
            raise InvalidToolDecision(tool_call_chunks=gathered.tool_call_chunks)
        _check_finish(gathered.response_metadata.get("finish_reason"))
        return parse_intent(_text(gathered))

    async def decide(self, messages: list, tools: list, ledger: BudgetLedger) -> AgentDecision:
        """汇总并校验只读申请或终止 JSON。参数：messages 完整事实、tools 既有只读工具、ledger 独占账本；返回：AgentDecision；异常：预算/协议/网络错误。"""
        if any(tool.name not in READ_ONLY_NAMES or "write" in (tool.tags or []) for tool in tools):
            raise InvalidToolDecision()
        available = {tool.name: tool for tool in tools}
        if len(available) != len(tools):
            raise InvalidToolDecision()
        schemas = [{"type":"function", "function":{"name":tool.name, "description":tool.description,
                    "parameters":tool.tool_call_schema.model_json_schema()}} for tool in tools]
        converted = convert_to_messages(messages)
        prepared = [SystemMessage(content=DECISION_PROMPT), *converted]
        input_units = _estimate_message_tokens(prepared) + (_estimate_token_units(schemas) if schemas else 0)
        answer_messages = [SystemMessage(content=ANSWER_PROMPT), *converted]
        retry_messages = [*answer_messages, HumanMessage(content=ANSWER_NUDGE)]
        # 未来正文仅检查剩余额度；工具结果增加后下一步再次核对，不重复扣账。
        ledger.ensure_capacity([("decide", input_units, 2048),
            ("answer", _estimate_message_tokens(answer_messages), 4096),
            ("answer", _estimate_message_tokens(retry_messages), 4096)])
        ledger.reserve("decide", input_units, 2048)
        model = self._model(2048)
        gathered = await self._gather(prepared, model.bind_tools(schemas, tool_choice="auto") if schemas else model,
                                      ledger, "decide")
        _check_finish(gathered.response_metadata.get("finish_reason"), tools=_has_calls(gathered))
        message = decision_message(gathered)
        if message.invalid_tool_calls:
            raise InvalidToolDecision(invalid_tool_calls=message.invalid_tool_calls)
        if not message.tool_calls:
            return parse_decision(_text(message))
        calls, seen = [], set()
        try:
            for call in message.tool_calls:
                call_id, name, args = call.get("id"), call.get("name"), call.get("args")
                if not isinstance(call_id, str) or not call_id.strip() or call_id in seen or name not in available:
                    raise ValueError("工具名称或 ID 非法")
                schema = available[name].tool_call_schema
                if not isinstance(args, dict) or set(args) - set(schema.model_fields):
                    raise ValueError("工具参数包含未公开字段")
                validated = schema.model_validate(args, strict=True).model_dump()
                calls.append({"id":call_id, "name":name, "args":validated})
                seen.add(call_id)
            return AgentDecision(kind="tools", tool_calls=calls)
        except (ValueError, TypeError, ValidationError) as error:
            raise InvalidToolDecision(tool_calls=message.tool_calls) from error

    async def stream_answer(self, messages: list, ledger: BudgetLedger) -> AsyncIterator[str]:
        """独立无工具正文流，最多一次有计费的空正文补救。参数：messages 完整事实、ledger 独占账本；返回：正文增量流；异常：预算/输出/网络错误及取消。"""
        prepared = [SystemMessage(content=ANSWER_PROMPT), *convert_to_messages(messages)]
        for attempt in range(2):
            current = prepared if attempt == 0 else [*prepared, HumanMessage(content=ANSWER_NUDGE)]
            ledger.reserve("answer", _estimate_message_tokens(current), 4096)
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
        stream = self._model(4096).astream(messages)
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
