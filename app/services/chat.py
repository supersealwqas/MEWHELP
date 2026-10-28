"""在线持久化工作流服务及显式旧章服务，负责会话互斥、增量历史对账和流资源清理。"""

import asyncio
import json
import logging
from time import perf_counter
from typing import AsyncIterator
from contextlib import aclosing, asynccontextmanager, suppress
from dataclasses import dataclass
from uuid import uuid4
import anyio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from app.core.memory import ContextBudgetExceeded, assemble_turns, prepare_messages, valid_tool_calls
from app.core.model_budget import WorkflowBudgetExceeded, budget_summary
from app.core.prompts import ANSWER_NUDGE, ANSWER_SYSTEM, CUSTOMER_SERVICE_SYSTEM, TOOL_DECISION_SYSTEM
from app.knowledge.answering import (
    KNOWLEDGE_ANSWER_RULES, business_facts_summary, collect_citations, decide_answer, has_knowledge_call,
    has_failed_knowledge_result, KnowledgeAnswerError, knowledge_answer_handoff, number_knowledge_results,
)
from app.repositories.chat import PersistenceUncertain
from app.schemas.chat import ChatEvent, ChatRequest, InvalidToolDecision
from app.schemas.tools import ToolContext, ToolResult
from app.tools.executor import ToolExecutor
from app.workflow.state import checkpoint_thread_id
from app.schemas.workflow import CurrentActions, OrderSelection, OrderSelectionAction, OrderSelectionConflict

logger = logging.getLogger(__name__)

# 工具执行流同时承载两类项：ChatEvent 直接转发给客户端，ToolResult 用于落库与拒绝判定。
# 用别名声明这个双协议，避免调用方误以为流里只有事件。
ToolStreamItem = ChatEvent | ToolResult

# 限制整个可见回答的字符数；即使全部是四字节 Unicode，也低于 MySQL TEXT 字节上限。
MAX_ANSWER_CHARACTERS = 12000


class AnswerTooLong(ValueError):
    """回答超过可展示和持久化的长度，必须关闭上游流并明确标记未完成。"""


@dataclass
class _ToolChainState:
    """跨步骤保留已发生写入和固定回复，异常或取消时也不能丢失已提交事实。"""

    write_started: bool = False  # 一旦尝试建单即保持为真，后续只读步骤不能重置。
    ticket_receipt: str = ""  # 已发送的真实工单编号提示，同时作为最终落库正文前缀。
    rejection: str | None = None  # 归属拒绝的固定答复，不交给模型改写。

# 归属拒绝的固定回复：未知订单与他人订单必须逐字一致，交给模型改写会泄露订单是否存在。
FIXED_REJECTION_REPLIES: dict[str, str] = {
    "order_not_owned": "没有找到您的这笔订单，请确认订单号或物流单号后重试。",
}


def fixed_reply(result: ToolResult) -> str | None:
    """取出不交给模型改写的固定回复。

    参数:
        result: 已落库的工具结果，成功结果或非拒绝错误返回 None。
    返回值:
        str | None: 命中固定回复表时返回固定文本，否则返回 None 表示仍走模型回答阶段。
    异常:
        无主动抛出的业务异常。
    """
    if result.ok or result.error is None:
        return None
    return FIXED_REJECTION_REPLIES.get(result.error.code)


class LegacyChatService:
    """保留CH02–CH04独立验收的旧编排；生产默认入口使用显式注图的ChatService。"""

    def __init__(self, chat_repository, support_repository, model_adapter, registry, settings):
        """注入业务依赖。参数：存储、模型、注册表和配置；返回：None；异常：执行器配置错误。"""
        self.chat, self.model, self.registry, self.settings = chat_repository, model_adapter, registry, settings
        self.executor = ToolExecutor(registry, chat_repository, support_repository, settings.tool_timeout_seconds,
                                     settings.tool_max_retries, settings.tool_retry_delay_seconds,
                                     knowledge_timeout_seconds=getattr(settings, "knowledge_timeout_seconds", None))
        self._active: set[str] = set()

    async def _execute(self, call, context):
        """将执行状态转换为流。

        参数:
            call: 单个工具申请。
            context: 已绑定 tool_call_id 的可信上下文。
        返回值:
            异步生成器：先产出 ChatEvent 状态帧，最后产出 ToolResult。
        异常:
            执行或持久化异常向外传播；取消时先清理工具任务。
        """
        queue = asyncio.Queue()

        async def status(frame):
            """将安全状态送入单轮队列。参数：状态字典；返回：None；异常：无。"""
            queue.put_nowait(ChatEvent(event="tool_status", data=frame))

        async def run():
            """执行并唤醒消费者。参数：无；返回：工具结果；异常：原执行异常。"""
            try:
                return await self.executor.execute(call, context, status)
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(run())
        try:
            while (event := await queue.get()) is not None:
                yield event
            yield await task
        finally:
            if not task.done():
                task.cancel()
            # Starlette 使用取消域；屏蔽重复取消直至工具事务清理完毕，禁止遗留业务后台任务。
            with anyio.CancelScope(shield=True):
                with suppress(asyncio.CancelledError, Exception):
                    await task

    async def _decide(self, messages) -> list[dict]:
        """请求模型做本轮工具决策，并校验申请协议。

        参数:
            messages: 决策阶段的完整消息列表。
        返回值:
            list[dict]: 已校验的工具申请，无工具时为空列表。
        异常:
            InvalidToolDecision: 模型返回残缺或损坏申请，携带仅供审计的原始字段。
        """
        decision = await self.model.decide(messages, self.registry.all())
        if decision.invalid_tool_calls or not valid_tool_calls(decision.tool_calls):
            raise InvalidToolDecision(tool_calls=decision.tool_calls, invalid_tool_calls=decision.invalid_tool_calls)
        return decision.tool_calls

    async def _skip_calls(
        self, calls: list[dict], context: ToolContext, current: list, code: str, message: str
    ):
        """为未执行的合法申请保存配对结果，并明确显示跳过状态。

        参数:
            calls: 已保存申请中本次不再执行的部分，调用ID必须唯一。
            context: 本轮可信上下文。
            current: 本轮消息列表，将被追加对应的 ToolMessage。
            code: 停止或限制的稳定错误码。
            message: 可回灌给模型的中文停止原因。
        返回值:
            异步 ChatEvent 流，每个申请产生一个未执行状态。
        异常:
            结果保存失败由仓储层抛出。
        """
        for call in calls:
            call_context = context.model_copy(update={"tool_call_id": call["id"]})
            result = await self.chat.save_tool_result(call_context, ToolResult.failed(code, message))
            current.append(ToolMessage(content=result.model_dump_json(), tool_call_id=call["id"]))
            yield ChatEvent(event="tool_status", data={
                "tool_call_id": call["id"], "name": call["name"][:100], "status": "failed",
                "label": "未执行", "is_mock": False,
            })

    @staticmethod
    def _request_key(call: dict) -> str:
        """计算同轮请求去重标识，匹配现有业务参数去首尾空白的规则。

        参数：call为经过协议校验的申请，当前五工具均使用扁平参数。
        返回：包含工具名与规范化参数的稳定JSON字符串，不包含调用ID。
        异常：参数包含非法JSON数值时抛出ValueError。
        """
        args = {key: value.strip() if isinstance(value, str) else value
                for key, value in call["args"].items()}
        return json.dumps([call["name"], args], sort_keys=True, ensure_ascii=False, allow_nan=False)

    async def _run_tool_chain(self, history, current, context: ToolContext, state: _ToolChainState):
        """按当前轮结果继续决策，在次数、失败和重复边界内依次执行工具。

        参数：history为完整历史轮；current为不可裁剪的当前轮消息；context为可信身份；
            state保存写入事实和固定答复，异常时由外层继续使用。
        返回：异步ChatEvent流；结束后current包含所有已保存且配对的工具事实。
        异常：模型、消息预算或持久化异常向外透传，取消会关闭正在执行的工具流。
        """
        limit = self.settings.tool_max_calls_per_turn
        used = 0
        seen_ids: set[str] = set()
        seen_requests: set[str] = set()
        tools = self.registry.all()
        while used < limit:
            # 每一步均重新核算完整请求，前一步结果始终保留在当前轮，不能被历史裁剪删除。
            messages = prepare_messages(
                history, current, self.settings.token_budget,
                system_text=CUSTOMER_SERVICE_SYSTEM + TOOL_DECISION_SYSTEM
                + f"\n本轮还可执行{limit - used}个工具；已完成的相同查询不要重复申请。",
                tools=tools, request_max_tokens=self.settings.request_token_budget,
            )
            try:
                calls = await self._decide(messages)
                if any(call["id"] in seen_ids for call in calls):
                    raise InvalidToolDecision(tool_calls=calls)
            except InvalidToolDecision as error:
                # 坏协议仅用于审计，不回灌原始申请；此前已经完成的事实仍然有效。
                await self.chat.save_tool_calls(context, [error.audit_record])
                current.append(SystemMessage(content=(
                    "本次工具申请格式无效或调用ID重复，未执行任何工具；"
                    "这仅指本次申请，此前已保存的工具结果仍有效。请据已有事实说明未完成部分。"
                )))
                return
            if not calls:
                return

            seen_ids.update(call["id"] for call in calls)
            await self.chat.save_tool_calls(context, calls)
            current.append(AIMessage(content="", tool_calls=calls))
            if len(calls) > limit - used:
                async with aclosing(self._skip_calls(
                    calls, context, current, "TOOL_CALL_LIMIT", "本批申请超过本轮剩余调用次数，未执行；请依据已有结果回答"
                )) as skipped:
                    async for event in skipped:
                        yield event
                return

            for index, call in enumerate(calls):
                request_key = self._request_key(call)
                if call["name"] == "create_ticket" and state.write_started:
                    code, reason = "WRITE_TOOL_LIMIT", "本轮已经尝试创建工单，禁止重复提交，请核实已有工单结果"
                elif request_key in seen_requests:
                    code, reason = "DUPLICATE_TOOL_CALL", "相同工具和参数本轮已经执行，未重复调用；请使用已有结果"
                else:
                    code = None
                if code is not None:
                    async with aclosing(self._skip_calls(calls[index:], context, current, code, reason)) as skipped:
                        async for event in skipped:
                            yield event
                    return

                seen_requests.add(request_key)
                used += 1
                state.write_started = state.write_started or call["name"] == "create_ticket"
                call_context = context.model_copy(update={"tool_call_id": call["id"]})
                result = None
                async with aclosing(self._run_single(call, call_context, current)) as runner:
                    async for item in runner:
                        if isinstance(item, ChatEvent):
                            yield item
                        else:
                            result = item
                            state.rejection = fixed_reply(item)
                            if call["name"] == "query_refund_policy" and item.ok:
                                # 一次三查询已覆盖完整退款问题，不让模型再申请同问题 FAQ 重复重排。
                                current.append(SystemMessage(content="本轮已完成退款政策三查询并保留全部证据，同一退款问题不要再查询 FAQ。"))
                            if call["name"] == "create_ticket" and item.ok and item.data:
                                number = item.data.get("ticket_no")
                                if isinstance(number, str) and number:
                                    state.ticket_receipt = f"工单已创建，工单号：{number}，状态：待处理。\n\n"
                                    # 先发送已提交编号；后续决策失败或用户停止也不能隐藏成功事实。
                                    yield ChatEvent(event="delta", data={"delta": state.ticket_receipt})
                if result is None:
                    raise RuntimeError("工具流缺少终态结果")
                if not result.ok:
                    async with aclosing(self._skip_calls(
                        calls[index + 1:], context, current, "TOOL_CHAIN_STOPPED", "前一步未完成，后续工具未执行"
                    )) as skipped:
                        async for event in skipped:
                            yield event
                    return

        current.append(SystemMessage(content="本轮已达到工具调用次数上限。依据已保存结果回答，剩余未完成的部分如实说明。"))

    async def _run_single(self, call: dict, context: ToolContext, current: list):
        """执行单个工具申请，转发状态帧并把结果写入本轮消息。

        参数:
            call: 单个工具申请。
            context: 已绑定 tool_call_id 的上下文。
            current: 本轮消息列表，工具结果会以 ToolMessage 追加。
        返回值:
            异步生成器：ChatEvent 状态帧与末尾的 ToolResult（见 ToolStreamItem）。
        异常:
            执行或持久化异常向外传播。
        """
        execution = self._execute(call, context)
        try:
            async for item in execution:
                if isinstance(item, ChatEvent):
                    yield item
                else:
                    current.append(ToolMessage(content=item.model_dump_json(), tool_call_id=context.tool_call_id))
                    yield item
        finally:
            await execution.aclose()

    @staticmethod
    def _error_message(error: Exception) -> str:
        """把内部异常映射为可安全展示给用户的中文说明。

        参数:
            error: 本轮捕获的异常。
        返回值:
            str: 面向用户的错误说明，不含异常原文。
        异常:
            无主动抛出的业务异常。
        """
        if isinstance(error, ContextBudgetExceeded):
            return str(error)
        if isinstance(error, PersistenceUncertain):
            return "暂时无法核实处理状态，请稍后查看，勿重复提交工单"
        if isinstance(error, AnswerTooLong):
            return "回答过长，已停止生成；请缩小问题范围后再询问"
        return "本轮暂时未能完成，请稍后重试"

    async def _stream_answer(self, messages, prefix: str = ""):
        """流式生成最终回答，返回已拼接的完整文本。

        参数:
            messages: 回答阶段的完整消息列表。
            prefix: 已发送的服务端工单提示，计入正文长度并随完成回答一起保存。
        返回值:
            异步生成器：先产出 ChatEvent 增量帧，最后产出完整回答字符串。
        异常:
            ValueError: 模型未返回有效文本。
            AnswerTooLong: 回答超过字符上限，流会立即关闭。
        """
        answer_stream = self.model.stream_answer(messages)
        text, length, has_answer = [prefix], len(prefix), False
        try:
            async for delta in answer_stream:
                if delta:
                    if length + len(delta) > MAX_ANSWER_CHARACTERS:
                        raise AnswerTooLong("回答超过长度限制")
                    text.append(delta)
                    length += len(delta)
                    has_answer = has_answer or bool(delta.strip())
                    yield ChatEvent(event="delta", data={"delta": delta})
        finally:
            await answer_stream.aclose()
        answer = "".join(text)
        if not has_answer:
            raise ValueError("模型未返回有效文本")
        yield answer

    async def stream_turn(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        """处理一次提问。参数：已校验请求；返回：ChatEvent 异步流；异常：客户端取消向外传播。"""
        key = request.session_id
        if key in self._active:
            yield ChatEvent(event="error", data={"error": "当前会话正在回复，请稍后再发", "retry_safe": True})
            return
        # 检查与占用之间没有 await，单 worker 中不会出现竞争窗口。
        self._active.add(key)
        context = ToolContext(session_id=key, user_id=request.user_id, turn_id=str(uuid4()), user_message=request.message)
        completed, started, terminal = False, False, "cancelled"
        chain = _ToolChainState()
        stage = "input"
        try:
            current = [HumanMessage(content=request.message)]
            decision_system = CUSTOMER_SERVICE_SYSTEM + TOOL_DECISION_SYSTEM
            tools = self.registry.all()
            # 先核算固定提示、工具定义和当前问题，超限时不创建无效轮次。
            prepare_messages([], current, self.settings.token_budget, system_text=decision_system,
                             tools=tools, request_max_tokens=self.settings.request_token_budget)
            context = await self.chat.start_turn(context, request.message)
            started = True
            stage = "history"
            history = assemble_turns(await self.chat.load_turns(context.conversation_id, exclude_turn_id=context.turn_id))
            stage = "tool_chain"
            async with aclosing(self._run_tool_chain(history, current, context, chain)) as runner:
                async for event in runner:
                    yield event
            if chain.rejection is not None:
                # 归属拒绝不进入模型阶段：固定回复保证未知订单与他人订单的措辞完全一致。
                yield ChatEvent(event="delta", data={"delta": chain.rejection})
                stage = "finish"
                metadata = await self.chat.finish_turn(context, chain.ticket_receipt + chain.rejection)
                completed = True
                yield ChatEvent(event="answer_metadata", data=metadata)
                yield ChatEvent(event="done")
                return
            stage = "answer"
            if (has_knowledge_call(current)
                    and getattr(self.executor.support, "uses_full_knowledge_query", False)):
                if has_failed_knowledge_result(current):
                    # 故障不得绕过严格知识生成，也不能制造useful=false污染问题池。
                    raise KnowledgeAnswerError("本轮知识查询未成功完成")
                citations = collect_citations(current)
                numbered = number_knowledge_results(current, citations)
                # 最后显式交接为生成任务，阻止网关沿工具尾帧继续申请不存在的工具；原问不改写。
                handoff = HumanMessage(content=json.dumps(knowledge_answer_handoff(context.user_message), ensure_ascii=False))
                messages = prepare_messages(history, [*numbered, handoff], self.settings.token_budget,
                    system_text=ANSWER_SYSTEM + KNOWLEDGE_ANSWER_RULES,
                    request_max_tokens=self.settings.request_token_budget)
                # 当前全部编号证据实际进入该请求；先严格自评，再原子提交，最后发布正文。
                decision = await self.model.answer_knowledge(messages)
                answer, low_confidence = decide_answer(decision, citations)
                if low_confidence is not None:
                    business = business_facts_summary(current)
                    if business:
                        answer = business + "\n\n" + answer
                if len(chain.ticket_receipt + answer) > MAX_ANSWER_CHARACTERS:
                    raise AnswerTooLong("回答超过长度限制")
                stage = "finish"
                metadata = await self.chat.finish_turn(context, chain.ticket_receipt + answer,
                                                       low_confidence=low_confidence, citations=citations)
                completed = True
                yield ChatEvent(event="delta", data={"delta": answer})
                yield ChatEvent(event="answer_metadata", data=metadata)
                yield ChatEvent(event="done")
                return
            # 回答阶段没有工具定义，必须改用不含决策指令的 ANSWER_SYSTEM；
            # 继续使用 CUSTOMER_SERVICE_SYSTEM 会命令模型再调工具，导致本轮整轮失败。
            # 同时为可能的空正文纠正预留两层预算，仍按整轮裁历史，不修改当前事实。
            messages = prepare_messages(history, [*current, HumanMessage(content=ANSWER_NUDGE)],
                self.settings.token_budget,
                system_text=ANSWER_SYSTEM,
                request_max_tokens=self.settings.request_token_budget)[:-1]
            answer = None
            async with aclosing(self._stream_answer(messages, chain.ticket_receipt)) as stream:
                async for item in stream:
                    if isinstance(item, ChatEvent):
                        yield item
                    else:
                        answer = item
            stage = "finish"
            metadata = await self.chat.finish_turn(context, answer)
            completed = True
            yield ChatEvent(event="answer_metadata", data=metadata)
            yield ChatEvent(event="done")
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as error:
            terminal = "failed"
            logger.warning("客服轮次失败 turn_id=%s stage=%s type=%s", context.turn_id, stage, type(error).__name__)
            retry_safe = not chain.write_started and not isinstance(error, PersistenceUncertain)
            if chain.ticket_receipt:
                message = chain.ticket_receipt.strip() + "补充回复未完成，勿重复提交工单。"
            elif not retry_safe:
                message = "本轮回复未完成，请先核实工单处理状态，勿重复提交工单"
            else:
                message = self._error_message(error)
            yield ChatEvent(event="error", data={"error": message, "retry_safe": retry_safe})
        finally:
            try:
                if started and not completed:
                    with anyio.CancelScope(shield=True):
                        await self.chat.mark_turn(context, terminal)
            except Exception as error:
                logger.warning("轮次终态保存失败 turn_id=%s type=%s", context.turn_id, type(error).__name__)
            finally:
                self._active.discard(key)


class ConversationBusy(RuntimeError):
    """聊天或确认动作已占用会话；调用者可以立即返回忙状态，不排队执行。"""


class ConversationGuard:
    """单进程共享的非阻塞会话互斥，聊天与后续确认服务必须注入同一实例。"""

    def __init__(self):
        """建立占用集合。参数：无；返回：None；异常：无。"""
        self._active: set[str] = set()

    @asynccontextmanager
    async def hold(self, session_id: str):
        """立即获取并保证释放会话。参数：session_id会话键；返回：异步上下文；异常：ConversationBusy，已有占用。"""
        # 检查和占用之间没有await；单worker事件循环内不存在抢占窗口。
        if session_id in self._active:
            raise ConversationBusy("当前会话正在处理")
        self._active.add(session_id)
        try:
            yield
        finally:
            self._active.discard(session_id)


class _WorkflowTrace(BaseCallbackHandler):
    """仅记录语义节点的安全摘要，排除用户原话、工具参数及模型内部推理。"""

    run_inline = True
    _names = frozenset({"resolve_reference", "classify_intent", "retrieve_knowledge", "confidence_check",
                       "main_agent", "complaint_reply", "chitchat_reply", "fallback_reply", "log_turn",
                       "decision", "tools", "answer"})

    def __init__(self):
        """建立本轮回调计时。参数：无；返回：None；异常：无。"""
        self.running = {}

    def _record(self, phase, name, state, elapsed=0, error=None):
        """输出允许字段。参数：阶段/节点/State/耗时/错误类型；返回：None；异常：日志错误由框架处理。"""
        context = state.get("context", {})
        gate = state.get("gate") or {}
        logger.info("workflow node=%s phase=%s turn=%s thread=%s intent=%s gate=%s tools=%s step=%s "
                    "budget=%s elapsed_ms=%.1f stop=%s error=%s", name, phase, context.get("turn_id"),
                    checkpoint_thread_id(context.get("conversation_id")), state.get("intent"),
                    {key: gate.get(key) for key in ("passed", "max_score", "threshold")},
                    [call.get("name") for call in state.get("pending_calls", [])], state.get("decision_count"),
                    budget_summary(state.get("budget")), elapsed * 1000, state.get("stop_reason"), error)

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """记录语义节点进入。参数：框架回调和运行ID；返回：None；异常：无。"""
        name = kwargs.get("name")
        if name in self._names and isinstance(inputs, dict):
            self.running[run_id] = (name, inputs, perf_counter())
            self._record("enter", name, inputs)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """合并节点增量后记录退出。参数：输出/运行ID；返回：None；异常：无。"""
        item = self.running.pop(run_id, None)
        if item:
            name, state, started = item
            self._record("exit", name, {**state, **(outputs if isinstance(outputs, dict) else {})}, perf_counter() - started)

    def on_chain_error(self, error, *, run_id, **kwargs):
        """只记录异常类型并结束计时。参数：error框架错误、run_id运行ID；返回：None；异常：无。"""
        item = self.running.pop(run_id, None)
        if item:
            name, state, started = item
            self._record("exit", name, state, perf_counter() - started, type(error).__name__)


class ChatService:
    """显式消费真实编译图的在线入口；完成事务和最终checkpoint均成功后才发布完成帧。"""

    def __init__(self, chat_repository, graph, executor, settings, *, guard: ConversationGuard | None = None, summaries=None):
        """注入共享资源。参数：仓储/编译图/执行器/配置、可选互斥和应用摘要管理器；返回：None；异常：无。"""
        self.chat, self.graph, self.executor, self.settings = chat_repository, graph, executor, settings
        self.guard = guard if guard is not None else ConversationGuard()
        self.summaries = summaries

    async def _schedule_summary(self, values: dict, conversation_id: int) -> None:
        """登记已有积压，不等待后台模型。参数：最终State/可信会话；返回：None；异常：调度错误隔离并留痕。"""
        try:
            # 旧等待续跑没有冻结快照时跳过；下一正常轮会从数据库重新冻结，不影响原轮完成。
            if self.summaries is not None and (values.get("history_context") or {}).get("summary_required"):
                await self.summaries.maybe_schedule(conversation_id)
        except Exception as error:
            logger.warning("summary fail conversation=%s reason=schedule type=%s", conversation_id, type(error).__name__)

    async def _stream_workflow(self, state: dict, config: dict):
        """由独占asyncio任务拥有整个图，隔离ASGI取消域的重复取消。

        参数：state本轮初态、config图配置；返回：原namespace/custom事件流。
        异常：图错误原样传播；消费者取消时只取消生产任务一次，等待节点和模型关闭后才退出。
        """
        queue = asyncio.Queue()
        finished = object()

        async def produce():
            """完整消费原图并在所有出口唤醒消费者。参数：无；返回：None；异常：原图错误/取消。"""
            try:
                async with aclosing(self.graph.astream(state, config, stream_mode="custom", subgraphs=True,
                                                       durability="sync")) as upstream:
                    async for item in upstream:
                        queue.put_nowait(item)
            finally:
                queue.put_nowait(finished)

        # 不放进AnyIO task group，避免其level cancellation打断LangGraph自己的退出任务。
        task = asyncio.create_task(produce())
        try:
            while (item := await queue.get()) is not finished:
                yield item
            await task
        finally:
            if not task.done():
                task.cancel()
            with anyio.CancelScope(shield=True):
                # 正常消费路径上方已传播原异常；关闭时只负责收回任务，不覆盖外层取消/发送错误。
                with suppress(asyncio.CancelledError, Exception):
                    await task

    async def _initial_state(self, context: ToolContext, config: dict):
        """先核验checkpoint身份，再增量对账并创建完整新input，禁止恢复旧申请。

        参数：context新轮可信身份；config只包含thread_id的图配置。
        返回：ConversationState，合并后的历史及全部重置的本轮字段。
        异常：数据库/SQLite读取失败；损坏历史校验错误。
        """
        from app.workflow.state import new_turn_state, full_history_messages
        from langgraph.types import Overwrite

        previous = (await self.graph.aget_state(config)).values
        old_context = previous.get("context")
        if not isinstance(old_context, dict) or any(
            old_context.get(key) != getattr(context, key) for key in ("conversation_id", "session_id", "user_id")
        ):
            # SQLite可独立于MySQL保留；身份缺失或变化时，不导入旧内容、游标或未完成轮。
            if previous:
                logger.warning("工作流checkpoint身份不符，重新读取可信历史 thread=%s",
                               config["configurable"]["thread_id"])
            previous, old_context = {}, {}
        refresh = old_context.get("turn_id") if not previous.get("completion_metadata") else None
        migrating = previous.get("context_version") != 7
        delta = await self.chat.load_history_delta(context.conversation_id, 0 if migrating else previous.get("audit_cursor", 0),
            refresh_turn_id=refresh)
        merged = {turn["turn_id"]: turn for turn in previous.get("history_turns", [])}
        if migrating:
            merged = {}
        for turn in delta.turns:
            if turn.turn_id != context.turn_id:
                merged[turn.turn_id] = turn.model_dump(mode="json")
        state = new_turn_state(context, list(merged.values()), delta.last_message_id)
        current = [turn.model_dump(mode="json") for turn in delta.turns if turn.turn_id == context.turn_id]
        if current:
            state["messages"] = [*full_history_messages(list(merged.values())), *full_history_messages(current)]
        if migrating:
            # 只在旧版或不可信checkpoint上完整重建；后续轮次由add_messages按ID合并。
            state["messages"] = Overwrite(state["messages"])
        return state

    def _waiting_action(self, snapshot, context: ToolContext) -> OrderSelectionAction | None:
        """核对官方待执行任务的interrupt与可信原轮，普通next不算等待。

        参数：snapshot官方StateSnapshot、context数据库原轮。
        返回：唯一匹配的选单或None；异常：ValueError，损坏动作。
        """
        from uuid import NAMESPACE_URL, uuid5
        saved = snapshot.values.get("context", {})
        if any(saved.get(key) != getattr(context, key) for key in ("conversation_id", "session_id", "user_id", "turn_id", "user_message")):
            return None
        interrupts = [item for task in getattr(snapshot, "tasks", ()) for item in task.interrupts]
        if len(interrupts) != 1:
            return None
        action = OrderSelectionAction.model_validate(interrupts[0].value)
        expected = str(uuid5(NAMESPACE_URL, f"mewhelp:ch06:{context.conversation_id}:{context.turn_id}:select_order"))
        return action if action.source_turn_id == context.turn_id and action.action_id == expected else None

    async def reconcile_waiting_turns(self) -> None:
        """启动后先对账官方保存器，再处理真正遗留的运行轮。

        参数：无；返回：None；匹配暂停修复等待，已接受暂停保留原选择。
        异常：任意数据库/检查点读取失败阻止启动；不猜测没有等待。
        """
        pending = await self.chat.list_pending_workflow_turns()
        resolved = []
        # 先完成所有检查点读取，任何读取失败都不能导致部分原轮被当作遗留取消。
        for row in pending:
            context = row["context"]
            config = {"configurable": {"thread_id": checkpoint_thread_id(context.conversation_id)}}
            snapshot = await self.graph.aget_state(config)
            resolved.append((row, self._waiting_action(snapshot, context)))
        for row, action in resolved:
            context, saved = row["context"], row["order_selection"]
            if action is not None:
                if isinstance(saved, dict) and saved.get("status") == "accepted":
                    expected = {key: saved.get(key) for key in ("type", "action_id", "source_turn_id", "orders")}
                    if (action.model_dump(mode="json") == expected and row["turn_status"] == "running"
                            and saved.get("accepted_order_id") in {order.order_id for order in action.orders}):
                        continue
                elif saved is None or saved.get("status") == "awaiting_input":
                    await self.chat.save_order_wait(context, action)
                    continue
            inconsistent = row["turn_status"] == "awaiting_input" or saved is not None
            if inconsistent:
                logger.warning("选单等待与检查点不一致 turn=%s", context.turn_id)
            await self.chat.mark_turn(context, "failed" if inconsistent else "cancelled", include_waiting=True)

    async def get_current_actions(self, session_id: str, user_id: str) -> CurrentActions:
        """读取当前保存动作，已接受未续跑时提供继续标记。参数：会话/用户；返回：CurrentActions；异常：身份或检查点读取错误。"""
        row = await self.chat.get_workflow_turn(session_id, user_id)
        if row is None:
            return CurrentActions()
        saved = row["order_selection"]
        action = None
        resume_required = False
        if isinstance(saved, dict):
            action = OrderSelectionAction.model_validate({key: saved[key] for key in ("type", "action_id", "source_turn_id", "orders")})
            if saved.get("status") == "accepted" and row["turn_status"] == "running":
                config = {"configurable": {"thread_id": checkpoint_thread_id(row["context"].conversation_id)}}
                pending = self._waiting_action(await self.graph.aget_state(config), row["context"])
                resume_required = pending == action
        metadata = row["completion_metadata"] or {}
        refund_receipt = None
        refund_action = metadata.get("refund_action")
        if refund_action:
            from app.repositories.actions import ActionRepository
            refund_receipt = await ActionRepository(self.chat.factory).get_refund_receipt(session_id, user_id, refund_action["action_id"])
        return CurrentActions(source_turn_id=row["context"].turn_id, turn_status=row["turn_status"],
            order_selection=action, accepted_order_id=saved.get("accepted_order_id") if saved else None,
            answer=row["answer"], completion_metadata=row["completion_metadata"], resume_required=resume_required,
            refund_action=refund_action, refund_receipt=refund_receipt)

    def _public_event(self, payload: dict, context: ToolContext) -> ChatEvent | None:
        """筛选并规范公开事件。参数：payload图custom帧、context原轮；返回：事件或None；异常：无。"""
        from app.workflow.stages import LABELS, FIELDS
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            return None
        data, event = payload["data"], payload.get("event")
        if event == "delta" and isinstance(data.get("delta"), str) and data["delta"]:
            return ChatEvent(event="delta", data={"delta": data["delta"]})
        if event == "tool_status":
            return ChatEvent(event="tool_status", data={key: data[key] for key in ("tool_call_id", "name", "status", "label", "is_mock") if key in data})
        stage, status = data.get("stage"), data.get("status")
        if event == "workflow_stage" and stage in LABELS and status in {"running", "completed"}:
            safe = {"source_turn_id": context.turn_id, "stage": stage, "status": status, "label": LABELS[stage]}
            if status == "completed":
                for key in FIELDS[stage]:
                    value = data.get(key)
                    if key == "confidence":
                        import math
                        if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
                            safe[key] = value
                    elif key == "queries":
                        if isinstance(value, list) and len(value) in {1, 3} and all(isinstance(item, str) and 0 < len(item) <= 2000 for item in value):
                            safe[key] = value
                    elif isinstance(value, str) and 0 < len(value) <= 4000:
                        safe[key] = value
            return ChatEvent(event="workflow_stage", data=safe)
        return None

    async def _run_graph(self, initial, context: ToolContext, outcome: dict):
        """消费图并确认完成或持久化等待，退出前收回图流。

        参数：initial新轮State或原轮Command；context可信原轮；outcome本请求清理状态。
        返回：公开增量和完成事件；选单只存outcome供锁外发送。
        异常：图/存储/取消错误；已确认暂停后不改原轮终态。
        """
        config = {"configurable": {"thread_id": checkpoint_thread_id(context.conversation_id)}}
        stream = self._stream_workflow(initial, {**config, "callbacks": [_WorkflowTrace()]})
        try:
            async for namespace, payload in stream:
                event = self._public_event(payload, context)
                if event is not None:
                    yield event
            final = await self.graph.aget_state(config)
            action = self._waiting_action(final, context)
            if action is not None:
                # SQLite已提交暂停，即使MySQL下一步失败也保留原轮，交启动对账修复。
                outcome["paused"] = True
                await self.chat.save_order_wait(context, action)
                outcome["waiting"] = action.model_dump(mode="json")
                await self._schedule_summary(final.values, context.conversation_id)
                return
            metadata = final.values.get("completion_metadata", {})
            if final.next or metadata.get("turn_id") != context.turn_id or metadata.get("status") != "completed":
                raise PersistenceUncertain("工作流最终状态尚未确认")
            outcome["completed"] = True
            await self._schedule_summary(final.values, context.conversation_id)
            yield ChatEvent(event="answer_metadata", data=metadata)
            yield ChatEvent(event="done")
        finally:
            with anyio.CancelScope(shield=True):
                await stream.aclose()

    async def _cleanup_turn(self, context: ToolContext, outcome: dict, terminal: str) -> None:
        """只标记本请求仍在运行的原轮，已等待或完成请求不再写状态。参数：context/outcome/terminal；返回：None；异常：记录后不抛。"""
        if outcome.get("started") and not outcome.get("completed") and not outcome.get("paused"):
            try:
                await self.chat.mark_turn(context, terminal)
            except Exception as error:
                logger.warning("轮次终态保存失败 turn=%s type=%s", context.turn_id, type(error).__name__)

    def _stream_error(self, error: Exception) -> ChatEvent:
        """生成安全错误，保留状态码与未知结果语义。参数：error异常；返回：error事件；异常：无。"""
        from app.repositories.actions import ActionNotFound
        status = 409 if isinstance(error, (ConversationBusy, OrderSelectionConflict)) else 404 if isinstance(error, ActionNotFound) or type(error) is ValueError else 503
        message = ("当前会话正在处理，请稍后继续原操作" if isinstance(error, ConversationBusy)
                   else str(error) if isinstance(error, OrderSelectionConflict)
                   else "该动作不存在或不属于当前会话" if status == 404 else LegacyChatService._error_message(error))
        return ChatEvent(event="error", data={"error": message, "status_code": status,
            "retry_safe": not isinstance(error, PersistenceUncertain), "outcome": "unknown" if status == 503 else "not_started"})

    async def validate_resume(self, request: OrderSelection) -> dict:
        """只读核验选单请求，供HTTP预检及忙碌重试复用。参数：request原键；返回：可信保存行；异常：404/409语义异常。"""
        from app.repositories.actions import ActionNotFound
        row = await self.chat.get_workflow_turn(request.session_id, request.user_id, request.source_turn_id)
        saved = row.get("order_selection") if row else None
        if not isinstance(saved, dict) or saved.get("action_id") != request.action_id:
            raise ActionNotFound("该选单动作不存在")
        from app.tools.business import owns_order
        if request.order_id not in {item["order_id"] for item in saved.get("orders", [])} or not owns_order(request.user_id, request.order_id):
            raise ActionNotFound("该订单不存在")
        if saved.get("status") == "cancelled" or row["turn_status"] in {"cancelled", "failed"}:
            raise OrderSelectionConflict("这张选单卡已失效，请使用当前页面的动作")
        if saved.get("status") == "accepted" and saved.get("accepted_order_id") != request.order_id:
            raise OrderSelectionConflict("同一动作不能更换订单，请继续原订单")
        return row

    async def stream_resume(self, request: OrderSelection) -> AsyncIterator[ChatEvent]:
        """取回原轮锁后以Command续跑，不创建用户消息或重置预算。

        参数：request冻结身份与原动作；返回：完成/等待/接受状态事件流。
        异常：取消透传；业务错误输出安全error，HTTP预检可提前返回对应状态码。
        """
        from langgraph.types import Command
        from app.repositories.actions import ActionRepository
        outcome, context, terminal = {}, None, "cancelled"
        try:
            async with self.guard.hold(request.session_id):
                try:
                    row = await self.validate_resume(request)
                    context = row["context"]
                    if row["turn_status"] == "completed":
                        if row["answer"] is None or not row["completion_metadata"]:
                            raise PersistenceUncertain("完成回执缺失")
                        yield ChatEvent(event="delta", data={"delta": row["answer"]})
                        yield ChatEvent(event="answer_metadata", data=row["completion_metadata"])
                        yield ChatEvent(event="done")
                        return
                    config = {"configurable": {"thread_id": checkpoint_thread_id(context.conversation_id)}}
                    waiting_snapshot = await self.graph.aget_state(config)
                    action = self._waiting_action(waiting_snapshot, context)
                    if action is None:
                        if row["order_selection"].get("status") == "accepted":
                            yield ChatEvent(event="action_status", data={"action_id": request.action_id, "source_turn_id": request.source_turn_id, "status": "accepted"})
                            return
                        raise PersistenceUncertain("选单与暂停检查点不匹配")
                    saved_action = {key: row["order_selection"][key] for key in ("type", "action_id", "source_turn_id", "orders")}
                    if action.model_dump(mode="json") != saved_action:
                        raise PersistenceUncertain("原选单内容不一致")
                    accepted = await ActionRepository(self.chat.factory).accept_order_selection(request)
                    # 提交接受后Command之前若断开，保留accepted/running以便原键继续。
                    outcome.update(started=True, paused=True)
                    migration_update = None
                    if waiting_snapshot.values.get("context_version") != 7:
                        from langgraph.types import Overwrite
                        from app.workflow.state import full_history_messages
                        delta = await self.chat.load_history_delta(context.conversation_id, 0)
                        turns = [turn.model_dump(mode="json") for turn in delta.turns]
                        migration_update = {"messages": Overwrite(full_history_messages(turns)),
                            "context_version": 7, "current_message_id": f"turn:{context.turn_id}:user",
                            "history_turns": [turn for turn in turns if turn["turn_id"] != context.turn_id],
                            "history_context": None, "context_notes": [], "audit_cursor": delta.last_message_id}
                    # 迁移与原Command.resume一并提交，保持同thread、同interrupt、同动作。
                    stream = self._run_graph(Command(resume=accepted.accepted_order_id, update=migration_update), context, outcome)
                    try:
                        async for event in stream:
                            # 任何新阶段表示Command已开始处理；此后普通断流沿用取消语义。
                            outcome["paused"] = False
                            yield event
                    finally:
                        with anyio.CancelScope(shield=True):
                            await stream.aclose()
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except Exception as error:
                    terminal = "failed"
                    yield self._stream_error(error)
                finally:
                    with anyio.CancelScope(shield=True):
                        if context is not None:
                            await self._cleanup_turn(context, outcome, terminal)
            if outcome.get("waiting"):
                yield ChatEvent(event="select_order", data=outcome["waiting"])
        except ConversationBusy as error:
            try:
                row = await self.validate_resume(request)
                if row["turn_status"] == "completed" and row["answer"] is not None and row["completion_metadata"]:
                    yield ChatEvent(event="delta", data={"delta": row["answer"]})
                    yield ChatEvent(event="answer_metadata", data=row["completion_metadata"])
                    yield ChatEvent(event="done")
                elif row["order_selection"].get("status") == "accepted":
                    yield ChatEvent(event="action_status", data={"action_id": request.action_id, "source_turn_id": request.source_turn_id, "status": "accepted"})
                else:
                    yield self._stream_error(error)
            except Exception as invalid:
                yield self._stream_error(invalid)

    async def stream_turn(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        """消费带namespace的安全custom事件并拥有整轮图资源。

        参数：request已校验的用户请求；返回：ChatEvent异步流，只有成功终态发布metadata/done。
        异常：取消和GeneratorExit透传；其他错误转换成不包含敏感原文的error帧。
        """
        context = ToolContext(session_id=request.session_id, user_id=request.user_id,
                              turn_id=str(uuid4()), user_message=request.message)
        outcome, terminal, stream = {}, "cancelled", None
        try:
            async with self.guard.hold(request.session_id):
                try:
                    from app.workflow.prompts import INTENT_PROMPT
                    from app.workflow.model import intent_tool_schema
                    prepare_messages([], [HumanMessage(content=request.message)], self.settings.token_budget,
                        system_text=INTENT_PROMPT, tools=[intent_tool_schema()], request_max_tokens=self.settings.request_token_budget)
                    current = await self.get_current_actions(request.session_id, request.user_id)
                    if current.resume_required:
                        raise OrderSelectionConflict("上一笔订单已经确认，请先点击继续处理")
                    await self.chat.cancel_order_wait(request.session_id, request.user_id)
                    context = await self.chat.start_turn(context, request.message)
                    outcome["started"] = True
                    config = {"configurable": {"thread_id": checkpoint_thread_id(context.conversation_id)}}
                    state = await self._initial_state(context, config)
                    stream = self._run_graph(state, context, outcome)
                    async for event in stream:
                        yield event
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except Exception as error:
                    terminal = "failed"
                    logger.warning("客服工作流失败 turn=%s thread=%s type=%s", context.turn_id,
                                   checkpoint_thread_id(context.conversation_id), type(error).__name__)
                    if isinstance(error, (InvalidToolDecision, WorkflowBudgetExceeded)) and error.audit_record:
                        # 分类等图外终止也保留诊断；此记录不具有可执行工具申请的形状。
                        try:
                            await self.chat.save_tool_calls(context, [error.audit_record])
                        except Exception as audit_error:
                            # 附加诊断保存失败不应吞掉用户可见错误，原finally仍负责标记终态。
                            logger.warning("模型诊断保存失败 turn=%s type=%s", context.turn_id, type(audit_error).__name__)
                    yield self._stream_error(error)
                finally:
                    # 关闭图会取消其模型/工具任务；屏蔽ASGI取消域直到上游和短事务真正退出。
                    with anyio.CancelScope(shield=True):
                        try:
                            if stream is not None:
                                await stream.aclose()
                        finally:
                            await self._cleanup_turn(context, outcome, terminal)
            # 等待提交、图流关闭及guard释放全部完成后，才进入可能受网络背压影响的发送。
            if outcome.get("waiting"):
                yield ChatEvent(event="select_order", data=outcome["waiting"])
        except ConversationBusy as error:
            yield self._stream_error(error)
