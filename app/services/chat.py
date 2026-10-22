"""有界多工具聊天服务，负责会话互斥、顺序执行、事实回灌和最终流式回答。"""

import asyncio
import json
import logging
from contextlib import aclosing, suppress
from dataclasses import dataclass
from uuid import uuid4
import anyio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.core.memory import ContextBudgetExceeded, assemble_turns, prepare_messages, valid_tool_calls
from app.core.prompts import ANSWER_NUDGE, ANSWER_SYSTEM, CUSTOMER_SERVICE_SYSTEM, TOOL_DECISION_SYSTEM
from app.repositories.chat import PersistenceUncertain
from app.schemas.chat import ChatEvent, ChatRequest, InvalidToolDecision
from app.schemas.tools import ToolContext, ToolResult
from app.tools.executor import ToolExecutor

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


class ChatService:
    """一个应用实例共享的聊天协调器；数据库事务不跨模型请求。"""

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

    async def stream_turn(self, request: ChatRequest):
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
                await self.chat.finish_turn(context, chain.ticket_receipt + chain.rejection)
                completed = True
                yield ChatEvent(event="done")
                return
            stage = "answer"
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
            await self.chat.finish_turn(context, answer)
            completed = True
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
