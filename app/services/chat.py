"""单轮 Function Calling 服务，负责会话互斥、工具事实回灌和流式收敛。"""

import asyncio
import logging
from contextlib import suppress
from uuid import uuid4
import anyio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.core.memory import ContextBudgetExceeded, assemble_turns, trim_turns, valid_tool_calls
from app.core.prompts import CUSTOMER_SERVICE_SYSTEM
from app.repositories.chat import PersistenceUncertain
from app.schemas.chat import ChatEvent, ChatRequest
from app.schemas.tools import ToolContext
from app.tools.executor import ToolExecutor, failure

logger = logging.getLogger(__name__)


class ChatService:
    """一个应用实例共享的聊天协调器；数据库事务不跨模型请求。"""

    def __init__(self, chat_repository, support_repository, model_adapter, registry, settings):
        """注入业务依赖。参数：存储、模型、注册表和配置；返回：None；异常：执行器配置错误。"""
        self.chat, self.model, self.registry, self.settings = chat_repository, model_adapter, registry, settings
        self.executor = ToolExecutor(registry, chat_repository, support_repository, settings.tool_timeout_seconds,
                                     settings.tool_max_retries, settings.tool_retry_delay_seconds)
        self._active: set[str] = set()

    async def _execute(self, call, context):
        """将执行状态转换为流。参数：申请和上下文；返回：状态及最终结果流；异常：执行或持久化异常。"""
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

    async def stream_turn(self, request: ChatRequest):
        """处理一次提问。参数：已校验请求；返回：ChatEvent 异步流；异常：客户端取消向外传播。"""
        key = request.session_id
        if key in self._active:
            yield ChatEvent(event="error", data={"error": "当前会话正在回复，请稍后再发"})
            return
        # 检查与占用之间没有 await，单 worker 中不会出现竞争窗口。
        self._active.add(key)
        context = ToolContext(session_id=key, user_id=request.user_id, turn_id=str(uuid4()), user_message=request.message)
        completed, started, terminal = False, False, "cancelled"
        try:
            current = [HumanMessage(content=request.message)]
            trim_turns([], current, self.settings.token_budget)
            context = await self.chat.start_turn(context, request.message)
            started = True
            history = assemble_turns(await self.chat.load_turns(context.conversation_id, exclude_turn_id=context.turn_id))
            messages = [SystemMessage(content=CUSTOMER_SERVICE_SYSTEM), *trim_turns(history, current, self.settings.token_budget)]
            decision = await self.model.decide(messages, self.registry.all())
            if decision.invalid_tool_calls or not valid_tool_calls(decision.tool_calls):
                raise ValueError("模型工具申请不完整")
            calls = decision.tool_calls
            if calls:
                await self.chat.save_tool_calls(context, calls)
                current.append(AIMessage(content="", tool_calls=calls))
                if len(calls) > 1:
                    # 拒绝整批，给每个合法 ID 配对安全错误；不能默默只执行第一项。
                    for call in calls:
                        call_context = context.model_copy(update={"tool_call_id": call["id"]})
                        result = await self.chat.save_tool_result(call_context, failure("SINGLE_TOOL_LIMIT", "每次只能查询一项，请明确最需要处理的问题"))
                        current.append(ToolMessage(content=result.model_dump_json(), tool_call_id=call["id"]))
                else:
                    context = context.model_copy(update={"tool_call_id": calls[0]["id"]})
                    execution = self._execute(calls[0], context)
                    try:
                        async for item in execution:
                            if isinstance(item, ChatEvent):
                                yield item
                            else:
                                current.append(ToolMessage(content=item.model_dump_json(), tool_call_id=context.tool_call_id))
                    finally:
                        await execution.aclose()
            messages = [SystemMessage(content=CUSTOMER_SERVICE_SYSTEM), *trim_turns(history, current, self.settings.token_budget)]
            text = []
            answer_stream = self.model.stream_answer(messages)
            try:
                async for delta in answer_stream:
                    if delta:
                        text.append(delta)
                        yield ChatEvent(event="delta", data={"delta": delta})
            finally:
                await answer_stream.aclose()
            answer = "".join(text)
            if not answer.strip():
                raise ValueError("模型未返回有效文本")
            await self.chat.finish_turn(context, answer)
            completed = True
            yield ChatEvent(event="done")
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as error:
            terminal = "failed"
            logger.warning("客服轮次失败：%s", type(error).__name__)
            if isinstance(error, ContextBudgetExceeded):
                message = str(error)
            elif isinstance(error, PersistenceUncertain):
                message = "暂时无法核实处理状态，请稍后查看，勿重复提交工单"
            else:
                message = "本轮暂时未能完成，请稍后重试；已提交的工具结果会保留"
            yield ChatEvent(event="error", data={"error": message})
        finally:
            try:
                if started and not completed:
                    with anyio.CancelScope(shield=True):
                        await self.chat.mark_turn(context, terminal)
            except Exception as error:
                logger.warning("轮次终态保存失败：%s", type(error).__name__)
            finally:
                self._active.discard(key)
