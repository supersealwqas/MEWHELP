"""聊天编排测试的外部边界替身；真实服务、工具和消息转换保持运行。"""

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.schemas.history import StoredMessage, StoredTurn


class MemoryRepository:
    """仅供离线测试记录持久化行为，不进入生产运行时。"""

    def __init__(self):
        """初始化记录。参数：无；返回：None；异常：无。"""
        self.turns, self.results, self.contexts = {}, {}, []
        self.fail_finish = False
        self.low_confidence = []
        self.metadata = {}
        self.message_ids, self.last_message_id = {}, 0

    def _message_added(self, turn_id):
        """分配单调审计主键。参数：turn_id新增消息的轮；返回：None；异常：无。"""
        self.last_message_id += 1
        self.message_ids.setdefault(turn_id, []).append(self.last_message_id)

    async def start_turn(self, context, message):
        """保存用户轮次。参数：上下文、原文；返回：带数字 ID 的上下文；异常：无。"""
        context = context.model_copy(update={"conversation_id": context.conversation_id or 1})
        self.contexts.append(context)
        self.turns[context.turn_id] = StoredTurn(turn_id=context.turn_id, turn_status="running",
            messages=[StoredMessage(role="user", content=message)])
        self._message_added(context.turn_id)
        return context

    async def load_turns(self, conversation_id, exclude_turn_id=None):
        """读取已存轮次。参数：会话和排除轮次；返回：历史列表；异常：无。"""
        return [t for key, t in self.turns.items() if key != exclude_turn_id]

    async def load_history_delta(self, conversation_id, after_message_id, *, refresh_turn_id=None, exclude_turn_id=None):
        """模拟按消息主键返回完整轮。参数：增量仓储契约；返回：HistoryDelta；异常：无。"""
        from app.schemas.workflow import HistoryDelta
        return HistoryDelta(turns=[turn.model_copy(deep=True) for key, turn in self.turns.items()
            if key != exclude_turn_id and (key == refresh_turn_id or any(i > after_message_id for i in self.message_ids.get(key, [])))],
            last_message_id=self.last_message_id)

    async def get_workflow_turn(self, session_id, user_id, turn_id=None):
        """读取内存原轮快照。参数：身份与可选轮；返回：状态或None；异常：无。"""
        contexts = [ctx for ctx in self.contexts if ctx.session_id == session_id and ctx.user_id == user_id
                    and (turn_id is None or ctx.turn_id == turn_id)]
        if not contexts:
            return None
        ctx = contexts[-1]
        turn = self.turns[ctx.turn_id]
        return {"context": ctx, "turn_status": turn.turn_status, "order_selection": turn.order_selection,
                "answer": turn.messages[-1].content if turn.turn_status == "completed" else None,
                "completion_metadata": self.metadata.get(ctx.turn_id)}

    async def cancel_order_wait(self, session_id, user_id):
        """普通聊天替身不产生等待。参数：会话/用户；返回：None；异常：无。"""

    async def save_tool_calls(self, context, calls, *, deduplicate=False):
        """保存申请。参数：上下文、申请；返回：None；异常：无。"""
        self.turns[context.turn_id].messages.append(StoredMessage(role="assistant", content="", tool_calls=calls))
        self._message_added(context.turn_id)

    async def get_tool_result(self, context):
        """读取事实。参数：调用上下文；返回：结果或空；异常：无。"""
        return self.results.get((context.turn_id, context.tool_call_id))

    async def save_tool_result(self, context, result):
        """模拟唯一写入。参数：上下文和结果；返回：已存结果；异常：无。"""
        key = (context.turn_id, context.tool_call_id)
        if key not in self.results:
            self.results[key] = result
            self.turns[context.turn_id].messages.append(StoredMessage(role="tool",
                content=result.model_dump_json(), tool_call_id=context.tool_call_id))
            self._message_added(context.turn_id)
        return self.results[key]

    async def finish_turn(self, context, content, *, low_confidence=None, citations=None, suggested_actions=None, workflow_metadata=None, refund_action=None):
        """模拟原子完成/入池。参数：上下文、回答、失败、引用、建议、流程摘要；返回：元数据；异常：ConnectionError。"""
        from app.knowledge.sources import answer_metadata
        if self.fail_finish:
            raise ConnectionError("SECRET")
        if self.turns[context.turn_id].turn_status == "completed":
            return self.metadata[context.turn_id]
        self.turns[context.turn_id].messages.append(StoredMessage(role="assistant", content=content))
        self._message_added(context.turn_id)
        if low_confidence is not None:
            self.low_confidence.append({"raw_question": context.user_message, "source": low_confidence.source,
                                        "reason": low_confidence.reason, "conversation_id": context.conversation_id})
        self.turns[context.turn_id].turn_status = "completed"
        self.metadata[context.turn_id] = answer_metadata(context, citations or [])
        if suggested_actions is not None:
            self.metadata[context.turn_id]["suggested_actions"] = [item.model_dump(mode="json") for item in suggested_actions]
        if workflow_metadata is not None:
            self.metadata[context.turn_id]["workflow"] = workflow_metadata
        if refund_action is not None:
            self.metadata[context.turn_id]["refund_action"] = refund_action.model_dump(mode="json")
        return self.metadata[context.turn_id]

    async def mark_turn(self, context, status):
        """记录失败/取消。参数：上下文、状态；返回：None；异常：无。"""
        if self.turns[context.turn_id].turn_status == "running":
            self.turns[context.turn_id].turn_status = status


class ScriptedModel:
    """提供预定决策与增量，并记录服务传给模型的真实上下文。"""

    def __init__(self, decision=None, chunks=None, *, decisions=None):
        """配置响应。参数：单次决策、增量或显式多步决策序列；返回：None；异常：无。"""
        self.decision = decision if decision is not None else AIMessage(content="规划不展示")
        self.decisions = list(decisions) if decisions is not None else None
        self._decision_index = 0
        self.chunks = chunks if chunks is not None else ["已", "查到"]
        self.inputs = []

    async def decide(self, messages, tools):
        """记录决策调用。参数：上下文、工具；返回：AIMessage；异常：预设异常。"""
        self.inputs.append(("decide", messages))
        if self.decisions is not None:
            decision = (self.decisions[self._decision_index]
                        if self._decision_index < len(self.decisions) else AIMessage(content=""))
            self._decision_index += 1
        else:
            # 旧用例每个用户轮只申请一次；已有当前轮结果时明确结束决策。
            last_user = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
            has_result = any(isinstance(message, ToolMessage) for message in messages[last_user + 1:])
            decision = AIMessage(content="") if has_result else self.decision
        if isinstance(decision, Exception):
            raise decision
        return decision

    async def answer_knowledge(self, messages):
        """提供旧聊天测试的显式生成边界，真实JSON解析另有定向测试。

        参数：messages 为实际知识自评输入；返回：KnowledgeAnswer；异常：预设模型错误。
        """
        from app.knowledge.answering import KnowledgeAnswer
        self.inputs.append(("answer", messages))
        text = []
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            text.append(chunk)
        answer = "".join(text)
        if len(answer) > 12000:
            from app.services.chat import AnswerTooLong
            raise AnswerTooLong("回答过长")
        if not answer.strip():
            raise ValueError("模型未返回有效文本")
        hits = []
        for message in messages:
            if isinstance(message, ToolMessage):
                data = json.loads(message.content).get("data")
                if isinstance(data, dict):
                    hits.extend(data.get("hits", []))
        if not hits:
            return KnowledgeAnswer(useful=False, reason="当前证据不足", answer="", citation_numbers=[])
        return KnowledgeAnswer(useful=True, reason="", answer=answer + "[1]", citation_numbers=[1])

    async def stream_answer(self, messages):
        """逐块生成最终文本。参数：模型上下文；返回：异步文本流；异常：预设异常。"""
        self.inputs.append(("answer", messages))
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class FAQRepository:
    """记录实际 FAQ 访问次数，返回零命中供编排测试。"""

    def __init__(self):
        """初始化计数。参数：无；返回：None；异常：无。"""
        self.keywords = []

    async def query_faq(self, keyword):
        """模拟数据库查询边界。参数：关键词；返回：空列表；异常：无。"""
        self.keywords.append(keyword)
        return []


def make_chat_service(decision=None, chunks=None, *, decisions=None):
    """构造真实聊天服务并仅替换外部模型和数据库。

    参数：decision为单决策；chunks为回答增量；decisions为可选多步决策序列。
    返回：服务、会话仓库、客服仓库和模型替身。
    异常：服务或工具注册构造错误。
    """
    from app.services.chat import LegacyChatService
    from app.tools.registry import build_registry

    repository, support = MemoryRepository(), FAQRepository()
    model = ScriptedModel(decision, chunks, decisions=decisions)
    config = SimpleNamespace(token_budget=12000, request_token_budget=16000,
                             tool_timeout_seconds=0.05, tool_max_retries=1,
                             tool_retry_delay_seconds=0, tool_max_calls_per_turn=4)
    return LegacyChatService(repository, support, model, build_registry(), config), repository, support, model


def make_workflow_service(*, intents=("订单",), decisions=None, answers=None, ranked=(), settings=None):
    """构造真实新图和执行器，内存保存器仅用于离线单测。

    参数：有限模型脚本、ranked检索结果、settings可选隔离配置。
    返回：新服务、仓储、同一执行器的support、模型替身；异常：图/配置错误。
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from app.services.chat import ChatService
    from app.schemas.workflow import AgentDecision
    from app.tools.executor import ToolExecutor
    from app.tools.registry import build_agent_registry
    from app.workflow.graph import build_workflow
    from tests.workflow.graph_fakes import dependencies

    deps = dependencies(intents=intents, decisions=decisions if decisions is not None else [AgentDecision(kind="answer")],
                        answers=answers if answers is not None else [["已", "查到"]], ranked=ranked)
    if settings is not None:
        deps.settings = settings
    repository, support = MemoryRepository(), FAQRepository()
    deps.chat = repository
    deps.executor = ToolExecutor(build_agent_registry(), repository, support, deps.settings.tool_timeout_seconds,
        deps.settings.tool_max_retries, deps.settings.tool_retry_delay_seconds)
    graph = build_workflow(deps, checkpointer=InMemorySaver())
    service = ChatService(repository, graph, deps.executor, deps.settings)
    return service, repository, support, deps.model
