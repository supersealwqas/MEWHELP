"""真实图测试的严格外部依赖替身；无网络、模型网关和业务数据库。"""
from copy import deepcopy
from uuid import uuid4
from langchain_core.callbacks import BaseCallbackHandler
from app import config
from app.config import Settings
from app.knowledge.rerank import RankedChunk
from app.knowledge.schemas import ChunkSnapshot
from app.schemas.tools import ToolContext, ToolResult
from app.schemas.workflow import RetrievalLowConfidence
from app.workflow.graph import WorkflowDependencies
from app.workflow.state import new_turn_state
from tests.workflow.fakes import ScriptedWorkflowModel

class NodeTrace(BaseCallbackHandler):
    """在真实节点开始时观测名称，而不是用模型假调用代替图顺序。"""
    def __init__(self):
        """参数：无；返回：None；异常：无。"""
        self.nodes = []
    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """参数：LangChain回调入参；返回：None；异常：无。"""
        self.nodes.append(kwargs.get("name"))

class StrictChat:
    """限制真实仓储边界的参数形态，记录配对写入和完成顺序。"""
    def __init__(self):
        """参数：无；返回：None；异常：无。"""
        self.calls, self.results, self.finished, self.order = [], [], [], []
        self.history, self.next_id = {}, 1
        self.snapshots = {}

    async def save_retrieval_snapshot(self, context, snapshot):
        """按原轮保存检索脚本实际证据。参数：可信上下文/快照；返回：None；异常：冲突事实。"""
        from app.knowledge.snapshots import merge_snapshots
        self.snapshots[context.turn_id] = merge_snapshots(self.snapshots.get(context.turn_id), snapshot)

    def _record(self, context, role, content, **fields):
        """记录本例等价原始流水。参数：上下文、角色、正文、工具字段；返回：None；异常：字段错误。"""
        from app.schemas.history import StoredMessage, StoredTurn
        if context.turn_id not in self.history:
            self.history[context.turn_id] = StoredTurn(turn_id=context.turn_id, turn_status="running", messages=[
                StoredMessage(id=self.next_id, role="user", content=context.user_message)])
            self.next_id += 1
        self.history[context.turn_id].messages.append(StoredMessage(id=self.next_id, role=role, content=content, **fields))
        self.next_id += 1

    async def load_history_delta(self, conversation_id, after_message_id, *, refresh_turn_id=None, exclude_turn_id=None):
        """模拟已提交轮的增量回读。参数：真实仓储游标契约；返回：独立HistoryDelta；异常：校验失败。"""
        from app.schemas.workflow import HistoryDelta
        turns = [turn.model_copy(deep=True) for turn in self.history.values() if turn.turn_id != exclude_turn_id
                 and (turn.turn_id == refresh_turn_id or any(row.id > after_message_id for row in turn.messages))]
        return HistoryDelta(turns=turns, last_message_id=max(after_message_id, self.next_id - 1))
    async def save_tool_calls(self, context, calls, *, deduplicate=False, bindings=None):
        """参数：可信上下文与申请；返回：None；异常：断言失败。"""
        assert isinstance(context, ToolContext)
        calls = [{**call, **({"binding": bindings[call["id"]]} if bindings and call["id"] in bindings else {})} for call in calls]
        if deduplicate:
            by_id = {call["id"]: call for call in self.calls if "id" in call}
            assert all(call["id"] not in by_id or by_id[call["id"]] == call for call in calls)
            calls = [call for call in calls if call["id"] not in by_id]
            if not calls:
                return
        self.calls.extend(deepcopy(calls))
        self.order.append("calls")
        self._record(context, "assistant", "", tool_calls=deepcopy(calls))
    async def save_ticket_intent(self, context, intent):
        """保存测试原轮短期诉求，供后续真实图决定读取。

        参数：context 原轮，intent 有界状态或空；返回：None；异常：无。
        """
        if context.turn_id in self.history:
            self.history[context.turn_id].ticket_intent = intent.model_dump(mode="json") if intent else None
    async def get_pending_ticket(self, context):
        """读取替身原轮确认状态。参数：context 原轮；返回：冻结流程或 None；异常：无。"""
        turn = self.history.get(context.turn_id)
        return deepcopy(turn.ticket_confirmation) if turn else None
    async def finish_ticket_preview(self, context, result):
        """没有预览的拒绝无需收尾，真实暂停测试使用 SQLite 仓储。

        参数：context 原调用，result 确定结果；返回：原冻结流程或 None；异常：无。
        """
        return await self.get_pending_ticket(context)
    async def save_tool_result(self, context, result):
        """参数：可信上下文与结果；返回：ToolResult；异常：断言失败。"""
        assert context.tool_call_id and isinstance(result, ToolResult)
        self.results.append((context.tool_call_id, result))
        self.order.append("result")
        self._record(context, "tool", result.model_dump_json(), tool_call_id=context.tool_call_id)
        return result
    async def finish_turn(self, context, content, *, low_confidence=None, citations, suggested_actions,
                          workflow_metadata, refund_action=None, retrieval_snapshot=None, pool_signals=None):
        """参数：最终仓储契约的全部字段；返回：完成元数据；异常：断言失败。"""
        assert isinstance(context, ToolContext) and content
        assert {"intent", "route", "stop_reason", "budget", "tool_count", "decision_count"} <= workflow_metadata.keys()
        assert all(item.model_dump(mode="json") for item in citations + suggested_actions)
        assert low_confidence is None or pool_signals is None
        if pool_signals:
            low = next((signal for signal in pool_signals if signal.source == "retrieval_low_conf"), None)
            low_confidence = RetrievalLowConfidence(reason=low.reason) if low else low_confidence
        if retrieval_snapshot is not None:
            assert retrieval_snapshot == self.snapshots.get(context.turn_id)
        # 复刻真实仓储的低证据契约：弱证据入池必须与固定拒答话术同时保存。
        # 替身不校验这一条时，图里传错 low_confidence 只会在真实服务上报错。
        if isinstance(low_confidence, RetrievalLowConfidence):
            from app.workflow.routing import fixed_reply
            assert content == fixed_reply("fallback")["answer"], (
                "检索弱证据入池必须与固定兜底一起保存")
        record = {"context": context, "content": content, "low_confidence": low_confidence,
                  "citations": citations, "suggested_actions": suggested_actions, "workflow_metadata": workflow_metadata}
        self.finished.append(record)
        self.order.append("finish")
        self._record(context, "assistant", content)
        self.history[context.turn_id].turn_status = "completed"
        return {"turn_id": context.turn_id, "citations": [x.model_dump(mode="json") for x in citations],
                "suggested_actions": [x.model_dump(mode="json") for x in suggested_actions], "workflow": workflow_metadata,
                "refund_action": refund_action.model_dump(mode="json") if refund_action else None}

class StrictExecutor:
    """只记录被图批准的执行；模拟已有执行器的结果落库和状态回调。"""
    def __init__(self, chat, results=()):
        """参数：chat仓储、results结果脚本；返回：None；异常：无。"""
        self.chat, self.script, self.calls = chat, list(results), []
    async def execute(self, call, context, on_status, *, scope=None):
        """参数：工具、身份、状态回调；返回：ToolResult；异常：脚本错误或断言失败。"""
        assert call["id"] == context.tool_call_id
        if isinstance(self.chat, StrictChat):
            assert any(item.get("id") == call["id"] for item in self.chat.calls)
        if scope is not None and scope.rejection is not None:
            result = await self.chat.save_tool_result(context, scope.rejection)
            await on_status({"tool_call_id": call["id"], "name": call["name"], "status": "failed", "label": "未执行", "is_mock": False})
            return result
        self.calls.append(deepcopy(call))
        await on_status({"tool_call_id": call["id"], "name": call["name"], "status": "running", "label": "正在查询", "is_mock": False})
        assert self.script, "发生未预期的工具执行"
        result = self.script.pop(0)
        if isinstance(result, Exception):
            from app.repositories.chat import PersistenceUncertain
            if isinstance(result, PersistenceUncertain):
                raise result
            result = ToolResult.failed("TOOL_EXECUTION_FAILED", "工具执行未能完成")
        result = await self.chat.save_tool_result(context, result)
        await on_status({"tool_call_id": call["id"], "name": call["name"],
                         "status": "succeeded" if result.ok else "failed",
                         "label": "已完成" if result.ok else "未完成", "is_mock": result.is_mock})
        return result

class StrictSearch:
    """普通与退款入口可独立断言，异常不自动转换为空结果。"""
    def __init__(self, ranked=(), error=None):
        """参数：ranked权威结果、error可选故障；返回：None；异常：无。"""
        self.ranked, self.error, self.calls = list(ranked), error, []
    async def retrieve(self, query, *, budget, prepared_queries=None, history="", filters=None, source_query=None, capture=None):
        """参数：query原问、budget独占账本；返回：排名；异常：脚本故障。"""
        return await self._take("retrieve", query, budget, capture, prepared_queries)
    async def retrieve_policy(self, query, *, budget, prepared_queries=None, history="", filters=None, capture=None):
        """参数：query原问、budget独占账本、filters来源范围；返回：排名；异常：脚本故障。"""
        return await self._take("retrieve_policy", query, budget, capture, prepared_queries)
    async def _take(self, kind, query, budget, capture, prepared_queries):
        """参数：kind入口、query原問、budget账本；返回：排名；异常：脚本故障。"""
        self.calls.append((kind, query))
        if capture is not None:
            from app.knowledge.snapshots import chunk_snapshot
            from app.schemas.flywheel import RetrievalAttempt
            for actual in prepared_queries or [query]:
                await capture.record(RetrievalAttempt(attempt_id=capture.attempt_id(actual), query=actual,
                    top_k=capture.top_k, candidate_count=None if self.error else len(self.ranked),
                    status="fault" if self.error else "ok" if self.ranked else "empty",
                    chunks=[] if self.error else [chunk_snapshot(item.chunk, i + 1, rerank_score=item.score)
                        for i, item in enumerate(self.ranked[:capture.top_k])],
                    fault=type(self.error).__name__ if self.error else None))
        if self.error:
            raise self.error
        return self.ranked

class StrictExpander:
    """明确返回标准问法的离线扩写边界，记录调用并扣除真实阶段预算。"""

    def __init__(self):
        """建立调用列表。参数：无；返回：None；异常：无。"""
        self.calls = []

    async def expand(self, query, *, simple_query, budget, required_count=None):
        """遵照流程条数返回查询。参数：query种子、simple_query标准问法、budget账本、required_count指定条数；返回：ExpandedQueries；异常：预算错误。"""
        from app.knowledge.query_rewrite import ExpandedQueries
        self.calls.append((query, simple_query))
        budget.reserve("expand", 100, 2048)
        return ExpandedQueries(queries=[f"{query} {aspect}" for aspect in ("适用条件", "限制", "流程")]
                               if required_count == 3 else [simple_query])


def dependencies(*, intents=("订单",), decisions=(), answers=(), ranked=(), results=(), understandings=(), **settings):
    """参数：有限依赖脚本与配置覆盖；返回：WorkflowDependencies；异常：配置校验错误。"""
    chat = StrictChat()
    # 沿用全局夹具的离线快照，避免SDK导入.env污染预算；用例显式覆盖仍参与校验。
    options = {**config.settings.model_dump(), **settings}
    from app.tools.registry import build_registry
    if not decisions and intents and all(isinstance(intent, str) and intent in {
            "商品咨询", "退款退货", "售后", "投诉", "闲聊"} for intent in intents):
        # CH08原固定出口也先真实决定一次，旧场景明确选择继续其原出口。
        from app.schemas.workflow import AgentDecision
        decisions = [AgentDecision(kind="answer") for _ in intents]
    return WorkflowDependencies(ScriptedWorkflowModel(intents=intents, decisions=decisions, answers=answers, understandings=understandings),
        StrictSearch(ranked), StrictExecutor(chat, results), chat,
        Settings(_env_file=None, **options), query_expander=StrictExpander(), registry=build_registry())

def turn(query="查订单1001", *, session="session", turn_id=None):
    """参数：query原问、session会话、turn_id可选轮ID；返回：新轮State；异常：原问错误。"""
    return new_turn_state(ToolContext(conversation_id=1, session_id=session, user_id="u", turn_id=turn_id or str(uuid4()), user_message=query), [], 0)

def ranked(score=0.9, *, number=1, answer="完整政策正文"):
    """参数：score归一分、number主键、answer原文；返回：RankedChunk；异常：模型校验错误。"""
    return RankedChunk(chunk=ChunkSnapshot(id=number, source_key="doc:test", ordinal=number,
        text={"category": "规则", "questions": "怎么退", "answer": answer}, content_hash="a" * 64,
        embedding_revision="test", content_type="policy", vector_id=str(number), vectorize_status="done", is_active=True), score=score)

def call(number, name="query_order", **args):
    """参数：number调用标识、name工具名、args业务参数；返回：严格申请；异常：无。"""
    return {"id": str(number), "name": name, "args": args or {"order_id": str(number)}}
