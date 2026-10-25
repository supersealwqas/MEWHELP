"""真实图测试的严格外部依赖替身；无网络、模型网关和业务数据库。"""
from copy import deepcopy
from uuid import uuid4
from langchain_core.callbacks import BaseCallbackHandler
from app.config import Settings
from app.knowledge.rerank import RankedChunk
from app.knowledge.schemas import ChunkSnapshot
from app.schemas.tools import ToolContext, ToolResult
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
    async def save_tool_calls(self, context, calls):
        """参数：可信上下文与申请；返回：None；异常：断言失败。"""
        assert isinstance(context, ToolContext)
        self.calls.extend(deepcopy(calls))
        self.order.append("calls")
    async def save_tool_result(self, context, result):
        """参数：可信上下文与结果；返回：ToolResult；异常：断言失败。"""
        assert context.tool_call_id and isinstance(result, ToolResult)
        self.results.append((context.tool_call_id, result))
        self.order.append("result")
        return result
    async def finish_turn(self, context, content, *, low_confidence, citations, suggested_actions, workflow_metadata):
        """参数：最终仓储契约的全部字段；返回：完成元数据；异常：断言失败。"""
        assert isinstance(context, ToolContext) and content
        assert {"intent", "route", "stop_reason", "budget", "tool_count", "decision_count"} <= workflow_metadata.keys()
        assert all(item.model_dump(mode="json") for item in citations + suggested_actions)
        record = {"context": context, "content": content, "low_confidence": low_confidence,
                  "citations": citations, "suggested_actions": suggested_actions, "workflow_metadata": workflow_metadata}
        self.finished.append(record)
        self.order.append("finish")
        return {"turn_id": context.turn_id, "citations": [x.model_dump(mode="json") for x in citations],
                "suggested_actions": [x.model_dump(mode="json") for x in suggested_actions], "workflow": workflow_metadata}

class StrictExecutor:
    """只记录被图批准的执行；模拟已有执行器的结果落库和状态回调。"""
    def __init__(self, chat, results=()):
        """参数：chat仓储、results结果脚本；返回：None；异常：无。"""
        self.chat, self.script, self.calls = chat, list(results), []
    async def execute(self, call, context, on_status):
        """参数：工具、身份、状态回调；返回：ToolResult；异常：脚本错误或断言失败。"""
        assert call["id"] == context.tool_call_id
        assert any(item.get("id") == call["id"] for item in self.chat.calls)
        self.calls.append(deepcopy(call))
        await on_status({"tool_call_id": call["id"], "name": call["name"], "status": "running", "label": "正在查询", "is_mock": False})
        assert self.script, "发生未预期的工具执行"
        result = self.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return await self.chat.save_tool_result(context, result)

class StrictSearch:
    """普通与退款入口可独立断言，异常不自动转换为空结果。"""
    def __init__(self, ranked=(), error=None):
        """参数：ranked权威结果、error可选故障；返回：None；异常：无。"""
        self.ranked, self.error, self.calls = list(ranked), error, []
    async def retrieve(self, query, *, budget):
        """参数：query原问、budget独占账本；返回：排名；异常：脚本故障。"""
        return self._take("retrieve", query, budget)
    async def retrieve_policy(self, query, *, budget):
        """参数：query原问、budget独占账本；返回：排名；异常：脚本故障。"""
        return self._take("retrieve_policy", query, budget)
    def _take(self, kind, query, budget):
        """参数：kind入口、query原問、budget账本；返回：排名；异常：脚本故障。"""
        self.calls.append((kind, query))
        budget.reserve("rewrite", 100, 2048)
        if self.error:
            raise self.error
        return self.ranked

def dependencies(*, intents=("订单",), decisions=(), answers=(), ranked=(), results=(), **settings):
    """参数：有限依赖脚本与配置覆盖；返回：WorkflowDependencies；异常：配置校验错误。"""
    chat = StrictChat()
    return WorkflowDependencies(ScriptedWorkflowModel(intents=intents, decisions=decisions, answers=answers),
        StrictSearch(ranked), StrictExecutor(chat, results), chat,
        Settings(_env_file=None, chat_base_url="http://127.0.0.1:9/v1", chat_model="offline", chat_api_key="offline", **settings))

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
