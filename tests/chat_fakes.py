"""聊天编排测试的外部边界替身；真实服务、工具和消息转换保持运行。"""

from langchain_core.messages import AIMessage
from app.schemas.history import StoredMessage, StoredTurn


class MemoryRepository:
    """仅供离线测试记录持久化行为，不进入生产运行时。"""

    def __init__(self):
        """初始化记录。参数：无；返回：None；异常：无。"""
        self.turns, self.results, self.contexts = {}, {}, []
        self.fail_finish = False

    async def start_turn(self, context, message):
        """保存用户轮次。参数：上下文、原文；返回：带数字 ID 的上下文；异常：无。"""
        context = context.model_copy(update={"conversation_id": context.conversation_id or 1})
        self.contexts.append(context)
        self.turns[context.turn_id] = StoredTurn(turn_id=context.turn_id, turn_status="running",
            messages=[StoredMessage(role="user", content=message)])
        return context

    async def load_turns(self, conversation_id, exclude_turn_id=None):
        """读取已存轮次。参数：会话和排除轮次；返回：历史列表；异常：无。"""
        return [t for key, t in self.turns.items() if key != exclude_turn_id]

    async def save_tool_calls(self, context, calls):
        """保存申请。参数：上下文、申请；返回：None；异常：无。"""
        self.turns[context.turn_id].messages.append(StoredMessage(role="assistant", content="", tool_calls=calls))

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
        return self.results[key]

    async def finish_turn(self, context, content):
        """模拟完成提交。参数：上下文、回答；返回：None；异常：ConnectionError。"""
        if self.fail_finish:
            raise ConnectionError("SECRET")
        self.turns[context.turn_id].messages.append(StoredMessage(role="assistant", content=content))
        self.turns[context.turn_id].turn_status = "completed"

    async def mark_turn(self, context, status):
        """记录失败/取消。参数：上下文、状态；返回：None；异常：无。"""
        if self.turns[context.turn_id].turn_status == "running":
            self.turns[context.turn_id].turn_status = status


class ScriptedModel:
    """提供预定决策与增量，并记录服务传给模型的真实上下文。"""

    def __init__(self, decision=None, chunks=None):
        """配置响应。参数：可选决策和增量；返回：None；异常：无。"""
        self.decision = decision if decision is not None else AIMessage(content="规划不展示")
        self.chunks = chunks if chunks is not None else ["已", "查到"]
        self.inputs = []

    async def decide(self, messages, tools):
        """记录决策调用。参数：上下文、工具；返回：AIMessage；异常：预设异常。"""
        self.inputs.append(("decide", messages))
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision

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
