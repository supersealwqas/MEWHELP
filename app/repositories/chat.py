"""会话与消息短事务存储，负责结果唯一性和启动时的中断恢复。"""

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from app.models.customer_service import Conversation, Message
from app.schemas.history import StoredMessage, StoredTurn
from app.schemas.tools import ToolContext, ToolResult


def result_query(context: ToolContext):
    """定位唯一工具结果。参数：context 调用身份；返回：Select；异常：无。"""
    return select(Message).where(Message.conversation_id == context.conversation_id,
                                 Message.turn_id == context.turn_id,
                                 Message.tool_call_id == context.tool_call_id, Message.role == "tool")


class PersistenceUncertain(RuntimeError):
    """数据库提交结果无法确认；不能当作业务失败重新执行。"""


class ChatRepository:
    """每次操作独立会话，不把数据库事务带入模型网络等待。"""

    def __init__(self, session_factory):
        """保存会话工厂。参数：session_factory 异步工厂；返回：None；异常：无。"""
        self.factory = session_factory

    async def start_turn(self, context: ToolContext, message: str) -> ToolContext:
        """映射客户端标识并保存用户消息。参数：上下文及原文；返回：含数字主键的上下文；异常：归属冲突或数据库错误。"""
        async with self.factory.begin() as session:
            conversation = await session.scalar(select(Conversation).where(Conversation.session_id == context.session_id))
            if conversation is None:
                conversation = Conversation(session_id=context.session_id, user_id=context.user_id)
                session.add(conversation)
                await session.flush()
            elif conversation.user_id != context.user_id:
                raise ValueError("当前会话的用户标识不匹配")
            result = context.model_copy(update={"conversation_id": conversation.id})
            session.add(Message(conversation_id=conversation.id, turn_id=context.turn_id,
                                role="user", content=message))
        return result

    async def load_turns(self, conversation_id: int, exclude_turn_id: str | None = None) -> list[StoredTurn]:
        """读取按主键有序的历史轮次。参数：会话及可排除轮次；返回：StoredTurn 列表；异常：数据库读取失败。"""
        statement = select(Message).where(Message.conversation_id == conversation_id)
        if exclude_turn_id:
            statement = statement.where(Message.turn_id != exclude_turn_id)
        async with self.factory() as session:
            rows = (await session.scalars(statement.order_by(Message.id))).all()
        turns: dict[str, StoredTurn] = {}
        for row in rows:
            turn = turns.setdefault(row.turn_id, StoredTurn(turn_id=row.turn_id, turn_status=row.turn_status))
            turn.messages.append(StoredMessage(role=row.role, content=row.content or "",
                                              tool_calls=row.tool_calls, tool_call_id=row.tool_call_id))
        return list(turns.values())

    async def save_tool_calls(self, context: ToolContext, calls: list[dict]) -> None:
        """执行前保存模型原申请。参数：上下文和申请列表；返回：None；异常：数据库写入失败。"""
        async with self.factory.begin() as session:
            session.add(Message(conversation_id=context.conversation_id, turn_id=context.turn_id,
                                role="assistant", content=None, tool_calls=calls))

    async def get_tool_result(self, context: ToolContext) -> ToolResult | None:
        """用新会话核实已提交结果。参数：调用上下文；返回：原结果或 None；异常：数据库/格式错误。"""
        async with self.factory() as session:
            row = await session.scalar(result_query(context))
            return ToolResult.model_validate_json(row.content) if row else None

    async def save_tool_result(self, context: ToolContext, result: ToolResult) -> ToolResult:
        """幂等保存唯一结果。参数：上下文及结果；返回：实际落库结果；异常：数据库写入失败。"""
        if not context.tool_call_id:
            raise ValueError("工具结果缺少调用 ID")
        try:
            async with self.factory.begin() as session:
                existing = await session.scalar(result_query(context))
                if existing:
                    return ToolResult.model_validate_json(existing.content)
                session.add(Message(conversation_id=context.conversation_id, turn_id=context.turn_id,
                                    role="tool", tool_call_id=context.tool_call_id, content=result.model_dump_json()))
        except IntegrityError:
            # 唯一键冲突事务已回滚，改用新会话读取获胜结果。
            existing_result = await self.get_tool_result(context)
            if existing_result is None:
                raise
            return existing_result
        return result

    async def finish_turn(self, context: ToolContext, content: str) -> None:
        """原子保存最终回复与完成状态。参数：上下文及完整回复；返回：None；异常：数据库写入失败。"""
        async with self.factory.begin() as session:
            session.add(Message(conversation_id=context.conversation_id, turn_id=context.turn_id,
                                role="assistant", content=content, turn_status="completed"))
            await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                                                        Message.turn_id == context.turn_id).values(turn_status="completed"))

    async def mark_turn(self, context: ToolContext, status: str) -> None:
        """标记失败或中断并保护已提交终态。参数：上下文与 failed/cancelled；返回：None；异常：状态或数据库错误。"""
        if status not in {"failed", "cancelled"}:
            raise ValueError("非法终止状态")
        async with self.factory.begin() as session:
            await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                                                        Message.turn_id == context.turn_id,
                                                        Message.turn_status == "running").values(turn_status=status))

    async def recover_interrupted_turns(self) -> int:
        """单实例启动前恢复遗留轮次。参数：无；返回：恢复轮数；异常：数据库错误，应终止启动。"""
        async with self.factory.begin() as session:
            turns = (await session.execute(select(Message.conversation_id, Message.turn_id).where(
                Message.turn_status == "running").distinct())).all()
            await session.execute(update(Message).where(Message.turn_status == "running").values(turn_status="cancelled"))
        return len(turns)
