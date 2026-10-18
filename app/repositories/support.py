"""FAQ 字面查询和工单原子写入，不依赖模型或 SSE。"""

import hashlib
import json
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from app.models.customer_service import Conversation, FAQ, Message, Ticket
from app.repositories.chat import ChatRepository, PersistenceUncertain, result_query
from app.schemas.tools import ToolContext, ToolResult


class SupportRepository:
    """查询和写入客服业务数据，每次调用独占短事务。"""

    def __init__(self, session_factory):
        """保存会话工厂。参数：session_factory 异步工厂；返回：None；异常：无。"""
        self.factory = session_factory

    async def query_faq(self, keyword: str, limit: int = 3) -> list[dict]:
        """LIKE 字面关键词查询。参数：非空 keyword，limit 取 1–3；返回：问答列表；异常：校验或数据库错误。"""
        if not keyword.strip() or not 1 <= limit <= 3:
            raise ValueError("FAQ 关键词或条数不合法")
        # 自动转义通配符，并使用绑定参数而非拼接 SQL。
        statement = select(FAQ).where(or_(*(field.contains(keyword, autoescape=True)
            for field in [FAQ.question, FAQ.answer, FAQ.category]))).order_by(FAQ.id).limit(limit)
        async with self.factory() as session:
            rows = (await session.scalars(statement)).all()
            return [{"id": row.id, "question": row.question, "answer": row.answer[:2000],
                     "category": row.category} for row in rows]

    async def create_ticket(self, context: ToolContext, description: str, ticket_type: str) -> ToolResult:
        """幂等创建工单并保存成功事实。参数：可信上下文、描述、售后/投诉/咨询；返回：原工单结果；异常：数据库或一致性错误。"""
        if not context.tool_call_id or not description.strip() or ticket_type not in {"售后", "投诉", "咨询"}:
            raise ValueError("工单参数不合法")
        identity = json.dumps([context.conversation_id, context.turn_id, context.tool_call_id])
        number = "T" + hashlib.sha256(identity.encode()).hexdigest()[:31]
        result = ToolResult(ok=True, data={"ticket_no": number, "ticket_type": ticket_type, "status": "待处理"})
        try:
            async with self.factory.begin() as session:
                existing = await session.scalar(result_query(context))
                if existing:
                    return ToolResult.model_validate_json(existing.content)
                if await session.get(Ticket, number):
                    raise PersistenceUncertain("工单与结果记录不一致，需核查")
                conversation = await session.get(Conversation, context.conversation_id)
                if conversation is None or conversation.user_id != context.user_id:
                    raise ValueError("工单会话不匹配")
                session.add(Ticket(ticket_no=number, conversation_id=context.conversation_id,
                                   description=description, ticket_type=ticket_type))
                conversation.status = "已转人工"
                # 业务数据与成功事实同事务提交，不留下孤立工单。
                session.add(Message(conversation_id=context.conversation_id, turn_id=context.turn_id,
                                    role="tool", tool_call_id=context.tool_call_id, content=result.model_dump_json()))
        except IntegrityError:
            persisted = await ChatRepository(self.factory).get_tool_result(context)
            if persisted is None:
                raise
            return persisted
        return result
