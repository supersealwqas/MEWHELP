"""客服 FAQ 知识检索注入和工单原子写入，不依赖模型或 SSE。"""

import hashlib
import json

from sqlalchemy.exc import IntegrityError

from app.knowledge.schemas import KnowledgeUnavailable
from app.knowledge.retrieval import HybridKnowledgeSearch, faq_rows
from app.knowledge.search import KnowledgeSearch
from app.models.customer_service import Conversation, Message, Ticket
from app.repositories.chat import ChatRepository, PersistenceUncertain, result_query
from app.schemas.tools import ToolContext, ToolResult


class SupportRepository:
    """查询和写入客服业务数据，每次调用独占短事务。"""

    def __init__(
        self,
        session_factory,
        knowledge_search: KnowledgeSearch | HybridKnowledgeSearch | None = None,
    ) -> None:
        """保存会话工厂与可选知识检索。

        参数:
            session_factory: 异步会话工厂。
            knowledge_search: 显式在线检索服务；为 None 时知识查询不可用，但工单仍可用。
        返回值:
            None
        异常:
            无
        """
        self.factory = session_factory
        self.knowledge_search = knowledge_search

    @property
    def uses_full_knowledge_query(self) -> bool:
        """判断新混合路径是否使用当前完整原话。参数：无；返回：bool；异常：无。"""
        return isinstance(self.knowledge_search, HybridKnowledgeSearch)

    async def query_faq(self, keyword: str, limit: int | None = None, *, filters: dict | None = None,
                        source_query: str | None = None) -> list[dict]:
        """通过注入检索服务查询 FAQ，不以旧 3 条覆盖混合默认 10。

        参数：keyword 混合路径为 1–2000 字符原话，旧 Dense 路径为 1–100 关键词；
            limit None 保留默认上限；filters 为白名单元数据；source_query 为可信本轮原问，旧Dense忽略此上下文。
        返回：list[dict]，最多对应默认或 limit 条 id/question/answer/category。
        异常：ValueError，参数不合法；KnowledgeUnavailable，未配置或依赖故障，不回退 LIKE。
        """
        if not keyword.strip() or (limit is not None and (type(limit) is not int or limit < 1)):
            raise ValueError("FAQ 关键词或条数不合法")
        if self.knowledge_search is None:
            raise KnowledgeUnavailable("知识检索未配置")
        if filters and not isinstance(self.knowledge_search, HybridKnowledgeSearch):
            raise KnowledgeUnavailable("元数据筛选需要混合检索服务")
        options = {"filters": filters} if filters else {}
        if source_query is not None and self.uses_full_knowledge_query:
            options["source_query"] = source_query
        if limit is None:
            return await self.knowledge_search.query_faq(keyword, **options)
        return await self.knowledge_search.query_faq(keyword, limit, **options)

    async def query_refund_policy(self, query: str, *, filters: dict | None = None) -> list[dict]:
        """独立退款证据入口，不截断去重后的三查询结果。

        参数：query 当前用户退款原话；filters 可选白名单元数据。
        返回：list[dict]，混合路径默认最多 30；不办理退款。
        异常：KnowledgeUnavailable，未配置混合检索；ValueError，输入非法；模型/存储错误。
        """
        if not isinstance(self.knowledge_search, HybridKnowledgeSearch):
            raise KnowledgeUnavailable("退款政策扩展检索需要已准备的混合索引")
        return faq_rows(await self.knowledge_search.retrieve_policy(query, filters=filters))

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
