"""会话上下文仓储：原文只读、边界单调推进、摘要段和投影在同一短事务提交。"""

from sqlalchemy import select, text

from app.core.context_budget import TokenEstimator
from app.core.context_layers import _safe_to_demote, project_summaries
from app.core.memory import assemble_turns
from app.models.customer_service import Conversation, ConversationSummary, Message
from app.repositories.chat import stored_turns


class ContextNotFound(LookupError):
    """未知会话或不属于当前用户，对外不区分。"""


def _identifier(value: int, *, zero: bool = False) -> None:
    """校验主键范围。参数：value数字ID、zero是否允许初始零；返回：None；异常：ValueError。"""
    if type(value) is not int or not (0 if zero else 1) <= value <= 2**64 - 1:
        raise ValueError("上下文主键必须是有效的无符号BIGINT")


def _segment(row: ConversationSummary) -> dict:
    """脱离session导出已保存段。参数：row摘要行；返回：字段字典；异常：无。"""
    return {column.name: getattr(row, column.name) for column in ConversationSummary.__table__.columns}


async def _range(session, conversation_id: int, after: int, upto: int) -> list:
    """核验连续完整源区间。参数：session短事务、会话ID、开/闭上界；返回：原始轮；异常：ValueError或数据库错误。"""
    rows = (await session.scalars(select(Message).where(Message.conversation_id == conversation_id)
                                  .order_by(Message.id))).all()
    chosen = []
    for turn in stored_turns(rows):
        first, last = turn.messages[0].id, turn.messages[-1].id
        if last <= after or first > upto:
            continue
        groups = assemble_turns([turn])
        if first <= after or last > upto or not _safe_to_demote(turn, groups[0] if groups else []):
            raise ValueError("摘要边界必须覆盖完整已结束且工具配对的轮")
        chosen.append(turn)
    if not chosen or chosen[-1].messages[-1].id != upto:
        raise ValueError("摘要边界不是当前会话的完整轮末尾")
    return chosen


class ContextRepository:
    """独立session仓储；模型请求不在本类中执行，不持事务等模型。"""

    def __init__(self, factory, *, estimator: TokenEstimator | None = None):
        """参数：factory异步工厂、estimator统一口径；返回：None；异常：无。"""
        self.factory = factory
        self.estimator = estimator or TokenEstimator()

    async def load_context(self, conversation_id: int, *, include_turns: bool = True) -> dict:
        """读取同一数据库快照。参数：conversation_id可信主键、include_turns是否读原文；返回：锚点/段/可选完整轮；异常：ContextNotFound、数据库错误。"""
        _identifier(conversation_id)
        async with self.factory() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv is None:
                raise ContextNotFound()
            segments = (await session.scalars(select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id).order_by(ConversationSummary.seq))).all()
            rows = (await session.scalars(select(Message).where(Message.conversation_id == conversation_id)
                                          .order_by(Message.id))).all() if include_turns else None
            return {"conversation_id": conversation_id, "summary": conv.summary or "",
                    "summary_upto_msg_id": conv.summary_upto_msg_id or 0,
                    "layer1_from_msg_id": conv.layer1_from_msg_id or 0,
                    "summaries": [_segment(row) for row in segments], "turns": stored_turns(rows) if rows is not None else None}

    async def advance_layer1(self, conversation_id: int, *, expected_summary_upto: int, upto_msg_id: int) -> bool:
        """只推进原文层边界。参数：会话、读取时摘要上界、新边界；返回：是否已达成；异常：非法范围或数据库错误。"""
        _identifier(conversation_id)
        _identifier(expected_summary_upto, zero=True)
        _identifier(upto_msg_id)
        async with self.factory.begin() as session:
            conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id).with_for_update())
            if conv is None:
                raise ContextNotFound()
            if (conv.summary_upto_msg_id or 0) != expected_summary_upto:
                return False
            if upto_msg_id <= (conv.layer1_from_msg_id or 0):
                return True
            await _range(session, conversation_id, expected_summary_upto, upto_msg_id)
            conv.layer1_from_msg_id = upto_msg_id
        return True

    async def pending_summary_ids(self, *, after_id: int = 0, limit: int = 32) -> list[int]:
        """只读扫描存在未摘要层2的会话，是否超量交计量器判断。

        参数：after_id主键开游标、limit为1至1000；返回：升序ID；异常：参数或数据库错误。
        """
        _identifier(after_id, zero=True)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("恢复分页数量须为1至1000")
        async with self.factory() as session:
            rows = await session.execute(text("SELECT id FROM conversations WHERE id>:after_id "
                "AND COALESCE(layer1_from_msg_id,0)>COALESCE(summary_upto_msg_id,0) ORDER BY id LIMIT :limit"),
                {"after_id": after_id, "limit": limit})
            return list(rows.scalars())

    async def _find_segment(self, conversation_id: int, first: int, last: int) -> dict | None:
        """提交结果不确定时独立回读。参数：会话及冻结范围；返回：已提交段或None；异常：数据库读取错误。"""
        async with self.factory() as session:
            row = await session.scalar(select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id,
                ConversationSummary.from_msg_id == first, ConversationSummary.upto_msg_id == last))
            return _segment(row) if row else None

    async def append_summary(self, conversation_id: int, *, expected_summary_upto: int,
                             from_msg_id: int, upto_msg_id: int, content: str,
                             projection_max_tokens: int) -> dict | None:
        """追加一段并原子更新投影与覆盖锚点。

        参数：conversation_id可信会话；expected_summary_upto任务旧锚点；from/upto为冻结闭区间；
            content为已校验摘要（空串代表显式无事实）；projection_max_tokens投影容量。
        返回：已保存段或None（任务边界过期）；重复范围回读原段。
        异常：ValueError非法源区间/摘要；数据库提交失败且无法确认已保存时保留原异常。
        """
        for value in (conversation_id, from_msg_id, upto_msg_id):
            _identifier(value)
        _identifier(expected_summary_upto, zero=True)
        if (from_msg_id > upto_msg_id or type(projection_max_tokens) is not int or projection_max_tokens < 0
                or not isinstance(content, str) or len(content) > 200 or (content and not content.strip())):
            raise ValueError("摘要区间、正文或投影预算不合法")
        try:
            async with self.factory.begin() as session:
                conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id).with_for_update())
                if conv is None:
                    raise ContextNotFound()
                segments = (await session.scalars(select(ConversationSummary).where(
                    ConversationSummary.conversation_id == conversation_id).order_by(ConversationSummary.seq))).all()
                duplicate = next((row for row in segments if row.from_msg_id == from_msg_id and row.upto_msg_id == upto_msg_id), None)
                if duplicate:
                    return _segment(duplicate)
                if (conv.summary_upto_msg_id or 0) != expected_summary_upto:
                    return None
                if upto_msg_id > (conv.layer1_from_msg_id or 0):
                    raise ValueError("不能总结尚未降级的原文层")
                turns = await _range(session, conversation_id, expected_summary_upto, upto_msg_id)
                if turns[0].messages[0].id != from_msg_id:
                    raise ValueError("摘要不能跳过未处理的首轮")
                row = ConversationSummary(conversation_id=conversation_id, seq=segments[-1].seq + 1 if segments else 1,
                    from_msg_id=from_msg_id, upto_msg_id=upto_msg_id, content=content)
                session.add(row)
                await session.flush()
                # MySQL不通过RETURNING带回服务端时间；显式异步读取，禁止属性访问触发隐式IO。
                row = await session.scalar(select(ConversationSummary).where(ConversationSummary.id == row.id))
                conv.summary, _ = project_summaries([*[_segment(item) for item in segments], _segment(row)],
                                                    projection_max_tokens, self.estimator)
                conv.summary_upto_msg_id = upto_msg_id
                saved = _segment(row)
            return saved
        except Exception:
            # 只在新session能确认该冻结区间已落库时认成功；没有记录就保留原失败。
            try:
                saved = await self._find_segment(conversation_id, from_msg_id, upto_msg_id)
            except Exception:
                saved = None
            if saved is not None:
                return saved
            raise

    async def list_conversations(self, user_id: str, *, before_id: int | None = None, limit: int = 50) -> dict:
        """分页读取本人会话。参数：身份、before_id游标、limit为1至100；返回：列表/后续游标；异常：非法参数、数据库错误。"""
        if before_id is not None:
            _identifier(before_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("分页数量须为1至100")
        # 预览使用有序索引子查询，只读首条用户原问；摘要updated_at不参与排序。
        query = text("SELECT c.id, c.session_id, c.created_at, "
            "(SELECT m.content FROM messages m WHERE m.conversation_id=c.id AND m.role='user' ORDER BY m.id LIMIT 1) AS preview, "
            "EXISTS(SELECT 1 FROM conversation_summaries s WHERE s.conversation_id=c.id) AS has_summary "
            "FROM conversations c WHERE c.user_id=:user_id AND (:before_id IS NULL OR c.id<:before_id) "
            "ORDER BY c.id DESC LIMIT :page_size")
        async with self.factory() as session:
            rows = (await session.execute(query, {"user_id": user_id, "before_id": before_id, "page_size": limit + 1})).mappings().all()
        items = [{**dict(row), "preview": (row["preview"] or "")[:80], "has_summary": bool(row["has_summary"])} for row in rows[:limit]]
        return {"items": items, "next_before_id": items[-1]["id"] if len(rows) > limit else None}

    async def list_messages(self, user_id: str, conversation_id: int, *, after_id: int = 0, limit: int = 100) -> dict:
        """分页原文回载。参数：本人身份/会话、after_id开游标、limit为1至200；返回：原文/元数据/游标；异常：ContextNotFound或数据库错误。"""
        _identifier(conversation_id)
        _identifier(after_id, zero=True)
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("分页数量须为1至200")
        async with self.factory() as session:
            conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id))
            if conv is None:
                raise ContextNotFound()
            rows = (await session.scalars(select(Message).where(Message.conversation_id == conversation_id, Message.id > after_id)
                .order_by(Message.id).limit(limit + 1))).all()
            items = [{column.name: getattr(row, column.name) for column in Message.__table__.columns} for row in rows[:limit]]
            return {"items": items, "session_id": conv.session_id,
                    "next_after_id": items[-1]["id"] if len(rows) > limit else None}
