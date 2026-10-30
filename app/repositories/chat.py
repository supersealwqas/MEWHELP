"""会话与消息短事务存储，负责结果唯一性和启动时的中断恢复。"""

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from uuid import NAMESPACE_URL, uuid5
from app.knowledge.answering import EvidenceCitation, FIXED_KNOWLEDGE_REFUSAL, SelfCheckFailure
from app.knowledge.sources import answer_metadata, verified_source_revision
from app.models.knowledge import KnowledgeChunk
from app.models.rag_quality import LowConfidenceQuestion
from app.models.customer_service import Conversation, Message
from app.schemas.history import StoredMessage, StoredTurn
from app.schemas.tools import ToolContext, ToolError, ToolResult
from app.schemas.workflow import HistoryDelta, RetrievalLowConfidence, SuggestedAction, OrderSelectionAction, RefundAction
from app.schemas.flywheel import FeedbackReceipt, PoolSignal, RetrievalSnapshot, TurnPoolState
from app.knowledge.snapshots import merge_snapshots, recover_legacy_snapshot


async def owned_source(session, session_id: str, user_id: str, turn_id: str | None = None, *, lock=False):
    """读取本人来源用户行，锁定时先定位主键避免范围间隙锁。

    参数：session短事务、session_id/user_id身份、turn_id可选原轮、lock是否行锁。
    返回：会话与用户消息二元组，缺失为(None,None)；异常：数据库错误。
    """
    conversation = await session.scalar(select(Conversation).where(Conversation.session_id == session_id, Conversation.user_id == user_id))
    if conversation is None:
        return None, None
    statement = select(Message.id).where(Message.conversation_id == conversation.id, Message.role == "user")
    if turn_id is not None:
        statement = statement.where(Message.turn_id == turn_id)
    else:
        # 独立工单确认轮不能替代最近问答的选单/退款来源。
        statement = statement.where(Message.rag_metadata["ticket_action"].as_string().is_(None))
    source_id = await session.scalar(statement.order_by(Message.id.desc()).limit(1))
    if source_id is None:
        return conversation, None
    query = select(Message).where(Message.id == source_id)
    return conversation, await session.scalar(query.with_for_update() if lock else query)


def source_context(conversation, source) -> ToolContext:
    """从不可变来源构造上下文。参数：conversation会话行、source用户行；返回：ToolContext；异常：字段校验错误。"""
    return ToolContext(conversation_id=conversation.id, session_id=conversation.session_id, user_id=conversation.user_id,
                       turn_id=source.turn_id, user_message=source.content)


async def validate_refund_evidence(session, context: ToolContext, action: RefundAction, workflow_metadata: dict | None) -> None:
    """核验退款建议必须来自本轮成功订单查询，不采信客户端或模型生成的快照。

    参数：session短事务、context可信原轮、action保存建议、workflow_metadata原轮摘要。
    返回：None；异常：ValueError，快照/意图/归属/查询事实不匹配；数据库错误。
    """
    from app.tools.business import owns_order
    expected = str(uuid5(NAMESPACE_URL, f"mewhelp:ch06:{context.conversation_id}:{context.turn_id}:submit_refund"))
    workflow_metadata = workflow_metadata or {}
    if (action.action_id != expected or action.source_turn_id != context.turn_id or action.question != context.user_message
            or not owns_order(context.user_id, action.order_id) or workflow_metadata.get("intent") != "退款退货"
            or workflow_metadata.get("route") != "order_policy" or workflow_metadata.get("order_id") != action.order_id):
        raise ValueError("退款动作与来源不一致")
    call_id = f"ch06:order:{context.turn_id}:{action.order_id}"
    result = await session.scalar(result_query(context.model_copy(update={"tool_call_id": call_id})))
    fact = ToolResult.model_validate_json(result.content) if result is not None else None
    calls = (await session.scalars(select(Message).where(Message.conversation_id == context.conversation_id,
        Message.turn_id == context.turn_id, Message.role == "assistant", Message.tool_calls.is_not(None)))).all()
    expected_call = {"id": call_id, "name": "query_order", "args": {"order_id": action.order_id}}
    if (fact is None or not fact.ok or not fact.data or fact.data.get("order_id") != action.order_id
            or not any(all(call.get(key) == value for key, value in expected_call.items())
                       and (not call.get("binding") or call["binding"].get("identity") == {
                           "source": "builtin", "server": None, "name": "query_order"})
                       for row in calls for call in row.tool_calls)):
        raise ValueError("退款动作缺少本轮已核实订单事实")


def result_query(context: ToolContext):
    """定位唯一工具结果。参数：context 调用身份；返回：Select；异常：无。"""
    return select(Message).where(Message.conversation_id == context.conversation_id,
                                 Message.turn_id == context.turn_id,
                                 Message.tool_call_id == context.tool_call_id, Message.role == "tool")


class PersistenceUncertain(RuntimeError):
    """数据库提交结果无法确认；不能当作业务失败重新执行。"""


class FeedbackNotFound(LookupError):
    """原轮不存在或不属于指定身份，对外统一404。"""


class FeedbackNotReady(ValueError):
    """原轮用户和最终助手尚未全部完成，暂不接受反馈。"""


async def _record_pool(session, source: Message, snapshot: RetrievalSnapshot | None,
                       signals: list[PoolSignal]) -> TurnPoolState:
    """在已锁原用户行的事务内统一落池，按用户→池→审核队列顺序加锁。

    参数：session 原事务；source 已核验并锁定的原问；snapshot 原轮事实；signals 固定来源理由。
    返回：本轮幂等状态；异常：ValueError，已保存落点损坏；数据库失败使外层事务一起回滚。
    """
    from app.models.flywheel import ReviewQueue
    metadata = dict(source.rag_metadata or {})
    state = TurnPoolState.model_validate(metadata.get("knowledge_pool") or {})
    rows = {}
    for identifier in sorted(set(state.pool_ids.values())):
        row = await session.scalar(select(LowConfidenceQuestion).where(LowConfidenceQuestion.id == identifier).with_for_update())
        if row is None or row.conversation_id != source.conversation_id or row.raw_question != source.content:
            raise ValueError("原轮问题池落点不一致")
        rows[identifier] = row
    matched = {row.matched_review_id for row in rows.values() if row.matched_review_id is not None}
    if state.matched_review_id is not None:
        matched.add(state.matched_review_id)
    if len(matched) > 1:
        raise ValueError("同轮问题池不能归并到多个审核问题")
    if matched:
        identifier = next(iter(matched))
        review = await session.scalar(select(ReviewQueue).where(ReviewQueue.id == identifier).with_for_update())
        if review is not None:
            state.matched_review_id = identifier
            if state.counted_review_id != identifier:
                review.occurrence_count += 1
                state.counted_review_id = identifier
    for signal in signals:
        if signal.source in state.pool_ids:
            continue
        row = LowConfidenceQuestion(conversation_id=source.conversation_id, raw_question=source.content,
            source=signal.source, reason=signal.reason,
            retrieved_chunks=snapshot.model_dump(mode="json") if snapshot is not None else None,
            matched_review_id=state.matched_review_id)
        session.add(row)
        await session.flush()
        state.pool_ids[signal.source] = row.id
    source.rag_metadata = {**metadata, "knowledge_pool": state.model_dump(mode="json")}
    return state


async def _original_snapshot(session, source: Message, *, current_citations: list[dict] | None = None) -> RetrievalSnapshot | None:
    """读取单个原轮事实。参数：事务/原用户行、current_citations仅当前完成时传真实引用；返回：快照或明确无检索；异常：数据库或契约错误。"""
    rows = (await session.scalars(select(Message).where(Message.conversation_id == source.conversation_id,
        Message.turn_id == source.turn_id).order_by(Message.id))).all()
    records = [{"turn_id": row.turn_id, "role": row.role, "content": row.content,
        "tool_calls": row.tool_calls, "tool_call_id": row.tool_call_id, "rag_metadata": row.rag_metadata} for row in rows]
    if current_citations is not None:
        known = ("retrieval_snapshot" in (source.rag_metadata or {}) or current_citations
            or any((row.rag_metadata or {}).get("citations") for row in rows)
            or any(call.get("name") in {"query_faq", "query_refund_policy"}
                for row in rows for call in row.tool_calls or [] if isinstance(call, dict)))
        if not known:
            return None
        if current_citations:
            records.append({"role": "assistant", "rag_metadata": {"citations": current_citations}})
    return recover_legacy_snapshot(records)


def stored_turns(rows: list[Message]) -> list[StoredTurn]:
    """把有序ORM流水转为完整轮。参数：rows按主键排序的本会话消息；返回：含真实ID的轮；异常：字段校验失败。"""
    turns: dict[str, StoredTurn] = {}
    for row in rows:
        turn = turns.setdefault(row.turn_id, StoredTurn(turn_id=row.turn_id, turn_status=row.turn_status))
        if row.role == "user" and isinstance(row.rag_metadata, dict):
            turn.order_selection = row.rag_metadata.get("order_selection")
            turn.ticket_intent = row.rag_metadata.get("ticket_intent")
            turn.ticket_confirmation = row.rag_metadata.get("pending_ticket")
            # 独立动作在恢复后可能标cancelled/failed，但成功回执仍可凭原键补写。
            turn.receipt_pending = row.rag_metadata.get("ticket_action") is not None and row.turn_status != "completed"
            if isinstance(turn.ticket_confirmation, dict):
                status = turn.ticket_confirmation.get("status")
                if status in {"awaiting_input", "accepted", "unknown"} or status == "completed" and row.turn_status != "completed":
                    turn.receipt_pending = True
        turn.messages.append(StoredMessage(id=row.id, role=row.role, content=row.content or "",
                                          tool_calls=row.tool_calls, tool_call_id=row.tool_call_id))
    for turn in turns.values():
        if turn.receipt_pending and turn.ticket_confirmation is None:
            results = [row for row in turn.messages if row.role == "tool"]
            try:
                # 已确认的工具失败不会生成成功回执；未知或损坏结果仍按可能补写保守处理。
                if len(results) == 1 and not ToolResult.model_validate_json(results[0].content).ok:
                    turn.receipt_pending = False
            except ValueError:
                pass
    return list(turns.values())


class ChatRepository:
    """每次操作独立会话，不把数据库事务带入模型网络等待。"""

    def __init__(self, session_factory):
        """保存会话工厂。参数：session_factory 异步工厂；返回：None；异常：无。"""
        self.factory = session_factory

    async def _complete_sources(self, session, citations: list[EvidenceCitation]) -> list[EvidenceCitation]:
        """从权威仓储补齐缺失来源，仅完整正文一致的主键可匹配。

        参数：session 当前短事务；citations 模型已经看到的证据。
        返回：新引用列表，正文不变；无法核对来源或旧版本时保留空字段。
        异常：数据库读取错误透传，使回答不能假装成功提交。
        """
        missing = [item.chunk_id for item in citations if item.source_key is None]
        if not missing:
            return citations
        columns = (KnowledgeChunk.id, KnowledgeChunk.source_key, KnowledgeChunk.ordinal, KnowledgeChunk.category,
                   KnowledgeChunk.questions, KnowledgeChunk.answer, KnowledgeChunk.section_path)
        rows = (await session.execute(select(*columns).where(KnowledgeChunk.id.in_(missing)))).all()
        by_id = {row.id: row for row in rows}
        revisions = {}
        completed = []
        for item in citations:
            row = by_id.get(item.chunk_id)
            if item.source_key is None and row is not None and (row.questions, row.answer, row.section_path) == (item.question, item.answer, item.section_path):
                if row.source_key.startswith("doc:") and row.source_key not in revisions:
                    source_rows = (await session.execute(select(*columns).where(
                        KnowledgeChunk.source_key == row.source_key, KnowledgeChunk.is_active == True))).all()
                    revisions[row.source_key] = verified_source_revision(row.source_key, source_rows)
                item = item.model_copy(update={"source_key": row.source_key, "source_revision": revisions.get(row.source_key)})
            completed.append(item)
        return completed

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
        return stored_turns(rows)

    async def load_history_delta(self, conversation_id: int, after_message_id: int, *,
                                 refresh_turn_id: str | None = None, exclude_turn_id: str | None = None) -> HistoryDelta:
        """只按新主键定位轮次，再读取这些轮的完整消息；可刷新无新消息的中断终态。

        参数：conversation_id可信会话；after_message_id非负游标；refresh_turn_id旧未完成轮；exclude_turn_id当前轮。
        返回：HistoryDelta，完整受影响轮和单调游标；零游标只用于首次导入。
        异常：ValueError，负游标；数据库读取错误透传。
        """
        if after_message_id < 0:
            raise ValueError("历史游标不能为负")
        async with self.factory() as session:
            changed = (await session.execute(select(Message.id, Message.turn_id).where(
                Message.conversation_id == conversation_id, Message.id > after_message_id).order_by(Message.id))).all()
            last_id = max((row.id for row in changed), default=after_message_id)
            affected = {row.turn_id for row in changed}
            if refresh_turn_id:
                affected.add(refresh_turn_id)
            affected.discard(exclude_turn_id)
            if not affected:
                return HistoryDelta(turns=[], last_message_id=last_id)
            rows = (await session.scalars(select(Message).where(Message.conversation_id == conversation_id,
                Message.turn_id.in_(affected)).order_by(Message.id))).all()
        return HistoryDelta(turns=stored_turns(rows), last_message_id=last_id)

    async def get_turn_context(self, session_id: str, user_id: str, turn_id: str) -> ToolContext | None:
        """读取本人原轮。参数：会话/用户/原轮标识；返回：可信上下文或None；异常：数据库错误。"""
        async with self.factory() as session:
            conversation, source = await owned_source(session, session_id, user_id, turn_id)
            return source_context(conversation, source) if source is not None else None

    async def get_workflow_turn(self, session_id: str, user_id: str, turn_id: str | None = None) -> dict | None:
        """只读返回原问状态和完成快照。参数：身份及可选原轮；返回：内部可信字典或None；异常：ValueError跨身份、数据库错误。"""
        async with self.factory() as session:
            conversation, source = await owned_source(session, session_id, user_id, turn_id)
            if conversation is None:
                if await session.scalar(select(Conversation.id).where(Conversation.session_id == session_id)) is not None:
                    raise ValueError("会话身份不匹配")
                return None
            if source is None:
                return None
            answer = await session.scalar(select(Message).where(Message.conversation_id == conversation.id,
                Message.turn_id == source.turn_id, Message.role == "assistant", Message.tool_calls.is_(None),
                Message.turn_status == "completed").order_by(Message.id.desc()).limit(1))
            return {"context": source_context(conversation, source), "turn_status": source.turn_status,
                    "trace_identity": (source.rag_metadata or {}).get("trace_identity"),
                    "retrieval_snapshot": (source.rag_metadata or {}).get("retrieval_snapshot"),
                    "pool_signals": (source.rag_metadata or {}).get("completion_inputs", {}).get("pool_signals", []),
                    "order_selection": (source.rag_metadata or {}).get("order_selection"),
                    "pending_ticket": (source.rag_metadata or {}).get("pending_ticket"),
                    "ticket_intent": (source.rag_metadata or {}).get("ticket_intent"),
                    "ticket_write_state": (source.rag_metadata or {}).get("ticket_write_state"),
                    "answer": answer.content if answer is not None else None,
                    "completion_metadata": answer.rag_metadata if answer is not None else None}

    async def save_retrieval_snapshot(self, context: ToolContext, snapshot: RetrievalSnapshot) -> None:
        """及时合并原轮快照，整份替换JSON并保留追踪/工单字段。

        参数：context 可信原轮原问；snapshot 检索核心收集的快照。
        返回：None；异常：ValueError，身份或同尝试事实冲突；数据库错误。
        """
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id, lock=True)
            if source is None or source_context(conversation, source) != context.model_copy(update={"tool_call_id": ""}):
                raise ValueError("检索快照缺少匹配的原始用户消息")
            metadata = dict(source.rag_metadata or {})
            saved = RetrievalSnapshot.model_validate(metadata["retrieval_snapshot"]) if metadata.get("retrieval_snapshot") else None
            merged = merge_snapshots(saved, snapshot)
            if source.turn_status == "completed" and merged != saved:
                raise ValueError("已完成原轮不能补写新的检索事实")
            source.rag_metadata = {**metadata, "retrieval_snapshot": merged.model_dump(mode="json")}

    async def record_unresolved_feedback(self, session_id: str, user_id: str, turn_id: str) -> FeedbackReceipt:
        """按服务端原轮原问和快照落池，重发复用已提交回执。

        参数：session_id/user_id 演示归属、turn_id 明确原轮；返回：字符串主键回执。
        异常：FeedbackNotFound 越权/未知；FeedbackNotReady 未完成；数据库错误。
        """
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, session_id, user_id, turn_id, lock=True)
            if source is None:
                raise FeedbackNotFound()
            answer = await session.scalar(select(Message).where(Message.conversation_id == conversation.id,
                Message.turn_id == turn_id, Message.role == "assistant", Message.tool_calls.is_(None))
                .order_by(Message.id.desc()).limit(1))
            if source.turn_status != "completed" or answer is None or answer.turn_status != "completed":
                raise FeedbackNotReady("原轮尚未完成")
            snapshot = await _original_snapshot(session, source)
            source.rag_metadata = {**(source.rag_metadata or {}),
                "retrieval_snapshot": snapshot.model_dump(mode="json") if snapshot is not None else None}
            state = await _record_pool(session, source, snapshot, [PoolSignal(source="user_feedback", reason="用户反馈未解决")])
            return FeedbackReceipt(pool_id=str(state.pool_ids["user_feedback"]),
                matched_review_id=str(state.matched_review_id) if state.matched_review_id is not None else None)

    async def save_trace_identity(self, context: ToolContext, identity: dict) -> dict:
        """在首次模型调用前锁原用户行保存追踪根，重试复用已有根。

        参数：context 可信原轮及原问，identity 仅含字符串身份与根 ID 的 JSON 字典。
        返回：dict，已经保存的原轮身份；异常：ValueError，来源/字段不符；数据库错误原样传播。
        """
        import re
        expected = {"trace_id", "session_id", "user_id", "turn_id", "root_observation_id"}
        if (set(identity) != expected or any(not isinstance(value, str) for value in identity.values())
                or not re.fullmatch(r"[0-9a-f]{32}", identity["trace_id"])
                or not re.fullmatch(r"[0-9a-f]{16}", identity["root_observation_id"])
                or any(identity[key] != getattr(context, key) for key in ("session_id", "user_id", "turn_id"))):
            raise ValueError("追踪身份必须属于当前原轮")
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id, lock=True)
            if source is None or source_context(conversation, source) != context.model_copy(update={"tool_call_id": ""}):
                raise ValueError("追踪身份缺少匹配的原始用户消息")
            metadata = dict(source.rag_metadata or {})
            if metadata.get("trace_identity") is not None:
                saved = metadata["trace_identity"]
                if any(saved.get(key) != identity[key] for key in expected - {"root_observation_id"}):
                    raise ValueError("原轮追踪身份不能被替换")
                return dict(saved)
            # JSON整体赋值使SQLAlchemy可靠标记脏字段，其他原轮数据保持完整。
            source.rag_metadata = {**metadata, "trace_identity": dict(identity)}
            return dict(identity)

    async def save_ticket_intent(self, context: ToolContext, intent) -> None:
        """把当前诉求保存到原用户行，不修改原始聊天正文。

        参数：context 可信原轮，intent 为有界意图或 None；返回：None；异常：PersistenceUncertain，来源不符；数据库错误。
        """
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id, lock=True)
            if source is None or source_context(conversation, source) != context.model_copy(update={"tool_call_id": ""}):
                raise PersistenceUncertain("建单诉求缺少可信原用户消息")
            metadata = dict(source.rag_metadata or {})
            # 有预览后其意图由确认流程更新，节点重放不得把它倒退为 collecting。
            if metadata.get("pending_ticket") is None:
                metadata["ticket_intent"] = intent.model_dump(mode="json") if intent else None
                source.rag_metadata = metadata

    async def get_pending_ticket(self, context: ToolContext) -> dict | None:
        """回读当前原轮唯一冻结流程，带调用标识时不接受串用其他申请。

        参数：context 可信会话和可选原调用标识；返回：冻结流程或 None；异常：PersistenceUncertain，身份不符；数据库错误。
        """
        async with self.factory() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id)
            if source is None or source_context(conversation, source) != context.model_copy(update={"tool_call_id": ""}):
                raise PersistenceUncertain("工单流程来源不匹配")
            pending = (source.rag_metadata or {}).get("pending_ticket")
            if pending is not None and context.tool_call_id and pending["action"]["tool_call_id"] != context.tool_call_id:
                return None
            return pending

    async def prepare_ticket_preview(self, context: ToolContext, binding, action, intent) -> dict:
        """在原用户行锁内冻结唯一预览和来源绑定，再结束事务等待用户。

        参数：context 原调用、binding 决定快照、action 稳定预览、intent 本诉求原话；返回：原冻结流程；
        异常：WriteAuthorizationRejected，意图/描述/原流程不符；PersistenceUncertain，来源损坏；数据库错误。
        """
        from app.tools.confirmation import ticket_description, ticket_preview
        from app.tools.contracts import WriteAuthorizationRejected, binding_dict, json_hash
        arguments = {"description": action.description, "ticket_type": action.ticket_type}
        if (binding.identity.source != "builtin" or binding.identity.server is not None
                or binding.identity.name != "create_ticket" or action != ticket_preview(context, arguments)
                or ticket_description(intent) != action.description):
            raise WriteAuthorizationRejected(ToolResult(ok=False, error=ToolError(code="NOT_CONFIRMED", message="建单申请与可信原话不匹配", category="permission_denied")))
        payload = {"action": action.model_dump(mode="json"), "binding": binding_dict(binding),
            "arguments_hash": json_hash(arguments), "status": "awaiting_input", "decision": None, "execution_started": False}
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id, lock=True)
            if source is None or source_context(conversation, source) != context.model_copy(update={"tool_call_id": ""}):
                raise PersistenceUncertain("工单预览缺少可信原轮")
            metadata = dict(source.rag_metadata or {})
            saved = metadata.get("pending_ticket")
            if saved is not None:
                if any(saved.get(key) != payload[key] for key in ("action", "binding", "arguments_hash")):
                    raise WriteAuthorizationRejected(ToolResult(ok=False, error=ToolError(code="TICKET_FLOW_USED", message="本轮已有工单流程，不能再次申请", category="permission_denied")))
                return saved
            if metadata.get("ticket_intent") != intent.model_dump(mode="json") or source.turn_status != "running":
                raise WriteAuthorizationRejected(ToolResult(ok=False, error=ToolError(code="NOT_CONFIRMED", message="原建单诉求已改变，本次没有执行", category="permission_denied")))
            calls = (await session.scalars(select(Message).where(Message.conversation_id == context.conversation_id,
                Message.turn_id == context.turn_id, Message.tool_calls.is_not(None)))).all()
            expected_call = {"id": context.tool_call_id, "name": "create_ticket", "args": arguments, "binding": binding_dict(binding)}
            if not any(expected_call in row.tool_calls for row in calls):
                raise PersistenceUncertain("工单预览没有对应的完整原申请")
            metadata.update(pending_ticket=payload, ticket_intent=intent.model_copy(update={"status": "preview"}).model_dump(mode="json"),
                ticket_write_state={"action_id": action.action_id, "tool_call_id": context.tool_call_id, "status": "awaiting_input"})
            source.rag_metadata = metadata
            await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                Message.turn_id == context.turn_id).values(turn_status="awaiting_input"))
            return payload

    async def save_ticket_wait(self, context: ToolContext, action) -> None:
        """核验图中断所展示的是已冻结卡片，不覆盖已接受的决定。

        参数：context 原轮，action 官方 interrupt 的卡片；返回：None；异常：PersistenceUncertain，快照不一致；数据库错误。
        """
        pending = await self.get_pending_ticket(context)
        if pending is None or pending.get("action") != action.model_dump(mode="json"):
            raise PersistenceUncertain("工单中断与已保存预览不一致")

    async def mark_ticket_execution(self, context: ToolContext, action_id: str) -> None:
        """确认后标记唯一执行边界，无确定结果的重入不得再次产生副作用。

        参数：context 原调用，action_id 已核对动作；返回：None；异常：PersistenceUncertain，来源/决定异常或已经开始；数据库错误。
        """
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id, lock=True)
            saved = (source.rag_metadata or {}).get("pending_ticket") if source is not None else None
            if (source is None or source_context(conversation, source) != context.model_copy(update={"tool_call_id": ""})
                    or not saved or saved["action"]["action_id"] != action_id
                    or saved["action"]["tool_call_id"] != context.tool_call_id
                    or saved.get("decision") != "confirm" or saved.get("status") != "accepted"
                    or saved.get("execution_started")):
                raise PersistenceUncertain("工单执行边界已经开始或无法核实")
            source.rag_metadata = {**source.rag_metadata, "pending_ticket": {**saved, "execution_started": True}}

    async def finish_ticket_preview(self, context: ToolContext, result: ToolResult | None) -> dict | None:
        """记录原冻结流程终态，保留首次决定和原参数供恢复核对。

        参数：context 原调用，result 确定工具结果或 None 表示提交未知；返回：最新流程或 None；异常：数据库错误。
        """
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id, lock=True)
            if source is None or source_context(conversation, source) != context.model_copy(update={"tool_call_id": ""}):
                raise PersistenceUncertain("工单收尾来源不匹配")
            metadata = dict(source.rag_metadata or {})
            saved = metadata.get("pending_ticket")
            if saved is None or saved["action"]["tool_call_id"] != context.tool_call_id:
                return saved
            # 写处理器开始后的故障不能证明没有业务副作用，保留未知供原键核实。
            unknown = result is None or bool(saved.get("execution_started") and not result.ok
                                            and result.error and result.error.category == "fault")
            status = "unknown" if unknown else "completed" if result.ok else "cancelled"
            saved = {**saved, "status": status}
            intent = dict(metadata.get("ticket_intent") or {})
            if intent:
                intent["status"] = "unknown" if unknown else "confirmed" if result.ok else "cancelled"
            source.rag_metadata = {**metadata, "pending_ticket": saved, "ticket_intent": intent or None,
                "ticket_write_state": {"action_id": saved["action"]["action_id"], "tool_call_id": context.tool_call_id, "status": status}}
            if result is not None:
                # 无效恢复也已有确定拒绝结果；结束等待后允许保存原轮最终答复。
                await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                    Message.turn_id == context.turn_id, Message.turn_status == "awaiting_input").values(turn_status="running"))
            return saved

    async def save_order_wait(self, context: ToolContext, action: OrderSelectionAction) -> None:
        """将匹配的原轮保存为等待，绝不倒退已接受或完成状态。

        参数：context可信原轮；action官方检查点选单快照。
        返回：None；同等待快照重复保存不新增。异常：PersistenceUncertain，身份/快照/状态冲突；数据库错误。
        """
        from app.tools.business import list_user_orders
        expected_id = str(uuid5(NAMESPACE_URL, f"mewhelp:ch06:{context.conversation_id}:{context.turn_id}:select_order"))
        payload = action.model_dump(mode="json")
        if action.source_turn_id != context.turn_id or action.action_id != expected_id or payload["orders"] != list_user_orders(context.user_id):
            raise PersistenceUncertain("等待动作与可信来源不一致")
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, context.session_id, context.user_id, context.turn_id, lock=True)
            if source is None or source_context(conversation, source) != context:
                raise PersistenceUncertain("等待原轮不存在")
            metadata = dict(source.rag_metadata or {})
            saved = metadata.get("order_selection")
            waiting = {**payload, "status": "awaiting_input", "accepted_order_id": None}
            if source.turn_status not in {"running", "awaiting_input"} or (saved is not None and saved != waiting):
                raise PersistenceUncertain("旧等待不能覆盖已改变的动作")
            metadata["order_selection"] = waiting
            source.rag_metadata = metadata
            await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                Message.turn_id == context.turn_id).values(turn_status="awaiting_input"))

    async def cancel_order_wait(self, session_id: str, user_id: str) -> None:
        """新问在持有会话锁时取消尚未接受的旧等待。参数：会话与用户；返回：None；异常：数据库错误。"""
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, session_id, user_id, lock=True)
            if source is None or source.turn_status != "awaiting_input":
                return
            metadata = dict(source.rag_metadata or {})
            saved = metadata.get("order_selection")
            if not isinstance(saved, dict) or saved.get("status") != "awaiting_input":
                raise PersistenceUncertain("等待状态与动作不一致")
            metadata["order_selection"] = {**saved, "status": "cancelled"}
            source.rag_metadata = metadata
            await session.execute(update(Message).where(Message.conversation_id == conversation.id,
                Message.turn_id == source.turn_id, Message.turn_status == "awaiting_input").values(turn_status="cancelled"))

    async def save_tool_calls(self, context: ToolContext, calls: list[dict], *, deduplicate: bool = False,
                              bindings: dict | None = None) -> None:
        """执行前保存完整申请，确定性节点可按键防重。

        参数：context可信上下文；calls完整申请；deduplicate是否按ID校验；bindings决定时可信来源快照。
        返回：None；相同完整申请不新增。
        异常：PersistenceUncertain，缺来源或相同ID内容冲突；数据库写入错误。
        """
        calls = [{**call, **({"binding": bindings[call["id"]]} if bindings and call.get("id") in bindings else {})}
                 for call in calls]
        async with self.factory.begin() as session:
            if deduplicate:
                source = await session.scalar(select(Message).where(Message.conversation_id == context.conversation_id,
                    Message.turn_id == context.turn_id, Message.role == "user").order_by(Message.id).limit(1).with_for_update())
                if source is None:
                    raise PersistenceUncertain("确定性查询缺少来源轮次")
                rows = (await session.scalars(select(Message).where(Message.conversation_id == context.conversation_id,
                    Message.turn_id == context.turn_id, Message.tool_calls.is_not(None)))).all()
                existing = {call["id"]: call for row in rows for call in row.tool_calls if "id" in call}
                pending = []
                for call in calls:
                    if call["id"] in existing and existing[call["id"]] != call:
                        raise PersistenceUncertain("相同查询调用键的内容不一致")
                    if call["id"] not in existing:
                        pending.append(call)
                        existing[call["id"]] = call
                calls = pending
                if not calls:
                    return
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

    async def finish_turn(self, context: ToolContext, content: str, *,
                          low_confidence: SelfCheckFailure | RetrievalLowConfidence | None = None,
                          citations: list[EvidenceCitation] | None = None,
                          suggested_actions: list[SuggestedAction] | None = None,
                          workflow_metadata: dict | None = None, refund_action: RefundAction | None = None,
                          retrieval_snapshot: RetrievalSnapshot | None = None,
                          pool_signals: list[PoolSignal] | None = None) -> dict:
        """原子保存最终回复、完成状态、完整原始引用及可选低证据问题池记录。

        参数：context可信身份/原话；content完整回复；low_confidence自评或检索弱证据；citations完整引用；
            suggested_actions服务端工单建议；workflow_metadata本轮流程摘要；refund_action独立本地申请快照；
            retrieval_snapshot本轮闸前快照；pool_signals统一入池信号，不能与low_confidence同时传入，均可省略。
        返回：dict，提交成功后的完成元数据；同轮一致结果重试幂等，不重复原话入池。
        异常：ValueError，正文过长/身份或原话不符；PersistenceUncertain，已完成结果不同；数据库错误。
        """
        if low_confidence is not None and pool_signals is not None:
            raise ValueError("旧低证据参数与统一入池信号不能同时使用")
        signals = [PoolSignal.model_validate(item) for item in (pool_signals or [])]
        if any(signal.source == "user_feedback" for signal in signals):
            raise ValueError("用户反馈只能从已完成原轮反馈入口提交")
        if len({signal.source for signal in signals}) != len(signals):
            raise ValueError("同轮入池来源不能重复")
        if (isinstance(low_confidence, SelfCheckFailure) or any(s.source == "self_check" for s in signals)) and not content.endswith(FIXED_KNOWLEDGE_REFUSAL):
            raise ValueError("自评不足入池必须与固定知识拒答一起保存")
        if low_confidence is not None and not isinstance(low_confidence, (SelfCheckFailure, RetrievalLowConfidence)):
            raise ValueError("未知的低证据入口")
        if len(content.encode("utf-8")) > 65535:
            raise ValueError("回答正文过长，无法保存")
        citations = citations or []
        originals = [item.model_dump(mode="json") for item in citations]
        actions = [action.model_dump(mode="json") for action in (suggested_actions or [])]
        low_snapshot = low_confidence.model_dump(mode="json") if low_confidence else None
        if low_confidence is not None:
            signals = [PoolSignal(source=low_confidence.source, reason=low_confidence.reason)]
        snapshot_input = retrieval_snapshot.model_dump(mode="json") if retrieval_snapshot is not None else None
        completion_inputs = {"retrieval_snapshot": snapshot_input,
            "pool_signals": [signal.model_dump(mode="json") for signal in signals]}
        refund_snapshot = refund_action.model_dump(mode="json") if refund_action else None
        async with self.factory.begin() as session:
            # 锁定同一轮最早用户消息，使并发完成互斥，无须给用户DDL增加turn_id字段。
            user = await session.scalar(select(Message).where(
                Message.conversation_id == context.conversation_id, Message.turn_id == context.turn_id,
                Message.role == "user").order_by(Message.id).limit(1).with_for_update())
            if user is None or user.content != context.user_message:
                raise ValueError("轮次原话与可信上下文不匹配")
            conversation = await session.get(Conversation, context.conversation_id)
            if conversation is None or conversation.user_id != context.user_id or conversation.session_id != context.session_id:
                raise ValueError("完成轮次会话身份不匹配")
            if any(signal.source == "retrieval_low_conf" for signal in signals):
                from app.repositories.actions import ActionRepository
                from app.schemas.workflow import TicketResume
                from app.workflow.routing import fixed_reply
                fallback = fixed_reply("fallback")["answer"]
                if content != fallback:
                    pending = (user.rag_metadata or {}).get("pending_ticket")
                    if (not isinstance(pending, dict) or pending.get("decision") != "confirm"
                            or not isinstance(pending.get("action"), dict)):
                        raise ValueError("检索弱证据入池必须与固定兜底或本轮真实回执一起保存")
                    request = TicketResume(type="confirm_ticket", session_id=context.session_id,
                        user_id=context.user_id, source_turn_id=context.turn_id,
                        action_id=pending["action"].get("action_id"), decision="confirm")
                    # 复用已锁定用户行的事务，不能另开会话；正文必须精确等于可信前缀与固定答复。
                    receipt = await ActionRepository(self.factory)._chat_ticket_receipt(session, request, lock=True)
                    if receipt is None or content != receipt.answer_prefix() + fallback:
                        raise ValueError("检索弱证据入池必须与固定兜底或本轮真实回执一起保存")
            if user.turn_status == "completed":
                existing = await session.scalar(select(Message).where(
                    Message.conversation_id == context.conversation_id, Message.turn_id == context.turn_id,
                    Message.role == "assistant", Message.tool_calls.is_(None)).order_by(Message.id.desc()).limit(1))
                saved = existing.rag_metadata if existing is not None else None
                if isinstance(saved, dict) and "citation_inputs" in saved:
                    # 原始输入独立于来源补全快照，None也是值，不能把字段删除当成一致重试。
                    same = saved["citation_inputs"] == originals
                else:
                    # 旧记录未保存原始输入，仅来源键/版本可能经过补全；章节等字段仍完整比较。
                    snapshots = saved.get("citations", []) if isinstance(saved, dict) else []
                    same = len(originals) == len(snapshots) and all(
                        all(snapshot.get(key) == value for key, value in original.items()
                            if value is not None or key not in {"source_key", "source_revision"})
                        for original, snapshot in zip(originals, snapshots))
                same_extra = isinstance(saved, dict) and saved.get("suggested_actions", []) == actions \
                    and saved.get("workflow") == workflow_metadata and saved.get("low_confidence") == low_snapshot \
                    and saved.get("refund_action") == refund_snapshot
                stored_inputs = (user.rag_metadata or {}).get("completion_inputs")
                same_extra = same_extra and (stored_inputs == completion_inputs if stored_inputs is not None
                    else retrieval_snapshot is None and pool_signals is None)
                if existing is None or existing.content != content or not same_extra or not same:
                    raise PersistenceUncertain("轮次已完成但结果不同，需核查")
                return existing.rag_metadata
            if user.turn_status != "running":
                raise PersistenceUncertain("轮次已结束，不能重新提交回复")
            if refund_action is not None:
                await validate_refund_evidence(session, context, refund_action, workflow_metadata)
            metadata = answer_metadata(context, await self._complete_sources(session, citations))
            metadata["citation_inputs"] = originals
            if suggested_actions is not None:
                metadata["suggested_actions"] = actions
            if workflow_metadata is not None:
                metadata["workflow"] = workflow_metadata
            if refund_action is not None:
                metadata["refund_action"] = refund_snapshot
            if low_snapshot is not None:
                metadata["low_confidence"] = low_snapshot
            saved_snapshot = await _original_snapshot(session, user, current_citations=originals if not signals else None)
            if retrieval_snapshot is not None:
                # 原轮先前实时保存的事实优先；无历史数据的标记不参与本轮新检索合并。
                prior = (user.rag_metadata or {}).get("retrieval_snapshot")
                saved_snapshot = merge_snapshots(RetrievalSnapshot.model_validate(prior) if prior else None, retrieval_snapshot)
            user.rag_metadata = {**(user.rag_metadata or {}), "completion_inputs": completion_inputs,
                "retrieval_snapshot": saved_snapshot.model_dump(mode="json") if saved_snapshot is not None else None}
            session.add(Message(conversation_id=context.conversation_id, turn_id=context.turn_id,
                                role="assistant", content=content, turn_status="completed", rag_metadata=metadata))
            if signals:
                await _record_pool(session, user, saved_snapshot, signals)
            await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                                                        Message.turn_id == context.turn_id).values(turn_status="completed"))
        return metadata

    async def get_citation(self, session_id: str, user_id: str, turn_id: str, n: int) -> dict | None:
        """只读取同一归属、已完成轮次中的保存快照，不查询当前知识替换旧证据。

        参数：session_id 客户端会话；user_id 演示归属；turn_id 服务端轮次；n 正引用编号。
        返回：dict或None，未知、跨会话用户、未完成与非法编号均返回None。
        异常：数据库读取错误透传。
        """
        if n < 1:
            return None
        async with self.factory() as session:
            row = await session.scalar(select(Message).join(Conversation, Message.conversation_id == Conversation.id).where(
                Conversation.session_id == session_id, Conversation.user_id == user_id,
                Message.turn_id == turn_id, Message.role == "assistant", Message.turn_status == "completed",
                Message.tool_calls.is_(None)).order_by(Message.id.desc()).limit(1))
            if row is None or not isinstance(row.rag_metadata, dict) or row.rag_metadata.get("status") != "completed":
                return None
            return next((item for item in row.rag_metadata.get("citations", []) if item.get("n") == n), None)

    async def list_pending_workflow_turns(self) -> list[dict]:
        """列出需启动对账的可信原轮，完成记录天然保护。

        参数：无；返回：包含context、turn_status、order_selection的内部列表。
        异常：数据库/字段校验错误，不能吞为空列表。
        """
        async with self.factory() as session:
            rows = (await session.execute(select(Conversation, Message).join(Message, Conversation.id == Message.conversation_id)
                .where(Message.role == "user", Message.turn_status.in_(["running", "awaiting_input"]))
                .order_by(Message.id))).all()
            return [{"context": source_context(conversation, source), "turn_status": source.turn_status,
                     "order_selection": (source.rag_metadata or {}).get("order_selection"),
                     "pending_ticket": (source.rag_metadata or {}).get("pending_ticket")}
                    for conversation, source in rows]

    async def mark_turn(self, context: ToolContext, status: str, *, include_waiting: bool = False) -> None:
        """标记失败或中断并保护已提交终态。参数：上下文与 failed/cancelled；返回：None；异常：状态或数据库错误。"""
        if status not in {"failed", "cancelled"}:
            raise ValueError("非法终止状态")
        async with self.factory.begin() as session:
            await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                                                        Message.turn_id == context.turn_id,
                                                        Message.turn_status.in_(["running", "awaiting_input"] if include_waiting else ["running"])).values(turn_status=status))

    async def recover_interrupted_turns(self) -> int:
        """单实例启动前恢复遗留轮次。参数：无；返回：恢复轮数；异常：数据库错误，应终止启动。"""
        async with self.factory.begin() as session:
            turns = (await session.execute(select(Message.conversation_id, Message.turn_id).where(
                Message.turn_status == "running").distinct())).all()
            await session.execute(update(Message).where(Message.turn_status == "running").values(turn_status="cancelled"))
        return len(turns)
