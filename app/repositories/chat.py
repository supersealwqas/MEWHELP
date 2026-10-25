"""会话与消息短事务存储，负责结果唯一性和启动时的中断恢复。"""

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from app.knowledge.answering import EvidenceCitation, FIXED_KNOWLEDGE_REFUSAL, SelfCheckFailure
from app.knowledge.sources import answer_metadata, verified_source_revision
from app.models.knowledge import KnowledgeChunk
from app.models.rag_quality import LowConfidenceQuestion
from app.models.customer_service import Conversation, Message
from app.schemas.history import StoredMessage, StoredTurn
from app.schemas.tools import ToolContext, ToolResult
from app.schemas.workflow import HistoryDelta, RetrievalLowConfidence, SuggestedAction


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
        turns: dict[str, StoredTurn] = {}
        for row in rows:
            turn = turns.setdefault(row.turn_id, StoredTurn(turn_id=row.turn_id, turn_status=row.turn_status))
            turn.messages.append(StoredMessage(role=row.role, content=row.content or "",
                                              tool_calls=row.tool_calls, tool_call_id=row.tool_call_id))
        return list(turns.values())

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
        turns: dict[str, StoredTurn] = {}
        for row in rows:
            turn = turns.setdefault(row.turn_id, StoredTurn(turn_id=row.turn_id, turn_status=row.turn_status))
            turn.messages.append(StoredMessage(role=row.role, content=row.content or "",
                                              tool_calls=row.tool_calls, tool_call_id=row.tool_call_id))
        return HistoryDelta(turns=list(turns.values()), last_message_id=last_id)

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

    async def finish_turn(self, context: ToolContext, content: str, *,
                          low_confidence: SelfCheckFailure | RetrievalLowConfidence | None = None,
                          citations: list[EvidenceCitation] | None = None,
                          suggested_actions: list[SuggestedAction] | None = None,
                          workflow_metadata: dict | None = None) -> dict:
        """原子保存最终回复、完成状态及可选 useful=false 问题池记录。

        参数：context可信身份/原话；content完整回复；low_confidence自评或检索弱证据；citations完整引用；
            suggested_actions服务端建议快照；workflow_metadata本轮流程摘要，两者均可省略以保留旧调用。
        返回：dict，提交成功后的完成元数据；同轮一致结果重试幂等，不重复原话入池。
        异常：ValueError，正文过长/身份或原话不符；PersistenceUncertain，已完成结果不同；数据库错误。
        """
        if isinstance(low_confidence, SelfCheckFailure) and not content.endswith(FIXED_KNOWLEDGE_REFUSAL):
            raise ValueError("自评不足入池必须与固定知识拒答一起保存")
        if isinstance(low_confidence, RetrievalLowConfidence):
            from app.workflow.routing import fixed_reply
            if content != fixed_reply("fallback")["answer"]:
                raise ValueError("检索弱证据入池必须与固定兜底一起保存")
        elif low_confidence is not None and not isinstance(low_confidence, SelfCheckFailure):
            raise ValueError("未知的低证据入口")
        if len(content.encode("utf-8")) > 65535:
            raise ValueError("回答正文过长，无法保存")
        citations = citations or []
        actions = [action.model_dump(mode="json") for action in (suggested_actions or [])]
        low_snapshot = low_confidence.model_dump(mode="json") if low_confidence else None
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
            if user.turn_status == "completed":
                existing = await session.scalar(select(Message).where(
                    Message.conversation_id == context.conversation_id, Message.turn_id == context.turn_id,
                    Message.role == "assistant", Message.tool_calls.is_(None)).order_by(Message.id.desc()).limit(1))
                saved = existing.rag_metadata if existing is not None else None
                # 重试核对模型原始证据；补齐来源不依赖可能已经更新的当前知识库。
                originals = [{key: value for key, value in item.model_dump().items() if value is not None} for item in citations]
                snapshots = saved.get("citations", []) if isinstance(saved, dict) else []
                same = len(originals) == len(snapshots) and all(
                    all(snapshot.get(key) == value for key, value in original.items())
                    for original, snapshot in zip(originals, snapshots))
                same_extra = isinstance(saved, dict) and saved.get("suggested_actions", []) == actions \
                    and saved.get("workflow") == workflow_metadata and saved.get("low_confidence") == low_snapshot
                if existing is None or existing.content != content or not same_extra or not same:
                    raise PersistenceUncertain("轮次已完成但结果不同，需核查")
                return existing.rag_metadata
            if user.turn_status != "running":
                raise PersistenceUncertain("轮次已结束，不能重新提交回复")
            metadata = answer_metadata(context, await self._complete_sources(session, citations))
            if suggested_actions is not None:
                metadata["suggested_actions"] = actions
            if workflow_metadata is not None:
                metadata["workflow"] = workflow_metadata
            if low_snapshot is not None:
                metadata["low_confidence"] = low_snapshot
            session.add(Message(conversation_id=context.conversation_id, turn_id=context.turn_id,
                                role="assistant", content=content, turn_status="completed", rag_metadata=metadata))
            if low_confidence is not None:
                session.add(LowConfidenceQuestion(conversation_id=context.conversation_id,
                    raw_question=context.user_message, source=low_confidence.source, reason=low_confidence.reason))
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
