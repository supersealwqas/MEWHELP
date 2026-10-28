"""确认建单的可信边界：锁定已完成建议、准备稳定动作轮并按真实票据幂等收尾。"""

from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError
from sqlalchemy import select, update

from app.models.customer_service import Conversation, Message, Ticket, RefundApplication
from app.repositories.chat import PersistenceUncertain, owned_source, source_context, validate_refund_evidence
from app.repositories.support import ticket_number
from app.schemas.tools import TicketArgs, ToolContext, ToolResult
from app.schemas.workflow import (PreparedTicketAction, SuggestedAction, TicketConfirmation, TicketReceipt,
                                 OrderSelection, OrderSelectionAction, AcceptedOrderSelection, OrderSelectionConflict)
from app.schemas.workflow import RefundAction, RefundSubmission, RefundReceipt


class ActionNotFound(ValueError):
    """动作不存在、来源未完成或身份不符；统一对外返回不存在以免泄漏归属。"""


class InvalidTicketArguments(ValueError):
    """保存原问不能作为工单参数；与数据库中的损坏结果严格区分。"""


class ActionRepository:
    """只处理用户确认后的独立建单动作，不重开或修改来源问答。"""

    def __init__(self, session_factory):
        """保存短事务工厂。参数：session_factory异步会话工厂；返回：None；异常：无。"""
        self.factory = session_factory

    async def accept_order_selection(self, request: OrderSelection) -> AcceptedOrderSelection:
        """来源行锁内接受原选单，仅首次选择可改变状态。

        参数：request严格身份/原轮/动作/订单请求；返回：已保存的原轮选单结果。
        异常：ActionNotFound，身份/动作/归属不符；OrderSelectionConflict，旧卡或同键换单；数据库错误。
        """
        from app.tools.business import owns_order
        async with self.factory.begin() as session:
            conversation, source = await owned_source(session, request.session_id, request.user_id, request.source_turn_id, lock=True)
            saved = (source.rag_metadata or {}).get("order_selection") if source is not None else None
            if not isinstance(saved, dict) or saved.get("action_id") != request.action_id:
                raise ActionNotFound("选单不存在")
            try:
                action = OrderSelectionAction.model_validate({key: saved[key] for key in ("type", "action_id", "source_turn_id", "orders")})
            except (KeyError, ValueError) as error:
                raise ActionNotFound("选单快照非法") from error
            context = source_context(conversation, source)
            expected_id = str(uuid5(NAMESPACE_URL, f"mewhelp:ch06:{context.conversation_id}:{context.turn_id}:select_order"))
            if (action.action_id != expected_id or action.source_turn_id != context.turn_id
                    or request.order_id not in {item.order_id for item in action.orders}
                    or not owns_order(request.user_id, request.order_id)):
                raise ActionNotFound("所选订单不存在")
            if saved.get("status") == "accepted":
                if saved.get("accepted_order_id") != request.order_id:
                    raise OrderSelectionConflict("同一动作不能改选其他订单")
                return AcceptedOrderSelection(context=context, action=action, accepted_order_id=request.order_id, turn_status=source.turn_status)
            if saved.get("status") != "awaiting_input" or source.turn_status != "awaiting_input":
                raise OrderSelectionConflict("这张选单卡已失效")
            source.rag_metadata = {**(source.rag_metadata or {}), "order_selection": {**saved, "status": "accepted", "accepted_order_id": request.order_id}}
            await session.execute(update(Message).where(Message.conversation_id == conversation.id,
                Message.turn_id == source.turn_id).values(turn_status="running"))
            return AcceptedOrderSelection(context=context, action=action, accepted_order_id=request.order_id, turn_status="running")

    def _refund_receipt(self, row: RefundApplication) -> RefundReceipt:
        """从真实申请行构造确定回执。参数：row已提交或将提交行；返回：回执；异常：数据损坏校验错误。"""
        return RefundReceipt(action_id=row.action_id, refund_no=row.refund_no, order_id=row.order_id,
                             reason_code=row.reason_code, status=row.status)

    async def get_refund_receipt(self, session_id: str, user_id: str, action_id: str) -> RefundReceipt | None:
        """独立新事务回读本人申请。参数：会话/用户/动作键；返回：确定回执或None；异常：数据库错误。"""
        async with self.factory() as session:
            row = await session.scalar(select(RefundApplication).join(Conversation, Conversation.id == RefundApplication.conversation_id)
                .where(Conversation.session_id == session_id, Conversation.user_id == user_id,
                       RefundApplication.user_id == user_id, RefundApplication.action_id == action_id))
            return self._refund_receipt(row) if row is not None else None

    def _same_refund(self, row, request, conversation) -> RefundReceipt:
        """已有真实回执优先，禁止同键换身份/来源/原因。参数：申请行/请求/会话；返回：回执；异常：404或409语义异常。"""
        if row.conversation_id != conversation.id or row.user_id != request.user_id or row.source_turn_id != request.source_turn_id:
            raise ActionNotFound("申请不存在")
        if row.reason_code != request.reason_code:
            raise OrderSelectionConflict("同一申请不能更换原因，已保存原申请")
        return self._refund_receipt(row)

    async def submit_refund(self, request: RefundSubmission) -> RefundReceipt:
        """独立事务保存本地申请，唯一动作键防重，响应丢失用新事务核实。

        参数：request仅包含身份、动作和已确认原因。
        返回：已提交的待处理回执，不表示批准退款。
        异常：ActionNotFound来源非法；OrderSelectionConflict同键改原因；数据库错误表示结果待核实。
        """
        try:
            async with self.factory.begin() as session:
                conversation = await session.scalar(select(Conversation).where(Conversation.session_id == request.session_id,
                                                                             Conversation.user_id == request.user_id))
                if conversation is None:
                    raise ActionNotFound("申请来源不存在")
                existing = await session.scalar(select(RefundApplication).where(RefundApplication.action_id == request.action_id))
                if existing is not None:
                    return self._same_refund(existing, request, conversation)
                source_id = await session.scalar(select(Message.id).where(Message.conversation_id == conversation.id,
                    Message.turn_id == request.source_turn_id, Message.role == "assistant", Message.tool_calls.is_(None),
                    Message.turn_status == "completed").order_by(Message.id.desc()).limit(1))
                if source_id is None:
                    raise ActionNotFound("申请来源未完成")
                source = await session.scalar(select(Message).where(Message.id == source_id).with_for_update())
                # 等待来源锁后再当前读，防止MySQL可重复读遗漏刚提交的同键申请。
                existing = await session.scalar(select(RefundApplication).where(RefundApplication.action_id == request.action_id).with_for_update())
                if existing is not None:
                    return self._same_refund(existing, request, conversation)
                metadata = source.rag_metadata or {}
                try:
                    action = RefundAction.model_validate(metadata.get("refund_action"))
                    _, user = await owned_source(session, request.session_id, request.user_id, request.source_turn_id)
                    if user is None or user.turn_status != "completed" or action.action_id != request.action_id:
                        raise ValueError("来源用户轮不一致")
                    context = source_context(conversation, user)
                    await validate_refund_evidence(session, context, action, metadata.get("workflow"))
                except ValueError as error:
                    raise ActionNotFound("退款建议未通过来源核验") from error
                row = RefundApplication(refund_no="RF" + uuid5(NAMESPACE_URL, f"mewhelp:ch06:refund:{action.action_id}").hex.upper(),
                    action_id=action.action_id, conversation_id=conversation.id, source_turn_id=request.source_turn_id,
                    user_id=request.user_id, order_id=action.order_id, reason_code=request.reason_code,
                    original_question=action.question, status="待处理")
                session.add(row)
                receipt = self._refund_receipt(row)
            return receipt
        except (ActionNotFound, OrderSelectionConflict):
            raise
        except Exception:
            # 失败session已退出/回滚；唯一键冲突或提交响应丢失后只用新事务回读，不重复插入。
            async with self.factory() as session:
                conversation = await session.scalar(select(Conversation).where(Conversation.session_id == request.session_id,
                                                                             Conversation.user_id == request.user_id))
                existing = await session.scalar(select(RefundApplication).where(RefundApplication.action_id == request.action_id))
                if conversation is not None and existing is not None:
                    return self._same_refund(existing, request, conversation)
            raise

    async def _trusted_action(self, session, request: TicketConfirmation) -> PreparedTicketAction:
        """先验归属再锁来源、读取建议。参数：session短事务、request确认；返回：可信派生动作；异常：ActionNotFound/InvalidTicketArguments/数据库错误。"""
        conversation = await session.scalar(select(Conversation).where(
            Conversation.session_id == request.session_id, Conversation.user_id == request.user_id))
        if conversation is None:
            raise ActionNotFound("建议不存在")
        # 先定位不可变来源主键，再锁单行，避免范围检索锁住动作插入间隙造成死锁。
        source_id = await session.scalar(select(Message.id).where(
            Message.conversation_id == conversation.id, Message.turn_id == request.source_turn_id,
            Message.role == "assistant", Message.tool_calls.is_(None), Message.turn_status == "completed"
        ).order_by(Message.id.desc()).limit(1))
        if source_id is None:
            raise ActionNotFound("建议不存在")
        source = await session.scalar(select(Message).where(Message.id == source_id).with_for_update())
        if (source is None or source.conversation_id != conversation.id or source.turn_id != request.source_turn_id
                or source.role != "assistant" or source.tool_calls is not None or source.turn_status != "completed"
                or not isinstance(source.rag_metadata, dict)):
            raise ActionNotFound("建议不存在")
        raw_actions = source.rag_metadata.get("suggested_actions", [])
        if not isinstance(raw_actions, list):
            raise ActionNotFound("建议不存在")
        candidates = [item for item in raw_actions if isinstance(item, dict) and item.get("action_id") == request.action_id]
        if len(candidates) != 1:
            raise ActionNotFound("建议不存在")
        try:
            suggestion = SuggestedAction.model_validate(candidates[0])
        except ValidationError as error:
            raise ActionNotFound("建议不存在") from error
        expected_id = str(uuid5(NAMESPACE_URL, f"mewhelp:ch05:{conversation.id}:{request.source_turn_id}:create_ticket"))
        if suggestion.kind != "create_ticket" or suggestion.source_turn_id != request.source_turn_id or suggestion.action_id != expected_id:
            raise ActionNotFound("建议不存在")
        original = await session.scalar(select(Message).where(
            Message.conversation_id == conversation.id, Message.turn_id == request.source_turn_id,
            Message.role == "user").order_by(Message.id).limit(1))
        if original is None or original.turn_status != "completed" or original.content != suggestion.question:
            raise ActionNotFound("建议不存在")
        try:
            arguments = TicketArgs(description=suggestion.question, ticket_type=suggestion.ticket_type)
        except ValidationError as error:
            raise InvalidTicketArguments("原始问题不能为空且不能超过2000字") from error
        context = ToolContext(conversation_id=conversation.id, session_id=conversation.session_id,
            user_id=conversation.user_id, turn_id=str(uuid5(NAMESPACE_URL, f"mewhelp:ch05:confirm:{expected_id}")),
            tool_call_id="ticket:" + expected_id, user_message="确认创建工单：" + suggestion.question)
        return PreparedTicketAction(request=request, context=context,
            call={"id": context.tool_call_id, "name": "create_ticket", "args": arguments.model_dump()}, turn_status="running")

    def _marker(self, action: PreparedTicketAction) -> dict:
        """生成独立动作身份标记。参数：action可信动作；返回：确定标记；异常：无。"""
        return {"action_id": action.request.action_id, "source_turn_id": action.request.source_turn_id}

    async def _action_rows(self, session, action: PreparedTicketAction) -> list[Message]:
        """在来源锁保护下核验独立动作记录。参数：session事务、action可信动作；返回：有序消息；异常：PersistenceUncertain/数据库错误。"""
        # MySQL可重复读中普通SELECT仍可能看见等锁前的快照；锁后必须当前读，
        # 否则第二个prepare会漏掉刚提交的确认消息，再追加一对消息。
        rows = list((await session.scalars(select(Message).where(
            Message.conversation_id == action.context.conversation_id, Message.turn_id == action.context.turn_id
        ).order_by(Message.id).with_for_update())).all())
        if not rows:
            return rows
        marker = self._marker(action)
        if (len(rows) < 2 or rows[0].role != "user" or rows[0].content != action.context.user_message
                or not isinstance(rows[0].rag_metadata, dict) or rows[0].rag_metadata.get("ticket_action") != marker
                or type(rows[0].rag_metadata.get("execution_started")) is not bool
                or rows[1].role != "assistant" or rows[1].tool_calls != [action.call]
                or rows[1].rag_metadata != {"ticket_action": marker}
                or any(row.role == "user" for row in rows[1:])):
            raise PersistenceUncertain("独立动作记录不一致")
        allowed = ["user", "assistant", "tool", "assistant"]
        if len(rows) > 4 or [row.role for row in rows] != allowed[:len(rows)]:
            raise PersistenceUncertain("独立动作消息配对不一致")
        if len(rows) >= 3 and rows[2].tool_call_id != action.context.tool_call_id:
            raise PersistenceUncertain("独立动作工具键不一致")
        return rows

    async def _verify_success(self, session, action: PreparedTicketAction, result: ToolResult) -> TicketReceipt:
        """核对成功工具事实及对应真实工单。参数：session事务、action动作、result已保存结果；返回：回执；异常：PersistenceUncertain。"""
        try:
            if not result.ok or result.error is not None or result.is_mock or not result.data:
                raise ValueError("没有真实成功事实")
            receipt = TicketReceipt(action_id=action.request.action_id, **result.data)
            ticket = await session.scalar(select(Ticket).where(Ticket.ticket_no == receipt.ticket_no).with_for_update())
            if (receipt.ticket_no != ticket_number(action.context)
                    or ticket is None or ticket.conversation_id != action.context.conversation_id
                    or ticket.description != action.call["args"]["description"]
                    or ticket.ticket_type != action.call["args"]["ticket_type"]
                    or receipt.ticket_type != ticket.ticket_type):
                raise ValueError("工单与建议不匹配")
        except (ValueError, TypeError) as error:
            raise PersistenceUncertain("成功事实与工单不一致") from error
        return receipt

    async def prepare_ticket(self, request: TicketConfirmation) -> PreparedTicketAction:
        """验证保存快照并只追加一次独立确认与申请。参数：request严格确认；返回：动作/既有状态与结果；异常：ActionNotFound/InvalidTicketArguments/一致性或数据库错误。"""
        async with self.factory.begin() as session:
            action = await self._trusted_action(session, request)
            rows = await self._action_rows(session, action)
            if not rows:
                marker = self._marker(action)
                session.add_all([
                    Message(conversation_id=action.context.conversation_id, turn_id=action.context.turn_id,
                        role="user", content=action.context.user_message,
                        rag_metadata={"ticket_action": marker, "execution_started": False}),
                    Message(conversation_id=action.context.conversation_id, turn_id=action.context.turn_id,
                        role="assistant", content=None, tool_calls=[action.call], rag_metadata={"ticket_action": marker}),
                ])
                return action
            result = ToolResult.model_validate_json(rows[2].content) if len(rows) >= 3 else None
            if result is not None and result.ok:
                await self._verify_success(session, action, result)
            return action.model_copy(update={"turn_status": rows[0].turn_status,
                "execution_started": rows[0].rag_metadata["execution_started"], "existing_result": result})

    async def get_ticket_receipt(self, session_id: str, user_id: str, source_turn_id: str, action_id: str) -> TicketReceipt | None:
        """只读回查原动作的已提交工单，不准备动作、不执行工具或补写审计。

        参数：session_id/user_id为来源身份，source_turn_id为已完成原问，action_id为保存建议的稳定键。
        返回：已核验真实工单的TicketReceipt；尚无成功工具结果或原问不能建单时返回None。
        异常：ActionNotFound，身份/建议不符；PersistenceUncertain，工具事实或回执损坏；数据库错误。
        """
        request = TicketConfirmation(session_id=session_id, user_id=user_id, source_turn_id=source_turn_id,
                                     action_id=action_id, confirmed=True)
        async with self.factory.begin() as session:
            try:
                action = await self._trusted_action(session, request)
            except InvalidTicketArguments:
                # 超长原问仍可展示；它未能进入合法建单流程，回读不得因此让聊天恢复失败。
                return None
            rows = await self._action_rows(session, action)
            if len(rows) < 3:
                return None
            result = ToolResult.model_validate_json(rows[2].content)
            if not result.ok:
                return None
            receipt = await self._verify_success(session, action, result)
            if len(rows) == 4:
                expected = {"status":"completed", "turn_id":action.context.turn_id,
                            "ticket_action":self._marker(action), "ticket_receipt":receipt.model_dump()}
                if rows[3].rag_metadata != expected or any(row.turn_status != "completed" for row in rows):
                    raise PersistenceUncertain("已保存工单回执与成功事实不一致")
                return receipt
            # cancelled/failed仅表示审计轮中断，不能覆盖真实工单已提交的事实。
            return receipt.model_copy(update={"audit_sync_pending": True})

    async def _revalidate(self, session, action: PreparedTicketAction) -> list[Message]:
        """收尾或执行前重新核验全部稳定身份。参数：session事务、action传入动作；返回：独立消息；异常：一致性/身份/数据库错误。"""
        trusted = await self._trusted_action(session, action.request)
        if trusted.context != action.context or trusted.call != action.call:
            raise PersistenceUncertain("动作稳定身份不匹配")
        rows = await self._action_rows(session, trusted)
        if not rows:
            raise PersistenceUncertain("独立动作尚未准备")
        return rows

    async def begin_ticket_execution(self, action: PreparedTicketAction) -> None:
        """记录写执行边界，未知提交不能重跑。参数：action已准备动作；返回：None；异常：PersistenceUncertain/数据库错误。"""
        async with self.factory.begin() as session:
            rows = await self._revalidate(session, action)
            # 无结果时只放行首次running或执行前cancelled；failed仍未知，原键重试不能变成写入。
            if (len(rows) != 2 or rows[0].rag_metadata["execution_started"]
                    or rows[0].turn_status not in {"running", "cancelled"}):
                raise PersistenceUncertain("动作执行状态待核实，请使用原动作重试查询")
            # JSON整体赋值触发ORM变更；只对明确未执行的旧取消动作恢复running。
            rows[0].rag_metadata = {**rows[0].rag_metadata, "execution_started": True}
            for row in rows:
                row.turn_status = "running"

    async def finish_ticket_action(self, action: PreparedTicketAction, result: ToolResult) -> dict:
        """凭真实成功事务修复独立动作终态并只追加一次回执。

        参数：action仓储派生动作；result执行器返回的成功事实，必须与库中完全一致。
        返回：dict，保存的确定回执元数据；completed重试比对后原样返回。
        异常：PersistenceUncertain，三键/事实/回执不一致；身份和数据库错误。
        """
        async with self.factory.begin() as session:
            rows = await self._revalidate(session, action)
            if len(rows) < 3 or ToolResult.model_validate_json(rows[2].content) != result:
                raise PersistenceUncertain("没有对应的已提交成功工具结果")
            receipt = await self._verify_success(session, action, result)
            metadata = {"status": "completed", "turn_id": action.context.turn_id,
                        "ticket_action": self._marker(action), "ticket_receipt": receipt.model_dump()}
            content = f"工单已创建，工单号：{receipt.ticket_no}，类型：{receipt.ticket_type}，状态：{receipt.status}。"
            if len(rows) == 4:
                if rows[3].content != content or rows[3].rag_metadata != metadata or any(row.turn_status != "completed" for row in rows):
                    raise PersistenceUncertain("动作已收尾但回执或终态不同")
                return rows[3].rag_metadata
            if any(row.turn_status == "completed" for row in rows):
                raise PersistenceUncertain("动作缺少完成回执")
            session.add(Message(conversation_id=action.context.conversation_id, turn_id=action.context.turn_id,
                role="assistant", content=content, turn_status="completed", rag_metadata=metadata))
            for row in rows:
                row.turn_status = "completed"
            return metadata
