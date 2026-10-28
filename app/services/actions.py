"""独立确认动作服务：复用聊天互斥与已有执行器，返回确定工单或明确未知状态。"""

import asyncio
import logging

import anyio
from app.repositories.actions import ActionNotFound, ActionRepository, InvalidTicketArguments
from app.repositories.chat import PersistenceUncertain
from app.schemas.workflow import TicketConfirmation, TicketReceipt, RefundSubmission, RefundReceipt, OrderSelectionConflict
from app.services.chat import ConversationBusy, ConversationGuard
from app.tools.contracts import ToolExecutionScope, WriteAuthorization, json_hash

logger = logging.getLogger(__name__)


class TicketActionError(RuntimeError):
    """可安全映射为HTTP的动作错误；未知状态不能被解释成未建单。"""

    def __init__(self, status_code: int, code: str, message: str, outcome: str):
        """保存公开错误字段。参数：HTTP状态/稳定代码/中文提示/结果状态；返回：None；异常：无。"""
        super().__init__(message)
        self.status_code = status_code
        self.detail = {"code": code, "message": message, "outcome": outcome}


async def _ignore_status(status: dict) -> None:
    """确认接口无SSE，消费已有执行器状态。参数：status安全状态帧；返回：None；异常：无。"""


class RefundActionService:
    """独立本地退款申请服务，与聊天/工单共用会话互斥，不运行模型或写退款工具。"""

    def __init__(self, repository: ActionRepository, guard: ConversationGuard):
        """注入原仓储及同一会话锁。参数：repository动作仓储、guard互斥；返回：None；异常：无。"""
        self.repository, self.guard = repository, guard

    async def submit_refund(self, request: RefundSubmission) -> RefundReceipt:
        """确认后保存本地待处理申请。参数：request原键及原因；返回：确定回执；异常：TicketActionError安全HTTP错误或取消。"""
        try:
            async with self.guard.hold(request.session_id):
                return await self.repository.submit_refund(request)
        except ConversationBusy as error:
            raise TicketActionError(409, "CONVERSATION_BUSY", "当前会话正在处理，请稍后重试原申请", "not_started") from error
        except OrderSelectionConflict as error:
            raise TicketActionError(409, "REFUND_CONFLICT", str(error), "not_started") from error
        except ActionNotFound as error:
            raise TicketActionError(404, "ACTION_NOT_FOUND", "退款建议不存在或不属于当前会话", "not_started") from error
        except Exception as error:
            logger.warning("refund outcome unknown error=%s", type(error).__name__)
            raise TicketActionError(503, "REFUND_OUTCOME_UNKNOWN", "暂时无法核实申请结果，请保留原操作查询", "unknown") from error


class TicketActionService:
    """确认服务不运行图或模型，只执行服务端保存建议对应的一次建单。"""

    def __init__(self, repository: ActionRepository, executor, guard: ConversationGuard):
        """注入可信边界及共享锁。参数：repository动作仓储、executor已有执行器、guard聊天同一互斥；返回：None；异常：无。"""
        self.repository, self.executor, self.guard = repository, executor, guard

    async def confirm_ticket(self, request: TicketConfirmation) -> TicketReceipt:
        """以原动作键确认或查询真实回执。参数：request严格确认；返回：真实票据回执；异常：TicketActionError/用户取消。"""
        try:
            async with self.guard.hold(request.session_id):
                return await self._confirm(request)
        except TicketActionError:
            raise
        except ConversationBusy as error:
            raise TicketActionError(409, "CONVERSATION_BUSY", "当前会话正在处理，请稍后重试原操作", "not_started") from error
        except ActionNotFound as error:
            raise TicketActionError(404, "ACTION_NOT_FOUND", "该建单建议不存在或不属于当前会话", "not_started") from error
        except InvalidTicketArguments as error:
            raise TicketActionError(422, "INVALID_TICKET_ARGUMENTS", "原始问题不能为空且不能超过2000字，请重新描述后确认", "not_started") from error
        except Exception as error:
            # 原始数据库异常可能含连接信息，不回显也不把unknown伪装成确定失败。
            logger.warning("ticket action outcome unknown error=%s", type(error).__name__)
            raise TicketActionError(503, "TICKET_OUTCOME_UNKNOWN", "暂时无法核实建单结果，请保留原操作重试查询", "unknown") from error

    async def _mark(self, action, status: str) -> None:
        """尽力标记动作中断，不覆盖已知业务事实。参数：action可信动作、status失败/取消；返回：None；异常：日志外不抛。"""
        try:
            await self.executor.chat.mark_turn(action.context, status)
        except Exception as error:
            logger.warning("ticket action terminal pending error=%s", type(error).__name__)

    async def _confirm(self, request: TicketConfirmation) -> TicketReceipt:
        """按准备、确定事实、专用收尾顺序执行。参数：request确认；返回：真实回执；异常：身份/参数/一致性/取消错误。"""
        action = await self.repository.prepare_ticket(request)
        result = action.existing_result
        if result is None:
            try:
                async def authorize(binding, arguments, context):
                    """把已验证的旧按钮决定绑定到原调用，不再次展示聊天预览。

                    参数：binding 当前工具来源、arguments 冻结参数、context 服务器调用；
                    返回：原动作专用授权；异常：PersistenceUncertain 或原来源核验错误。
                    """
                    verified = await self.repository.prepare_ticket(request)
                    if (binding.identity.source != "builtin" or binding.identity.server is not None
                            or binding.identity.name != "create_ticket" or context != verified.context
                            or arguments != verified.call["args"] or verified.context != action.context
                            or verified.call != action.call or not verified.execution_started):
                        raise PersistenceUncertain("按钮确认与原工具申请不一致")
                    return WriteAuthorization(conversation_id=context.conversation_id, turn_id=context.turn_id,
                        action_id=request.action_id, tool_call_id=context.tool_call_id,
                        arguments_hash=json_hash(arguments), decision="confirm", origin="action")

                await self.repository.begin_ticket_execution(action)
                result = await self.executor.execute(action.call, action.context, _ignore_status,
                                                     scope=ToolExecutionScope(authorize_write=authorize))
            except asyncio.CancelledError:
                with anyio.CancelScope(shield=True):
                    await self._mark(action, "cancelled")
                raise
            except Exception:
                # 工具事务可能已成功但响应丢失；先重新验明身份并核实，不换ID重放。
                refreshed = await self.repository.prepare_ticket(request)
                result = refreshed.existing_result
                if result is None:
                    await self._mark(action, "failed")
                    raise PersistenceUncertain("动作没有可核实的执行结果")
        if not result.ok:
            await self._mark(action, "failed")
            error = result.error
            raise TicketActionError(422, error.code if error else "TICKET_FAILED",
                error.message if error else "该次建单已确定失败", "failed")
        receipt = TicketReceipt(action_id=request.action_id, **(result.data or {}))
        try:
            metadata = await self.repository.finish_ticket_action(action, result)
            return TicketReceipt.model_validate(metadata["ticket_receipt"])
        except Exception as error:
            # 这里已有执行器核实的成功事实；审计失败只能标待修复，不能重跑写工具。
            logger.warning("ticket action audit pending error=%s", type(error).__name__)
            return receipt.model_copy(update={"audit_sync_pending": True})
