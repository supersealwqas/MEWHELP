"""独立确认建单HTTP接口；严格请求校验和稳定错误映射，不启动图或模型。"""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.chat import get_chat_service
from app.repositories.actions import ActionRepository
from app.schemas.workflow import TicketConfirmation, TicketReceipt, CurrentActions
from app.services.actions import TicketActionError, TicketActionService
from app.tools.executor import ToolExecutor
from app.tools.registry import build_ticket_registry

router = APIRouter(prefix="/api/actions", tags=["actions"])


@router.get("/current", response_model=CurrentActions)
async def current_actions(session_id: str, user_id: str = "demo-user", service=Depends(get_chat_service)) -> CurrentActions:
    """回读当前会话动作。参数：身份及共享服务；返回：保存状态；异常：404身份、503读取故障。"""
    try:
        return await service.get_current_actions(session_id.strip(), user_id.strip())
    except ValueError as error:
        raise HTTPException(status_code=404, detail={"message": "会话不存在或不属于当前用户"}) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail={"message": "暂时无法读取处理状态", "outcome": "unknown"}) from error


def get_ticket_action_service(request: Request) -> TicketActionService:
    """复用线上聊天的仓储、业务资源和同一锁。参数：request HTTP请求；返回：确认服务；异常：运行时未就绪。"""
    chat = get_chat_service(request)
    settings = chat.settings
    executor = ToolExecutor(build_ticket_registry(), chat.chat, chat.executor.support,
        settings.tool_timeout_seconds, settings.tool_max_retries, settings.tool_retry_delay_seconds)
    return TicketActionService(ActionRepository(chat.chat.factory), executor, chat.guard)


@router.post("/tickets", response_model=TicketReceipt)
async def confirm_ticket(request: TicketConfirmation,
                         service: TicketActionService = Depends(get_ticket_action_service)) -> TicketReceipt:
    """确认保存建议并返回原票据。参数：request严格请求、service注入服务；返回：成功回执；异常：404/409/422/503安全HTTP错误。"""
    try:
        return await service.confirm_ticket(request)
    except TicketActionError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error
