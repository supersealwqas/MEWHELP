"""CH08 真实 MySQL 确认恢复验收，仅使用显式 mewhelp_test 夹具，不调用外部模型。"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict

import pytest
from sqlalchemy import select, text

from app.models.customer_service import Message, Ticket
from app.models.tool_audit import ToolAuditLog
from app.repositories.actions import ActionRepository
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.repositories.tool_audit import ToolAuditRepository
from app.schemas.workflow import AgentDecision, OrderSelectionConflict, TicketResume
from app.services.chat import ChatService
from app.tools.contracts import WriteAuthorization
from app.tools.executor import ToolExecutor
from app.workflow.runtime import workflow_runtime
from tests.workflow.graph_fakes import dependencies
from tests.workflow.test_ch08_resume import open_preview, saved_preview
from tests.workflow.test_ch08_ticket_interrupt import ticket_call


pytestmark = pytest.mark.mysql


@asynccontextmanager
async def mysql_service(factory, path, *, answer=None):
    """真实数据库、图及执行器配有限模型脚本，资源在退出时关闭。

    参数：factory mewhelp_test 会话工厂、path 检查点目录、answer 可选正文故障；
    返回：服务和依赖；异常：数据库、检查点或断言错误。
    """
    async with factory() as session:
        assert await session.scalar(text("SELECT DATABASE()")) == "mewhelp_test"
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
        AgentDecision(kind="answer")], answers=[[answer or "已记录。"]])
    deps.chat = ChatRepository(factory)
    deps.executor = ToolExecutor(deps.registry, deps.chat, SupportRepository(factory),
        audit_repository=ToolAuditRepository(factory), retry_delay_seconds=0)
    async with workflow_runtime(deps, path / "ticket-checkpoints.sqlite3") as graph:
        yield ChatService(deps.chat, graph, deps.executor, deps.settings), deps


async def snapshot(factory):
    """读取提交后的业务和审计事实。参数：factory 独占工厂；返回：工单、审计、用户消息；异常：数据库错误。"""
    async with factory() as session:
        tickets = list((await session.scalars(select(Ticket))).all())
        audits = list((await session.scalars(select(ToolAuditLog))).all())
        users = list((await session.scalars(select(Message).where(Message.role == "user"))).all())
        return tickets, audits, users


async def test_confirm_cancel_race_has_one_winner(database):
    """真实 MySQL 行锁下确认与取消必须只接受一项，重发回读相同凭据。

    参数：database 独占测试库；返回：None；异常：数据库或断言错误。
    """
    _, factory = database
    _, _, body = await saved_preview(ChatRepository(factory), factory)
    results = await asyncio.gather(*(ActionRepository(factory).accept_ticket_resume(
        TicketResume(**{**body, "decision": decision})) for decision in ("confirm", "cancel")), return_exceptions=True)
    winners = [item for item in results if isinstance(item, WriteAuthorization)]
    conflicts = [item for item in results if isinstance(item, OrderSelectionConflict)]
    assert len(winners) == len(conflicts) == 1, [type(item).__name__ for item in results]
    repeated = await ActionRepository(factory).accept_ticket_resume(TicketResume(**{**body, "decision": winners[0].decision}))
    assert asdict(repeated) == asdict(winners[0])
    tickets, audits, users = await snapshot(factory)
    assert tickets == [] and audits == [] and len(users) == 1


@pytest.mark.parametrize("decision", ["confirm", "cancel"])
async def test_mysql_resume_preserves_one_ticket_and_one_audit(database, tmp_path, decision):
    """确认只增一单一审计，取消零工单且留权限拒绝；响应丢失重发不能重复写入。

    参数：database 测试库、tmp_path 检查点目录、decision 首次决定；返回：None；异常：数据库或断言错误。
    """
    _, factory = database
    async with mysql_service(factory, tmp_path) as (service, _):
        body = await open_preview(service)
        request = TicketResume(**{**body, "decision": decision})
        first = [event async for event in service.stream_resume(request)]
        assert first[-1].event == "done", [(event.event, event.data) for event in first]
        repeated = [event async for event in service.stream_resume(request)]
        assert repeated[-1].event == "done"
        tickets, audits, users = await snapshot(factory)
        assert len(users) == 1
        assert len(tickets) == (1 if decision == "confirm" else 0)
        assert len(audits) == 1 and audits[0].status == ("成功" if decision == "confirm" else "权限拒绝")
        if tickets:
            assert tickets[0].ticket_no in "".join(event.data["delta"] for event in repeated if event.event == "delta")


@pytest.mark.parametrize("fault", ["answer", "audit"])
async def test_mysql_committed_ticket_survives_later_fault(database, tmp_path, monkeypatch, fault):
    """正文和独立审计故障不能撤回已提交工单或丢失原回执。

    参数：database 测试库、tmp_path 检查点目录、monkeypatch 故障注入、fault 故障边界；
    返回：None；异常：数据库或断言错误。
    """
    _, factory = database
    async with mysql_service(factory, tmp_path, answer=RuntimeError("正文故障") if fault == "answer" else None) as (service, deps):
        if fault == "audit":
            async def fail_audit(entry):
                """只阻断独立审计事务，不影响工具业务提交。参数：entry 审计输入；返回：无；异常：OSError。"""
                raise OSError("独立审计不可用")

            monkeypatch.setattr(deps.executor.audit, "record", fail_audit)
        body = await open_preview(service)
        events = [event async for event in service.stream_resume(TicketResume(**body))]
        tickets, audits, users = await snapshot(factory)
        assert len(tickets) == len(users) == 1
        assert len(audits) == (0 if fault == "audit" else 1)
        assert tickets[0].ticket_no in "".join(event.data["delta"] for event in events if event.event == "delta")
        current = await service.get_current_actions("ticket-session", "demo-user")
        assert current.ticket_receipt.ticket_no == tickets[0].ticket_no
        repeated = [event async for event in service.stream_resume(TicketResume(**body))]
        assert tickets[0].ticket_no in "".join(event.data["delta"] for event in repeated if event.event == "delta")
        assert len((await snapshot(factory))[0]) == 1
