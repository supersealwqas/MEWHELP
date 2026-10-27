"""确认建单的真实仓储、HTTP边界和中断恢复测试；离线SQLite不冒充MySQL行锁验收。"""

import asyncio
from contextlib import asynccontextmanager
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import select, text

from app.models.customer_service import Conversation, Message, Ticket
from app.repositories.chat import PersistenceUncertain
from app.repositories.support import SupportRepository
from app.schemas.tools import TicketArgs, ToolContext, ToolResult
from app.schemas.workflow import AgentDecision, SuggestedAction
from tests.workflow.service_fakes import collect, service_runtime


@asynccontextmanager
async def action_runtime(path, **script):
    """真实仓储/执行器/API的离线入口。参数：path独占目录、script模型脚本；返回：服务/动作/API/工厂；异常：资源或断言失败。"""
    from app.main import create_app
    async with service_runtime(path, **script) as (chat, deps, factory):
        async with factory.begin() as session:
            await session.execute(text("CREATE TABLE IF NOT EXISTS tickets (ticket_no TEXT PRIMARY KEY, conversation_id INTEGER, description TEXT, ticket_type TEXT, status TEXT DEFAULT '待处理', created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
        app = create_app()
        from app.repositories.actions import ActionRepository
        from app.services.actions import TicketActionService
        from app.tools.executor import ToolExecutor
        from app.tools.registry import build_ticket_registry
        executor = ToolExecutor(build_ticket_registry(), deps.chat, SupportRepository(factory), retry_delay_seconds=0)
        actions = TicketActionService(ActionRepository(factory), executor, chat.guard)
        from app.api.actions import get_ticket_action_service
        app.dependency_overrides[get_ticket_action_service] = lambda: actions
        app.state.chat_service = chat
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/api/actions/tickets", json={})).status_code == 422, "独立确认API必须拒绝缺失请求字段"
            yield chat, actions, client, factory


async def saved_suggestion(chat, *, question="  旧订单未收到  ", session_id="s", user_id="demo-user", turn_id="old-question"):
    """保存已完成真实来源快照。参数：chat仓储与原问/身份/轮次；返回：确认请求字典；异常：数据库错误。"""
    context = await chat.start_turn(ToolContext(session_id=session_id, user_id=user_id, turn_id=turn_id, user_message=question), question)
    action_id = str(uuid5(NAMESPACE_URL, f"mewhelp:ch05:{context.conversation_id}:{turn_id}:create_ticket"))
    action = SuggestedAction(action_id=action_id, kind="create_ticket", source_turn_id=turn_id, question=question, ticket_type="售后")
    await chat.finish_turn(context, "可以确认创建工单", suggested_actions=[action])
    return {"session_id": session_id, "user_id": user_id, "source_turn_id": turn_id, "action_id": action_id, "confirmed": True}


async def records(factory):
    """读取本例所有真实行。参数：factory独占会话工厂；返回：会话/消息/工单；异常：数据库错误。"""
    async with factory() as session:
        return tuple([list((await session.scalars(select(model).order_by(model.id if model is not Ticket else model.ticket_no))).all()) for model in (Conversation, Message, Ticket)])


@pytest.mark.parametrize("question", ["  原样提交  ", " " + "字" * 1998 + " "], ids=["raw", "raw-2000"])
def test_ticket_args_preserve_raw_whitespace(question):
    """去首尾空白后入库会失败。参数：question原始描述；返回：None；异常：断言失败。"""
    assert TicketArgs(description=question, ticket_type="售后").description == question


@pytest.mark.parametrize("question", [" " * 3, " " + "字" * 1999 + " "], ids=["blank", "raw-2001"])
def test_ticket_args_reject_blank_and_raw_length_2001(question):
    """把2001字裁成1999或接受纯空白会失败。参数：question非法原文；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError):
        TicketArgs(description=question, ticket_type="售后")


@pytest.mark.parametrize("confirmed", [None, False, 1, 1.0, "true"])
async def test_only_explicit_json_true_can_confirm(tmp_path, confirmed):
    """缺确认或数字冒充布尔写库时失败。参数：tmp_path/confirmed；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        if confirmed is None:
            body.pop("confirmed")
        else:
            body["confirmed"] = confirmed
        response = await client.post("/api/actions/tickets", json=body)
        assert response.status_code == 422
        assert len((await records(factory))[1]) == 2 and (await records(factory))[2] == []


@pytest.mark.parametrize("override", [{"user_id": "other"}, {"session_id": "other"}, {"source_turn_id": "other"}, {"action_id": "00000000-0000-5000-8000-000000000000"}])
async def test_identity_and_source_checked_before_cached_receipt(tmp_path, override):
    """已有票据先读缓存再验身份会泄漏回执。参数：tmp_path/override伪造字段；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        assert (await client.post("/api/actions/tickets", json=body)).status_code == 200
        response = await client.post("/api/actions/tickets", json={**body, **override})
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "ACTION_NOT_FOUND"
        assert len((await records(factory))[2]) == 1


@pytest.mark.parametrize("extra", [{"ticket_type": "投诉"}, {"description": "替换原文"}, {"audit_sync_pending": True}])
async def test_client_cannot_override_snapshot_or_receipt(tmp_path, extra):
    """接受客户端类型/原文/回执字段时失败。参数：tmp_path/extra非法字段；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        assert (await client.post("/api/actions/tickets", json={**body, **extra})).status_code == 422
        assert (await records(factory))[2] == []


async def test_plain_chat_and_suggestions_do_not_write_then_old_action_uses_original(tmp_path):
    """普通建单话术写库、旧按钮取最新问题或顺便转人工会失败。参数：tmp_path；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path, intents=["其他", "闲聊"], decisions=[AgentDecision(kind="answer", suggested_actions=["create_ticket"])], answers=[["请确认"]]) as (chat, _, client, factory):
        before = len((await records(factory))[2])
        events = await collect(chat, "  帮我建单，订单没收到  ")
        suggestion = events[-2].data["suggested_actions"][0]
        assert len((await records(factory))[2]) == before
        await collect(chat, "你好，这是另一个问题")
        body = {"session_id": "s", "source_turn_id": suggestion["source_turn_id"], "action_id": suggestion["action_id"], "confirmed": True}
        first = await client.post("/api/actions/tickets", json=body)
        retry = await client.post("/api/actions/tickets", json=body)
        assert first.status_code == retry.status_code == 200 and first.json() == retry.json()
        conversations, rows, tickets = await records(factory)
        assert len(tickets) == 1 and tickets[0].description == "  帮我建单，订单没收到  "
        assert conversations[0].status == "进行中"
        assert all(row.turn_status == "completed" for row in rows)
        action_turn = str(uuid5(NAMESPACE_URL, f"mewhelp:ch05:confirm:{suggestion['action_id']}"))
        assert [row.role for row in rows if row.turn_id == action_turn] == ["user", "assistant", "tool", "assistant"]
        delta = await chat.chat.load_history_delta(conversations[0].id, rows[3].id)
        assert len(delta.turns) == 1 and tickets[0].ticket_no in delta.turns[0].messages[-1].content


@pytest.mark.parametrize("length, expected", [(2000, 200), (2001, 422), (4000, 422)])
async def test_saved_raw_length_is_not_trimmed_or_truncated(tmp_path, length, expected):
    """以去空白长度建超限工单或裁剪4000字时失败。参数：tmp_path/length/expected；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        question = " " + "字" * (length - 2) + " "
        body = await saved_suggestion(chat.chat, question=question)
        response = await client.post("/api/actions/tickets", json=body)
        assert response.status_code == expected
        tickets = (await records(factory))[2]
        assert (len(tickets) == 1 and tickets[0].description == question) if expected == 200 else tickets == []


async def test_shared_chat_guard_and_double_click_return_busy_then_same_receipt(tmp_path):
    """确认未复用聊天锁或双击执行两次时失败。参数：tmp_path；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        async with chat.guard.hold("s"):
            busy = await client.post("/api/actions/tickets", json=body)
        assert busy.status_code == 409 and (await records(factory))[2] == []
        pair = await asyncio.gather(*(client.post("/api/actions/tickets", json=body) for _ in range(2)))
        assert sorted(item.status_code for item in pair) == [200, 409]
        repeated = await client.post("/api/actions/tickets", json=body)
        assert repeated.json() == next(item.json() for item in pair if item.status_code == 200)


@pytest.mark.parametrize("session_id", ["s ", " s", "\ts \n"])
async def test_confirmation_normalizes_identity_while_real_chat_is_active(tmp_path, monkeypatch, session_id):
    """确认身份空白绕过正在运行的聊天锁时失败。参数：tmp_path目录、monkeypatch等待注入、session_id等价身份；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path, intents=["闲聊"]) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        entered, release = asyncio.Event(), asyncio.Event()
        original = chat.chat.finish_turn

        async def waiting_finish(*args, **kwargs):
            """仅在真实事务开始前等待，避免SQLite写锁干扰互斥断言。参数：完成事务参数；返回：真实元数据；异常：数据库错误。"""
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(chat.chat, "finish_turn", waiting_finish)
        task = asyncio.create_task(collect(chat, "你好", session=" s "))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            before = await records(factory)
            response = await client.post("/api/actions/tickets", json={**body, "session_id": session_id, "user_id": " demo-user "})
            after = await records(factory)
            assert response.status_code == 409
            assert [row.id for row in after[1]] == [row.id for row in before[1]]
            assert after[2] == []
        finally:
            release.set()
            assert (await task)[-1].event == "done"
        receipt = await client.post("/api/actions/tickets", json={**body, "session_id": session_id, "user_id": " demo-user "})
        assert receipt.status_code == 200
        assert (await records(factory))[2][0].description == "  旧订单未收到  "


@pytest.mark.parametrize("field", ["session_id", "user_id"])
async def test_confirmation_rejects_blank_identity_without_writing(tmp_path, field):
    """纯空白身份进入确认流程时失败。参数：tmp_path独占目录、field身份字段；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        response = await client.post("/api/actions/tickets", json={**body, field: " \t "})
        assert response.status_code == 422
        _, rows, tickets = await records(factory)
        assert len(rows) == 2 and tickets == []


@pytest.mark.parametrize("terminal", ["cancelled", "failed"])
async def test_committed_ticket_failed_audit_returns_real_receipt_then_repairs(tmp_path, monkeypatch, terminal):
    """提交后收尾失败被误报没建成、恢复改源轮或多回执时失败。参数：tmp_path/monkeypatch/terminal中断终态；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, actions, client, factory):
        body = await saved_suggestion(chat.chat)
        original = actions.repository.finish_ticket_action

        async def fail_finish(action, result):
            """仅注入审计失败。参数：已提交动作/结果；返回：无；异常：OSError。"""
            raise OSError("审计暂不可写")

        monkeypatch.setattr(actions.repository, "finish_ticket_action", fail_finish)
        first = await client.post("/api/actions/tickets", json=body)
        assert first.status_code == 200 and first.json()["audit_sync_pending"] is True
        if terminal == "cancelled":
            assert await chat.chat.recover_interrupted_turns() == 1
        else:
            from app.schemas.workflow import TicketConfirmation
            prepared = await actions.repository.prepare_ticket(TicketConfirmation(**body))
            await chat.chat.mark_turn(prepared.context, "failed")
        monkeypatch.setattr(actions.repository, "finish_ticket_action", original)
        second = await client.post("/api/actions/tickets", json=body)
        assert second.status_code == 200 and second.json()["audit_sync_pending"] is False
        assert second.json()["ticket_no"] == first.json()["ticket_no"]
        _, rows, tickets = await records(factory)
        assert len(tickets) == 1 and all(row.turn_status == "completed" for row in rows)
        assert len(rows) == 6


async def test_known_failure_reuses_result_without_reexecuting(tmp_path):
    """已缓存确定失败改键重执行时失败。参数：tmp_path；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, actions, client, factory):
        from app.schemas.workflow import TicketConfirmation
        body = await saved_suggestion(chat.chat)
        prepared = await actions.repository.prepare_ticket(TicketConfirmation(**body))
        await chat.chat.save_tool_result(prepared.context, ToolResult.failed("INVALID_ARGUMENTS", "原请求不能执行"))
        first = await client.post("/api/actions/tickets", json=body)
        second = await client.post("/api/actions/tickets", json=body)
        assert first.status_code == second.status_code == 422 and first.json() == second.json()
        assert first.json()["detail"]["outcome"] == "failed" and (await records(factory))[2] == []


async def test_unknown_database_outcome_does_not_claim_failure(tmp_path, monkeypatch):
    """无法读库时给确定失败或重复写入会失败。参数：tmp_path/monkeypatch；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, actions, client, factory):
        body = await saved_suggestion(chat.chat)

        async def unavailable(request):
            """仅注入不可读数据库。参数：request确认；返回：无；异常：OSError。"""
            raise OSError("离线数据库")

        monkeypatch.setattr(actions.repository, "prepare_ticket", unavailable)
        response = await client.post("/api/actions/tickets", json=body)
        assert response.status_code == 503 and response.json()["detail"]["outcome"] == "unknown"
        assert (await records(factory))[2] == []


async def test_cancelled_before_execution_can_reconfirm_but_started_unknown_cannot(tmp_path):
    """执行前取消不能重试或执行过却未知时盲目重跑会失败。参数：tmp_path；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, actions, client, factory):
        from app.schemas.workflow import TicketConfirmation
        body = await saved_suggestion(chat.chat)
        prepared = await actions.repository.prepare_ticket(TicketConfirmation(**body))
        await chat.chat.mark_turn(prepared.context, "cancelled")
        assert (await client.post("/api/actions/tickets", json=body)).status_code == 200
        another = await saved_suggestion(chat.chat, turn_id="second-question")
        prepared = await actions.repository.prepare_ticket(TicketConfirmation(**another))
        await actions.repository.begin_ticket_execution(prepared)
        await chat.chat.mark_turn(prepared.context, "cancelled")
        response = await client.post("/api/actions/tickets", json=another)
        assert response.status_code == 503 and response.json()["detail"]["outcome"] == "unknown"
        assert len((await records(factory))[2]) == 1


async def test_failed_before_execution_stays_unknown_when_original_action_retried(tmp_path, monkeypatch):
    """执行前异常产生failed后，重试查询不能变成新写入。参数：tmp_path独占目录、monkeypatch故障注入；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, actions, client, factory):
        body = await saved_suggestion(chat.chat)

        async def fail_before_execution(action):
            """在真实服务写执行边界之前失败。参数：action已准备动作；返回：无；异常：受控OSError。"""
            raise OSError("执行边界暂不可写")

        with monkeypatch.context() as patch:
            patch.setattr(actions.repository, "begin_ticket_execution", fail_before_execution)
            first = await client.post("/api/actions/tickets", json=body)
        assert first.status_code == 503 and first.json()["detail"]["outcome"] == "unknown"
        _, before_rows, before_tickets = await records(factory)
        assert before_tickets == [] and len(before_rows) == 4
        assert [row.turn_status for row in before_rows] == ["completed", "completed", "failed", "failed"]
        assert before_rows[2].rag_metadata["execution_started"] is False

        # 恢复真实begin实现后，原请求仍只能查结果；failed不能按取消状态放行执行。
        for _ in range(2):
            retry = await client.post("/api/actions/tickets", json=body)
            assert retry.status_code == 503 and retry.json()["detail"]["outcome"] == "unknown"
        _, after_rows, after_tickets = await records(factory)
        assert after_tickets == [] and [row.id for row in after_rows] == [row.id for row in before_rows]
        assert [row.turn_status for row in after_rows] == ["completed", "completed", "failed", "failed"]
        assert after_rows[2].rag_metadata["execution_started"] is False


async def test_special_finish_rejects_forged_context_and_never_reopens_normal_turn(tmp_path):
    """专用收尾信任调用方结果/三键或放宽普通轮终态时失败。参数：tmp_path；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, actions, client, factory):
        from app.schemas.workflow import TicketConfirmation
        body = await saved_suggestion(chat.chat)
        prepared = await actions.repository.prepare_ticket(TicketConfirmation(**body))
        with pytest.raises(PersistenceUncertain):
            await actions.repository.finish_ticket_action(prepared, ToolResult.success({"ticket_no": "fake", "ticket_type": "售后", "status": "待处理"}))
        await client.post("/api/actions/tickets", json=body)
        result = await chat.chat.get_tool_result(prepared.context)
        forged = prepared.model_copy(update={"context": prepared.context.model_copy(update={"turn_id": body["source_turn_id"]})})
        with pytest.raises((ValueError, PersistenceUncertain)):
            await actions.repository.finish_ticket_action(forged, result)
        original = ToolContext(conversation_id=prepared.context.conversation_id, session_id="s", user_id="demo-user", turn_id="cancelled-chat", user_message="普通问答")
        original = await chat.chat.start_turn(original, original.user_message)
        await chat.chat.mark_turn(original, "cancelled")
        with pytest.raises(PersistenceUncertain):
            await chat.chat.finish_turn(original, "不应重开")


@pytest.mark.parametrize("damage", ["malformed_result", "different_ticket"])
async def test_damaged_saved_result_is_unknown_not_a_client_error_or_receipt(tmp_path, damage):
    """损坏缓存被报为客户端问题或把同描述别票当本动作成功时失败。参数：tmp_path/damage；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        receipt = await client.post("/api/actions/tickets", json=body)
        assert receipt.status_code == 200
        async with factory.begin() as session:
            result_row = await session.scalar(select(Message).where(Message.role == "tool"))
            if damage == "malformed_result":
                result_row.content = "{}"
            else:
                original = await session.get(Ticket, receipt.json()["ticket_no"])
                session.add(Ticket(ticket_no="T-unrelated", conversation_id=original.conversation_id,
                    description=original.description, ticket_type=original.ticket_type))
                altered = ToolResult.model_validate_json(result_row.content)
                altered.data["ticket_no"] = "T-unrelated"
                result_row.content = altered.model_dump_json()
        response = await client.post("/api/actions/tickets", json=body)
        assert response.status_code == 503 and response.json()["detail"]["outcome"] == "unknown"


@pytest.mark.parametrize("damage", ["unfinished", "wrong_question", "wrong_kind"])
async def test_source_snapshot_is_authoritative_and_must_be_completed(tmp_path, damage):
    """未完成来源或保存快照不符仍可建单时失败。参数：tmp_path/damage；返回：None；异常：断言失败。"""
    async with action_runtime(tmp_path) as (chat, _, client, factory):
        body = await saved_suggestion(chat.chat)
        async with factory.begin() as session:
            source = await session.scalar(select(Message).where(Message.role == "assistant"))
            if damage == "unfinished":
                source.turn_status = "running"
            else:
                metadata = source.rag_metadata
                snapshot = dict(metadata["suggested_actions"][0])
                snapshot.update({"question": "不同原文"} if damage == "wrong_question" else {"kind": "handoff", "ticket_type": None})
                source.rag_metadata = {**metadata, "suggested_actions": [snapshot]}
        response = await client.post("/api/actions/tickets", json=body)
        assert response.status_code == 404 and (await records(factory))[2] == []


async def test_default_http_dependency_reuses_real_chat_guard_and_support(tmp_path):
    """线上依赖接错guard或只读工具注册表导致不建单时失败。参数：tmp_path；返回：None；异常：断言失败。"""
    from app.main import create_app
    async with action_runtime(tmp_path) as (chat, actions, _, factory):
        body = await saved_suggestion(chat.chat)
        chat.executor = actions.executor
        app = create_app()
        app.state.chat_service = chat
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            async with chat.guard.hold("s"):
                assert (await client.post("/api/actions/tickets", json=body)).status_code == 409
            response = await client.post("/api/actions/tickets", json=body)
        assert response.status_code == 200 and len((await records(factory))[2]) == 1
