"""CH05非破坏性真实MySQL确认验收：父持资源锁、真实硬退出与新进程恢复同票。

只使用已准备的mewhelp_ch04_test；本模块不请求旧清库fixture、不建表、不接模型或Milvus。
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
import pytest
from sqlalchemy import delete, func, select, text

from app.db.session import create_database
from app.models.customer_service import Conversation, Message, Ticket
from app.repositories.actions import ActionRepository
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.schemas.tools import ToolContext, ToolResult
from app.schemas.workflow import SuggestedAction, TicketConfirmation
from app.services.actions import TicketActionService
from app.services.chat import ConversationGuard
from app.tools.executor import ToolExecutor
from app.tools.registry import build_ticket_registry
from tests.verify_ch04_live import build_config, parser, resource_lock

pytestmark = pytest.mark.mysql
DATABASE = "mewhelp_ch04_test"
LOCK_NAME = "mewhelp:ch04:live:exclusive"


def _config(env_file: Path, report_dir: Path):
    """显式加载指定测试配置，不准备资源。参数：env_file用户配置路径、report_dir临时输出；返回：Settings；异常：配置错误。"""
    return build_config(parser().parse_args(["--test", "--env-file", str(env_file), "--report-dir", str(report_dir)]))


async def _guard_database(factory, owner: int, allowed: list[int]):
    """核实真实库/父锁/外来running。参数：factory工厂、owner锁拥有者、allowed本次会话；返回：None；异常：断言失败，拒绝恢复或写入。"""
    async with factory() as session:
        assert await session.scalar(text("SELECT DATABASE()")) == DATABASE
        assert await session.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": LOCK_NAME}) == owner
        running = list((await session.scalars(select(Message.conversation_id).where(Message.turn_status == "running").distinct())).all())
        assert set(running) <= set(allowed), "存在本次登记之外running，拒绝触碰或全库恢复"


def _service(factory):
    """构造真实动作运行时，不启动图或知识资源。参数：factory独占会话工厂；返回：动作服务；异常：构造错误。"""
    chat = ChatRepository(factory)
    executor = ToolExecutor(build_ticket_registry(), chat, SupportRepository(factory), retry_delay_seconds=0)
    return TicketActionService(ActionRepository(factory), executor, ConversationGuard())


async def _snapshot(factory, body: dict) -> dict:
    """读取精确会话与票据证据。参数：factory工厂、body原确认；返回：无凭据JSON证据；异常：数据库或断言失败。"""
    async with factory() as session:
        conversation = await session.scalar(select(Conversation).where(
            Conversation.session_id == body["session_id"], Conversation.user_id == body["user_id"]))
        assert conversation is not None
        rows = list((await session.scalars(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id))).all())
        tickets = list((await session.scalars(select(Ticket).where(Ticket.conversation_id == conversation.id))).all())
    action_turn = str(uuid5(NAMESPACE_URL, f"mewhelp:ch05:confirm:{body['action_id']}"))
    action_rows = [row for row in rows if row.turn_id == action_turn]
    source = [row for row in rows if row.turn_id == body["source_turn_id"]]
    results = [ToolResult.model_validate_json(row.content).model_dump() for row in action_rows if row.role == "tool"]
    return {"database": DATABASE, "conversation_id": conversation.id, "session_id": conversation.session_id,
        "conversation_status": conversation.status, "source_statuses": [row.turn_status for row in source],
        "action_id": body["action_id"], "action_turn_id": action_turn, "tool_call_id": "ticket:" + body["action_id"],
        "action_statuses": [row.turn_status for row in action_rows], "action_roles": [row.role for row in action_rows],
        "ticket_numbers": [ticket.ticket_no for ticket in tickets], "descriptions": [ticket.description for ticket in tickets],
        "tool_results": results, "receipt_count": sum(bool((row.rag_metadata or {}).get("ticket_receipt")) for row in action_rows)}


@pytest.fixture
async def isolated_action_mysql(tmp_path):
    """父全程持锁并只精确清理登记资源。参数：tmp_path本次目录；返回：工厂/来源构造器/锁证据；异常：目标/归属/数据库错误。"""
    env_file = Path(os.environ["MEWHELP_TEST_ENV_FILE"])
    config = _config(env_file, tmp_path)
    registered = []
    engine, factory = create_database(config, database=DATABASE)
    try:
        async with resource_lock(config, prepare=False) as checked:
            owner = await checked.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": LOCK_NAME})
            assert owner is not None
            await _guard_database(factory, owner, [])
            async with factory() as session:
                isolation = await session.scalar(text("SELECT @@transaction_isolation"))
                version = await session.scalar(text("SELECT VERSION()"))
                before = {model.__tablename__: await session.scalar(select(func.count()).select_from(model)) for model in (Conversation, Message, Ticket)}

            async def source(question="  订单迟迟未收到，请处理  "):
                """登记不存在的唯一会话并保存真实建议。参数：question原文；返回：确认请求；异常：数据库/碰撞错误。"""
                session_id, user_id, turn_id = "ch05-action-" + uuid4().hex, "ch05-user-" + uuid4().hex, str(uuid4())
                await _guard_database(factory, owner, [item[0] for item in registered])
                async with factory.begin() as session:
                    assert await session.scalar(select(Conversation.id).where(Conversation.session_id == session_id)) is None
                    conversation = Conversation(session_id=session_id, user_id=user_id)
                    session.add(conversation)
                    await session.flush()
                    # 准确主键/会话/用户已核验后登记，异常清理也仅限这三者共同匹配。
                    registered.append((conversation.id, session_id, user_id))
                chat = ChatRepository(factory)
                context = await chat.start_turn(ToolContext(session_id=session_id, user_id=user_id, turn_id=turn_id, user_message=question), question)
                action_id = str(uuid5(NAMESPACE_URL, f"mewhelp:ch05:{context.conversation_id}:{turn_id}:create_ticket"))
                suggestion = SuggestedAction(action_id=action_id, kind="create_ticket", source_turn_id=turn_id, question=question, ticket_type="售后")
                await chat.finish_turn(context, "请确认是否创建工单", suggested_actions=[suggestion])
                return {"session_id": session_id, "user_id": user_id, "source_turn_id": turn_id, "action_id": action_id, "confirmed": True}

            try:
                yield {"factory": factory, "source": source, "owner": owner, "env_file": env_file, "registered": registered, "path": tmp_path}
            finally:
                # 全库恢复的保护同样用于清理前核验；不取消或改写外来轮次。
                await _guard_database(factory, owner, [item[0] for item in registered])
                async with factory.begin() as session:
                    for identifier, session_id, user_id in registered:
                        exact = await session.scalar(select(Conversation.id).where(Conversation.id == identifier,
                            Conversation.session_id == session_id, Conversation.user_id == user_id))
                        if exact is None:
                            continue
                        await session.execute(delete(Ticket).where(Ticket.conversation_id == identifier))
                        await session.execute(delete(Message).where(Message.conversation_id == identifier))
                        await session.execute(delete(Conversation).where(Conversation.id == identifier,
                            Conversation.session_id == session_id, Conversation.user_id == user_id))
                async with factory() as session:
                    after = {model.__tablename__: await session.scalar(select(func.count()).select_from(model)) for model in (Conversation, Message, Ticket)}
                evidence = {"database": DATABASE, "version": version, "isolation": isolation, "lock": LOCK_NAME, "owner": owner, "registered": registered, "before": before, "after": after}
                (tmp_path / "cleanup.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
                print("MYSQL_CLEANUP " + json.dumps(evidence, ensure_ascii=False))
                assert after == before
    finally:
        await engine.dispose()


async def _run_child(resource: dict, phase: str, request_file: Path, evidence: Path) -> int:
    """启动真实新解释器且不请求子锁。参数：resource父资源、phase阶段、请求和证据路径；返回：退出码；异常：超时/子进程错误。"""
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.integration.test_ch05_actions",
        "--phase", phase, "--request", str(request_file), "--evidence", str(evidence),
        "--lock-owner", str(resource["owner"]), "--env-file", str(resource["env_file"]),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    # 不打印原始异常（可能有连接串）；必要时仅留无凭据阶段输出用于判定。
    assert process.returncode == (73 if phase == "crash" else 0), f"子进程{phase}异常退出：{process.returncode}，stdout字节数={len(stdout)}，stderr字节数={len(stderr)}"
    return process.returncode


async def test_mysql_hard_exit_recover_and_same_action_receipt(isolated_action_mysql):
    """提交成功后硬退出，新进程真实恢复应保持同票并只补动作回执。参数：隔离资源；返回：None；异常：断言失败。"""
    resource = isolated_action_mysql
    body = await resource["source"]()
    request_file = resource["path"] / "request.json"
    request_file.write_text(json.dumps({"body": body, "allowed": [item[0] for item in resource["registered"]]}), encoding="utf-8")
    first_file, second_file = resource["path"] / "committed.json", resource["path"] / "recovered.json"
    first_exit = await _run_child(resource, "crash", request_file, first_file)
    committed = json.loads(first_file.read_text(encoding="utf-8"))
    assert committed["snapshot"] == await _snapshot(resource["factory"], body)
    assert committed["snapshot"]["action_statuses"] == ["running"] * 3
    await _guard_database(resource["factory"], resource["owner"], [item[0] for item in resource["registered"]])
    second_exit = await _run_child(resource, "recover", request_file, second_file)
    recovered = json.loads(second_file.read_text(encoding="utf-8"))
    assert recovered["cancelled"]["action_statuses"] == ["cancelled"] * 3
    assert recovered["snapshot"]["source_statuses"] == ["completed", "completed"]
    assert recovered["snapshot"]["action_statuses"] == ["completed"] * 4
    assert recovered["snapshot"]["ticket_numbers"] == committed["snapshot"]["ticket_numbers"] == [recovered["receipt"]["ticket_no"]]
    assert recovered["receipt"] == recovered["retry"] and recovered["snapshot"]["receipt_count"] == 1
    assert recovered["snapshot"]["conversation_status"] == "进行中"
    print("MYSQL_HARD_EXIT " + json.dumps({"exit_codes": [first_exit, second_exit], "owner": resource["owner"], "committed": committed, "recovered": recovered}, ensure_ascii=False))


async def test_mysql_double_click_identity_and_original_whitespace(isolated_action_mysql):
    """真实双击只有一票，伪造/跨用户会话不泄漏，原文保留。参数：隔离资源；返回：None；异常：断言失败。"""
    from fastapi import FastAPI
    from app.api.actions import get_ticket_action_service, router
    resource = isolated_action_mysql
    question = " \n请处理旧订单　 "
    body = await resource["source"](question)
    other = await resource["source"]("另一个会话")
    # 绕过进程内guard直接并发prepare，验证MySQL来源行锁本身防重且读取锁后新提交。
    prepared = await asyncio.gather(*(ActionRepository(resource["factory"]).prepare_ticket(TicketConfirmation(**body)) for _ in range(2)))
    assert prepared[0] == prepared[1]
    assert (await _snapshot(resource["factory"], body))["action_roles"] == ["user", "assistant"]
    actions = _service(resource["factory"])
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_ticket_action_service] = lambda: actions
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        pair = await asyncio.gather(*(client.post("/api/actions/tickets", json=body) for _ in range(2)))
        assert sorted(item.status_code for item in pair) == [200, 409]
        repeated = await client.post("/api/actions/tickets", json=body)
        assert repeated.json() == next(item.json() for item in pair if item.status_code == 200)
        before_read = await _snapshot(resource["factory"], body)
        query = {key:value for key,value in body.items() if key != "confirmed"}
        receipts = await asyncio.gather(*(ActionRepository(resource["factory"]).get_ticket_receipt(**query) for _ in range(2)))
        assert [receipt.model_dump() for receipt in receipts] == [repeated.json(), repeated.json()]
        assert await _snapshot(resource["factory"], body) == before_read
        for override in ({"user_id": other["user_id"]}, {"session_id": other["session_id"]},
                         {"user_id": other["user_id"], "session_id": other["session_id"]},
                         {"action_id": other["action_id"]}, {"source_turn_id": other["source_turn_id"]}):
            rejected = await client.post("/api/actions/tickets", json={**body, **override})
            assert rejected.status_code == 404 and rejected.json()["detail"]["code"] == "ACTION_NOT_FOUND"
    snapshot = await _snapshot(resource["factory"], body)
    assert snapshot["descriptions"] == [question] and snapshot["receipt_count"] == 1
    assert snapshot["conversation_status"] == "进行中" and snapshot["source_statuses"] == ["completed", "completed"]
    assert (await _snapshot(resource["factory"], other))["ticket_numbers"] == []
    print("MYSQL_DOUBLE_CLICK " + json.dumps(snapshot, ensure_ascii=False))


async def _child(args):
    """子进程只核实父锁并跑真实提交/恢复。参数：args阶段及身份路径；返回：None；异常：断言/数据库错误；crash阶段硬退出73。"""
    config = _config(args.env_file, args.evidence.parent)
    engine, factory = create_database(config, database=DATABASE)
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    body, allowed = payload["body"], payload["allowed"]
    try:
        await _guard_database(factory, args.lock_owner, allowed)
        actions = _service(factory)
        if args.phase == "crash":

            async def crash_after_commit(action, result):
                """只在真实工具提交且独立读取确认后硬退出。参数：action/result真实对象；返回：无；异常：断言或退出73。"""
                await _guard_database(factory, args.lock_owner, allowed)
                snapshot = await _snapshot(factory, body)
                assert len(snapshot["ticket_numbers"]) == 1 and snapshot["tool_results"] == [result.model_dump()]
                assert snapshot["ticket_numbers"] == [result.data["ticket_no"]]
                args.evidence.write_text(json.dumps({"phase": "committed_before_finish", "snapshot": snapshot}, ensure_ascii=False, indent=2), encoding="utf-8")
                print("COMMITTED_BEFORE_FINISH", flush=True)
                os._exit(73)

            actions.repository.finish_ticket_action = crash_after_commit
            await actions.confirm_ticket(TicketConfirmation(**body))
            raise AssertionError("硬退出注入未执行")
        await _guard_database(factory, args.lock_owner, allowed)
        recovered_count = await actions.executor.chat.recover_interrupted_turns()
        cancelled = await _snapshot(factory, body)
        assert recovered_count == 1 and cancelled["action_statuses"] == ["cancelled"] * 3
        pending = await actions.repository.get_ticket_receipt(**{key:value for key,value in body.items() if key != "confirmed"})
        assert pending.audit_sync_pending is True and cancelled["ticket_numbers"] == [pending.ticket_no]
        assert await _snapshot(factory, body) == cancelled
        receipt = await actions.confirm_ticket(TicketConfirmation(**body))
        retry = await actions.confirm_ticket(TicketConfirmation(**body))
        snapshot = await _snapshot(factory, body)
        args.evidence.write_text(json.dumps({"phase": "recovered_and_reconfirmed", "recovered_count": recovered_count,
            "cancelled": cancelled, "receipt": receipt.model_dump(), "retry": retry.model_dump(), "snapshot": snapshot}, ensure_ascii=False, indent=2), encoding="utf-8")
        await _guard_database(factory, args.lock_owner, allowed)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    child_parser = argparse.ArgumentParser()
    child_parser.add_argument("--phase", choices=["crash", "recover"], required=True)
    child_parser.add_argument("--request", type=Path, required=True)
    child_parser.add_argument("--evidence", type=Path, required=True)
    child_parser.add_argument("--lock-owner", type=int, required=True)
    child_parser.add_argument("--env-file", type=Path, required=True)
    asyncio.run(_child(child_parser.parse_args()))
