"""本地退款申请的独立快照、严格请求和同键回执测试。"""

from uuid import NAMESPACE_URL, uuid5
import pytest
from sqlalchemy import select, text
from app.schemas import workflow
from app.models import customer_service
from app.repositories.actions import ActionRepository
from app.repositories.chat import PersistenceUncertain
from tests.workflow.service_fakes import service_runtime, collect
from tests.workflow.test_ch06_graph import order_result


def submission(action, **override):
    """用保存动作构造确认。参数：action原快照、override覆盖；返回：严格请求；异常：请求非法。"""
    return workflow.RefundSubmission(**{"session_id": "s", "user_id": "demo-user", "source_turn_id": action["source_turn_id"],
        "action_id": action["action_id"], "reason_code": "quality_issue", "confirmed": True, **override})


async def test_refund_receipt_is_idempotent_and_independent(tmp_path):
    """同键同原因得到同编号，换原因409且工单表不写入。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["需核实政策，可提交待处理申请。"]]) as (service, deps, factory):
        events = await collect(service, "订单1001退款")
        action = events[-2].data["refund_action"]
        request = submission(action)
        repository = ActionRepository(factory)
        first = await repository.submit_refund(request)
        retry = await repository.submit_refund(request)
        assert first == retry and first.status == "待处理"
        assert first.refund_no == "RF" + uuid5(NAMESPACE_URL, f"mewhelp:ch06:refund:{action['action_id']}").hex.upper()
        with pytest.raises(workflow.OrderSelectionConflict):
            await repository.submit_refund(submission(action, reason_code="other"))
        async with factory() as session:
            rows = (await session.scalars(select(customer_service.RefundApplication))).all()
            assert len(rows) == 1 and rows[0].original_question == "订单1001退款"
            assert rows[0].reason_code == "quality_issue"
        current = await service.get_current_actions("s", "demo-user")
        assert current.refund_action.action_id == action["action_id"] and current.refund_receipt == first


async def test_refund_snapshot_rolls_back_with_answer(tmp_path):
    """完成事务失败不留下退款动作，保持原轮未完成。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["待处理申请"]]) as (service, deps, factory):
        async with factory.begin() as session:
            await session.execute(text("CREATE TRIGGER refuse_answer BEFORE INSERT ON messages WHEN NEW.role = 'assistant' AND NEW.content IS NOT NULL BEGIN SELECT RAISE(ABORT, 'reject answer'); END"))
        events = await collect(service, "订单1001退款")
        assert events[-1].event == "error"
        current = await service.get_current_actions("s", "demo-user")
        assert current.refund_action is None and current.answer is None


async def test_finish_retry_compares_refund_snapshot(tmp_path):
    """已完成重试必须比较退款动作，不能移除或换订单。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["待处理申请"]]) as (service, deps, _):
        events = await collect(service, "订单1001退款")
        metadata = events[-2].data
        ctx = await deps.chat.get_turn_context("s", "demo-user", metadata["turn_id"])
        action = workflow.RefundAction.model_validate(metadata["refund_action"])
        answer = "".join(event.data["delta"] for event in events if event.event == "delta")
        from app.schemas.flywheel import RetrievalSnapshot
        row = await deps.chat.get_workflow_turn("s", "demo-user", metadata["turn_id"])
        snapshot = RetrievalSnapshot.model_validate(row["retrieval_snapshot"])
        assert await deps.chat.finish_turn(ctx, answer, citations=[], suggested_actions=[],
            workflow_metadata=metadata["workflow"], refund_action=action, retrieval_snapshot=snapshot, pool_signals=[]) == metadata
        with pytest.raises(PersistenceUncertain):
            await deps.chat.finish_turn(ctx, answer, citations=[], suggested_actions=[], workflow_metadata=metadata["workflow"],
                retrieval_snapshot=snapshot, pool_signals=[])


@pytest.mark.parametrize("override", [{"confirmed": False}, {"confirmed": 1}, {"amount": 1}, {"order_id": "1002"},
    {"reason_code": "custom"}, {"status": "已退款"}])
def test_refund_submission_rejects_client_business_fields(override):
    """客户端不能审批或改订单/金额。参数：override非法输入；返回：None；异常：断言失败。"""
    action = {"action_id": str(uuid5(NAMESPACE_URL, "test")), "source_turn_id": "t"}
    with pytest.raises(ValueError):
        submission(action, **override)


async def test_unverified_order_never_creates_refund_action(tmp_path):
    """本人无订单时不能出现退款申请动作。参数：tmp_path目录；返回：None；异常：断言失败。"""
    from app.schemas.chat import ChatRequest
    async with service_runtime(tmp_path, intents=["退款退货"]) as (service, _, _):
        events = [event async for event in service.stream_turn(ChatRequest(session_id="s", user_id="nobody", message="我要退款"))]
        assert events[-1].event == "done" and events[-2].data.get("refund_action") is None


async def test_forged_snapshot_is_rejected_and_guard_is_shared(tmp_path):
    """篡改保存订单或身份不写申请，共享会话锁阻止并行提交。参数：tmp_path目录；返回：None；异常：断言失败。"""
    from app.repositories.actions import ActionNotFound
    from app.services.actions import RefundActionService, TicketActionError
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["待处理"]]) as (service, _, factory):
        events = await collect(service, "订单1001退款")
        action = events[-2].data["refund_action"]
        request = submission(action)
        refund_service = RefundActionService(ActionRepository(factory), service.guard)
        async with service.guard.hold("s"):
            with pytest.raises(TicketActionError) as error:
                await refund_service.submit_refund(request)
            assert error.value.status_code == 409
        with pytest.raises(ActionNotFound):
            await ActionRepository(factory).submit_refund(submission(action, user_id="demo-other-user"))
        async with factory.begin() as session:
            row = await session.scalar(select(customer_service.Message).where(customer_service.Message.turn_status == "completed",
                customer_service.Message.role == "assistant", customer_service.Message.tool_calls.is_(None)))
            row.rag_metadata = {**row.rag_metadata, "refund_action": {**action, "order_id": "1002"}}
        with pytest.raises(ActionNotFound):
            await ActionRepository(factory).submit_refund(request)
        async with factory() as session:
            assert not (await session.scalars(select(customer_service.RefundApplication))).all()


async def test_refund_http_reason_catalog_and_stable_receipt(tmp_path):
    """真实API提供单一目录和相同申请回执，工单保持独立。参数：tmp_path目录；返回：None；异常：断言失败。"""
    from tests.workflow.test_ticket_actions import action_runtime
    async with action_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["待核实"]]) as (service, _, client, factory):
        events = await collect(service, "订单1001退款")
        request = submission(events[-2].data["refund_action"]).model_dump()
        reasons = await client.get("/api/actions/refund-reasons")
        assert reasons.status_code == 200 and len(reasons.json()) == 7
        assert {item["code"]: item["label"] for item in reasons.json()} == workflow.REFUND_REASONS
        first = await client.post("/api/actions/refunds", json=request)
        second = await client.post("/api/actions/refunds", json=request)
        assert first.status_code == second.status_code == 200 and first.json() == second.json()
        assert first.json()["status"] == "待处理"
        assert (await client.post("/api/actions/refunds", json={**request, "reason_code": "other"})).status_code == 409
        assert (await client.post("/api/actions/refunds", json={**request, "confirmed": False})).status_code == 422
        assert (await client.post("/api/actions/refunds", json={**request, "user_id": "other"})).status_code == 404
        async with factory() as session:
            assert not (await session.scalars(select(customer_service.Ticket))).all()
