"""退款对象回归：真实服务、图和仓储不能把附带查询变成退款目标。"""

import json
import socket
import tempfile

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.models.customer_service import Message, RefundApplication
from app.repositories.actions import ActionRepository
from app.schemas.workflow import AgentDecision, OrderSelection, RefundSubmission
from app.services.actions import RefundActionService
from tests.workflow.ch08_ticket_fakes import ticket_runtime
from tests.workflow.service_fakes import collect


@pytest_asyncio.fixture(autouse=True)
async def isolated_refund_target_runtime(tmp_path, monkeypatch):
    """将临时制品限制在本例目录并拒绝真实网络。

    参数：tmp_path 为独占临时目录；monkeypatch 为还原补丁的夹具。
    返回：None；异常：任何真实连接尝试均抛出 AssertionError。
    """
    def deny_network(*args, **kwargs):
        """阻止网络连接。参数：原套接字参数；返回：无；异常：AssertionError。"""
        raise AssertionError("退款对象回归禁止连接真实外部服务")

    # 异步夹具先等待 Windows 事件循环建立内部唤醒管道，再封禁业务网络。
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def query_decisions(orders):
    """按顺序构造每次仅查询一单的有限模型输出。

    参数：orders 为完整订单号列表；返回：决定列表；异常：模型字段不合法时抛出校验错误。
    """
    return [AgentDecision(kind="tools", tool_calls=[{
        "id": f"query-{order}", "name": "query_order", "args": {"order_id": order},
    }]) for order in orders] + [AgentDecision(kind="answer")]


def select_first_order(action):
    """使用服务端保存的卡片身份明确选择 1001。

    参数：action 为当前选单卡；返回：OrderSelection；异常：字段不合法时抛出校验错误。
    """
    return OrderSelection(session_id="s", user_id="demo-user", source_turn_id=action.source_turn_id,
                          action_id=action.action_id, order_id="1001")


async def assert_saved_refund_target(service, factory):
    """经真实确认服务提交，并核对持久化的申请对象。

    参数：service 为真实聊天服务；factory 为独占 SQLite 会话工厂。
    返回：None；异常：服务、数据库或行为断言失败。
    """
    current = await service.get_current_actions("s", "demo-user")
    action = current.refund_action
    assert action is not None
    receipt = await RefundActionService(ActionRepository(factory), service.guard).submit_refund(
        RefundSubmission(session_id="s", user_id="demo-user", source_turn_id=action.source_turn_id,
                         action_id=action.action_id, confirmed=True, reason_code="quality_issue"))
    async with factory() as session:
        applications = (await session.scalars(select(RefundApplication))).all()
        assert [row.order_id for row in applications] == ["1001"]
    assert action.order_id == receipt.order_id == "1001"


@pytest.mark.parametrize("orders", [["1001"], ["1001", "1002"], ["1002", "1001"]])
async def test_other_order_query_cannot_replace_refund_target(tmp_path, orders):
    """附带查询不得产生被明确排除的 1002 申请，允许先选单再提交。

    参数：tmp_path 为隔离目录；orders 为单次独立查询的顺序。
    返回：None；异常：服务、数据库或行为断言失败。
    """
    async with ticket_runtime(tmp_path, intents=["退款退货"], decisions=query_decisions(orders)) as (service, deps, factory, _):
        events = await collect(service, "订单1001想退款，订单1002不要退款，只查一下它的状态")
        assert events[-1].event in {"done", "select_order"}
        async with factory() as session:
            results = (await session.scalars(select(Message).where(Message.role == "tool").order_by(Message.id))).all()
        # 核对真实工具流水，防止通过漏执行第二次查询让目标断言虚假通过。
        assert [(row.tool_call_id, json.loads(row.content)["data"]["order_id"]) for row in results] == [
            (f"query-{order}", order) for order in orders]
        current = await service.get_current_actions("s", "demo-user")
        if current.order_selection is not None:
            assert current.refund_action is None
            resumed = [event async for event in service.stream_resume(select_first_order(current.order_selection))]
            assert resumed[-1].event == "done"
        await assert_saved_refund_target(service, factory)
        saved = await deps.chat.get_workflow_turn("s", "demo-user")
        assert saved["turn_status"] == "completed"


async def test_two_order_queries_require_selection_and_survive_accepted_resume(tmp_path):
    """两笔退款对象有歧义时先选单，已接受的选择跨运行时重建仍绑定原轮。

    参数：tmp_path 为数据库与检查点的独占目录。
    返回：None；异常：服务、数据库或行为断言失败。
    """
    async with ticket_runtime(tmp_path, intents=["退款退货"],
                              decisions=query_decisions(["1001", "1002"])) as (service, _, factory, _):
        events = await collect(service, "订单1001和订单1002能退款吗")
        assert events[-1].event == "select_order"
        current = await service.get_current_actions("s", "demo-user")
        assert current.refund_action is None
        request = select_first_order(current.order_selection)
        # 模拟接受已落库、图尚未恢复时运行时退出；重建后不能重复分类或换办理对象。
        await ActionRepository(factory).accept_order_selection(request)

    async with ticket_runtime(tmp_path, intents=[]) as (service, deps, factory, _):
        current = await service.get_current_actions("s", "demo-user")
        assert current.resume_required and current.accepted_order_id == "1001"
        events = [event async for event in service.stream_resume(request)]
        assert events[-1].event == "done"
        await assert_saved_refund_target(service, factory)
        turns = await deps.chat.load_turns(1)
        assert len(turns) == 1 and turns[0].turn_id == request.source_turn_id
        assert turns[0].turn_status == "completed"
        assert not any(row["stage"] == "classify" for row in deps.model.inputs)


async def test_query_limit_cannot_create_refund_for_undetermined_target(tmp_path):
    """工具决定达到上限而提前结束时，附带查询也不能直接变成退款卡。

    参数：tmp_path 为隔离目录；返回：None；异常：服务、数据库或行为断言失败。
    """
    async with ticket_runtime(tmp_path, intents=["退款退货"], workflow_max_decisions=2,
                              decisions=query_decisions(["1001", "1002"])) as (service, _, factory, _):
        events = await collect(service, "订单1001想退款，订单1002不要退款，只查一下它的状态")
        assert events[-1].event == "done"
        current = await service.get_current_actions("s", "demo-user")
        assert current.refund_action is None or current.refund_action.order_id == "1001"
        async with factory() as session:
            assert not (await session.scalars(select(RefundApplication))).all()


async def test_single_order_target_reuses_current_dynamic_query(tmp_path):
    """单一目标保留动态工具配对并复用结果，不能补查固定编号来通过退款证据检查。

    参数：tmp_path 为隔离目录；返回：None；异常：服务、数据库或行为断言失败。
    """
    async with ticket_runtime(tmp_path, intents=["退款退货"],
                              decisions=query_decisions(["1001"])) as (service, _, factory, _):
        events = await collect(service, "订单1001想退款")
        assert events[-1].event == "done"
        await assert_saved_refund_target(service, factory)
        async with factory() as session:
            messages = (await session.scalars(select(Message).order_by(Message.id))).all()
        calls = [call for row in messages for call in row.tool_calls or []]
        results = [row for row in messages if row.role == "tool"]
        assert [(call["id"], call["args"]) for call in calls] == [("query-1001", {"order_id": "1001"})]
        assert [(row.tool_call_id, json.loads(row.content)["data"]["order_id"]) for row in results] == [
            ("query-1001", "1001")]
