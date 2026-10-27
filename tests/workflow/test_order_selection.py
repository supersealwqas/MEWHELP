"""真实SQLite原轮选单服务测试；事件屏障验证等待锁释放与旧流收尾。"""

import asyncio
from contextlib import aclosing
import pytest
from app.repositories.actions import ActionRepository
from app.schemas.chat import ChatRequest
from app.schemas.workflow import OrderSelection
from app.workflow.state import checkpoint_thread_id
from tests.workflow.service_fakes import service_runtime, collect
from tests.workflow.test_ch06_graph import order_result


def resume_request(action, order_id="1001"):
    """冻结卡片身份构造请求。参数：action卡片、order_id本人订单；返回：严格请求；异常：校验错误。"""
    return OrderSelection(session_id="s", user_id="demo-user", source_turn_id=action["source_turn_id"],
                          action_id=action["action_id"], order_id=order_id)


async def test_waiting_lock_released_before_event_and_resume_keeps_original_turn(tmp_path):
    """选单帧可见前数据库已提交、图已关闭、会话锁已释放。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["签收时间未知，需核实政策。"]]) as (service, deps, _):
        stream = service.stream_turn(ChatRequest(session_id="s", message="我要退款"))
        async with aclosing(stream):
            events = []
            async for event in stream:
                events.append(event)
                if event.event == "select_order":
                    action = event.data
                    async with service.guard.hold("s"):
                        current = await service.get_current_actions("s", "demo-user")
                        assert current.turn_status == "awaiting_input"
                    break
            assert not any(event.event in {"done", "error"} for event in events)
            snapshot = await service.graph.aget_state({"configurable": {"thread_id": checkpoint_thread_id(1)}})
            records = snapshot.values["budget"]["records"]
            resumed = [event async for event in service.stream_resume(resume_request(action))]
            assert resumed[-1].event == "done"
            # 旧等待生成器此时才关闭，不能取消已完成的原轮。
        turns = await deps.chat.load_turns(1)
        assert len(turns) == 1 and turns[0].turn_status == "completed"
        assert turns[0].turn_id == action["source_turn_id"]
        final = await service.graph.aget_state({"configurable": {"thread_id": checkpoint_thread_id(1)}})
        assert final.values["budget"]["records"][:len(records)] == records
        assert sum(row["stage"] == "classify" for row in deps.model.inputs) == 1
        replay = [event async for event in service.stream_resume(resume_request(action))]
        assert replay[-1].event == "done" and not deps.model.answers


async def test_new_question_wins_before_late_select_event(tmp_path):
    """网络暂缓旧选单帧时新问可开始，晚到卡片续跑被拒。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货", "闲聊"]) as (service, deps, _):
        ready, release = asyncio.Event(), asyncio.Event()
        action = {}

        async def delayed_send():
            """在网络发送边界暂停。参数：无；返回：None；异常：事件等待或服务错误。"""
            async with aclosing(service.stream_turn(ChatRequest(session_id="s", message="我要退款"))) as stream:
                async for event in stream:
                    if event.event == "select_order":
                        action.update(event.data)
                        ready.set()
                        await release.wait()

        task = asyncio.create_task(delayed_send())
        await asyncio.wait_for(ready.wait(), 3)
        try:
            assert (await collect(service, "你好"))[-1].event == "done"
            refused = [event async for event in service.stream_resume(resume_request(action))]
            assert refused[-1].event == "error" and refused[-1].data["status_code"] == 409
        finally:
            release.set()
            await task
        turns = await deps.chat.load_turns(1)
        assert [turn.turn_status for turn in turns] == ["cancelled", "completed"]


async def test_new_question_cannot_overwrite_accepted_pending_checkpoint(tmp_path):
    """接受事务后崩溃仍可同键继续，新问不能覆盖原checkpoint。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["需核实退货政策。"]]) as (service, deps, factory):
        waiting = await collect(service, "我要退款")
        assert waiting[-1].event == "select_order"
        request = resume_request(waiting[-1].data)
        await ActionRepository(factory).accept_order_selection(request)
        current = await service.get_current_actions("s", "demo-user")
        assert current.resume_required and current.accepted_order_id == "1001"
        refused = await collect(service, "新问题")
        assert refused[-1].event == "error" and refused[-1].data["status_code"] == 409
        assert len(await deps.chat.load_turns(1)) == 1
        assert [event async for event in service.stream_resume(request)][-1].event == "done"


async def test_resume_wins_and_duplicate_returns_accepted(tmp_path):
    """续跑占锁时同键只回状态，新问不能插入。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["完成"]]) as (service, deps, _):
        waiting = await collect(service, "我要退款")
        assert waiting[-1].event == "select_order"
        request = resume_request(waiting[-1].data)
        entered, release = asyncio.Event(), asyncio.Event()
        original = deps.query_expander.expand

        async def blocked_expand(*args, **kwargs):
            """卡住已接受的扩写阶段。参数：原参数；返回：扩写；异常：原实现错误。"""
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        deps.query_expander.expand = blocked_expand

        async def consume():
            """消费续跑。参数：无；返回：事件；异常：服务错误。"""
            return [event async for event in service.stream_resume(request)]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 3)
        try:
            duplicate = [event async for event in service.stream_resume(request)]
            assert [event.event for event in duplicate] == ["action_status"]
            assert (await collect(service, "新问题"))[-1].event == "error"
        finally:
            release.set()
            assert (await task)[-1].event == "done"
        assert len(await deps.chat.load_turns(1)) == 1


async def test_wait_save_failure_keeps_checkpoint_for_reconciliation(tmp_path):
    """跨库暂停已保存但MySQL失败时，不误取消原轮。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"]) as (service, deps, _):
        async def fail(*args):
            """注入等待提交失败。参数：原参数；返回：无；异常：ConnectionError。"""
            raise ConnectionError("save failed")
        deps.chat.save_order_wait = fail
        events = await collect(service, "我要退款")
        assert events[-1].event == "error" and not any(event.event == "select_order" for event in events)
        assert (await deps.chat.load_turns(1))[0].turn_status == "running"
        state = await service.graph.aget_state({"configurable": {"thread_id": checkpoint_thread_id(1)}})
        assert state.tasks[0].interrupts


async def test_failed_accept_never_drives_command(tmp_path, monkeypatch):
    """接受事务失败必须保持等待，不调用订单或扩写。参数：tmp_path/monkeypatch；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"]) as (service, deps, _):
        action = (await collect(service, "我要退款"))[-1].data

        async def fail(*args):
            """模拟接受提交前故障。参数：原参数；返回：无；异常：ConnectionError。"""
            raise ConnectionError("failed before commit")

        monkeypatch.setattr(ActionRepository, "accept_order_selection", fail)
        events = [event async for event in service.stream_resume(resume_request(action))]
        assert events[-1].event == "error" and not deps.executor.calls
        assert (await deps.chat.load_turns(1))[0].turn_status == "awaiting_input"


async def test_workflow_stage_events_are_ordered_and_sanitized(tmp_path):
    """阶段按执行顺序公开，伪造身份和内部字段不流出。参数：tmp_path目录；返回：None；异常：断言失败。"""
    from langgraph.config import get_stream_writer
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["待核实"]]) as (service, deps, _):
        original = deps.model.understand

        async def understand(*args):
            """额外注入内部字段。参数：原模型参数；返回：理解；异常：模型错误。"""
            get_stream_writer()({"event": "workflow_stage", "data": {"stage": "understand", "status": "completed",
                "source_turn_id": "fake", "failure": "SECRET", "raw_json": "SECRET", "resolved_query": "可见问题"}})
            return await original(*args)

        deps.model.understand = understand
        events = await collect(service, "订单1001退款")
        running = [event.data["stage"] for event in events if event.event == "workflow_stage" and event.data["status"] == "running"]
        assert running == ["understand", "classify", "order", "expand", "policy", "answer"]
        assert "SECRET" not in str(events) and "fake" not in str(events)
        assert events[-1].event == "done"


async def test_resume_http_contract_and_current_identity(tmp_path):
    """HTTP选单流无DONE，非法/跨身份/旧卡分别被拒，完成回放。参数：tmp_path目录；返回：None；异常：断言失败。"""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.chat import router as chat_router, get_chat_service
    from app.api.actions import router as actions_router
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()], answers=[["待核实"]]) as (service, _, _):
        application = FastAPI()
        application.include_router(chat_router)
        application.include_router(actions_router)
        application.dependency_overrides[get_chat_service] = lambda: service
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as client:
            response = await client.post("/api/chat", json={"session_id": "s", "message": "我要退款"})
            assert "event: select_order" in response.text and "[DONE]" not in response.text
            current = (await client.get("/api/actions/current", params={"session_id": "s"})).json()
            request = resume_request(current["order_selection"]).model_dump()
            assert (await client.get("/api/actions/current", params={"session_id": "s", "user_id": "demo-other-user"})).status_code == 404
            assert (await client.post("/api/chat/resume", json={**request, "budget": {}})).status_code == 422
            assert (await client.post("/api/chat/resume", json={**request, "user_id": "demo-other-user"})).status_code == 404
            completed = await client.post("/api/chat/resume", json=request)
            assert completed.status_code == 200 and "[DONE]" in completed.text
            assert (await client.post("/api/chat/resume", json={**request, "order_id": "1002"})).status_code == 409
            replay = await client.post("/api/chat/resume", json=request)
            assert "[DONE]" in replay.text and "待核实" in replay.text
