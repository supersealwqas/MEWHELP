"""增量历史完整配对及业务已提交、checkpoint失败后的新轮对账验证。"""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.schemas.tools import ToolContext, ToolResult
from app.schemas.workflow import AgentDecision
from tests.workflow.service_fakes import collect, service_runtime


async def test_history_delta_reads_complete_affected_turn_and_refreshes_terminal(tmp_path):
    """增量仅返回新消息导致工具丢配对，或不刷新无新消息终态时失败。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (_, deps, _):
        context = await deps.chat.start_turn(ToolContext(session_id="s", user_id="u", turn_id="old", user_message="查物流"), "查物流")
        first = await deps.chat.load_history_delta(context.conversation_id, 0)
        await deps.chat.save_tool_calls(context, [{"id": "c1", "name": "query_order", "args": {"order_id": "1001"}}])
        await deps.chat.save_tool_result(context.model_copy(update={"tool_call_id": "c1"}), ToolResult.success({"order_id": "1001"}))
        delta = await deps.chat.load_history_delta(context.conversation_id, first.last_message_id)
        assert [item.role for item in delta.turns[0].messages] == ["user", "assistant", "tool"]
        assert delta.last_message_id > first.last_message_id
        await deps.chat.mark_turn(context, "cancelled")
        refreshed = await deps.chat.load_history_delta(context.conversation_id, delta.last_message_id, refresh_turn_id="old")
        assert refreshed.turns[0].turn_status == "cancelled"
        assert refreshed.last_message_id == delta.last_message_id


async def test_next_input_merges_history_without_full_reload_and_understands_1001(tmp_path, monkeypatch):
    """后续全量查历史、丢已追问上下文或带入旧意图时失败。参数：tmp_path/monkeypatch；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["物流", "物流", "闲聊"],
            decisions=[AgentDecision(kind="clarify"), AgentDecision(kind="answer")], answers=[["请给订单号"], ["正在查询1001"]]) as (service, deps, _):
        cursors = []
        original = deps.chat.load_history_delta

        async def record(conversation_id, after_message_id, **kwargs):
            """记录真实增量查询边界。参数：仓储契约；返回：真实增量；异常：数据库错误。"""
            cursors.append(after_message_id)
            return await original(conversation_id, after_message_id, **kwargs)

        monkeypatch.setattr(deps.chat, "load_history_delta", record)
        assert (await collect(service, "查物流"))[-1].event == "done"
        assert (await collect(service, "1001"))[-1].event == "done"
        classify = [item for item in deps.model.inputs if item["stage"] == "classify"][-1]
        assert classify["messages"] == [{"role": "user", "content": "查物流"}, {"role": "assistant", "content": "请给订单号"}, {"role": "user", "content": "1001"}]
        assert (await collect(service, "谢谢"))[-1].event == "done"
        assert cursors[0] == 0 and all(cursor > 0 for cursor in cursors[1:])
        state = (await service.graph.aget_state({"configurable": {"thread_id": "s"}})).values
        assert state["intent"] == "闲聊" and state["pending_calls"] == state["suggested_actions"] == []
        assert len(state["history_turns"]) == 2


async def test_completed_database_failed_checkpoint_reopens_without_replaying_work(tmp_path, monkeypatch):
    """SQLite写失败误发DONE或新轮重放旧业务时失败。参数：tmp_path/monkeypatch；返回：None；异常：断言失败。"""
    original = AsyncSqliteSaver.aput

    async def fail_final(saver, config, checkpoint, metadata, new_versions):
        """仅拒绝最终checkpoint。参数：官方保存契约；返回：保存结果；异常：受控OSError。"""
        if checkpoint["channel_values"].get("completion_metadata"):
            raise OSError("磁盘失败")
        return await original(saver, config, checkpoint, metadata, new_versions)

    monkeypatch.setattr(AsyncSqliteSaver, "aput", fail_final)
    async with service_runtime(tmp_path, intents=["商品咨询"]) as (service, deps, _):
        events = await collect(service, "旧问题")
        assert events[-1].event == "error" and all(event.event != "done" for event in events)
        turns = await deps.chat.load_turns(1)
        assert turns[0].turn_status == "completed"
    monkeypatch.setattr(AsyncSqliteSaver, "aput", original)
    async with service_runtime(tmp_path, intents=["闲聊"]) as (service, deps, factory):
        assert await deps.chat.recover_interrupted_turns() == 0
        events = await collect(service, "你好")
        assert events[-1].event == "done" and deps.executor.calls == []
        state = (await service.graph.aget_state({"configurable": {"thread_id": "s"}})).values
        assert state["history_turns"][0]["turn_status"] == "completed"
        from sqlalchemy import select
        from app.models.rag_quality import LowConfidenceQuestion
        async with factory() as session:
            assert len((await session.scalars(select(LowConfidenceQuestion))).all()) == 1
