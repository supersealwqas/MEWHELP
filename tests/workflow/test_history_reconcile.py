"""增量历史完整配对及业务已提交、checkpoint失败后的新轮对账验证。"""

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.schemas.tools import ToolContext, ToolResult
from app.schemas.workflow import AgentDecision
from tests.workflow.service_fakes import collect, service_runtime


@pytest.mark.parametrize("damage", ["conversation_id", "session_id", "user_id", "missing_conversation", "missing_session", "missing_user", "missing_context", "null_context"])
async def test_untrusted_checkpoint_cannot_supply_history_or_cursor(tmp_path, monkeypatch, damage):
    """旧身份不符仍导入历史/游标/刷新轮时失败。参数：tmp_path目录、monkeypatch记录边界、damage旧身份变体；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (service, deps, _):
        old = await deps.chat.start_turn(ToolContext(session_id="s", user_id="u", turn_id="db-old", user_message="当前会话的旧问题"), "当前会话的旧问题")
        await deps.chat.finish_turn(old, "数据库可信回复")
        current = await deps.chat.start_turn(ToolContext(session_id="s", user_id="u", turn_id="current", user_message="新问题"), "新问题")
        identity = old.model_dump(mode="json")
        identity["turn_id"] = "foreign-unfinished"
        if damage.startswith("missing_"):
            identity.pop({"missing_conversation": "conversation_id", "missing_session": "session_id", "missing_user": "user_id"}.get(damage, "turn_id"), None)
        elif damage in {"conversation_id", "session_id", "user_id"}:
            identity[damage] = 7 if damage == "conversation_id" else "foreign"
        previous = {"context": identity, "audit_cursor": 1000, "completion_metadata": {}, "history_turns": [{
            "turn_id": "foreign-turn", "turn_status": "completed", "messages": [{"role": "user", "content": "其他身份的秘密"}]}]}
        if damage == "missing_context":
            previous.pop("context")
        elif damage == "null_context":
            previous["context"] = None
        config = {"configurable": {"thread_id": "mewhelp:ch05:1"}}
        await service.graph.aupdate_state(config, previous, as_node="log_turn")
        observed = []
        original = deps.chat.load_history_delta

        async def record(conversation_id, after_message_id, **kwargs):
            """保留真实仓储，仅记录游标和刷新轮。参数：增量仓储契约；返回：真实增量；异常：数据库错误。"""
            observed.append((conversation_id, after_message_id, kwargs.get("refresh_turn_id")))
            return await original(conversation_id, after_message_id, **kwargs)

        monkeypatch.setattr(deps.chat, "load_history_delta", record)
        state = await service._initial_state(current, config)
        assert [item["turn_id"] for item in state["history_turns"]] == ["db-old"]
        assert state["history_turns"][0]["messages"][-1]["content"] == "数据库可信回复"
        assert observed == [(1, 0, None)] and 0 < state["audit_cursor"] < 1000
        assert state["context"]["conversation_id"] == 1 and state["pending_calls"] == []


async def test_service_checkpoint_uses_trusted_id_and_reopens_isolated_history(tmp_path):
    """客户端键决定checkpoint、同文件重开丢历史或跨会话导入时失败。参数：tmp_path独占目录；返回：None；异常：断言失败。"""
    first_config = {"configurable": {"thread_id": "mewhelp:ch05:1"}}
    second_config = {"configurable": {"thread_id": "mewhelp:ch05:2"}}
    # 客户端字符串刻意等于另一个会话的可信键，证明不能直接当作SQLite键。
    async with service_runtime(tmp_path, intents=["物流", "闲聊"], decisions=[AgentDecision(kind="clarify")], answers=[["请给订单号"]]) as (service, _, _):
        first = await collect(service, "查物流", session="mewhelp:ch05:2")
        second = await collect(service, "你好", session="another-client-key")
        assert first[-1].event == second[-1].event == "done"
        saved = (await service.graph.aget_state(first_config)).values
        other = (await service.graph.aget_state(second_config)).values
        assert saved.get("raw_query") == "查物流" and saved["context"]["conversation_id"] == 1
        assert other["raw_query"] == "你好" and other["context"]["conversation_id"] == 2
        assert (await service.graph.aget_state({"configurable": {"thread_id": "another-client-key"}})).values == {}
    async with service_runtime(tmp_path, intents=["物流"], decisions=[AgentDecision(kind="answer")], answers=[["正在查询1001"]]) as (service, deps, _):
        assert (await service.graph.aget_state(first_config)).values["context"]["turn_id"] == first[-2].data["turn_id"]
        continued = await collect(service, "1001", session="mewhelp:ch05:2")
        assert continued[-1].event == "done"
        classify = next(item for item in deps.model.inputs if item["stage"] == "classify")
        assert classify["messages"][:-1] == [{"role": "user", "content": "查物流"}, {"role": "assistant", "content": "请给订单号"}, {"role": "user", "content": "1001"}]
        assert classify["messages"][-1]["content"].startswith("上下文资料")
        assert (await service.graph.aget_state(second_config)).values["context"]["turn_id"] == second[-2].data["turn_id"]


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
        assert classify["messages"][:-1] == [{"role": "user", "content": "查物流"}, {"role": "assistant", "content": "请给订单号"}, {"role": "user", "content": "1001"}]
        assert classify["messages"][-1]["content"].startswith("上下文资料")
        assert (await collect(service, "谢谢"))[-1].event == "done"
        assert cursors[0] == 0 and all(cursor > 0 for cursor in cursors[1:])
        state = (await service.graph.aget_state({"configurable": {"thread_id": "mewhelp:ch05:1"}})).values
        assert state["intent"] == "闲聊" and state["pending_calls"] == state["suggested_actions"] == []
        assert len(state["history_turns"]) == 3  # 完成阶段已对账当前轮，不再故意让缓存落后一轮。


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
        state = (await service.graph.aget_state({"configurable": {"thread_id": "mewhelp:ch05:1"}})).values
        assert state["history_turns"][0]["turn_status"] == "completed"
        from sqlalchemy import select
        from app.models.rag_quality import LowConfidenceQuestion
        async with factory() as session:
            assert len((await session.scalars(select(LowConfidenceQuestion))).all()) == 1
