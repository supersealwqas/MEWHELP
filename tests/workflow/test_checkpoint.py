"""官方SQLite根图持久化、会话隔离和新dict启动契约测试。"""
from pathlib import Path
import pytest
from app.schemas.workflow import AgentDecision
from app.schemas.tools import ToolResult
from app.repositories.chat import PersistenceUncertain
from app.workflow.runtime import workflow_runtime
from tests.workflow.graph_fakes import dependencies, turn, call

async def test_file_reopen_isolated_sessions_and_finish_before_final_checkpoint(tmp_path):
    """文件未持久化、会话串线或完成提交后checkpoint顺序错误时失败。参数：tmp_path独占目录；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["闲聊", "投诉"])
    path = tmp_path / "nested" / "workflow.sqlite3"
    config = {"configurable": {"thread_id": "one"}}
    async with workflow_runtime(deps, path) as graph:
        assert graph.config["recursion_limit"] == 64
        events = [event async for event in graph.astream(turn("你好"), config, stream_mode="custom", subgraphs=True, durability="sync")]
        assert events and deps.chat.order[-1] == "finish"
        saved = await graph.aget_state(config)
        assert not saved.next and saved.values["completion_metadata"]["turn_id"] == deps.chat.finished[0]["context"].turn_id
        await graph.ainvoke(turn("投诉", session="two"), {"configurable": {"thread_id": "two"}}, durability="sync")
    async with workflow_runtime(deps, path) as reopened:
        assert (await reopened.aget_state(config)).values["raw_query"] == "你好"
        assert (await reopened.aget_state({"configurable": {"thread_id": "two"}})).values["intent"] == "投诉"

async def test_new_dict_starts_new_turn_without_replaying_pending_tool(tmp_path):
    """重启后新轮恢复旧pending工具或旧意图/建议时失败。参数：tmp_path独占目录；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["订单", "闲聊"], decisions=[AgentDecision(kind="tools", tool_calls=[call("old")]), AgentDecision(kind="answer")],
        results=[PersistenceUncertain("结果未查清")])
    path, config = tmp_path / "workflow.sqlite3", {"configurable": {"thread_id": "one"}}
    async with workflow_runtime(deps, path) as graph:
        with pytest.raises(PersistenceUncertain):
            await graph.ainvoke(turn(), config, durability="sync")
        interrupted = await graph.aget_state(config, subgraphs=True)
        assert interrupted.next == ("main_agent",)
        assert interrupted.tasks[0].state.values["pending_calls"] == [call("old")]
    async with workflow_runtime(deps, path) as graph:
        result = await graph.ainvoke(turn("谢谢", turn_id="new"), config, durability="sync")
        assert result["intent"] == "闲聊" and result["tool_count"] == 0
        assert result["pending_calls"] == [] and result["evidence"] == [] and result["suggested_actions"] == []
        assert deps.executor.calls == [call("old")]

async def test_final_checkpoint_failure_after_business_finish_is_propagated(tmp_path, monkeypatch):
    """MySQL完成后SQLite写失败却让图成功耗尽时失败。参数：tmp_path目录、monkeypatch故障注入；返回：None；异常：受控错误。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    deps = dependencies(intents=["闲聊"])
    original = AsyncSqliteSaver.aput
    async def save(saver, config, checkpoint, metadata, new_versions):
        """参数：官方保存器写入契约；返回：官方结果；异常：终态写入受控OSError。"""
        if checkpoint["channel_values"].get("completion_metadata"):
            deps.chat.order.append("final-checkpoint")
            raise OSError("磁盘写入失败")
        return await original(saver, config, checkpoint, metadata, new_versions)
    monkeypatch.setattr(AsyncSqliteSaver, "aput", save)
    async with workflow_runtime(deps, tmp_path / "workflow.sqlite3") as graph:
        with pytest.raises(OSError, match="磁盘写入失败"):
            async for event in graph.astream(turn(), {"configurable": {"thread_id": "one"}}, stream_mode="custom", subgraphs=True, durability="sync"):
                assert event[1]["event"] not in {"done", "answer_metadata"}
    assert deps.chat.order == ["finish", "final-checkpoint"]

async def test_finish_failure_prevents_successful_final_state(tmp_path):
    """业务完成提交失败却保存最终成功元数据时失败。参数：tmp_path目录；返回：None；异常：受控错误。"""
    deps = dependencies(intents=["闲聊"])
    async def fail(*args, **kwargs):
        """参数：完成契约；返回：无；异常：受控数据库错误。"""
        raise ConnectionError("事务失败")
    deps.chat.finish_turn = fail
    config = {"configurable": {"thread_id": "one"}}
    async with workflow_runtime(deps, tmp_path / "workflow.sqlite3") as graph:
        with pytest.raises(ConnectionError):
            await graph.ainvoke(turn(), config, durability="sync")
        state = await graph.aget_state(config)
        assert state.next == ("log_turn",) and state.values["completion_metadata"] == {}

async def test_runtime_rejects_memory_path():
    """运行入口悄悄使用内存数据库时失败。参数：无；返回：None；异常：预期ValueError。"""
    with pytest.raises(ValueError):
        async with workflow_runtime(dependencies(), Path(":memory:")):
            pass
