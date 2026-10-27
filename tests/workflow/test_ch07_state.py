"""CH07完整历史和旧检查点迁移：使用真实SQLite及正式服务，模型只替换外部调用。"""

from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, Overwrite, interrupt

from tests.workflow.service_fakes import collect, service_runtime


async def test_sdk_overwrite_with_resume_preserves_old_interrupt(tmp_path):
    """旧普通字段转reducer不能丢原interrupt；专门验证已查文档未保证的升级组合。参数：tmp_path；返回：None；异常：断言失败。"""
    class LegacyState(TypedDict):
        """旧检查点消息字段整体替换。"""
        messages: list
        accepted: str

    class CurrentState(TypedDict):
        """新检查点按稳定消息ID合并。"""
        messages: Annotated[list[AnyMessage], add_messages]
        accepted: str

    def wait(state):
        """参数：state旧/新状态；返回：确认结果；异常：GraphInterrupt正常等待。"""
        return {"accepted": interrupt({"action_id": "same-action"})}

    def graph(schema, saver):
        """参数：schema状态定义、saver真实保存器；返回：图；异常：编译失败。"""
        builder = StateGraph(schema)
        builder.add_node("wait", wait)
        builder.add_edge(START, "wait")
        builder.add_edge("wait", END)
        return builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "old-thread"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "upgrade.sqlite")) as saver:
        await graph(LegacyState, saver).ainvoke({"messages": [{"role": "user", "content": "原始问题"}], "accepted": ""}, config)
        result = await graph(CurrentState, saver).ainvoke(Command(resume="original-order",
            update={"messages": Overwrite([HumanMessage(content="原始问题", id="same-user")])}), config)
        assert result["accepted"] == "original-order"
        assert [(m.id, m.content) for m in result["messages"]] == [("same-user", "原始问题")]


async def test_three_turns_keep_all_original_messages_in_checkpoint_and_do_not_replay(tmp_path):
    """每轮覆盖messages或父子图重复追加都会失败。参数：tmp_path；返回：None；异常：断言失败。"""
    config = {"configurable": {"thread_id": "mewhelp:ch05:1"}}
    async with service_runtime(tmp_path, intents=["闲聊"] * 3) as (service, deps, _):
        for question in ["第一轮  ", "第二轮\n", "第三轮"]:
            assert (await collect(service, question))[-1].event == "done"
        state = (await service.graph.aget_state(config)).values
        assert state["context_version"] == 7
        assert len(state["messages"]) == 6 and all(isinstance(message, BaseMessage) for message in state["messages"])
        assert [m.content for m in state["messages"] if m.type == "human"] == ["第一轮  ", "第二轮\n", "第三轮"]
        assert len({m.id for m in state["messages"]}) == 6
        assert [m.additional_kwargs["db_message_id"] for m in state["messages"]] == [1, 2, 3, 4, 5, 6]
        assert not deps.executor.calls
        # 重放同一已保存消息只能更新元数据，不能追加第二个气泡。
        await service.graph.aupdate_state(config, {"messages": [state["messages"][-1]]}, as_node="log_turn")
        assert len((await service.graph.aget_state(config)).values["messages"]) == 6


async def test_legacy_completed_checkpoint_rebuilds_full_history_from_database(tmp_path):
    """旧版只保存本轮messages时，升级不能遗失前轮或带进临时system。参数：tmp_path；返回：None；异常：断言失败。"""
    config = {"configurable": {"thread_id": "mewhelp:ch05:1"}}
    async with service_runtime(tmp_path, intents=["闲聊", "闲聊"]) as (service, deps, _):
        await collect(service, "原始第一轮")
        # 模拟上一版本已结束状态，正文仍在可信数据库中。
        await service.graph.aupdate_state(config, {"context_version": 6,
            "messages": Overwrite([HumanMessage(content="原始第一轮"),
                {"role": "system", "content": "旧的临时资料"}])}, as_node="log_turn")
        assert (await collect(service, "继续"))[-1].event == "done"
        state = (await service.graph.aget_state(config)).values
        assert state["context_version"] == 7 and len(state["messages"]) == 4
        assert all(m.type != "system" for m in state["messages"])
        assert [m.content for m in state["messages"] if m.type == "human"] == ["原始第一轮", "继续"]


async def test_same_tool_keys_in_different_turns_keep_both_original_pairs(tmp_path):
    """稳定身份不能只用工具键，否则跨轮同键会覆盖旧事实。参数：tmp_path；返回：None；异常：断言失败。"""
    from app.schemas.tools import ToolResult
    from app.schemas.workflow import AgentDecision
    from tests.workflow.graph_fakes import call

    decisions = [AgentDecision(kind="tools", tool_calls=[call("order", order_id="1001")]),
                 AgentDecision(kind="tools", tool_calls=[call("shipping", "query_logistics", tracking_no="SF1")]),
                 AgentDecision(kind="answer")]
    config = {"configurable": {"thread_id": "mewhelp:ch05:1"}}
    results = [ToolResult.success({"order_id": "1001", "tracking_no": "SF1"}),
               ToolResult.success({"status": "运输中"}),
               ToolResult.success({"order_id": "1001", "tracking_no": "SF1"}),
               ToolResult.success({"status": "已签收"})]
    async with service_runtime(tmp_path, intents=["订单"] * 2, decisions=decisions * 2, results=results,
                               answers=[["运输中"], ["已签收"]]) as (service, deps, _):
        for question in ("订单1001现在在哪里", "再查一下订单1001"):
            assert (await collect(service, question))[-1].event == "done"
        state = (await service.graph.aget_state(config)).values
        messages = state["messages"]
        assert len(messages) == len({message.id for message in messages}) == 12
        assert [m.additional_kwargs["db_message_id"] for m in messages] == list(range(1, 13))
        assert [m.tool_call_id for m in messages if m.type == "tool"] == ["order", "shipping"] * 2
        assert "运输中" in messages[4].content and "已签收" in messages[10].content
        assert len(deps.executor.calls) == 4
        await service.graph.aupdate_state(config, {"messages": messages[-4:]}, as_node="log_turn")
        assert len((await service.graph.aget_state(config)).values["messages"]) == 12


async def test_legacy_waiting_checkpoint_resumes_same_action_without_reclassification(tmp_path):
    """旧等待升级不能换动作、重跑分类或保留临时system。参数：tmp_path；返回：None；异常：断言失败。"""
    from tests.workflow.test_ch06_graph import order_result
    from tests.workflow.test_order_selection import resume_request
    config = {"configurable": {"thread_id": "mewhelp:ch05:1"}}
    async with service_runtime(tmp_path, intents=["退款退货"], results=[order_result()],
                               answers=[["签收时间还需核实。"]]) as (service, deps, _):
        action = (await collect(service, "我要退款"))[-1].data
        await service.graph.aupdate_state(config, {"context_version": 6, "current_message_id": "",
            "messages": Overwrite([{"role": "user", "content": "我要退款"},
                {"role": "system", "content": "旧临时内容"}])}, as_node="classify_intent")
        await service.graph.ainvoke(None, config)  # 在同一节点再次挂回原来的确定性动作。
        resumed = [event async for event in service.stream_resume(resume_request(action))]
        assert resumed[-1].event == "done"
        state = (await service.graph.aget_state(config)).values
        assert state["context_version"] == 7
        assert state["context"]["turn_id"] == action["source_turn_id"]
        assert sum(item["stage"] == "classify" for item in deps.model.inputs) == 1
        assert all(message.type != "system" for message in state["messages"])
        assert len({message.id for message in state["messages"]}) == len(state["messages"]) == 4
