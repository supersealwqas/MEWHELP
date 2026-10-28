"""CH08 真实 SQLite 图暂停验收，确认前无写入，恢复复用原决定与已完成查询。"""

import pytest

from langgraph.types import Command
from sqlalchemy import text

from app.schemas.tools import ToolContext
from app.schemas.workflow import AgentDecision
from app.workflow.state import checkpoint_thread_id, new_turn_state
from tests.workflow.ch08_ticket_fakes import saved_decision, ticket_runtime


def ticket_call(identifier="ticket-1", description="订单1001收到时破损"):
    """构造公开建单工具申请，不包含任何用户确认字段。

    参数：identifier 为调用标识，description 为候选描述；返回：申请字典；异常：无。
    """
    return {"id": identifier, "name": "create_ticket", "args": {"description": description, "ticket_type": "售后"}}


async def begin(graph, chat, query="帮我建个工单，订单1001收到时破损"):
    """保存真实原问并消费 custom 流，返回官方根快照。

    参数：graph 真实图、chat 真实仓储、query 原话；返回：上下文、图配置、快照；异常：图或数据库错误。
    """
    context = await chat.start_turn(ToolContext(session_id="ticket-test", user_id="demo-user",
        turn_id="ticket-turn", user_message=query), query)
    config = {"configurable": {"thread_id": checkpoint_thread_id(context.conversation_id)}}
    async for _ in graph.astream(new_turn_state(context, [], 0), config, stream_mode="custom", durability="sync"):
        pass
    return context, config, await graph.aget_state(config)


async def rows(factory, statement):
    """读取临时业务数据库，用外部可见事实验证行为。

    参数：factory 临时工厂、statement 固定测试 SQL；返回：结果行；异常：SQL 错误。
    """
    async with factory() as session:
        return (await session.execute(text(statement))).all()


async def test_missing_description_does_not_interrupt(tmp_path):
    """模型误发建单时，缺少实际问题仍回灌补参且不占用预览。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call(description="帮我建个工单")]),
            AgentDecision(kind="clarify")], answers=[["请补充实际问题。"]]) as (service, deps, factory, _):
        context, _, snapshot = await begin(service.graph, deps.chat, "帮我建个工单")
        assert not snapshot.interrupts
        result = await deps.chat.get_tool_result(context.model_copy(update={"tool_call_id": "ticket-1"}))
        assert result.error.category == "invalid_arguments"
        assert await deps.chat.get_pending_ticket(context) is None
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []


async def test_preview_precedes_ticket_write(tmp_path):
    """真实写入只能发生在保存预览并接受用户决定之后。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer", suggested_actions=["create_ticket", "handoff"])], answers=[["工单已创建。"]]) as (service, deps, factory, _):
        context, config, paused = await begin(service.graph, deps.chat)
        assert len(paused.interrupts) == 1
        preview = paused.interrupts[0].value
        assert preview["type"] == "confirm_ticket" and preview["description"] == "订单1001收到时破损"
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []
        assert await rows(factory, "SELECT tool_call_id FROM tool_audit_logs") == []
        assert (await deps.chat.get_pending_ticket(context))["action"] == preview
        authorization = await saved_decision(factory, context)
        final = await service.graph.ainvoke(Command(resume=authorization), config)
        assert final["ticket_write_state"]["status"] == "completed"
        assert [action["kind"] for action in final["suggested_actions"]] == ["handoff"]
        last_decision = [row for row in deps.model.inputs if row["stage"] == "decide"][-1]
        assert not any("仍处于待确认阶段" in row.get("content", "") for row in last_decision["messages"])
        assert len(await rows(factory, "SELECT ticket_no FROM tickets")) == 1
        assert await rows(factory, "SELECT status FROM tool_audit_logs") == [("成功",)]


async def test_cancel_has_no_ticket_and_permission_audit(tmp_path):
    """取消恢复经统一执行器保存权限拒绝，不留下工单。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()])]) as (service, deps, factory, _):
        context, config, paused = await begin(service.graph, deps.chat)
        assert paused.interrupts
        await service.graph.ainvoke(Command(resume=await saved_decision(factory, context, decision="cancel")), config)
        result = await deps.chat.get_tool_result(context.model_copy(update={"tool_call_id": "ticket-1"}))
        assert result.error.code == "USER_CANCELLED"
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []
        assert await rows(factory, "SELECT status FROM tool_audit_logs") == [("权限拒绝",)]


async def test_query_then_interrupt_resumes_without_redecide(tmp_path):
    """工具节点重放必须回读前序查询，暂停期间不得重复模型决定或消费预算。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    calls = [{"id": "order-1", "name": "query_order", "args": {"order_id": "1001"}}, ticket_call()]
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=calls),
            AgentDecision(kind="answer")], answers=[["订单已核实，工单已创建。"]]) as (service, deps, factory, queried):
        context, config, paused = await begin(service.graph, deps.chat)
        assert paused.interrupts and queried == ["order-1"]
        assert [item["stage"] for item in deps.model.inputs] == ["understand", "classify", "decide"]
        before = paused.values["budget"]
        final = await service.graph.ainvoke(Command(resume=await saved_decision(factory, context)), config)
        assert queried == ["order-1"] and final["tool_count"] == 2 and final["decision_count"] == 2
        assert [item["stage"] for item in deps.model.inputs] == ["understand", "classify", "decide", "decide", "answer"]
        assert final["budget"]["records"][:len(before["records"])] == before["records"]
        assert await rows(factory, "SELECT tool_call_id FROM tool_audit_logs ORDER BY id") == [("order-1",), ("ticket-1",)]


async def test_second_ticket_in_original_turn_is_rejected(tmp_path):
    """首个有效预览占用原轮流程，后续不同调用 ID 不能再建第二张工单。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    second = ticket_call("ticket-2")
    second["args"]["ticket_type"] = "投诉"
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call(), second]),
            AgentDecision(kind="answer")], answers=[["已创建一张工单。"]]) as (service, deps, factory, _):
        context, config, paused = await begin(service.graph, deps.chat)
        assert paused.interrupts
        await service.graph.ainvoke(Command(resume=await saved_decision(factory, context)), config)
        assert not (await service.graph.aget_state(config)).interrupts
        result = await deps.chat.get_tool_result(context.model_copy(update={"tool_call_id": "ticket-2"}))
        assert result.error.category == "permission_denied"
        assert len(await rows(factory, "SELECT ticket_no FROM tickets")) == 1
        assert await rows(factory, "SELECT status FROM tool_audit_logs ORDER BY id") == [("成功",), ("权限拒绝",)]


async def test_forged_resume_without_saved_decision_cannot_write(tmp_path):
    """直接提交形似凭据的 Command 不能绕过数据库保存的用户决定。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer")], answers=[["没有有效确认，未创建。"]]) as (service, deps, factory, _):
        context, config, paused = await begin(service.graph, deps.chat)
        assert paused.interrupts
        pending = await deps.chat.get_pending_ticket(context)
        action = pending["action"]
        fake = {"conversation_id": context.conversation_id, "turn_id": context.turn_id,
            "tool_call_id": action["tool_call_id"], "action_id": action["action_id"],
            "arguments_hash": pending["arguments_hash"], "decision": "confirm", "origin": "chat"}
        await service.graph.ainvoke(Command(resume=fake), config)
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []
        assert await rows(factory, "SELECT status FROM tool_audit_logs") == [("权限拒绝",)]


async def test_narrow_authorization_rejection_is_audited_but_plain_error_propagates(tmp_path):
    """只有专门的授权拒绝可以变成业务结果，普通程序异常仍原样上抛。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    from app.schemas.tools import ToolError, ToolResult
    from app.tools import contracts
    async with ticket_runtime(tmp_path) as (_, deps, factory, _):
        call = ticket_call()
        context = await deps.chat.start_turn(ToolContext(session_id="narrow", user_id="demo-user",
            turn_id="narrow-turn", tool_call_id=call["id"], user_message="帮我建个工单，订单1001收到时破损"),
            "帮我建个工单，订单1001收到时破损")
        binding = next(row.binding for row in deps.registry.snapshot() if row.name == "create_ticket")

        async def status(frame):
            """接收无 UI 的执行状态。参数：frame 状态帧；返回：None；异常：无。"""

        async def reject(*args):
            """模拟事务中发现原轮已占用。参数：授权参数；返回：无；异常：专门授权拒绝。"""
            raise contracts.WriteAuthorizationRejected(ToolResult(ok=False, error=ToolError(code="TICKET_FLOW_USED", message="原轮已有工单流程", category="permission_denied")))

        result = await deps.executor.execute(call, context, status,
            scope=contracts.ToolExecutionScope(binding=binding, authorize_write=reject))
        assert result.error.code == "TICKET_FLOW_USED"
        assert await rows(factory, "SELECT status FROM tool_audit_logs") == [("权限拒绝",)]

        async def broken(*args):
            """模拟授权代码自身错误。参数：授权参数；返回：无；异常：ValueError。"""
            raise ValueError("ordinary programming error")

        with pytest.raises(ValueError, match="ordinary programming error"):
            await deps.executor.execute(ticket_call("ticket-2"), context.model_copy(update={"tool_call_id": "ticket-2"}), status,
                scope=contracts.ToolExecutionScope(binding=binding, authorize_write=broken))
        assert await deps.chat.get_tool_result(context.model_copy(update={"tool_call_id": "ticket-2"})) is None
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []


async def test_invalid_description_can_be_corrected_before_flow_is_claimed(tmp_path):
    """预览前描述错误允许用新调用修正，不能提前用掉原轮唯一流程。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call("bad", "模型添加了事实")]),
            AgentDecision(kind="tools", tool_calls=[ticket_call("good")]), AgentDecision(kind="answer")],
            answers=[["已创建。"]]) as (service, deps, factory, _):
        context, config, paused = await begin(service.graph, deps.chat)
        assert paused.interrupts and paused.interrupts[0].value["tool_call_id"] == "good"
        first = await deps.chat.get_tool_result(context.model_copy(update={"tool_call_id": "bad"}))
        assert first.error.category == "invalid_arguments"
        history = await deps.chat.load_turns(context.conversation_id)
        assert history[-1].receipt_pending
        await service.graph.ainvoke(Command(resume=await saved_decision(factory, context)), config)
        assert len(await rows(factory, "SELECT ticket_no FROM tickets")) == 1


async def test_collecting_survives_completed_turn_history_and_is_provided_to_decision(tmp_path):
    """待补诉求从真实用户行历史恢复到下一轮，模型获得服务端完整描述。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    async with ticket_runtime(tmp_path, intents=["订单", "订单"], decisions=[AgentDecision(kind="clarify"),
            AgentDecision(kind="tools", tool_calls=[ticket_call()])], answers=[["请补充实际问题。"]]) as (service, deps, _, _):
        context, config, first = await begin(service.graph, deps.chat, "帮我建个工单")
        assert not first.interrupts
        history = [turn.model_dump(mode="json") for turn in await deps.chat.load_turns(context.conversation_id)]
        next_context = await deps.chat.start_turn(context.model_copy(update={"turn_id": "next-turn", "user_message": "订单1001收到时破损"}), "订单1001收到时破损")
        await service.graph.ainvoke(new_turn_state(next_context, history, 0), config)
        paused = await service.graph.aget_state(config)
        assert paused.interrupts[0].value["description"] == "订单1001收到时破损"
        saved = await deps.chat.get_workflow_turn(context.session_id, context.user_id, "next-turn")
        assert saved["ticket_intent"]["source_message_ids"] == ["turn:ticket-turn:user", "turn:next-turn:user"]
        decision = [row for row in deps.model.inputs if row["stage"] == "decide"][-1]
        assert any("description 必须逐字" in row.get("content", "") for row in decision["messages"])


async def test_reopened_sqlite_graph_resumes_original_pending_tool(tmp_path):
    """重新建立官方保存器后仍恢复原工具节点，不重新理解或查询。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    from app.workflow.runtime import workflow_runtime
    calls = [{"id": "order-1", "name": "query_order", "args": {"order_id": "1001"}}, ticket_call()]
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=calls),
            AgentDecision(kind="answer")], answers=[["已创建。"]]) as (service, deps, factory, queried):
        context, config, first = await begin(service.graph, deps.chat)
        assert first.interrupts
        async with workflow_runtime(deps, tmp_path / "workflow.sqlite3") as reopened:
            paused = await reopened.aget_state(config)
            assert paused.interrupts[0].value == first.interrupts[0].value
            await reopened.ainvoke(Command(resume=await saved_decision(factory, context)), config)
        assert queried == ["order-1"]
        assert len(await rows(factory, "SELECT ticket_no FROM tickets")) == 1


async def test_already_started_without_result_stays_unknown_and_never_writes(tmp_path):
    """确认后执行边界已越过但结果缺失时，恢复必须拒绝第二次写入。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    from app.repositories.chat import PersistenceUncertain
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()])]) as (service, deps, factory, _):
        context, config, paused = await begin(service.graph, deps.chat)
        authorization = await saved_decision(factory, context)
        await deps.chat.mark_ticket_execution(context.model_copy(update={"tool_call_id": "ticket-1"}), paused.interrupts[0].value["action_id"])
        with pytest.raises(PersistenceUncertain):
            await service.graph.ainvoke(Command(resume=authorization), config)
        assert (await deps.chat.get_pending_ticket(context))["status"] == "unknown"
        assert await rows(factory, "SELECT ticket_no FROM tickets") == []
        assert await deps.chat.get_tool_result(context.model_copy(update={"tool_call_id": "ticket-1"})) is None
        with pytest.raises(PersistenceUncertain):
            await service.graph.ainvoke(Command(resume=authorization), config)
        assert (await deps.chat.get_pending_ticket(context))["status"] == "unknown"
        assert await deps.chat.get_tool_result(context.model_copy(update={"tool_call_id": "ticket-1"})) is None


async def test_confirmed_receipt_is_streamed_and_saved_even_when_answer_fails(tmp_path):
    """正文模型故障不能隐藏已经提交的真实工单号，SSE 与落库答复均先保留回执。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer")], answers=[[RuntimeError("answer crashed")]]) as (service, deps, factory, _):
        context, config, _ = await begin(service.graph, deps.chat)
        frames = [frame async for frame in service.graph.astream(Command(resume=await saved_decision(factory, context)),
            config, stream_mode="custom", subgraphs=True, durability="sync")]
        final = (await service.graph.aget_state(config)).values
        number = (await rows(factory, "SELECT ticket_no FROM tickets"))[0][0]
        visible = "".join(frame[1]["data"]["delta"] for frame in frames if frame[1].get("event") == "delta")
        assert final["answer"].startswith(f"工单已创建，工单号：{number}")
        assert visible == final["answer"] and "待处理" in visible
        saved = await deps.chat.get_workflow_turn(context.session_id, context.user_id, context.turn_id)
        assert saved["answer"] == visible


async def test_committed_ticket_with_unfinished_reply_remains_summary_boundary(tmp_path):
    """工单已提交但回复尚未保存时，失败原轮仍不能被摘要越过。

    参数：tmp_path 临时目录；返回：None；异常：断言或图错误。
    """
    from sqlalchemy import update
    from app.models.customer_service import Message
    async with ticket_runtime(tmp_path, decisions=[AgentDecision(kind="tools", tool_calls=[ticket_call()]),
            AgentDecision(kind="answer")], answers=[["已创建。"]]) as (service, deps, factory, _):
        context, config, _ = await begin(service.graph, deps.chat)
        await service.graph.ainvoke(Command(resume=await saved_decision(factory, context)), config)
        # 模拟工单可靠结果保留、正文事务没有完成的原轮，不修改票据或确认决定。
        async with factory.begin() as session:
            await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
                Message.turn_id == context.turn_id).values(turn_status="failed"))
        history = await deps.chat.load_turns(context.conversation_id)
        assert history[-1].receipt_pending
