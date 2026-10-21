"""同轮多工具的真实编排回归，覆盖依赖查询、调用限制、重复写入与中断事实。"""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.core.memory import assemble_turns
from app.schemas.chat import ChatRequest
from app.schemas.tools import ToolResult
from tests.chat_fakes import make_chat_service


def call(name: str, identifier: str, **args) -> dict:
    """构造模型业务申请。参数：工具名、调用ID与参数；返回：申请字典；异常：无。"""
    return {"name": name, "id": identifier, "args": args}


def decision(*calls: dict) -> AIMessage:
    """构造一个完整决策批次。参数：工具申请；返回：模型消息；异常：字段校验错误。"""
    return AIMessage(content="", tool_calls=list(calls))


async def run(service, message="订单1001的物流到哪了") -> list:
    """执行真实单轮服务。参数：服务与原话；返回：事件列表；异常：服务未捕获异常。"""
    return [event async for event in service.stream_turn(ChatRequest(session_id="multi", message=message))]


def install_ticket_store(repository, support) -> list:
    """模拟数据库写入并记录实际创建次数。参数：两个仓库；返回：写入记录；异常：无。"""
    writes = []

    async def create_ticket(context, description, ticket_type):
        """保存唯一编号用于发现重复建单。参数：上下文、描述、类型；返回：结果；异常：存储错误。"""
        writes.append((description, ticket_type))
        return await repository.save_tool_result(context, ToolResult.success({
            "ticket_no": f"T-{len(writes)}", "ticket_type": ticket_type, "status": "待处理"}))

    support.create_ticket = create_ticket
    return writes


async def test_order_logistics_chain_uses_current_turn_result():
    """下一次决策必须收到本轮真实订单结果。参数：无；返回：None；异常：断言失败。"""
    service, repository, _, model = make_chat_service(decisions=[
        decision(call("query_order", "order", order_id="1001")),
        decision(call("query_logistics", "shipping", tracking_no="MOCK213502378238")),
    ])
    events = await run(service)
    assert [event.data["name"] for event in events if event.event == "tool_status"] == [
        "query_order", "query_order", "query_logistics", "query_logistics"]
    assert [phase for phase, _ in model.inputs] == ["decide", "decide", "decide", "answer"]
    previous_result = next(message for message in model.inputs[1][1] if isinstance(message, ToolMessage))
    assert json.loads(previous_result.content)["data"]["tracking_no"] == "MOCK213502378238"
    results = list(repository.results.values())
    assert len(results) == 2 and results[1].data["latest_node"] == "包裹已到达杭州转运中心"
    assert events[-1].event == "done" and sum(event.event == "done" for event in events) == 1


async def test_same_batch_tools_run_sequentially():
    """同批工具状态必须依次结束后开始下一项。参数：无；返回：None；异常：断言失败。"""
    service, repository, _, _ = make_chat_service(decisions=[decision(
        call("query_order", "first", order_id="1001"), call("query_order", "second", order_id="1005"))])
    events = await run(service, "查询订单1001和1005")
    assert [(event.data["tool_call_id"], event.data["status"]) for event in events
            if event.event == "tool_status"] == [
                ("first", "running"), ("first", "succeeded"), ("second", "running"), ("second", "succeeded")]
    assert len(repository.results) == 2


@pytest.mark.parametrize("limit", [1, 2, 4])
async def test_call_budget_stops_further_business_execution(limit):
    """模型持续申请时仍遵守配置的执行上限。参数：上限；返回：None；异常：断言失败。"""
    service, repository, _, model = make_chat_service(decisions=[
        decision(call("query_product", f"p{i}", keyword=f"商品{i}")) for i in range(6)])
    service.settings.tool_max_calls_per_turn = limit
    events = await run(service, "查询这些商品")
    assert len(repository.results) == limit
    assert sum(phase == "decide" for phase, _ in model.inputs) == limit
    assert events[-1].event == "done"


async def test_batch_exceeding_remaining_budget_is_not_partially_executed():
    """剩余额度不足时明确拒绝整批并配对结果。参数：无；返回：None；异常：断言失败。"""
    service, repository, _, _ = make_chat_service(decisions=[decision(*[
        call("query_product", f"p{i}", keyword=f"商品{i}") for i in range(5)])])
    events = await run(service)
    assert len(repository.results) == 5
    assert all(result.error.code == "TOOL_CALL_LIMIT" for result in repository.results.values())
    assert not any(event.event == "tool_status" and event.data["status"] == "running" for event in events)


async def test_reused_call_id_across_decisions_is_audited_without_execution():
    """后续同ID不能复用另一工具的旧结果。参数：无；返回：None；异常：断言失败。"""
    service, repository, _, _ = make_chat_service(decisions=[
        decision(call("query_order", "same", order_id="1001")),
        decision(call("query_logistics", "same", tracking_no="MOCK213502378238")),
    ])
    await run(service)
    turn = next(iter(repository.turns.values()))
    assert len(repository.results) == 1
    assert any(message.tool_calls and message.tool_calls[0].get("type") == "invalid_tool_decision"
               for message in turn.messages)


async def test_same_arguments_with_new_id_are_not_executed_twice():
    """更换ID不能绕过相同请求去重。参数：无；返回：None；异常：断言失败。"""
    service, repository, _, _ = make_chat_service(decisions=[
        decision(call("query_order", "one", order_id="1001")),
        decision(call("query_order", "two", order_id="1001")),
    ])
    events = await run(service)
    results = list(repository.results.values())
    assert len(results) == 2 and results[1].error.code == "DUPLICATE_TOOL_CALL"
    assert sum(event.event == "tool_status" and event.data["status"] == "running" for event in events) == 1


@pytest.mark.parametrize("same_batch", [True, False])
async def test_new_call_id_cannot_create_second_ticket(same_batch):
    """改ID和描述也不能在同轮重复建单。参数：是否同批；返回：None；异常：断言失败。"""
    first = call("create_ticket", "t1", description="订单未送达", ticket_type="售后")
    second = call("create_ticket", "t2", description="补充订单未送达", ticket_type="售后")
    decisions = [decision(first, second)] if same_batch else [decision(first), decision(second)]
    service, repository, support, _ = make_chat_service(decisions=decisions)
    writes = install_ticket_store(repository, support)
    events = await run(service, "请帮我创建工单")
    assert len(writes) == 1 and len(repository.results) == 2
    assert list(repository.results.values())[1].error.code == "WRITE_TOOL_LIMIT"
    assert "T-1" in "".join(event.data.get("delta", "") for event in events)


async def test_ticket_then_later_decision_failure_keeps_receipt_and_unsafe_retry():
    """建单后只读步骤不能重置已写入标记。参数：无；返回：None；异常：断言失败。"""
    service, repository, support, _ = make_chat_service(decisions=[
        decision(call("create_ticket", "ticket", description="订单未送达", ticket_type="售后")),
        decision(call("query_order", "order", order_id="1001")), RuntimeError("PRIVATE"),
    ])
    install_ticket_store(repository, support)
    events = await run(service, "创建工单并查询订单1001")
    assert events[-1].event == "error" and events[-1].data["retry_safe"] is False
    assert "T-1" in events[-1].data["error"] and "PRIVATE" not in str(events)
    assert len(repository.results) == 2 and not any(event.event == "done" for event in events)


async def test_cancel_after_ticket_keeps_partial_batch_in_history():
    """批次中途停止时已建工单仍可被后续会话读取。参数：无；返回：None；异常：断言失败。"""
    service, repository, support, model = make_chat_service(decisions=[decision(
        call("create_ticket", "ticket", description="订单未送达", ticket_type="售后"),
        call("query_order", "order", order_id="1001"))])
    install_ticket_store(repository, support)
    stream = service.stream_turn(ChatRequest(session_id="cancel", message="创建工单并查订单1001"))
    try:
        while (event := await anext(stream)).event != "delta":
            pass
        assert "T-1" in event.data["delta"]
    finally:
        await stream.aclose()
    assert len(repository.results) == 1
    history = assemble_turns(list(repository.turns.values()))[0]
    assert any(isinstance(message, ToolMessage) and "T-1" in message.content for message in history)
    assert [phase for phase, _ in model.inputs] == ["decide"]


async def test_foreign_order_stops_remaining_batch_with_fixed_reply():
    """归属拒绝后不执行剩余写操作，仍完整配对。参数：无；返回：None；异常：断言失败。"""
    service, repository, support, model = make_chat_service(decisions=[decision(
        call("query_order", "foreign", order_id="2003"),
        call("create_ticket", "ticket", description="查询这笔订单", ticket_type="咨询"))])
    writes = install_ticket_store(repository, support)
    events = await run(service)
    assert not writes and len(repository.results) == 2
    assert list(repository.results.values())[1].error.code == "TOOL_CHAIN_STOPPED"
    assert "".join(event.data.get("delta", "") for event in events) == "没有找到您的这笔订单，请确认订单号或物流单号后重试。"
    assert [phase for phase, _ in model.inputs] == ["decide"]


async def test_saved_ticket_is_not_lost_when_later_order_is_rejected():
    """固定拒绝回复也必须与先发的工单编号一起落库。参数：无；返回：None；异常：断言失败。"""
    service, repository, support, _ = make_chat_service(decisions=[decision(
        call("create_ticket", "ticket", description="处理问题", ticket_type="咨询"),
        call("query_order", "foreign", order_id="2003"))])
    install_ticket_store(repository, support)
    events = await run(service, "创建工单并查询订单2003")
    answer = "".join(event.data.get("delta", "") for event in events)
    assert "T-1" in answer and "没有找到您的这笔订单" in answer
    assert next(iter(repository.turns.values())).messages[-1].content == answer


@pytest.mark.parametrize("is_ticket", [False, True], ids=["order", "ticket"])
async def test_second_decision_budget_overflow_preserves_committed_facts(is_ticket):
    """新增工具结果超预算时不得继续决策或丢失事实。参数：is_ticket 是否已建工单；返回：None；异常：断言失败。"""
    from langchain_core.messages import HumanMessage
    from app.core.memory import ContextBudgetExceeded, _estimate_message_tokens, prepare_messages
    from app.core.prompts import CUSTOMER_SERVICE_SYSTEM, TOOL_DECISION_SYSTEM

    first = (call("create_ticket", "first", description="订单未送达", ticket_type="售后") if is_ticket
             else call("query_order", "first", order_id="1001"))
    question = "订单1001未送达，请创建工单" if is_ticket else "订单1001的物流到哪了"
    service, repository, support, model = make_chat_service(decisions=[
        decision(first), decision(call("query_product", "unused", keyword="猫粮")),
    ])
    if is_ticket:
        install_ticket_store(repository, support)
    before_result = [HumanMessage(content=question), decision(first)]
    budget = _estimate_message_tokens(before_result)
    service.settings.token_budget = budget
    budget_options = {
        "system_text": CUSTOMER_SERVICE_SYSTEM + TOOL_DECISION_SYSTEM,
        "tools": service.registry.all(),
        "request_max_tokens": service.settings.request_token_budget,
    }
    # 用户原话和申请本身放得下；超限必须由随后保存的成功工具结果触发。
    assert prepare_messages([], before_result, budget, **budget_options)[1:] == before_result
    events = await run(service, question)
    assert [phase for phase, _ in model.inputs] == ["decide"]
    assert events[-1].event == "error" and not any(event.event == "done" for event in events)
    assert len(repository.results) == 1
    result = next(iter(repository.results.values()))
    assert result.ok
    fact = ToolMessage(content=result.model_dump_json(), tool_call_id="first")
    with pytest.raises(ContextBudgetExceeded):
        prepare_messages([], [*before_result, fact], budget, **budget_options)
    turn = next(iter(repository.turns.values()))
    assert turn.turn_status == "failed"
    assert [message.role for message in turn.messages] == ["user", "assistant", "tool"]
    history = assemble_turns([turn])[0]
    assert any(isinstance(message, ToolMessage) and message.content == fact.content for message in history)
    if is_ticket:
        assert result.data["ticket_no"] == "T-1"
        assert "T-1" in "".join(event.data.get("delta", "") for event in events)
        assert "T-1" in events[-1].data["error"] and events[-1].data["retry_safe"] is False
    else:
        assert result.data["order_id"] == "1001"
        assert "工具结果过长" in events[-1].data["error"] and events[-1].data["retry_safe"] is True
