"""聊天编排的通用行为测试：协议、增量、故障、并发与取消。"""

import asyncio
import json
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from app.schemas.chat import ChatRequest
from tests.chat_fakes import make_chat_service as make_service


def request(session_id="s"):
    """构造用户请求。参数：会话 ID；返回：ChatRequest；异常：校验失败。"""
    return ChatRequest(session_id=session_id, message="邮费是多少")


def faq_call(call_id="c1", args=None, name="query_faq"):
    """构造完整工具申请。参数：ID、业务参数和名称；返回：字典；异常：无。"""
    return {"id": call_id, "name": name, "args": args if args is not None else {"keyword": "邮费"}}


async def test_one_tool_status_before_answer_and_commit_before_done():
    """缺少工具配对或提前 done 都应失败。参数：无；返回：None；异常：断言失败。"""
    service, repo, support, model = make_service(AIMessage(content="", tool_calls=[faq_call()]))
    events = [e async for e in service.stream_turn(request())]
    assert [e.event for e in events] == ["tool_status", "tool_status", "delta", "delta", "answer_metadata", "done"]
    assert [e.data["status"] for e in events[:2]] == ["running", "succeeded"]
    assert support.keywords == ["邮费"]
    assert [step for step, _ in model.inputs] == ["decide", "decide", "answer"]
    messages = model.inputs[-1][1]
    results = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(results) == 1 and results[0].tool_call_id == "c1"
    # 空命中现在以 hits=[] 表达，并带未找到说明，模型可据此拒答。
    assert '"hits":[]' in results[0].content and "未找到" in results[0].content
    turn = next(iter(repo.turns.values()))
    assert turn.turn_status == "completed" and turn.messages[-1].content == "已查到"


async def test_no_tool_still_streams_and_does_not_save_planning_text():
    """无工具轮不泄露规划或制造徽章。参数：无；返回：None；异常：断言失败。"""
    service, repo, _, model = make_service()
    iterator = service.stream_turn(request())
    assert (await anext(iterator)).data == {"delta": "已"}
    assert next(iter(repo.turns.values())).turn_status == "running"
    assert (await anext(iterator)).data == {"delta": "查到"}
    assert (await anext(iterator)).event == "answer_metadata"
    assert (await anext(iterator)).event == "done"
    assert next(iter(repo.turns.values())).turn_status == "completed"
    await iterator.aclose()
    assert len(model.inputs) == 2
    assert all("规划" not in m.content for m in next(iter(repo.turns.values())).messages)


async def test_model_inputs_hide_source_marker_for_current_and_historical_facts():
    """当前及历史工具事实不向模型暴露来源标记，后台审计仍保留。

    参数：无，运行真实聊天服务、订单工具与历史组装，仅替换模型网络和存储。
    返回：None，断言两轮模型输入只含业务事实，原始结果未被修改。
    异常：AssertionError 表示标记泄露、业务事实丢失或审计被改写。
    """
    service, repo, _, model = make_service(decisions=[
        AIMessage(content="", tool_calls=[faq_call(name="query_order", args={"order_id": "1001"})]),
        AIMessage(content=""), AIMessage(content=""),
    ], chunks=["订单信息已查询。"])
    first = [event async for event in service.stream_turn(ChatRequest(session_id="source", message="查订单1001"))]
    original = next(iter(repo.results.values())).model_dump()
    second_start = len(model.inputs)
    second = [event async for event in service.stream_turn(ChatRequest(session_id="source", message="订单金额是多少"))]
    assert first[-1].event == second[-1].event == "done"
    assert original["is_mock"] is True
    assert next(iter(repo.results.values())).model_dump() == original
    assert json.loads(next(row.content for row in next(iter(repo.turns.values())).messages
                           if row.role == "tool"))["is_mock"] is True
    for inputs in (model.inputs[:second_start], model.inputs[second_start:]):
        facts = [json.loads(message.content) for _, messages in inputs
                 for message in messages if isinstance(message, ToolMessage)]
        assert facts and all("is_mock" not in fact for fact in facts)
        assert all(fact["ok"] and fact["data"]["order_id"] == "1001"
                   and fact["data"]["total_amount"] == 39.9 for fact in facts)


async def test_failed_tool_fact_keeps_error_without_source_marker():
    """来源标记过滤不能隐藏工具失败的错误事实。

    参数：无，使用真实订单参数校验触发失败。
    返回：None，断言模型得到失败状态与错误，后台保留来源。
    异常：AssertionError 表示错误被吞掉或来源标记泄露。
    """
    service, repo, _, model = make_service(AIMessage(content="", tool_calls=[
        faq_call(name="query_order", args={"order_id": " "})]))
    events = [event async for event in service.stream_turn(request())]
    assert events[-1].event == "done"
    fact = json.loads(next(message.content for message in model.inputs[-1][1] if isinstance(message, ToolMessage)))
    assert fact == {"ok": False, "data": None,
                    "error": next(iter(repo.results.values())).error.model_dump()}
    assert fact["error"]["code"] == "INVALID_ARGUMENTS"


async def test_multiple_calls_execute_in_order_and_pair_all_results():
    """同批独立查询依次执行且结果全部配对。参数：无；返回：None；异常：断言失败。"""
    service, repo, support, model = make_service(AIMessage(content="", tool_calls=[
        faq_call(args={"keyword": "邮费"}), faq_call("c2", args={"keyword": "是多少"})]))
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].event == "done" and support.keywords == ["邮费", "是多少"]
    results = [m for m in model.inputs[-1][1] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in results} == {"c1", "c2"}
    assert all('"ok":true' in m.content for m in results)


@pytest.mark.parametrize("decision", [
    AIMessage(content="", tool_calls=[faq_call(None)]),
    AIMessage(content="", tool_calls=[faq_call(), faq_call()]),
    AIMessage(content="", invalid_tool_calls=[{"name": "query_faq", "id": "x", "args": "{broken", "error": "bad"}]),
])
async def test_broken_protocol_is_audited_before_safe_answer(decision):
    """坏协议只留审计，不执行或回灌。参数：损坏决策；返回：None；异常：断言失败。"""
    service, repo, support, model = make_service(decision)
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].event == "done" and not support.keywords
    assert [step for step, _ in model.inputs] == ["decide", "answer"]
    audit = next(m.tool_calls for m in next(iter(repo.turns.values())).messages if m.tool_calls)
    assert audit[0]["type"] == "invalid_tool_decision"
    assert not any(getattr(m, "tool_calls", None) for m in model.inputs[-1][1])
    assert any("未执行任何工具" in m.content for m in model.inputs[-1][1])
    from app.core.memory import assemble_turns
    assert not any(getattr(m, "tool_calls", None) for m in assemble_turns(list(repo.turns.values()))[0])


@pytest.mark.parametrize("call,code", [(faq_call(name="not_registered"), "UNKNOWN_TOOL"),
    (faq_call(args={"keyword": " "}), "INVALID_ARGUMENTS")])
async def test_tool_failure_is_safely_returned_once(call, code):
    """未知工具或坏参数回灌一次错误，不执行业务。参数：申请与错误码；返回：None；异常：断言失败。"""
    service, repo, support, model = make_service(AIMessage(content="", tool_calls=[call]))
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].event == "done" and not support.keywords
    assert code in next(m.content for m in model.inputs[-1][1] if isinstance(m, ToolMessage))
    assert len(repo.results) == 1


@pytest.mark.parametrize("chunks,fail_commit", [([""], False), (["部分", RuntimeError("SECRET")], False), (["答案"], True)])
async def test_no_done_on_empty_answer_model_or_commit_failure(chunks, fail_commit):
    """错误不得发成功结束帧或泄露异常。参数：增量、提交故障；返回：None；异常：断言失败。"""
    service, repo, _, _ = make_service(chunks=chunks)
    repo.fail_finish = fail_commit
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].event == "error" and all(e.event != "done" for e in events)
    assert "SECRET" not in str(events) and next(iter(repo.turns.values())).turn_status == "failed"


async def test_concurrent_same_session_rejected_and_cancel_releases_lock():
    """同会话并发不写入；关闭流后允许下一次发送。参数：无；返回：None；异常：断言失败。"""
    service, repo, _, _ = make_service()
    first = service.stream_turn(request())
    await anext(first)
    second = [e async for e in service.stream_turn(request())]
    assert second[0].event == "error" and len(repo.turns) == 1
    other = [e async for e in service.stream_turn(request("other"))]
    assert other[-1].event == "done"
    await first.aclose()
    assert list(repo.turns.values())[0].turn_status == "cancelled"
    again = [e async for e in service.stream_turn(request())]
    assert again[-1].event == "done"


async def test_cancel_during_tool_waits_for_cleanup_and_stops_retry():
    """取消时工具完成清理后才能释放会话。参数：无；返回：None；异常：断言失败。"""
    service, repo, support, _ = make_service(AIMessage(content="", tool_calls=[faq_call()]))
    entered, cleaned = asyncio.Event(), asyncio.Event()
    async def slow(keyword):
        """控制工具阻塞。参数：关键词；返回：不返回；异常：取消。"""
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    support.query_faq = slow
    stream = service.stream_turn(request())
    assert (await anext(stream)).event == "tool_status"
    await entered.wait()
    await stream.aclose()
    assert cleaned.is_set() and not repo.results
    assert next(iter(repo.turns.values())).turn_status == "cancelled"


async def test_padded_session_id_cannot_bypass_same_conversation_lock():
    """MySQL 忽略末尾空格，客户端变体不得绕过互斥。参数：无；返回：None；异常：断言失败。"""
    service, repo, _, _ = make_service()
    first = service.stream_turn(request("same"))
    await anext(first)
    try:
        events = [e async for e in service.stream_turn(request("same "))]
        assert events[0].event == "error" and len(repo.turns) == 1
    finally:
        await first.aclose()


async def test_cleanup_failure_still_releases_execution_right():
    """失败收尾异常不能永久锁住会话。参数：无；返回：None；异常：断言失败。"""
    service, repo, _, model = make_service(chunks=[RuntimeError("private")])
    async def broken_mark(context, status):
        """模拟收尾写失败。参数：上下文和状态；返回：不返回；异常：ConnectionError。"""
        raise ConnectionError("private")
    repo.mark_turn = broken_mark
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].event == "error"
    model.chunks = ["恢复"]
    assert [e async for e in service.stream_turn(request())][-1].event == "done"


def owned_call(order_id="2003", call_id="c1"):
    """构造订单查询申请。参数：订单号和调用 ID；返回：字典；异常：无。"""
    return {"id": call_id, "name": "query_order", "args": {"order_id": order_id}}


async def test_order_not_owned_bypasses_model_and_uses_fixed_reply():
    """归属拒绝不得交给模型改写，且必须完成持久化后再 done。参数：无；返回：None；异常：断言失败。"""
    service, repo, _, model = make_service(AIMessage(content="", tool_calls=[owned_call()]))
    events = [e async for e in service.stream_turn(request())]
    # 仍保留 running/succeeded 之外的失败状态徽章，随后是固定回复与完成帧。
    assert [e.event for e in events] == ["tool_status", "tool_status", "delta", "answer_metadata", "done"]
    assert events[1].data["status"] == "failed"
    assert events[2].data["delta"] == "没有找到您的这笔订单，请确认订单号或物流单号后重试。"
    # 只发生决策阶段调用，回答阶段被完全跳过。
    assert [step for step, _ in model.inputs] == ["decide"]
    turn = next(iter(repo.turns.values()))
    assert turn.turn_status == "completed" and turn.messages[-1].content == events[2].data["delta"]
    stored = next(iter(repo.results.values()))
    assert stored.ok is False and stored.error.code == "order_not_owned"


async def test_unknown_and_foreign_orders_produce_identical_reply():
    """未知订单与他人订单的回复必须逐字一致，不泄露存在性。参数：无；返回：None；异常：断言失败。"""
    replies = []
    for order_id in ("2003", "missing-order"):
        service, _, _, _ = make_service(AIMessage(content="", tool_calls=[owned_call(order_id)]))
        events = [e async for e in service.stream_turn(request())]
        replies.append(events[2].data["delta"])
    assert replies[0] == replies[1]
    assert all("2003" not in text and "不存在" not in text for text in replies)


async def test_faq_error_still_uses_model_answer():
    """非归属错误不能被固定回复劫持。参数：无；返回：None；异常：断言失败。"""
    service, _, _, model = make_service(AIMessage(content="", tool_calls=[faq_call(args={"keyword": " "})]))
    events = [e async for e in service.stream_turn(request())]
    # 空关键词仍会得到 INVALID_ARGUMENTS，非归属错误应交给回答阶段。
    assert [step for step, _ in model.inputs] == ["decide", "answer"]
    assert events[-1].event == "done"
    assert all(e.data.get("delta") != "没有找到您的这笔订单，请确认订单号或物流单号后重试。" for e in events)


async def test_faq_followup_accepts_keyword_absent_from_current_message():
    """包邮追问的查询词不在原句中时，真实编排仍应执行查询并保存成功事实。

    参数：无，使用原问题和模型曾提交的“运费”，仅替换外部模型与数据库。
    返回：None，断言工具成功、查询词传递正确且当前轮保存正常空命中。
    异常：AssertionError 表示原句硬校验再次拦截查询或破坏工具结果持久化。
    """
    service, repo, support, model = make_service(AIMessage(content="", tool_calls=[
        faq_call(args={"keyword": "运费"})]))
    events = [event async for event in service.stream_turn(ChatRequest(
        session_id="faq-followup", message="那如果没有满99元的话，我需要付什么费用？是多少"))]
    statuses = [event.data["status"] for event in events if event.event == "tool_status"]
    assert statuses == ["running", "succeeded"] and events[-1].event == "done"
    assert support.keywords == ["运费"]
    result = next(iter(repo.results.values()))
    assert result.ok and result.data["hits"] == [] and result.error is None
    assert [step for step, _ in model.inputs] == ["decide", "decide", "answer"]


def ticket_service(chunks):
    """构造工单成功的离线服务。参数：回答分片；返回：服务及仓库；异常：构造错误。"""
    from app.schemas.tools import ToolResult
    decision = AIMessage(content="", tool_calls=[{"id": "ticket-call", "name": "create_ticket",
        "args": {"description": "订单未送达", "ticket_type": "售后"}}])
    service, repo, support, model = make_service(decision, chunks)

    async def create_ticket(context, description, ticket_type):
        """原子保存模拟工单事实。参数：上下文、描述、类型；返回：成功结果；异常：保存错误。"""
        return await repo.save_tool_result(context, ToolResult.success({
            "ticket_no": "T-saved", "status": "待处理", "ticket_type": ticket_type}))

    support.create_ticket = create_ticket
    return service, repo, model


@pytest.mark.parametrize("chunks,fail_finish", [([RuntimeError("private")], False),
    (["部分回答", RuntimeError("private")], False), (["模型没复述编号"], True)])
async def test_saved_ticket_always_exposes_receipt_on_answer_failure(chunks, fail_finish):
    """已建单后的故障不隐藏编号或鼓励重试。参数：分片、写库故障；返回：None；异常：断言失败。"""
    service, repo, _ = ticket_service(chunks)
    repo.fail_finish = fail_finish
    events = [e async for e in service.stream_turn(request())]
    assert "T-saved" in "".join(e.data.get("delta", "") for e in events)
    assert events[-1].event == "error" and events[-1].data["retry_safe"] is False
    assert "T-saved" in events[-1].data["error"] and "勿重复" in events[-1].data["error"]
    assert not any(e.event == "done" for e in events)
    assert len(repo.results) == 1


async def test_saved_ticket_receipt_is_persisted_with_successful_answer():
    """正常回答也必须保存服务端编号提示。参数：无；返回：None；异常：断言失败。"""
    service, repo, _ = ticket_service(["请耐心等待"])
    events = [e async for e in service.stream_turn(request())]
    answer = "".join(e.data.get("delta", "") for e in events)
    assert "T-saved" in answer and events[-1].event == "done"
    assert next(iter(repo.turns.values())).messages[-1].content == answer


async def test_oversized_answer_stops_stream_and_preserves_ticket():
    """超长回答在内存及写库前截停。参数：无；返回：None；异常：断言失败。"""
    service, repo, _ = ticket_service(["好" * 12001])
    events = [e async for e in service.stream_turn(request())]
    answer = "".join(e.data.get("delta", "") for e in events)
    assert len(answer) <= 12000 and "T-saved" in answer
    assert events[-1].event == "error" and events[-1].data["retry_safe"] is False
    assert next(iter(repo.turns.values())).turn_status == "failed"


async def test_plain_failure_does_not_claim_saved_tool_results(caplog):
    """未执行工具时使用普通错误并记录轮次。参数：日志夹具；返回：None；异常：断言失败。"""
    service, repo, _, _ = make_service(decision=ValueError("private"))
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].data["retry_safe"] is True
    assert "已提交" not in events[-1].data["error"]
    assert repo.contexts[0].turn_id in caplog.text and "private" not in caplog.text


@pytest.mark.parametrize("text", ["好" * 22000, "🐱" * 17000], ids=["chinese", "emoji"])
async def test_repository_rejects_text_overflow_before_opening_transaction(text):
    """仓库按UTF-8字节拦截TEXT溢出。参数：超长正文；返回：None；异常：断言失败。"""
    from app.repositories.chat import ChatRepository
    from app.schemas.tools import ToolContext
    context = ToolContext(session_id="s", user_id="u", turn_id="t", user_message="问题")
    with pytest.raises(ValueError, match="正文过长"):
        await ChatRepository(None).finish_turn(context, text)


async def test_request_budget_rejects_decision_before_model_or_storage():
    """完整请求预算包含系统提示和工具定义。参数：无；返回：None；异常：断言失败。"""
    service, repo, _, model = make_service()
    service.settings.request_token_budget = 100
    events = [event async for event in service.stream_turn(request())]
    assert events[-1].event == "error"
    assert not repo.turns and not model.inputs


async def test_answer_stage_checks_its_own_complete_request_budget(monkeypatch):
    """回答阶段也必须核算系统提示，不能只检查决策。参数：替换器；返回：None；异常：断言失败。"""
    import app.services.chat as chat_module
    service, repo, _, model = make_service()
    # 回答阶段的 system 已独立为 ANSWER_SYSTEM；改它才能单独撑大回答请求。
    monkeypatch.setattr(chat_module, "ANSWER_SYSTEM", "答" * 9000)
    events = [event async for event in service.stream_turn(request())]
    assert events[-1].event == "error"
    assert [step for step, _ in model.inputs] == ["decide"]
    assert next(iter(repo.turns.values())).turn_status == "failed"


@pytest.mark.parametrize("budget_kind,spare", [("request", 0), ("dialogue", 0), ("request", 100)])
async def test_answer_retry_respects_both_input_budgets(monkeypatch, budget_kind, spare):
    """纠正重试必须预留预算，不能在检查后向上游发送超限请求。

    参数:
        monkeypatch: 仅替换模型网络边界与回答提示。
        budget_kind: 本用例贴近完整请求预算或对话预算。
        spare: 初次输入之外的可用余量；0 无法容纳纠正，100 足够。
    返回值:
        None: 断言超限时零网络调用，余量足够时两次请求均符合预算。
    异常:
        AssertionError: 超限仍发送请求、正常重试失败或修改用户原话。
    """
    from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
    from app.core import llm
    from app.core.memory import _estimate_message_tokens
    from app.services import chat as chat_module

    # 放大回答提示，使完整请求预算先约束回答阶段，决策阶段仍可正常完成。
    answer_system = "答" * 7000
    monkeypatch.setattr(chat_module, "ANSWER_SYSTEM", answer_system)
    service, repo, _, _ = make_service()
    original = request("retry-budget")
    user = HumanMessage(content=original.message)
    if budget_kind == "request":
        service.settings.request_token_budget = _estimate_message_tokens([
            SystemMessage(content=answer_system), user,
        ]) + spare
    else:
        service.settings.token_budget = _estimate_message_tokens([user])
    seen = []

    class NetworkModel:
        """只记录实际网络输入；首轮空正文，重试返回正文。"""

        async def astream(self, messages):
            """模拟网络流。参数：真实请求消息；返回：模型分片；异常：无。"""
            seen.append(list(messages))
            yield AIMessageChunk(content="" if len(seen) == 1 else "重试回答")

    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: NetworkModel())
    service.model.stream_answer = llm.ChatModelAdapter().stream_answer
    events = [event async for event in service.stream_turn(original)]
    for messages in seen:
        assert _estimate_message_tokens(messages) <= service.settings.request_token_budget
        assert _estimate_message_tokens(messages[1:]) <= service.settings.token_budget
        assert messages[1].content == original.message
    if spare == 0:
        assert seen == [] and events[-1].event == "error"
        assert next(iter(repo.turns.values())).turn_status == "failed"
    else:
        assert len(seen) == 2 and events[-1].event == "done"
        assert events[-3].data["delta"] == "重试回答"


async def test_answer_retry_reservation_trims_history_as_a_whole(monkeypatch):
    """为纠正预留预算时整轮裁旧历史，首轮与重试均保留当前问题且不超限。

    参数:
        monkeypatch: 替换回答提示及模型网络边界。
    返回值:
        None: 断言旧问答整轮裁掉，两次实际请求均符合预算，存储未改变。
    异常:
        AssertionError: 重试超限、历史只裁半轮或当前事实被纠正消息改写。
    """
    from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
    from app.core import llm
    from app.core.memory import _estimate_message_tokens
    from app.schemas.history import StoredMessage, StoredTurn
    from app.services import chat as chat_module

    answer_system = "答" * 7000
    monkeypatch.setattr(chat_module, "ANSWER_SYSTEM", answer_system)
    service, repo, _, _ = make_service()
    old_question, old_answer = "旧问题" * 30, "旧回答" * 30
    old = StoredTurn(turn_id="old", turn_status="completed", messages=[
        StoredMessage(role="user", content=old_question),
        StoredMessage(role="assistant", content=old_answer),
    ])
    repo.turns["old"] = old
    original = request("retry-history")
    # 此预算可刚好容纳旧轮及原请求，但无法再容纳纠正；必须裁掉完整旧轮。
    service.settings.request_token_budget = _estimate_message_tokens([
        SystemMessage(content=answer_system), HumanMessage(content=old_question),
        AIMessage(content=old_answer), HumanMessage(content=original.message),
    ])
    seen = []

    class NetworkModel:
        """首轮空响应、下一轮正文，记录真正送往网络的消息。"""

        async def astream(self, messages):
            """模拟网络分片。参数：请求；返回：分片流；异常：无。"""
            seen.append(list(messages))
            yield AIMessageChunk(content="" if len(seen) == 1 else "重试回答")

    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: NetworkModel())
    service.model.stream_answer = llm.ChatModelAdapter().stream_answer
    events = [event async for event in service.stream_turn(original)]
    assert len(seen) == 2 and events[-1].event == "done"
    assert len(seen[0]) == 2 and len(seen[1]) == 3
    for messages in seen:
        assert _estimate_message_tokens(messages) <= service.settings.request_token_budget
        assert messages[1].content == original.message
        assert all(message.content not in {old_question, old_answer} for message in messages)
    assert repo.turns["old"] == old
    current = next(turn for key, turn in repo.turns.items() if key != "old")
    assert current.messages[0].content == original.message
    assert all(message.content != llm.ChatModelAdapter.ANSWER_NUDGE for message in current.messages)


async def test_cancel_after_ticket_receipt_preserves_fact_without_answer_call():
    """编号发出后停止仍保留工单，且不继续调用回答模型。参数：无；返回：None；异常：断言失败。"""
    service, repo, model = ticket_service(["不应生成"])
    stream = service.stream_turn(request())
    try:
        while (event := await anext(stream)).event != "delta":
            pass
        assert "T-saved" in event.data["delta"]
    finally:
        await stream.aclose()
    assert len(repo.results) == 1
    assert [step for step, _ in model.inputs] == ["decide"]
    assert next(iter(repo.turns.values())).turn_status == "cancelled"
