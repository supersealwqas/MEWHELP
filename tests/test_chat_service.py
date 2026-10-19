"""单工具聊天编排的行为测试：协议、增量、故障、并发与取消。"""

import asyncio
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from app.schemas.chat import ChatRequest
from app.tools.registry import build_registry
from tests.chat_fakes import FAQRepository, MemoryRepository, ScriptedModel


def make_service(decision=None, chunks=None):
    """建立真实服务及外部替身。参数：决策、增量；返回：服务和依赖；异常：导入错误。"""
    from app.services.chat import ChatService
    repo, support, model = MemoryRepository(), FAQRepository(), ScriptedModel(decision, chunks)
    config = SimpleNamespace(token_budget=2000, request_token_budget=16000, tool_timeout_seconds=0.05,
                             tool_max_retries=1, tool_retry_delay_seconds=0)
    return ChatService(repo, support, model, build_registry(), config), repo, support, model


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
    assert [e.event for e in events] == ["tool_status", "tool_status", "delta", "delta", "done"]
    assert [e.data["status"] for e in events[:2]] == ["running", "succeeded"]
    assert support.keywords == ["邮费"]
    assert [step for step, _ in model.inputs] == ["decide", "answer"]
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
    assert (await anext(iterator)).event == "done"
    assert next(iter(repo.turns.values())).turn_status == "completed"
    await iterator.aclose()
    assert len(model.inputs) == 2
    assert all("规划" not in m.content for m in next(iter(repo.turns.values())).messages)


async def test_multiple_calls_execute_none_but_pair_all_results():
    """多申请不能偷偷执行第一项。参数：无；返回：None；异常：断言失败。"""
    service, repo, support, model = make_service(AIMessage(content="", tool_calls=[faq_call(), faq_call("c2")]))
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].event == "done" and not support.keywords
    results = [m for m in model.inputs[-1][1] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in results} == {"c1", "c2"}
    assert all("SINGLE_TOOL_LIMIT" in m.content for m in results)


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
    assert [e.event for e in events] == ["tool_status", "tool_status", "delta", "done"]
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
    service, _, _, model = make_service(AIMessage(content="", tool_calls=[faq_call(args={"keyword": "运费"})]))
    events = [e async for e in service.stream_turn(request())]
    # query_faq 关键词不在原话中会得到 INVALID_ARGUMENTS，仍应走回答阶段。
    assert [step for step, _ in model.inputs] == ["decide", "answer"]
    assert events[-1].event == "done"
    assert all(e.data.get("delta") != "没有找到您的这笔订单，请确认订单号或物流单号后重试。" for e in events)


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
    monkeypatch.setattr(chat_module, "FINAL_ANSWER_SYSTEM", "答" * 9000)
    events = [event async for event in service.stream_turn(request())]
    assert events[-1].event == "error"
    assert [step for step, _ in model.inputs] == ["decide"]
    assert next(iter(repo.turns.values())).turn_status == "failed"


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
