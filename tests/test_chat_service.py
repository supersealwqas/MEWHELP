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
    config = SimpleNamespace(token_budget=2000, tool_timeout_seconds=0.05,
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
    assert '"data":[]' in results[0].content
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
async def test_broken_protocol_never_reaches_answer_or_tools(decision):
    """缺 ID、重 ID、坏 JSON 不能形成损坏的模型请求。参数：决策；返回：None；异常：断言失败。"""
    service, repo, support, model = make_service(decision)
    events = [e async for e in service.stream_turn(request())]
    assert events[-1].event == "error" and not support.keywords
    assert len(model.inputs) == 1 and next(iter(repo.turns.values())).turn_status == "failed"


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
