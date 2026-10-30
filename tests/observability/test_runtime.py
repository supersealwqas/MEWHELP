"""CH09 使用真实 LangGraph、SDK 和可记录导出器核验原轮归属、用量及有界退出。"""

import asyncio
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.config import Settings
from app.schemas.tools import ToolContext
from app.services.chat import _WorkflowTrace
from app.workflow.state import ConversationState
from tests.test_context_repository import context_db


def runtime_module():
    """定位待实现运行时。参数：无；返回：模块；异常：断言表示 CH09 行为缺失。"""
    assert importlib.util.find_spec("app.observability") is not None, "尚未实现 CH09 原轮观测运行时"
    from app.observability import runtime
    return runtime


@pytest.fixture
def observed_runtime():
    """只替换外部导出端，保留真实 SDK。参数：无；返回：运行时/导出器；异常：配置或 SDK 错误。"""
    module = runtime_module()
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    config = Settings(_env_file=None, chat_base_url="http://localhost", chat_model="test", chat_api_key="test",
        observability_enabled=True, langfuse_base_url="http://127.0.0.1:13000",
        langfuse_public_key="pk-test-" + uuid4().hex, langfuse_secret_key="test-secret")
    runtime = module.ObservabilityRuntime(config, span_exporter=exporter)
    yield runtime, exporter
    runtime.client.shutdown()
    runtime.tracer_provider.shutdown()


def context(turn_id="original", session_id="session"):
    """构造可信测试身份。参数：原轮/会话键；返回：ToolContext；异常：字段校验错误。"""
    return ToolContext(conversation_id=1, session_id=session_id, user_id="user", turn_id=turn_id, user_message="你好")


def metadata(span):
    """读取真实 OTel 元数据属性。参数：完成 span；返回：字典；异常：非法 JSON。"""
    attrs = span.attributes
    result = json.loads(attrs.get("langfuse.observation.metadata", "{}"))
    for key, value in attrs.items():
        if key.startswith("langfuse.observation.metadata."):
            result[key.removeprefix("langfuse.observation.metadata.")] = value
    return result


def generations(exporter):
    """筛选可计费原始观测。参数：真实导出器；返回：generation 列表；异常：无。"""
    return [span for span in exporter.get_finished_spans()
            if span.attributes.get("langfuse.observation.type") == "generation"]


async def test_concurrent_turns_and_resume_keep_separate_trace_trees(observed_runtime):
    """当前身份共享或恢复丢父 ID 时失败。参数：运行时夹具；返回：None；异常：断言失败。"""
    runtime, exporter = observed_runtime
    model = FakeMessagesListChatModel(responses=[AIMessage(content="回答", usage_metadata={
        "input_tokens": 11, "output_tokens": 3, "total_tokens": 14}, response_metadata={"model_name": "test-model"})])

    async def classify(state):
        """先产生模型消费再保存意图。参数：状态；返回：分类字段；异常：模型错误。"""
        await model.ainvoke("分类前调用")
        runtime.record_intent("闲聊", 0.9, "chitchat")
        return {"intent": "闲聊"}

    async def confirm(state):
        """模拟原轮确认后才调用模型。参数：状态；返回：正文；异常：正常 interrupt。"""
        interrupt("确认")
        result = await model.ainvoke("确认后调用")
        return {"answer": result.content}

    builder = StateGraph(ConversationState)
    builder.add_node("classify_intent", classify)
    builder.add_node("answer", confirm)
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=InMemorySaver()).with_config(callbacks=[runtime.callback_handler])
    first, other = runtime.identity_for(context()), runtime.identity_for(context("other", "another"))

    async def execute(identity, initial, kind):
        """在真实图调用边界建立上下文。参数：身份/输入/执行类别；返回：None；异常：图错误。"""
        trace = _WorkflowTrace()
        with runtime.execution(identity, kind=kind):
            await graph.ainvoke(initial, {"configurable": {"thread_id": identity.turn_id}, "callbacks": [trace]})
        assert not trace.running

    await asyncio.gather(execute(first, {"context": context().model_dump()}, "chat"),
                         execute(other, {"context": context("other", "another").model_dump()}, "chat"))
    resumed = runtime_module().TraceIdentity(**json.loads(json.dumps(asdict(first))))
    await execute(resumed, Command(resume=True), "resume")
    runtime.client.flush()
    assert first.trace_id == resumed.trace_id != other.trace_id
    calls = generations(exporter)
    assert len(calls) == 3
    assert len({span.context.span_id for span in calls}) == 3
    first_spans = [span for span in exporter.get_finished_spans() if f"{span.context.trace_id:032x}" == first.trace_id]
    roots = [span for span in first_spans if span.name.startswith("execution.")]
    assert len(roots) == 2
    root = next(span for span in roots if span.name == "execution.chat")
    resume = next(span for span in roots if span.name == "execution.resume")
    assert metadata(root)["intent"] == "闲聊"
    assert resume.parent.span_id == root.context.span_id == int(first.root_observation_id, 16)
    assert len([span for span in calls if f"{span.context.trace_id:032x}" == first.trace_id]) == 2


async def test_compiled_workflow_keeps_one_handler_and_original_trace(observed_runtime, caplog):
    """生产构图漏挂或请求回调覆盖应用回调时失败。参数：运行时/日志；返回：None；异常：断言失败。"""
    from app.workflow.graph import build_workflow
    from tests.workflow.graph_fakes import dependencies, turn
    runtime, exporter = observed_runtime
    deps = dependencies(intents=["闲聊"])
    deps.observability = runtime
    graph = build_workflow(deps, checkpointer=InMemorySaver())
    state = turn("你好")
    with runtime.execution(runtime.identity_for(ToolContext.model_validate(state["context"])), kind="chat"):
        with caplog.at_level("INFO", logger="app.services.chat"):
            await graph.ainvoke(state, {"configurable": {"thread_id": "production"}, "callbacks": [_WorkflowTrace()]})
    runtime.client.flush()
    spans = [span for span in exporter.get_finished_spans() if span.name == "classify_intent"]
    assert len(spans) == 1
    assert "node=classify_intent phase=enter" in caplog.text
    roots = [span for span in exporter.get_finished_spans() if span.name == "execution.chat"]
    assert metadata(roots[0])["intent"] == "闲聊"


@pytest.mark.parametrize("usage, origin", [(None, "inferred_or_unknown"), (
    {"input_tokens": 0, "output_tokens": 3, "total_tokens": 3}, "model_reported")])
async def test_generation_preserves_reported_usage_presence(observed_runtime, usage, origin):
    """自动估算掩盖缺失 usage 或零 token 被视为缺失时失败。参数：导出器/用量；返回：None；异常：断言失败。"""
    runtime, exporter = observed_runtime
    model = FakeMessagesListChatModel(responses=[AIMessage(content="答复", usage_metadata=usage,
        response_metadata={"model_name": "test-model"})]).with_config(callbacks=[runtime.callback_handler])
    with runtime.execution(runtime.identity_for(context()), kind="chat"):
        await model.ainvoke("输入")
    runtime.client.flush()
    calls = generations(exporter)
    assert len(calls) == 1
    assert metadata(calls[0])["usage_origin"] == origin
    assert metadata(calls[0])["input_usage_present"] is (usage is not None)
    assert metadata(calls[0])["output_usage_present"] is (usage is not None)


async def test_completed_receipt_replay_does_not_create_generation(observed_runtime):
    """完成工单回放重新执行模型时失败。参数：运行时；返回：None；异常：断言失败。"""
    from app.schemas.workflow import TicketResume
    from app.services.chat import ChatService
    runtime, exporter = observed_runtime
    service = ChatService(None, None, None, None, observability=runtime)
    request = TicketResume(type="confirm_ticket", session_id="session", user_id="user", source_turn_id="original",
                           action_id=str(uuid5(NAMESPACE_URL, "test")), decision="confirm")
    row = {"context": context(), "turn_status": "completed", "answer": "已创建工单",
           "completion_metadata": {"status": "completed"}}
    events = [event async for event in service._ticket_replay(request, row)]
    runtime.client.flush()
    assert [event.event for event in events] == ["delta", "answer_metadata", "done"]
    assert len(generations(exporter)) == 0


async def test_shutdown_export_failure_does_not_break_turn_cleanup(observed_runtime, tmp_path, monkeypatch, caplog):
    """导出失败中断原轮取消事务时失败。参数：运行时/目录/补丁/日志；返回：None；异常：受控取消。"""
    from tests.workflow.service_fakes import collect, service_runtime
    runtime, exporter = observed_runtime
    from opentelemetry.sdk.trace.export import SpanExportResult
    monkeypatch.setattr(exporter, "export", lambda spans: SpanExportResult.FAILURE)
    from app.schemas.workflow import AgentDecision
    async with service_runtime(tmp_path, intents=["订单"], decisions=[AgentDecision(kind="answer")]) as (service, deps, _):
        service.observability = runtime
        entered, closed = asyncio.Event(), asyncio.Event()

        async def answer(messages, ledger):
            """阻塞模型，验证取消确实释放流。参数：输入/预算；返回：流；异常：取消。"""
            entered.set()
            try:
                await asyncio.Event().wait()
                yield "不可到达"
            finally:
                closed.set()

        deps.model.stream_answer = answer
        task = asyncio.create_task(collect(service))
        await asyncio.wait_for(entered.wait(), 3)
        saved = await deps.chat.get_workflow_turn("s", "demo-user")
        assert len(saved["trace_identity"]["root_observation_id"]) == 16
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await runtime.aclose()
        stream_closed = closed.is_set()
        transaction_finished = (await deps.chat.load_turns(1))[0].turn_status == "cancelled"
        exporter_failure_visible = runtime.health["status"] == "export_failed"
        assert stream_closed and transaction_finished and exporter_failure_visible
        assert "observability export_failed" in caplog.text


def test_blocked_exporter_process_exits_within_shutdown_deadline(tmp_path):
    """只超时协程却让 SDK atexit 或非守护线程卡住进程时失败。参数：目录；返回：None；异常：子进程超时。"""
    runtime_module()
    script = tmp_path / "blocked_exporter.py"
    script.write_text('''"""独立进程中模拟永久阻塞的真实 SDK 导出器。"""
import asyncio
import threading
from app.config import Settings
from app.observability.runtime import ObservabilityRuntime
from app.schemas.tools import ToolContext
from opentelemetry.sdk.trace.export import SpanExporter

class BlockedExporter(SpanExporter):
    """永久阻塞导出，不以协程假替身冒充线程。"""
    def export(self, spans):
        """参数：待导出spans；返回：不返回；异常：无。"""
        threading.Event().wait()

async def main():
    """参数：无；返回：None；异常：断言失败。"""
    runtime = ObservabilityRuntime(Settings(_env_file=None, chat_base_url="http://localhost", chat_model="test", chat_api_key="test",
        observability_enabled=True, langfuse_base_url="http://127.0.0.1:13000", langfuse_public_key="blocked-test",
        langfuse_secret_key="test", observability_shutdown_timeout_seconds=0.2), span_exporter=BlockedExporter())
    with runtime.execution(runtime.identity_for(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="q")), kind="chat"):
        pass
    await runtime.aclose()
    assert runtime.health["status"] == "shutdown_timeout"
    print("business-cleanup-finished; exporter-thread-not-reclaimed", flush=True)

asyncio.run(main())
''', encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], cwd=".", env={**__import__("os").environ, "PYTHONPATH": "."},
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert "business-cleanup-finished; exporter-thread-not-reclaimed" in result.stdout


async def test_original_root_identity_is_saved_before_model_and_preserves_metadata(tmp_path):
    """首个检查点前取消丢根或覆盖原轮元数据时失败。参数：目录；返回：None；异常：断言失败。"""
    from sqlalchemy import select
    from app.models.customer_service import Message
    from tests.workflow.service_fakes import service_runtime
    async with service_runtime(tmp_path) as (service, deps, factory):
        trusted = await deps.chat.start_turn(context(), "你好")
        assert hasattr(deps.chat, "save_trace_identity"), "首次模型前需要持久保存原轮追踪根"
        async with factory.begin() as session:
            source = await session.scalar(select(Message).where(Message.role == "user"))
            source.rag_metadata = {"existing_field": "保留"}
        identity = {"trace_id": "1" * 32, "root_observation_id": "2" * 16,
                    "session_id": trusted.session_id, "turn_id": trusted.turn_id, "user_id": trusted.user_id}
        saved = await deps.chat.save_trace_identity(trusted, identity)
        assert saved == identity
        assert await deps.chat.save_trace_identity(trusted, {**identity, "root_observation_id": "3" * 16}) == identity
        async with factory() as session:
            source = await session.scalar(select(Message).where(Message.role == "user"))
            assert source.rag_metadata["existing_field"] == "保留"
            assert source.rag_metadata["trace_identity"] == identity
        row = await deps.chat.get_workflow_turn(trusted.session_id, trusted.user_id, trusted.turn_id)
        assert row["trace_identity"] == identity
        with pytest.raises(ValueError):
            await deps.chat.save_trace_identity(trusted.model_copy(update={"user_message": "伪造原问"}), identity)


async def test_summary_model_is_a_separate_background_trace(observed_runtime, context_db):
    """摘要继承客服trace或漏掉真实模型回调时失败。参数：运行时/真实SQLite；返回：None；异常：断言失败。"""
    from tests.test_summaries import ControlledGenerator, manager, seed_pending, wait_idle
    runtime, exporter = observed_runtime
    ctx, _ = await seed_pending(context_db)
    model = FakeMessagesListChatModel(responses=[AIMessage(content="模型摘要", response_metadata={"model_name": "summary-model"})])

    class Generator(ControlledGenerator):
        """只替换模型网络，摘要任务和持久保存保持真实。"""

        async def generate(self, messages, background):
            """参数：冻结消息与背景；返回：合法草稿；异常：模型错误。"""
            await model.ainvoke("摘要输入")
            return await super().generate(messages, background)

    worker = manager(context_db, Generator(), observability=runtime)
    identity = runtime.identity_for(context())
    try:
        with runtime.execution(identity, kind="chat"):
            runtime.record_intent("订单", 0.8, "business")
            assert await worker.maybe_schedule(ctx.conversation_id)
            await wait_idle(worker)
    finally:
        await worker.close()
    runtime.client.flush()
    calls = generations(exporter)
    assert len(calls) == 1
    assert f"{calls[0].context.trace_id:032x}" != identity.trace_id
    assert metadata(calls[0])["task_kind"] == "summary"
    assert metadata(calls[0])["intent"] == "unclassified"


async def test_actual_tool_retry_and_retrieval_results_are_recorded(observed_runtime):
    """工具重试、召回原分数或原文未进入观测时失败。参数：运行时；返回：None；异常：断言失败。"""
    import httpx
    from app.tools.executor import ToolExecutor
    from app.tools.registry import ToolRegistry
    from app.schemas.tools import ToolResult
    from tests.tools.test_execution import definition
    from tests.test_tool_executor import ResultStore, call, context as tool_context, ignore_status
    from tests.rag.test_retrieval import _chunk, _search
    runtime, exporter = observed_runtime
    attempts = 0

    async def handler(arguments, ctx, scope):
        """模拟一次真实读取超时。参数：工具参数/身份/资源；返回：结果；异常：首次超时。"""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("读取超时")
        return ToolResult.success({"verified": True})

    registry = ToolRegistry()
    registry.register_definition(definition(handler))
    executor = ToolExecutor(registry, ResultStore(), None, retry_delay_seconds=0)
    search, _, _, _ = _search([_chunk(1)], scores={1: 0.75})
    with runtime.execution(runtime.identity_for(context()), kind="chat"):
        result = await executor.execute(call("probe", {}), tool_context(), ignore_status)
        ranked = await search._retrieve_one("实际查询", 1)
    runtime.client.flush()
    assert result.ok and ranked[0].score == 0.75
    tool_span = next(span for span in exporter.get_finished_spans() if span.name == "tool.execute")
    retrieval = next(span for span in exporter.get_finished_spans() if span.name == "knowledge.retrieve")
    assert metadata(tool_span)["attempts"] == 2
    assert metadata(tool_span)["preflight_allowed"] is True
    assert metadata(tool_span)["last_attempt_error"] == "ReadTimeout"
    candidates = metadata(retrieval)["recall_candidates"]
    candidates = json.loads(candidates) if isinstance(candidates, str) else candidates
    assert candidates[0]["score"] == 0.01
    assert json.loads(retrieval.attributes["langfuse.observation.output"])["ranked"][0]["chunk"]["text"]["answer"] == "权威答案1。"
    assert not generations(exporter)


async def test_confirmation_interrupt_is_observed_as_pause_not_failure(observed_runtime):
    """正常确认暂停被错误计为依赖失败时失败。参数：运行时；返回：None；异常：受控GraphInterrupt。"""
    from langgraph.errors import GraphInterrupt
    runtime, exporter = observed_runtime
    with runtime.execution(runtime.identity_for(context()), kind="chat"):
        with pytest.raises(GraphInterrupt):
            with runtime.observe("waiting-tool", kind="tool", input={}):
                raise GraphInterrupt(())
    runtime.client.flush()
    span = next(item for item in exporter.get_finished_spans() if item.name == "waiting-tool")
    assert span.attributes["langfuse.observation.level"] == "DEFAULT"
    assert span.attributes["langfuse.observation.status_message"] == "paused"


async def test_streaming_model_and_following_retrieval_stay_inside_live_tool(observed_runtime):
    """SDK流结束后遗留已结束generation上下文、忽略手工工具边界时失败。参数：运行时；返回：None；异常：断言失败。"""
    runtime, exporter = observed_runtime
    model = FakeMessagesListChatModel(responses=[AIMessage(content="扩写", response_metadata={"model_name": "test-model"})])

    async def node(state):
        """实际流调用后继续非Runnable检索。参数：状态；返回：正文；异常：模型错误。"""
        with runtime.observe("tool-test", kind="tool", input={}):
            async for _ in model.astream("扩写输入"):
                pass
            with runtime.observe("retrieval-test", kind="retriever", input={}):
                pass
        return {"answer": "完成"}

    graph = StateGraph(ConversationState)
    graph.add_node("tools", node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    compiled = graph.compile().with_config(callbacks=[runtime.callback_handler])
    with runtime.execution(runtime.identity_for(context()), kind="chat"):
        await compiled.ainvoke({})
    runtime.client.flush()
    spans = exporter.get_finished_spans()
    tool = next(span for span in spans if span.name == "tool-test")
    retrieval = next(span for span in spans if span.name == "retrieval-test")
    assert generations(exporter)[0].parent.span_id == tool.context.span_id
    assert retrieval.parent.span_id == tool.context.span_id


async def test_classified_intent_cannot_be_overwritten_by_propagated_placeholder(observed_runtime):
    """本机服务端用初始trace元数据覆盖最新观测意图时失败。参数：运行时；返回：None；异常：断言失败。"""
    runtime, exporter = observed_runtime
    model = FakeMessagesListChatModel(responses=[AIMessage(content="分类前调用")])

    async def classify(state):
        """真实SDK链先传播未分类再更新根。参数：状态；返回：新意图；异常：模型回调错误。"""
        await model.ainvoke("分类前输入")
        runtime.record_intent("商品咨询", 0.95, "knowledge")
        return {"intent": "商品咨询"}

    builder = StateGraph(ConversationState)
    builder.add_node("classify_intent", classify)
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", END)
    graph = builder.compile().with_config(callbacks=[runtime.callback_handler])
    with runtime.execution(runtime.identity_for(context()), kind="chat"):
        await graph.ainvoke({})
    runtime.client.flush()
    root = next(item for item in exporter.get_finished_spans() if item.name == "execution.chat")
    assert metadata(root)["intent"] == "商品咨询"
    # Langfuse v4服务端合并trace.metadata与observation.metadata，前者不能带旧的占位分类。
    assert root.attributes.get("langfuse.trace.metadata.intent", "商品咨询") == "商品咨询"
    assert all(item.attributes.get("langfuse.trace.metadata.intent") != "unclassified"
               for item in exporter.get_finished_spans())
