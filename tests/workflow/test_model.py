"""验证严格工作流协议、真实 SDK 流式发包与请求独占的改写预算。"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI

from app.schemas.chat import InvalidToolDecision
from app.tools.business import query_order, query_logistics, query_product, create_ticket


def test_intent_parser_accepts_only_exact_enum_object():
    """合法中文枚举无法解析时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.model import parse_intent

    assert parse_intent('{"intent":"物流"}').intent == "物流"


@pytest.mark.parametrize("raw", ['{"intent":"物流"', '{"intent":"物流","intent":"订单"}',
    '{"intent":NaN}', '{"intent":"物流","reason":"猜测"}', '[]', '{"intent":"未知"}',
    '```json\n{"intent":"物流"}\n```'])
def test_intent_parser_never_repairs_or_guesses(raw):
    """损坏 JSON 被修补或未知意图被默认映射时失败。参数：raw 输出；返回：None；异常：断言失败。"""
    from app.workflow.model import parse_intent

    with pytest.raises(InvalidToolDecision):
        parse_intent(raw)


@pytest.mark.parametrize("raw", ['{"kind":"answer","suggested_actions":["delete_order"]}',
    '{"kind":"answer","suggested_actions":["handoff","handoff"]}',
    '{"kind":"answer","suggested_actions":[],"answer":"内部正文"}',
    '{"kind":"answer"}', '{"kind":"answer","kind":"clarify","suggested_actions":[]}',
    '{"kind":"tools","tool_calls":[],"suggested_actions":[]}',
    '{"kind":"answer","suggested_actions":[NaN]}', '{}', ''])
def test_decision_parser_rejects_nonterminal_or_ambiguous_metadata(raw):
    """未知动作、重复键或缺字段终止进入工作流时失败。参数：raw 输出；返回：None；异常：断言失败。"""
    from app.workflow.model import parse_decision

    with pytest.raises(InvalidToolDecision):
        parse_decision(raw)


def test_terminal_decision_contains_no_final_body():
    """内部终止元数据错误夹带正文或丢动作时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.model import parse_decision

    result = parse_decision('{"kind":"clarify","suggested_actions":["handoff","create_ticket"]}')
    assert result.kind == "clarify" and result.tool_calls == []
    assert result.suggested_actions == ["handoff", "create_ticket"]


def sse(delta=None, *, finish=None, usage=None, tail=False):
    """构造真实 SSE JSON 块。参数：delta 增量、finish 原因、usage 用量、tail 无 choices；返回：dict；异常：无。"""
    return {"id":"ch05-offline", "object":"chat.completion.chunk", "created":0, "model":"test-model",
            "choices":[] if tail else [{"index":0, "delta":delta or {}, "finish_reason":finish, "logprobs":None}],
            "usage":usage}


def text_response(content, finish="stop", *, usage=False):
    """构造流式正文与终止块。参数：content 文本、finish 原因、usage 是否尾块；返回：列表；异常：无。"""
    chunks = [sse({"role":"assistant", "content":content}), sse(finish=finish)]
    if usage:
        chunks.append(sse(tail=True, usage={"prompt_tokens":100,"completion_tokens":12,"total_tokens":112}))
    return chunks


def call_response(name="query_order", raw='{"order_id":"1001"}', *, duplicate=False):
    """构造原始工具分片。参数：name 工具、raw 参数、duplicate 是否重复 ID；返回：列表；异常：无。"""
    calls = [{"index":0,"id":"call-1","type":"function","function":{"name":name,"arguments":raw}}]
    if duplicate:
        calls.append({"index":1,"id":"call-1","type":"function","function":{"name":name,"arguments":raw}})
    return [sse({"role":"assistant","tool_calls":calls}), sse(finish="tool_calls")]


@pytest.fixture
async def sdk_runtime():
    """仅替代外部 HTTP，保留 LangChain 与 SDK 实际序列化。参数：无；返回：网络脚本边界；异常：SDK 错误。"""
    requests, responses, options = [], [], []

    async def handle(request):
        """记录实际发包。参数：request HTTP 请求；返回：SSE 响应；异常：超脚本次数时断言失败。"""
        requests.append(json.loads(request.content))
        assert responses, "不允许额外模型请求"
        body = "".join("data: " + json.dumps(item) + "\n\n" for item in responses.pop(0)) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type":"text/event-stream"}, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        def factory(**kwargs):
            """使用真实客户端和外部假网络。参数：kwargs 适配器配置；返回：ChatOpenAI；异常：配置错误。"""
            options.append(kwargs)
            return ChatOpenAI(base_url="http://offline.invalid/v1", api_key="offline", model="test-model",
                              http_async_client=client, **kwargs)
        yield SimpleNamespace(factory=factory, requests=requests, responses=responses, options=options)


async def test_classify_counts_usage_tail_and_keeps_full_history(sdk_runtime):
    """无 choices 用量尾块被漏记或输入丢历史时失败。参数：sdk_runtime 网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    rt, ledger = sdk_runtime, BudgetLedger()
    rt.responses.append(text_response('{"intent":"物流"}', usage=True))
    history = [{"role":"user","content":"我要查物流"}, {"role":"assistant","content":"请提供订单号。"},
               {"role":"user","content":"1001"}]
    result = await WorkflowModelAdapter(model_factory=rt.factory, stream_usage=True).classify(history, ledger)
    assert result.intent == "物流" and ledger.snapshot()["actual_tokens"] == 112
    assert rt.requests[0]["messages"][1:] == history
    assert rt.requests[0]["stream"] is True and rt.requests[0]["temperature"] == 0
    assert rt.requests[0]["max_completion_tokens"] == 128
    assert rt.requests[0]["stream_options"] == {"include_usage":True}
    assert "response_format" not in rt.requests[0] and "tools" not in rt.requests[0]


@pytest.mark.parametrize("name,raw,duplicate", [("query_order",'{"order_id":"1001"',False),
    ("query_order",'{"order_id":"1001","order_id":"1002"}',False),
    ("query_order",'{"order_id":NaN}',False), ("query_order",'{"order_id":1e999}',False),
    ("query_order",'{"order_id":"1001"}',True), ("unknown",'{}',False),
    ("create_ticket",'{"description":"投诉","ticket_type":"投诉"}',False),
    ("query_order",'{"order_id":"1001","user_id":"victim"}',False),
    ("query_order",'{"order_id":1001}',False), ("query_order",'{}',False)])
async def test_workflow_rejects_invalid_readonly_batch(sdk_runtime, name, raw, duplicate):
    """坏参数、重复 ID、未知或写工具通过时失败。参数：网络夹具、工具名/参数/重复标记；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.append(call_response(name, raw, duplicate=duplicate))
    with pytest.raises(InvalidToolDecision):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order, query_logistics, query_product], BudgetLedger())
    assert len(sdk_runtime.requests) == 1


async def test_valid_tool_decision_preserves_raw_metadata_and_no_hidden_schema(sdk_runtime):
    """工具决定丢结果字段、身份泄露或用量记错时失败。参数：sdk_runtime 网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    ledger = BudgetLedger()
    sdk_runtime.responses.append(call_response())
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], ledger)
    assert result.kind == "tools" and result.tool_calls[0]["args"] == {"order_id":"1001"}
    assert result.tool_calls[0]["id"] == "call-1" and result.suggested_actions == []
    assert ledger.snapshot()["records"][0]["finish_reason"] == "tool_calls"
    assert ledger.snapshot()["actual_tokens"] is None
    request = sdk_runtime.requests[0]
    assert request["max_completion_tokens"] == 2048
    assert "user_id" not in request["tools"][0]["function"]["parameters"]["properties"]
    assert "stream_options" not in request


@pytest.mark.parametrize("stage", ["classify","decide","answer"])
async def test_length_finish_is_never_normal_completion(sdk_runtime, stage):
    """SDK 返回 length 却当正常完成时失败。参数：网络夹具、阶段；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter, WorkflowOutputError

    payload = {"classify":'{"intent":"物流"}', "decide":'{"kind":"answer","suggested_actions":[]}', "answer":"截断的正文"}[stage]
    sdk_runtime.responses.append(text_response(payload, "length"))
    ledger, adapter = BudgetLedger(), WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    with pytest.raises(WorkflowOutputError):
        if stage == "classify":
            await adapter.classify([], ledger)
        elif stage == "decide":
            await adapter.decide([], [], ledger)
        else:
            [delta async for delta in adapter.stream_answer([], ledger)]
    assert ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["records"][0]["finish_reason"] == "length"


async def test_no_headroom_stops_before_any_network_request(sdk_runtime):
    """只够决策、不够两次正文却发包时失败。参数：网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
    from app.workflow.model import WorkflowModelAdapter

    with pytest.raises(WorkflowBudgetExceeded):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [], BudgetLedger(max_model_calls=2))
    assert sdk_runtime.requests == []


async def test_answer_is_unbound_real_stream_with_one_counted_empty_retry(sdk_runtime):
    """正文仍输出内部 JSON、绑定工具或免费重试时失败。参数：网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
    from app.workflow.model import WorkflowModelAdapter

    rt, ledger = sdk_runtime, BudgetLedger()
    rt.responses.extend([text_response(""), text_response("请提供订单号。")])
    adapter = WorkflowModelAdapter(model_factory=rt.factory)
    assert [delta async for delta in adapter.stream_answer([HumanMessage(content="查物流")], ledger)] == ["请提供订单号。"]
    assert ledger.snapshot()["model_calls"] == 2
    assert all(request["stream"] and request["max_completion_tokens"] == 4096 and "tools" not in request for request in rt.requests)
    with pytest.raises(WorkflowBudgetExceeded):
        [delta async for delta in adapter.stream_answer([], ledger)]
    assert len(rt.requests) == 2


async def test_answer_yields_before_upstream_finishes_and_close_closes_upstream():
    """聚合完整答案才逐字输出或关闭外层不释放上游时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    closed = []
    class WaitingModel:
        """首块后等待的外部流，用于证明真正按需输出及关闭。"""
        async def astream(self, messages):
            """参数：messages 完整输入；返回：分片流；异常：取消原样传播。"""
            try:
                yield AIMessageChunk(content=[{"type":"reasoning","text":"SECRET"}, {"type":"text","text":"首块"}])
                await asyncio.Event().wait()
            finally:
                closed.append(True)
    adapter = WorkflowModelAdapter(model_factory=lambda **kwargs: WaitingModel())
    stream = adapter.stream_answer([], BudgetLedger())
    assert await asyncio.wait_for(anext(stream), timeout=1) == "首块"
    await stream.aclose()
    assert closed == [True]


async def test_old_adapter_keeps_aggregated_usage_and_rejects_duplicate_keys(monkeypatch):
    """兼容适配器重建消息丢 metadata 或容忍重复键时失败。参数：monkeypatch 替换器；返回：None；异常：断言失败。"""
    from app.core import llm
    from tests.test_model_adapter import ChunkModel

    model = ChunkModel([AIMessageChunk(content="完成", response_metadata={"finish_reason":"stop"}),
                        AIMessageChunk(content="", usage_metadata={"input_tokens":10,"output_tokens":2,"total_tokens":12})])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: model)
    result = await llm.ChatModelAdapter().decide([], [])
    assert result.usage_metadata["total_tokens"] == 12 and result.response_metadata["finish_reason"] == "stop"
    model.chunks = [AIMessageChunk(content="", tool_call_chunks=[{"name":"query_order","id":"x",
        "args":'{"order_id":"1001","order_id":"1002"}',"index":0}])]
    with pytest.raises(InvalidToolDecision):
        await llm.ChatModelAdapter().decide([], [])


async def test_shared_retrieval_keeps_ledgers_request_local():
    """共享检索器临时换计量器导致串轮或漏计退款改写时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
    from app.knowledge.query_rewrite import QueryRewriter
    from tests.rag.test_retrieval import _chunk, _search
    from tests.rag.test_query_rewrite import RewriteModel

    search, _, _, _ = _search([_chunk(1)])
    model = RewriteModel('{"standard_query":"退款怎么申请","synonyms":[]}')
    search.query_rewriter = QueryRewriter(model.factory)
    first, second = BudgetLedger(), BudgetLedger()
    await asyncio.gather(search.retrieve("退款怎么申请", budget=first), search.retrieve_policy("退款怎么申请", budget=second))
    assert first.snapshot()["model_calls"] == second.snapshot()["model_calls"] == 1
    assert len(model.inputs) == 2 and not hasattr(search, "budget")
    with pytest.raises(WorkflowBudgetExceeded):
        await search.retrieve("退款怎么申请", budget=BudgetLedger(total_token_budget=1))
    assert len(model.inputs) == 2


async def test_answer_preserves_whitespace_chunks_between_words(sdk_runtime):
    """逐块 strip 使单词粘连或段落消失时失败。参数：sdk_runtime 网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.append([sse({"content":text}) for text in ["Hello", " ", "world", "\n\n", "下一段"]] + [sse(finish="stop")])
    chunks = [part async for part in WorkflowModelAdapter(model_factory=sdk_runtime.factory).stream_answer([], BudgetLedger())]
    assert "".join(chunks) == "Hello world\n\n下一段"


async def test_answer_whitespace_only_is_empty_and_retry_remains_bounded(sdk_runtime):
    """纯空白被当有效答案或补救无次数上限时失败。参数：sdk_runtime 网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter, WorkflowOutputError

    sdk_runtime.responses.extend([text_response(" \n"), text_response("\t")])
    ledger = BudgetLedger()
    with pytest.raises(WorkflowOutputError):
        [part async for part in WorkflowModelAdapter(model_factory=sdk_runtime.factory).stream_answer([], ledger)]
    assert ledger.snapshot()["model_calls"] == 2


async def test_cancel_while_classifying_keeps_reservation_and_closes_stream():
    """等待网络时取消未关闭流或退还已发请求占额时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    started, closed = asyncio.Event(), []
    class WaitingModel:
        """在首块前阻塞的外部网络边界。"""
        async def astream(self, messages):
            """参数：messages 完整输入；返回：分片流；异常：取消原样传播。"""
            try:
                started.set()
                await asyncio.Event().wait()
                yield AIMessageChunk(content="")
            finally:
                closed.append(True)
    ledger = BudgetLedger()
    task = asyncio.create_task(WorkflowModelAdapter(model_factory=lambda **kwargs: WaitingModel()).classify([], ledger))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True] and ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["reserved_units"] > 0 and ledger.snapshot()["actual_tokens"] is None


async def test_full_input_is_not_silently_trimmed_and_writes_never_bound(sdk_runtime):
    """完整事实太大被截断继续调用或写工具被暴露时失败。参数：网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
    from app.workflow.model import WorkflowModelAdapter

    adapter = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    with pytest.raises(WorkflowBudgetExceeded):
        await adapter.decide([HumanMessage(content="事实" * 10000)], [], BudgetLedger())
    with pytest.raises(InvalidToolDecision):
        await adapter.decide([], [create_ticket], BudgetLedger())
    assert sdk_runtime.requests == []


async def test_rewriter_preserves_usage_and_stops_on_length(sdk_runtime):
    """改写缺计量、截断后仍作为检索输入或错误额外调用时失败。参数：网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError

    payload = '{"standard_query":"如何退款","synonyms":[]}'
    sdk_runtime.responses.append(text_response(payload, usage=True))
    ledger = BudgetLedger()
    await QueryRewriter(sdk_runtime.factory).rewrite("如何退款", budget=ledger)
    assert ledger.snapshot()["actual_tokens"] == 112
    sdk_runtime.responses.append(text_response(payload, "length"))
    ledger = BudgetLedger()
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(sdk_runtime.factory).rewrite("如何退款", budget=ledger)
    assert ledger.snapshot()["records"][0]["finish_reason"] == "length"
    assert len(sdk_runtime.requests) == 2
