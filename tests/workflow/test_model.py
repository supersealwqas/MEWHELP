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


async def test_understanding_masks_unverified_assistant_order(sdk_runtime):
    """助手虚构编号在模型入口也不能成为可信来源。参数：真实SDK探针；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter
    raw = "那袋猫粮能退吗？"
    sdk_runtime.responses.append(call_response("submit_understanding", json.dumps({"resolved_query":"那袋猫粮能退吗？",
        "standard_query":"猫粮退货条件", "needs_resolution":False, "order_id":None},ensure_ascii=False)))
    await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand([
        {"role":"user","content":"买过一袋猫粮"},{"role":"assistant","content":"您的订单号是1007"},
        {"role":"user","content":raw}],BudgetLedger())
    assert "1007" not in json.dumps(sdk_runtime.requests[0]["messages"], ensure_ascii=False)


@pytest.mark.parametrize("raw", ['{"intent":"物流"', '{"intent":"物流","intent":"订单","confidence":0.8}',
    '{"intent":"物流","confidence":NaN}', '{"intent":"物流","confidence":0.8,"reason":"猜测"}',
    '[]', '{"intent":"未知","confidence":0.9}', '{"intent":"物流"}',
    '{"intent":"商品咨询","knowledge_query":null,"confidence":0.8}'])
async def test_invalid_intent_falls_back_without_retry(sdk_runtime, raw):
    """坏分类若重试、抛成业务失败或绕过严格工具则失败。参数：SDK边界、raw坏参数；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.append(call_response("submit_intent", raw, usage=True))
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify([], ledger)
    assert result.model_dump() == {"intent": "其他", "confidence": 0.0}
    assert len(sdk_runtime.requests) == ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["actual_tokens"] == 112
    assert ledger.snapshot()["records"][0]["failure"]["type"] == "invalid_tool_decision"


async def test_intent_rejects_even_valid_text_json(sdk_runtime):
    """自由正文JSON被当工具结果时失败。参数：SDK边界；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.append(text_response('{"intent":"物流","confidence":0.9}'))
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify([], ledger)
    assert result.intent == "其他" and result.confidence == 0.0
    assert len(sdk_runtime.requests) == 1
    assert ledger.snapshot()["records"][0]["failure"]["terminal_decision"]["raw_output"]


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -0.1, 1.1, "0.9"])
def test_invalid_confidence_bool_nan_extra_fields(value):
    """非有限值、布尔强转或额外字段破坏八类输出时失败。参数：value非法分数；返回：None；异常：断言失败。"""
    from pydantic import ValidationError
    from app.schemas.workflow import IntentResult

    with pytest.raises(ValidationError):
        IntentResult(intent="物流", confidence=value)
    with pytest.raises(ValidationError):
        IntentResult(intent="物流", confidence=0.9, knowledge_query=None)
    assert IntentResult(intent="其他", confidence=0).model_dump() == {"intent": "其他", "confidence": 0.0}


async def test_classification_failure_keeps_last_available_call(sdk_runtime):
    """最后一次分类协议失败不能再占额补救。参数：SDK边界；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.append(text_response("无效分类"))
    ledger = BudgetLedger(max_model_calls=1)
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).classify([], ledger)
    assert result.intent == "其他" and ledger.snapshot()["model_calls"] == 1


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


def call_response(name="query_order", raw='{"order_id":"1001"}', *, duplicate=False, usage=False, call_id="call-1"):
    """构造原始工具分片。参数：name工具、raw参数、duplicate重复ID、usage用量尾块、call_id标识；返回：列表；异常：无。"""
    calls = [{"index":0,"id":call_id,"type":"function","function":{"name":name,"arguments":raw}}]
    if duplicate:
        calls.append({"index":1,"id":call_id,"type":"function","function":{"name":name,"arguments":raw}})
    chunks = [sse({"role":"assistant","tool_calls":calls}), sse(finish="tool_calls")]
    if usage:
        chunks.append(sse(tail=True, usage={"prompt_tokens":100,"completion_tokens":12,"total_tokens":112}))
    return chunks


def terminal_response(kind="answer", actions=None, *, usage=False):
    """构造内部终止控制工具响应。参数：kind终止类型、actions建议数组、usage用量尾块；返回：SSE块列表；异常：无。"""
    return call_response("submit_decision", json.dumps({"kind": kind, "suggested_actions": actions or []}), usage=usage)


def intent_response(intent="订单", *, usage=False, confidence=0.96):
    """构造分类工具响应。参数：intent意图、usage用量尾块、confidence有限分数；返回：SSE列表；异常：无。"""
    return call_response("submit_intent", json.dumps({"intent": intent, "confidence": confidence}, ensure_ascii=False), usage=usage)


def understanding_response(query, *, standard=None, needs_resolution=False, order_id=None, usage=False):
    """构造完整理解工具响应。参数：query完整问、standard标准问、消解标志、订单候选、用量；返回：SSE列表；异常：无。"""
    return call_response("submit_understanding", json.dumps({"resolved_query": query,
        "standard_query": query if standard is None else standard, "needs_resolution": needs_resolution,
        "order_id": order_id}, ensure_ascii=False), usage=usage)


@pytest.fixture
async def sdk_runtime():
    """仅替代外部 HTTP，保留 LangChain 与 SDK 实际序列化。参数：无；返回：网络脚本边界；异常：SDK 错误。"""
    requests, responses, options = [], [], []

    async def handle(request):
        """记录实际发包。参数：request HTTP 请求；返回：SSE 响应；异常：超脚本次数时断言失败。"""
        requests.append(json.loads(request.content))
        assert responses, "不允许额外模型请求"
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, httpx.Response):
            return response
        body = "".join("data: " + json.dumps(item) + "\n\n" for item in response) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type":"text/event-stream"}, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        def factory(**kwargs):
            """使用真实客户端和外部假网络。参数：kwargs 适配器配置；返回：ChatOpenAI；异常：配置错误。"""
            options.append(kwargs)
            return ChatOpenAI(base_url="http://offline.invalid/v1", api_key="offline", model=kwargs.pop("model", "test-model"),
                              http_async_client=client, **kwargs)
        yield SimpleNamespace(factory=factory, requests=requests, responses=responses, options=options)


async def test_classify_counts_usage_tail_and_keeps_full_history(sdk_runtime):
    """无 choices 用量尾块被漏记或输入丢历史时失败。参数：sdk_runtime 网络夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    rt, ledger = sdk_runtime, BudgetLedger()
    rt.responses.append(intent_response("物流", usage=True))
    history = [{"role":"user","content":"我要查物流"}, {"role":"assistant","content":"请提供订单号。"},
               {"role":"user","content":"1001"}]
    result = await WorkflowModelAdapter(model_factory=rt.factory, stream_usage=True).classify(history, ledger)
    assert result.intent == "物流" and ledger.snapshot()["actual_tokens"] == 112
    assert rt.requests[0]["messages"][1:] == history
    assert rt.requests[0]["stream"] is True and rt.requests[0]["temperature"] == 0
    assert rt.requests[0]["max_completion_tokens"] == 4096
    assert rt.requests[0]["stream_options"] == {"include_usage":True}
    assert "response_format" not in rt.requests[0]
    assert rt.requests[0]["tool_choice"] == {"type": "function", "function": {"name": "submit_intent"}}
    assert rt.requests[0]["parallel_tool_calls"] is False
    assert rt.requests[0]["tools"][0]["function"]["strict"] is True
    from app.core.memory import _estimate_message_tokens, _estimate_token_units
    from langchain_core.messages import convert_to_messages
    assert ledger.snapshot()["records"][0]["input_units"] == (
        _estimate_message_tokens(convert_to_messages(rt.requests[0]["messages"])) + _estimate_token_units(rt.requests[0]["tools"]))


async def test_understanding_and_classification_use_distinct_strict_tools(sdk_runtime):
    """理解与分类混用工具、预算或模型配置时失败。参数：SDK边界；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    raw = "  订单1001现在是什么状态？  "
    sdk_runtime.responses.extend([understanding_response(raw, order_id="1001", usage=True),
                                  intent_response("订单", confidence=0.92, usage=True)])
    ledger, adapter = BudgetLedger(), WorkflowModelAdapter(model_factory=sdk_runtime.factory, intent_model="intent-test")
    understood = await adapter.understand([{"role": "user", "content": raw}], ledger)
    result = await adapter.classify([{"role": "user", "content": understood.resolved_query}], ledger)
    assert result.model_dump() == {"intent": "订单", "confidence": 0.92}
    assert [r["stage"] for r in ledger.snapshot()["records"]] == ["understand", "classify"]
    assert ledger.snapshot()["actual_tokens"] == 224
    for request, tool in zip(sdk_runtime.requests, ["submit_understanding", "submit_intent"]):
        assert request["tool_choice"] == {"type": "function", "function": {"name": tool}}
        assert [item["function"]["name"] for item in request["tools"]] == [tool]
        assert request["parallel_tool_calls"] is False and request["tools"][0]["function"]["strict"] is True
    assert sdk_runtime.requests[0]["model"] == "test-model"
    assert sdk_runtime.requests[1]["model"] == "intent-test"
    assert [r["max_completion_tokens"] for r in sdk_runtime.requests] == [2048, 4096]


async def test_understanding_order_id_required_but_nullable(sdk_runtime):
    """可空订单漏字段被默认补齐或重试时失败。参数：SDK边界；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter, understanding_tool_schema

    raw = "  运费怎么算？  "
    sdk_runtime.responses.extend([understanding_response(raw), call_response("submit_understanding",
        json.dumps({"resolved_query": "变动原问", "standard_query": "变动检索问", "needs_resolution": False}))])
    adapter = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    first = await adapter.understand([{"role": "user", "content": raw}], BudgetLedger())
    ledger = BudgetLedger()
    second = await adapter.understand([{"role": "user", "content": raw}], ledger)
    assert first.order_id is None
    assert second.resolved_query == second.standard_query == raw and second.order_id is None
    assert ledger.snapshot()["model_calls"] == 1 and ledger.snapshot()["reserved_units"] > 0
    assert ledger.snapshot()["records"][0]["failure"]["missing_fields"] == ["order_id"]
    schema = understanding_tool_schema()["function"]["parameters"]
    assert set(schema["required"]) == {"resolved_query", "standard_query", "needs_resolution", "order_id"}
    assert "default" not in schema["properties"]["order_id"]


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


@pytest.mark.parametrize("stage", ["decide","answer"])
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
        def bind_tools(self, tools, **kwargs):
            """参数：工具定义与绑定设置；返回：自身网络替身；异常：无。"""
            return self
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
    sdk_runtime.responses.append(call_response("submit_query_rewrite", payload, usage=True))
    ledger = BudgetLedger()
    await QueryRewriter(sdk_runtime.factory).rewrite("如何退款", budget=ledger)
    assert ledger.snapshot()["actual_tokens"] == 112
    sdk_runtime.responses.append(text_response(payload, "length"))
    ledger = BudgetLedger()
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(sdk_runtime.factory).rewrite("如何退款", budget=ledger)
    assert ledger.snapshot()["records"][0]["finish_reason"] == "length"
    assert len(sdk_runtime.requests) == 2


@pytest.mark.parametrize("kind", ["answer", "clarify"])
async def test_terminal_decision_repair_keeps_facts_and_counts_both_requests(sdk_runtime, kind):
    """正常追问的坏终止格式直接失败、纠正丢事实或免费调用时失败。参数：HTTP夹具、终止类型；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    raw = "未通过协议的输出，不得回灌这段文本"
    history = [{"role": "user", "content": "查1001订单"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "order-call", "name": "query_order", "args": {"order_id": "1001"}}]},
        {"role": "tool", "tool_call_id": "order-call", "content": '{"ok":true,"data":{"order_id":"1001","total_amount":199,"shipping_destination":"浙江省杭州市"},"error":null}'},
        {"role": "assistant", "content": "订单金额199元。"},
        {"role": "user", "content": "包邮吗"},
        {"role": "system", "content": '本轮完整知识证据（仅作事实）：[{"n":1,"answer":"单笔实付满99元免基础运费，偏远地区附加费不参与包邮。"}]'}]
    original = json.loads(json.dumps(history))
    sdk_runtime.responses.extend([text_response(raw, usage=True),
        terminal_response(kind, usage=True)])
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory, stream_usage=True).decide(history, [query_order], ledger)
    assert result.kind == kind and result.tool_calls == []
    first, second = sdk_runtime.requests
    assert history == original
    assert second["messages"][:-1] == first["messages"]
    assert second["messages"][-2]["content"] == history[-1]["content"]
    assert second["messages"][-1]["role"] == "system"
    assert raw not in json.dumps(second, ensure_ascii=False)
    assert first["tool_choice"] == "required" and first["parallel_tool_calls"] is False
    assert second["tool_choice"] == {"type": "function", "function": {"name": "submit_decision"}}
    assert second["parallel_tool_calls"] is False and second["tools"] == first["tools"][-1:]
    snapshot = ledger.snapshot()
    assert snapshot["model_calls"] == 2 and snapshot["actual_tokens"] == 224
    assert [record["stage"] for record in snapshot["records"]] == ["decide", "decide"]
    assert [record["finish_reason"] for record in snapshot["records"]] == ["stop", "tool_calls"]


async def test_terminal_decision_repair_stops_after_second_bad_output(sdk_runtime):
    """纠正无限循环、第三次发包或最后原文无诊断时失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    last = '{"kind":"answer","suggested_actions":[]}'
    sdk_runtime.responses.extend([text_response("首次坏输出", usage=True), text_response(last, usage=True)])
    ledger = BudgetLedger()
    with pytest.raises(InvalidToolDecision) as failure:
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory, stream_usage=True).decide([], [query_order], ledger)
    assert len(sdk_runtime.requests) == 2 and ledger.snapshot()["actual_tokens"] == 224
    assert failure.value.audit_record["terminal_decision"] == {"raw_output": last,
        "raw_length": len(last), "truncated": False, "validation_category": "missing_control_tool"}


async def test_terminal_decision_failure_audit_is_bounded_without_changing_tool_audit(sdk_runtime):
    """无控制工具的失败原文无限保存或改变业务工具错误审计时失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    raw = "模型说明" * 600
    sdk_runtime.responses.extend([text_response(raw), text_response(raw)])
    with pytest.raises(InvalidToolDecision) as failure:
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [], BudgetLedger())
    diagnostic = failure.value.audit_record["terminal_decision"]
    assert diagnostic == {"raw_output": raw[:1000], "raw_length": len(raw),
        "truncated": True, "validation_category": "missing_control_tool"}
    assert raw not in str(failure.value)
    calls = [{"name": "query_order", "args": {"order_id": 1001}}]
    assert InvalidToolDecision(tool_calls=calls).audit_record == {"type": "invalid_tool_decision",
        "tool_calls": calls, "invalid_tool_calls": [], "tool_call_chunks": []}


@pytest.mark.parametrize("limits", [{"max_model_calls": 3}, {"max_decisions": 1}])
async def test_terminal_decision_repair_respects_remaining_budget_before_request(sdk_runtime, limits):
    """纠正挤占两次正文余量或突破决策额度时失败。参数：HTTP夹具、预算限制；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.append(text_response("普通正文", usage=True))
    ledger = BudgetLedger(**limits)
    with pytest.raises(WorkflowBudgetExceeded):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], ledger)
    assert len(sdk_runtime.requests) == 1
    assert ledger.snapshot()["model_calls"] == 1 and ledger.snapshot()["actual_tokens"] == 112


async def test_terminal_decision_repair_never_accepts_a_new_tool_request(sdk_runtime):
    """网关违背纠正轮禁用工具时仍执行已有或新查询则失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.extend([text_response("非JSON正文"), call_response()])
    ledger = BudgetLedger()
    with pytest.raises(InvalidToolDecision) as failure:
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], ledger)
    assert len(sdk_runtime.requests) == 2
    assert failure.value.audit_record["tool_calls"][0]["name"] == "query_order"


@pytest.mark.parametrize("attempt", [0, 1])
async def test_decision_cancellation_propagates_without_another_request(sdk_runtime, attempt):
    """首轮或纠正轮取消被当成格式错误继续发包时失败。参数：HTTP夹具、取消所在请求索引；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    if attempt:
        sdk_runtime.responses.append(text_response("非JSON正文"))
    sdk_runtime.responses.append(asyncio.CancelledError())
    ledger = BudgetLedger()
    with pytest.raises(asyncio.CancelledError):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], ledger)
    assert len(sdk_runtime.requests) == attempt + 1
    assert ledger.snapshot()["model_calls"] == attempt + 1
    assert ledger.snapshot()["records"][-1]["finish_reason"] is None


@pytest.mark.parametrize("attempt", [0, 1])
async def test_decision_broken_stream_propagates_and_closes_without_repair(sdk_runtime, attempt):
    """流网络异常被格式纠正吞掉或未关闭响应时失败。参数：HTTP夹具、故障所在请求索引；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter
    from openai import APIConnectionError

    closed = []

    class BrokenStream(httpx.AsyncByteStream):
        """产生首块后断开的HTTP字节流，保留真实SDK异常边界。"""

        async def __aiter__(self):
            """产生一个SSE块后失败。参数：无；返回：异步字节迭代；异常：httpx.ReadError模拟断流。"""
            yield ("data: " + json.dumps(sse({"role": "assistant", "content": "坏"})) + "\n\n").encode()
            raise httpx.ReadError("上游流中断")

        async def aclose(self):
            """记录真实HTTP响应关闭。参数：无；返回：None；异常：无。"""
            closed.append(True)

    if attempt:
        sdk_runtime.responses.append(text_response("非JSON正文"))
    sdk_runtime.responses.append(httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=BrokenStream()))
    ledger = BudgetLedger()
    with pytest.raises(APIConnectionError):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], ledger)
    assert len(sdk_runtime.requests) == attempt + 1 and closed == [True]
    assert ledger.snapshot()["model_calls"] == attempt + 1


@pytest.mark.parametrize("kind,actions,with_business_tools", [
    ("answer", [], True), ("clarify", [], True), ("answer", ["handoff"], True),
    ("answer", ["create_ticket"], True), ("answer", ["handoff", "create_ticket"], True),
    ("answer", [], False)])
async def test_control_tool_ends_in_one_request_without_business_calls(sdk_runtime, kind, actions, with_business_tools):
    """合法控制工具被当未知工具、需要二次请求或泄露为业务调用时失败。参数：HTTP夹具、类型、建议、业务工具开关；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.append(terminal_response(kind, actions, usage=True))
    ledger = BudgetLedger()
    tools = [query_order, query_logistics, query_product] if with_business_tools else []
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], tools, ledger)
    assert result.kind == kind and result.suggested_actions == actions and result.tool_calls == []
    assert len(sdk_runtime.requests) == 1 and ledger.snapshot()["actual_tokens"] == 112


def wire_input_units(request):
    """按预算公开字符口径核对实际发包，适用于仅含system的测试请求。参数：request实际HTTP体；返回：输入单位整数；异常：非system测试输入断言失败。"""
    units = 0
    values = [{"role": row["role"], "content": row["content"]} for row in request["messages"]]
    assert all(row["role"] == "system" for row in request["messages"])
    for value in [*values, request["tools"]]:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        units += (sum(1 if char.isascii() else 8 for char in serialized) + 3) // 4
    return units


async def test_decision_sends_strict_required_schemas_and_budgets_actual_bound_tools(sdk_runtime):
    """发包仍可自由文本、公开身份字段或预算漏掉控制schema时失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.extend([text_response("必须恢复"), terminal_response()])
    ledger = BudgetLedger()
    tools = [query_order, query_logistics, query_product]
    await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], tools, ledger)
    from app.workflow.model import decision_tool_schemas

    first, second = sdk_runtime.requests
    assert first["tools"] == decision_tool_schemas(tools)
    assert [schema["function"]["name"] for schema in first["tools"]] == [
        "query_order", "query_logistics", "query_product", "submit_decision"]
    assert first["tool_choice"] == "required" and first["parallel_tool_calls"] is False
    for schema in first["tools"]:
        function = schema["function"]
        assert function["strict"] is True
        parameters = function["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])
        assert "user_id" not in parameters["properties"]
    terminal = first["tools"][-1]["function"]["parameters"]
    assert set(terminal["properties"]) == {"kind", "suggested_actions"}
    assert terminal["properties"]["kind"]["enum"] == ["answer", "clarify"]
    assert terminal["properties"]["suggested_actions"]["maxItems"] == 2
    assert terminal["properties"]["suggested_actions"]["items"]["enum"] == ["handoff", "create_ticket"]
    assert second["tools"] == first["tools"][-1:]
    assert [row["input_units"] for row in ledger.snapshot()["records"]] == [wire_input_units(first), wire_input_units(second)]


@pytest.mark.parametrize("entries", [
    [("submit_decision", "d1", '{"kind":"answer","suggested_actions":[]}'), ("query_order", "q1", '{"order_id":"1001"}')],
    [("submit_decision", "d1", '{"kind":"answer","suggested_actions":[]}'), ("submit_decision", "d2", '{"kind":"clarify","suggested_actions":[]}')],
    [("query_order", "q1", '{"order_id":"1001"}'), ("query_order", "q1", '{"order_id":"1002"}')],
    [("query_order", "q1", '{"order_id":"1001"}'), ("query_order", "q2", '{"order_id":1002}')],
    [("submit_decision", "", '{"kind":"answer","suggested_actions":[]}')],
    [("submit_decision", " ", '{"kind":"answer","suggested_actions":[]}')],
    [("submit_decision", "d1", '{"kind":"answer"'), ("query_order", "q1", '{"order_id":"1001"}')],
    [("submit_decision", "d1", '{"kind":"answer"'), ("submit_decision", "d2", '{"kind":"answer","suggested_actions":[]}')],
    [("submit_decision", " ", '{"kind":"answer"')],
], ids=["mixed", "two-terminal", "duplicate-id", "bad-business-in-batch", "empty-id", "blank-id",
        "broken-control-and-business", "broken-control-and-control", "broken-control-blank-id"])
async def test_invalid_decision_batches_are_rejected_without_partial_result_or_retry(sdk_runtime, entries):
    """混合批次、多个终止、非法ID或坏业务参数仍返回部分申请时失败。参数：HTTP夹具、完整批次；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    calls = [{"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}
        for index, (name, call_id, raw) in enumerate(entries)]
    sdk_runtime.responses.append([sse({"role": "assistant", "tool_calls": calls}), sse(finish="tool_calls")])
    with pytest.raises(InvalidToolDecision):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], BudgetLedger())
    assert len(sdk_runtime.requests) == 1


@pytest.mark.parametrize("raw", ['{"kind":"answer"}',
    '{"kind":"answer","suggested_actions":["handoff","handoff"]}',
    '{"kind":"answer","suggested_actions":["unknown"]}',
    '{"kind":"tools","suggested_actions":[]}',
    '{"kind":"answer","suggested_actions":[],"answer":"隐藏正文"}'],
    ids=["missing-field", "duplicate-action", "unknown-action", "bad-kind", "extra-field"])
async def test_single_control_bad_parameters_get_one_repair_but_are_never_accepted(sdk_runtime, raw):
    """唯一合法ID控制工具的坏参数被放行或不能有限纠正时失败。参数：HTTP夹具、参数JSON；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.extend([call_response("submit_decision", raw), call_response("submit_decision", raw)])
    with pytest.raises(InvalidToolDecision) as failure:
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [], BudgetLedger())
    assert len(sdk_runtime.requests) == 2
    assert failure.value.audit_record["terminal_decision"]["validation_category"] in {"shape", "schema"}


async def test_single_control_parameter_repair_preserves_clarification(sdk_runtime):
    """控制参数纠正把需要追問强制改为answer时失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.extend([call_response("submit_decision", '{"kind":"clarify"}'), terminal_response("clarify")])
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [], BudgetLedger())
    assert result.kind == "clarify" and result.tool_calls == [] and len(sdk_runtime.requests) == 2


@pytest.mark.parametrize("raw", ['{"kind":"answer","suggested_actions":[],"broken_probe"',
    '{"kind":"answer","kind":"clarify","suggested_actions":[]}'], ids=["broken-json", "duplicate-key"])
async def test_single_control_raw_json_failure_can_recover_without_replaying_raw(sdk_runtime, raw):
    """唯一合法控制分片的坏JSON直接失败、修补参数或回灌原文时失败。参数：HTTP夹具、原始坏参数；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.extend([call_response("submit_decision", raw, usage=True), terminal_response("clarify", usage=True)])
    ledger = BudgetLedger()
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], ledger)
    assert result.kind == "clarify" and result.tool_calls == []
    assert len(sdk_runtime.requests) == 2 and ledger.snapshot()["actual_tokens"] == 224
    second = sdk_runtime.requests[1]
    assert [schema["function"]["name"] for schema in second["tools"]] == ["submit_decision"]
    assert raw not in "\n".join(message["content"] for message in second["messages"])


@pytest.mark.parametrize("raw", ['{"kind":"answer","suggested_actions":[],"broken_probe"',
    '{"kind":"answer","kind":"clarify","suggested_actions":[]}'], ids=["broken-json", "duplicate-key"])
async def test_single_control_raw_json_failure_stops_after_one_repair_with_original_audit(sdk_runtime, raw):
    """控制分片坏JSON无限重试或最终改写原始异常审计时失败。参数：HTTP夹具、原始坏参数；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter

    sdk_runtime.responses.extend([call_response("submit_decision", raw, usage=True), call_response("submit_decision", raw, usage=True)])
    ledger = BudgetLedger()
    with pytest.raises(InvalidToolDecision) as failure:
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [query_order], ledger)
    assert len(sdk_runtime.requests) == 2 and ledger.snapshot()["actual_tokens"] == 224
    audit = failure.value.audit_record
    assert audit["tool_calls"] == [] and audit["invalid_tool_calls"] == []
    assert len(audit["tool_call_chunks"]) == 1 and audit["tool_call_chunks"][0]["args"] == raw
    assert "terminal_decision" not in audit


async def test_broken_control_json_with_abnormal_finish_does_not_retry(sdk_runtime):
    """控制参数同时截断时把异常结束误当可恢复格式失败则失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter, WorkflowOutputError

    response = call_response("submit_decision", '{"kind":"answer"')
    response[-1] = sse(finish="length")
    sdk_runtime.responses.append(response)
    ledger = BudgetLedger()
    with pytest.raises(WorkflowOutputError):
        await WorkflowModelAdapter(model_factory=sdk_runtime.factory).decide([], [], ledger)
    assert len(sdk_runtime.requests) == 1 and ledger.snapshot()["records"][0]["finish_reason"] == "length"
