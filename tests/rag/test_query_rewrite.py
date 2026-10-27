"""单轮归一的真实解析、检索编排及取消边界测试，仅替换模型流。"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tests.rag.test_retrieval import _chunk, _search


class RewriteModel:
    """可观测的外部模型流，支持坏协议、网络故障和等待取消。"""

    def __init__(self, payload=None, *, error=None, wait=False):
        """参数：payload 文本或分片，error 网络异常，wait 等待取消；返回：None；异常：无。"""
        self.payload = payload if payload is not None else '{"standard_query":"饮水机怎么清洗","synonyms":["清洁","保养"]}'
        self.error, self.wait = error, wait
        self.inputs, self.options = [], []
        self.closed = False
        self.started = asyncio.Event()

    def factory(self, **options):
        """记录客户端参数。参数：模型设置；返回：流模型；异常：无。"""
        self.options.append(options)
        return self

    def bind_tools(self, tools, **options):
        """接受生产工具绑定后提供受控响应。参数：tools公开schema、options强制选择；返回：自身；异常：无。"""
        return self

    async def astream(self, messages):
        """模拟网络分片。参数：完整模型输入；返回：分片流；异常：配置的网络错误或取消。"""
        self.inputs.append(messages)
        self.started.set()
        try:
            if self.wait:
                await asyncio.Event().wait()
            if self.error:
                raise self.error
            if isinstance(self.payload, list):
                for chunk in self.payload:
                    yield chunk
            else:
                yield AIMessageChunk(content="", tool_call_chunks=[{"name": "submit_query_rewrite",
                    "args": self.payload, "id": "rewrite-1", "index": 0}],
                    response_metadata={"finish_reason": "tool_calls"})
        finally:
            self.closed = True


async def test_rewriter_receives_no_history_or_ground_truth():
    """改写仅发送原问和当前关键词。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter

    model = RewriteModel()
    raw = "  饮水机脏了咋洗？\n"
    result = await QueryRewriter(model.factory).rewrite(raw, keyword="饮水机清洗")
    assert result.standard_query == "饮水机怎么清洗" and result.synonyms == ["清洁", "保养"]
    assert len(model.inputs) == len(model.options) == 1 and model.closed
    assert json.loads(model.inputs[0][-1].content) == {"query": raw, "keyword": "饮水机清洗"}
    assert [message.type for message in model.inputs[0]] == ["system", "human"]


@pytest.mark.parametrize("payload", [
    '{"standard_query":"清洗","synonyms":[],"history":[]}',
    '{"standard_query":"清洗","standard_query":"改写","synonyms":[]}',
    '{"standard_query":123,"synonyms":[]}',
    '{"standard_query":"清洗","synonyms":[1]}',
    '{"standard_query":"清洗","synonyms":["a","b","c","d","e"]}',
    '{"standard_query":" ","synonyms":[]}',
    '{"standard_query":"清洗","synonyms":[" "]}',
    '{"standard_query":"MH-W99清洗","synonyms":[]}',
    '{"standard_query":"清洗","synonyms":["MH-W99"]}',
    '```json\n{"standard_query":"清洗","synonyms":[]}\n```',
    '{"standard_query":"清洗","synonyms":[NaN]}',
])
async def test_invalid_rewrite_fails_without_fallback(payload):
    """宽松解析或新增型号会让错误改写漏过。参数：非法输出；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError

    model = RewriteModel(payload)
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(model.factory).rewrite("饮水机怎么洗")
    assert model.closed and len(model.inputs) == 1


@pytest.mark.parametrize("standard", ["MH-W20清洗", "MH-W99用了2天怎么洗", "怎么清洗"])
async def test_rewrite_cannot_change_or_drop_model_and_quantity(standard):
    """型号、数量不能被邻近型号或标准问法吞掉。参数：错误改写；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError

    model = RewriteModel(json.dumps({"standard_query": standard, "synonyms": []}))
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(model.factory).rewrite("MH-W99用了3天怎么洗")


async def test_rewrite_network_error_and_cancel_close_stream():
    """网络故障不静默回退，取消不转换为知识不足。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError
    from app.core.model_budget import BudgetLedger

    broken = RewriteModel(error=OSError("network"))
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(broken.factory).rewrite("如何清洗")
    assert broken.closed
    waiting = RewriteModel(wait=True)
    budget = BudgetLedger()
    task = asyncio.create_task(QueryRewriter(waiting.factory).rewrite("如何清洗", budget=budget))
    await waiting.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert waiting.closed
    assert budget.snapshot()["model_calls"] == 1 and budget.snapshot()["records"][0]["usage"] is None


async def test_budget_and_tool_call_are_explicit_errors():
    """超预算或工具申请不能转成截断/成功输出。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError

    model = RewriteModel()
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(model.factory, request_budget=1).rewrite("如何清洗")
    assert not model.inputs
    model = RewriteModel([AIMessageChunk(content="", tool_call_chunks=[{"name": "query_faq", "args": "{}", "id": "x", "index": 0}])])
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(model.factory).rewrite("如何清洗")
    assert model.closed


async def test_synonyms_affect_bm25_not_knowledge_storage():
    """实际编码采用标准问，BM25追加同义词，重排保留原问且不修改知识。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter

    search, encoder, index, reranker = _search([_chunk(1)])
    model = RewriteModel()
    search.query_rewriter = QueryRewriter(model.factory)
    before = await search.store.get_chunks([1])
    rows = await search.query_faq("饮水机清洗", source_query="饮水机脏了咋洗")
    assert rows and len(model.inputs) == 1
    assert encoder.inputs == ["饮水机怎么清洗"]
    assert all(word in index.calls[0][0] for word in ["饮水机脏了咋洗", "饮水机怎么清洗", "清洁", "保养"])
    assert "饮水机脏了咋洗" in reranker.calls[0][0]
    assert "保养" not in reranker.calls[0][0]
    assert await search.store.get_chunks([1]) == before


async def test_refund_expansions_share_one_rewrite():
    """退款三角度不能重复付费改写。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter

    search, _, index, _ = _search([_chunk(1)])
    model = RewriteModel('{"standard_query":"退款怎么申请","synonyms":["退货"]}')
    search.query_rewriter = QueryRewriter(model.factory)
    await search.retrieve_policy("想退钱怎么弄")
    assert len(model.inputs) == 1 and len(index.calls) == 3


def test_preparation_preserves_raw_and_keeps_synonyms_out_of_dense():
    """共享准备结果不得覆盖原问或把同义词写入Dense。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewrite
    from app.knowledge.retrieval import prepare_query

    raw = " 新疆呢，不退款，MH-W99用了3天如何清洗？ "
    prepared = prepare_query(raw, rewrite=QueryRewrite(standard_query="新疆不退款，MH-W99用了3天如何清洗", synonyms=["清洁"]))
    assert prepared.raw == raw
    assert all("清洁" not in query for group in prepared.groups for query in group)
    assert all(raw in text and "清洁" in text for text in prepared.bm25_queries.values())
    assert all(raw in text and "清洁" not in text for text in prepared.rerank_queries.values())


@pytest.fixture
async def rewrite_sdk():
    """只替换HTTP响应，保留真实SDK工具序列化。参数：无；返回：工厂/请求/响应脚本；异常：SDK错误。"""
    requests, responses = [], []

    async def handle(request):
        """记录实际发包并提供SSE。参数：request请求；返回：httpx.Response；异常：多余请求断言失败。"""
        requests.append(json.loads(request.content))
        assert responses, "不得额外调用或回退"
        body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in responses.pop(0)) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        def factory(**options):
            """创建使用离线HTTP的真实客户端。参数：options生产模型选项；返回：ChatOpenAI；异常：SDK配置错误。"""
            return ChatOpenAI(base_url="http://offline.invalid/v1", api_key="offline", model="test-model",
                              http_async_client=client, **options)
        yield SimpleNamespace(factory=factory, requests=requests, responses=responses)


def rewrite_response(raw, *, tool=True, name="submit_query_rewrite", finish=None, usage=True):
    """构造完整改写SSE响应。参数：raw参数原文、tool工具或正文、name工具名、finish结束原因、usage用量；返回：块列表；异常：无。"""
    delta = {"role": "assistant"}
    if tool:
        delta["tool_calls"] = [{"index": 0, "id": "rewrite-1", "type": "function",
            "function": {"name": name, "arguments": raw}}]
    else:
        delta["content"] = raw
    rows = [{"id": "rewrite-test", "object": "chat.completion.chunk", "created": 0,
        "model": "test-model", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {"id": "rewrite-test", "object": "chat.completion.chunk", "created": 0,
        "model": "test-model", "choices": [{"index": 0, "delta": {},
            "finish_reason": finish or ("tool_calls" if tool else "stop")}]}]
    if usage:
        rows.append({"id": "rewrite-test", "object": "chat.completion.chunk", "created": 0,
            "model": "test-model", "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    return rows


async def test_rewrite_uses_forced_strict_tool_and_budgets_schema(rewrite_sdk):
    """未强制结构化工具、原问被改写或schema漏计预算时失败。参数：真实SDK夹具；返回：None；异常：断言失败。"""
    from app.core.memory import _estimate_message_tokens, _estimate_token_units
    from app.core.model_budget import BudgetLedger
    from app.knowledge.query_rewrite import QueryRewriter

    raw = "  MH-W99用了3天，脏了咋洗？\n"
    rewrite_sdk.responses.append(rewrite_response('{"standard_query":"MH-W99用了3天如何清洗","synonyms":["清洁"]}'))
    ledger = BudgetLedger()
    result = await QueryRewriter(rewrite_sdk.factory).rewrite(raw, keyword="保养", budget=ledger)
    assert result.standard_query == "MH-W99用了3天如何清洗" and result.synonyms == ["清洁"]
    assert len(rewrite_sdk.requests) == 1
    request = rewrite_sdk.requests[0]
    assert json.loads(request["messages"][1]["content"]) == {"query": raw, "keyword": "保养"}
    assert request["tool_choice"] == {"type": "function", "function": {"name": "submit_query_rewrite"}}
    assert request["parallel_tool_calls"] is False
    assert len(request["tools"]) == 1
    schema = request["tools"][0]["function"]
    assert schema["name"] == "submit_query_rewrite" and schema["strict"] is True
    assert schema["parameters"]["additionalProperties"] is False
    assert set(schema["parameters"]["required"]) == {"standard_query", "synonyms"}
    actual_messages = [SystemMessage(content=request["messages"][0]["content"]), HumanMessage(content=request["messages"][1]["content"])]
    record = ledger.snapshot()["records"][0]
    assert record["input_units"] == _estimate_message_tokens(actual_messages) + _estimate_token_units(request["tools"])
    assert record["output_limit"] == request["max_completion_tokens"] == 2048
    assert record["finish_reason"] == "tool_calls" and ledger.snapshot()["actual_tokens"] == 120


@pytest.mark.parametrize("tool,name,finish", [(False, "submit_query_rewrite", "stop"),
    (True, "query_order", "tool_calls"), (True, "submit_query_rewrite", "length")],
    ids=["text-json", "wrong-tool", "truncated-tool"])
async def test_rewrite_has_no_text_or_wrong_tool_fallback(rewrite_sdk, tool, name, finish):
    """自由文本JSON、错误工具或截断被接受或触发额外调用时失败。参数：SDK夹具、响应形式/名称/结束原因；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError

    rewrite_sdk.responses.append(rewrite_response('{"standard_query":"如何清洗","synonyms":[]}', tool=tool, name=name, finish=finish))
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(rewrite_sdk.factory).rewrite("如何清洗")
    assert len(rewrite_sdk.requests) == 1


async def test_rewrite_request_budget_cannot_omit_tool_definition():
    """只容得下消息却遗漏工具schema仍发请求时失败。参数：无；返回：None；异常：断言失败。"""
    from app.core.memory import _estimate_message_tokens
    from app.knowledge.query_rewrite import QUERY_REWRITE_RULES, QueryRewriter, QueryRewriteError

    query = "饮水机怎么清洗"
    messages = [SystemMessage(content=QUERY_REWRITE_RULES),
        HumanMessage(content=json.dumps({"query": query, "keyword": None}, ensure_ascii=False))]
    model = RewriteModel()
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(model.factory, request_budget=_estimate_message_tokens(messages)).rewrite(query)
    assert model.inputs == [] and model.options == []


async def test_rewrite_tool_argument_length_is_bounded(rewrite_sdk):
    """工具参数超10000字符未被输出长度保护拦截时失败。参数：SDK夹具；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryRewriter, QueryRewriteError

    # 参数本身能通过JSON与语义校验，尾部空白只应由完整输出长度保护拒绝。
    rewrite_sdk.responses.append(rewrite_response('{"standard_query":"如何清洗","synonyms":[]}' + " " * 10000))
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(rewrite_sdk.factory).rewrite("如何清洗")
    assert len(rewrite_sdk.requests) == 1


async def test_query_expander_simple_and_core_queries(rewrite_sdk):
    """扩写器在简单FAQ返回单条标准问法，核心多条件返回三条不重复查询并计入expand账本。

    参数：rewrite_sdk 离线HTTP与真实ChatOpenAI夹具。
    返回：None。
    异常：AssertionError，工具定义、账本记录或返回条数不符合契约。
    """
    from app.core.memory import _estimate_message_tokens, _estimate_token_units
    from app.core.model_budget import BudgetLedger
    from app.knowledge.query_rewrite import QueryExpander

    rewrite_sdk.responses.append(
        rewrite_response('{"queries":["退货运费由谁承担"]}', name="submit_expanded_queries")
    )
    simple_ledger = BudgetLedger()
    expander = QueryExpander(rewrite_sdk.factory)
    simple = await expander.expand("订单1001退货运费谁出 已签收", simple_query="退货运费由谁承担", budget=simple_ledger)
    assert simple.queries == ["退货运费由谁承担"]
    assert len(rewrite_sdk.requests) == 1
    first_req = rewrite_sdk.requests[0]
    assert json.loads(first_req["messages"][1]["content"]) == {
        "query": "订单1001退货运费谁出 已签收",
        "simple_query": "退货运费由谁承担",
    }
    assert first_req["tool_choice"] == {"type": "function", "function": {"name": "submit_expanded_queries"}}
    assert first_req["parallel_tool_calls"] is False
    schema = first_req["tools"][0]["function"]
    assert schema["name"] == "submit_expanded_queries" and schema["strict"] is True
    assert set(schema["parameters"]["required"]) == {"queries"}
    actual_messages = [
        SystemMessage(content=first_req["messages"][0]["content"]),
        HumanMessage(content=first_req["messages"][1]["content"]),
    ]
    record = simple_ledger.snapshot()["records"][0]
    assert record["stage"] == "expand"
    assert record["input_units"] == _estimate_message_tokens(actual_messages) + _estimate_token_units(first_req["tools"])
    assert record["output_limit"] == first_req["max_completion_tokens"] == 2048
    assert simple_ledger.snapshot()["actual_tokens"] == 120

    core_queries = ["已签收商品退货申请条件与时限", "已签收拆封商品退货限制与除外情形", "退货办理流程与运费承担规则"]
    rewrite_sdk.responses.append(
        rewrite_response(json.dumps({"queries": core_queries}, ensure_ascii=False), name="submit_expanded_queries")
    )
    core_ledger = BudgetLedger()
    core = await expander.expand("订单1001的猫砂盆拆封了还能退吗 已签收", simple_query="拆封猫砂盆能否退货", budget=core_ledger)
    assert core.queries == core_queries
    assert core_ledger.snapshot()["model_calls"] == 1
    assert "failure" not in simple_ledger.snapshot()["records"][0]
    assert "failure" not in core_ledger.snapshot()["records"][0]


@pytest.mark.parametrize("single_query", ["取消订单的办理流程", " 退货运费由谁承担 "])
async def test_query_expander_different_single_query_falls_back_with_usage(rewrite_sdk, single_query: str) -> None:
    """单条扩写偏离标准问法时仅回退完整种子，保留实际用量和内容失败诊断。

    参数：rewrite_sdk 离线 HTTP 与真实 ChatOpenAI 夹具；single_query 与标准问法不严格相等的模型输出。
    返回：None。
    异常：AssertionError，错问法被接受、触发再次调用或账本和失败类型丢失。
    """
    from app.core.model_budget import BudgetLedger
    from app.knowledge.query_rewrite import QueryExpander

    rewrite_sdk.responses.append(
        rewrite_response(json.dumps({"queries": [single_query]}, ensure_ascii=False), name="submit_expanded_queries")
    )
    ledger = BudgetLedger()
    ledger.reserve("understand", 10, 100)
    ledger.record_usage("understand", {"total_tokens": 20}, "tool_calls")
    previous_record = ledger.snapshot()["records"][0]
    query = "订单1001退货运费谁出 已签收"
    result = await QueryExpander(rewrite_sdk.factory).expand(
        query, simple_query="退货运费由谁承担", budget=ledger
    )

    assert result.queries == [query]
    assert len(rewrite_sdk.requests) == 1
    snapshot = ledger.snapshot()
    assert snapshot["model_calls"] == 2 and snapshot["records"][0] == previous_record
    assert snapshot["actual_tokens"] == 140
    record = snapshot["records"][1]
    assert record["stage"] == "expand" and record["reserved_units"] > 0
    assert record["usage"]["total_tokens"] == 120 and record["finish_reason"] == "tool_calls"
    assert record["failure"]["terminal_decision"]["validation_category"] == "expansion_validation"
    assert record["failure"]["error_type"] == "QueryRewriteError"
    assert record["failure"]["query_fallback"] == "seed_query"


@pytest.mark.parametrize(
    "payload,error",
    [
        (None, OSError("network down")),
        ('{"queries":[]}', None),
        ('{"queries":["单条1","单条2"]}', None),
        ('{"queries":["重复","重复","重复"]}', None),
        ('{"queries":["  "]}', None),
        ('{"queries":["合法","extra":1]}', None),
    ],
)
async def test_query_expander_network_and_parse_failures_fallback_and_preserve_budget(payload, error):
    """扩写网络或结构解析失败时回退为[query]并保留账本占额与失败诊断。

    参数：payload 模型工具参数字符串；error 模拟的网络异常。
    返回：None。
    异常：AssertionError，未回退为单条种子或账本丢失。
    """
    from app.core.model_budget import BudgetLedger
    from app.knowledge.query_rewrite import QueryExpander

    chunks = None
    if payload is not None:
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": "submit_expanded_queries", "args": payload, "id": "exp-1", "index": 0}],
                response_metadata={"finish_reason": "tool_calls"},
            )
        ]
    model = RewriteModel(chunks, error=error)
    ledger = BudgetLedger()
    query = "订单1001的猫砂盆能退吗 已签收"
    failed_expansion = await QueryExpander(model.factory).expand(query, simple_query="猫砂盆能否退货", budget=ledger)
    assert failed_expansion.queries == [query]
    assert len(model.inputs) == 1 and model.closed
    snap = ledger.snapshot()
    assert snap["model_calls"] == 1 and snap["records"][0]["stage"] == "expand"
    assert snap["records"][0].get("failure") is not None


async def test_expansion_validation_failure_uses_seed_once():
    """扩写内容校验失败仅回退一次[query]且保留账本，直接向检索层注入非法prepared_queries必须抛错。

    参数：无。
    返回：None。
    异常：AssertionError，发生多次模型调用、账本丢失或检索层放行非法查询。
    """
    from app.core.model_budget import BudgetLedger
    from app.knowledge.query_rewrite import QueryExpander, QueryRewriteError

    bad_queries = ["订单1002退货条件", "MH-W99退货限制", "退货运费规则"]
    chunks = [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "submit_expanded_queries",
                    "args": json.dumps({"queries": bad_queries}, ensure_ascii=False),
                    "id": "exp-bad",
                    "index": 0,
                }
            ],
            response_metadata={"finish_reason": "tool_calls"},
        )
    ]
    model = RewriteModel(chunks)
    ledger = BudgetLedger()
    query = "订单1001能退货吗 已签收"
    failed_expansion = await QueryExpander(model.factory).expand(query, simple_query="订单1001退货条件", budget=ledger)
    assert failed_expansion.queries == [query]
    assert len(model.inputs) == 1
    snap = ledger.snapshot()
    assert snap["model_calls"] == 1 and snap["records"][0]["stage"] == "expand"
    assert snap["records"][0].get("failure") is not None

    search, encoder, index, reranker = _search([_chunk(1)])
    rewriter_model = RewriteModel()
    search.query_rewriter = QueryExpander(rewriter_model.factory)
    with pytest.raises(QueryRewriteError):
        await search.retrieve_policy(query, prepared_queries=bad_queries)
    assert len(rewriter_model.inputs) == 0
    assert not encoder.inputs and not index.calls and not reranker.calls


async def test_query_expander_propagates_cancel_budget_and_invalid_seed():
    """取消信号、工作流预算耗尽与非法种子不得伪装为扩写回退。

    参数：无。
    返回：None。
    异常：AssertionError，控制信号或输入边界被吞掉。
    """
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
    from app.knowledge.query_rewrite import QueryExpander, QueryRewriteError

    model = RewriteModel()
    with pytest.raises(QueryRewriteError):
        await QueryExpander(model.factory).expand("   ", simple_query="退货条件")
    with pytest.raises(QueryRewriteError):
        await QueryExpander(model.factory).expand("退" * 2001, simple_query="退货条件")
    with pytest.raises(QueryRewriteError):
        await QueryExpander(model.factory).expand("订单1001退货", simple_query="退" * 2001)
    with pytest.raises(QueryRewriteError):
        await QueryExpander(model.factory, request_budget=1).expand("订单1001退货", simple_query="退货条件")
    assert not model.inputs

    exhausted = BudgetLedger(max_model_calls=1)
    exhausted.reserve("understand", 10, 100)
    with pytest.raises(WorkflowBudgetExceeded):
        await QueryExpander(model.factory).expand("订单1001退货", simple_query="退货条件", budget=exhausted)
    assert not model.inputs

    waiting = RewriteModel(wait=True)
    budget = BudgetLedger()
    task = asyncio.create_task(
        QueryExpander(waiting.factory).expand("订单1001退货", simple_query="退货条件", budget=budget)
    )
    await waiting.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert waiting.closed and budget.snapshot()["model_calls"] == 1


@pytest.mark.parametrize("payload", [
    {"queries": ["已购滤芯能否退货"]},
    {"queries": ["凭空增加999天退货期"]},
    {"queries": ["", "", ""]},
])
async def test_required_policy_coverage_survives_bad_expansion(rewrite_sdk, payload):
    """核心政策检索不能因模型单条或坏输出丢失覆盖，也不能新增重试。参数：HTTP夹具/坏结果；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    from app.knowledge.query_rewrite import QueryExpander, validate_prepared_queries
    seed = "已购滤芯未拆封能否退货 已发货"
    rewrite_sdk.responses.append(rewrite_response(json.dumps(payload), name="submit_expanded_queries"))
    ledger = BudgetLedger()
    result = await QueryExpander(rewrite_sdk.factory).expand(seed, simple_query="已购滤芯能否退货",
                                                            required_count=3, budget=ledger)
    assert len(result.queries) == len(set(result.queries)) == 3
    assert all(seed in query for query in result.queries)
    validate_prepared_queries(seed, result.queries)
    assert len(rewrite_sdk.requests) == 1 and ledger.snapshot()["model_calls"] == 1
    assert ledger.snapshot()["records"][0]["failure"]


def test_policy_queries_preserve_product_models_but_not_order_identifiers():
    """订单号与商品型号必须有不同保留规则，不能为通过订单用例放开真实型号。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import validate_prepared_queries, QueryRewriteError
    seed = "订单SO-4801的ZX-W77滤芯如何保修"
    validate_prepared_queries(seed, ["ZX-W77保修条件", "ZX-W77保修限制", "ZX-W77保修流程"])
    with pytest.raises(QueryRewriteError, match="型号"):
        validate_prepared_queries(seed, ["其他滤芯如何保修"])
