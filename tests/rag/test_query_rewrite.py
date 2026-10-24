"""单轮归一的真实解析、检索编排及取消边界测试，仅替换模型流。"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessageChunk

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
                yield AIMessageChunk(content=self.payload)
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

    broken = RewriteModel(error=OSError("network"))
    with pytest.raises(QueryRewriteError):
        await QueryRewriter(broken.factory).rewrite("如何清洗")
    assert broken.closed
    waiting = RewriteModel(wait=True)
    task = asyncio.create_task(QueryRewriter(waiting.factory).rewrite("如何清洗"))
    await waiting.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert waiting.closed


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
