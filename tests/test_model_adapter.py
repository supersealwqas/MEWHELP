"""模型适配层的真实消息分片聚合及单轮终止约束。"""

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage


class ChunkModel:
    """替代模型网络连接，保留 LangChain 真实消息类型。"""
    def __init__(self, chunks):
        """设置网络分片。参数：列表；返回：None；异常：无。"""
        self.chunks, self.binds = chunks, []

    def bind_tools(self, tools, **kwargs):
        """记录模型可见参数。参数：工具、选项；返回：自身；异常：无。"""
        self.binds.append(kwargs)
        return self

    async def astream(self, messages):
        """生成网络分片。参数：消息；返回：异步分片流；异常：无。"""
        for chunk in self.chunks:
            yield chunk


async def test_decide_aggregates_complete_json_without_leaking_partial_args(monkeypatch):
    """分片未聚合或遗漏 auto 绑定会失败。参数：替换器；返回：None；异常：断言失败。"""
    from app.core import llm
    model = ChunkModel([
        AIMessageChunk(content="", tool_call_chunks=[{"name": "query_logistics", "id": "c", "args": '{"order_', "index": 0}]),
        AIMessageChunk(content="", tool_call_chunks=[{"name": None, "id": None, "args": 'id":"1001"}', "index": 0}])])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: model)
    result = await llm.ChatModelAdapter().decide([HumanMessage(content="查物流")], [])
    assert result.tool_calls[0]["args"] == {"order_id": "1001"} and result.tool_calls[0]["id"] == "c"
    assert model.binds == [{"tool_choice": "auto"}]


async def test_truncated_json_not_accepted_by_partial_json_parser(monkeypatch):
    """LangChain 宽松补全不能把截断参数当成完整申请。参数：替换器；返回：None；异常：断言失败。"""
    from app.core import llm
    model = ChunkModel([AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_logistics", "id": "c", "args": '{"order_id":"1001"', "index": 0}])])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: model)
    with pytest.raises(ValueError):
        await llm.ChatModelAdapter().decide([], [])


@pytest.mark.parametrize("raw", ['{"keyword":"退货"', "", "[]",
    '{"keyword":NaN}', '{"keyword":Infinity}', '{"keyword":-Infinity}',
    '{"keyword":1e999}', '{"keyword":{"nested":[NaN]}}'])
async def test_invalid_arguments_preserve_raw_audit(monkeypatch, raw):
    """非法参数保留原始诊断而不伪造合法调用。参数：替换器、原参数；返回：None；异常：断言失败。"""
    from app.core import llm
    from app.schemas.chat import InvalidToolDecision
    model = ChunkModel([AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_faq", "id": "bad", "args": raw, "index": 0}])])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: model)
    with pytest.raises(InvalidToolDecision) as failure:
        await llm.ChatModelAdapter().decide([], [])
    assert failure.value.audit_record["type"] == "invalid_tool_decision"
    assert failure.value.audit_record["tool_call_chunks"][0]["args"] == raw


async def test_final_model_unbound_filters_reasoning_and_rejects_tool_calls(monkeypatch):
    """回答阶段不能继续选工具，也不能展示推理块。参数：替换器；返回：None；异常：断言失败。"""
    from app.core import llm
    model = ChunkModel([AIMessageChunk(content=[{"type": "text", "text": "你好"}, {"type": "reasoning", "text": "SECRET"}])])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: model)
    assert [v async for v in llm.ChatModelAdapter().stream_answer([])] == ["你好"]
    assert not model.binds
    model.chunks = [AIMessageChunk(content="", tool_call_chunks=[{"name": "query_order", "id": "c", "args": "{}", "index": 0}])]
    with pytest.raises(ValueError):
        [v async for v in llm.ChatModelAdapter().stream_answer([])]


async def test_closing_answer_immediately_closes_upstream_stream(monkeypatch):
    """停止回复必须等待模型流关闭，不依赖垃圾回收。参数：替换器；返回：None；异常：断言失败。"""
    from app.core import llm
    closed = []
    class StreamingModel:
        """可观察关闭的网络边界。"""
        async def astream(self, messages):
            """返回首块后等待下一次拉取。参数：消息；返回：分片流；异常：取消。"""
            try:
                yield AIMessageChunk(content="第一块")
                yield AIMessageChunk(content="第二块")
            finally:
                closed.append(True)
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: StreamingModel())
    stream = llm.ChatModelAdapter().stream_answer([])
    assert await anext(stream) == "第一块"
    await stream.aclose()
    assert closed == [True]
