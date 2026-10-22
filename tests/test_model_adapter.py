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


async def test_final_model_unbound_filters_reasoning_and_ignores_tool_calls(monkeypatch):
    """回答阶段不展示推理块；模型违规申请工具时丢弃该分片并继续等正文。

    参数：替换器；返回：None；异常：断言失败。
    背景：真实上游在“历史缺工具轮 + 本轮工具失败”时会在无工具绑定的回答请求里
    继续输出 tool_call 分片，旧实现直接抛 ValueError，用户看到“本轮暂时未能完成”。
    """
    from app.core import llm
    model = ChunkModel([AIMessageChunk(content=[{"type": "text", "text": "你好"}, {"type": "reasoning", "text": "SECRET"}])])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: model)
    assert [v async for v in llm.ChatModelAdapter().stream_answer([])] == ["你好"]
    assert not model.binds
    # 工具申请分片被丢弃，同一条流里的正常正文仍要送出，不能因违规分片整体失败。
    model.chunks = [
        AIMessageChunk(content="", tool_call_chunks=[{"name": "query_order", "id": "c", "args": "{}", "index": 0}]),
        AIMessageChunk(content="订单 1005 待发货"),
    ]
    assert [v async for v in llm.ChatModelAdapter().stream_answer([])] == ["订单 1005 待发货"]


async def test_answer_stream_retries_once_when_model_returns_no_text(monkeypatch):
    """模型整轮只返回工具申请或空正文时，重试一次而不是直接判失败。

    参数：替换器；返回：None；异常：断言失败。
    背景：上游偶发把回答阶段的分片全部用在 tool_call 上，正文为空；
    单次失败会让用户看到“本轮暂时未能完成”，重试一次可显著降低误报。
    """
    from app.core import llm

    class RecordingModel(ChunkModel):
        """记录每次收到的消息，用于确认重试确实改变了输入。"""
        def __init__(self, chunks):
            """初始化并创建输入记录。参数：分片；返回：None；异常：无。"""
            super().__init__(chunks)
            self.seen = []

        async def astream(self, messages):
            """记录输入后转发分片。参数：消息；返回：异步分片流；异常：无。"""
            self.seen.append(list(messages))
            for chunk in self.chunks:
                yield chunk

    first = RecordingModel([AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_faq", "id": "c1", "args": '{"keyword":"运费"}', "index": 0}])])
    second = RecordingModel([AIMessageChunk(content="重试后的正文")])
    models = [first, second]
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: models.pop(0))
    assert [v async for v in llm.ChatModelAdapter().stream_answer([HumanMessage(content="问")])] == ["重试后的正文"]
    # 首次请求不带纠正指令，重试时追加，保证正常路径的输入不被改变。
    assert len(first.seen[0]) == 1
    assert second.seen[0][-1].content == llm.ChatModelAdapter.ANSWER_NUDGE
    assert len(second.seen[0]) == 2


async def test_answer_stream_reports_error_when_both_attempts_have_no_text(monkeypatch):
    """两次尝试都没有正文时必须显式失败，不能返回空回答。

    参数：替换器；返回：None；异常：断言失败。
    """
    from app.core import llm
    empty = AIMessageChunk(content="", tool_call_chunks=[
        {"name": "query_faq", "id": "c1", "args": '{"keyword":"运费"}', "index": 0}])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: ChunkModel([empty]))
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
