"""知识抽取/关系判定 Adapter 的流式聚合、校验与取消测试。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest
from langchain_core.messages import AIMessageChunk

from app.knowledge.llm import KnowledgeModel, KnowledgeModelError
from app.knowledge.schemas import (
    CompletedTurn,
    EvidenceItem,
    ExtractionBatch,
    KnowledgeText,
)


def _batch() -> ExtractionBatch:
    """构造单轮抽取批次。参数：无；返回：批次；异常：校验失败。"""
    turn = CompletedTurn(
        message_id=1,
        conversation_id=1,
        turn_id="turn-1",
        question="邮费怎么算",
        answer="未满99元收6元。",
    )
    return ExtractionBatch(
        batch_no="batch-1",
        source_ids=[1],
        snapshot_upper_id=1,
        input_hash="input-hash",
        prompt_version="prompt-v1",
        turns=[turn],
    )


def _valid_extraction() -> str:
    """返回合法抽取 JSON。参数：无；返回：文本；异常：无。"""
    return json.dumps(
        {
            "qas": [
                {
                    "source_ids": [1],
                    "question": "邮费怎么算",
                    "answer": "未满99元收6元。",
                    "category": "物流",
                    "answer_excerpts": [{"message_id": 1, "quote": "未满99元收6元。"}],
                }
            ]
        },
        ensure_ascii=False,
    )


def _tool_chunk(payload: str, name: str = "submit_extracted_knowledge") -> AIMessageChunk:
    """构造已正常完成的结构化工具结果。参数：原始参数与工具名；返回值：分片；异常：无。"""
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": payload, "id": "call-1", "index": 0}],
        response_metadata={"finish_reason": "tool_calls"},
    )


class _ScriptedStream:
    """按预置分片返回内容并可记录关闭。"""

    def __init__(self, chunks: list[AIMessageChunk | Exception]) -> None:
        """保存分片。参数：分片列表；返回：None；异常：无。"""
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> "_ScriptedStream":
        """返回自身。参数：无；返回：异步迭代器；异常：无。"""
        return self

    async def __anext__(self) -> AIMessageChunk:
        """返回下一个分片。参数：无；返回：分片；异常：预置异常或 StopAsyncIteration。"""
        if not self._chunks:
            raise StopAsyncIteration
        chunk = self._chunks.pop(0)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    async def aclose(self) -> None:
        """标记流关闭。参数：无；返回：None；异常：无。"""
        self.closed = True


class _WaitingStream:
    """首片后永久等待，用于取消测试。"""

    def __init__(self) -> None:
        """初始化事件。参数：无；返回：None；异常：无。"""
        self.started = anyio.Event()
        self.closed = False
        self._first = True

    def __aiter__(self) -> "_WaitingStream":
        """返回自身。参数：无；返回：异步迭代器；异常：无。"""
        return self

    async def __anext__(self) -> AIMessageChunk:
        """首片后等待取消。参数：无；返回：首片；异常：CancelledError。"""
        if self._first:
            self._first = False
            self.started.set()
            return AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "submit_extracted_knowledge", "args": '{"qas": [',
                    "id": "call-1", "index": 0,
                }],
            )
        await anyio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        """标记流关闭。参数：无；返回：None；异常：无。"""
        self.closed = True


class _FakeChatModel:
    """返回预置流的聊天模型。"""

    def __init__(self, stream: Any) -> None:
        """保存流。参数：流；返回：None；异常：无。"""
        self.stream = stream
        self.messages: Any = None

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> "_FakeChatModel":
        """接受工具绑定并保留预置流。参数：工具定义和绑定选项；返回值：当前模型；异常：无。"""
        return self

    def astream(self, messages: Any) -> Any:
        """记录消息并返回流。参数：消息；返回：流；异常：无。"""
        self.messages = messages
        return self.stream


def _model(stream: Any) -> tuple[KnowledgeModel, _FakeChatModel]:
    """构造注入假模型的 KnowledgeModel。参数：流；返回：模型与假客户端；异常：无。"""
    client = _FakeChatModel(stream)
    return KnowledgeModel(lambda **kwargs: client), client


async def test_aggregate_tool_stream_before_argument_validation() -> None:
    """完整聚合工具参数分片后才校验。参数：无；返回：None；异常：无。"""
    text = _valid_extraction()
    stream = _ScriptedStream(
        [
            AIMessageChunk(content="", tool_call_chunks=[{
                "name": "submit_extracted_knowledge", "args": text[:10], "id": "call-1", "index": 0,
            }]),
            AIMessageChunk(content="", tool_call_chunks=[{
                "name": None, "args": text[10:40], "id": None, "index": 0,
            }]),
            AIMessageChunk(content="", tool_call_chunks=[{
                "name": None, "args": text[40:], "id": None, "index": 0,
            }]),
            AIMessageChunk(content="", response_metadata={"finish_reason": "tool_calls"}),
        ]
    )
    model, _ = _model(stream)
    qas = await model.extract(_batch())
    assert len(qas) == 1
    assert qas[0].source_ids == [1]
    assert stream.closed is True


@pytest.mark.parametrize("answer", ["未满99元收600元。", "满99元包邮。"])
async def test_grounding_rejects_invented_amount_or_shipping(answer: str) -> None:
    """原文不支持的金额或包邮政策必须拒绝。参数：伪造答案；返回：None；异常：无。"""
    raw = json.loads(_valid_extraction())
    raw["qas"][0]["answer"] = answer
    model, _ = _model(_ScriptedStream([_tool_chunk(json.dumps(raw, ensure_ascii=False))]))
    with pytest.raises(KnowledgeModelError, match="原句|证据|答案"):
        await model.extract(_batch())


async def test_grounding_accepts_ordered_complete_sentence_summary() -> None:
    """允许省略寒暄并组合多个完整原句。参数：无；返回：None；异常：无。"""
    batch = _batch()
    turn = batch.turns[0].model_copy(update={"answer": "您好。未满99元收6元。欢迎咨询。偏远地区不包邮。"})
    batch = batch.model_copy(update={"turns": [turn]})
    raw = json.loads(_valid_extraction())
    raw["qas"][0].update(answer="未满99元收6元。\n偏远地区不包邮。", answer_excerpts=[
        {"message_id": 1, "quote": "未满99元收6元。"},
        {"message_id": 1, "quote": "偏远地区不包邮。"},
    ])
    model, _ = _model(_ScriptedStream([_tool_chunk(json.dumps(raw, ensure_ascii=False))]))
    assert len(await model.extract(batch)) == 1


@pytest.mark.parametrize("answer,quotes", [
    ("未满99元收6元。", ["未满99元收6元。"]),
    ("偏远地区包邮。", ["偏远地区包邮。"]),
    ("未满99元收600元。偏远地区不包邮。", ["未满99元收600元。", "偏远地区不包邮。"]),
])
async def test_grounding_rejects_omitted_condition_or_fake_excerpt(answer, quotes) -> None:
    """伪摘录与漏掉偏远不包邮条件不可发布。参数：答案与摘录；返回：None；异常：无。"""
    batch = _batch()
    batch = batch.model_copy(update={"turns": [batch.turns[0].model_copy(update={"answer": "未满99元收6元。偏远地区不包邮。"})]})
    raw = json.loads(_valid_extraction())
    raw["qas"][0].update(answer=answer, answer_excerpts=[{"message_id": 1, "quote": quote} for quote in quotes])
    model, _ = _model(_ScriptedStream([_tool_chunk(json.dumps(raw, ensure_ascii=False))]))
    with pytest.raises(KnowledgeModelError, match="原句|条件|证据"):
        await model.extract(batch)


async def test_grounding_rejects_omitted_activity_end_condition() -> None:
    """无否定关键词的活动截止条件也必须保留。参数：无；返回：None；异常：无。"""
    batch = _batch()
    batch = batch.model_copy(update={"turns": [batch.turns[0].model_copy(update={"answer": "商品售价6元。活动结束后恢复原价。"})]})
    raw = json.loads(_valid_extraction())
    raw["qas"][0].update(answer="商品售价6元。", answer_excerpts=[{"message_id": 1, "quote": "商品售价6元。"}])
    model, _ = _model(_ScriptedStream([_tool_chunk(json.dumps(raw, ensure_ascii=False))]))
    with pytest.raises(KnowledgeModelError, match="条件"):
        await model.extract(batch)


async def test_grounding_preserves_closing_quote() -> None:
    """原句的闭合引号不可被当作可删除碎片。参数：无；返回：None；异常：无。"""
    batch = _batch()
    quote = "客服说明：“未满99元收6元。”"
    batch = batch.model_copy(update={"turns": [batch.turns[0].model_copy(update={"answer": quote})]})
    raw = json.loads(_valid_extraction())
    raw["qas"][0].update(answer=quote, answer_excerpts=[{"message_id": 1, "quote": quote}])
    model, _ = _model(_ScriptedStream([_tool_chunk(json.dumps(raw, ensure_ascii=False))]))
    assert (await model.extract(batch))[0].answer == quote


@pytest.mark.parametrize("answer,accepted", [("商品售价6元。", False), ("活动结束后恢复原价\n商品售价6元。", True)])
async def test_grounding_covers_unpunctuated_business_lines(answer, accepted) -> None:
    """无句号业务行不得被跳过，完整多行合法。参数：答案/是否通过；返回：None；异常：无。"""
    batch = _batch()
    batch = batch.model_copy(update={"turns": [batch.turns[0].model_copy(update={"answer": "活动结束后恢复原价\n商品售价6元。"})]})
    raw = json.loads(_valid_extraction())
    raw["qas"][0].update(answer=answer, answer_excerpts=[{"message_id": 1, "quote": quote} for quote in answer.splitlines()])
    model, _ = _model(_ScriptedStream([_tool_chunk(json.dumps(raw, ensure_ascii=False))]))
    if accepted:
        assert len(await model.extract(batch)) == 1
    else:
        with pytest.raises(KnowledgeModelError):
            await model.extract(batch)


async def test_grounding_never_normalizes_number_internal_space() -> None:
    """6 00不得在原句核验时变为600。参数：无；返回：None；异常：无。"""
    batch = _batch()
    batch = batch.model_copy(update={"turns": [batch.turns[0].model_copy(update={"answer": "运费6 00元。"})]})
    raw = json.loads(_valid_extraction())
    raw["qas"][0].update(answer="运费600元。", answer_excerpts=[{"message_id": 1, "quote": "运费600元。"}])
    model, _ = _model(_ScriptedStream([_tool_chunk(json.dumps(raw, ensure_ascii=False))]))
    with pytest.raises(KnowledgeModelError):
        await model.extract(batch)


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps(
            {
                "qas": [
                    {
                        "source_ids": [999],
                        "question": "邮费怎么算",
                        "answer": "答案。",
                        "category": "物流",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "qas": [
                    {
                        "source_ids": [1],
                        "question": "改写后的问题",
                        "answer": "答案。",
                        "category": "物流",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ],
    ids=["unknown-source", "rewritten-question"],
)
async def test_unknown_source_or_rewritten_question_rejected(payload: str) -> None:
    """未知来源或改写问题被拒绝。参数：非法 JSON；返回：None；异常：无。"""
    model, _ = _model(_ScriptedStream([_tool_chunk(payload)]))
    with pytest.raises(KnowledgeModelError, match="来源|问题"):
        await model.extract(_batch())


async def test_nan_or_unexpected_tool_rejected() -> None:
    """NaN 数值或未绑定工具被拒绝。参数：无；返回：None；异常：无。"""
    nan_payload = '{"qas": [{"source_ids": [1], "question": "邮费怎么算", "answer": NaN, "category": "物流"}]}'
    model, _ = _model(_ScriptedStream([_tool_chunk(nan_payload)]))
    with pytest.raises(KnowledgeModelError):
        await model.extract(_batch())

    tool_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "leak", "args": "{}", "id": "call-1", "index": 0}],
        response_metadata={"finish_reason": "tool_calls"},
    )
    model, _ = _model(_ScriptedStream([tool_chunk]))
    with pytest.raises(KnowledgeModelError, match="工具"):
        await model.extract(_batch())


async def test_empty_valid_batch_is_distinct_from_model_failure() -> None:
    """空 qas 是成功，空响应是模型失败。参数：无；返回：None；异常：无。"""
    model, _ = _model(_ScriptedStream([_tool_chunk('{"qas": []}')]))
    assert await model.extract(_batch()) == []

    model, _ = _model(_ScriptedStream([AIMessageChunk(content="")]))
    with pytest.raises(KnowledgeModelError):
        await model.extract(_batch())


async def test_cancel_closes_stream() -> None:
    """取消抽取时关闭底层流。参数：无；返回：None；异常：无。"""
    stream = _WaitingStream()
    model, _ = _model(stream)
    task = asyncio.create_task(model.extract(_batch()))
    await stream.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed is True


def _relation_inputs() -> tuple[list[EvidenceItem], list[tuple[str, str]]]:
    """构造关系判定输入。参数：无；返回：证据与 pair；异常：校验失败。"""
    items = [
        EvidenceItem(
            ref="left",
            origin="document",
            text=KnowledgeText(category="分类", questions="问法", answer="答案一。"),
            source_refs=["doc:a"],
        ),
        EvidenceItem(
            ref="right",
            origin="conversation",
            text=KnowledgeText(category="分类", questions="问法", answer="答案二。"),
            source_refs=["qa:1"],
        ),
    ]
    return items, [("left", "right")]


async def test_relate_rejects_unknown_or_missing_pairs() -> None:
    """关系结果必须覆盖请求 pair 且引用合法。参数：无；返回：None；异常：无。"""
    items, pairs = _relation_inputs()
    bad_payloads = [
        json.dumps(
            {
                "decisions": [
                    {
                        "left_ref": "left",
                        "right_ref": "unknown",
                        "relation": "conflict",
                        "evidence_refs": ["left"],
                        "reason": "错误引用",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        json.dumps({"decisions": []}, ensure_ascii=False),
    ]
    for payload in bad_payloads:
        model, _ = _model(_ScriptedStream([_tool_chunk(payload, "submit_knowledge_relations")]))
        with pytest.raises(KnowledgeModelError):
            await model.relate(items, pairs)


async def test_relate_valid_decision() -> None:
    """合法关系判定按请求 pair 返回。参数：无；返回：None；异常：无。"""
    items, pairs = _relation_inputs()
    payload = json.dumps(
        {
            "decisions": [
                {
                    "left_ref": "left",
                    "right_ref": "right",
                    "relation": "conflict",
                    "evidence_refs": ["left", "right"],
                    "reason": "数字不同",
                }
            ]
        },
        ensure_ascii=False,
    )
    model, _ = _model(_ScriptedStream([_tool_chunk(payload, "submit_knowledge_relations")]))
    decisions = await model.relate(items, pairs)
    assert len(decisions) == 1
    assert decisions[0].relation == "conflict"
