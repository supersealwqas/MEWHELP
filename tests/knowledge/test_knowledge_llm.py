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
                }
            ]
        },
        ensure_ascii=False,
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
            return AIMessageChunk(content='{"qas": [')
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

    def astream(self, messages: Any) -> Any:
        """记录消息并返回流。参数：消息；返回：流；异常：无。"""
        self.messages = messages
        return self.stream


def _model(stream: Any) -> tuple[KnowledgeModel, _FakeChatModel]:
    """构造注入假模型的 KnowledgeModel。参数：流；返回：模型与假客户端；异常：无。"""
    client = _FakeChatModel(stream)
    return KnowledgeModel(lambda **kwargs: client), client


async def test_aggregate_stream_before_json_validation() -> None:
    """完整聚合分片后才做 JSON 校验。参数：无；返回：None；异常：无。"""
    text = _valid_extraction()
    stream = _ScriptedStream(
        [
            AIMessageChunk(content=text[:10]),
            AIMessageChunk(content=text[10:40]),
            AIMessageChunk(content=text[40:]),
        ]
    )
    model, _ = _model(stream)
    qas = await model.extract(_batch())
    assert len(qas) == 1
    assert qas[0].source_ids == [1]
    assert stream.closed is True


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
    model, _ = _model(_ScriptedStream([AIMessageChunk(content=payload)]))
    with pytest.raises(KnowledgeModelError, match="来源|问题"):
        await model.extract(_batch())


async def test_nan_or_tool_call_rejected() -> None:
    """NaN 数值或工具申请被拒绝。参数：无；返回：None；异常：无。"""
    nan_payload = '{"qas": [{"source_ids": [1], "question": "邮费怎么算", "answer": NaN, "category": "物流"}]}'
    model, _ = _model(_ScriptedStream([AIMessageChunk(content=nan_payload)]))
    with pytest.raises(KnowledgeModelError):
        await model.extract(_batch())

    tool_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "leak", "args": "{}", "id": "call-1"}],
    )
    model, _ = _model(_ScriptedStream([tool_chunk]))
    with pytest.raises(KnowledgeModelError, match="工具"):
        await model.extract(_batch())


async def test_empty_valid_batch_is_distinct_from_model_failure() -> None:
    """空 qas 是成功，空响应是模型失败。参数：无；返回：None；异常：无。"""
    model, _ = _model(_ScriptedStream([AIMessageChunk(content='{"qas": []}')]))
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
        model, _ = _model(_ScriptedStream([AIMessageChunk(content=payload)]))
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
    model, _ = _model(_ScriptedStream([AIMessageChunk(content=payload)]))
    decisions = await model.relate(items, pairs)
    assert len(decisions) == 1
    assert decisions[0].relation == "conflict"
