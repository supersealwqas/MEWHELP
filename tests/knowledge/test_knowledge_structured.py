"""通过真实聊天 SDK 验证知识模型的强制工具协议、字段边界和字符预算。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI

from app.knowledge.llm import KnowledgeModel, KnowledgeModelError
from app.knowledge.prompts import extraction_messages, messages_size, relation_input_size
from tests.knowledge.test_knowledge_llm import _batch, _relation_inputs, _valid_extraction


def _response(raw: str, name: str, *, plain: bool = False) -> list[dict[str, Any]]:
    """构造 SDK 接收的完整事件流。

    参数：raw 为原始参数文本，name 为工具名，plain 表示改为自由文本响应。
    返回值：包含首块与正常终止块的事件列表。
    异常：无。
    """
    delta: dict[str, Any] = {"role": "assistant"}
    if plain:
        delta["content"] = raw
    else:
        delta["tool_calls"] = [{
            "index": 0,
            "id": "knowledge-call-1",
            "type": "function",
            "function": {"name": name, "arguments": raw},
        }]
    return [
        {
            "id": "knowledge-response",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "offline-model",
            "choices": [{"index": 0, "delta": content, "finish_reason": finish}],
        }
        for content, finish in [(delta, None), ({}, "stop" if plain else "tool_calls")]
    ]


@pytest.fixture
async def sdk_runtime():
    """仅替代 HTTP 边界，保留 SDK 的请求序列化和流解析。

    参数：无。
    返回值：含预置响应、实际请求与模型工厂的夹具。
    异常：额外请求触发 AssertionError，SDK 错误原样传播。
    """
    requests: list[dict[str, Any]] = []
    responses: list[list[dict[str, Any]]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        """返回本地预置事件流。参数：实际 HTTP 请求；返回值：响应；异常：额外请求断言失败。"""
        requests.append(json.loads(request.content))
        assert responses, "不允许额外模型请求"
        body = "".join("data: " + json.dumps(item) + "\n\n" for item in responses.pop(0))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body + "data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        def factory(**kwargs: Any) -> ChatOpenAI:
            """创建真实模型客户端。参数：适配器选项；返回值：聊天模型；异常：配置错误。"""
            return ChatOpenAI(
                base_url="http://offline.invalid/v1",
                api_key="offline",
                model="offline-model",
                http_async_client=client,
                **kwargs,
            )

        yield SimpleNamespace(factory=factory, requests=requests, responses=responses)


def _relation_payload() -> str:
    """构造完整合法的关系参数。参数：无；返回值：参数文本；异常：无。"""
    return json.dumps({"decisions": [{
        "left_ref": "left",
        "right_ref": "right",
        "relation": "uncertain",
        "evidence_refs": ["left", "right"],
        "reason": "缺少足以比较的事实",
    }]}, ensure_ascii=False)


def _assert_strict_objects(schema: Any) -> None:
    """核对嵌套对象不允许额外字段。参数：schema 子树；返回值：None；异常：断言失败。"""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
        for value in schema.values():
            _assert_strict_objects(value)
    elif isinstance(schema, list):
        for value in schema:
            _assert_strict_objects(value)


@pytest.mark.parametrize("kind", ["extraction", "relation"])
async def test_sdk_requires_one_named_strict_knowledge_tool(sdk_runtime, kind: str) -> None:
    """知识任务未强制唯一严格工具或不能读取工具结果时失败。参数：网络夹具、任务；返回值：None；异常：断言失败。"""
    name = "submit_extracted_knowledge" if kind == "extraction" else "submit_knowledge_relations"
    payload = _valid_extraction() if kind == "extraction" else _relation_payload()
    sdk_runtime.responses.append(_response(payload, name))
    model = KnowledgeModel(sdk_runtime.factory)
    if kind == "extraction":
        result = await model.extract(_batch())
        assert result[0].question == "邮费怎么算"
        assert result[0].source_ids == [1]
    else:
        result = await model.relate(*_relation_inputs())
        assert result[0].relation == "uncertain"

    assert len(sdk_runtime.requests) == 1
    request = sdk_runtime.requests[0]
    assert request["tool_choice"] == {"type": "function", "function": {"name": name}}
    assert request["parallel_tool_calls"] is False
    assert len(request["tools"]) == 1
    tool = request["tools"][0]["function"]
    assert tool["name"] == name and tool["strict"] is True
    assert set(tool["parameters"]["properties"]) == ({"qas"} if kind == "extraction" else {"decisions"})
    _assert_strict_objects(tool["parameters"])
    serialized = json.dumps(tool["parameters"])
    assert all(f'"{field}"' not in serialized for field in (
        "verified", "offset", "hash", "source_evidence", "calibration_digest",
    ))
    assert request["stream"] is True and request["temperature"] == 0


@pytest.mark.parametrize("kind", ["extraction", "relation"])
async def test_sdk_rejects_valid_json_returned_as_free_text(sdk_runtime, kind: str) -> None:
    """自由文本 JSON 仍被当作结构结果解析时失败。参数：网络夹具、任务；返回值：None；异常：断言失败。"""
    name = "submit_extracted_knowledge" if kind == "extraction" else "submit_knowledge_relations"
    payload = _valid_extraction() if kind == "extraction" else _relation_payload()
    sdk_runtime.responses.append(_response(payload, name, plain=True))
    model = KnowledgeModel(sdk_runtime.factory)
    with pytest.raises(KnowledgeModelError):
        if kind == "extraction":
            await model.extract(_batch())
        else:
            await model.relate(*_relation_inputs())
    assert len(sdk_runtime.requests) == 1


@pytest.mark.parametrize("field,value", [
    ("source_ids", ["1"]),
    ("verified", True),
    ("answer_excerpts", [{"message_id": 1, "quote": "未满99元收6元。", "verified": True}]),
])
async def test_sdk_rejects_coerced_identifiers_or_server_owned_fields(sdk_runtime, field: str, value: Any) -> None:
    """来源 ID 被强转或模型自报可信标记时失败。参数：网络夹具、被篡改字段和值；返回值：None；异常：断言失败。"""
    payload = json.loads(_valid_extraction())
    payload["qas"][0][field] = value
    sdk_runtime.responses.append(_response(json.dumps(payload), "submit_extracted_knowledge"))
    with pytest.raises(KnowledgeModelError):
        await KnowledgeModel(sdk_runtime.factory).extract(_batch())


async def test_extraction_input_budget_includes_tool_schema_before_request(sdk_runtime) -> None:
    """消息未超限但加入工具后超限仍发包时失败。参数：网络夹具；返回值：None；异常：断言失败。"""
    batch = _batch()
    padding = 11990 - messages_size(extraction_messages(batch))
    turn = batch.turns[0].model_copy(update={"answer": batch.turns[0].answer + "甲" * padding})
    batch = batch.model_copy(update={"turns": [turn]})
    assert messages_size(extraction_messages(batch)) == 11990
    sdk_runtime.responses.append(_response('{"qas": []}', "submit_extracted_knowledge"))
    with pytest.raises(KnowledgeModelError, match="预算"):
        await KnowledgeModel(sdk_runtime.factory).extract(batch)
    assert sdk_runtime.requests == []


async def test_relation_batch_estimate_matches_adapter_budget(sdk_runtime, monkeypatch) -> None:
    """分批算法未计工具或与调用侧预算不一致时失败。参数：网络夹具、替换器；返回值：None；异常：断言失败。"""
    from app.knowledge import llm
    from app.knowledge.prompts import relation_messages

    items, pairs = _relation_inputs()
    estimate = relation_input_size(items, pairs)
    assert estimate > messages_size(relation_messages(items, pairs))
    sdk_runtime.responses.append(_response(_relation_payload(), "submit_knowledge_relations"))
    monkeypatch.setattr(llm, "EXTRACTION_INPUT_BUDGET", estimate)
    assert len(await KnowledgeModel(sdk_runtime.factory).relate(items, pairs)) == 1
    monkeypatch.setattr(llm, "EXTRACTION_INPUT_BUDGET", estimate - 1)
    with pytest.raises(KnowledgeModelError, match="预算"):
        await KnowledgeModel(sdk_runtime.factory).relate(items, pairs)
    assert len(sdk_runtime.requests) == 1


async def test_tool_arguments_count_toward_output_limit(sdk_runtime) -> None:
    """仅限制正文而漏掉工具参数长度时失败。参数：网络夹具；返回值：None；异常：断言失败。"""
    payload = json.loads(_valid_extraction())
    payload["qas"][0]["answer"] = "甲" * 24001
    sdk_runtime.responses.append(_response(json.dumps(payload, ensure_ascii=False), "submit_extracted_knowledge"))
    with pytest.raises(KnowledgeModelError, match="上限|超限|长度"):
        await KnowledgeModel(sdk_runtime.factory).extract(_batch())
