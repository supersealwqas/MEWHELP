"""用真实SDK请求验证结构化结果只来自指定工具，拒绝纯文本JSON及协议违约。"""

import json

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from app.core.model_budget import BudgetLedger
from tests.workflow.test_model import sdk_runtime, call_response, text_response


class ResultPayload(BaseModel):
    """测试使用的严格结果契约，不允许未知字段或隐式类型转换。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    count: int = Field(ge=0, description="结果条数，必须是非负整数")


async def invoke_result(runtime, response, *, max_chars=1000, budget=None):
    """通过真实SDK调用结构化边界。参数：HTTP夹具、响应、字符上限、可选账本；返回：解析结果；异常：协议或预算错误。"""
    from app.core.structured import output_tool, bind_output_tool, collect_tool_output, parse_tool_output

    runtime.responses.append(response)
    tool = output_tool(ResultPayload, "submit_result")
    model = bind_output_tool(runtime.factory(streaming=True, max_tokens=128), tool)
    gathered = await collect_tool_output(model, [HumanMessage(content="统计结果")], max_chars=max_chars,
                                        budget=budget, stage="classify" if budget else None)
    return parse_tool_output(gathered, ResultPayload, "submit_result")


async def test_valid_result_uses_named_strict_tool_and_records_usage(sdk_runtime):
    """合法结构化结果须通过强制单工具且保留实际用量。参数：HTTP夹具；返回：None；异常：断言失败。"""
    ledger = BudgetLedger()
    ledger.reserve("classify", 100, 128)
    result = await invoke_result(sdk_runtime, call_response("submit_result", '{"count":2}', usage=True), budget=ledger)

    assert result.count == 2 and ledger.snapshot()["actual_tokens"] == 112
    request = sdk_runtime.requests[0]
    assert request["tool_choice"] == {"type": "function", "function": {"name": "submit_result"}}
    assert request["parallel_tool_calls"] is False
    assert request["tools"][0]["function"]["strict"] is True
    assert request["tools"][0]["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("response", [
    text_response('{"count":2}'),
    call_response("wrong_result", '{"count":2}'),
    call_response("submit_result", '{"count":"2"}'),
    call_response("submit_result", '{"count":1,"count":2}'),
    call_response("submit_result", '{"count":2,"extra":true}'),
    call_response("submit_result", '{"count":2}', duplicate=True),
], ids=["text-json", "wrong-tool", "coerced-type", "duplicate-key", "extra-field", "multiple-tools"])
async def test_result_rejects_text_json_and_invalid_tool_protocol(sdk_runtime, response):
    """文本JSON、错误工具及坏字段不得成为业务结果。参数：HTTP夹具及违约响应；返回：None；异常：断言失败。"""
    from app.core.structured import StructuredOutputError

    with pytest.raises(StructuredOutputError):
        await invoke_result(sdk_runtime, response)
    assert len(sdk_runtime.requests) == 1


async def test_tool_arguments_are_counted_in_output_limit(sdk_runtime):
    """只计算正文而漏掉工具参数长度时失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.structured import StructuredOutputError

    with pytest.raises(StructuredOutputError):
        await invoke_result(sdk_runtime, call_response("submit_result", '{"count":12345}'), max_chars=5)


async def test_truncated_tool_result_is_not_accepted(sdk_runtime):
    """参数看似完整但结束原因是截断时也必须失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.core.structured import StructuredOutputError

    response = call_response("submit_result", '{"count":2}')
    response[-1]["choices"][0]["finish_reason"] = "length"
    with pytest.raises(StructuredOutputError):
        await invoke_result(sdk_runtime, response)
