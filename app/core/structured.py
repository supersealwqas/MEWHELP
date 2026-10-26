"""统一的结构化输出边界：强制单个结果工具，保留原始分片并验证结构与结束状态。"""

import json
from typing import TYPE_CHECKING, TypeVar

import anyio
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from app.core.model_budget import BudgetLedger

ResultModel = TypeVar("ResultModel", bound=BaseModel)


class StructuredOutputError(ValueError):
    """结构化工具缺失、参数非法、输出超限或异常结束；不得退回解析自由文本。"""


def output_tool(schema: type[BaseModel], name: str) -> dict:
    """生成严格的结果工具定义。参数：schema结果模型、name固定工具名；返回：OpenAI工具定义；异常：不支持的schema转换错误。"""
    tool = convert_to_openai_tool(schema, strict=True)
    tool["function"]["name"] = name
    return tool


def bind_output_tool(model, tool: dict):
    """强制模型提交唯一结果工具。参数：model兼容聊天模型、tool严格定义；返回：绑定模型；异常：SDK绑定或网关参数错误。"""
    return model.bind_tools([tool], tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
                            parallel_tool_calls=False)


def output_chunk_size(chunk) -> int:
    """统计流分片的正文和原始工具参数长度。参数：chunk消息分片；返回：字符数；异常：不可序列化的非标准工具参数。"""
    content = getattr(chunk, "content", "")
    size = len(content) if isinstance(content, str) else sum(
        len(block) if isinstance(block, str) else len(block.get("text", ""))
        for block in (content or []) if isinstance(block, str) or
        (isinstance(block, dict) and isinstance(block.get("text"), str)))
    chunks = getattr(chunk, "tool_call_chunks", None) or []
    if chunks:
        return size + sum(len(call.get("args") or "") for call in chunks)
    # 完整消息替身或非增量兼容实现可能仅提供已解析参数，也必须计入输出上限。
    return size + sum(len(json.dumps(call.get("args"), ensure_ascii=False))
                      for call in (getattr(chunk, "tool_calls", None) or []))


async def collect_tool_output(model, messages, *, max_chars: int,
                              budget: "BudgetLedger | None" = None, stage: str | None = None):
    """有界聚合原始结构化流，在所有退出路径关闭连接并记录可观测用量。

    参数：model已绑定工具的模型、messages完整消息、max_chars正数字符上限；budget/stage可选已占额账本与阶段。
    返回：完整AIMessageChunk，保留原始工具参数、usage及结束原因，交由parse_tool_output验证。
    异常：StructuredOutputError，空流或超限；网络和取消原样传播；账本未占额时ValueError。
    """
    if max_chars < 1 or (budget is not None and stage is None):
        raise ValueError("结构化输出需有效字符上限；提供账本时必须指定阶段")
    gathered, size = None, 0
    stream = model.astream(messages)
    try:
        async for chunk in stream:
            gathered = chunk if gathered is None else gathered + chunk
            size += output_chunk_size(chunk)
            if size > max_chars:
                raise StructuredOutputError("结构化输出超过长度上限")
    finally:
        try:
            if budget is not None:
                budget.record_usage(stage, gathered.usage_metadata if gathered is not None else None,
                    gathered.response_metadata.get("finish_reason") if gathered is not None else None)
        finally:
            with anyio.CancelScope(shield=True):
                await stream.aclose()
    if gathered is None:
        raise StructuredOutputError("模型未返回结构化结果")
    return gathered


def parse_tool_output(gathered, schema: type[ResultModel], name: str) -> ResultModel:
    """只验证指定结果工具，绝不从模型正文读取JSON。

    参数：gathered完整原始消息、schema严格结果模型、name本次唯一允许的结果工具名。
    返回：通过结构和类型校验的结果模型；来源、引用等业务校验仍由调用方负责。
    异常：StructuredOutputError，异常结束、工具数量/名称/ID不符、参数JSON或字段非法。
    """
    # 懒导入使旧适配器也能复用本边界，不形成core.llm与知识模块的初始化循环。
    from app.core.llm import decision_message
    from app.schemas.chat import InvalidToolDecision

    if gathered.response_metadata.get("finish_reason") != "tool_calls":
        raise StructuredOutputError("结构化结果没有以工具调用正常结束")
    try:
        message = decision_message(gathered)
        if message.invalid_tool_calls or len(message.tool_calls) != 1:
            raise ValueError("必须提交唯一结构化工具")
        call = message.tool_calls[0]
        if call.get("name") != name or not isinstance(call.get("id"), str) or not call["id"].strip():
            raise ValueError("结构化工具名称或ID非法")
        return schema.model_validate(call["args"], strict=True)
    except (InvalidToolDecision, ValueError, TypeError, ValidationError) as error:
        raise StructuredOutputError("模型结构化工具结果不合法") from error
