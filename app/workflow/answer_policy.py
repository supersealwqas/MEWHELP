"""本轮查询结果的确定性答复策略，供生产正文节点与离线评估共用。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_messages
from pydantic import ValidationError

from app.core.memory import valid_tool_calls
from app.schemas.tools import ToolResult
from app.tools.contracts import binding_from_dict
from app.tools.registry import ToolRegistry
from app.workflow.state import ConversationState


QUERY_FAULT_ANSWER = "本次查询未能完成，暂时无法核实相关信息。请稍后重试。"


def _strict_result(content: str) -> ToolResult | None:
    """校验服务端结果结构，拒绝类型强转、矛盾结果和缺失的错误分类。

    参数：content 为完整 ToolMessage 的 JSON 正文，不接受用户消息。
    返回：严格且成功/失败互斥的 ToolResult；结构不可靠时为 None。
    异常：无；JSON 和模型校验失败均保守返回 None。
    """
    try:
        payload = json.loads(content)
        result = ToolResult.model_validate(payload, strict=True, extra="forbid")
    except (ValueError, TypeError, ValidationError):
        return None
    if result.ok:
        return result if result.error is None and result.data is not None else None
    # ToolError 会给旧错误自动补分类；固定话术只接受原始结果的显式分类，不能猜测。
    error = payload.get("error")
    if result.data is not None or result.error is None or not isinstance(error, dict) or error.get("category") is None:
        return None
    return result


def query_fault_answer(state: ConversationState, *, registry: ToolRegistry) -> str | None:
    """仅在本轮可信读查询明确故障且没有可用成功事实时给出固定答复。

    参数：state 为服务器维护的原轮状态，须含 context.turn_id、raw_query、
        current_message_id、messages、tool_call_facts；evidence/gate、order_data 为当前可信事实。
        registry 保留生产与评估的共享调用接口，不用最新定义覆盖原调用的服务器快照。
    返回：应跳过正文模型时返回固定字符串，其余情况返回 None 保留既有答复流程。
    异常：无主动业务异常；消息、绑定或结果无法核实时保守返回 None。

    不读取用户 JSON、自然语言、历史任意失败或旧轮结果；来源卸载、schema 或读写
    配置改变不撤销原快照。旧检查点缺快照或配对不完整时保守保留原正文流程。
    """
    gate = state.get("gate") or {}
    if state.get("order_data") or state.get("evidence") and gate.get("passed") is True:
        return None
    turn_id = state.get("context", {}).get("turn_id")
    facts = state.get("tool_call_facts")
    if not turn_id or not isinstance(facts, dict):
        return None
    try:
        current = [message for message in convert_to_messages(state.get("messages", []))
                   if message.additional_kwargs.get("turn_id") == turn_id]
    except (ValueError, TypeError, NotImplementedError):
        return None
    anchors = [index for index, message in enumerate(current)
               if message.id == state.get("current_message_id") and isinstance(message, HumanMessage)
               and message.content == state.get("raw_query")]
    if len(anchors) != 1:
        return None

    calls, completed, has_fault = {}, set(), False
    for message in current[anchors[0] + 1:]:
        if isinstance(message, HumanMessage):
            # 原轮只能有一个真实用户入口，不能跨过另一条用户消息拼接工具配对。
            return None
        if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls):
            if message.invalid_tool_calls or not valid_tool_calls(message.tool_calls):
                return None
            bindings = message.additional_kwargs.get("tool_bindings", {})
            for call in message.tool_calls:
                if call["id"] in calls:
                    return None
                try:
                    binding = binding_from_dict(bindings[call["id"]])
                    fact = facts[call["id"]]
                    original = binding_from_dict(fact["binding"])
                except (KeyError, TypeError, ValueError):
                    return None
                if (set(fact) != {"turn_id", "binding", "access"} or fact["turn_id"] != turn_id
                        or not isinstance(fact["access"], str) or fact["access"] not in {"read", "write"}
                        or binding != original
                        or call["name"] != binding.identity.name):
                    return None
                calls[call["id"]] = fact["access"]
        elif isinstance(message, ToolMessage):
            identifier = message.tool_call_id
            if identifier not in calls or identifier in completed:
                return None
            result = _strict_result(message.content)
            if result is None:
                return None
            completed.add(identifier)
            if calls[identifier] == "read":
                # 空列表也是已完成的查询；同轮混合成功/失败保留成功事实，不整段覆盖。
                if result.ok:
                    return None
                # 编排层为剩余申请补齐的 TOOL_SKIPPED 只证明未执行，不是一次查询故障。
                has_fault = has_fault or (result.error.category == "fault" and result.error.code != "TOOL_SKIPPED")
            # 成功建单等写结果不表示读查询成功；真实票号由正文节点另行保留。
    return QUERY_FAULT_ANSWER if has_fault and completed == calls.keys() else None
