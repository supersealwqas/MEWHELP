"""订单政策阶段的来源隔离：构造已确认目标、检索上下文及确定性的订单事实展示。"""

import json

from app.workflow.state import ConversationState


async def verified_business_prefix(state: ConversationState, repository) -> str:
    """仅由仓储核实本轮冻结调用和真实结果生成订单/工单前缀。

    参数：state服务端原轮及tool_call_facts，repository聊天仓储。
    返回：可信前缀，无可核实业务结果为空；异常：身份、数据库或结果一致性错误。
    """
    return await repository.verified_business_prefix(state)


def optional_probe_miss(state: ConversationState) -> bool:
    """识别确实进入无业务对象补检索后的未命中，保留纯业务澄清而不认定政策缺口。

    参数：state服务端原轮；返回：严格窄场景布尔，旧checkpoint缺标记为False；异常：无。
    """
    from app.workflow.routing import _active_order_tokens, _trusted_history
    return bool(state.get("optional_knowledge_probe") is True and state.get("route") == "business"
        and not state.get("route_ready") and not state.get("tool_count") and not state.get("order_id")
        and not state.get("pending_ticket") and not _active_order_tokens(state["raw_query"])
        and not _trusted_history(state["history_turns"])[1]
        and not any(message.get("tool_calls") or message.get("role") == "tool"
                    for turn in state["history_turns"] for message in turn.get("messages", []))
        and state.get("gate") is not None and state["gate"].get("passed") is False)


def knowledge_retrieval_fault(state: ConversationState) -> bool:
    """识别未恢复的知识处理责任（含预算），不让历史故障永久阻断同请求成功重试。

    参数：state服务端责任字典及快照；返回：仍有未恢复故障时为True；异常：无。
    """
    from app.tools.knowledge import unresolved_knowledge_faults
    return bool(unresolved_knowledge_faults(state))


def knowledge_failure_reason(state: ConversationState) -> str | None:
    """从未恢复责任保留预算停止语义，不把完整知识处理失败视为资料缺口。

    参数：state原轮责任字典；返回：预算/依赖停止原因或None；异常：无。
    """
    from app.tools.knowledge import unresolved_knowledge_faults
    faults = unresolved_knowledge_faults(state)
    if "budget_exceeded" in faults.values():
        return "budget_exceeded"
    return "retrieval_unavailable" if faults else None


def policy_query(state: ConversationState) -> str:
    """为检索绑定已核实对象，保留用户原始条件。

    参数：state含成功订单事实、完整问题及意图；返回：不截断的检索种子；异常：必需订单字段缺失时KeyError。
    """
    order = state["order_data"]
    # 编号先作为明确的订单类型传递，避免与商品型号混淆；用户条件保留以供多角度检索。
    target = f"订单{state['order_id']}，商品：{order.get('product', '')}，状态：{order.get('status', '')}"
    return f"当前已确认对象：{target}。{state['intent']}需求：{state['resolved_query']}"


def verified_order_text(state: ConversationState) -> str:
    """只使用成功工具字段生成事实区，不从原话提取金额或签收日期。

    参数：state已核实订单；返回：Markdown事实区及说明标题；异常：必需状态缺失时KeyError。
    """
    order = state["order_data"] or {}
    fields = (("order_id", "订单"), ("product", "商品"), ("quantity", "数量"),
              ("total_amount", "商品总金额"), ("status", "状态"), ("signed_at", "签收时间"))
    lines = ["**已核实订单信息**"]
    for key, label in fields:
        value = order.get(key)
        if value is not None and value != "":
            lines.append(f"- {label}：{value}" + (f" {order.get('currency', 'CNY')}" if key == "total_amount" else ""))
    if not order.get("signed_at"):
        lines.append("- 签收时间：暂未查到")
    if state["intent"] == "退款退货":
        # 此阶段只生成建议，尚未执行申请；后续提交结果由独立表单回执展示。
        lines.append("- 退款申请：尚未提交（本条答复生成时）；提交结果以表单回执为准")
    return "\n".join(lines) + "\n\n**政策与办理说明**\n\n"


def policy_facts(state: ConversationState) -> dict:
    """将用户陈述、已确认对象、工具事实与表单职责分别传给政策解释阶段。

    参数：state含订单事实、选单状态和本轮证据；返回：动态事实字典；异常：必需字段缺失时KeyError。
    """
    selected = state.get("selection_action") or {}
    facts = {"用户陈述（未经核实）": state["raw_query"], "完整问题": state["resolved_query"],
             "已确认对象": {"订单号": state["order_id"], "来源": "用户选单" if selected.get("status") == "accepted" else "本轮订单查询"},
             "订单事实": state["order_data"], "政策证据": state["evidence"],
             "办理进度": {"当前阶段": "尚未提交" if state["intent"] == "退款退货" else "仅查询信息",
                          "执行回执": None, "展示方式": "由页面表单及服务端回执展示"},
             "后续表单收集": ["退款原因"] if state["intent"] == "退款退货" else []}
    return facts
