"""订单政策阶段的来源隔离：构造已确认目标、检索上下文及确定性的订单事实展示。"""

import json

from app.workflow.state import ConversationState


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


def policy_answer_messages(state: ConversationState) -> list[dict]:
    """将用户陈述、已确认对象、工具事实与表单职责分别传给政策解释阶段。

    参数：state含订单事实、选单状态和本轮证据；返回：模型消息列表；异常：数据不能JSON序列化时TypeError。
    """
    selected = state.get("selection_action") or {}
    facts = {"用户陈述（未经核实）": state["raw_query"], "完整问题": state["resolved_query"],
             "已确认对象": {"订单号": state["order_id"], "来源": "用户选单" if selected.get("status") == "accepted" else "本轮订单查询"},
             "订单事实": state["order_data"], "政策证据": state["evidence"],
             "办理进度": {"当前阶段": "尚未提交" if state["intent"] == "退款退货" else "仅查询信息",
                          "执行回执": None, "展示方式": "由页面表单及服务端回执展示"},
             "后续表单收集": ["退款原因"] if state["intent"] == "退款退货" else []}
    # 这是各业务通用的来源/职责契约，不包含具体题目、商品、渠道或政策答案。
    rules = ("本阶段只输出政策与办理说明，程序会单独展示已核实订单信息；不要重复订单事实或重新列金额。"
             "已确认对象是当前唯一处理对象，即使用户原话记录了此前的犹豫，也不再让用户确认或选择该对象。"
             "用户陈述仅是用户提供的条件，不能提升为已核实事实；引用时注明来源。"
             "后续表单收集的字段不在正文追问，也不列为本轮待补信息。"
             "分别说明证据支持的条件和未覆盖的部分，不用相近规则替代用户实际所问；缺证据时仍给出可继续办理的说明。"
             "created_at是创建时间，不是signed_at签收时间；缺少签收时间就说明未知。"
             "办理进度由程序及执行回执展示，本阶段不叙述操作已提交、已保存或已执行；提供建议不等于执行操作。"
             "只给政策规则标注本轮引用，不为订单事实加引用。\n")
    return [{"role": "user", "content": state["raw_query"]},
            {"role": "system", "content": rules + json.dumps(facts, ensure_ascii=False)}]
