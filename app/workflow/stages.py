"""工作流阶段事件白名单；仅发布用户可理解的状态，不泄露模型内部诊断。"""

from langgraph.config import get_stream_writer

LABELS = {"understand": "理解问题", "classify": "判断需求", "order": "核实订单",
          "expand": "准备政策查询", "policy": "查阅政策", "answer": "生成回答"}
FIELDS = {"understand": {"resolved_query", "standard_query"}, "classify": {"intent", "confidence"},
          "order": {"order_id"}, "expand": {"queries"}, "policy": set(), "answer": set()}


def publish_stage(state: dict, stage: str, status: str, **details) -> None:
    """发布安全阶段快照。参数：state可信上下文、stage固定阶段、status运行/完成、details白名单详情；返回：None；异常：非法阶段或流故障。"""
    if stage not in LABELS or status not in {"running", "completed"}:
        raise ValueError("未知工作流阶段")
    payload = {"source_turn_id": state["context"]["turn_id"], "stage": stage, "status": status, "label": LABELS[stage]}
    if status == "completed":
        payload.update({key: value for key, value in details.items() if key in FIELDS[stage]})
    get_stream_writer()({"event": "workflow_stage", "data": payload})
