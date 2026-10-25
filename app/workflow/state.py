"""可持久化的共享工作流State及新轮初始化，隔离历史事实和本轮运行字段。"""

from copy import deepcopy
from typing import Literal, TypedDict

from app.core.model_budget import BudgetLedger
from app.schemas.tools import ToolContext
from app.schemas.workflow import ActionKind, Intent


def checkpoint_thread_id(conversation_id: int) -> str:
    """由可信会话主键派生CH05持久化键。参数：conversation_id经仓储核验归属的MySQL主键；返回：命名空间线程键；异常：无。"""
    return f"mewhelp:ch05:{conversation_id}"


class ConversationState(TypedDict, total=False):
    """根图与Agent共用的纯数据；字段整体替换，不使用累加reducer或保存运行依赖。"""

    context: dict  # ToolContext的JSON快照，身份只由服务端提供。
    raw_query: str  # 当前用户完整原话，保留首尾空白。
    resolved_query: str  # 指代节点完成后的透传原话，未执行时为空。
    history_turns: list[dict]  # 既有StoredTurn字典列表，按原格式和历史预算使用。
    audit_cursor: int  # 已对账消息主键，零表示尚未导入历史。
    messages: list[dict]  # 本轮模型与工具消息的可序列化字典列表。
    intent: Intent | None  # 当前轮的严格七类意图，分类前为空。
    route: Literal["knowledge", "business", "complaint", "chitchat"] | None  # 固定四出口。
    evidence: list[dict]  # 本轮完整EvidenceCitation字典，编号保持稳定。
    evidence_scores: list[float]  # 本轮全部归一化重排分，与权威证据同源。
    gate: dict | None  # GateDecision的JSON快照，未执行证据闸时为空。
    pending_calls: list[dict]  # 本轮下一批待执行的只读工具申请。
    seen_call_ids: list[str]  # 已见工具调用ID，用于跨决策查重。
    seen_call_signatures: list[str]  # 已见工具名称和参数签名，用于拒绝重复申请。
    tool_count: int  # 本轮逻辑工具调用次数。
    decision_count: int  # 本轮Agent决策次数。
    budget: dict  # 每轮独占的BudgetLedger快照，各节点结束后整体替换。
    suggested_kinds: list[ActionKind]  # 待服务端构造快照的建议类型，不执行动作。
    suggested_actions: list[dict]  # 服务端构造的稳定建议快照，未生成时为空。
    answer: str  # 本轮完整用户可见正文。
    stop_reason: str | None  # 本轮结束原因，尚未结束时为空。
    completion_metadata: dict  # 本轮最终审计与界面元数据。


def new_turn_state(context: ToolContext, history_turns: list[dict], audit_cursor: int) -> ConversationState:
    """开始新轮并覆盖全部运行字段，不恢复旧工具申请。

    参数：context为可信身份及1–4000字非空原话；history_turns为已对账历史字典；audit_cursor为非负整数。
    返回：ConversationState，历史深拷贝、原话完整保留，使用独占的默认预算；入口可再按配置替换新账本。
    异常：ValueError，原话为空、超过4000字或游标非法；不调用模型、检索器或仓储。
    """
    query = context.user_message
    if not query.strip() or len(query) > 4000:
        raise ValueError("当前原话必须为1–4000字符的非空文本")
    if type(audit_cursor) is not int or audit_cursor < 0:
        raise ValueError("历史审计游标必须为非负整数")
    # 新input必须显式覆盖每个字段，防止同thread的旧checkpoint残留运行数据。
    return {
        "context": context.model_dump(mode="json"), "raw_query": query, "resolved_query": "",
        "history_turns": deepcopy(history_turns), "audit_cursor": audit_cursor, "messages": [],
        "intent": None, "route": None, "evidence": [], "evidence_scores": [], "gate": None,
        "pending_calls": [], "seen_call_ids": [], "seen_call_signatures": [],
        "tool_count": 0, "decision_count": 0, "budget": BudgetLedger().snapshot(),
        "suggested_kinds": [], "suggested_actions": [], "answer": "", "stop_reason": None,
        "completion_metadata": {},
    }
