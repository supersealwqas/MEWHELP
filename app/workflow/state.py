"""可持久化的共享工作流State及新轮初始化，隔离历史事实和本轮运行字段。"""

from copy import deepcopy
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, BaseMessage, convert_to_messages
from langgraph.graph import add_messages

from app.core.model_budget import BudgetLedger
from app.schemas.tools import ToolContext
from app.schemas.workflow import ActionKind, Intent


def checkpoint_thread_id(conversation_id: int) -> str:
    """由可信会话主键派生CH05持久化键。参数：conversation_id经仓储核验归属的MySQL主键；返回：命名空间线程键；异常：无。"""
    return f"mewhelp:ch05:{conversation_id}"


class ConversationState(TypedDict, total=False):
    """根图与Agent共用状态；完整messages按ID合并，其余控制字段每轮覆盖。"""

    context: dict  # ToolContext的JSON快照，身份只由服务端提供。
    raw_query: str  # 当前用户完整原话，保留首尾空白。
    resolved_query: str  # 一次理解补全的完整问题，未执行时为空。
    standard_query: str  # 同次理解的标准问法；优化失败保留完整原句。
    order_id: str | None  # 可信输入中的完整订单候选，执行前仍须校验本人归属。
    order_data: dict | None  # 本轮成功核实的订单事实，未查询时为空。
    expanded_queries: list[str]  # 本轮政策检索的一条或三条扩写查询。
    selection_action: dict | None  # 当前原轮的服务端订单选择动作，未等待时为空。
    history_turns: list[dict]  # 既有StoredTurn字典列表，按原格式和历史预算使用。
    audit_cursor: int  # 已对账消息主键，零表示尚未导入历史。
    messages: Annotated[list[AnyMessage], add_messages]  # 全会话原始消息；裁剪只发生在模型副本。
    context_version: int  # 完整历史格式版本，CH07为7。
    current_message_id: str  # 当前用户原话稳定身份，不能由最后一条human推断。
    history_context: dict | None  # 每轮冻结的三层投影，模型调用时另行组装。
    context_notes: list[str]  # 当前动态资料，绝不写入完整messages。
    intent: Intent | None  # 当前轮的严格八类意图，分类前为空。
    intent_confidence: float | None  # 分类自报有限分数，仅用于记录与评估。
    route: Literal["knowledge", "business", "complaint", "chitchat", "order_policy"] | None  # 原四出口加订单政策出口。
    evidence: list[dict]  # 本轮完整EvidenceCitation字典，编号保持稳定。
    evidence_scores: list[float]  # 本轮全部归一化重排分，与权威证据同源。
    gate: dict | None  # GateDecision的JSON快照，未执行证据闸时为空。
    retrieval_attempted: bool  # 本轮是否已经尝试知识检索，防止分类补救重复检索。
    degradations: list[dict]  # 本轮外部故障的阶段和类型，不含错误原文。
    pending_calls: list[dict]  # 本轮下一批待执行的只读工具申请。
    seen_call_ids: list[str]  # 已见工具调用ID，用于跨决策查重。
    seen_call_signatures: list[str]  # 已见工具名称和参数签名，用于拒绝重复申请。
    tool_count: int  # 本轮逻辑工具调用次数。
    decision_count: int  # 本轮Agent决策次数。
    budget: dict  # 每轮独占的BudgetLedger快照，各节点结束后整体替换。
    suggested_kinds: list[ActionKind]  # 待服务端构造快照的建议类型，不执行动作。
    suggested_actions: list[dict]  # 服务端构造的稳定建议快照，未生成时为空。
    refund_action: dict | None  # 独立本地退款建议，未经核实或新轮时为空。
    answer: str  # 本轮完整用户可见正文。
    stop_reason: str | None  # 本轮结束原因，尚未结束时为空。
    completion_metadata: dict  # 本轮最终审计与界面元数据。


def message_row(turn_id: str, role: str, content: str, *, calls: list[dict] | None = None,
                call_id: str | None = None, db_id: int | None = None) -> dict:
    """生成跨落库稳定的消息身份。

    参数：turn_id轮次、role角色、content原文、calls申请、call_id结果键、db_id可选持久化主键。
    返回：可供add_messages转换的字典；异常：无，坏申请保存在审计元数据中而非可执行协议。
    """
    from app.core.memory import valid_tool_calls
    valid = bool(calls) and valid_tool_calls(calls)
    key = "user" if role == "user" else f"result:{call_id}" if role == "tool" else (
        f"calls:{calls[0]['id']}" if valid else f"audit:{db_id}" if calls else "answer")
    metadata = {"turn_id": turn_id}
    if db_id is not None:
        metadata["db_message_id"] = db_id
    row = {"id": f"turn:{turn_id}:{key}", "role": role, "content": content, "additional_kwargs": metadata}
    if valid:
        row["tool_calls"] = deepcopy(calls)
    elif calls:
        metadata["raw_tool_calls"] = deepcopy(calls)
    if role == "tool":
        row["tool_call_id"] = call_id or f"missing:{db_id}"
    return row


def full_history_messages(turns: list[dict]) -> list[dict]:
    """转换完整原始流水，不进行配对过滤或文本压缩。

    参数：turns含真实主键的StoredTurn字典，旧测试允许空主键。
    返回：所有原文消息的独立字典；异常：KeyError表示原始契约损坏。
    """
    return [message_row(turn["turn_id"], row["role"], row["content"], calls=row.get("tool_calls"),
                        call_id=row.get("tool_call_id"), db_id=row.get("id"))
            for turn in turns for row in turn["messages"]]


def current_messages(state: ConversationState) -> list[BaseMessage]:
    """从完整历史提取本轮或显式模型副本。参数：state当前状态；返回：独立消息；异常：消息转换错误。"""
    turn_id = state["context"]["turn_id"]
    return [message.model_copy(deep=True) for message in convert_to_messages(state["messages"])
            if message.additional_kwargs.get("turn_id", turn_id) == turn_id]


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
        "context": context.model_dump(mode="json"), "raw_query": query, "resolved_query": "", "standard_query": "",
        "order_id": None, "order_data": None, "expanded_queries": [], "selection_action": None,
        "history_turns": deepcopy(history_turns), "audit_cursor": audit_cursor,
        "messages": [*full_history_messages(history_turns), message_row(context.turn_id, "user", query)],
        "context_version": 7, "current_message_id": f"turn:{context.turn_id}:user",
        "history_context": None, "context_notes": [],
        "intent": None, "intent_confidence": None, "route": None, "evidence": [], "evidence_scores": [], "gate": None,
        "retrieval_attempted": False, "degradations": [],
        "pending_calls": [], "seen_call_ids": [], "seen_call_signatures": [],
        "tool_count": 0, "decision_count": 0, "budget": BudgetLedger().snapshot(),
        "suggested_kinds": [], "suggested_actions": [], "refund_action": None, "answer": "", "stop_reason": None,
        "completion_metadata": {},
    }
