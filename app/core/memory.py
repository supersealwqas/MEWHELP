"""客服会话历史内存存储与 Token 裁剪模块。

提供客服服务端纯内存会话历史管理器 (SessionStore) 与
基于 LangChain trim_messages 封装的上下文预算裁剪函数 (trim_history)。
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from app.schemas.history import StoredTurn


class SessionStore:
    """客服会话历史纯内存存储管理器。

    设计意图：
        提供快速、无外部数据库依赖的多会话历史管理能力，支持不同 session_id
        间的历史隔离、多条消息追加、历史读取与会话清理。
    """

    def __init__(self) -> None:
        """初始化内存会话存储字典。

        参数:
            无。

        返回值:
            None: 构造函数无返回值。

        异常:
            无业务异常抛出。
        """
        # 使用字典保存每个 session_id 对应的消息列表
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        """获取指定会话的历史消息列表副本。

        参数:
            session_id (str): 会话唯一标识符

        返回值:
            list[BaseMessage]: 历史消息列表的浅拷贝副本；若会话不存在则返回空列表

        异常:
            无业务异常抛出
        """
        # 返回列表浅拷贝，避免外部直接修改污染内部存储
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        """向指定会话按顺序追加一条或多条消息。

        参数:
            session_id (str): 会话唯一标识符
            *messages (BaseMessage): 待追加的用户或客服消息，可为空。

        返回值:
            None: 无返回值

        异常:
            无业务异常抛出
        """
        # 若会话尚不存在，先初始化空列表，再按传入顺序写入消息。
        self._sessions.setdefault(session_id, []).extend(messages)

    def clear(self, session_id: str) -> None:
        """清理指定会话的全部历史记录。

        参数:
            session_id (str): 会话唯一标识符

        返回值:
            None: 无返回值

        异常:
            无业务异常抛出
        """
        # 安全移除目标会话，若不存在则静默忽略
        self._sessions.pop(session_id, None)


def trim_history(messages: list[BaseMessage], max_tokens: int = 2000) -> list[BaseMessage]:
    """基于 Token 预算控制的历史消息裁剪函数。

    使用 LangChain 官方 `trim_messages` 工具实现，确保对话历史在超出预算时
    丢弃较早轮次，并始终完整保留 SystemMessage，截断后对话首条消息保持为 HumanMessage。

    参数:
        messages (list[BaseMessage]): 待裁剪的消息列表
        max_tokens (int): 允许保留的最大 Token 上限，默认为 2000

    返回值:
        list[BaseMessage]: 裁剪后的消息列表；若输入为空则返回空列表

    异常:
        无业务异常抛出
    """
    # 边界防御：空列表直接返回
    if not messages:
        return []

    # 调用 LangChain trim_messages 执行裁剪
    # strategy="last": 从尾部向头部保留最新消息
    # token_counter="approximate": 采用快速近似 token 估算，无外部网络模型依赖
    # start_on="human": 截断后除 SystemMessage 之外首条对话消息保证为 HumanMessage
    # include_system=True: 始终完整保留 SystemMessage 系统设定
    # allow_partial=False: 不截断单条消息的中间内容，保持语义完整
    trimmed: list[BaseMessage] = trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter="approximate",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    return trimmed


class ContextBudgetExceeded(ValueError):
    """当前用户问题与工具事实无法完整放入上下文预算。"""


def valid_tool_calls(calls: list[dict]) -> bool:
    """校验可回灌的申请协议。参数：申请列表；返回：是否完整且 ID 唯一；异常：无。"""
    ids = []
    for call in calls:
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 64:
            return False
        if not isinstance(call.get("name"), str) or not call["name"].strip() or not isinstance(call.get("args"), dict):
            return False
        ids.append(identifier)
    return len(ids) == len(set(ids))


def assemble_turns(turns: list[StoredTurn]) -> list[list[BaseMessage]]:
    """恢复完整轮及已提交事实。参数：按时间排列的存储轮；返回：逐轮消息；异常：无业务异常。"""
    history = []
    for turn in turns:
        if turn.turn_status == "running":
            continue
        group = []
        for row in turn.messages:
            if row.role == "user":
                group.append(HumanMessage(content=row.content))
            elif row.role == "assistant" and row.tool_calls:
                calls = row.tool_calls
                results = [m for m in turn.messages if m.role == "tool" and m.tool_call_id in {c.get("id") for c in calls}]
                # 任一结果缺失、重复或申请损坏时整组跳过，原始记录仍留数据库审计。
                if valid_tool_calls(calls) and len(results) == len(calls) and len({m.tool_call_id for m in results}) == len(calls):
                    group.append(AIMessage(content="", tool_calls=calls))
                    by_id = {m.tool_call_id: m for m in results}
                    group.extend(ToolMessage(content=by_id[c["id"]].content, tool_call_id=c["id"]) for c in calls)
            elif row.role == "assistant" and row.content and turn.turn_status == "completed":
                group.append(AIMessage(content=row.content))
        if group and turn.turn_status in {"cancelled", "failed"}:
            group.append(AIMessage(content="上一轮回答未完成；以上工具结果是已保存的事实，请勿重复提交已创建的工单。"))
        if group:
            history.append(group)
    return history


def trim_turns(history: list[list[BaseMessage]], current: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """按整轮保留最近历史。参数：历史轮、不可裁剪当前轮和预算；返回：平铺消息；异常：ContextBudgetExceeded。"""
    if count_tokens_approximately(current) > max_tokens:
        raise ContextBudgetExceeded("当前消息或工具结果过长，请缩短问题后重试")
    kept = list(current)
    for turn in reversed(history):
        candidate = [*turn, *kept]
        if count_tokens_approximately(candidate) > max_tokens:
            break
        kept = candidate
    return kept
