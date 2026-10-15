"""客服会话历史内存存储与 Token 裁剪模块。

提供客服服务端纯内存会话历史管理器 (SessionStore) 与
基于 LangChain trim_messages 封装的上下文预算裁剪函数 (trim_history)。
"""

from langchain_core.messages import BaseMessage, trim_messages


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
