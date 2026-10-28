"""客服会话历史内存存储与 Token 裁剪模块。

提供旧版纯内存历史与裁剪接口，以及持久化工具轮次恢复、整轮裁剪和完整输入预算。
新预算按 ASCII 约四字符一单位、非 ASCII 约每字符两单位估算，不代表模型真实分词。
"""

import json
from collections import Counter

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from app.core.context_budget import ContextBudgetExceeded, TokenEstimator
from app.schemas.history import StoredMessage, StoredTurn
from app.schemas.tools import ToolResult
from app.tools.contracts import ResultPolicy
from app.tools.presentation import model_tool_result


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


def trim_history(messages: list[BaseMessage], max_tokens: int = 12000) -> list[BaseMessage]:
    """基于 Token 预算控制的历史消息裁剪函数。

    使用 LangChain 官方 `trim_messages` 工具实现，确保对话历史在超出预算时
    丢弃较早轮次，并始终完整保留 SystemMessage，截断后对话首条消息保持为 HumanMessage。

    参数:
        messages (list[BaseMessage]): 待裁剪的消息列表
        max_tokens (int): 允许保留的最大 Token 上限，默认为 12000

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


def valid_tool_calls(calls: list[dict]) -> bool:
    """校验可回灌的申请协议。参数：申请列表；返回：是否完整且 ID 唯一、无首尾空白；异常：无。"""
    ids = []
    for call in calls:
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier or identifier != identifier.strip() or len(identifier) > 64:
            return False
        if not isinstance(call.get("name"), str) or not call["name"].strip() or not isinstance(call.get("args"), dict):
            return False
        ids.append(identifier)
    return len(ids) == len(set(ids))


def _message_identity(row: StoredMessage, turn: StoredTurn) -> dict:
    """附加可信消息身份。参数：row原始消息、turn所属轮；返回：消息构造参数；异常：无。"""
    if row.id is None:
        return {"additional_kwargs": {"turn_id": turn.turn_id}}
    return {"id": f"db:{row.id}", "additional_kwargs": {"db_message_id": row.id, "turn_id": turn.turn_id}}


def assemble_turns(turns: list[StoredTurn]) -> list[list[BaseMessage]]:
    """按申请区段恢复各轮已提交事实，保留中断批次中的可信结果。

    参数：turns 按时间排列的存储轮，其消息已按主键排序。
    返回：逐轮消息列表；仅回灌唯一配对的调用与结果，不改写原始审计记录。
    异常：无主动抛出的业务异常。
    """
    history = []
    for turn in turns:
        if turn.turn_status == "running":
            continue
        group = []
        seen_ids: set[str] = set()
        for index, row in enumerate(turn.messages):
            if row.role == "user":
                group.append(HumanMessage(content=row.content, **_message_identity(row, turn)))
            elif row.role == "assistant" and row.tool_calls:
                calls = row.tool_calls
                if not valid_tool_calls(calls):
                    continue
                call_ids = {call["id"] for call in calls}
                reused_ids = seen_ids.intersection(call_ids)
                seen_ids.update(call_ids)
                # 同轮 ID 不能再次指代另一申请，即使首次申请尚无结果也不能借用后续结果。
                if reused_ids:
                    continue
                results = []
                for result_row in turn.messages[index + 1:]:
                    # 新的助手或用户消息结束当前申请区段，禁止跨组搜索同 ID 的结果。
                    if result_row.role != "tool":
                        break
                    if result_row.tool_call_id in call_ids:
                        results.append(result_row)
                result_counts = Counter(result.tool_call_id for result in results)
                # 只丢弃缺失或重复结果的调用，已唯一配对的工单事实必须保留。
                paired_calls = [call for call in calls if result_counts[call["id"]] == 1]
                if paired_calls:
                    public_calls = [{key: call[key] for key in ("id", "name", "args")} for call in paired_calls]
                    identity = _message_identity(row, turn)
                    bindings = {call["id"]: call["binding"] for call in paired_calls if call.get("binding")}
                    if bindings:
                        identity["additional_kwargs"]["tool_bindings"] = bindings
                    group.append(AIMessage(content=row.content, tool_calls=public_calls, **identity))
                    by_id = {m.tool_call_id: m for m in results}
                    group.extend(ToolMessage(content=by_id[c["id"]].content, tool_call_id=c["id"],
                                             **_message_identity(by_id[c["id"]], turn)) for c in paired_calls)
            elif row.role == "assistant" and row.content and turn.turn_status == "completed":
                group.append(AIMessage(content=row.content, **_message_identity(row, turn)))
        if group and turn.turn_status in {"cancelled", "failed"}:
            group.append(AIMessage(content="上一轮回答未完成；以上工具结果是已保存的事实，请勿重复提交已创建的工单。"))
        if group:
            history.append(group)
    return history


def trim_turns(history: list[list[BaseMessage]], current: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """按整轮保留最近历史。

    参数：history 按时间排列的历史轮；current 不可裁剪的当前消息；max_tokens 对话估算预算。
    返回：list[BaseMessage]，保留最近连续完整轮及当前消息的平铺列表。
    异常：ContextBudgetExceeded 表示当前问题或工具事实本身已超预算。
    """
    used = _estimate_message_tokens(current)
    if used > max_tokens:
        raise ContextBudgetExceeded("当前消息或工具结果过长，请缩短问题后重试")
    kept = list(current)
    for turn in reversed(history):
        turn_tokens = _estimate_message_tokens(turn)
        # 当前轮不可拆分；旧轮也必须连同工具申请和结果一起保留或舍弃。
        if used + turn_tokens > max_tokens:
            break
        kept = [*turn, *kept]
        used += turn_tokens
    return kept


def _estimate_token_units(value: object) -> int:
    """用真实 Unicode 序列化估算文本与协议字段的预算。

    参数：value 可 JSON 序列化的消息字段或工具定义。
    返回：int，ASCII 四字符约一单位、非 ASCII 每字符约两单位，向上取整。
    异常：TypeError 表示输入含不可 JSON 序列化的数据。
    """
    return TokenEstimator().value_tokens(value)


def _estimate_message_tokens(messages: list[BaseMessage]) -> int:
    """估算对话正文与会影响请求长度的消息协议字段。

    参数：messages 当前轮或完整历史轮的 LangChain 消息列表。
    返回：int，逐条累加的近似输入单位数，不是模型真实 Token 数。
    异常：TypeError 表示消息正文或工具参数不能 JSON 序列化。
    """
    return TokenEstimator().message_tokens(messages)


def _without_tool_metadata(value):
    """递归移除内部来源与知识声明。参数：原始JSON值；返回：独立公开副本；异常：无。"""
    if isinstance(value, list):
        return [_without_tool_metadata(item) for item in value]
    if isinstance(value, dict):
        hidden = {"verified_knowledge", "tool_bindings", "bindings", "is_mock", "mcp_server", "tool_source",
                  "source_key", "source_revision"}
        return {key: _without_tool_metadata(item) for key, item in value.items() if key not in hidden}
    return value


def _model_message(message: BaseMessage, *, policy: ResultPolicy | None = None,
                   max_tokens: int | None = None, estimator: TokenEstimator | None = None) -> BaseMessage:
    """为模型生成工具事实视图，隔离后台来源标记，不改写原消息。

    参数：message当前或历史消息；policy可选本地结果规则；max_tokens可选正文预算；estimator统一估算器。
    返回：BaseMessage，结果副本保留调用配对；默认移除内部来源及知识字段，显式规则或预算启用统一投影。
    异常：ContextBudgetExceeded，核心事实超预算；兼容历史非JSON或非统一结果正文。
    """
    if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
        return message
    try:
        payload = json.loads(message.content)
    except (ValueError, TypeError):
        return message
    if not isinstance(payload, dict):
        return message
    payload = _without_tool_metadata(payload)
    if (policy is not None or max_tokens is not None) and type(payload.get("ok")) is bool:
        estimator = estimator or TokenEstimator()
        result = ToolResult.model_validate(payload)
        payload = model_tool_result(result, policy or ResultPolicy(),
            max_tokens=max_tokens if max_tokens is not None else max(1, estimator.value_tokens(payload) * 4),
            estimator=estimator)
        return message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))})
    # 仅改模型副本，完整数据库回执与内部证据继续保留。
    return message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))})


def prepare_messages(
    history: list[list[BaseMessage]],
    current: list[BaseMessage],
    max_tokens: int,
    *,
    system_text: str,
    tools: list[BaseTool | dict] | None = None,
    request_max_tokens: int = 16000,
) -> list[BaseMessage]:
    """按对话和完整输入两层预算组装模型消息，完整保留当前问题及工具事实。

    参数：history 按时间排列的完整旧轮；current 不可裁剪当前轮；max_tokens 对话估算预算；
        system_text 本阶段系统提示；tools 仅决策阶段传入的工具或 OpenAI 工具定义；
        request_max_tokens 包含系统、对话和工具定义的完整输入估算预算，默认 16000，不含输出。
    返回：list[BaseMessage]，系统消息在首位，其后为不含后台来源标记的完整对话。
    异常：ContextBudgetExceeded 表示固定提示、工具定义与当前消息无法共同放入预算；
        TypeError、ValueError 表示工具定义或消息内容无法转换为合法 JSON 数据。
    """
    system = SystemMessage(content=system_text)
    fixed_tokens = _estimate_message_tokens([system])
    if tools:
        fixed_tokens += _estimate_token_units([convert_to_openai_tool(tool) for tool in tools])
    # 固定说明只扣完整输入预算，不能挤占独立的 12000 单位对话预算。
    available = min(max_tokens, request_max_tokens - fixed_tokens)
    # 两个阶段共用此出口；先投影当前及历史事实，再按模型实际可见内容核算预算。
    model_history = [[_model_message(message) for message in turn] for turn in history]
    model_current = [_model_message(message) for message in current]
    messages = trim_turns(model_history, model_current, available)
    return [system, *messages]
