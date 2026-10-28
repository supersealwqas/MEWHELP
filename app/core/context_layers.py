"""按消息锚点生成三层历史的只读模型视图。

本模块只建议边界，不写数据库；临时省略半压旧轮不会伪造摘要完成。
原始消息始终完整保存，工具配对与完整轮边界在官方trim_messages之外另行校验。
"""

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, trim_messages

from app.core.context_budget import ContextBudget, ContextBudgetExceeded, TokenEstimator
from app.core.memory import _model_message, assemble_turns, valid_tool_calls
from app.schemas.history import StoredMessage, StoredTurn


@dataclass(frozen=True)
class HistoryBlock:
    """消息区间相交的完整轮次，共用裁剪和摘要边界，不修改原始消息。"""

    turns: list[StoredTurn]  # 按首条消息ID排列的原始轮。
    messages: list[StoredMessage]  # 块内按真实主键排序的完整原文。

    @property
    def first_id(self) -> int:
        """读取共同起点。参数：无；返回：已校验的首条主键；异常：无。"""
        return self.messages[0].id

    @property
    def last_id(self) -> int:
        """读取共同末尾。参数：无；返回：已校验的末条主键；异常：无。"""
        return self.messages[-1].id


def history_blocks(turns: list[StoredTurn]) -> list[HistoryBlock]:
    """将延迟补写形成的相交区间合为不可拆分块。

    参数：turns完整历史，各轮内部消息须按真实主键升序，空轮忽略。
    返回：按时间排列的块，块内原文亦按真实主键排序。
    异常：ValueError表示缺主键、轮内错序或全局主键重复。
    """
    nonempty, seen = [], set()
    for turn in turns:
        ids = [row.id for row in turn.messages]
        if any(type(value) is not int or value <= 0 for value in ids) or ids != sorted(set(ids)) or seen.intersection(ids):
            raise ValueError("历史必须提供真实且轮内有序、全局不重复的消息主键ID")
        seen.update(ids)
        if ids:
            nonempty.append(turn)
    groups, last = [], 0
    for turn in sorted(nonempty, key=lambda item: item.messages[0].id):
        if not groups or turn.messages[0].id > last:
            groups.append([])
        # 延迟回执可能跨过多个轮，须按累计末尾合并传递相交，不能仅比较相邻轮。
        groups[-1].append(turn)
        last = max(last, turn.messages[-1].id)
    return [HistoryBlock(group, sorted([row for turn in group for row in turn.messages], key=lambda row: row.id))
            for group in groups]


@dataclass(frozen=True)
class ContextSnapshot:
    """一次对话共用的历史快照；正文均为独立副本，锚点须由仓储另行提交。"""

    summary_upto_msg_id: int  # 数据库已确认摘要覆盖上界，不随省略前移。
    previous_layer1_from_msg_id: int  # 本次加载的原文层旧边界。
    layer1_from_msg_id: int  # 建议提交的原文层新边界。
    original_from_msg_id: int  # 输入完整历史的首条主键。
    original_upto_msg_id: int  # 输入完整历史的末条主键。
    summary: str  # 可容纳的完整摘要段组成的有界投影。
    summary_seqs: list[int]  # 实际注入的段号，按时间正序。
    layer1: list[BaseMessage]  # 最近原文层。
    layer2: list[BaseMessage]  # 实际发送的半压层。
    budget: ContextBudget  # 快照对应预算。
    layer1_tokens: int  # 原文层实际用量。
    layer2_tokens: int  # 所有待摘要半压轮用量，省略前统计。
    sent_layer2_tokens: int  # 省略后实际发送量。
    omitted_ranges: list[tuple[int, int]]  # 临时未发送的完整轮区间。
    summary_required: bool  # 半压用量超预算时请求后台摘要。


def project_summaries(summaries: list[dict], max_tokens: int, estimator: TokenEstimator) -> tuple[str, list[int]]:
    """选取能容纳的最近完整摘要段。

    参数：summaries含seq/content的已提交段；max_tokens投影预算；estimator统一估算器。
    返回：正文与实际段号；异常：KeyError/TypeError/ValueError表示存储数据不合法。
    """
    chosen = []
    for segment in reversed(sorted(summaries, key=lambda row: row["seq"])):
        content = segment["content"]
        if not isinstance(content, str):
            raise ValueError("摘要正文必须是字符串")
        if not content:
            continue  # 显式无事实段仅留覆盖记录，不注入占位文字。
        candidate = [segment, *chosen]
        if estimator.value_tokens("\n".join(row["content"] for row in candidate)) > max_tokens:
            break
        chosen = candidate
    return "\n".join(row["content"] for row in chosen), [row["seq"] for row in chosen]


def _safe_to_demote(turn: StoredTurn, messages: list[BaseMessage]) -> bool:
    """判断轮次能否跨入摘要区。参数：原始轮及已配对视图；返回：安全性；异常：无。"""
    if turn.receipt_pending or turn.turn_status not in {"completed", "failed", "cancelled"}:
        return False
    calls = [call for row in turn.messages for call in (row.tool_calls or [])]
    if not valid_tool_calls(calls):
        return False
    expected = [call["id"] for call in calls]
    actual = [message.tool_call_id for message in messages if isinstance(message, ToolMessage)]
    stored = [row.tool_call_id for row in turn.messages if row.role == "tool"]
    if any(not isinstance(identifier, str) or not identifier for identifier in stored):
        return False
    return sorted(expected) == sorted(actual) == sorted(stored)


def _block_view(block: HistoryBlock) -> tuple[list[BaseMessage], bool]:
    """按时间渲染整块，同时保持工具申请与唯一结果紧邻。

    参数：block已校验的完整相交轮组；返回：独立模型消息及整块是否可降级。
    异常：消息协议转换错误，不会改变数据库原文顺序。
    """
    units, safe = [], True
    for turn in block.turns:
        # 运行轮仍计入容量；任何未完成或未配对轮都会阻止整块降级。
        visible = turn if turn.turn_status != "running" else turn.model_copy(update={"turn_status": "awaiting_input"})
        groups = assemble_turns([visible])
        group = [_model_message(message).model_copy(deep=True) for message in (groups[0] if groups else [])]
        safe = _safe_to_demote(turn, group) and safe
        for index, message in enumerate(group):
            if isinstance(message, ToolMessage):
                # assemble_turns只输出唯一配对结果，工具结果与前面的申请作为协议整体排序。
                units[-1][2].append(message)
            else:
                # 合成的未完成说明跟在本轮末尾后；排序键不会写成持久化消息ID。
                key = message.additional_kwargs.get("db_message_id", turn.messages[-1].id)
                units.append((key, index, [message]))
    return [message for _, _, unit in sorted(units, key=lambda unit: unit[:2]) for message in unit], safe


def _tool_marker(message: ToolMessage, name: str) -> str:
    """生成有界单行标识。参数：message原始结果、name配对工具名；返回：说明；异常：无，坏JSON标记未知。"""
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    state = "成功" if payload.get("ok") is True else "失败" if payload.get("ok") is False else "状态未知"
    data = payload.get("data")
    identifiers = {}
    if isinstance(data, dict):
        for key, value in data.items():
            # 只显示源结果中的短标识，不从助手正文或参数猜测业务成功。
            if (key == "id" or key.endswith(("_id", "_no"))) and type(value) in (str, int) and len(str(value)) <= 64:
                identifiers[key] = value
                if len(identifiers) == 3:
                    break
    identity = json.dumps(identifiers, ensure_ascii=False, separators=(",", ":"))
    source = (f"原始消息ID={message.additional_kwargs['db_message_id']}" if message.additional_kwargs.get("db_message_id")
              else f"调用键={message.tool_call_id}")
    return f"[工具结果已省略；工具={name}；{state}；对象={identity}；{source}]"


def _shorten(messages: list[BaseMessage], estimator: TokenEstimator) -> list[BaseMessage]:
    """渲染半压副本。参数：已配对消息及计量器；返回：用户不改、助手前60字、大结果一行；异常：序列化错误。"""
    result = []
    names = {}
    for source in messages:
        message = source.model_copy(deep=True)
        if isinstance(message, AIMessage):
            names.update({call["id"]: call["name"] for call in message.tool_calls})
            if len(message.content) > 60:
                message.content = message.content[:60] + "…[已截短]"
        elif isinstance(message, ToolMessage) and estimator.message_tokens([message]) > 256:
            # 256是“大块结果”的渲染阈值，不替代窗口预算或用条数计量。
            message.content = _tool_marker(message, names.get(message.tool_call_id, "未知工具"))
        result.append(message)
    return result


def _suffix(groups: list[list[BaseMessage]], max_tokens: int, estimator: TokenEstimator) -> list[list[BaseMessage]]:
    """官方裁剪后对齐完整轮。参数：groups原顺序消息组、预算、计量器；返回：连续后缀；异常：消息协议错误。"""
    if not groups:
        return []
    messages = [message for group in groups for message in group]
    trimmed = trim_messages(messages, max_tokens=max_tokens, token_counter=estimator.message_tokens,
                            strategy="last", start_on="human", allow_partial=False, include_system=False)
    retained = {id(message) for message in trimmed}
    # 不依赖框架是否复制实例；稳定消息ID用于找回整轮起点。
    retained_ids = {message.id for message in trimmed if message.id is not None}
    for index, group in enumerate(groups):
        if group and (id(group[0]) in retained or group[0].id in retained_ids):
            return groups[index:]
    return []


def build_context_snapshot(
    turns: list[StoredTurn], *, summary_upto_msg_id: int, layer1_from_msg_id: int,
    summaries: list[dict], budget: ContextBudget, estimator: TokenEstimator,
) -> ContextSnapshot:
    """生成三层视图并建议降级边界，不修改原始记录。

    参数：turns轮内按消息ID有序的完整轮；两个锚点为非负真实主键；summaries已提交段；budget预算；estimator计量器。
    返回：ContextSnapshot，含未省略用量、实际发送消息和临时省略区间。
    异常：ValueError表示缺ID、错序、轮内锚点或不安全持久化边界；ContextBudgetExceeded表示未完成轮阻止安全降级。
    """
    if any(type(value) is not int or value < 0 for value in (summary_upto_msg_id, layer1_from_msg_id)):
        raise ValueError("上下文锚点必须是非负消息ID")
    if summary_upto_msg_id > layer1_from_msg_id:
        raise ValueError("摘要锚点不能超过原文层边界")
    blocks = history_blocks(turns)
    entries = []
    for block in blocks:
        first, last = block.first_id, block.last_id
        if any(first <= anchor < last for anchor in (summary_upto_msg_id, layer1_from_msg_id)):
            raise ValueError("上下文锚点不能切开完整轮或相交轮组")
        if last <= summary_upto_msg_id:
            continue
        group, safe = _block_view(block)
        if last <= layer1_from_msg_id and not safe:
            raise ValueError("半压边界覆盖未完成或工具未配对的轮")
        entries.append((block, group, safe))
    old = [entry for entry in entries if entry[0].last_id <= layer1_from_msg_id]
    recent = [entry for entry in entries if entry[0].last_id > layer1_from_msg_id]
    recent_groups = [entry[1] for entry in recent]
    kept = _suffix(recent_groups, budget.layer1, estimator)
    demotion_count = len(recent) - len(kept)
    demoted = recent[:demotion_count]
    if any(not safe for _, _, safe in demoted):
        raise ContextBudgetExceeded("上下文预算不足：未完成或工具未配对轮阻止安全降级")
    proposed = demoted[-1][0].last_id if demoted else layer1_from_msg_id
    layer2_entries = [*old, *demoted]
    half_groups = [_shorten(group, estimator) for _, group, _ in layer2_entries]
    full_half = [message for group in half_groups for message in group]
    half_tokens = estimator.message_tokens(full_half)
    sent_half = _suffix(half_groups, budget.layer2, estimator)
    omitted = layer2_entries[:len(half_groups) - len(sent_half)]
    layer1 = [message for group in kept for message in group]
    layer2 = [message for group in sent_half for message in group]
    committed = [row for row in summaries if row["upto_msg_id"] <= summary_upto_msg_id]
    summary, seqs = project_summaries(committed, budget.summary_tokens, estimator)
    return ContextSnapshot(summary_upto_msg_id=summary_upto_msg_id,
        previous_layer1_from_msg_id=layer1_from_msg_id, layer1_from_msg_id=proposed,
        original_from_msg_id=blocks[0].first_id if blocks else 0, original_upto_msg_id=blocks[-1].last_id if blocks else 0,
        summary=summary, summary_seqs=seqs, layer1=layer1, layer2=layer2, budget=budget,
        layer1_tokens=estimator.message_tokens(layer1), layer2_tokens=half_tokens,
        sent_layer2_tokens=estimator.message_tokens(layer2),
        omitted_ranges=[(block.first_id, block.last_id) for block, _, _ in omitted],
        summary_required=half_tokens > budget.layer2)
