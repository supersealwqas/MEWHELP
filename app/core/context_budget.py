"""会话上下文的统一用量估算与窗口预算推导。

TokenEstimator只计算实际消息字段；ContextBudget保留每项扣减，供分层、日志和启动自检共用。
估算单位是可校准的容量指标，不冒充上游真实分词或计费数据。
"""

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

if TYPE_CHECKING:
    from app.config import Settings


class ContextBudgetExceeded(ValueError):
    """固定开销、当前轮或最低历史保留量超出模型窗口。"""


@dataclass(frozen=True)
class TokenEstimator:
    """统一文本及消息协议计量；只读系数避免同一快照中途改变口径。"""

    non_ascii_tokens: float = 2.0  # 每个非ASCII字符的保守估算单位。

    def __post_init__(self) -> None:
        """校验系数。参数：无；返回：None；异常：ValueError表示系数非有限正数。"""
        value = self.non_ascii_tokens
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("非ASCII字符估算系数必须是有限正数")

    def value_tokens(self, value: object) -> int:
        """估算JSON字段。参数：value可序列化内容；返回：向上取整用量；异常：TypeError，内容不可序列化。"""
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        ascii_count = sum(character.isascii() for character in serialized)
        # 中文不先变成六字符转义；只在完整字段计量后取整。
        return math.ceil(ascii_count / 4 + (len(serialized) - ascii_count) * self.non_ascii_tokens)

    def message_tokens(self, messages: list[BaseMessage]) -> int:
        """估算实际消息字段。参数：messages模型消息；返回：累计用量；异常：TypeError，正文或参数不可序列化。"""
        total = 0
        for message in messages:
            fields = {"role": message.type, "content": message.content}
            if message.name:
                fields["name"] = message.name
            if isinstance(message, AIMessage) and message.tool_calls:
                fields["tool_calls"] = message.tool_calls
            if isinstance(message, ToolMessage):
                fields["tool_call_id"] = message.tool_call_id
            # checkpoint身份、来源和usage不属于发给模型的正文。
            total += self.value_tokens(fields)
        return total


@dataclass(frozen=True)
class ContextBudget:
    """一轮共享的预算明细；所有字段都使用同一估算口径。"""

    window: int  # 应用声明的模型上下文窗口。
    fixed_tokens: int  # 固定system与工具定义。
    evidence_tokens: int  # 有界检索证据预留。
    summary_tokens: int  # 有界摘要投影预留。
    output_tokens: int  # 本次输出预留。
    safety_tokens: int  # 分词偏差和协议安全余量。
    current_peak_tokens: int  # 当前问题和累积工具消息的峰值。
    target_turns: int  # 期望保留的历史轮数。
    steady_turn_tokens: int  # 校准后的单轮稳态用量。
    available_history: int  # 窗口扣减后实际能分配的历史容量。
    history: int  # 最终历史预算。
    layer1: int  # 最近原文预算，整数七成。
    layer2: int  # 半压层预算，保留舍入余数。
    request_limit: int  # 不含输出及安全余量的完整输入上限。


def derive_history_budget(
    *, window: int, fixed_tokens: int, evidence_tokens: int, summary_tokens: int,
    output_tokens: int, safety_tokens: int, current_peak_tokens: int,
    target_turns: int, steady_turn_tokens: int,
) -> ContextBudget:
    """从窗口逐项扣减后分配历史，供配置校验与在线投影共用。

    参数：window窗口；fixed_tokens固定前缀；evidence_tokens证据；summary_tokens摘要；
        output_tokens输出；safety_tokens余量；current_peak_tokens当前轮峰值；
        target_turns目标轮数；steady_turn_tokens稳态单轮，均为非负整数，窗口/轮数/稳态须正数。
    返回：ContextBudget，包含全部扣减及七三分层。
    异常：ValueError表示非法参数；ContextBudgetExceeded表示原文层容不下一稳态轮。
    """
    parts = dict(window=window, fixed_tokens=fixed_tokens, evidence_tokens=evidence_tokens,
                 summary_tokens=summary_tokens, output_tokens=output_tokens, safety_tokens=safety_tokens,
                 current_peak_tokens=current_peak_tokens, target_turns=target_turns,
                 steady_turn_tokens=steady_turn_tokens)
    if any(type(value) is not int or value < 0 for value in parts.values()):
        raise ValueError("上下文预算各项必须为非负整数")
    if min(window, target_turns, steady_turn_tokens) == 0:
        raise ValueError("窗口、目标轮数和稳态用量必须为正整数")
    request_limit = window - output_tokens - safety_tokens
    available = request_limit - fixed_tokens - evidence_tokens - summary_tokens - current_peak_tokens
    history = min(target_turns * steady_turn_tokens, available)
    layer1 = history * 7 // 10
    if layer1 < steady_turn_tokens:
        raise ContextBudgetExceeded(f"上下文预算不足：原文层{layer1} < 单轮{steady_turn_tokens}；扣减明细={parts}")
    return ContextBudget(**parts, available_history=available, history=history,
                         layer1=layer1, layer2=history - layer1, request_limit=request_limit)


def context_budget(settings: "Settings", *, fixed_tokens: int) -> ContextBudget:
    """将应用配置换算为可复算预算。

    参数：settings已校验配置；fixed_tokens使用相同中文系数实测的system/schema占用。
    返回：ContextBudget；异常：ContextBudgetExceeded表示容量不足，ValueError表示非法扣减。
    """
    window = settings.model_context_window
    # 一步可能并行申请多个工具，峰值必须按整轮工具数，而非决策步数计算。
    peak = settings.max_user_input_tokens + settings.tool_max_calls_per_turn * (
        settings.tool_result_max_tokens + settings.tool_call_max_tokens)
    return derive_history_budget(
        window=window, fixed_tokens=fixed_tokens,
        evidence_tokens=min(settings.context_evidence_max_tokens, window // 16),
        summary_tokens=min(settings.context_summary_max_tokens, window // 24),
        output_tokens=settings.max_output_tokens, safety_tokens=max(512, (window + 19) // 20),
        current_peak_tokens=peak, target_turns=settings.context_target_turns,
        steady_turn_tokens=settings.context_steady_turn_tokens,
    )
