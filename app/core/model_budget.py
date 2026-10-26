"""每轮独占的保守模型预算：请求前占额，保留一次 SDK 重试余量，另存真实用量。"""

from collections import Counter
from copy import deepcopy

OUTPUT_LIMITS = {"understand": 2048, "classify": 4096, "expand": 2048,
                 "rewrite": 2048, "decide": 2048, "answer": 4096}


def budget_summary(snapshot: dict | None) -> dict | None:
    """生成日志和客户端可用的计量摘要，排除原始模型诊断。

    参数：snapshot内部账本快照，可为None；返回：独立摘要或None；异常：输入结构非法时TypeError。
    """
    if snapshot is None:
        return None
    result = deepcopy(snapshot)
    for record in result.get("records", []):
        record.pop("failure", None)
    return result


class WorkflowBudgetExceeded(ValueError):
    """调用次数、完整输入、阶段输出或保守总额不足，调用方必须固定停止。"""

    audit_record: dict | None = None  # 补救被额度拦截时携带此前失败诊断，默认没有诊断。


class BudgetLedger:
    """单轮账本；不绑定模型或共享检索器，可通过纯字典快照跨节点传递。"""

    def __init__(self, *, max_model_calls: int = 9, total_token_budget: int = 240000,
                 max_decisions: int = 5, request_token_budget: int = 16000) -> None:
        """创建空账本。参数：四个限制均为正整数，分别限制总次数/总额/决策次数/单次输入；返回：None；异常：ValueError。"""
        limits = {"max_model_calls": max_model_calls, "total_token_budget": total_token_budget,
                  "max_decisions": max_decisions, "request_token_budget": request_token_budget}
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("模型预算限制必须为正整数")
        self._limits = limits
        self._records: list[dict] = []

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "BudgetLedger":
        """恢复可信 State 快照。参数：snapshot 为 snapshot() 结果；返回：独占账本；异常：KeyError/ValueError，快照错误。"""
        ledger = cls(**{key: snapshot[key] for key in (
            "max_model_calls", "total_token_budget", "max_decisions", "request_token_budget")})
        ledger._records = deepcopy(snapshot["records"])
        return ledger

    def ensure_capacity(self, requests: list[tuple[str, int, int]]) -> None:
        """检查未来请求合计是否放得下，但不扣账。参数：requests 为阶段/输入估算/输出上限列表；返回：None；异常：WorkflowBudgetExceeded。"""
        counts = Counter(record["stage"] for record in self._records)
        reserved = sum(record["reserved_units"] for record in self._records)
        for stage, input_units, output_limit in requests:
            if (stage not in OUTPUT_LIMITS or type(input_units) is not int or input_units < 0
                    or input_units > self._limits["request_token_budget"]
                    or type(output_limit) is not int or not 1 <= output_limit <= OUTPUT_LIMITS[stage]):
                raise WorkflowBudgetExceeded("完整输入或阶段输出超过预算")
            counts[stage] += 1
            # 即便首次成功也不退回 SDK 重试保守占额，避免把估算当作精确账单。
            reserved += (input_units + output_limit) * 2
        if (sum(counts.values()) > self._limits["max_model_calls"]
                or counts["decide"] > self._limits["max_decisions"] or counts["answer"] > 2
                or counts["classify"] > 1 or counts["understand"] > 1
                or counts["expand"] > 1 or counts["rewrite"] > 1
                or reserved > self._limits["total_token_budget"]):
            raise WorkflowBudgetExceeded("本轮模型调用次数或总预算不足")

    def reserve(self, stage: str, input_units: int, output_limit: int) -> None:
        """发起逻辑请求前占额。参数：stage 固定阶段、input_units 完整输入估算、output_limit 输出上限；返回：None；异常：WorkflowBudgetExceeded。"""
        self.ensure_capacity([(stage, input_units, output_limit)])
        self._records.append({"stage": stage, "input_units": input_units, "output_limit": output_limit,
                              "reserved_units": (input_units + output_limit) * 2,
                              "usage": None, "finish_reason": None})

    def record_usage(self, stage: str, usage: dict | None, finish_reason: str | None) -> None:
        """记录最近请求的真实观测。参数：stage 阶段、usage 上游用量或未知、finish_reason 原因或未知；返回：None；异常：ValueError，未先占额。"""
        if not self._records or self._records[-1]["stage"] != stage:
            raise ValueError("用量只能写入最近已占额的同阶段请求")
        self._records[-1]["usage"] = deepcopy(usage)
        self._records[-1]["finish_reason"] = finish_reason

    def record_failure(self, stage: str, diagnostic: dict) -> None:
        """保存最近请求的有界协议诊断，成功补救后仍可追查首次失败。

        参数：stage最近已占额阶段；diagnostic由InvalidToolDecision生成的审计字典。
        返回：None，诊断随独占账本写入检查点；异常：ValueError，阶段未占额或不匹配。
        """
        if not self._records or self._records[-1]["stage"] != stage:
            raise ValueError("诊断只能写入最近已占额的同阶段请求")
        self._records[-1]["failure"] = deepcopy(diagnostic)

    def snapshot(self) -> dict:
        """生成可持久化且独立的账本。参数：无；返回：dict，任一次缺用量则 actual_tokens 为 None；异常：无。"""
        totals = [(record["usage"] or {}).get("total_tokens") for record in self._records]
        known = bool(totals) and all(type(value) is int and value >= 0 for value in totals)
        return {**self._limits, "model_calls": len(self._records),
                "reserved_units": sum(record["reserved_units"] for record in self._records),
                "actual_tokens": sum(totals) if known else None, "records": deepcopy(self._records)}
