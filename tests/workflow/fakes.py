"""后续图/服务测试使用的有限工作流模型替身；不能用于真实 Prompt 评估。"""

from collections import deque

from app.core.model_budget import BudgetLedger
from app.schemas.workflow import AgentDecision, IntentResult


class ScriptedWorkflowModel:
    """消费有限脚本并保留消息记录；任何额外请求立即失败，不无限返回默认答案。"""

    def __init__(self, *, intents=(), decisions=(), answers=()) -> None:
        """参数：intents 意图结果/名称、decisions 严格决策、answers 各次正文块序列；返回：None；异常：无。"""
        self.intents, self.decisions, self.answers = deque(intents), deque(decisions), deque(answers)
        self.inputs: list[dict] = []

    def _take(self, script, stage: str, messages: list, ledger: BudgetLedger, output_limit: int):
        """消费一次逻辑请求。参数：script 有限队列、stage/messages 观测输入、ledger/output_limit 计量；返回：脚本项；异常：AssertionError/脚本异常/预算错误。"""
        if not script:
            raise AssertionError(f"未预期的额外 {stage} 模型请求")
        ledger.reserve(stage, 100, output_limit)
        self.inputs.append({"stage":stage, "messages":list(messages)})
        result = script.popleft()
        ledger.record_usage(stage, None, None if isinstance(result, BaseException) else "stop")
        if isinstance(result, BaseException):
            raise result
        return result

    async def classify(self, messages: list, ledger: BudgetLedger) -> IntentResult:
        """返回脚本分类。参数：messages 输入、ledger 本轮账本；返回：IntentResult；异常：脚本/额度/验证错误。"""
        result = self._take(self.intents, "classify", messages, ledger, 128)
        return IntentResult(intent=result) if isinstance(result, str) else result

    async def decide(self, messages: list, tools: list, ledger: BudgetLedger) -> AgentDecision:
        """返回脚本内部决定。参数：messages 输入、tools 可见工具、ledger 本轮账本；返回：AgentDecision；异常：脚本/额度错误。"""
        result = self._take(self.decisions, "decide", messages, ledger, 2048)
        self.inputs[-1]["tools"] = list(tools)
        return result

    async def stream_answer(self, messages: list, ledger: BudgetLedger):
        """按脚本转发最终正文块。参数：messages 输入、ledger 本轮账本；返回：异步字符串流；异常：脚本/额度错误。"""
        for delta in self._take(self.answers, "answer", messages, ledger, 4096):
            if isinstance(delta, BaseException):
                raise delta
            yield delta
