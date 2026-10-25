"""CH05 工作流严格契约：七类意图、工具/终止决策及前置证据闸快照。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]
ActionKind = Literal["handoff", "create_ticket"]


class IntentResult(BaseModel):
    """仅保存分类结果，禁止模型补充解释或创造路由。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    intent: Intent = Field(description="本轮主诉求所属的七类固定意图")


class AgentDecision(BaseModel):
    """内部决策；工具申请由适配器严格验证，终止元数据不能充当用户正文。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    kind: Literal["tools", "answer", "clarify"] = Field(description="申请只读工具、已有信息回答或缺信息追问")
    tool_calls: list[dict] = Field(default_factory=list, description="适配器校验后的完整 id/name/args 工具申请")
    suggested_actions: list[ActionKind] = Field(default_factory=list, max_length=2, description="最多两个不重复的界面建议，不自动执行")

    @model_validator(mode="after")
    def validate_shape(self) -> "AgentDecision":
        """拒绝空工具决定与混合元数据。参数：无；返回：自身；异常：ValueError，结构不一致。"""
        if len(set(self.suggested_actions)) != len(self.suggested_actions):
            raise ValueError("建议动作不能重复")
        if self.kind == "tools":
            if not self.tool_calls or self.suggested_actions:
                raise ValueError("工具决定必须有申请且不能夹带终止建议")
        elif self.tool_calls:
            raise ValueError("终止决定不能申请工具")
        return self


class RetrievalLowConfidence(BaseModel):
    """仅表示成功检索后的空命中或低分，不含模型自报身份或用户原话。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    source: Literal["retrieval_low_conf"] = Field(default="retrieval_low_conf", description="固定的前置检索弱证据入口")
    reason: str = Field(min_length=1, max_length=1000, description="成功检索后无证据或最高重排分不足的原因")


class GateDecision(BaseModel):
    """成功检索的最高归一化分判定；坏数据和依赖失败必须作为异常处理。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    passed: bool = Field(description="证据非空且最高归一化重排分达到门槛时为真")
    max_score: float | None = Field(ge=0, le=1, allow_inf_nan=False, description="本轮全部重排分中的最高值，成功零命中时为空")
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False, description="已有rerank_min_score门槛，不代表回答正确概率")
    low_confidence: RetrievalLowConfidence | None = Field(default=None, description="仅未通过时携带的弱证据入池草案，不直接写库")

    @model_validator(mode="after")
    def validate_outcome(self) -> "GateDecision":
        """拒绝相互矛盾的闸快照。参数：无；返回：自身；异常：ValueError，通过标记或弱证据草案与分数不符。"""
        expected = self.max_score is not None and self.max_score >= self.threshold
        if self.passed != expected or (self.low_confidence is None) != self.passed:
            raise ValueError("证据闸通过标记、分数和弱证据草案不一致")
        return self
