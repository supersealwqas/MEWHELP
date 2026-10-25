"""CH05 工作流严格模型契约：七类意图及不包含用户正文的工具/终止决策。"""

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
