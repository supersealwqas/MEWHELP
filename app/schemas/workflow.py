"""CH05 工作流严格契约：七类意图、工具/终止决策、前置证据闸和服务端动作建议快照。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from app.schemas.history import StoredTurn
from app.schemas.tools import ToolContext, ToolResult

Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]
ActionKind = Literal["handoff", "create_ticket"]


class HistoryDelta(BaseModel):
    """从消息游标定位的完整受影响轮次，终态刷新不要求产生新消息。"""

    turns: list[StoredTurn] = Field(default_factory=list, description="按原消息顺序返回完整配对的受影响轮次")
    last_message_id: int = Field(ge=0, description="本次已扫描消息的最大主键，不因排除当前轮倒退")


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


class SuggestedAction(BaseModel):
    """服务端保存的动作建议快照；不接受模型生成链接、按钮HTML或覆盖原问。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    action_id: str = Field(min_length=36, max_length=36, description="服务端按会话/来源轮/类型生成的稳定UUIDv5")
    kind: ActionKind = Field(description="仅转人工或建议创建工单，不代表动作已经执行")
    source_turn_id: str = Field(min_length=1, description="建议产生的原始问答轮次标识")
    question: str = Field(min_length=1, max_length=4000, description="来源轮完整用户原话，保留首尾空白")
    ticket_type: Literal["售后", "投诉", "咨询"] | None = Field(description="建单建议的类型；转人工固定为空")

    @model_validator(mode="after")
    def validate_snapshot(self) -> "SuggestedAction":
        """验证稳定键及动作类型一致。参数：无；返回：自身；异常：ValueError，键或快照非法。"""
        from uuid import UUID

        identifier = UUID(self.action_id)
        if identifier.version != 5 or str(identifier) != self.action_id or not self.question.strip():
            raise ValueError("动作键必须是规范UUIDv5，原问不得为空白")
        if (self.kind == "handoff") != (self.ticket_type is None):
            raise ValueError("转人工不能带工单类型，建单建议必须带类型")
        return self


class TicketConfirmation(BaseModel):
    """客户端只能确认保存的建议，不能提供业务类型、描述或回执状态。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    session_id: str = Field(min_length=1, max_length=64, description="建议所属会话标识")
    user_id: str = Field(default="demo-user", min_length=1, max_length=64, description="当前演示用户标识，必须匹配原会话")
    source_turn_id: str = Field(min_length=1, max_length=36, description="已完成来源问答轮次")
    action_id: str = Field(min_length=36, max_length=36, description="已保存的服务端建议UUIDv5")
    confirmed: Literal[True] = Field(description="用户明确确认，只接受JSON布尔true")

    @field_validator("confirmed", mode="before")
    @classmethod
    def require_real_boolean(cls, value):
        """阻止数字1被Literal强收为确认。参数：value原始输入；返回：True；异常：ValueError，非真正布尔true。"""
        if value is not True:
            raise ValueError("必须由用户明确确认")
        return value


class TicketReceipt(BaseModel):
    """真实已提交票据的回执；审计待同步不改变建单成功事实。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    action_id: str = Field(min_length=36, max_length=36, description="本次确认的稳定动作标识")
    ticket_no: str = Field(min_length=1, max_length=32, description="真实工单号，重试不变化")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="来源建议保存的工单分类")
    status: Literal["待处理"] = Field(description="创建时确定的处理状态，不代表人工接入")
    audit_sync_pending: bool = Field(default=False, description="工单已提交但独立动作审计收尾仍待修复")


class PreparedTicketAction(BaseModel):
    """仓储从已完成来源验证后派生的独立动作，不接受HTTP直接构造。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    request: TicketConfirmation = Field(description="已验证归属的原确认请求，供收尾重新核验")
    context: ToolContext = Field(description="稳定动作轮及工具调用三键的可信身份")
    call: dict = Field(description="仅从保存建议快照生成的create_ticket申请")
    turn_status: Literal["running", "completed", "failed", "cancelled"] = Field(description="动作轮当前持久化状态")
    execution_started: bool = Field(default=False, description="是否已越过执行前边界；无结果时不能盲目重复写入")
    existing_result: ToolResult | None = Field(default=None, description="验明归属之后读出的既有确定工具事实")
