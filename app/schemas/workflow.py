"""工作流严格契约：一次理解、八类意图、工具/终止决策、前置证据闸和服务端动作快照。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from app.schemas.history import StoredTurn
from app.schemas.tools import ToolContext, ToolResult
from app.tools.confirmation import TicketPreviewAction

Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]
ActionKind = Literal["handoff", "create_ticket"]


class OrderSelectionConflict(ValueError):
    """旧选单动作已取消或相同键更换订单，必须返回409而不是重新开轮。"""


class OwnedOrder(BaseModel):
    """订单卡片的最小只读快照，不含其他用户或支付数据。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    order_id: str = Field(min_length=1, max_length=64, description="本人完整订单主键")
    product_name: str = Field(min_length=1, description="商品名称")
    status: str = Field(min_length=1, description="订单当前状态")


class OrderSelectionAction(BaseModel):
    """官方interrupt持久化的稳定选单卡片。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    type: Literal["select_order"] = Field(default="select_order", description="固定选单事件类型")
    action_id: str = Field(min_length=36, max_length=36, description="按会话/原轮生成的UUIDv5")
    source_turn_id: str = Field(min_length=1, max_length=36, description="等待中的原问轮次")
    orders: list[OwnedOrder] = Field(min_length=1, description="只包含当前用户的可选订单")

    @model_validator(mode="after")
    def validate_action(self):
        """验证稳定键及订单唯一性。参数：无；返回：自身；异常：ValueError，结构非法。"""
        from uuid import UUID
        identifier = UUID(self.action_id)
        if identifier.version != 5 or str(identifier) != self.action_id or len({row.order_id for row in self.orders}) != len(self.orders):
            raise ValueError("选单动作键或订单列表非法")
        return self


class OrderSelection(BaseModel):
    """客户端只能确认原卡片中的订单，不能覆盖检查点或预算。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    session_id: str = Field(min_length=1, max_length=64, description="来源会话标识")
    user_id: str = Field(min_length=1, max_length=64, description="来源用户标识")
    source_turn_id: str = Field(min_length=1, max_length=36, description="原问轮次")
    action_id: str = Field(min_length=36, max_length=36, description="服务端生成的选单键")
    order_id: str = Field(min_length=1, max_length=64, description="选中的本人完整订单号")

    @field_validator("session_id", "user_id", "source_turn_id", "order_id")
    @classmethod
    def normalize_nonblank(cls, value):
        """统一身份及订单空白规则。参数：value字符串；返回：去首尾空白值；异常：ValueError，空白。"""
        if not value.strip():
            raise ValueError("选单标识不能为空")
        return value.strip()

    @field_validator("action_id")
    @classmethod
    def validate_identifier(cls, value):
        """只接受规范UUIDv5。参数：value动作键；返回：原值；异常：ValueError，非法键。"""
        from uuid import UUID
        identifier = UUID(value)
        if identifier.version != 5 or str(identifier) != value:
            raise ValueError("选单键必须为规范UUIDv5")
        return value


class AcceptedOrderSelection(BaseModel):
    """已提交的原轮选单结果，由仓储构造供同轮Command使用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    context: ToolContext = Field(description="经会话/来源核验的原轮上下文")
    action: OrderSelectionAction = Field(description="保存的原选单卡片")
    accepted_order_id: str = Field(description="首次接受后不可改变的订单号")
    turn_status: str = Field(description="原轮当前持久化状态")


class TicketResume(BaseModel):
    """浏览器只能决定保存的工单预览，不能覆盖描述、类型或服务器授权身份。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    type: Literal["confirm_ticket"] = Field(description="必须显式指定工单恢复类型")
    session_id: str = Field(min_length=1, max_length=64, description="预览所属会话")
    user_id: str = Field(min_length=1, max_length=64, description="预览所属用户")
    source_turn_id: str = Field(min_length=1, max_length=36, description="保存工具申请及预览的原轮")
    action_id: str = Field(min_length=36, max_length=36, description="服务器冻结预览的 UUIDv5")
    decision: Literal["confirm", "cancel"] = Field(description="用户首次确认或取消决定")

    @field_validator("session_id", "user_id", "source_turn_id")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        """在取得会话锁前统一身份空白。参数：value 严格字符串；返回：去空白标识；异常：ValueError，空标识。"""
        return OrderSelection.normalize_nonblank(value)

    @field_validator("action_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """拒绝非规范预览键。参数：value 动作标识；返回：原值；异常：ValueError，非法 UUIDv5。"""
        return OrderSelection.validate_identifier(value)


REFUND_REASONS = {"wrong_purchase": "不想要/买错了", "quality_issue": "商品质量问题",
                  "not_as_described": "收到商品与描述不符", "wrong_item": "发错商品", "missing_item": "少发漏发",
                  "shipping_damage": "运输破损", "other": "其他"}


class RefundAction(BaseModel):
    """与完成答案原子保存的独立退款建议，不表示已批准或执行退款。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    action_id: str = Field(min_length=36, max_length=36, description="按原轮生成的退款动作UUIDv5")
    kind: Literal["submit_refund"] = Field(default="submit_refund", description="固定本地申请动作")
    source_turn_id: str = Field(min_length=1, max_length=36, description="已核实订单的原问轮次")
    question: str = Field(min_length=1, max_length=4000, description="完整用户原话")
    order_id: str = Field(min_length=1, max_length=64, description="服务端核实的本人订单号")

    @field_validator("action_id")
    @classmethod
    def validate_identifier(cls, value):
        """核对规范UUIDv5。参数：value动作键；返回：原值；异常：ValueError，非法键。"""
        return OrderSelection.validate_identifier(value)


class RefundReceipt(BaseModel):
    """仅确认本地申请已保存，不包含实际退款或到账字段。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    action_id: str = Field(description="申请原动作键")
    refund_no: str = Field(min_length=34, max_length=34, description="稳定的RF前缀本地申请编号")
    order_id: str = Field(description="已核实订单")
    reason_code: str = Field(description="服务端原因目录代码")
    status: Literal["待处理"] = Field(default="待处理", description="本地申请处理状态，不等于退款成功")


class CurrentActions(BaseModel):
    """刷新页面时回读的原轮状态，数据库快照优先于客户端旧卡片。"""

    model_config = ConfigDict(extra="forbid")
    source_turn_id: str | None = Field(default=None, description="最近问答原轮，没有则为空")
    turn_status: str | None = Field(default=None, description="原轮保存状态")
    order_selection: OrderSelectionAction | None = Field(default=None, description="保存的原选单快照")
    accepted_order_id: str | None = Field(default=None, description="首次接受的订单号")
    answer: str | None = Field(default=None, description="已保存完成正文")
    completion_metadata: dict | None = Field(default=None, description="已保存完成元数据")
    resume_required: bool = Field(default=False, description="已接受但仍停在匹配interrupt，需要原键继续")
    refund_action: RefundAction | None = Field(default=None, description="独立已保存退款申请建议")
    refund_receipt: RefundReceipt | None = Field(default=None, description="同动作已提交的本地申请回执")
    ticket_receipt: "TicketReceipt | None" = Field(default=None, description="当前原问对应的真实工单回执，审计待同步仍保留成功事实")
    ticket_confirmation: TicketPreviewAction | None = Field(default=None, description="服务器冻结的当前工单预览")
    ticket_decision: Literal["confirm", "cancel"] | None = Field(default=None, description="已持久化且不可改变的首次工单决定")
    ticket_status: Literal["awaiting_input", "accepted", "completed", "cancelled", "unknown"] | None = Field(
        default=None, description="服务端核实的工单流程状态；确定失败与待核实结果分开，客户端不能据首次确认猜测")


class HistoryDelta(BaseModel):
    """从消息游标定位的完整受影响轮次，终态刷新不要求产生新消息。"""

    turns: list[StoredTurn] = Field(default_factory=list, description="按原消息顺序返回完整配对的受影响轮次")
    last_message_id: int = Field(ge=0, description="本次已扫描消息的最大主键，不因排除当前轮倒退")


class UnderstandingResult(BaseModel):
    """一次理解的扁平结果；完整问题和标准问法分开，订单候选仍须由业务节点验本人归属。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    resolved_query: str = Field(min_length=1, max_length=4000, description="指代补全后的完整问题；无需消解时保留原话")
    standard_query: str = Field(min_length=1, max_length=4000, description="本次理解产生的标准检索问法；失败回退完整原话，检索上限另行检查")
    needs_resolution: bool = Field(description="当前原话是否需要当前会话历史才能明确对象或业务")
    order_id: str | None = Field(min_length=1, max_length=64, description="来自可信输入的完整订单候选；没有或歧义时必须显式JSON null，不得省略")

    @field_validator("resolved_query", "standard_query", "order_id")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        """拒绝空白但不改原字符。参数：value已通过类型/长度校验的字段；返回：原值；异常：ValueError，纯空白。"""
        if value is not None and not value.strip():
            raise ValueError("理解结果不能只有空白")
        return value


class IntentResult(BaseModel):
    """严格八类意图与有限置信度；不承载检索问题、动作或模型升级策略。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    intent: Intent = Field(description="本轮主诉求所属的八类固定意图")
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False, description="模型自报的0–1有限置信度，仅记录评估，不是已校准准确率")


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

    @field_validator("session_id", "user_id")
    @classmethod
    def normalize_identity(cls, value: str) -> str:
        """与聊天入口统一身份键。参数：value通过类型/长度校验的身份；返回：去首尾空白的标识；异常：ValueError，纯空白身份。"""
        # 必须在取得共享互斥之前规范化；仅处理身份，不修改来源问题或工单描述。
        if not value.strip():
            raise ValueError("会话和用户标识不能为空或纯空白字符")
        return value.strip()

    @field_validator("confirmed", mode="before")
    @classmethod
    def require_real_boolean(cls, value):
        """阻止数字1被Literal强收为确认。参数：value原始输入；返回：True；异常：ValueError，非真正布尔true。"""
        if value is not True:
            raise ValueError("必须由用户明确确认")
        return value


class RefundSubmission(TicketConfirmation):
    """仅确认保存建议与固定原因，沿用严格身份和真实布尔确认契约。"""

    reason_code: str = Field(description="必须来自服务端七种固定原因目录")

    @field_validator("reason_code")
    @classmethod
    def validate_reason(cls, value):
        """拒绝客户端编造原因。参数：value原因代码；返回：原值；异常：ValueError，目录外原因。"""
        if value not in REFUND_REASONS:
            raise ValueError("请选择有效的退款原因")
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
