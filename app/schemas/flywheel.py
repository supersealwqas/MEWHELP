"""CH09 检索历史、入池信号与幂等状态的共享契约，不补造未知历史证据。"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RetrievalStatus = Literal["ok", "empty", "fault", "legacy_incomplete", "legacy_unavailable"]
PoolSource = Literal["retrieval_low_conf", "self_check", "user_feedback"]
PositiveId = Annotated[int, Field(gt=0)]
DecimalId = Annotated[str, Field(pattern=r"^[1-9][0-9]*$")]


class FlywheelContract(BaseModel):
    """禁止未知字段和宽松类型转换，避免客户端或旧数据静默改变共享契约。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class RetrievalChunkSnapshot(FlywheelContract):
    """当轮召回片段的原文、身份与分数；旧数据无法确认的身份保持缺失。"""

    chunk_id: PositiveId | None = Field(default=None, description="当轮知识主键；旧证据未记录时为 None")
    source_key: str | None = Field(default=None, description="当轮来源键；历史未知时缺失")
    content_hash: str | None = Field(default=None, description="当轮内容指纹；不可用新检索补写")
    embedding_revision: str | None = Field(default=None, description="当轮编码模型与配置版本；历史未知时缺失")
    question: str = Field(description="当轮召回原文中的问题")
    answer: str = Field(description="当轮召回原文中的答案")
    rank: int = Field(gt=0, description="该次检索内从 1 开始的名次")
    recall_score: float | None = Field(default=None, allow_inf_nan=False, description="实际召回分数；未取得时缺失")
    rerank_score: float | None = Field(default=None, allow_inf_nan=False, description="实际重排分数；未取得时缺失")


class RetrievalAttempt(FlywheelContract):
    """一次真实查询或可追溯历史记录；故障不伪装成零结果。"""

    attempt_id: str = Field(min_length=1, description="本轮检索尝试的稳定标识")
    query: str | None = Field(default=None, description="实际检索查询；仅旧历史无法确认时可缺失")
    status: RetrievalStatus = Field(description="成功、零结果、故障或两类历史缺失状态")
    candidate_count: int | None = Field(default=None, ge=0, description="真实候选总数；尚未获得或旧历史未知时缺失")
    top_k: int | None = Field(default=None, gt=0, description="实际使用的返回上限；仅旧历史未知时可缺失")
    chunks: list[RetrievalChunkSnapshot] = Field(default_factory=list, description="本次真实召回原文快照")
    fault: str | None = Field(default=None, description="检索故障说明；无故障时缺失")

    @model_validator(mode="after")
    def require_current_retrieval_facts(self) -> Self:
        """约束新检索必需事实，保留旧历史的缺失值。

        参数：self 已完成字段校验的尝试。
        返回：Self，事实完整的新尝试或明确标注的旧记录。
        异常：ValueError，新尝试缺查询/上限、成功检索缺候选数/片段主键。
        """
        if self.status in {"ok", "empty", "fault"} and (not self.query or self.top_k is None):
            raise ValueError("实际检索必须记录 query 和 top_k")
        if self.status in {"ok", "empty"}:
            if self.candidate_count is None or any(chunk.chunk_id is None for chunk in self.chunks):
                raise ValueError("成功检索必须记录候选总数和可追溯片段主键")
        return self


class RetrievalSnapshot(FlywheelContract):
    """一轮所有检索尝试的版本化快照；整列 SQL NULL 另表示确认没走检索。"""

    schema_version: Literal[1] = Field(default=1, description="快照结构版本，固定为 1")
    status: RetrievalStatus = Field(description="整轮检索状态，不用空值混淆故障与历史缺失")
    attempts: list[RetrievalAttempt] = Field(default_factory=list, description="按发生顺序保存的各次检索")


class PoolSignal(FlywheelContract):
    """固定三入口之一产生的落池理由。"""

    source: PoolSource = Field(description="检索证据低、生成自评不足或用户反馈未解决")
    reason: str = Field(description="该入口判定需入池的真实原因")


class TurnPoolState(FlywheelContract):
    """轮内各入口落池与审核计数的持久幂等状态，内部主键保持整数。"""

    schema_version: Literal[1] = Field(default=1, description="轮内状态版本，固定为 1")
    pool_ids: dict[PoolSource, PositiveId] = Field(default_factory=dict, description="各已落池入口对应的内部问题池主键")
    matched_review_id: PositiveId | None = Field(default=None, description="本轮归并命中的审核队列主键")
    counted_review_id: PositiveId | None = Field(default=None, description="本轮已计入出现次数的审核队列主键")


class FeedbackReceipt(FlywheelContract):
    """对外反馈回执，十进制字符串防止浏览器丢失 BIGINT 精度。"""

    pool_id: DecimalId = Field(description="十进制正整数形式的问题池主键")
    matched_review_id: DecimalId | None = Field(default=None, description="已归并审核主键的十进制字符串，未处理时为空")


class NormalizedGap(FlywheelContract):
    """经隐私泛化的FAQ问题和仅供人工参考的示例答案。"""

    normalized_question: str = Field(min_length=1, max_length=512, description="保留关键条件、去除私人身份的FAQ问题，不截断事实")
    ai_suggested_answer: str | None = Field(description="仅供人工审核参考的示例答案；无法可靠建议时为null")


class ReviewCandidate(FlywheelContract):
    """单页中真实、尚未冻结的待审缺口。"""

    id: PositiveId = Field(description="本页真实审核主键，只能原样选择")
    normalized_question: str = Field(min_length=1, max_length=512, description="该待审缺口的标准问题")


class MatchDecision(FlywheelContract):
    """当前候选页的语义等价判定，不包含审核或发布权限。"""

    matched_review_id: PositiveId | None = Field(description="本页语义相同的真实主键；没有重复为null")


class ApprovalRequest(FlywheelContract):
    """人工明确提交的核准答案；不从模型示例自动取值。"""

    answer: str = Field(min_length=1, description="人工核准正文；保留空白和标点，三格字符/token硬限由实际编码器校验")

    @field_validator("answer")
    @classmethod
    def require_human_answer(cls, value: str) -> str:
        """拒绝纯空白但不改写人工输入。参数：答案；返回：原字符串；异常：ValueError空答案。"""
        if not value.strip():
            raise ValueError("人工答案不能为空")
        return value
