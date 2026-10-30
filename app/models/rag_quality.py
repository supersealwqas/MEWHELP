"""用户 CH04 DDL 的低置信问题池与忠实度编造个案 ORM。

问题池只记录原话和入池理由，不保存 useful 或新增轮次字段；编造台账一题一行，
保留当轮证据全集、累计次数、首次/最近时间与人工处置/复发时间。使用独立元数据，
避免旧业务初始化及测试库重置误处理质量表；会话外键直接引用既有列。
"""

from datetime import datetime

from sqlalchemy import Enum, ForeignKey, Index, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models.customer_service import Conversation, TABLE_OPTIONS
from app.models.flywheel import ReviewQueue


class QualityBase(DeclarativeBase):
    """仅管理两张质量表；建表和结构核验由专用迁移入口负责。"""


class LowConfidenceQuestion(QualityBase):
    """生成自评知识不足时的原话问题池，为后续数据归并保留真实诉求。"""

    __tablename__ = "low_confidence_questions"
    __table_args__ = (Index("idx_source", "source"), Index("idx_created_at", "created_at"),
                      Index("idx_matched_review_id", "matched_review_id"),
                      {**TABLE_OPTIONS, "comment": "低置信度问题池"})

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键")
    conversation_id: Mapped[int | None] = mapped_column(
        # 直接引用既有列，使外键可跨元数据解析，同时不复制或注册业务表。
        BIGINT(unsigned=True), ForeignKey(Conversation.__table__.c.id, name="fk_lcq_conversation"), nullable=True, comment="来源会话")
    raw_question: Mapped[str] = mapped_column(Text, nullable=False, comment="用户原话,带情绪口语")
    source: Mapped[str] = mapped_column(Enum("retrieval_low_conf", "self_check", "user_feedback"), nullable=False,
        comment="入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True, comment="判不能的原因,留作复盘")
    retrieved_chunks: Mapped[dict | None] = mapped_column(JSON(none_as_null=True), nullable=True,
        comment="落池时的召回片段快照:Top 几条的原文与得分,审核页展示用;没走检索为 NULL")
    matched_review_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True),
        ForeignKey(ReviewQueue.__table__.c.id, name="fk_lcq_review", ondelete="SET NULL"), nullable=True,
        comment="查重后归并到的缺口,指向 review_queue.id")
    created_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="入池时间")


class FaithCase(QualityBase):
    """忠实度裁判判出的编造台账，eval_id 唯一并支持人工处置后复发。"""

    __tablename__ = "faith_cases"
    __table_args__ = (UniqueConstraint("eval_id", name="uk_eval_id"), Index("idx_status", "status"),
                      Index("idx_last_seen_at", "last_seen_at"), {**TABLE_OPTIONS, "comment": "ch04 忠实度编造个案台账"})

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键")
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, comment="评估集题号,如 A43;一题一行")
    bucket: Mapped[str] = mapped_column(String(24), nullable=False, comment="题目所属桶:A_policy / B_model / C_colloquial / E_multi")
    query: Mapped[str] = mapped_column(String(512), nullable=False, comment="用户问题原文")
    strategy: Mapped[str] = mapped_column(String(24), nullable=False, server_default="hybrid_rerank", comment="产出这条答案的检索策略")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="被判编造的那版生成答案原文")
    reason: Mapped[str] = mapped_column(Text, nullable=False, comment="裁判给的理由:编在哪一句")
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True, comment="本轮喂给模型的Top-K证据全集快照,旧数据未记为NULL")
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="判这条的裁判模型")
    status: Mapped[str] = mapped_column(Enum("未解决", "已解决", "无需解决"), nullable=False, server_default="未解决", comment="处置状态,人工点按钮改")
    seen_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False, server_default=text("1"), comment="被判编造的累计次数(跨轮)")
    first_seen_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="第一次被判编造的时间")
    last_seen_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="最近一次被判编造的时间")
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True, comment="处置说明;已解决/无需解决要说明原因,退回未解决时清空")
    resolved_at: Mapped[datetime | None] = mapped_column(DATETIME, nullable=True, comment="最近处置时间;复发后仍保留用来标复发")


QUALITY_TABLES = (LowConfidenceQuestion.__table__, FaithCase.__table__)
