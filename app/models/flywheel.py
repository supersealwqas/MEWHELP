"""CH09 固定 DDL 的待审知识缺口与评估轮次 ORM，独立于业务和 CH04 元数据。"""

from datetime import datetime

from sqlalchemy import Enum, FetchedValue, Index, JSON, String, Text, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models.customer_service import TABLE_OPTIONS


class FlywheelBase(DeclarativeBase):
    """仅管理 CH09 两表；表结构创建和增量升级由显式迁移负责。"""


class ReviewQueue(FlywheelBase):
    """标准化去重后的知识缺口；只保存固定审核字段，不扩展轮次或发布状态。"""

    __tablename__ = "review_queue"
    __table_args__ = (Index("idx_review_status", "review_status"), {**TABLE_OPTIONS, "comment": "飞轮待审队列"})

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="缺口主键,也是查重命中要返回的 matched_question_id")
    normalized_question: Mapped[str] = mapped_column(String(512), nullable=False, comment="标准化后的 FAQ 式问题")
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True, comment="模型生成的示例答案,备查")
    occurrence_count: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False, server_default=text("1"), comment="出现次数,查重命中累加,越高越该优先补")
    review_status: Mapped[str] = mapped_column(Enum("待审", "通过", "驳回"), nullable=False, server_default="待审", comment="人工审核状态")
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True, comment="审核通过时补的核准答案,走 ch03 落库流程写回知识库")
    created_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="首次入队时间")
    updated_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False,
        # FetchedValue 只通知 ORM 回读；真正自动更新由固定数据库默认表达式承担。
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"), server_onupdate=FetchedValue(), comment="更新时间")


class EvalRun(FlywheelBase):
    """一次自动或手动评估的指标快照，保留数据量与真实产生时间。"""

    __tablename__ = "eval_runs"
    __table_args__ = (Index("idx_created_at", "created_at"), {**TABLE_OPTIONS, "comment": "自动化评估流水线轮次结果"})

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="评估轮次主键")
    triggered_by: Mapped[str] = mapped_column(Enum("定时", "手动"), nullable=False, server_default="定时", comment="这轮怎么起的:定时任务,或某次改动后手动跑")
    dataset_size: Mapped[int] = mapped_column(INTEGER(unsigned=True), nullable=False, comment="这轮跑的评估集条数")
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, comment="各指标分数及本轮评估口径")
    created_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="跑完落表时间")
