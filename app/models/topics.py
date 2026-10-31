"""CH10固定主题结果ORM，仅注册用户指定的四字段单表。"""

from datetime import datetime

from sqlalchemy import ForeignKey, JSON, UniqueConstraint, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.models.customer_service import TABLE_OPTIONS
from app.models.rag_quality import LowConfidenceQuestion


class TopicBase(DeclarativeBase):
    """仅管理主题结果表；结构变更必须使用显式固定迁移。"""


class TopicClassification(TopicBase):
    """每条池记录最新确定主题；概率、版本与批次证据保存在本地原件。"""

    __tablename__ = "topic_classifications"
    __table_args__ = (UniqueConstraint("question_id", name="uk_question_id"),
                      {**TABLE_OPTIONS, "comment": "主题分类结果"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键")
    question_id: Mapped[int] = mapped_column(BIGINT(unsigned=True),
        ForeignKey(LowConfidenceQuestion.__table__.c.id, name="fk_topic_question"), nullable=False,
        comment="归类的问题,指向 low_confidence_questions.id")
    labels: Mapped[list[str]] = mapped_column(JSON, nullable=False, comment="多标签,17 类权威类目里命中的若干个")
    classified_at: Mapped[datetime] = mapped_column(DATETIME, nullable=False,
        server_default=text("CURRENT_TIMESTAMP"), comment="归类时间")
