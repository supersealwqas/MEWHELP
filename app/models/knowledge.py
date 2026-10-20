"""CH03 知识库三表 ORM。

使用独立 KnowledgeBase 元数据，不扩展现有 CH02 四张业务表；MySQL 保存知识
原文、发布资格、向量同步状态、抽取暂存与批次身份，Milvus 只保存向量。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Enum,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER, TINYINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

TABLE_OPTIONS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}


class KnowledgeBase(DeclarativeBase):
    """CH03 三张知识库表共用的声明式元数据。"""


class KnowledgeChunk(KnowledgeBase):
    """知识库原文权威源与发布/同步状态。"""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("source_key", "ordinal", name="uq_chunks_source_ordinal"),
        Index("idx_chunks_active_status", "is_active", "vectorize_status"),
        Index("idx_chunks_source_key", "source_key"),
        Index("idx_chunks_ordinal", "source_key", "ordinal"),
        {**TABLE_OPTIONS, "comment": "知识库 chunk 原文权威源"},
    )

    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="与 Milvus 对齐的正整数主键"
    )
    source_key: Mapped[str] = mapped_column(
        String(255, collation="utf8mb4_bin"), nullable=False, comment="稳定来源键：doc:相对路径或 qa:暂存ID"
    )
    ordinal: Mapped[int] = mapped_column(INTEGER, nullable=False, comment="来源内从 0 开始的连续序号")
    category: Mapped[str] = mapped_column(String(255), nullable=False, comment="分类，进入向量")
    questions: Mapped[str] = mapped_column(Text, nullable=False, comment="问法，进入向量")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="答案原文，进入向量")
    section_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="完整章节路径，仅元数据"
    )
    content_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="内容类型，仅元数据"
    )
    is_key_clause: Mapped[bool] = mapped_column(
        TINYINT(1), nullable=False, server_default=text("0"), comment="是否关键条款，仅元数据"
    )
    content_hash: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_bin"), nullable=False, comment="三格与格式规范指纹"
    )
    embedding_revision: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="编码模型与推理配置标识"
    )
    is_active: Mapped[bool] = mapped_column(
        TINYINT(1), nullable=False, server_default=text("1"), comment="是否具备在线发布资格"
    )
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True, comment="同来源前一块"
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True, comment="同来源后一块"
    )
    vector_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Milvus 主键，成功后回填"
    )
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done"), nullable=False, server_default="pending", comment="向量同步状态"
    )
    created_at: Mapped[datetime] = mapped_column(
        DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DATETIME,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        comment="更新时间",
    )


class QAExtractionBatch(KnowledgeBase):
    """历史对话抽取的固定输入快照与恢复状态。"""

    __tablename__ = "qa_extraction_batches"
    __table_args__ = (
        UniqueConstraint("batch_no", name="uq_batches_batch_no"),
        Index("idx_batches_status", "status"),
        {**TABLE_OPTIONS, "comment": "历史对话抽取批次"},
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_bin"), nullable=False, comment="稳定批次号"
    )
    source_ids: Mapped[list] = mapped_column(JSON, nullable=False, comment="本批最终回答消息 ID 列表")
    snapshot_upper_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), nullable=False, comment="快照的最终回答消息上界"
    )
    input_hash: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_bin"), nullable=False, comment="批次输入指纹"
    )
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, comment="抽取 Prompt 规范版本")
    status: Mapped[str] = mapped_column(
        Enum("pending", "done"), nullable=False, server_default="pending", comment="pending/done"
    )
    created_at: Mapped[datetime] = mapped_column(
        DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DATETIME,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
    )


class QAExtractionStaging(KnowledgeBase):
    """模型抽取出的候选知识与持久冲突/处置状态。"""

    __tablename__ = "qa_extraction_staging"
    __table_args__ = (
        UniqueConstraint("source_fingerprint", name="uq_staging_source_fingerprint"),
        Index("idx_staging_batch_no", "batch_no"),
        Index("idx_staging_status", "status"),
        Index("idx_staging_conflict_group", "conflict_group"),
        {**TABLE_OPTIONS, "comment": "历史对话抽 QA 暂存表"},
    )

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False, comment="来源抽取批次")
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="来源会话或导出标识")
    source_ids: Mapped[list] = mapped_column(JSON, nullable=False, comment="来源最终回答消息 ID")
    source_fingerprint: Mapped[str] = mapped_column(
        String(64, collation="utf8mb4_bin"), nullable=False, comment="来源与问答规范化指纹"
    )
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="真实用户问法")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="有原文支持的答案")
    category: Mapped[str] = mapped_column(String(255), nullable=False, comment="知识分类")
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded"), nullable=False, server_default="extracted"
    )
    conflict_group: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="持久冲突组身份")
    evidence_refs: Mapped[list] = mapped_column(JSON, nullable=False, comment="处置证据引用")
    reason: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="处置原因")
    knowledge_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
        comment="关联的已发布知识",
    )
    created_at: Mapped[datetime] = mapped_column(
        DATETIME, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DATETIME,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
    )
