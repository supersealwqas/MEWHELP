"""按用户 DDL 定义四张业务表，补充会话映射、轮次恢复与结果防重字段。"""

from datetime import datetime
from sqlalchemy import CheckConstraint, Enum, ForeignKey, Index, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

TABLE_OPTIONS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}


class Base(DeclarativeBase):
    """四张业务表共用的声明式元数据。"""


class Conversation(Base):
    """数字主键会话壳；session_id 兼容原有网页请求。"""
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_user_id", "user_id"), {**TABLE_OPTIONS, "comment": "客服会话"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="会话主键")
    session_id: Mapped[str] = mapped_column(String(64, collation="utf8mb4_bin"), unique=True, comment="兼容客户端会话标识")
    user_id: Mapped[str] = mapped_column(String(64, collation="utf8mb4_bin"), comment="用户标识")
    status: Mapped[str] = mapped_column(Enum("进行中", "已转人工", "已结束"), server_default="进行中", comment="处理状态")
    created_at: Mapped[datetime] = mapped_column(DATETIME, server_default=text("CURRENT_TIMESTAMP"), comment="开启时间")
    updated_at: Mapped[datetime] = mapped_column(DATETIME, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"), comment="更新时间")


class Message(Base):
    """有序消息流水，技术字段记录轮次及幂等工具结果。"""
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_conversation_id", "conversation_id"),
        Index("idx_message_turn", "conversation_id", "turn_id"),
        UniqueConstraint("conversation_id", "turn_id", "tool_call_id", name="uq_tool_result"),
        CheckConstraint("(role = 'tool' AND tool_call_id IS NOT NULL AND tool_call_id <> '') OR (role <> 'tool' AND tool_call_id IS NULL)", name="ck_tool_call_role"),
        CheckConstraint("turn_status IN ('running','completed','failed','cancelled')", name="ck_turn_status"),
        {**TABLE_OPTIONS, "comment": "会话消息流水"},
    )
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="消息主键")
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_messages_conversation"), comment="所属会话")
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool"), comment="角色:用户/助手/工具结果")
    content: Mapped[str | None] = mapped_column(Text, nullable=True, comment="消息正文")
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True, comment="assistant 工具调用申请单")
    rag_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="最终回答状态与本轮完整引用快照")
    tool_call_id: Mapped[str | None] = mapped_column(String(64, collation="utf8mb4_bin"), nullable=True, comment="tool 对应申请单 id")
    turn_id: Mapped[str] = mapped_column(String(36, collation="utf8mb4_bin"), comment="服务端轮次编号")
    turn_status: Mapped[str] = mapped_column(String(16), default="running", server_default="running", comment="轮次执行状态")
    created_at: Mapped[datetime] = mapped_column(DATETIME, server_default=text("CURRENT_TIMESTAMP"), comment="产生时间")


class FAQ(Base):
    """query_faq 的字面关键词数据源。"""
    __tablename__ = "faq"
    __table_args__ = (Index("idx_category", "category"), {**TABLE_OPTIONS, "comment": "常见问答"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="FAQ 主键")
    question: Mapped[str] = mapped_column(String(512), comment="问题")
    answer: Mapped[str] = mapped_column(Text, comment="答案")
    category: Mapped[str] = mapped_column(String(64), comment="分类")
    created_at: Mapped[datetime] = mapped_column(DATETIME, server_default=text("CURRENT_TIMESTAMP"), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DATETIME, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"), comment="更新时间")


class Ticket(Base):
    """人工待处理工单；创建不代表人工已接入。"""
    __tablename__ = "tickets"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), {**TABLE_OPTIONS, "comment": "人工工单"})
    ticket_no: Mapped[str] = mapped_column(String(32, collation="utf8mb4_bin"), primary_key=True, comment="工单号")
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_tickets_conversation"), comment="关联会话")
    description: Mapped[str] = mapped_column(Text, comment="问题描述")
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询"), comment="工单类型")
    status: Mapped[str] = mapped_column(Enum("待处理", "已处理"), server_default="待处理", comment="处理状态")
    created_at: Mapped[datetime] = mapped_column(DATETIME, server_default=text("CURRENT_TIMESTAMP"), comment="创建时间")
