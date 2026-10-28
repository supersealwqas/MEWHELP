"""CH08工具审计ORM，与sql/ch08-ddl.sql的无外键审计表保持一致。"""

from datetime import datetime

from sqlalchemy import Enum, Index, JSON, String, Text, text
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER, TINYINT
from sqlalchemy.orm import Mapped, mapped_column

from app.models.customer_service import Base, TABLE_OPTIONS


class ToolAuditLog(Base):
    """工具逻辑调用的独立审计记录，不参与业务结果的提交与防重。"""

    __tablename__ = "tool_audit_logs"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), Index("idx_tool_name", "tool_name"),
        Index("idx_status", "status"), {**TABLE_OPTIONS, "comment": "工具调用审计留痕"})
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="审计主键")
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True, comment="所属会话,无会话上下文的调用为 NULL")
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="模型申请单 id,可对回 messages 流水")
    tool_name: Mapped[str] = mapped_column(String(128), comment="工具名")
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp"), comment="工具来源:内置 / MCP 接入")
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="来源 MCP Server 名,内置工具为 NULL")
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="调用参数")
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="返回结果,过长截断存摘要")
    status: Mapped[str] = mapped_column(Enum("成功", "失败", "超时", "校验拦下", "权限拒绝"), comment="本次调用结局")
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="失败 / 拦下时的原因说明")
    retry_count: Mapped[int] = mapped_column(TINYINT(unsigned=True), server_default=text("0"), comment="实际重试次数,写操作默认不重试恒为 0")
    duration_ms: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True, comment="耗时毫秒")
    created_at: Mapped[datetime] = mapped_column(DATETIME, server_default=text("CURRENT_TIMESTAMP"), comment="调用时间")
