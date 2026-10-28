"""独立审计短事务与有界审计输入；数据库失败由调用方隔离，不覆盖业务事实。"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.tool_audit import ToolAuditLog


class ToolAuditEntry(BaseModel):
    """一次逻辑工具调用的终态；不增加原DDL未定义的数据库字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    conversation_id: int | None = Field(default=None, ge=0, description="可信会话主键，无会话为空")
    tool_call_id: str | None = Field(default=None, max_length=64, description="原申请单ID")
    tool_name: str = Field(min_length=1, max_length=128, description="可信注册名")
    tool_source: Literal["builtin", "mcp"] = Field(description="可信来源类型")
    mcp_server: str | None = Field(default=None, max_length=64, description="本地配置的MCP Server名")
    arguments: dict | None = Field(default=None, description="完整原始公开参数")
    result_summary: str | None = Field(default=None, description="最多4000字符的审计摘要")
    status: Literal["成功", "失败", "超时", "校验拦下", "权限拒绝"] = Field(description="最终调用状态")
    error_message: str | None = Field(default=None, description="最多512字符的安全失败说明")
    retry_count: int = Field(default=0, ge=0, le=255, description="实际重试次数")
    duration_ms: int | None = Field(default=None, ge=0, le=4294967295, description="不含确认等待和审计的耗时毫秒")

    @field_validator("result_summary", "error_message")
    @classmethod
    def bound_text(cls, value, info):
        """按字段截断审计副本。参数：文本及字段信息；返回：有界文本或空；异常：无。"""
        return value[:4000 if info.field_name == "result_summary" else 512] if value is not None else None


class ToolAuditRepository:
    """每次新开会话提交审计；不持有也不操作调用方业务事务。"""

    def __init__(self, factory):
        """注入会话工厂。参数：异步SQLAlchemy会话工厂；返回：None；异常：无。"""
        self.factory = factory

    async def record(self, entry: ToolAuditEntry) -> None:
        """独立提交一条终态审计。参数：已验证的审计输入；返回：None；异常：数据库异常透传。"""
        async with self.factory() as session:
            async with session.begin():
                session.add(ToolAuditLog(**entry.model_dump()))
