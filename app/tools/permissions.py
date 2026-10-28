"""MCP本地权限热加载和业务归属核验；远端描述与annotations永不授予权限。"""

from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.tools import ToolContext, ToolResult
from app.tools.contracts import ToolIdentity
from app.tools.mock_data import NOT_OWNED, order_id_for_tracking_no, owns_order

logger = logging.getLogger(__name__)


class ResultRules(BaseModel):
    """本地结果字段/枚举/演示标记规则，禁止未声明字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    fields: list[str] | None = Field(default=None, description="允许展示的顶层字段；空为通用JSON")
    enum_labels: dict[str, dict[str, str]] = Field(default_factory=dict, description="字段到中文枚举映射")
    is_mock: bool = Field(default=False, description="本地确认的演示标记")


class MCPPermission(BaseModel):
    """一个明确来源的本地授权；写规则可记录但本章不会放行。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    source: Literal["mcp"] = Field(description="本规则只授权MCP来源")
    server: str = Field(min_length=1, max_length=64, description="本地配置中的Server名")
    name: str = Field(min_length=1, max_length=64, description="公开工具名")
    enabled: bool = Field(description="本地是否启用")
    access: Literal["read", "write"] = Field(description="本地指定操作类型")
    ownership: Literal["none", "order_id", "tracking_no"] = Field(description="发送请求前的归属检查")
    result_policy: ResultRules = Field(default_factory=ResultRules, description="本地结果呈现规则")

    @model_validator(mode="after")
    def validate_identity(self):
        """复用统一身份约束。参数：自身字段；返回：自身；异常：ValueError，来源名非法。"""
        ToolIdentity(self.source, self.server, self.name)
        return self


class PermissionFile(BaseModel):
    """完整权限文件，只有全部校验成功才替换运行策略。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: int = Field(ge=1, le=1, description="权限文件版本，固定1")
    tools: list[MCPPermission] = Field(description="逐来源工具授权规则")


class ToolPermissions:
    """应用级本地授权快照；格式错误保留完整旧规则，文件不可读时拒绝调用。"""

    def __init__(self, path: Path | str):
        """设置本地路径。参数：权限JSON文件；返回：None；异常：路径类型错误。"""
        self.path = Path(path)
        self._rules: dict[ToolIdentity, MCPPermission] = {}
        self._content: bytes | None = None
        self._read_error: str | None = None

    def reload(self) -> None:
        """完整验证后热替换。参数：无；返回：None；异常：仅记录文件/格式错误，不抛到聊天层。"""
        try:
            content = self.path.read_bytes()
            self._read_error = None
        except OSError as error:
            self._rules, self._content = {}, None
            if self._read_error != type(error).__name__:
                logger.warning("工具权限文件不可读，MCP默认拒绝 error=%s", type(error).__name__)
            self._read_error = type(error).__name__
            return
        if content == self._content:
            return
        self._content = content
        try:
            config = PermissionFile.model_validate(json.loads(content))
            rules = {ToolIdentity(row.source, row.server, row.name): row for row in config.tools}
            if len(rules) != len(config.tools):
                raise ValueError("权限规则身份重复")
        except (ValueError, TypeError) as error:
            logger.warning("工具权限格式不合法，保留最后完整有效规则 error=%s", type(error).__name__)
            return
        self._rules = rules

    def policy(self, identity: ToolIdentity) -> dict | None:
        """返回本地规则副本。参数：完整来源；返回：规则或空；异常：无。"""
        rule = self._rules.get(identity)
        return deepcopy(rule.model_dump()) if rule else None

    def allowed(self, identity: ToolIdentity) -> bool:
        """检查可向模型暴露的来源。参数：完整身份；返回：是否允许；异常：无。"""
        if identity.source == "builtin":
            return True
        rule = self._rules.get(identity)
        return bool(rule and rule.enabled and rule.access == "read")

    def check(self, identity: ToolIdentity, arguments: dict, context: ToolContext) -> ToolResult | None:
        """发请求前重验权限及本人业务对象。参数：来源/公开参数/可信身份；返回：拒绝结果或None；异常：无。"""
        self.reload()
        if not self.allowed(identity):
            return ToolResult.failed("PERMISSION_DENIED", "本地尚未授权此工具，本次没有执行")
        if identity.source == "builtin":
            return None
        rule = self._rules[identity]
        field = rule.ownership
        if field == "none":
            return None
        value = arguments.get(field)
        if not isinstance(value, str):
            return ToolResult.failed("INVALID_ARGUMENTS", f"请提供有效的{field}")
        identifier = order_id_for_tracking_no(value.strip()) if field == "tracking_no" else value.strip()
        if not owns_order(context.user_id, identifier):
            return ToolResult.failed(NOT_OWNED["code"], NOT_OWNED["error"])
        return None
