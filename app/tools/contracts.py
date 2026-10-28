"""工具系统共享契约：可信来源、注册定义、调用快照与执行资源。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from app.schemas.tools import ToolContext, ToolResult

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from app.core.model_budget import BudgetLedger


def json_hash(value: Any) -> str:
    """计算稳定JSON摘要。参数：可JSON序列化值；返回：SHA256文本；异常：非法JSON类型或非有限数值。"""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolIdentity:
    """工具可信来源身份，避免同名工具跨来源接管原申请。"""

    source: Literal["builtin", "mcp"]  # 本地登记的来源类型。
    server: str | None  # MCP Server名；内置为空。
    name: str  # 模型平台可接受的公开名称。

    def __post_init__(self) -> None:
        """检查名称与来源关系。参数：构造字段；返回：None；异常：ValueError，身份非法。"""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.name):
            raise ValueError("工具名称必须为1至64位字母、数字、下划线或连字符")
        if self.source not in {"builtin", "mcp"} or (self.source == "builtin" and self.server is not None):
            raise ValueError("工具来源不合法")
        if self.source == "mcp" and (not isinstance(self.server, str) or not self.server.strip() or len(self.server) > 64):
            raise ValueError("MCP工具必须包含有效Server名")


@dataclass(frozen=True)
class ToolBinding:
    """决定时冻结的来源与schema版本；只含可持久化数据。"""

    identity: ToolIdentity  # 申请所属原来源。
    schema_hash: str  # 原公开schema摘要。


@dataclass(frozen=True)
class ResultPolicy:
    """本地控制的模型字段视图，远端声明不能影响权限或结果可信度。"""

    fields: tuple[str, ...] | None = None  # 需要展示的顶层字段；空为通用JSON。
    enum_labels: dict[str, dict[str, str]] = field(default_factory=dict)  # 字段到中文枚举映射。
    is_mock: bool = False  # 由本地确认的演示数据标记。


@dataclass(frozen=True)
class WriteAuthorization:
    """绑定某一冻结动作的确认决定，不是可复用的布尔写权限。"""

    conversation_id: int  # 已验明归属的会话主键。
    turn_id: str  # 原工具申请轮。
    action_id: str  # 已保存动作的稳定ID。
    tool_call_id: str  # 已保存工具申请ID。
    arguments_hash: str  # 用户所见冻结参数摘要。
    decision: Literal["confirm", "cancel"]  # 首次接受的确认或取消。
    origin: Literal["chat", "action"]  # 聊天暂停或旧按钮确认。


@dataclass
class KnowledgeCallContext:
    """知识工具共享本轮证据与预算，只在运行时存在。"""

    state: dict  # 已保存的本轮原问、政策条件与可信证据。
    ledger: BudgetLedger  # 当前轮独占预算账本。
    search: Any = None  # 本实例权威检索器，不进入检查点。
    settings: Any = None  # 本实例证据和预算配置。
    query_expander: Any = None  # 本实例政策扩写器。


@dataclass
class ToolExecutionScope:
    """执行器接收的可信资源；所有成员均不得由模型参数注入。"""

    binding: ToolBinding | None = None  # 原决定来源；程序内置调用可推导。
    rejection: ToolResult | None = None  # 编排层已决定的跳过结果，仍须统一留痕。
    authorize_write: Callable[[ToolBinding, dict, ToolContext], Awaitable[WriteAuthorization]] | None = None  # 保存动作的授权回调。
    knowledge: KnowledgeCallContext | None = None  # 知识工具共享原轮预算。
    support: Any = None  # 既有业务仓储，由执行器注入。


@dataclass(frozen=True)
class ToolDefinition:
    """注册记录；运行处理器与本地策略不进入检查点或模型参数。"""

    identity: ToolIdentity  # 唯一来源身份。
    description: str  # 模型可见用途。
    parameters: dict  # 公开JSON Schema。
    invoke: Callable[[dict, ToolContext, ToolExecutionScope], Awaitable[ToolResult]]  # 异步业务边界。
    access: Literal["read", "write"]  # 本地指定读写类型。
    result_policy: ResultPolicy = field(default_factory=ResultPolicy)  # 本地结果呈现规则。
    legacy_tool: BaseTool | None = field(default=None, repr=False, compare=False)  # 迁移期BaseTool投影，非另一份注册表。

    @property
    def name(self) -> str:
        """提供公开名称。参数：无；返回：str；异常：无。"""
        return self.identity.name

    @property
    def binding(self) -> ToolBinding:
        """冻结当前来源和schema。参数：无；返回：ToolBinding；异常：schema不能JSON序列化。"""
        return ToolBinding(self.identity, json_hash(self.parameters))


def binding_dict(binding: ToolBinding) -> dict:
    """序列化已冻结的可信来源。参数：绑定；返回：可持久化字典；异常：无。"""
    return {"identity": {"source": binding.identity.source, "server": binding.identity.server,
                         "name": binding.identity.name}, "schema_hash": binding.schema_hash}


def binding_from_dict(value: dict) -> ToolBinding:
    """恢复服务器保存的绑定。参数：原绑定字典；返回：ToolBinding；异常：ValueError/KeyError，快照损坏。"""
    identity = ToolIdentity(**value["identity"])
    digest = value["schema_hash"]
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("工具schema摘要不合法")
    return ToolBinding(identity, digest)
