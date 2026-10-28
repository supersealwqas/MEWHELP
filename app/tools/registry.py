"""统一动态工具注册表：来源原子更新、冲突隔离及内置BaseTool兼容投影。"""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import inspect
from langchain_core.tools import BaseTool
from app.tools.business import create_ticket, query_faq, query_logistics, query_order, query_product, query_refund_policy
from app.schemas.tools import ToolResult, is_business_rejection
from app.tools.contracts import ResultPolicy, ToolDefinition, ToolIdentity
from app.tools.validation import check_schema, public_schema

_CONTROL_NAMES = {"submit_decision", "submit_intent", "submit_understanding"}


def _snapshot(definition: ToolDefinition) -> ToolDefinition:
    """复制可变定义但共享异步处理器。参数：登记记录；返回：独立快照；异常：无。"""
    return replace(definition, parameters=deepcopy(definition.parameters), result_policy=deepcopy(definition.result_policy))


class ToolRegistry:
    """应用级唯一工具记录；逐来源候选集合与可执行快照分开以确定性隔离撞名。"""
    def __init__(self):
        """建立空注册表。参数：无；返回：None；异常：无。"""
        self._builtin: dict[str, ToolDefinition] = {}
        self._sources: dict[str, list[ToolDefinition]] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, tool: BaseTool) -> None:
        """兼容受信任本地BaseTool登记。参数：tool业务工具；返回：None；异常：ValueError，冲突或非异步。"""
        if not getattr(tool, "coroutine", None):
            raise ValueError("工具必须异步且名称唯一")

        async def invoke(arguments, context, scope):
            """适配旧内置工具且只注入服务端身份。参数：公开参数/可信身份/资源；返回：ToolResult；异常：业务异常。"""
            values = dict(arguments)
            fields = getattr(tool.args_schema, "model_fields", {})
            if "user_id" in fields and "user_id" not in public_schema(tool).get("properties", {}):
                values["user_id"] = context.user_id
            payload = await tool.ainvoke(values, config={"configurable": {"context": context, "support_repository": scope.support}})
            if not isinstance(payload, dict):
                raise TypeError("内置工具返回格式错误")
            if is_business_rejection(payload):
                return ToolResult.failed(payload["code"], payload["error"])
            return ToolResult.success(payload)

        self.register_definition(ToolDefinition(ToolIdentity("builtin", None, tool.name), tool.description,
            public_schema(tool), invoke, "write" if tool is create_ticket else "read",
            ResultPolicy(is_mock=tool in (query_order, query_product, query_logistics)), legacy_tool=tool))

    def get(self, name: str) -> BaseTool:
        """按名称查找。参数：name 技术名称；返回：BaseTool；异常：KeyError 表示未知工具。"""
        definition = self._definitions[name]
        if definition.legacy_tool is None:
            raise KeyError(name)
        return definition.legacy_tool

    def all(self) -> list[BaseTool]:
        """导出模型可见工具。参数：无；返回：独立工具列表；异常：无。"""
        return [row.legacy_tool for row in self._definitions.values() if row.legacy_tool is not None]

    @staticmethod
    def _validate(definition: ToolDefinition) -> None:
        """验证完整登记记录。参数：工具定义；返回：None；异常：ValueError，不完整或非法。"""
        if not definition.description.strip() or definition.access not in {"read", "write"} or not inspect.iscoroutinefunction(definition.invoke):
            raise ValueError("工具必须包含用途、显式读写类型和异步处理器")
        check_schema(definition.parameters)

    def register_definition(self, definition: ToolDefinition) -> None:
        """运行中登记唯一内置工具。参数：完整可信定义；返回：None；异常：ValueError，冲突或非法。"""
        self._validate(definition)
        occupied = set(self._builtin) | {row.name for rows in self._sources.values() for row in rows}
        if definition.identity.source != "builtin" or definition.name in occupied | _CONTROL_NAMES:
            raise ValueError("工具名称已占用；MCP必须按Server原子登记")
        self._builtin = {**self._builtin, definition.name: _snapshot(definition)}
        self._rebuild()

    def get_definition(self, identity: ToolIdentity) -> ToolDefinition:
        """按完整原来源获取最新定义。参数：可信身份；返回：独立定义；异常：KeyError，来源不存在或已变。"""
        definition = self._definitions[identity.name]
        if definition.identity != identity:
            raise KeyError(identity)
        return _snapshot(definition)

    def snapshot(self) -> tuple[ToolDefinition, ...]:
        """读取完整可执行快照。参数：无；返回：不可扩充且内容隔离的元组；异常：无。"""
        return tuple(_snapshot(row) for row in self._definitions.values())

    def replace_source(self, server: str, definitions: list[ToolDefinition]) -> list[str]:
        """原子替换单Server候选，空列表撤下来源。参数：来源名/本轮清单；返回：隔离说明；异常：ValueError，非法定义。"""
        for definition in definitions:
            self._validate(definition)
            if definition.identity.source != "mcp" or definition.identity.server != server:
                raise ValueError("Server与工具来源不匹配")
        self._sources = {**self._sources, server: [_snapshot(row) for row in definitions]}
        return self._rebuild()

    def _rebuild(self) -> list[str]:
        """从完整候选集确定性产生下一快照。参数：无；返回：冲突说明；异常：无。"""
        candidates = [row for server in sorted(self._sources) for row in self._sources[server]]
        counts = Counter(row.name for row in candidates)
        updated, rejected = dict(self._builtin), []
        for row in candidates:
            if counts[row.name] > 1 or row.name in self._builtin or row.name in _CONTROL_NAMES:
                rejected.append(f"工具名冲突：{row.identity.server}/{row.name}")
            else:
                updated[row.name] = row
        # 读者永远只看到上一整份或下一整份，不观察逐条更新过程。
        self._definitions = updated
        return rejected


def build_registry(*, include_refund_policy: bool = True) -> ToolRegistry:
    """注册业务工具。参数：include_refund_policy 是否提供混合退款入口；返回：注册表；异常：注册冲突。"""
    registry = ToolRegistry()
    tools = [query_order, query_product, query_logistics, query_faq, create_ticket]
    if include_refund_policy:
        tools.append(query_refund_policy)
    for business_tool in tools:
        registry.register(business_tool)
    return registry


def build_agent_registry() -> ToolRegistry:
    """提供Agent固定只读范围。参数：无；返回：三项只读注册表；异常：工具注册冲突。"""
    registry = ToolRegistry()
    for business_tool in (query_order, query_logistics, query_product):
        registry.register(business_tool)
    return registry


def build_ticket_registry() -> ToolRegistry:
    """提供用户确认后的单一建单工具。参数：无；返回：仅create_ticket的注册表；异常：工具注册冲突。"""
    registry = ToolRegistry()
    registry.register(create_ticket)
    return registry
