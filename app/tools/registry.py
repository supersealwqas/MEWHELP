"""集中注册可供模型选择的异步业务工具，拒绝重名和未知名称。"""

from langchain_core.tools import BaseTool
from app.tools.business import create_ticket, query_faq, query_logistics, query_order, query_product, query_refund_policy


class ToolRegistry:
    """工具名称到真实异步 LangChain 工具的映射。"""
    def __init__(self):
        """建立空注册表。参数：无；返回：None；异常：无。"""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册唯一异步工具。参数：tool 业务工具；返回：None；异常：ValueError 表示重复或非异步。"""
        if tool.name in self._tools or not getattr(tool, "coroutine", None):
            raise ValueError("工具必须异步且名称唯一")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """按名称查找。参数：name 技术名称；返回：BaseTool；异常：KeyError 表示未知工具。"""
        return self._tools[name]

    def all(self) -> list[BaseTool]:
        """导出模型可见工具。参数：无；返回：独立工具列表；异常：无。"""
        return list(self._tools.values())


def build_registry(*, include_refund_policy: bool = True) -> ToolRegistry:
    """注册业务工具。参数：include_refund_policy 是否提供混合退款入口；返回：注册表；异常：注册冲突。"""
    registry = ToolRegistry()
    tools = [query_order, query_product, query_logistics, query_faq, create_ticket]
    if include_refund_policy:
        tools.append(query_refund_policy)
    for business_tool in tools:
        registry.register(business_tool)
    return registry
