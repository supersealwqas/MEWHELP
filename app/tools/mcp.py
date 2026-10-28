"""MCP逐来源发现管理器与LangChain内容适配；权限只来自本地配置。"""

import asyncio
from copy import deepcopy
import logging

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.schemas.tools import ToolResult, is_business_rejection
from app.tools.contracts import ResultPolicy, ToolDefinition, ToolIdentity
from app.tools.validation import public_schema

logger = logging.getLogger(__name__)
IDENTITY_FIELDS = {"user_id", "conversation_id", "session_id", "turn_id", "tool_call_id", "config", "confirmed"}


class MCPToolManager:
    """合并同时刷新并隔离单Server故障；仅管理临时发现任务，不持久保存网络会话。"""

    def __init__(self, registry, permissions, connections: dict, *, timeout_seconds: float = 3, client=None):
        """注入共享资源。参数：注册表/权限/连接/独立超时/可选客户端；返回：None；异常：ValueError，超时非法。"""
        if timeout_seconds <= 0:
            raise ValueError("发现时限必须为正")
        self.registry, self.permissions = registry, permissions
        self.connections, self.timeout = deepcopy(connections), timeout_seconds
        self.client = client or MultiServerMCPClient(self.connections, handle_tool_errors=False)
        self._refresh_task: asyncio.Task | None = None
        self._closed = False

    async def refresh(self) -> None:
        """获取本轮完整清单；单等待者取消不取消共享请求。参数：无；返回：None；异常：调用者取消。"""
        if self._closed:
            return
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_all())
        await asyncio.shield(self._refresh_task)

    async def _refresh_all(self) -> None:
        """并发刷新来源，按来源分别发布快照。参数：无；返回：None；异常：取消。"""
        self.permissions.reload()
        await asyncio.gather(*(self._refresh_server(server) for server in self.connections))

    async def _refresh_server(self, server: str) -> None:
        """限时发现并替换一份来源清单。参数：Server名；返回：None；异常：取消透传。"""
        definitions = []
        try:
            async with asyncio.timeout(self.timeout):
                discovered = await self.client.get_tools(server_name=server)
            for tool in discovered:
                try:
                    identity = ToolIdentity("mcp", server, tool.name)
                    if self.permissions.allowed(identity):
                        definitions.append(self._definition(identity, tool))
                except (ValueError, TypeError) as error:
                    logger.warning("MCP工具定义被隔离 server=%s error=%s", server, type(error).__name__)
        except Exception as error:
            # 失联不能继续执行来源的旧缓存，其他来源由各自任务维护。
            logger.warning("MCP发现失败，撤下当前来源 server=%s error=%s", server, type(error).__name__)
        for reason in self.registry.replace_source(server, definitions):
            logger.warning("MCP注册隔离 %s", reason)

    def _definition(self, identity: ToolIdentity, tool) -> ToolDefinition:
        """绑定来源、schema及本地策略。参数：可信来源/发现工具；返回：定义；异常：ValueError，不安全参数。"""
        # 不可信用途须在逐工具隔离边界内拒绝，避免整份来源发布时才抛异常。
        if not isinstance(tool.description, str) or not tool.description.strip():
            raise ValueError("MCP工具必须提供非空用途")
        schema = public_schema(tool)
        if IDENTITY_FIELDS.intersection(schema.get("properties", {})):
            raise ValueError("MCP不得要求模型提供身份或确认字段")
        policy = self.permissions.policy(identity)
        rule = policy["result_policy"]
        result_policy = ResultPolicy(tuple(rule["fields"]) if rule["fields"] is not None else None, rule["enum_labels"], rule["is_mock"])

        async def invoke(arguments, context, scope):
            """发送前再检查本地权限，解析内容与结构化附件。参数：公开参数/身份/运行域；返回：ToolResult；异常：真实传输故障或取消。"""
            if IDENTITY_FIELDS.intersection(arguments):
                return ToolResult.failed("INVALID_ARGUMENTS", "工具参数不得包含身份或确认字段")
            rejected = self.permissions.check(identity, arguments, context)
            if rejected is not None:
                return rejected
            try:
                message = await tool.ainvoke({"type": "tool_call", "id": context.tool_call_id,
                                              "name": identity.name, "args": arguments})
            except ToolException:
                return ToolResult.failed("MCP_TOOL_ERROR", "外部工具未能完成查询", is_mock=result_policy.is_mock)
            if not isinstance(message, ToolMessage):
                raise TypeError("MCP适配器没有返回配对工具消息")
            if message.status == "error":
                return ToolResult.failed("MCP_TOOL_ERROR", "外部工具返回查询错误", is_mock=result_policy.is_mock)
            artifact = message.artifact
            payload = artifact.get("structured_content") if isinstance(artifact, dict) else None
            if payload is None:
                payload = {"content": message.content}
            if not isinstance(payload, dict):
                raise TypeError("MCP结构化内容必须为对象")
            if is_business_rejection(payload):
                return ToolResult.failed(payload["code"], payload["error"], is_mock=result_policy.is_mock)
            # 远端字段只放在data内，不能伪造本地verified_knowledge或写确认元数据。
            return ToolResult.success(deepcopy(payload), is_mock=result_policy.is_mock)

        return ToolDefinition(identity, tool.description, schema, invoke, policy["access"], result_policy)

    async def aclose(self) -> None:
        """回收本管理器拥有的在途任务。参数：无；返回：None；异常：无。"""
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)
