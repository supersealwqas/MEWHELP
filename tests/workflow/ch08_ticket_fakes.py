"""CH08 工单确认测试资源：真实图、SQLite 仓储和执行器配合有限模型脚本。"""

from contextlib import asynccontextmanager
from dataclasses import asdict, replace

from sqlalchemy import select, update

from app.models.customer_service import Message
from app.repositories.support import SupportRepository
from app.repositories.tool_audit import ToolAuditRepository
from app.tools.contracts import WriteAuthorization
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from tests.workflow.service_fakes import service_runtime


@asynccontextmanager
async def ticket_runtime(path, **script):
    """创建只使用临时 SQLite 的完整业务边界。

    参数：path 为独占临时目录，script 为有限模型输出；返回：服务、依赖、工厂和真实查询计数；异常：数据库错误。
    """
    async with service_runtime(path, **script) as (service, deps, factory):
        async with factory.begin() as session:
            connection = await session.connection()
            await connection.exec_driver_sql("CREATE TABLE IF NOT EXISTS tickets (ticket_no TEXT PRIMARY KEY, conversation_id INTEGER, description TEXT, ticket_type TEXT, status TEXT DEFAULT '待处理', created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
            await connection.exec_driver_sql("CREATE TABLE IF NOT EXISTS tool_audit_logs (id INTEGER PRIMARY KEY, conversation_id INTEGER, tool_call_id TEXT, tool_name TEXT, tool_source TEXT, mcp_server TEXT, arguments JSON, result_summary TEXT, status TEXT, error_message TEXT, retry_count INTEGER, duration_ms INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        queried = []
        registry = ToolRegistry()
        for definition in deps.registry.snapshot():
            if definition.name == "query_order":
                original = definition.invoke

                async def query(arguments, context, scope):
                    """记录真实查询进入次数并完整调用生产实现。

                    参数：生产工具参数、上下文和scope；返回：生产ToolResult；异常：生产工具异常。
                    """
                    queried.append(context.tool_call_id)
                    return await original(arguments, context, scope)

                definition = replace(definition, invoke=query)
            registry.register_definition(definition)
        deps.registry = registry
        deps.executor = ToolExecutor(registry, deps.chat, SupportRepository(factory),
            audit_repository=ToolAuditRepository(factory), retry_delay_seconds=0)
        service.executor = deps.executor
        yield service, deps, factory, queried


async def saved_decision(factory, context, *, decision="confirm"):
    """在真实数据库写入已接受决定，隔离尚由 Task7 实现的 HTTP 接口。

    参数：factory 临时工厂，context 原工具轮，decision 确认或取消；返回：序列化服务器凭据；异常：缺少真实预览或数据库错误。
    """
    async with factory.begin() as session:
        source = await session.scalar(select(Message).where(Message.conversation_id == context.conversation_id,
            Message.turn_id == context.turn_id, Message.role == "user"))
        pending = source.rag_metadata["pending_ticket"]
        assert pending["status"] == "awaiting_input"
        source.rag_metadata = {**source.rag_metadata, "pending_ticket": {**pending,
            "status": "accepted", "decision": decision}}
        await session.execute(update(Message).where(Message.conversation_id == context.conversation_id,
            Message.turn_id == context.turn_id).values(turn_status="running"))
        action = pending["action"]
        return asdict(WriteAuthorization(conversation_id=context.conversation_id, turn_id=context.turn_id,
            action_id=action["action_id"], tool_call_id=action["tool_call_id"],
            arguments_hash=pending["arguments_hash"], decision=decision, origin="chat"))
