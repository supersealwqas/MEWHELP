"""CH08真实验收的独占进程、生产运行时和安全证据采集，不修改用户配置。"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    """以替换方式保存本轮证据。参数：目标文件和JSON内容；返回：None；异常：文件或序列化错误。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def free_port() -> int:
    """取得本机空闲临时端口。参数：无；返回：端口号；异常：socket错误。"""
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


class OwnedServer:
    """只启动和结束本验收创建的子进程，不按端口查杀其他服务。"""

    def __init__(self, module: str, port: int, report: Path, name: str):
        """保存启动参数。参数：模块、端口、报告目录、日志名；返回：None；异常：无。"""
        self.module, self.port, self.report, self.name = module, port, report, name
        self.process = self.output = None
        self.generation = 0

    async def start(self, *, hot: bool = False) -> None:
        """启动并等候端口就绪。参数：hot控制售后新增工具；返回：None；异常：启动失败或超时。"""
        if self.process is not None:
            raise ValueError("本服务已启动")
        self.generation += 1
        self.output = (self.report / f"{self.name}-{self.generation}.log").open("w", encoding="utf-8")
        arguments = [sys.executable, "-m", self.module, "--port", str(self.port)]
        if hot:
            arguments.append("--hot")
        self.process = subprocess.Popen(arguments, cwd=ROOT, stdout=self.output, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(100):
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name}启动失败，见独立日志")
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", self.port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                await asyncio.sleep(0.1)
        raise TimeoutError(f"{self.name}启动超时")

    async def stop(self) -> None:
        """关闭自己拥有的进程并收回句柄。参数：无；返回：None；异常：系统进程错误。"""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    await asyncio.to_thread(self.process.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait, timeout=5)
            self.process = None
        if self.output is not None:
            self.output.close()
            self.output = None

    def identity(self) -> dict:
        """返回可审计进程身份。参数：无；返回：名称、PID和端口；异常：无。"""
        return {"name": self.name, "pid": self.process.pid if self.process else None,
                "port": self.port, "generation": self.generation}


def configure(env_file: Path, database: str, report: Path, ports: tuple[int, int]):
    """显式加载配置并只覆盖本轮隔离资源。参数：env、测试库、报告、MCP端口；返回：业务/知识配置；异常：配置错误。"""
    from app.config import Settings
    from tests.verify_ch08_live import validate_database
    validate_database(database)
    original = Settings(_env_file=env_file.resolve())
    replacements = {}
    for name in ("knowledge_model_path", "rerank_model_path"):
        path = Path(getattr(original, name))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            raise ValueError(f"缺少本地权重：{name}")
        replacements[name] = str(path.resolve())
    knowledge_config = original.model_copy(update=replacements)
    permission_path = report / "mcp-permissions.json"
    permission_path.write_bytes((ROOT / "data/tools/mcp-permissions.json").read_bytes())
    config = knowledge_config.model_copy(update={"mysql_database": database,
        "mcp_permissions_path": str(permission_path),
        "mcp_logistics_url": f"http://127.0.0.1:{ports[0]}/mcp",
        "mcp_after_sales_url": f"http://127.0.0.1:{ports[1]}/mcp",
        "workflow_checkpoint_path": str(report / "checkpoints.sqlite3")})
    return config, knowledge_config


@asynccontextmanager
async def live_runtime(config, knowledge_config, report: Path):
    """组装真实模型/MySQL/MCP/SQLite及只读已有知识，资源始终独立清理。

    参数：显式业务/知识配置和新报告目录；返回：共享service与观察资源；
    异常：配置、数据库、模型或知识连接失败，不切换为替身。
    """
    import httpx
    from langchain_openai import ChatOpenAI
    from sqlalchemy import text
    from app.db.session import create_database
    from app.models.customer_service import Base
    import app.models.tool_audit  # 注册审计元数据，不连接或修改应用库。
    from app.db.migrate_ch08 import migrate_connection
    from app.knowledge.runtime import knowledge_storage, online_search_runtime
    from app.knowledge.query_rewrite import QueryExpander, QueryRewriter
    from app.repositories.chat import ChatRepository
    from app.repositories.support import SupportRepository
    from app.repositories.context import ContextRepository
    from app.repositories.tool_audit import ToolAuditRepository
    from app.services.chat import ChatService
    from app.services.summaries import SummaryManager
    from app.workflow.summarization import SummaryGenerator
    from app.tools.registry import build_agent_registry
    from app.tools.executor import ToolExecutor
    from app.tools.permissions import ToolPermissions
    from app.tools.mcp import MCPToolManager
    from app.workflow.graph import WorkflowDependencies
    from app.workflow.model import WorkflowModelAdapter
    from app.workflow.runtime import workflow_runtime
    from tests.ch05_live_runtime import RecordingChat
    from tests.verify_ch08_live import validate_database

    validate_database(config.mysql_database)
    engine, factory = create_database(config)
    models = []
    lock = await engine.connect()
    mcp = summaries = None
    sync_http, async_http = httpx.Client(), httpx.AsyncClient()

    def model_factory(*, evaluation_stage=None, **options):
        """记录真实SDK输入和输出。参数：生产模型选项；返回：透明包装；异常：模型配置错误。"""
        selected = options.pop("model", None) or config.chat_model
        model = ChatOpenAI(base_url=config.chat_base_url, model=selected, api_key=config.chat_api_key,
            http_client=sync_http, http_async_client=async_http, **options)
        return RecordingChat(model, models, evaluation_stage or "answer")

    try:
        validate_database(await lock.scalar(text("SELECT DATABASE()")))
        if await lock.scalar(text("SELECT GET_LOCK('mewhelp:ch08:live:exclusive', 0)")) != 1:
            raise ValueError("已有CH08验收占用测试库")
        async with engine.begin() as connection:
            # 仅补本章测试库缺表，不drop已有表、数据或知识资源。
            await connection.run_sync(Base.metadata.create_all)
            await connection.run_sync(lambda conn: migrate_connection(conn, apply=True))
        async with knowledge_storage(knowledge_config) as store, online_search_runtime(knowledge_config, store) as search:
            search.query_rewriter = QueryRewriter(model_factory=model_factory, request_budget=config.request_token_budget)
            repository = ChatRepository(factory)
            registry = build_agent_registry()
            permissions = ToolPermissions(config.mcp_permissions_path)
            permissions.reload()
            mcp = MCPToolManager(registry, permissions, {
                "logistics": {"transport": "streamable_http", "url": config.mcp_logistics_url},
                "after_sales": {"transport": "streamable_http", "url": config.mcp_after_sales_url}},
                timeout_seconds=config.mcp_discovery_timeout_seconds)
            await mcp.refresh()
            executor = ToolExecutor(registry, repository, SupportRepository(factory, search),
                config.tool_timeout_seconds, config.tool_max_retries, config.tool_retry_delay_seconds,
                knowledge_timeout_seconds=config.knowledge_timeout_seconds, permissions=permissions,
                audit_repository=ToolAuditRepository(factory), audit_timeout_seconds=config.tool_audit_timeout_seconds)
            deps = WorkflowDependencies(WorkflowModelAdapter(model_factory=model_factory, stream_usage=config.workflow_stream_usage),
                search, executor, repository, config,
                QueryExpander(model_factory=model_factory, request_budget=config.request_token_budget),
                registry=registry, permissions=permissions, mcp=mcp)
            async with workflow_runtime(deps, Path(config.workflow_checkpoint_path)) as graph:
                summaries = SummaryManager(ContextRepository(factory), SummaryGenerator(config), config)
                service = ChatService(repository, graph, executor, config, summaries=summaries)
                await service.reconcile_waiting_turns()
                await summaries.recover()
                yield SimpleNamespace(service=service, factory=factory, engine=engine, models=models,
                    registry=registry, mcp=mcp, config=config, report=report, executor=executor)
    finally:
        write_json(report / "model-calls.json", models)
        if summaries is not None:
            await summaries.close()
        if mcp is not None:
            await mcp.aclose()
        await async_http.aclose()
        sync_http.close()
        await lock.close()
        await engine.dispose()


def runtime_identity(runtime) -> dict:
    """读取运行中实例和schema身份。参数：真实运行时；返回：JSON证据；异常：无。"""
    return {"pid": os.getpid(), "registry_id": id(runtime.registry), "graph_id": id(runtime.service.graph),
            "tools": [{**asdict(tool.identity), "schema": tool.parameters} for tool in runtime.registry.snapshot()]}


async def database_snapshot(runtime, session_id: str) -> dict:
    """回读指定会话的原文、真实工单及审计。参数：运行时和session_id；返回：证据字典；异常：数据库错误。"""
    from sqlalchemy import select
    from app.models.customer_service import Conversation, Message, Ticket
    from app.models.tool_audit import ToolAuditLog
    async with runtime.factory() as session:
        conversation = await session.scalar(select(Conversation).where(Conversation.session_id == session_id))
        if conversation is None:
            return {"session_id": session_id, "messages": [], "tickets": [], "audits": []}
        result = {"session_id": session_id, "conversation_id": conversation.id}
        for name, model in (("messages", Message), ("tickets", Ticket), ("audits", ToolAuditLog)):
            rows = (await session.scalars(select(model).where(model.conversation_id == conversation.id))).all()
            result[name] = [{column.name: getattr(row, column.name) for column in model.__table__.columns} for row in rows]
        return result


async def sample_turn(runtime, message: str, *, session_id: str | None = None) -> dict:
    """通过生产聊天服务驱动真实模型一轮并回读数据库。参数：运行时/原话/可选会话；返回：完整证据；异常：真实运行错误。"""
    from app.schemas.chat import ChatRequest
    session_id = session_id or "ch08-" + uuid4().hex
    before = len(runtime.models)
    events = []
    async for event in runtime.service.stream_turn(ChatRequest(session_id=session_id, user_id="demo-user", message=message)):
        events.append(event.model_dump(mode="json"))
    result = {"input": message, "events": events, "model_calls": runtime.models[before:],
              "database": await database_snapshot(runtime, session_id)}
    # 即使后续验收断言失败也保留已观察到的完整一轮，不只留错误文字。
    write_json(runtime.report / ("turn-" + uuid4().hex + ".json"), result)
    return result
