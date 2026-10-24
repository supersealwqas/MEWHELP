"""FastAPI 应用程序核心入口模块。

负责构建 FastAPI 主应用实例，配置全局基础元数据（服务名称、版本号与描述），
注册业务 API，并在首页提供同源客服聊天页面及其专用静态资源。
"""

from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.chat import router as chat_router
from app.api.citations import router as citations_router
from app.api.extract import router as extract_router
from app.api.rageval import router as rageval_router
from app.core.logging import setup_logging
from app.config import settings
from app.core.llm import ChatModelAdapter
from app.db.session import create_database
from app.knowledge.runtime import online_search_runtime
from app.api.kb import router as kb_router
from app.knowledge.store import KnowledgeStore
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.services.chat import ChatService
from app.tools.registry import build_registry

# 在应用创建前初始化业务日志，确保所有模块的 logger 都使用统一格式输出。
setup_logging()

# 仅公开前端专用目录，不能挂载项目根目录，避免泄露 .env 等配置。
STATIC_DIR = Path(__file__).resolve().parent / "static"


class SecurityHeadersMiddleware:
    """统一提供聊天 HTML 防护与 MIME 保护，仅修改响应头，不处理流正文或取消。

    首页和静态 HTML 使用同源资源策略；API 文档保留自身的外部脚本加载方式。
    """

    def __init__(self, app: ASGIApp) -> None:
        """保存下游应用。参数：app 为 ASGI 应用；返回：None；异常：无。"""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """转发请求并补头。参数：scope 请求环境、receive 接收、send 发送；返回：None；异常：透传下游异常与取消。"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            """修改首帧后原样发送。参数：message 为响应帧；返回：None；异常：透传网络错误和取消。"""
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                is_chat_path = scope["path"] == "/" or scope["path"].startswith("/static/")
                if is_chat_path and headers.get("content-type", "").startswith("text/html"):
                    # 首页和静态 HTML 共用策略，避免直接访问 index.html 绕过防护。
                    headers["Content-Security-Policy"] = (
                        "default-src 'self'; script-src 'self'; style-src 'self'; "
                        "img-src 'self'; connect-src 'self'; "
                        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
                    )
            # 不建立额外任务或缓冲正文，SSE 增量和取消直接交给原有调用链。
            await send(message)

        await self.app(scope, receive, send_with_headers)


async def chat_page() -> FileResponse:
    """返回可直接连接本服务聊天 API 的网页。

    参数:
        无；页面与 API 使用相同来源，无需跨域配置或前端密钥。
    返回值:
        FileResponse: UTF-8 HTML 页面；浏览器随后加载专用静态目录中的样式与脚本。
    异常:
        RuntimeError: 部署时缺失 index.html 文件，会在发送响应时抛出。
    """
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


async def rag_eval_page() -> FileResponse:
    """返回四策略评估看板页面。

    参数：无；页面与评估 API 同源，不接外部图表库。
    返回：FileResponse，UTF-8 HTML；数据由 /api/rag-eval/overview 提供。
    异常：RuntimeError，部署时缺失 rag-eval.html。
    """
    return FileResponse(
        STATIC_DIR / "rag-eval.html",
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@asynccontextmanager
async def database_runtime():
    """建立单 worker 运行时、恢复中断轮并加载知识检索。

    参数:
        无
    返回值:
        ChatService: 共享数据库、编码器、Milvus 与检索策略的聊天服务。
    异常:
        恢复/数据库/模型/向量/策略初始化失败均阻止启动。
    """
    engine, factory = create_database()
    lease_engine = None
    try:
        repository = ChatRepository(factory)
        await repository.recover_interrupted_turns()
        lease_engine = create_async_engine(
            engine.url,
            poolclass=NullPool,
            connect_args={"connect_timeout": 5},
        )
        store = KnowledgeStore(factory, lease_engine)
        async with online_search_runtime(settings, store) as search:
            yield ChatService(
                repository,
                SupportRepository(factory, search),
                ChatModelAdapter(),
                build_registry(include_refund_policy=settings.knowledge_retrieval_mode == "hybrid"),
                settings,
            )
    finally:
        with anyio.CancelScope(shield=True):
            try:
                if lease_engine is not None:
                    await lease_engine.dispose()
            finally:
                await engine.dispose()


# 运行时工厂可在测试边界替换，生产默认始终连接 MySQL，不提供内存后备。
create_runtime = database_runtime


def create_app(runtime_factory=None) -> FastAPI:
    """创建应用。参数：可选异步运行时工厂；返回：FastAPI；异常：资源错误在 lifespan 启动时抛出。"""
    @asynccontextmanager
    async def lifespan(application):
        """先恢复后接收请求。参数：应用；返回：生命周期；异常：运行时初始化失败。"""
        async with (runtime_factory or create_runtime)() as service:
            application.state.chat_service = service
            try:
                yield
            finally:
                del application.state.chat_service

    application = FastAPI(title="MewHelp E-Commerce Customer Service",
        description="MewHelp 电商智能客服系统与售后工单结构化提取服务 API", version="0.1.0", lifespan=lifespan)
    application.add_middleware(SecurityHeadersMiddleware)
    application.include_router(extract_router)
    application.include_router(chat_router)
    application.include_router(citations_router)
    application.include_router(kb_router)
    application.include_router(rageval_router)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    application.add_api_route("/", chat_page, response_class=FileResponse, include_in_schema=False)
    application.add_api_route("/rag-eval", rag_eval_page, response_class=FileResponse, include_in_schema=False)
    return application


app = create_app()
