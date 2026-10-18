"""FastAPI 应用程序核心入口模块。

负责构建 FastAPI 主应用实例，配置全局基础元数据（服务名称、版本号与描述），
注册业务 API，并在首页提供同源客服聊天页面及其专用静态资源。
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.core.logging import setup_logging
from app.config import settings
from app.core.llm import ChatModelAdapter
from app.db.session import create_database
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.services.chat import ChatService
from app.tools.registry import build_registry

# 在应用创建前初始化业务日志，确保所有模块的 logger 都使用统一格式输出。
setup_logging()

# 仅公开前端专用目录，不能挂载项目根目录，避免泄露 .env 等配置。
STATIC_DIR = Path(__file__).resolve().parent / "static"


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
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
            # 脚本、样式及网络请求只允许同源；页面没有外部 CDN 依赖。
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self'; connect-src 'self'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            ),
        },
    )


@asynccontextmanager
async def database_runtime():
    """建立单 worker 运行时并恢复中断轮。参数：无；返回：聊天服务；异常：恢复/连接失败阻止启动。"""
    engine, factory = create_database()
    try:
        repository = ChatRepository(factory)
        await repository.recover_interrupted_turns()
        yield ChatService(repository, SupportRepository(factory), ChatModelAdapter(), build_registry(), settings)
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
    application.include_router(extract_router)
    application.include_router(chat_router)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    application.add_api_route("/", chat_page, response_class=FileResponse, include_in_schema=False)
    return application


app = create_app()
