"""FastAPI 应用程序核心入口模块。

负责构建 FastAPI 主应用实例，配置全局基础元数据（服务名称、版本号与描述），
注册业务 API，并在首页提供同源客服聊天页面及其专用静态资源。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.core.logging import setup_logging

# 在应用创建前初始化业务日志，确保所有模块的 logger 都使用统一格式输出。
setup_logging()

# 实例化主 FastAPI 应用，注入全局服务元数据
app = FastAPI(
    title="MewHelp E-Commerce Customer Service",
    description="MewHelp 电商智能客服系统与售后工单结构化提取服务 API",
    version="0.1.0",
)

# 注册售后诉求提取接口路由
app.include_router(extract_router)
# 注册客服多轮对话流式接口路由
app.include_router(chat_router)

# 仅公开前端专用目录，不能挂载项目根目录，避免泄露 .env 等配置。
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse, include_in_schema=False)
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
