"""聊天前端入口测试：验证页面、静态资源、目录隔离及原有 API 契约。

所有用例只读取本地页面或 OpenAPI，不调用真实模型。
"""

from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse

from app.main import app, create_app


def test_chat_homepage() -> None:
    """验证首页直接提供中文聊天界面和本地资源引用。

    参数:
        无。
    返回值:
        None: 断言首页可访问、必要控件存在以及安全响应头有效。
    异常:
        AssertionError: 页面缺失、类型不正确或资源未注册时抛出。
    """
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<html lang="zh-CN">' in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="message-input"' in response.text
    assert 'id="messages"' in response.text
    assert 'src="/static/chat.js"' in response.text
    assert 'href="/static/styles.css"' in response.text
    assert response.headers["cache-control"] == "no-cache"
    assert "connect-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("url", ["/", "/static/index.html", "/flywheel", "/static/flywheel.html", "/rag-eval", "/static/rag-eval.html"])
def test_chat_html_entrypoints_have_same_security_headers(url: str) -> None:
    """验证聊天、审核和评估的 HTML 入口均限制脚本与请求来源。

    参数:
        url: 聊天、审核、评估页面或静态 HTML 的直接访问路径。
    返回值:
        None: 断言所有入口使用一致的内容安全策略和 MIME 保护。
    异常:
        AssertionError: 某个入口缺少关键防护指令时抛出。
    """
    with TestClient(app) as client:
        response = client.get(url)
        homepage = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    policy = response.headers.get("content-security-policy", "")
    assert policy == homepage.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "connect-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert response.headers.get("x-content-type-options") == "nosniff"


@pytest.mark.asyncio
async def test_security_headers_preserve_incremental_stream_and_cancellation() -> None:
    """验证统一响应头不会缓存 SSE 正文，且取消仍能到达流生成器。

    参数:
        无；使用只产生一帧后等待的本地路由，不调用模型或数据库。
    返回值:
        None: 断言首帧立即交付、正文不被改写并且取消可执行清理。
    异常:
        AssertionError: 头部遗漏、首帧被缓存或取消被吞掉时抛出。
        TimeoutError: 等待首帧超过本地测试时限时抛出。
    """
    application = create_app()
    first_chunk = anyio.Event()
    stream_closed = anyio.Event()
    messages = []

    async def chunks():
        """提供可取消数据流。参数：无；返回：异步字节迭代器；异常：透传任务取消。"""
        try:
            yield b"data: first\n\n"
            await anyio.sleep_forever()
        finally:
            stream_closed.set()

    async def stream_page():
        """返回测试 SSE。参数：无；返回：StreamingResponse；异常：无。"""
        return StreamingResponse(chunks(), media_type="text/event-stream")

    async def receive():
        """等待取消。参数：无；返回：本例不产生接收帧；异常：透传任务取消。"""
        await anyio.sleep_forever()

    async def send(message):
        """收集真实输出帧。参数：ASGI 响应帧；返回：None；异常：无。"""
        messages.append(message)
        if message["type"] == "http.response.body":
            first_chunk.set()

    application.add_api_route("/test-stream", stream_page)
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": "/test-stream", "raw_path": b"/test-stream", "root_path": "",
        "query_string": b"", "headers": [], "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(application, scope, receive, send)
            await first_chunk.wait()
            tasks.cancel_scope.cancel()
    assert stream_closed.is_set()
    response_headers = dict(messages[0]["headers"])
    assert response_headers.get(b"x-content-type-options") == b"nosniff"
    assert b"content-security-policy" not in response_headers
    assert messages[1] == {
        "type": "http.response.body", "body": b"data: first\n\n", "more_body": True,
    }


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("styles.css", "text/css"),
        ("chat.js", "javascript"),
        ("chat-api.mjs", "javascript"),
        ("actions.mjs", "javascript"),
        ("markdown.mjs", "javascript"),
        ("vendor/markdown-it.mjs", "javascript"),
        ("mew.svg", "image/svg+xml"),
    ],
)
def test_frontend_assets(filename: str, content_type: str) -> None:
    """验证聊天页引用的本地资源都可加载。

    参数:
        filename: 前端专用目录内的资源文件名。
        content_type: 响应类型必须包含的片段，兼容不同平台的 JavaScript MIME。
    返回值:
        None: 断言资源存在、非空且 MIME 正确。
    异常:
        AssertionError: 资源缺失或被当作错误类型发送时抛出。
    """
    with TestClient(app) as client:
        response = client.get(f"/static/{filename}")
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert response.content


@pytest.mark.parametrize(
    "url",
    [
        "/.env",
        "/static/.env",
        "/static/%2e%2e/config.py",
        "/static/%2e%2e/%2e%2e/.env",
        "/static/does-not-exist.js",
    ],
)
def test_frontend_does_not_expose_project_files(url: str) -> None:
    """验证静态文件入口不能访问项目源码、配置或不存在的文件。

    参数:
        url: 敏感文件路径、路径穿越尝试或缺失资源路径。
    返回值:
        None: 断言所有请求均被拒绝为 404。
    异常:
        AssertionError: 静态资源路由意外泄露项目文件时抛出。
    """
    with TestClient(app) as client:
        response = client.get(url)
    assert response.status_code == 404


def test_frontend_paths_are_independent_of_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证资源路径以模块位置为基准，不依赖进程工作目录。

    参数:
        monkeypatch: 用于临时切换当前目录的夹具。
    返回值:
        None: 断言切换目录后首页与脚本仍可读取。
    异常:
        AssertionError: 资源错误地依赖相对工作目录时抛出。
    """
    # 此测试只需切换目录，不创建文件，避免依赖系统临时目录的写权限。
    monkeypatch.chdir(Path(__file__).resolve().parent)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/chat-api.mjs").status_code == 200


def test_frontend_preserves_api_schema() -> None:
    """验证首页不污染 API 文档，也没有覆盖原有业务端点。

    参数:
        无。
    返回值:
        None: 断言聊天和提取接口继续保留在 OpenAPI 中。
    异常:
        AssertionError: 页面影响业务路由或意外成为 API 契约时抛出。
    """
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    paths = response.json()["paths"]
    assert "post" in paths["/api/chat"]
    assert "post" in paths["/api/extract"]
    assert "/" not in paths
