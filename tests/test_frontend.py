"""聊天前端入口测试：验证页面、静态资源、目录隔离及原有 API 契约。

所有用例只读取本地页面或 OpenAPI，不调用真实模型。
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


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


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("styles.css", "text/css"),
        ("chat.js", "javascript"),
        ("chat-api.mjs", "javascript"),
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
