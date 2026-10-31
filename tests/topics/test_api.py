"""主题HTTP接口的关闭、后台占用和状态事实验证。"""

from contextlib import asynccontextmanager
import asyncio
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient
import pytest

from tests.topics.test_batch import MemoryRepository, make_service, response


def test_disabled_topics_returns_explicit_503():
    """关闭模式不能冒充空分布。参数：无；返回：None；异常：断言失败。"""
    from app.main import create_app
    @asynccontextmanager
    async def runtime():
        """参数：无；返回：占位服务；异常：无。"""
        yield NS()
    with TestClient(create_app(runtime_factory=runtime)) as client:
        for path in ["/api/flywheel/topics", "/api/flywheel/topics/status"]:
            assert client.get(path).status_code == 503
        assert client.post("/api/flywheel/topics/classify", json={"reclassify": False}).status_code == 503


def test_trigger_returns_before_long_http_and_busy_is_409(tmp_path):
    """阻塞HTTP或仅本地布尔占用会失败。参数：目录；返回：None；异常：断言失败。"""
    from app.main import create_app
    repo = MemoryRepository(1)
    async def handler(request):
        """参数：请求；返回：响应；异常：取消。"""
        if request.url.path == "/classify": await asyncio.Future()
        return response(request)
    @asynccontextmanager
    async def runtime():
        """参数：无；返回：共享主题服务；异常：无。"""
        service = make_service(tmp_path, repo, handler)
        try: yield NS(topics=service)
        finally: await service.close()
    with TestClient(create_app(runtime_factory=runtime)) as client:
        assert client.post("/api/flywheel/topics/classify", json={"reclassify": False}).status_code == 202
        assert client.post("/api/flywheel/topics/classify", json={"reclassify": False}).status_code == 409
        assert client.get("/api/flywheel/topics/status").json()["status"] == "running"
        assert client.post("/api/flywheel/topics/classify", json={"reclassify": "false"}).status_code == 422


@pytest.mark.parametrize("url", ["http://example.com", "http://127.0.0.1.evil", "http://user@127.0.0.1", "ftp://127.0.0.1"])
def test_config_rejects_non_loopback_urls(url, offline_settings):
    """前缀校验不能接受外网或凭据URL。参数：URL/配置；返回：None；异常：断言失败。"""
    from app.config import Settings
    with pytest.raises(ValueError):
        Settings(**{**offline_settings.model_dump(), "topic_service_url": url})
