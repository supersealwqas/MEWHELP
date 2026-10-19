"""原聊天 API 的 SSE 契约与数据库服务边界回归，不连接真实模型或数据库。"""

import json
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from app.main import app


@pytest.fixture
def client():
    """启动使用隔离运行时的真实应用。参数：无；返回：客户端；异常：启动失败。"""
    with TestClient(app) as test_client:
        yield test_client


def send(client, message="你好", session_id="api-session"):
    """发送一次同源聊天请求。参数：客户端、问题、会话；返回：响应；异常：HTTP 错误。"""
    return client.post("/api/chat", json={"session_id": session_id, "message": message})


def test_message_length_limit_is_validated_before_service(client, offline_chat_runtime):
    """超出4000字符时不进入业务服务。参数：客户端、隔离依赖；返回：None；异常：断言失败。"""
    response = send(client, "好" * 4001)
    assert response.status_code == 422
    assert not offline_chat_runtime[1].turns and not offline_chat_runtime[-1].inputs


def test_chat_sse_normal_multi_chunk_stream(client, offline_chat_runtime):
    """验证真实增量和完成帧，数据库完成在 done 之前。参数：客户端、依赖；返回：None；异常：断言失败。"""
    _, repo, _, model = offline_chat_runtime
    model.chunks = ["您", "好", "！我是", "小喵。"]
    response = send(client)
    assert response.status_code == 200 and "text/event-stream" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache" and response.headers["connection"] == "keep-alive"
    for word in model.chunks:
        assert "data: " + json.dumps({"delta": word}, ensure_ascii=False) + "\n\n" in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    turn = next(iter(repo.turns.values()))
    assert turn.turn_status == "completed" and turn.messages[-1].content == "您好！我是小喵。"


def test_empty_deltas_filtered(client, offline_chat_runtime):
    """空增量不能进入 SSE。参数：客户端、依赖；返回：None；异常：断言失败。"""
    offline_chat_runtime[-1].chunks = ["", "喵", "", "购"]
    response = send(client)
    assert response.text.count('"delta"') == 2 and '"delta": ""' not in response.text


def test_second_request_contains_completed_history(client, offline_chat_runtime):
    """丢弃数据库历史会使第二次模型输入错误。参数：客户端、依赖；返回：None；异常：断言失败。"""
    model = offline_chat_runtime[-1]
    model.chunks = ["你好，我是小喵"]
    send(client, "我是张三")
    send(client, "我是谁")
    inputs = model.inputs[2][1]
    assert [m.content for m in inputs[1:]] == ["我是张三", "你好，我是小喵", "我是谁"]
    assert isinstance(inputs[1], HumanMessage)


@pytest.mark.parametrize("chunks", [[RuntimeError("SECRET")], ["部分", TimeoutError("SECRET")], [""]])
def test_error_has_no_done_and_preserves_failed_audit(client, offline_chat_runtime, chunks):
    """上游故障和空回复不得伪装成功。参数：客户端、依赖、分片；返回：None；异常：断言失败。"""
    _, repo, _, model = offline_chat_runtime
    model.chunks = chunks
    response = send(client)
    assert "event: error" in response.text and "[DONE]" not in response.text and "SECRET" not in response.text
    turn = next(iter(repo.turns.values()))
    assert turn.turn_status == "failed" and len(turn.messages) == 1


def test_model_factory_failure_is_safe(client, offline_chat_runtime):
    """初始化错误也被编排捕获。参数：客户端、依赖；返回：None；异常：断言失败。"""
    offline_chat_runtime[-1].decision = ValueError("SECRET")
    response = send(client)
    assert "event: error" in response.text and "SECRET" not in response.text


def test_oversized_current_message_is_not_dropped(client, offline_chat_runtime, monkeypatch):
    """过预算先拒绝，不能以空历史问模型。参数：客户端、依赖、替换器；返回：None；异常：断言失败。"""
    service, repo, _, model = offline_chat_runtime
    monkeypatch.setattr(service.settings, "token_budget", 1)
    response = send(client, "很长问题" * 100)
    assert "消息或工具结果过长" in response.text and not repo.turns and not model.inputs


def test_tool_named_event_precedes_text_and_is_not_text(client, offline_chat_runtime):
    """工具状态必须独立于回答流。参数：客户端、依赖；返回：None；异常：断言失败。"""
    offline_chat_runtime[-1].decision = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "id": "c1", "args": {"tracking_no": "MOCK213502378238"}}])
    response = send(client, "订单 1001 的物流到哪了")
    assert response.text.count("event: tool_status") == 2
    assert response.text.index("event: tool_status") < response.text.index('"delta"')
    assert '"is_mock": true' in response.text


@pytest.mark.parametrize("payload", [{}, {"session_id": "s"}, {"message": "hello"},
    *[{"session_id": v, "message": "hello"} for v in ["", "   ", "\t\n", "x" * 65]],
    *[{"session_id": "s", "message": v} for v in ["", "   ", "\r\n"]],
    {"session_id": "s", "message": "hello", "user_id": " "}])
def test_invalid_request_rejected(client, payload):
    """缺字段、空白和过长 ID 不进入数据库。参数：客户端、请求；返回：None；异常：断言失败。"""
    assert client.post("/api/chat", json=payload).status_code == 422


def test_route_and_package_export():
    """保留原路由注册及包导出。参数：无；返回：None；异常：断言失败。"""
    from app.api import chat_router
    from app.api.chat import router
    assert router is chat_router and "post" in app.openapi()["paths"]["/api/chat"]
