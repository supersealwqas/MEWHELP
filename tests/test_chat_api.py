"""原聊天 API 的 SSE 契约与数据库服务边界回归，不连接真实模型或数据库。"""

import json
import pytest
from fastapi.testclient import TestClient
from app.schemas.workflow import AgentDecision
from app.workflow.model import WorkflowOutputError
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
    chunks = ["您", "好", "！我是", "小喵。"]
    model.answers.clear()
    model.answers.append(chunks)
    response = send(client, "我的订单进度")
    assert response.status_code == 200 and "text/event-stream" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache" and response.headers["connection"] == "keep-alive"
    for word in chunks:
        assert "data: " + json.dumps({"delta": word}, ensure_ascii=False) + "\n\n" in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    turn = next(iter(repo.turns.values()))
    assert turn.turn_status == "completed" and turn.messages[-1].content == "您好！我是小喵。"


def test_empty_deltas_filtered(client, offline_chat_runtime):
    """空增量不能进入 SSE。参数：客户端、依赖；返回：None；异常：断言失败。"""
    offline_chat_runtime[-1].answers.clear()
    offline_chat_runtime[-1].answers.append(["", "喵", "", "购"])
    response = send(client)
    assert response.text.count('"delta"') == 2 and '"delta": ""' not in response.text


def test_second_request_contains_completed_history(client, offline_chat_runtime):
    """丢弃数据库历史会使第二次模型输入错误。参数：客户端、依赖；返回：None；异常：断言失败。"""
    model = offline_chat_runtime[-1]
    model.intents.append("订单")
    model.decisions.append(AgentDecision(kind="answer"))
    model.answers.clear()
    model.answers.extend([["你好，我是小喵"], ["你是张三"]])
    send(client, "我是张三")
    send(client, "我是谁")
    inputs = [row for row in model.inputs if row["stage"] == "classify"][-1]["messages"]
    assert inputs[:-1] == [{"role": "user", "content": "我是张三"},
                      {"role": "assistant", "content": "你好，我是小喵"}, {"role": "user", "content": "我是谁"}]
    assert inputs[-1]["role"] == "user" and inputs[-1]["content"].startswith("上下文资料")


@pytest.mark.parametrize("chunks", [[RuntimeError("SECRET")]])
def test_error_has_no_done_and_preserves_failed_audit(client, offline_chat_runtime, chunks):
    """未知程序故障不得伪装完成。参数：客户端、依赖、分片；返回：None；异常：断言失败。"""
    _, repo, _, model = offline_chat_runtime
    model.answers.clear()
    model.answers.append(chunks)
    response = send(client)
    assert "event: error" in response.text and "[DONE]" not in response.text and "SECRET" not in response.text
    turn = next(iter(repo.turns.values()))
    assert turn.turn_status == "failed" and len(turn.messages) == 1


@pytest.mark.parametrize("chunks", [["部分", TimeoutError("SECRET")], [WorkflowOutputError("空正文")]])
def test_upstream_failure_finishes_with_saved_safe_reply(client, offline_chat_runtime, chunks):
    """已知上游故障返回可保存说明，SSE与数据库正文一致。参数：客户端、依赖、分片；返回：None；异常：断言失败。"""
    _, repo, _, model = offline_chat_runtime
    model.answers.clear()
    model.answers.append(chunks)
    response = send(client)
    assert "event: error" not in response.text and response.text.endswith("data: [DONE]\n\n")
    assert "SECRET" not in response.text and "answer_unavailable" in response.text
    deltas = [json.loads(block.removeprefix("data: "))["delta"] for block in response.text.split("\n\n")
              if block.startswith('data: {"delta":')]
    turn = next(iter(repo.turns.values()))
    assert turn.turn_status == "completed" and turn.messages[-1].content == "".join(deltas)
    assert "未能完整生成" in turn.messages[-1].content


def test_model_factory_failure_is_safe(client, offline_chat_runtime):
    """初始化错误也被编排捕获。参数：客户端、依赖；返回：None；异常：断言失败。"""
    offline_chat_runtime[-1].intents.clear()
    offline_chat_runtime[-1].intents.append(ValueError("SECRET"))
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
    offline_chat_runtime[-1].decisions.appendleft(AgentDecision(kind="tools", tool_calls=[
        {"name": "query_logistics", "id": "c1", "args": {"tracking_no": "MOCK213502378238"}}]))
    response = send(client, "订单 1001 的物流到哪了")
    assert response.text.count("event: tool_status") == 2
    assert response.text.index("event: tool_status") < response.text.index('"delta"')
    assert '"is_mock": true' in response.text


def test_dependent_tools_share_one_sse_reply(client, offline_chat_runtime):
    """同一HTTP请求传输两步工具状态，最后只发送一次完成帧。

    参数：client为真实路由客户端；offline_chat_runtime为外部边界替身。
    返回：None，通过断言核对状态顺序、工具配对与已保存回答。
    异常：断言失败。
    """
    _, repository, _, model = offline_chat_runtime
    model.decisions.clear()
    model.decisions.extend([
        AgentDecision(kind="tools", tool_calls=[{"id": "order", "name": "query_order", "args": {"order_id": "1001"}}]),
        AgentDecision(kind="tools", tool_calls=[{"id": "logistics", "name": "query_logistics",
                                                 "args": {"tracking_no": "MOCK213502378238"}}]),
        AgentDecision(kind="answer"),
    ])
    response = send(client, "订单1001的物流到哪了")
    assert response.status_code == 200
    assert response.text.count("event: tool_status") == 4
    assert response.text.index('"name": "query_order"') < response.text.index('"name": "query_logistics"')
    assert response.text.count("data: [DONE]") == 1 and response.text.endswith("data: [DONE]\n\n")
    turn = next(iter(repository.turns.values()))
    assert [message.role for message in turn.messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert turn.turn_status == "completed"


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
