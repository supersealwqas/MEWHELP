"""客服多轮对话 SSE 流式输出 API 端点测试模块。

本模块针对 POST /api/chat 接口进行完备的单元测试，涵盖：
1. 正常多 Token 流式推送（验证 HTTP 200、text/event-stream 响应头、逐 chunk 增量数据帧与 [DONE] 完成帧）；
2. 会话历史自动回写（验证请求处理完成后 memory 中记录了完整的用户输入与合并的 AI 回答）；
3. 多轮对话上下文继承（验证第二轮请求时，传给 chain.astream 的 history 包含上一轮的用户与 AI 消息）；
4. 上游模型异常安全处理（验证立即异常与流式中途异常时推送 event: error 帧，且不写回不完整历史）；
5. 请求体参数边界校验（缺少字段、空字符或纯空白字符返回 HTTP 422）；
6. 路由注册与 OpenAPI 契约核验。
所有测试通过 Mock 阻断真实大模型调用，确保单测执行的轻量与幂等。
"""

import json
from typing import Any, Generator
import uuid
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables.base import RunnableSequence

from app.core.memory import memory
from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """创建 FastAPI 测试客户端 Fixture。

    提供与 FastAPI 应用程序交互的轻量级同步客户端实例。

    返回值:
        Generator[TestClient, None, None]: TestClient 实例生成器
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id() -> Generator[str, None, None]:
    """生成唯一的会话标识符，并在测试完成后自动清理内存中的会话数据。

    返回值:
        Generator[str, None, None]: 唯一的会话 ID 字符串
    """
    sid = f"test_chat_sess_{uuid.uuid4().hex[:8]}"
    try:
        yield sid
    finally:
        # 测试结束后清理当前测试产生的会话历史，保证单测之间隔离
        memory.clear(sid)


class TestChatSSEStreamOutput:
    """POST /api/chat 正常流式输出测试类。"""

    def test_chat_sse_normal_multi_chunk_stream(
        self, client: TestClient, session_id: str
    ) -> None:
        """测试正常多 token 流式输出，验证 HTTP 状态码、响应头、逐帧 JSON delta 及完成帧。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            session_id (str): 隔离的会话 ID
        """
        # 准备 Mock 的增量 chunk 序列
        mock_chunks = [
            AIMessageChunk(content="您"),
            AIMessageChunk(content="好"),
            AIMessageChunk(content="！我是"),
            AIMessageChunk(content="小喵。"),
        ]

        def make_stream(chunks: list[AIMessageChunk]):
            async def _astream(*args: Any, **kwargs: Any):
                for chunk in chunks:
                    yield chunk

            return _astream

        with patch.object(RunnableSequence, "astream", side_effect=make_stream(mock_chunks)):
            payload = {"session_id": session_id, "message": "你好"}
            response = client.post("/api/chat", json=payload)

            # 验证 HTTP 状态码与响应头
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            assert response.headers.get("cache-control") == "no-cache"
            assert response.headers.get("connection") == "keep-alive"

            # 验证 SSE 文本流内容
            text = response.text
            expected_chunks = [
                'data: {"delta": "您"}\n\n',
                'data: {"delta": "好"}\n\n',
                'data: {"delta": "！我是"}\n\n',
                'data: {"delta": "小喵。"}\n\n',
                "data: [DONE]\n\n",
            ]
            for expected in expected_chunks:
                assert expected in text

            # 验证 [DONE] 为最后一帧
            assert text.rstrip().endswith("data: [DONE]")

    def test_chat_sse_empty_chunk_ignored(
        self, client: TestClient, session_id: str
    ) -> None:
        """测试模型返回空内容的 chunk 时不会产出多余的空增量帧。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            session_id (str): 隔离的会话 ID
        """
        mock_chunks = [
            AIMessageChunk(content=""),
            AIMessageChunk(content="喵"),
            AIMessageChunk(content=""),
            AIMessageChunk(content="购"),
        ]

        def make_stream(chunks: list[AIMessageChunk]):
            async def _astream(*args: Any, **kwargs: Any):
                for chunk in chunks:
                    yield chunk

            return _astream

        with patch.object(RunnableSequence, "astream", side_effect=make_stream(mock_chunks)):
            payload = {"session_id": session_id, "message": "介绍一下"}
            response = client.post("/api/chat", json=payload)

            assert response.status_code == 200
            text = response.text
            # 确认只有非空内容被序列化为增量帧
            assert 'data: {"delta": "喵"}\n\n' in text
            assert 'data: {"delta": "购"}\n\n' in text
            assert 'data: {"delta": ""}\n\n' not in text
            assert "data: [DONE]\n\n" in text


class TestChatSSEMemoryHistory:
    """POST /api/chat 会话历史管理与上下文继承测试类。"""

    def test_chat_sse_writes_memory_history(
        self, client: TestClient, session_id: str
    ) -> None:
        """测试流式结束后，用户提问与完整拼接的 AI 回答被正确回写进 memory。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            session_id (str): 隔离的会话 ID
        """
        mock_chunks = [
            AIMessageChunk(content="喵购"),
            AIMessageChunk(content="为您"),
            AIMessageChunk(content="服务"),
        ]

        def make_stream(chunks: list[AIMessageChunk]):
            async def _astream(*args: Any, **kwargs: Any):
                for chunk in chunks:
                    yield chunk

            return _astream

        with patch.object(RunnableSequence, "astream", side_effect=make_stream(mock_chunks)):
            payload = {"session_id": session_id, "message": "人工客服在吗"}
            response = client.post("/api/chat", json=payload)
            assert response.status_code == 200

            # 校验全局 memory 中存储的消息列表
            history = memory.get_messages(session_id)
            assert len(history) == 2
            # 第一条为用户输入的提问
            assert isinstance(history[0], HumanMessage)
            assert history[0].content == "人工客服在吗"
            # 第二条为流中拼接完整的 AI 回答
            assert isinstance(history[1], AIMessage)
            assert history[1].content == "喵购为您服务"

    def test_chat_sse_multi_turn_context_inheritance(
        self, client: TestClient, session_id: str
    ) -> None:
        """测试连续两轮对话交互，第二轮传递给 chain.astream 的 history 完整继承第一轮上下文。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            session_id (str): 隔离的会话 ID
        """
        captured_histories: list[list[Any]] = []

        def make_recording_stream(response_text: str):
            async def _astream(input_dict: dict[str, Any], *args: Any, **kwargs: Any):
                # 记录每次调用链接收到的 history 入参
                captured_histories.append(input_dict.get("history", []))
                yield AIMessageChunk(content=response_text)

            return _astream

        # 第一轮提问
        with patch.object(
            RunnableSequence,
            "astream",
            side_effect=make_recording_stream("订单一般在 24 小时内发出。"),
        ):
            res1 = client.post(
                "/api/chat",
                json={"session_id": session_id, "message": "我的订单什么时候发货？"},
            )
            assert res1.status_code == 200
            assert "24 小时内发出" in res1.text

        # 验证第一轮传给 prompt 的 history 仅有当前首轮消息
        assert len(captured_histories) == 1
        round1_history = captured_histories[0]
        assert len(round1_history) == 1
        assert isinstance(round1_history[0], HumanMessage)
        assert round1_history[0].content == "我的订单什么时候发货？"

        # 第二轮提问 (继承同一 session_id)
        with patch.object(
            RunnableSequence,
            "astream",
            side_effect=make_recording_stream("可以在发货前修改收货地址。"),
        ):
            res2 = client.post(
                "/api/chat",
                json={"session_id": session_id, "message": "可以改地址吗？"},
            )
            assert res2.status_code == 200
            assert "修改收货地址" in res2.text

        # 验证第二轮传给 prompt 的 history 包含第一轮问答与当前提问
        assert len(captured_histories) == 2
        round2_history = captured_histories[1]
        assert len(round2_history) == 3
        # 第一轮用户问
        assert isinstance(round2_history[0], HumanMessage)
        assert round2_history[0].content == "我的订单什么时候发货？"
        # 第一轮 AI 答
        assert isinstance(round2_history[1], AIMessage)
        assert round2_history[1].content == "订单一般在 24 小时内发出。"
        # 第二轮用户问
        assert isinstance(round2_history[2], HumanMessage)
        assert round2_history[2].content == "可以改地址吗？"


class TestChatSSEErrorHandling:
    """POST /api/chat 上游模型异常帧输出测试类。"""

    def test_chat_sse_upstream_immediate_exception(
        self, client: TestClient, session_id: str
    ) -> None:
        """测试模型在开始流式输出前抛出异常时，向客户端推送 event: error 帧且不回写 AI 消息。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            session_id (str): 隔离的会话 ID
        """
        def make_error_stream(err_msg: str):
            async def _astream(*args: Any, **kwargs: Any):
                raise ConnectionResetError(err_msg)
                yield  # 使函数成为异步生成器

            return _astream

        with patch.object(
            RunnableSequence,
            "astream",
            side_effect=make_error_stream("上游连接已断开"),
        ):
            payload = {"session_id": session_id, "message": "你好"}
            response = client.post("/api/chat", json=payload)

            # 流式响应已建立，返回 200
            assert response.status_code == 200
            text = response.text
            # 必须包含规范的错误事件帧
            assert "event: error\n" in text
            expected_data = json.dumps({"error": "上游模型调用失败: 上游连接已断开"}, ensure_ascii=False)
            assert f"data: {expected_data}\n\n" in text
            # 异常情况下不应该输出完成标志帧
            assert "data: [DONE]" not in text

            # 验证 memory 中不记录失败的 AI 消息（仅有用户的提问）
            history = memory.get_messages(session_id)
            assert len(history) == 1
            assert isinstance(history[0], HumanMessage)

    def test_chat_sse_upstream_midstream_exception(
        self, client: TestClient, session_id: str
    ) -> None:
        """测试模型在输出部分 token 后中途发生网络异常，输出已生成部分后追加 event: error 帧。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            session_id (str): 隔离的会话 ID
        """
        def make_midstream_error():
            async def _astream(*args: Any, **kwargs: Any):
                yield AIMessageChunk(content="正在查询中")
                raise TimeoutError("上游模型响应超时")

            return _astream

        with patch.object(
            RunnableSequence,
            "astream",
            side_effect=make_midstream_error(),
        ):
            payload = {"session_id": session_id, "message": "查下物流"}
            response = client.post("/api/chat", json=payload)

            assert response.status_code == 200
            text = response.text
            # 确认中途产生的文本被输出
            assert 'data: {"delta": "正在查询中"}\n\n' in text
            # 确认中途崩溃被捕获并推送错误帧
            assert "event: error\n" in text
            expected_data = json.dumps({"error": "上游模型调用失败: 上游模型响应超时"}, ensure_ascii=False)
            assert f"data: {expected_data}\n\n" in text
            assert "data: [DONE]" not in text

            # 验证异常中断未向 memory 写入不完整的 AI 回答
            history = memory.get_messages(session_id)
            assert len(history) == 1
            assert isinstance(history[0], HumanMessage)


class TestChatSSEValidationAndRouting:
    """POST /api/chat 请求体校验与路由注册测试类。"""

    def test_chat_missing_required_fields(self, client: TestClient) -> None:
        """测试缺失必填字段时返回 HTTP 422 校验失败。

        参数:
            client (TestClient): FastAPI 测试客户端实例
        """
        # 缺失 message
        res1 = client.post("/api/chat", json={"session_id": "s1"})
        assert res1.status_code == 422

        # 缺失 session_id
        res2 = client.post("/api/chat", json={"message": "hello"})
        assert res2.status_code == 422

        # 空请求体
        res3 = client.post("/api/chat", json={})
        assert res3.status_code == 422

    @pytest.mark.parametrize("empty_val", ["", "   ", "\t\n  "])
    def test_chat_empty_or_whitespace_session_id_rejected(
        self, client: TestClient, empty_val: str
    ) -> None:
        """测试 session_id 为空或纯空白字符时返回 HTTP 422。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            empty_val (str): 空或纯空白字符串
        """
        res = client.post("/api/chat", json={"session_id": empty_val, "message": "hello"})
        assert res.status_code == 422

    @pytest.mark.parametrize("empty_val", ["", "   ", "\r\n\t "])
    def test_chat_empty_or_whitespace_message_rejected(
        self, client: TestClient, empty_val: str
    ) -> None:
        """测试 message 为空或纯空白字符时返回 HTTP 422。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            empty_val (str): 空或纯空白字符串
        """
        res = client.post("/api/chat", json={"session_id": "sess_1", "message": empty_val})
        assert res.status_code == 422

    def test_chat_route_registered(self) -> None:
        """测试 POST /api/chat 路由已被正确挂载在 FastAPI app 中。"""
        paths = app.openapi()["paths"]
        assert "/api/chat" in paths
        assert "post" in paths["/api/chat"]


class TestSubmoduleExports:
    """API 模块导出一致性测试类。"""

    def test_api_package_exports_chat_router(self) -> None:
        """测试 app.api 包层导出的 chat_router 与子模块 chat.router 一致。"""
        from app.api import chat_router as package_chat_router
        from app.api.chat import chat_router as direct_chat_router, router as direct_router

        assert package_chat_router is direct_chat_router
        assert package_chat_router is direct_router
