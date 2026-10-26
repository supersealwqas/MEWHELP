"""售后信息结构化提取 API 端点测试模块。

本模块针对 POST /api/extract 接口进行完备的单元测试，涵盖：
1. 正常售后文本结构化提取（包含订单号、诉求类型、期望方案）；
2. 缺失订单号文本提取（order_id 为 None，验证非臆造约束）；
3. 上游大模型异常/超时情况下的 HTTP 502 错误封装；
4. 请求体缺失或非法的 HTTP 422 校验机制（空字符串、纯空白字符、缺少必填字段）；
5. 路由注册与元数据核验。
所有测试通过 Mock 阻断真实网络请求，确保单测执行的轻量与幂等。
"""

from typing import Generator
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables.base import RunnableSequence

from app.api.extract import get_extractor
from app.core.llm import get_chat_model
from app.main import app
from app.schemas.extract import AfterSalesTicket
from tests.workflow.test_model import sdk_runtime, call_response, text_response


async def test_extractor_forces_named_strict_schema_over_real_sdk(sdk_runtime, monkeypatch):
    """售后抽取必须强制调用结果工具，不能继续使用auto。参数：HTTP夹具与补丁；返回：None；异常：断言失败。"""
    monkeypatch.setattr("app.api.extract.get_chat_model", sdk_runtime.factory)
    sdk_runtime.responses.append(call_response("AfterSalesTicket",
        '{"order_id":null,"request_type":"退款","expected_solution":"申请退款"}'))
    result = await get_extractor().ainvoke({"text": "我想退款，没有订单号"})
    assert result.order_id is None and result.request_type.value == "退款"
    request = sdk_runtime.requests[0]
    assert request["tool_choice"] == {"type": "function", "function": {"name": "AfterSalesTicket"}}
    assert request["parallel_tool_calls"] is False
    assert request["tools"][0]["function"]["strict"] is True


async def test_extractor_has_no_text_json_fallback(sdk_runtime, monkeypatch):
    """合法正文JSON也不能绕过工具协议返回售后结果。参数：HTTP夹具与补丁；返回：None；异常：断言失败。"""
    monkeypatch.setattr("app.api.extract.get_chat_model", sdk_runtime.factory)
    sdk_runtime.responses.append(text_response('{"order_id":null,"request_type":"退款","expected_solution":"申请退款"}'))
    assert await get_extractor().ainvoke({"text": "申请退款"}) is None


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """创建 FastAPI 测试客户端 Fixture。

    提供与 FastAPI 应用程序交互的轻量级同步客户端实例。

    返回值:
        Generator[TestClient, None, None]: TestClient 实例生成器
    """
    with TestClient(app) as test_client:
        yield test_client


class TestExtractEndpointSuccess:
    """POST /api/extract 正常提取功能测试类。"""

    def test_extract_success_with_order_id(self, client: TestClient) -> None:
        """测试正常包含订单号的售后文本提取，返回 HTTP 200 及完整结构化字段。

        参数:
            client (TestClient): FastAPI 测试客户端实例
        """
        # 构造预期的上游提取结果对象
        mock_ticket = AfterSalesTicket(
            order_id="MH20260701123",
            request_type="退款",
            expected_solution="商品到货损坏，要求退货退款",
        )

        # Mock RunnableSequence.ainvoke 模拟大模型结构化抽取链异步返回
        with patch.object(RunnableSequence, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = mock_ticket

            payload = {
                "text": "我上周买的猫爬架，订单号 MH20260701123，到货就散架了，我要退款"
            }
            response = client.post("/api/extract", json=payload)

            # 验证 HTTP 状态码与响应体
            assert response.status_code == 200
            data = response.json()
            assert data["order_id"] == "MH20260701123"
            assert data["request_type"] == "退款"
            assert data["expected_solution"] == "商品到货损坏，要求退货退款"

            # 验证上游 chain.ainvoke 被正确调用且入参匹配
            mock_ainvoke.assert_awaited_once_with({"text": payload["text"]})

    def test_extract_success_without_order_id(self, client: TestClient) -> None:
        """测试无订单号的售后文本提取，返回 HTTP 200 且 order_id 为 None。

        参数:
            client (TestClient): FastAPI 测试客户端实例
        """
        # 构造未包含订单号的提取结果，order_id 显式为 None
        mock_ticket = AfterSalesTicket(
            order_id=None,
            request_type="换货",
            expected_solution="衣服尺码偏小，申请更换大一号",
        )

        with patch.object(RunnableSequence, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = mock_ticket

            payload = {
                "text": "昨天收到的毛衣尺码有点偏小，穿不进去，能帮忙换个L码吗？"
            }
            response = client.post("/api/extract", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["order_id"] is None
            assert data["request_type"] == "换货"
            assert data["expected_solution"] == "衣服尺码偏小，申请更换大一号"
            mock_ainvoke.assert_awaited_once_with({"text": payload["text"]})

    @pytest.mark.parametrize(
        ("request_type", "expected_solution"),
        [
            ("维修", "吸尘器电机异响，在保修期内申请免费检修"),
            ("投诉", "承诺次日达但延误三天，客服态度恶劣要求道歉并赔付"),
            ("其他", "咨询发票开具抬头修改流程"),
        ],
    )
    def test_extract_various_request_types(
        self, client: TestClient, request_type: str, expected_solution: str
    ) -> None:
        """测试不同售后枚举类型的正确提取与序列化。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            request_type (str): 诉求枚举类型字符串
            expected_solution (str): 期望方案内容
        """
        mock_ticket = AfterSalesTicket(
            order_id="ORD-998877",
            request_type=request_type,  # type: ignore[arg-type]
            expected_solution=expected_solution,
        )

        with patch.object(RunnableSequence, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = mock_ticket

            response = client.post(
                "/api/extract",
                json={"text": f"订单 ORD-998877 售后诉求：{expected_solution}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["order_id"] == "ORD-998877"
            assert data["request_type"] == request_type
            assert data["expected_solution"] == expected_solution


class TestExtractEndpointUpstreamFailure:
    """POST /api/extract 上游异常处理测试类。"""

    def test_missing_tool_result_returns_502(self, client: TestClient) -> None:
        """验证上游未返回结构化工具结果时使用受控错误响应。

        参数:
            client: FastAPI 测试客户端。
        返回值:
            None: 断言空结果被转换为 HTTP 502。
        异常:
            AssertionError: 接口未安全处理空结果时抛出。
        """
        with patch.object(RunnableSequence, "ainvoke", new_callable=AsyncMock) as invoke:
            invoke.return_value = None
            response = client.post("/api/extract", json={"text": "请帮我退款"})
        assert response.status_code == 502
        assert response.json()["detail"] == "上游模型暂时不可用，请稍后重试"

    def test_extract_factory_failure_returns_502(self, client: TestClient) -> None:
        """验证提取链工厂初始化失败时，接口仍按原约定返回 HTTP 502。

        参数:
            client: FastAPI 测试客户端实例。

        返回值:
            None: 断言初始化错误被接口统一包装。

        异常:
            AssertionError: 错误响应不符合原接口约定时抛出。
        """
        with patch(
            "app.api.extract.get_extractor", side_effect=ValueError("模型参数无效")
        ):
            response = client.post("/api/extract", json={"text": "商品损坏，申请退款"})
        assert response.status_code == 502
        assert response.json()["detail"] == "上游模型暂时不可用，请稍后重试"

    def test_extract_upstream_exception_returns_502(self, client: TestClient) -> None:
        """测试上游模型调用抛出通用异常时，接口转换为 HTTP 502 错误。

        参数:
            client (TestClient): FastAPI 测试客户端实例
        """
        error_message = "Connection reset by peer; sensitive-upstream-detail"

        with patch.object(RunnableSequence, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            # 模拟上游 LLM 调用故障抛出异常
            mock_ainvoke.side_effect = RuntimeError(error_message)

            response = client.post("/api/extract", json={"text": "订单已损坏，申请退货"})

            assert response.status_code == 502
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "上游模型暂时不可用，请稍后重试"
            assert "sensitive-upstream-detail" not in response.text

    def test_extract_upstream_timeout_returns_502(self, client: TestClient) -> None:
        """测试上游模型调用超时时，接口转换为 HTTP 502 错误。

        参数:
            client (TestClient): FastAPI 测试客户端实例
        """
        with patch.object(RunnableSequence, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.side_effect = TimeoutError("Request timed out after 30.0s")

            response = client.post("/api/extract", json={"text": "申请退款"})

            assert response.status_code == 502
            data = response.json()
            assert data["detail"] == "上游模型暂时不可用，请稍后重试"


class TestExtractEndpointValidation:
    """POST /api/extract 请求体参数校验测试类。"""

    @pytest.mark.parametrize("character", ["猫", "🐱", "𠮷"])
    def test_extract_accepts_4000_unicode_code_points(
        self, client: TestClient, character: str
    ) -> None:
        """验证4000码点仍能提取，补充平面字符不能按UTF-16长度误拒绝。

        参数：client HTTP客户端、character 单个Unicode码点。
        返回：None，断言边界输入完整传到提取链。
        异常：AssertionError，输入被拒绝或截断。
        """
        text = character * 4000
        ticket = AfterSalesTicket(request_type="其他", expected_solution="咨询售后")
        with patch.object(RunnableSequence, "ainvoke", new_callable=AsyncMock) as invoke:
            invoke.return_value = ticket
            response = client.post("/api/extract", json={"text": text})
        assert response.status_code == 200
        assert response.json()["expected_solution"] == "咨询售后"
        invoke.assert_awaited_once_with({"text": text})

    @pytest.mark.parametrize("character", ["猫", "🐱", "𠮷"])
    def test_extract_rejects_4001_code_points_before_model_creation(
        self, client: TestClient, character: str
    ) -> None:
        """验证超长请求在HTTP校验阶段被拒绝，绝不创建或调用模型。

        参数：client HTTP客户端、character 单个Unicode码点。
        返回：None，断言422、字段错误及零模型调用。
        异常：AssertionError，超长文本进入模型边界。
        """
        with patch("app.api.extract.get_chat_model") as factory:
            response = client.post("/api/extract", json={"text": character * 4001})
        assert response.status_code == 422
        assert any(error["loc"] == ["body", "text"] for error in response.json()["detail"])
        factory.assert_not_called()

    def test_extract_missing_text_field(self, client: TestClient) -> None:
        """测试缺失 text 必填字段时，返回 HTTP 422 校验失败。

        参数:
            client (TestClient): FastAPI 测试客户端实例
        """
        response = client.post("/api/extract", json={})
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("text" in error["loc"] for error in errors)

    @pytest.mark.parametrize("empty_value", ["", "   ", "\n\t  "])
    def test_extract_empty_or_whitespace_text_rejected(
        self, client: TestClient, empty_value: str
    ) -> None:
        """测试 text 为空字符串或纯空白字符时，返回 HTTP 422 校验失败。

        参数:
            client (TestClient): FastAPI 测试客户端实例
            empty_value (str): 空值或纯空白字符串
        """
        response = client.post("/api/extract", json={"text": empty_value})
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("text" in error["loc"] for error in errors)


class TestAppMetadataAndRouting:
    """FastAPI 应用与路由元数据测试类。"""

    def test_extract_model_has_explicit_output_limit(self) -> None:
        """验证实际提取工厂显式传入输出额度，结构化客户端保留原兼容参数。

        参数：无，使用离线配置仅构造客户端，不访问模型。
        返回：None，断言模型获得1024 tokens的单次输出上限。
        异常：AssertionError，工厂遗漏上限或改变既有参数。
        """
        models = []

        def recording_factory(**kwargs):
            """记录真实模型工厂入参与返回值。参数：kwargs 客户端参数；返回：模型；异常：配置错误。"""
            model = get_chat_model(**kwargs)
            models.append(model)
            return model

        with patch("app.api.extract.get_chat_model", side_effect=recording_factory) as factory:
            chain = get_extractor()
        factory.assert_called_once_with(temperature=0.0, streaming=True, max_tokens=1024)
        assert models[0].max_tokens == 1024
        assert isinstance(chain, RunnableSequence)

    def test_app_title_and_version(self) -> None:
        """测试 FastAPI 应用实例的元数据符合规划规范。"""
        assert app.title == "MewHelp E-Commerce Customer Service"
        assert app.version == "0.1.0"

    def test_extract_route_registered(self) -> None:
        """测试 POST /api/extract 路由已被正确挂载在 FastAPI app 中。"""
        paths = app.openapi()["paths"]
        assert "/api/extract" in paths
        assert "post" in paths["/api/extract"]
