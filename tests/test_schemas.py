"""数据契约与 Schema 校验单元测试模块。

覆盖对话请求模型 (ChatRequest)、流式响应块 (ChatChunk)、售后抽取请求模型 (ExtractRequest)
以及售后工单抽取结构模型 (AfterSalesTicket) 的合法解析与异常边界校验。
"""

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from app.schemas import (
    AfterSalesTicket,
    ChatChunk,
    ChatRequest,
    ExtractRequest,
    RequestType,
)
from app.schemas.extract import (
    AfterSalesTicket as DirectAfterSalesTicket,
    ExtractRequest as DirectExtractRequest,
)
from app.schemas.chat import (
    ChatChunk as DirectChatChunk,
    ChatRequest as DirectChatRequest,
)


class TestChatRequest:
    """测试对话请求模型 (ChatRequest) 的校验行为。"""

    def test_valid_chat_request(self) -> None:
        """测试正常参数下的 ChatRequest 实例化。

        返回值:
            None: 无返回值，通过断言验证字段值正确。
        """
        req = ChatRequest(session_id="session-001", message="请问如何退货？")
        assert req.session_id == "session-001"
        assert req.message == "请问如何退货？"

    @pytest.mark.parametrize("invalid_session_id", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_session_id_rejected(self, invalid_session_id: str) -> None:
        """测试空字符串或纯空白字符的 session_id 被拦截并抛出 ValidationError。

        参数:
            invalid_session_id (str): 非法的空白会话 ID 字符串

        返回值:
            None: 无返回值，通过断言确认抛出校验异常。
        """
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(session_id=invalid_session_id, message="你好")
        errors = exc_info.value.errors()
        # 验证错误定位到 session_id 字段
        assert any(err["loc"] == ("session_id",) for err in errors)

    @pytest.mark.parametrize("invalid_message", ["", "   ", "\r\n  \t"])
    def test_empty_or_whitespace_message_rejected(self, invalid_message: str) -> None:
        """测试空字符串或纯空白字符的 message 被拦截并抛出 ValidationError。

        参数:
            invalid_message (str): 非法的空白消息字符串

        返回值:
            None: 无返回值，通过断言确认抛出校验异常。
        """
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(session_id="session-001", message=invalid_message)
        errors = exc_info.value.errors()
        # 验证错误定位到 message 字段
        assert any(err["loc"] == ("message",) for err in errors)

    def test_missing_fields_rejected(self) -> None:
        """测试缺少必填字段时抛出 ValidationError。

        返回值:
            None: 无返回值，通过断言确认缺少参数时报错。
        """
        with pytest.raises(ValidationError):
            ChatRequest()  # type: ignore[call-arg]


class TestChatChunk:
    """测试流式文本块模型 (ChatChunk) 的构建。"""

    def test_valid_chat_chunk(self) -> None:
        """测试正常参数下的 ChatChunk 实例化。

        返回值:
            None: 无返回值，断言 delta 属性。
        """
        chunk = ChatChunk(delta="您好，很高兴为您服务")
        assert chunk.delta == "您好，很高兴为您服务"

    def test_empty_delta_allowed(self) -> None:
        """测试流式传输结束或保活心跳时的空字符串 delta 合法性。

        返回值:
            None: 无返回值，断言允许空字符串 delta。
        """
        chunk = ChatChunk(delta="")
        assert chunk.delta == ""


class TestExtractRequest:
    """测试售后信息提取请求模型 (ExtractRequest) 的校验行为。"""

    def test_valid_extract_request(self) -> None:
        """测试正常文本内容的 ExtractRequest 实例化。

        返回值:
            None: 无返回值，断言 text 属性。
        """
        req = ExtractRequest(text="我想退掉昨天买的键盘，订单是 20240501001")
        assert req.text == "我想退掉昨天买的键盘，订单是 20240501001"

    @pytest.mark.parametrize("invalid_text", ["", "   ", "\n\t  "])
    def test_empty_or_whitespace_text_rejected(self, invalid_text: str) -> None:
        """测试空文本或纯空白字符被拦截并抛出 ValidationError。

        参数:
            invalid_text (str): 非法的空白输入文本

        返回值:
            None: 无返回值，通过断言确认抛出校验异常。
        """
        with pytest.raises(ValidationError) as exc_info:
            ExtractRequest(text=invalid_text)
        errors = exc_info.value.errors()
        # 验证错误定位到 text 字段
        assert any(err["loc"] == ("text",) for err in errors)

    def test_missing_text_rejected(self) -> None:
        """测试缺失 text 必填字段时抛出 ValidationError。

        返回值:
            None: 无返回值，确认必填校验生效。
        """
        with pytest.raises(ValidationError):
            ExtractRequest()  # type: ignore[call-arg]


class TestAfterSalesTicket:
    """测试售后工单模型 (AfterSalesTicket) 的类型与枚举校验。"""

    @pytest.mark.parametrize("placeholder", ["", "   ", "null", " None ", "N/A", "无"])
    def test_missing_order_placeholders_become_none(self, placeholder: str) -> None:
        """验证模型用占位文本表示缺失订单号时统一返回空值。

        参数:
            placeholder: 不表示真实订单号的模型输出。
        返回值:
            None: 断言订单号归一化结果。
        异常:
            AssertionError: 占位文本被误当作订单号时抛出。
        """
        ticket = AfterSalesTicket(
            order_id=placeholder, request_type="退款", expected_solution="退全款"
        )
        assert ticket.order_id is None

    @pytest.mark.parametrize("solution", ["", " \n\t"])
    def test_blank_solution_is_rejected(self, solution: str) -> None:
        """验证缺少有效期望方案的模型输出被拒绝。

        参数:
            solution: 空字符串或纯空白方案。
        返回值:
            None: 断言触发模型校验错误。
        异常:
            AssertionError: 无效方案被接受时抛出。
        """
        with pytest.raises(ValidationError):
            AfterSalesTicket(request_type="退款", expected_solution=solution)

    def test_ticket_tool_schema_preserves_request_types(self) -> None:
        """验证改用 RequestType 后，模型调用仍能获得全部中文诉求选项。

        参数:
            无。

        返回值:
            None: 断言发往模型的工具定义完整且保留可选订单号。

        异常:
            AssertionError: 工具字段定义与既有接口约定不符时抛出。
        """
        tool = convert_to_openai_tool(AfterSalesTicket)
        parameters = tool["function"]["parameters"]
        assert parameters["properties"]["request_type"]["enum"] == [
            "退款", "换货", "维修", "投诉", "其他"
        ]
        assert "order_id" not in parameters["required"]

    def test_valid_ticket_with_order_id(self) -> None:
        """测试包含订单号的工单正常解析。

        返回值:
            None: 无返回值，断言各字段属性正确。
        """
        ticket = AfterSalesTicket(
            order_id="ORD-20240501-9999",
            request_type="退款",
            expected_solution="商品破损严重，要求全额退款至原支付渠道",
        )
        assert ticket.order_id == "ORD-20240501-9999"
        assert ticket.request_type == "退款"
        assert ticket.expected_solution == "商品破损严重，要求全额退款至原支付渠道"

    def test_valid_ticket_without_order_id(self) -> None:
        """测试未提供订单号时默认为 None。

        返回值:
            None: 无返回值，断言 order_id 缺省为 None。
        """
        ticket = AfterSalesTicket(
            request_type="换货",
            expected_solution="衣服尺码偏小，换大一码 L码",
        )
        assert ticket.order_id is None
        assert ticket.request_type == "换货"
        assert ticket.expected_solution == "衣服尺码偏小，换大一码 L码"

    @pytest.mark.parametrize("valid_type", ["退款", "换货", "维修", "投诉", "其他"])
    def test_all_valid_request_types(self, valid_type: str) -> None:
        """测试支持的所有诉求类型枚举值。

        参数:
            valid_type (str): 合法的诉求类型枚举字符串

        返回值:
            None: 无返回值，断言类型枚举被正确接受。
        """
        ticket = AfterSalesTicket(
            request_type=valid_type,  # type: ignore[arg-type]
            expected_solution="处理诉求",
        )
        assert ticket.request_type == valid_type
        assert ticket.request_type is RequestType(valid_type)
        assert ticket.model_dump(mode="json")["request_type"] == valid_type

    @pytest.mark.parametrize("invalid_type", ["退货", "咨询", "催发货", "unknown", ""])
    def test_invalid_request_type_rejected(self, invalid_type: str) -> None:
        """测试非法诉求类型被严格拦截并抛出 ValidationError。

        参数:
            invalid_type (str): 非法的诉求类型字符串

        返回值:
            None: 无返回值，断言抛出校验异常。
        """
        with pytest.raises(ValidationError) as exc_info:
            AfterSalesTicket(
                request_type=invalid_type,  # type: ignore[arg-type]
                expected_solution="处理诉求",
            )
        errors = exc_info.value.errors()
        # 验证错误定位到 request_type 字段
        assert any(err["loc"] == ("request_type",) for err in errors)

    def test_missing_required_fields_rejected(self) -> None:
        """测试缺失诉求类型或期望解决方案等必填字段时抛出 ValidationError。

        返回值:
            None: 无返回值，确认缺失必填字段报错。
        """
        with pytest.raises(ValidationError):
            AfterSalesTicket(order_id="ORD-123")  # type: ignore[call-arg]


class TestDirectSubmoduleImports:
    """测试直接从子模块导入模型的可用性与一致性。"""

    def test_submodule_classes_match_package_exports(self) -> None:
        """测试直接从 app.schemas.chat 和 app.schemas.extract 导入的模型与顶层包导出一致。

        返回值:
            None: 无返回值，断言导出的类标识符完全一致。
        """
        assert DirectChatRequest is ChatRequest
        assert DirectChatChunk is ChatChunk
        assert DirectExtractRequest is ExtractRequest
        assert DirectAfterSalesTicket is AfterSalesTicket
