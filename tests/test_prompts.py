"""客服与售后提取 Prompt 模板单元测试模块。

覆盖客服对话 Prompt 模板 (CUSTOMER_SERVICE_PROMPT)、售后信息提取 Prompt 模板 (EXTRACT_PROMPT)
的变量校验、消息格式化与导出一致性行为。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate

from app.core import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT
from app.core.prompts import (
    CUSTOMER_SERVICE_PROMPT as DIRECT_CUSTOMER_SERVICE_PROMPT,
    EXTRACT_PROMPT as DIRECT_EXTRACT_PROMPT,
)


class TestCustomerServicePrompt:
    """客服对话 Prompt 模板 (CUSTOMER_SERVICE_PROMPT) 测试类。"""

    def test_customer_service_prompt_type_and_variables(self) -> None:
        """测试客服 Prompt 是 ChatPromptTemplate 实例且输入变量仅包含 'history'。

        返回值:
            None: 无返回值，断言模板类型与输入变量定义。
        """
        assert isinstance(CUSTOMER_SERVICE_PROMPT, ChatPromptTemplate)
        assert "history" in CUSTOMER_SERVICE_PROMPT.input_variables
        assert CUSTOMER_SERVICE_PROMPT.input_variables == ["history"]

    def test_customer_service_prompt_formatting(self) -> None:
        """测试使用历史消息列表格式化客服 Prompt，产出符合预期的 ChatPromptValue。

        返回值:
            None: 无返回值，断言消息列表包含开头的 SystemMessage 与后续的历史消息。
        """
        history_messages = [
            HumanMessage(content="你好，我的包裹还没到。"),
            AIMessage(content="您好，请问方便提供一下您的订单编号或手机号吗？"),
            HumanMessage(content="单号是 ORD-2026-999。"),
        ]

        prompt_value = CUSTOMER_SERVICE_PROMPT.format_prompt(history=history_messages)
        assert isinstance(prompt_value, ChatPromptValue)

        messages = prompt_value.to_messages()
        # 消息总数应为 1 (SystemMessage) + 3 (history 消息) = 4
        assert len(messages) == 4

        # 首条消息必须为 SystemMessage
        assert isinstance(messages[0], SystemMessage)
        assert len(messages[0].content) > 0

        # 后续消息严格对应传入的 history
        assert messages[1] == history_messages[0]
        assert messages[2] == history_messages[1]
        assert messages[3] == history_messages[2]

    def test_customer_service_prompt_formatting_empty_history(self) -> None:
        """测试传入空历史列表时的格式化行为，应仅包含单条 SystemMessage。

        返回值:
            None: 无返回值，断言空历史仅产出系统人设消息。
        """
        prompt_value = CUSTOMER_SERVICE_PROMPT.format_prompt(history=[])
        messages = prompt_value.to_messages()
        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)

    def test_refund_timing_rule_distinguishes_processing_and_arrival(self) -> None:
        """测试提示词区分处理时限与到账时效，避免把不承诺到账扩大成不承诺审核。

        返回值:
            None: 无返回值，断言两条提示词都保留处理时限与到账时效的分离约束。
        """
        from app.core.prompts import CUSTOMER_SERVICE_SYSTEM, FINAL_ANSWER_SYSTEM

        assert "处理时限" in CUSTOMER_SERVICE_SYSTEM
        assert "到账时间" in CUSTOMER_SERVICE_SYSTEM
        assert "退款审核要几天" in CUSTOMER_SERVICE_SYSTEM
        assert "不能截成" in CUSTOMER_SERVICE_SYSTEM
        assert "处理/审核时限" in FINAL_ANSWER_SYSTEM
        assert "不能因为不承诺到账就否认知识库给出的处理时限" in FINAL_ANSWER_SYSTEM

    def test_customer_service_system_prompt_content(self) -> None:
        """测试客服 System Prompt 文本包含官方人设、严禁臆造虚假单号、礼貌引导等核心业务约束。

        返回值:
            None: 无返回值，断言关键约束词汇存在于 System 提示词中。
        """
        # 获取第一条 SystemMessage 模板
        system_template = CUSTOMER_SERVICE_PROMPT.messages[0]
        prompt_text = system_template.prompt.template  # type: ignore[attr-defined]

        # 验证核心业务人设与约束关键词
        assert "客服" in prompt_text
        # 严禁编造/臆造虚假信息的红线约束
        assert any(keyword in prompt_text for keyword in ["臆造", "捏造", "虚构", "杜绝", "真实", "编造"])
        # 偏离主题时的礼貌拉回/引导
        assert any(keyword in prompt_text for keyword in ["引导", "无关", "电商", "购物", "拉回", "职责之外"])


class TestExtractionPrompt:
    """售后信息提取 Prompt 模板 (EXTRACT_PROMPT) 测试类。"""

    def test_extraction_prompt_type_and_variables(self) -> None:
        """测试售后提取 Prompt 是 ChatPromptTemplate 实例且输入变量仅包含 'text'。

        返回值:
            None: 无返回值，断言模板类型与输入变量定义。
        """
        assert isinstance(EXTRACT_PROMPT, ChatPromptTemplate)
        assert "text" in EXTRACT_PROMPT.input_variables
        assert EXTRACT_PROMPT.input_variables == ["text"]

    def test_extraction_prompt_formatting(self) -> None:
        """测试传入待分析文本格式化提取 Prompt，产出包含 System 与 Human 的消息列表。

        返回值:
            None: 无返回值，断言消息列表结构及用户输入正确填入 HumanMessage。
        """
        user_raw_text = "前天买的衣服尺码偏小了，订单号 ORD-8888，希望退款。"
        prompt_value = EXTRACT_PROMPT.format_prompt(text=user_raw_text)
        assert isinstance(prompt_value, ChatPromptValue)

        messages = prompt_value.to_messages()
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == user_raw_text

    def test_extraction_system_prompt_content(self) -> None:
        """测试提取 System Prompt 包含售后字段规范（单号未提及输出 None、诉求枚举、期望方案）。

        返回值:
            None: 无返回值，断言提取说明包含 5 种售后枚举类型与字段要求。
        """
        system_template = EXTRACT_PROMPT.messages[0]
        prompt_text = system_template.prompt.template  # type: ignore[attr-defined]

        # 验证 5 种诉求类型的枚举约束
        for req_type in ["退款", "换货", "维修", "投诉", "其他"]:
            assert req_type in prompt_text

        # 验证对 order_id 未提供时的说明 (None / null)
        assert any(keyword in prompt_text for keyword in ["None", "null", "未提及", "未提供"])


class TestSubmoduleExports:
    """导出一致性测试类。"""

    def test_submodule_exports_match_package_exports(self) -> None:
        """测试 app.core 包层导出的对象与子模块直接导出的对象完全一致。

        返回值:
            None: 无返回值，断言对象引用一致。
        """
        assert CUSTOMER_SERVICE_PROMPT is DIRECT_CUSTOMER_SERVICE_PROMPT
        assert EXTRACT_PROMPT is DIRECT_EXTRACT_PROMPT
