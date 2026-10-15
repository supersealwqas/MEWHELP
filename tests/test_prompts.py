"""Prompt 模板与 LLM 客户端工厂单元测试模块。

覆盖客服对话 Prompt 模板 (CUSTOMER_SERVICE_PROMPT)、售后信息提取 Prompt 模板 (EXTRACTION_PROMPT)
以及大语言模型客户端工厂 (get_llm) 的属性配置、变量校验与消息格式化行为。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core import CUSTOMER_SERVICE_PROMPT, EXTRACTION_PROMPT, get_llm
from app.core.llm import get_llm as direct_get_llm
from app.core.prompts import (
    CUSTOMER_SERVICE_PROMPT as DIRECT_CUSTOMER_SERVICE_PROMPT,
    EXTRACTION_PROMPT as DIRECT_EXTRACTION_PROMPT,
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
    """售后信息提取 Prompt 模板 (EXTRACTION_PROMPT) 测试类。"""

    def test_extraction_prompt_type_and_variables(self) -> None:
        """测试售后提取 Prompt 是 ChatPromptTemplate 实例且输入变量仅包含 'text'。

        返回值:
            None: 无返回值，断言模板类型与输入变量定义。
        """
        assert isinstance(EXTRACTION_PROMPT, ChatPromptTemplate)
        assert "text" in EXTRACTION_PROMPT.input_variables
        assert EXTRACTION_PROMPT.input_variables == ["text"]

    def test_extraction_prompt_formatting(self) -> None:
        """测试传入待分析文本格式化提取 Prompt，产出包含 System 与 Human 的消息列表。

        返回值:
            None: 无返回值，断言消息列表结构及用户输入正确填入 HumanMessage。
        """
        user_raw_text = "前天买的衣服尺码偏小了，订单号 ORD-8888，希望退款。"
        prompt_value = EXTRACTION_PROMPT.format_prompt(text=user_raw_text)
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
        system_template = EXTRACTION_PROMPT.messages[0]
        prompt_text = system_template.prompt.template  # type: ignore[attr-defined]

        # 验证 5 种诉求类型的枚举约束
        for req_type in ["退款", "换货", "维修", "投诉", "其他"]:
            assert req_type in prompt_text

        # 验证对 order_id 未提供时的说明 (None / null)
        assert any(keyword in prompt_text for keyword in ["None", "null", "未提及", "未提供"])


class TestGetLLM:
    """大模型客户端工厂函数 (get_llm) 测试类。"""

    def test_get_llm_default_configuration(self) -> None:
        """测试 get_llm 工厂在默认参数下返回配置正确的 ChatOpenAI 实例。

        返回值:
            None: 无返回值，断言 model, base_url, api_key, temperature 与全局配置一致。
        """
        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == settings.chat_model
        assert str(llm.openai_api_base).rstrip("/") == settings.chat_base_url.rstrip("/")
        assert llm.openai_api_key.get_secret_value() == settings.chat_api_key
        assert llm.temperature == 0.0

    def test_get_llm_custom_parameters(self) -> None:
        """测试 get_llm 允许自定义 temperature 与额外关键字参数。

        返回值:
            None: 无返回值，断言自定义参数被正确透传至 ChatOpenAI 实例。
        """
        custom_temp = 0.7
        llm = get_llm(temperature=custom_temp, max_tokens=512, request_timeout=15.0)
        assert isinstance(llm, ChatOpenAI)
        assert llm.temperature == custom_temp
        assert llm.max_tokens == 512
        assert llm.request_timeout == 15.0


class TestSubmoduleExports:
    """导出一致性测试类。"""

    def test_submodule_exports_match_package_exports(self) -> None:
        """测试 app.core 包层导出的对象与子模块直接导出的对象完全一致。

        返回值:
            None: 无返回值，断言对象引用一致。
        """
        assert CUSTOMER_SERVICE_PROMPT is DIRECT_CUSTOMER_SERVICE_PROMPT
        assert EXTRACTION_PROMPT is DIRECT_EXTRACTION_PROMPT
        assert get_llm is direct_get_llm
