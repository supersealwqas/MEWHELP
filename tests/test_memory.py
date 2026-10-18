"""内存会话存储与 Token 裁剪单元测试模块。

覆盖客服会话历史内存存储组件 (SessionStore) 与
基于 Token 预算控制的历史裁剪函数 (trim_history) 的核心行为与边界场景。
"""

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.memory import SessionStore

store = SessionStore()
from app.core import SessionStore, trim_history
from app.core.memory import (
    SessionStore as DirectSessionStore,
    trim_history as direct_trim_history,
)


class TestSessionStore:
    """测试内存会话存储管理器 (SessionStore) 的功能。"""

    def test_session_store_basic(self) -> None:
        """测试添加用户消息与客服消息、按顺序读取历史以及不同会话间的隔离性。

        返回值:
            None: 无返回值，通过断言验证消息顺序、类型与多会话隔离。
        """
        chat_memory = SessionStore()
        session_a = "session_user_001"
        session_b = "session_user_002"

        # 会话 A 添加对话轮次
        chat_memory.append(
            session_a,
            HumanMessage(content="你好，我想咨询退货流程"),
            AIMessage(content="您好！请提供您的订单编号，我为您查询。"),
        )
        chat_memory.append(session_a, HumanMessage(content="订单号是 ORD-2026-001"))

        # 验证会话 A 的消息内容与顺序
        msgs_a = chat_memory.get(session_a)
        assert len(msgs_a) == 3
        assert isinstance(msgs_a[0], HumanMessage)
        assert msgs_a[0].content == "你好，我想咨询退货流程"
        assert isinstance(msgs_a[1], AIMessage)
        assert msgs_a[1].content == "您好！请提供您的订单编号，我为您查询。"
        assert isinstance(msgs_a[2], HumanMessage)
        assert msgs_a[2].content == "订单号是 ORD-2026-001"

        # 验证会话 B 隔离：此时会话 B 尚无历史
        msgs_b_initial = chat_memory.get(session_b)
        assert msgs_b_initial == []

        # 会话 B 添加自己的消息，互不干扰
        chat_memory.append(session_b, HumanMessage(content="快递什么时候能发货？"))
        msgs_b = chat_memory.get(session_b)
        assert len(msgs_b) == 1
        assert msgs_b[0].content == "快递什么时候能发货？"

        # 会话 A 的历史数量不受会话 B 影响
        assert len(chat_memory.get(session_a)) == 3

    def test_session_store_returns_copy(self) -> None:
        """测试 get 返回的是内部消息列表的浅拷贝副本，防止外部修改污染内部状态。

        返回值:
            None: 无返回值，断言外部变更不污染内部会话存储。
        """
        chat_memory = SessionStore()
        session_id = "session_copy_test"
        chat_memory.append(session_id, HumanMessage(content="原始消息"))

        # 获取消息列表并在外部修改
        copied_list = chat_memory.get(session_id)
        assert len(copied_list) == 1
        copied_list.append(HumanMessage(content="外部追加的消息"))

        # 再次获取内部存储，长度应仍为 1
        fresh_list = chat_memory.get(session_id)
        assert len(fresh_list) == 1
        assert fresh_list[0].content == "原始消息"

    def test_session_store_clear(self) -> None:
        """测试清除指定 session_id 的历史记录，且不影响其他 session_id。

        返回值:
            None: 无返回值，断言目标会话被清空而其他会话保留。
        """
        chat_memory = SessionStore()
        session_1 = "session_clear_1"
        session_2 = "session_clear_2"

        chat_memory.append(session_1, HumanMessage(content="用户1的消息"))
        chat_memory.append(session_2, HumanMessage(content="用户2的消息"))

        # 清除 session_1
        chat_memory.clear(session_1)

        # 验证 session_1 已被清空，而 session_2 依然保留
        assert chat_memory.get(session_1) == []
        assert len(chat_memory.get(session_2)) == 1

        # 测试清除不存在的 session_id 不会抛出异常
        chat_memory.clear("non_existent_session")

    def test_store_singleton_instance(self) -> None:
        """测试对话入口单例 store 实例存在且属于 SessionStore 类型。

        返回值:
            None: 无返回值，验证单例实例。
        """
        assert isinstance(store, SessionStore)
        assert SessionStore is DirectSessionStore


class TestTrimHistory:
    """测试基于 Token 预算控制的历史裁剪函数 (trim_history)。"""

    def test_trim_history_empty(self) -> None:
        """测试空消息列表安全返回空列表。

        返回值:
            None: 无返回值，断言空列表返回 []。
        """
        assert trim_history([]) == []
        assert direct_trim_history([]) == []

    def test_trim_history_within_budget(self) -> None:
        """测试消息总 token 未超标时，完整保留所有消息（含 SystemMessage）。

        返回值:
            None: 无返回值，断言消息完整保留且顺序一致。
        """
        messages: list[BaseMessage] = [
            SystemMessage(content="你是 MewHelp 电商智能客服。"),
            HumanMessage(content="你好，退货有什么要求吗？"),
            AIMessage(content="您好，商品签收 7 天内无理由退货，需保证商品完好。"),
        ]

        trimmed = trim_history(messages, max_tokens=2000)

        # 预算足够大，所有 3 条消息均应保留
        assert len(trimmed) == 3
        assert trimmed == messages

    def test_trim_history_exceed_budget(self) -> None:
        """测试消息超长时丢弃较早轮次，但始终保留开头的 SystemMessage，且截断后第一条对话消息必须是 HumanMessage。

        返回值:
            None: 无返回值，验证 SystemMessage 保持、首条对话为 HumanMessage、最近轮次保留。
        """
        # 构建一个包含系统提示和 5 轮对话的长对话历史
        messages: list[BaseMessage] = [
            SystemMessage(content="你是 MewHelp 电商智能客服系统。"),
            HumanMessage(content="轮次1 用户提问：" + "关于保修期的详细政策说明 " * 15),
            AIMessage(content="轮次1 客服回答：" + "国家三包规定整机保修一年，主要部件两年 " * 15),
            HumanMessage(content="轮次2 用户提问：" + "发票丢了还能享受全国联保吗 " * 15),
            AIMessage(content="轮次2 客服回答：" + "可以凭电子购买记录和出厂序列号享受保修 " * 15),
            HumanMessage(content="轮次3 用户提问：" + "换货需要多长时间寄回 " * 15),
            AIMessage(content="轮次3 客服回答：" + "售后仓库收到商品后 48 小时内为您发出换新商品 " * 15),
            HumanMessage(content="轮次4 用户提问：" + "物流运费是谁承担呢 " * 15),
            AIMessage(content="轮次4 客服回答：" + "质量问题由商家全额承担双向运费 " * 15),
            HumanMessage(content="轮次5 用户提问：" + "好的，那我申请换货 " * 15),
            AIMessage(content="轮次5 客服回答：" + "请您点击订单详情中的申请售后即可发起 " * 15),
        ]

        # 设置较小的 token 预算（约保留最近 1-2 轮与系统提示词）
        trimmed = trim_history(messages, max_tokens=250)

        # 验证裁剪发生
        assert len(trimmed) < len(messages)
        assert len(trimmed) >= 2  # 至少保留系统提示词和最近的人类消息

        # 始终保留开头的 SystemMessage
        assert isinstance(trimmed[0], SystemMessage)
        assert trimmed[0].content == "你是 MewHelp 电商智能客服系统。"

        # 截断后除 SystemMessage 外的第一条对话消息必须是 HumanMessage (start_on="human")
        assert isinstance(trimmed[1], HumanMessage)

        # 验证保留的是最新的轮次内容（例如最后一条 AI 回答包含申请售后）
        assert "申请售后" in trimmed[-1].content

    def test_trim_history_only_system_message(self) -> None:
        """测试仅包含单条 SystemMessage 时的安全处理。

        返回值:
            None: 无返回值，断言单个 SystemMessage 正常返回。
        """
        messages: list[BaseMessage] = [
            SystemMessage(content="你是 MewHelp 电商智能客服。")
        ]
        trimmed = trim_history(messages, max_tokens=100)
        assert len(trimmed) == 1
        assert isinstance(trimmed[0], SystemMessage)
        assert trimmed[0].content == "你是 MewHelp 电商智能客服。"
