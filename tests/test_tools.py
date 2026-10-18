"""验证真实 LangChain 工具注册、参数边界和可信上下文注入。"""

import pytest
from pydantic import ValidationError
from app.schemas.tools import ToolContext
from app.tools.registry import build_registry


def test_five_registered_tools():
    """仅暴露五个业务工具。参数：无；返回：None；异常：断言失败。"""
    assert {tool.name for tool in build_registry().all()} == {
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}


def test_runtime_context_hidden():
    """模型不能指定运行时身份。参数：无；返回：None；异常：断言失败。"""
    for tool in build_registry().all():
        fields = tool.tool_call_schema.model_json_schema()["properties"]
        assert not {"config", "conversation_id", "user_id", "turn_id", "tool_call_id"}.intersection(fields)


def test_duplicate_or_unknown_tool():
    """拒绝重复注册和未知名称。参数：无；返回：None；异常：断言失败。"""
    registry = build_registry()
    with pytest.raises(ValueError):
        registry.register(registry.get("query_order"))
    with pytest.raises(KeyError):
        registry.get("drop_database")


@pytest.mark.parametrize("name,args", [
    ("query_order", {"order_id": " "}), ("query_order", {"order_id": 1001}),
    ("query_order", {"order_id": "a" * 65}), ("query_product", {"keyword": "a" * 101}),
    ("query_faq", {"keyword": "退货", "conversation_id": 1}),
    ("create_ticket", {"description": "x", "ticket_type": "退款"}),
    ("create_ticket", {"description": "x" * 2001, "ticket_type": "咨询"}),
])
async def test_validate_args_before_execution(name, args):
    """真实工具拒绝非法参数，尚未取仓库即失败。参数：工具名和非法输入；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError):
        await build_registry().get(name).ainvoke(args)


@pytest.mark.parametrize("name", ["query_order", "query_logistics"])
async def test_mock_preserves_order_identifier(name):
    """随机查询保留订单号并标注演示。参数：订单工具名；返回：None；异常：断言失败。"""
    result = await build_registry().get(name).ainvoke({"order_id": " 1001 "})
    assert result.is_mock and result.ok and result.data["order_id"] == "1001"


async def test_faq_context_is_injected_and_keyword_is_literal():
    """关键词不能被模型替换同义词。参数：无；返回：None；异常：断言失败。"""
    class Support:
        """只替换外部数据库读取，记录传入的字面词。"""
        async def query_faq(self, keyword):
            """返回受控数据库结果。参数：原文词；返回：空列表；异常：断言失败。"""
            assert keyword == "邮费"
            return []

    context = ToolContext(session_id="s", user_id="u", turn_id="t", user_message="邮费是多少")
    config = {"configurable": {"context": context, "support_repository": Support()}}
    tool = build_registry().get("query_faq")
    result = await tool.ainvoke({"keyword": "邮费"}, config=config)
    assert result.ok and result.data == []
    with pytest.raises(ValueError):
        await tool.ainvoke({"keyword": "运费"}, config=config)
