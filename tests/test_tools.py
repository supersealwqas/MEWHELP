"""验证业务工具注册、参数边界、固定演示数据与可信上下文注入。"""

import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import pytest
from pydantic import ValidationError
from app.schemas.tools import ToolContext, ToolResult
from app.tools.registry import build_registry


# 独立解释器只导入业务工具，不初始化模型或数据库，完整结果用于重启稳定性比较。
_MOCK_SUBPROCESS_SCRIPT = '''"""独立进程读取三个演示工具的完整结果，不连接模型或数据库。"""
import asyncio
import json
from app.tools.business import query_order, query_product, query_logistics


async def read_mock_results():
    """读取固定输入。参数：无；返回：list[dict]，完整工具结果；异常：工具调用错误。"""
    results = await asyncio.gather(
        query_order.ainvoke({"order_id": "1001", "user_id": "demo-user"}),
        query_product.ainvoke({"keyword": "猫粮"}),
        query_logistics.ainvoke({"tracking_no": "MOCK213502378238", "user_id": "demo-user"}),
    )
    return [result if isinstance(result, dict) else result.model_dump(mode="json") for result in results]


print(json.dumps(asyncio.run(read_mock_results()), ensure_ascii=False, sort_keys=True))
'''


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
    ("query_order", {"order_id": " ", "user_id": "demo-user"}),
    ("query_order", {"order_id": 1001, "user_id": "demo-user"}),
    ("query_order", {"order_id": "a" * 65, "user_id": "demo-user"}),
    ("query_logistics", {"tracking_no": " ", "user_id": "demo-user"}),
    ("query_logistics", {"tracking_no": 1001, "user_id": "demo-user"}),
    ("query_logistics", {"tracking_no": "a" * 65, "user_id": "demo-user"}),
    ("query_logistics", {"order_id": "1001", "user_id": "demo-user"}),
    ("query_order", {"order_id": "1001", "user_id": "demo-user", "extra": "x"}),
    ("query_product", {"keyword": "a" * 101}),
    ("query_faq", {"keyword": "退货", "conversation_id": 1}),
    ("create_ticket", {"description": "x", "ticket_type": "退款"}),
    ("create_ticket", {"description": "x" * 2001, "ticket_type": "咨询"}),
])
async def test_validate_args_before_execution(name, args):
    """真实工具拒绝非法参数，尚未取仓库即失败。参数：工具名和非法输入；返回：None；异常：断言失败。"""
    with pytest.raises(ValidationError):
        await build_registry().get(name).ainvoke(args)


@pytest.mark.parametrize("name,args", [
    ("query_order", {"order_id": " 1001 ", "user_id": "demo-user"}),
    ("query_logistics", {"tracking_no": " MOCK213502378238 ", "user_id": "demo-user"}),
])
async def test_mock_preserves_order_identifier(name, args):
    """归属校验前清理业务字段空白并保留订单号。参数：工具名及带空白业务参数；返回：None；异常：断言失败。"""
    result = await build_registry().get(name).ainvoke(args)
    assert isinstance(result, dict) and result["order_id"] == "1001"


def _serialize_result(result: dict | ToolResult) -> dict:
    """统一比较原始业务字典和既有工具结果。参数：result 工具结果；返回：可序列化字典；异常：无。"""
    return result if isinstance(result, dict) else result.model_dump(mode="json")


@pytest.mark.parametrize("name,args,padded_args", [
    ("query_order", {"order_id": "1001", "user_id": "demo-user"}, {"order_id": " 1001 ", "user_id": "demo-user"}),
    ("query_product", {"keyword": "猫粮"}, {"keyword": " 猫粮 "}),
    ("query_logistics", {"tracking_no": "MOCK213502378238", "user_id": "demo-user"},
     {"tracking_no": " MOCK213502378238 ", "user_id": "demo-user"}),
])
async def test_mock_result_stable_after_interleaved_and_concurrent_calls(name, args, padded_args):
    """交错及并发查询不改变同一输入。参数：工具名、标准及带空白参数；返回：None；异常：断言失败。"""
    registry = build_registry()
    tool = registry.get(name)
    expected = _serialize_result(await tool.ainvoke(args))
    if name == "query_product":
        assert expected["name"] == "演示商品：猫粮" and expected["keyword"] == "猫粮"
    else:
        assert expected["order_id"] == "1001"

    for index in range(3):
        # 其他输入穿插调用，防止结果取决于全局随机数状态或上一次查询。
        await asyncio.gather(
            registry.get("query_order").ainvoke({"order_id": str(1002 + index), "user_id": "demo-user"}),
            registry.get("query_product").ainvoke({"keyword": f"猫砂-{index}"}),
            registry.get("query_logistics").ainvoke({
                "tracking_no": ["MOCK487816333527", "MOCK973791694360", "MOCK587093061117"][index],
                "user_id": "demo-user",
            }),
        )
        results = await asyncio.gather(tool.ainvoke(args), tool.ainvoke(padded_args))
        assert [_serialize_result(result) for result in results] == [expected, expected]


@pytest.mark.parametrize("name,args", [
    ("query_order", {"order_id": "1001", "user_id": "demo-user"}),
    ("query_product", {"keyword": "猫粮"}),
    ("query_logistics", {"tracking_no": "MOCK213502378238", "user_id": "demo-user"}),
])
async def test_mock_queries_preserve_global_random_state(name, args):
    """工具不扰动其他业务的随机序列。参数：工具名及合法输入；返回：None；异常：断言失败。"""
    original_state = random.getstate()
    try:
        await build_registry().get(name).ainvoke(args)
        assert random.getstate() == original_state
    finally:
        # 即使旧实现导致断言失败，也恢复进程状态，避免污染其他测试。
        random.setstate(original_state)


@pytest.mark.parametrize("order_id", ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "A-1001"])
async def test_order_and_logistics_share_shipment_facts(order_id):
    """同一订单的发货事实一致。参数：实际演示订单号；返回：None；异常：断言失败。"""
    registry = build_registry()
    order = await registry.get("query_order").ainvoke({"order_id": order_id, "user_id": "demo-user"})
    if order["tracking_no"] is None:
        # 未发货订单没有可供工具查询的物流号，只核对数据层的待发货事实。
        from app.tools.mock_data import _mock_order_snapshot
        _, logistics = _mock_order_snapshot(order_id)
    else:
        logistics = await registry.get("query_logistics").ainvoke({
            "tracking_no": order["tracking_no"], "user_id": "demo-user",
        })
    assert order["order_id"] == logistics["order_id"] == order_id
    assert order["total_amount"] == round(order["quantity"] * order["unit_price"], 2)
    assert order["tracking_no"] == logistics["tracking_no"]
    assert datetime.fromisoformat(order["created_at"]) == datetime.fromisoformat(logistics["updated_at"]) - timedelta(days=2)

    # 订单只呈现发货阶段，物流的运输中和派送中都属于已发货。
    order_status_by_logistics = {
        "待发货": "待发货", "运输中": "已发货", "派送中": "已发货", "已签收": "已签收",
    }
    assert logistics["status"] in order_status_by_logistics
    assert order["status"] == order_status_by_logistics[logistics["status"]]
    if order["status"] == "待发货":
        assert logistics["tracking_no"] is None
        assert logistics["carrier"] is None
        assert "未发货" in logistics["latest_node"]
    else:
        assert isinstance(logistics["tracking_no"], str) and logistics["tracking_no"]
        assert logistics["carrier"]


async def test_documented_order_1001_snapshot():
    """锁定文档中可复现的订单与物流样例。参数：无；返回：None；异常：断言失败。"""
    registry = build_registry()
    order = await registry.get("query_order").ainvoke({"order_id": "1001", "user_id": "demo-user"})
    logistics = await registry.get("query_logistics").ainvoke({"tracking_no": "MOCK213502378238", "user_id": "demo-user"})
    assert order == {
        "order_id": "1001", "product": "演示猫粮", "quantity": 1, "unit_price": 39.9,
        "total_amount": 39.9, "currency": "CNY", "status": "已发货", "tracking_no": "MOCK213502378238",
        "created_at": "2026-02-03T16:45:00+00:00",
    }
    assert logistics == {
        "order_id": "1001", "carrier": "演示速递", "tracking_no": "MOCK213502378238",
        "status": "运输中", "latest_node": "包裹已到达杭州转运中心",
        "updated_at": "2026-02-05T16:45:00+00:00",
    }


def test_mock_results_stable_across_process_restarts_and_hash_seeds():
    """重启且改变哈希种子后结果相同。参数：无；返回：None；异常：断言、子进程或超时错误。"""
    snapshots = []
    for hash_seed in ("1", "987654"):
        # 仅传递解释器运行所需的系统变量，避免继承模型或数据库凭据。
        env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH") if key in os.environ}
        env.update({"PYTHONHASHSEED": hash_seed, "PYTHONIOENCODING": "utf-8"})
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", _MOCK_SUBPROCESS_SCRIPT],
            cwd=Path(__file__).resolve().parents[1], env=env,
            capture_output=True, text=True, encoding="utf-8", check=True, timeout=30,
        )
        snapshots.append(json.loads(completed.stdout))
    assert snapshots[0] == snapshots[1]


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
    assert result["hits"] == [] and "未找到" in result["message"]
    with pytest.raises(ValueError):
        await tool.ainvoke({"keyword": "运费"}, config=config)
