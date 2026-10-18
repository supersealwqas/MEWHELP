"""五个异步业务工具：三项随机演示查询、FAQ 字面检索与真实工单写入。"""

import random
from datetime import datetime, timezone
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from app.schemas.tools import KeywordArgs, OrderArgs, TicketArgs, ToolResult


@tool(args_schema=OrderArgs, description="按已知订单号查询订单摘要、金额和状态，返回随机演示数据；物流进度请直接用 query_logistics。")
async def query_order(order_id: str) -> ToolResult:
    """生成自洽演示订单。参数：非空 order_id；返回：标注 mock 的摘要；异常：无业务异常。"""
    quantity, price = random.randint(1, 3), random.choice([39.9, 89.0, 129.0])
    return ToolResult(ok=True, is_mock=True, data={"order_id": order_id, "product": "演示猫粮",
        "quantity": quantity, "unit_price": price, "total_amount": round(quantity * price, 2),
        "currency": "CNY", "status": random.choice(["待发货", "已发货", "已签收"])})


@tool(args_schema=KeywordArgs, description="按商品名称或编号查询价格、库存和规格，返回随机演示数据。")
async def query_product(keyword: str) -> ToolResult:
    """生成演示商品。参数：keyword 名称或编号；返回：价格库存字典；异常：无业务异常。"""
    return ToolResult(ok=True, is_mock=True, data={"keyword": keyword, "name": f"演示商品：{keyword}",
        "price": random.choice([39.9, 89.0, 129.0]), "currency": "CNY", "stock": random.randint(0, 80),
        "specification": "标准装"})


@tool(args_schema=OrderArgs, description="直接按订单号查询物流到哪了、承运方和最近节点，无需先查订单；结果为随机演示数据。")
async def query_logistics(order_id: str) -> ToolResult:
    """生成状态与节点一致的物流。参数：已知 order_id；返回：物流字典；异常：无业务异常。"""
    status, node = random.choice([("运输中", "包裹已到达杭州转运中心"),
                                ("派送中", "包裹由目的地配送站安排派送"),
                                ("已签收", "包裹已在收货地址签收")])
    return ToolResult(ok=True, is_mock=True, data={"order_id": order_id, "carrier": "演示速递",
        "status": status, "latest_node": node, "updated_at": datetime.now(timezone.utc).isoformat()})


@tool(args_schema=KeywordArgs, description="查商城政策等常见问题。keyword 必须取自当前用户原话的连续片段；例如邮费就查邮费，不能改成运费。不命中时返回空列表。")
async def query_faq(keyword: str, config: RunnableConfig) -> ToolResult:
    """查询 FAQ 表。参数：原文 keyword，config 为服务端注入资源；返回：最多三条问答；异常：ValueError 或数据库错误。"""
    resources = config["configurable"]
    if keyword not in resources["context"].user_message:
        raise ValueError("FAQ 关键词必须来自当前用户原话")
    data = await resources["support_repository"].query_faq(keyword)
    return ToolResult(ok=True, data=data)


@tool(args_schema=TicketArgs, description="仅在用户明确要求人工协助、创建工单或提交处理时创建待处理工单；不能退款、不能表示人工已接入，不因 FAQ 无结果自动建单。")
async def create_ticket(description: str, ticket_type: str, config: RunnableConfig) -> ToolResult:
    """原子保存人工工单。参数：描述、类型和可信运行配置；返回：工单事实；异常：数据库或一致性错误。"""
    resources = config["configurable"]
    return await resources["support_repository"].create_ticket(resources["context"], description, ticket_type)
