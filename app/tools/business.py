"""五个异步业务工具：三项可复现演示查询、FAQ 字面检索与真实工单写入。"""

import random
from typing import Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field
from app.schemas.tools import CODE_FIELD, ERROR_FIELD, KeywordArgs, LogisticsArgs, OwnedOrderArgs, TicketArgs
from app.tools.mock_data import NOT_OWNED, logistics_snapshot, order_id_for_tracking_no, order_snapshot, owns_order

# 写操作标记。写操作会改变外部状态，不能像只读查询那样重试；
# 由工具自身声明，执行器据此调整重试策略与超时文案。
WRITE_TAG = "write"


@tool(args_schema=OwnedOrderArgs, description="根据订单号查询当前用户的订单商品、数量、单价、总金额、创建时间、当前状态和物流单号。订单号须由用户提供或在会话中已明确；查物流时先查询订单取得 tracking_no，再调用 query_logistics。待发货订单的 tracking_no 为空。")
async def query_order(
    order_id: Annotated[str, Field(min_length=1, max_length=64, description="用户提供或历史中已明确的订单号")],
    user_id: Annotated[str, InjectedToolArg],
) -> dict:
    """校验订单归属后查询商品、金额与处理状态。

    参数:
        order_id: 已去除首尾空白的非空订单号，最多 64 字符。
        user_id: 系统注入的当前用户标识，不由模型填写。
    返回值:
        dict: 本人订单摘要；未知或非本人订单统一返回 NOT_OWNED 的副本。
    异常:
        无主动抛出的业务异常；参数不合法由工具参数模型抛出校验错误。
    """
    if not owns_order(user_id, order_id):
        return dict(NOT_OWNED)
    return order_snapshot(order_id)


query_order.tags = ["mock"]
# 当前 LangChain 版本的 tool() 不接受 tags 关键字，只能事后赋值；删掉这行会退回真实数据语义。


@tool(args_schema=KeywordArgs, description="根据商品名称或编号查询商品价格、库存和规格，适用于商品咨询和购买前的信息确认。")
async def query_product(keyword: str) -> dict:
    """查询商品价格、库存与规格。

    参数:
        keyword: 非空商品名称或编号，最多 100 字符。
    返回值:
        dict: 包含商品名称、价格、库存和规格的业务字典。
    异常:
        无主动抛出的业务异常；参数不合法由工具参数模型抛出校验错误。
    """
    rng = random.Random(f"product:{keyword}")
    return {
        "keyword": keyword,
        "name": f"演示商品：{keyword}",
        "price": rng.choice([39.9, 89.0, 129.0]),
        "currency": "CNY",
        "stock": rng.randint(0, 80),
        "specification": "标准装",
    }


query_product.tags = ["mock"]
# 同上：tags 事后赋值，缺这行则该工具的演示数据会被当作真实数据播报。


@tool(args_schema=LogisticsArgs, description="根据 query_order 返回的 tracking_no 物流单号查询当前用户的配送状态、承运方、最近物流节点及节点更新时间。只接受物流单号，不接受订单号；必须先查订单取得非空 tracking_no，待发货时不调用。")
async def query_logistics(
    tracking_no: Annotated[str, Field(min_length=1, max_length=64, description="query_order 返回的非空物流单号")],
    user_id: Annotated[str, InjectedToolArg],
) -> dict:
    """按物流单号校验对应订单归属后查询配送进度。

    参数:
        tracking_no: 订单查询返回的非空物流单号，最多 64 字符。
        user_id: 系统注入的当前用户标识，不由模型填写。
    返回值:
        dict: 本人订单的物流详情；未知或非本人物流统一返回 NOT_OWNED 的副本。
    异常:
        无主动抛出的业务异常；参数不合法由工具参数模型抛出校验错误。
    """
    order_id = order_id_for_tracking_no(tracking_no)
    if not owns_order(user_id, order_id):
        return dict(NOT_OWNED)
    return logistics_snapshot(tracking_no)


query_logistics.tags = ["mock"]
# 同上：tags 事后赋值，缺这行则该工具的演示数据会被当作真实数据播报。


@tool(args_schema=KeywordArgs, description="按关键词查询退换货、配送、发票等商城政策和常见问题，返回最多三条相关问答。关键词必须取自当前用户原话的连续片段，不能改成同义词；例如邮费不能改成运费。无匹配时 hits 为空列表。")
async def query_faq(keyword: str, config: RunnableConfig) -> dict:
    """查询商城政策和常见问答。

    参数:
        keyword: 原文关键词，必须取自当前用户原话。
        config: 当前请求身份与查询资源的可信运行配置。
    返回值:
        dict: 命中时包含 hits 列表；未命中时 hits 为空列表并附带 message 说明。
            使用 hits 而非泛化的 data 键，避免与执行器兼容分支的字段名撞车。
    异常:
        ValueError: 关键词不在当前用户原话中或参数不合法。
    """
    resources = config["configurable"]
    if keyword not in resources["context"].user_message:
        raise ValueError("FAQ 关键词必须来自当前用户原话")
    rows = await resources["support_repository"].query_faq(keyword)
    if not rows:
        return {"hits": [], "message": f"未找到与「{keyword}」相关的常见问题"}
    return {"hits": rows}


@tool(args_schema=TicketArgs, description="为用户明确提出的人工协助、创建工单或提交处理请求创建工单，返回工单号、类型和处理状态。类型仅支持售后、投诉和咨询，创建后状态为待处理；不执行退款，也不表示人工已接入。不能仅因政策查询无结果而自动创建工单。")
async def create_ticket(description: str, ticket_type: str, config: RunnableConfig) -> dict:
    """创建待处理人工工单。

    参数:
        description: 非空问题描述，最多 2000 字符。
        ticket_type: 工单类型，仅支持售后、投诉、咨询。
        config: 当前请求身份与工单服务的可信运行配置。
    返回值:
        dict: 包含工单号和处理状态的业务字典。
    异常:
        ValueError: 参数不合法。
        PersistenceUncertain: 数据库写入状态未确认。
    """
    resources = config["configurable"]
    result = await resources["support_repository"].create_ticket(resources["context"], description, ticket_type)
    # 仓储层返回内部事实载体；工具层必须交出业务字典，失败则按共享协议转成拒绝载荷，
    # 使具体业务错误码（如工单号冲突）能传到执行器，而不是被降级成笼统的工具故障。
    if not result.ok or result.error is not None:
        return {ERROR_FIELD: result.error.message, CODE_FIELD: result.error.code}
    return result.data


create_ticket.tags = [WRITE_TAG]
