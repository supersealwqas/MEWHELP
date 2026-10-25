"""异步业务工具：演示查询、普通/退款知识检索与真实工单写入。"""

import random
from typing import Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field
from app.schemas.tools import CODE_FIELD, ERROR_FIELD, KeywordArgs, KnowledgeQueryArgs, LogisticsArgs, OwnedOrderArgs, RefundPolicyArgs, TicketArgs
from app.tools.mock_data import NOT_OWNED, logistics_snapshot, order_id_for_tracking_no, order_snapshot, owns_order

# 写操作标记。写操作会改变外部状态，不能像只读查询那样重试；
# 由工具自身声明，执行器据此调整重试策略与超时文案。
WRITE_TAG = "write"


@tool(args_schema=OwnedOrderArgs, description="根据订单号查询当前用户的订单商品、数量、单价、总金额、创建时间、当前状态、物流单号、发出地省市 shipping_origin 和目的地省市 shipping_destination。订单号须由用户提供或在会话中已明确；查物流时先查询订单取得 tracking_no，再调用 query_logistics。待发货订单的 tracking_no 为空，发出地和目的地表示计划配送起终点，不代表已发货。")
async def query_order(
    order_id: Annotated[str, Field(min_length=1, max_length=64, description="用户提供或历史中已明确的订单号")],
    user_id: Annotated[str, InjectedToolArg],
) -> dict:
    """校验订单归属后查询商品、金额、处理状态与配送省市。

    参数:
        order_id: 已去除首尾空白的非空订单号，最多 64 字符。
        user_id: 系统注入的当前用户标识，不由模型填写。
    返回值:
        dict: 本人订单摘要，包含发出地省市 shipping_origin 和目的地省市 shipping_destination；
            待发货时两字段表示计划配送起终点，不代表已发货。
            未知或非本人订单统一返回 NOT_OWNED 的副本。
    异常:
        无主动抛出的业务异常；参数不合法由工具参数模型抛出校验错误。
    """
    if not owns_order(user_id, order_id):
        return dict(NOT_OWNED)
    return order_snapshot(order_id)


query_order.tags = ["mock"]
# 当前 LangChain 版本的 tool() 不接受 tags 关键字，只能事后赋值；保留来源标记供后台审计。


@tool(args_schema=KeywordArgs, description="根据商品名称或编号查询价格、库存和基础规格，仅用于价格/库存类商品咨询；知识库中的型号、使用方法和售后政策应使用 query_faq。")
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
        "name": keyword,
        "price": rng.choice([39.9, 89.0, 129.0]),
        "currency": "CNY",
        "stock": rng.randint(0, 80),
        "specification": "标准装",
    }


query_product.tags = ["mock"]
# 同上：tags 事后赋值，保留演示数据的内部来源标记供后台审计。


@tool(args_schema=LogisticsArgs, description="根据用户提供或query_order成功结果中的tracking_no物流单号查询当前用户的配送状态、承运方、最近物流节点及节点更新时间。只接受物流单号，不接受订单号；只有订单号时先查订单，再使用结果里的非空tracking_no继续查询，无需用户重复提供。待发货且没有单号时不调用。")
async def query_logistics(
    tracking_no: Annotated[str, Field(min_length=1, max_length=64, description="用户明确提供或query_order成功结果中的非空物流单号")],
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
# 同上：tags 事后赋值，保留演示数据的内部来源标记供后台审计。


@tool(args_schema=KnowledgeQueryArgs, description="查询商城政策、商品规格/使用、配送与会员知识。keyword 提交当前问题的核心检索词并保留时限/型号/否定；按子句独立召回重排，默认最多10条。category/content_type可选精确预过滤，不得猜测。无匹配时 hits 为空。综合退款申请优先 query_refund_policy，该工具不办理退款。")
async def query_faq(keyword: str, config: RunnableConfig, category: str | None = None,
                    content_type: str | None = None) -> dict:
    """查询商城政策和常见问答。

    参数:
        keyword: 模型提交的查询词，提示词要求取自当前原话；代码不校验原句包含关系。
        config: 当前请求身份与查询资源的可信运行配置。
        category/content_type: 可选权威元数据筛选，不放宽无命中范围。
    返回值:
        dict: 命中时包含 hits 列表；未命中时 hits 为空列表并附带 message 说明。
            使用 hits 而非泛化的 data 键，避免与执行器兼容分支的字段名撞车。
    异常:
        ValueError: 关键词为空、长度超限或检索参数不合法。
        KnowledgeUnavailable: 知识检索依赖不可用。
    """
    resources = config["configurable"]
    support = resources["support_repository"]
    # keyword和可信原问分别传入：允许检索同义词，但不能丢失原问型号和条件。
    filters = {key: value for key, value in {"category": category, "content_type": content_type}.items()
               if value is not None}
    options = {"filters": filters} if filters else {}
    if getattr(support, "uses_full_knowledge_query", False):
        options["source_query"] = resources["context"].user_message
    rows = await support.query_faq(keyword, **options)
    if not rows:
        return {"hits": [], "message": f"未找到与「{keyword}」相关的常见问题"}
    return {"hits": rows}


@tool(args_schema=RefundPolicyArgs, description="检索当前用户的综合退款/退货申请政策，扩展3个检索角度，每条最多10个块，按chunk id去重后全部返回，默认最多30条，不再截为10。仅查询申请条件、流程、处理时限和运费规则，不执行退款、不承诺到账或人工已经接入。")
async def query_refund_policy(query: str, config: RunnableConfig, category: str | None = None,
                              content_type: str | None = None) -> dict:
    """通过可信本轮原话检索三组退款政策证据。

    参数：query 模型提交的当前诉求说明；config 注入本轮原话与仓储身份；category/content_type 可选筛选。
    返回：dict，hits 为去重后的全部权威证据，默认最多 30；空时附安全未命中说明。
    异常：ValueError，输入非法；KnowledgeUnavailable，混合检索未配置或依赖错误。
    """
    resources = config["configurable"]
    raw = resources["context"].user_message
    filters = {key: value for key, value in {"category": category, "content_type": content_type}.items()
               if value is not None}
    rows = await resources["support_repository"].query_refund_policy(raw, **({"filters": filters} if filters else {}))
    if not rows:
        return {"hits": [], "message": "未找到与当前退款问题相关的政策证据"}
    return {"hits": rows}


@tool(args_schema=TicketArgs, description="仅在独立确认接口验证用户明确确认后，以保存的原问和类型创建待处理工单，返回工单号、类型和处理状态。普通聊天及模型建议不执行此工具；建单不改变转人工状态，也不执行退款。")
async def create_ticket(description: str, ticket_type: str, config: RunnableConfig) -> dict:
    """创建待处理人工工单。

    参数:
        description: 已确认的保存原问，保留首尾空白，原始长度最多 2000 字符。
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
