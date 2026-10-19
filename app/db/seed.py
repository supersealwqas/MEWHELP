"""显式写入四表演示数据，重复运行仅补缺失项，不覆盖已有记录。"""

from sqlalchemy import select
from app.models.customer_service import Conversation, FAQ
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.schemas.tools import ToolContext

# 问题、答案、分类及顺序以 sql/ch02-seed.sql 为准；固定 ID 用于初始化时只补缺失项。
FAQ_SEEDS = [
    (1, "退货政策", "支持 7 天无理由退货,商品需保持完好、不影响二次销售,具体以平台售后规则为准。", "售后"),
    (2, "如何申请退款", "在「我的订单」找到对应订单点击「申请退款」,按提示提交,审核通过后原路退回。", "售后"),
    (3, "换货流程", "收到商品 7 天内可申请换货,联系客服登记后将商品寄回,平台核验无误后补发新品。", "售后"),
    (4, "发货时效", "现货商品付款后 48 小时内发货,预售商品以商品详情页标注的发货时间为准。", "物流"),
    (5, "运费怎么算", "单笔订单满 99 元包邮,未满收取 10 元运费,偏远地区运费另计。", "物流"),
    (6, "发票如何开具", "在「我的订单」-「申请开票」提交抬头与税号,电子发票将于 3 个工作日内发送至邮箱。", "账务"),
]


async def seed_database(factory) -> None:
    """幂等补充演示记录。参数：factory 异步会话工厂；返回：None；异常：数据库错误。"""
    async with factory.begin() as session:
        for identifier, question, answer, category in FAQ_SEEDS:
            if await session.get(FAQ, identifier) is None:
                session.add(FAQ(id=identifier, question=question, answer=answer, category=category))
    chat, support = ChatRepository(factory), SupportRepository(factory)
    async with factory() as session:
        existing = await session.scalar(select(Conversation).where(Conversation.session_id == "demo-seed-session"))
    ctx = ToolContext(session_id="demo-seed-session", user_id="demo-seed-user", turn_id="demo-seed-turn",
                      tool_call_id="demo-seed-call", user_message="请帮我创建人工工单，查询演示订单延迟问题")
    if existing and existing.user_id != ctx.user_id:
        raise ValueError("演示会话标识已被其他用户占用，未覆盖记录")
    messages = []
    if existing:
        ctx = ctx.model_copy(update={"conversation_id": existing.id})
        turns = await chat.load_turns(existing.id)
        messages = [message for turn in turns if turn.turn_id == ctx.turn_id for message in turn.messages]
    # 逐个补齐上次初始化已提交的前缀，保留已有消息内容和用户修改。
    if not any(message.role == "user" for message in messages):
        ctx = await chat.start_turn(ctx, ctx.user_message)
    if not any(message.tool_calls for message in messages):
        await chat.save_tool_calls(ctx, [{"id": ctx.tool_call_id, "name": "create_ticket",
                                         "args": {"description": "演示订单延迟", "ticket_type": "咨询"}}])
    result = await support.create_ticket(ctx, "演示订单延迟", "咨询")
    if not any(message.role == "assistant" and not message.tool_calls for message in messages):
        await chat.finish_turn(ctx, f"已创建演示工单 {result.data['ticket_no']}，状态为待处理。")
