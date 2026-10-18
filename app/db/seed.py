"""显式写入四表演示数据，重复运行仅补缺失项，不覆盖已有记录。"""

from sqlalchemy import select
from app.models.customer_service import Conversation, FAQ
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.schemas.tools import ToolContext

FAQ_SEEDS = [
    (1, "退货政策是什么", "【演示商城政策】签收后 7 天内，商品完好且不影响二次销售可申请退货；质量问题由商城承担退回运费，其他情况由买家承担。生鲜和定制商品不适用无理由退货。", "退换货"),
    (2, "运费标准", "【演示商城政策】普通配送运费为 8 元，订单实付满 99 元免运费；偏远地区费用需人工核实。", "配送"),
    (3, "如何联系客服", "【演示商城政策】可以在此聊天中明确提出创建人工工单，由工作人员后续处理。", "客服"),
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
