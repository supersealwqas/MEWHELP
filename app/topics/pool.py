"""只读问题池派生视图：精确轮次映射、匿名家族和保守来源证据。"""

import hashlib
from sqlalchemy import select


def derive_pool(pool: list, messages: list, conversations: list, reviews: list) -> list[dict]:
    """生成全池派生视图。参数：池、用户消息、会话和审核实体；返回：脱敏题及证据；异常：歧义ValueError。"""
    from app.topics.data import clean_question
    convs = {row.id: row for row in conversations}
    review_map = {row.id: row for row in reviews}
    result = []
    for row in pool:
        matches = [m for m in messages if m.role == "user" and m.conversation_id == row.conversation_id
                   and (m.rag_metadata or {}).get("knowledge_pool", {}).get("pool_ids", {}).get(row.source) == row.id]
        if len(matches) > 1:
            raise ValueError(f"问题池{row.id}存在歧义原轮映射")
        match = matches[0] if matches else None
        conv = convs.get(row.conversation_id)
        # 只把可见演示标记作为证据，不能把缺少标记解释为真实客户。
        context = " ".join(str(getattr(conv, name, "") or "") for name in ("session_id", "user_id"))
        markers = [word for word in ("demo", "test", "ch09", "验收", "演示") if word in context.lower()]
        source = "application_demo" if markers else "application_history"
        identity = f"{match.conversation_id}:{match.id}" if match else None
        group = "turn:" + hashlib.sha256(identity.encode()).hexdigest()[:24] if identity else f"legacy-pool:{row.id}"
        review = review_map.get(row.matched_review_id)
        result.append(dict(question_id=str(row.id), text=clean_question(row.raw_question), source=source, group_id=group,
                           history_uncertain=match is None, provenance=dict(pool_source=row.source,
                           origin="demonstration_marker" if markers else "unverified_application_history",
                           markers=markers, mapping="exact_pool_ids" if match else "legacy_no_exact_mapping"),
                           review_status=getattr(review, "review_status", None),
                           created_at=row.created_at.isoformat() if row.created_at else None))
    return result


async def read_pool(factory) -> list[dict]:
    """只读全池。参数：factory异步SQLAlchemy会话工厂；返回：派生视图；异常：数据库或歧义错误。"""
    from app.models.rag_quality import LowConfidenceQuestion
    from app.models.customer_service import Conversation, Message
    from app.models.flywheel import ReviewQueue
    async with factory() as session:
        pool = (await session.scalars(select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id))).all()
        messages = (await session.scalars(select(Message).where(Message.role == "user"))).all()
        conversations = (await session.scalars(select(Conversation))).all()
        reviews = (await session.scalars(select(ReviewQueue))).all()
        return derive_pool(pool, messages, conversations, reviews)
