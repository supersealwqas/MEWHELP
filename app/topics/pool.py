"""只读问题池派生视图：精确轮次映射、匿名家族和保守来源证据。"""

import hashlib
from asyncio import to_thread
from collections import defaultdict

from sqlalchemy import or_, select
from sqlalchemy.orm import load_only


def derive_pool(pool: list, messages: list, conversations: list, reviews: list) -> list[dict]:
    """通过一次原轮索引生成派生视图。

    参数：pool待派生池记录，messages候选用户消息，conversations关联会话，reviews关联审核实体；均须已加载所用字段。
    返回：list[dict]，包含脱敏题、原轮分组、来源证据及审核状态。
    异常：待派生池记录匹配多个原轮时抛ValueError；批外映射不参与歧义判断。
    """
    from app.topics.data import clean_question
    convs = {row.id: row for row in conversations}
    review_map = {row.id: row for row in reviews}
    needed = {(row.conversation_id, row.source, row.id) for row in pool}
    by_origin = defaultdict(list)
    for message in messages:
        if message.role != "user":
            continue
        mapping = (message.rag_metadata or {}).get("knowledge_pool", {}).get("pool_ids", {})
        for source, identifier in mapping.items():
            # 原映射要求数字ID精确匹配，不能把历史字符串ID猜成已核实原轮。
            if type(identifier) is int and (message.conversation_id, source, identifier) in needed:
                by_origin[(message.conversation_id, source, identifier)].append(message)
    result = []
    for row in pool:
        matches = by_origin.get((row.conversation_id, row.source, row.id), [])
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


async def read_pool(factory, *, after_id: int = 0, limit: int | None = None,
                    unclassified_only: bool = False) -> list[dict]:
    """先在数据库筛选候选，再读取其派生所需元信息。

    参数：factory异步SQLAlchemy会话工厂；after_id非负排他游标；limit为1至200的页大小或None读取全部；
    unclassified_only为True时排除已有分类行，False用于完整分布/导出或显式重跑。
    返回：list[dict]，按池ID递增的脱敏派生行；空页不再查询关联表。
    异常：分页参数非法ValueError、数据库错误或精确原轮映射歧义ValueError。
    """
    from app.models.rag_quality import LowConfidenceQuestion
    from app.models.customer_service import Conversation, Message
    from app.models.flywheel import ReviewQueue
    from app.models.topics import TopicClassification

    if type(after_id) is not int or after_id < 0 or (limit is not None and (type(limit) is not int or not 1 <= limit <= 200)):
        raise ValueError("问题池分页参数非法")
    candidates = select(LowConfidenceQuestion).options(load_only(
        LowConfidenceQuestion.id, LowConfidenceQuestion.conversation_id, LowConfidenceQuestion.raw_question,
        LowConfidenceQuestion.source, LowConfidenceQuestion.matched_review_id, LowConfidenceQuestion.created_at,
    )).where(LowConfidenceQuestion.id > after_id).order_by(LowConfidenceQuestion.id)
    if unclassified_only:
        candidates = candidates.where(~select(TopicClassification.id).where(
            TopicClassification.question_id == LowConfidenceQuestion.id).exists())
    if limit is not None:
        candidates = candidates.limit(limit)
    async with factory() as session:
        pool = (await session.scalars(candidates)).all()
        if not pool:
            return []
        conversation_ids = {row.conversation_id for row in pool if row.conversation_id is not None}
        review_ids = {row.matched_review_id for row in pool if row.matched_review_id is not None}
        source_ids = defaultdict(list)
        for row in pool:
            source_ids[row.source].append(row.id)
        # 只读取能关联当前池ID的消息元信息，不搬运聊天正文、工具结果或整段会话摘要。
        messages = (await session.scalars(select(Message).options(load_only(
            Message.id, Message.conversation_id, Message.role, Message.rag_metadata,
        )).where(Message.role == "user", Message.conversation_id.in_(conversation_ids), or_(*[
            Message.rag_metadata["knowledge_pool"]["pool_ids"][source].as_integer().in_(identifiers)
            for source, identifiers in source_ids.items()
        ])))).all() if conversation_ids else []
        conversations = (await session.scalars(select(Conversation).options(load_only(
            Conversation.id, Conversation.session_id, Conversation.user_id,
        )).where(Conversation.id.in_(conversation_ids)))).all() if conversation_ids else []
        reviews = (await session.scalars(select(ReviewQueue).options(load_only(
            ReviewQueue.id, ReviewQueue.review_status,
        )).where(ReviewQueue.id.in_(review_ids)))).all() if review_ids else []
    # 全量导出/分布仍可能含许多池行；纯内存派生移出主事件循环，字段已显式加载。
    return await to_thread(derive_pool, pool, messages, conversations, reviews)
