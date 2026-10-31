"""主题仓储：全池精确分组统计与同租约连接上的短事务保存。"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy import func, select, text
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.lease import MySqlWriteLease, finish_cleanup
from app.knowledge.errors import LeaseLost
from app.models.rag_quality import LowConfidenceQuestion
from app.models.topics import TopicClassification
from app.topics.pool import read_pool
from app.topics.taxonomy import label_names, load_taxonomy, validate_labels

TOPIC_EXECUTION_LOCK = "mewhelp_ch10_topic_batch"


def aggregate_distribution(pool, classifications, scope="pending"):
    """按可信轮次作主题并集，旧行仅按ID回退。参数：全池/分类行/scope；返回：17类及组/行覆盖；异常：scope或标签非法。"""
    if scope not in {"pending", "all"}:
        raise ValueError("scope必须为pending或all")
    taxonomy = load_taxonomy()
    labels = {str(r["question_id"]): set(validate_labels(r["labels"], taxonomy)) for r in classifications}
    selected = [r for r in pool if scope == "all" or r["review_status"] in {None, "待审"}]
    groups, known = {}, set()
    classified_rows = 0
    for row in selected:
        group = row["group_id"]
        groups.setdefault(group, set()).update(labels.get(row["question_id"], set()))
        classified_rows += int(row["question_id"] in labels)
        if not row["history_uncertain"]:
            known.add(group)
    classified = sum(bool(values) for values in groups.values())
    # 来源说明按所选池行计数，同轮多入口仍是多条可见来源记录。
    source_counts = {source: sum(row.get("source") == source for row in selected)
                     for source in ("application_demo", "application_history")}
    return dict(labels=[dict(label=name, count=sum(name in values for values in groups.values()))
                        for name in label_names(taxonomy)],
        total_questions=len(groups), classified_questions=classified, unclassified_questions=len(groups) - classified,
        known_original_turns=len(known), history_uncertain_pool_rows=sum(r["history_uncertain"] for r in selected),
        pool_rows=len(selected), classified_pool_rows=classified_rows, unclassified_pool_rows=len(selected) - classified_rows,
        source_counts=source_counts,
        source_note=f"演示记录{source_counts['application_demo']}条；来源未核实历史{source_counts['application_history']}条。")


def classification_payload(row):
    """输出可与批次原件对账的字段。参数：ORM分类行；返回：字符串ID/UTC无偏移时间；异常：无。"""
    return dict(question_id=str(row.question_id), labels=row.labels, classified_at=row.classified_at.isoformat())


class TopicRepository:
    """读写分离，写入只绑定调用方持有主题锁的连接。"""

    def __init__(self, factory, lease_engine):
        """参数：会话工厂/NullPool租约引擎；返回：None；异常：无。"""
        self.factory, self.lease_engine = factory, lease_engine

    @asynccontextmanager
    async def execution(self):
        """先于池读取争用共同执行锁。参数：无；返回：lease；异常：占用/失锁/连接/取消。"""
        lease = await MySqlWriteLease.acquire(self.lease_engine, timeout_seconds=0, lock_name=TOPIC_EXECUTION_LOCK)
        try:
            yield lease
        finally:
            await finish_cleanup(lease.release())

    async def execution_busy(self):
        """读取跨实例锁事实。参数：无；返回：bool；异常：数据库错误。"""
        async with self.factory() as session:
            return await session.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": TOPIC_EXECUTION_LOCK}) is not None

    async def _assert_topic_held(self, lease):
        """核验传入物理连接实际拥有主题锁，拒绝其他业务租约。参数：lease；返回：None；异常：失锁或数据库错误。"""
        await lease.assert_held()
        owner = await lease.connection.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": TOPIC_EXECUTION_LOCK})
        current = await lease.connection.scalar(text("SELECT CONNECTION_ID()"))
        if owner is None or current is None or int(owner) != int(current):
            raise LeaseLost("传入租约未持有主题共同执行锁")

    async def high_watermark(self):
        """冻结当前轮次上界。参数：无；返回：最大整数ID或0；异常：数据库错误。"""
        async with self.factory() as session:
            return int(await session.scalar(select(func.max(LowConfidenceQuestion.id))) or 0)

    async def classified_rows(self):
        """读取全部当前分类，供全表版本对账。参数：无；返回：分类列表；异常：数据库错误。"""
        async with self.factory() as session:
            return [classification_payload(row) for row in (await session.scalars(
                select(TopicClassification).order_by(TopicClassification.question_id))).all()]

    async def pending(self, limit: int, reclassify: bool = False, after_id: int = 0):
        """递增读取候选，沿用Task1全池映射。参数：1至200条/reclassify/非负游标；返回：字符串ID列表；异常：输入/数据库/映射错误。"""
        if type(limit) is not int or not 1 <= limit <= 200 or type(after_id) is not int or after_id < 0:
            raise ValueError("主题分页参数非法")
        pool = await read_pool(self.factory)
        classified = set() if reclassify else {r["question_id"] for r in await self.classified_rows()}
        return [r for r in pool if int(r["question_id"]) > after_id and r["question_id"] not in classified][:limit]

    async def save(self, rows: list[dict], *, lease):
        """核验整批后在同连接显式UTC短事务upsert。参数：最多200个唯一ID合法标签/主题lease；返回：saved计数；异常：输入/失锁/数据库/取消。"""
        if not isinstance(rows, list) or len(rows) > 200:
            raise ValueError("主题保存最多200行")
        taxonomy, identifiers, values = load_taxonomy(), set(), []
        now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        for row in rows:
            identifier = row.get("question_id")
            if not isinstance(identifier, str) or not identifier.isdecimal() or not 1 <= int(identifier) <= 2**64 - 1:
                raise ValueError("主题问题ID必须为合法十进制字符串")
            if int(identifier) in identifiers:
                raise ValueError("同一批次问题ID重复")
            identifiers.add(int(identifier))
            labels = validate_labels(row.get("labels"), taxonomy)
            stamp = datetime.fromisoformat(row["classified_at"]) if "classified_at" in row else now
            if stamp.tzinfo is not None or stamp.microsecond:
                raise ValueError("归类时间必须为UTC无偏移整秒")
            values.append(dict(question_id=int(identifier), labels=labels, classified_at=stamp))
        await self._assert_topic_held(lease)
        # 核锁查询触发autobegin，先结束只读事务才建立真正提交的Session事务。
        await lease.connection.commit()
        async with AsyncSession(bind=lease.connection, expire_on_commit=False) as session:
            async with session.begin():
                if values:
                    statement = insert(TopicClassification).values(values)
                    await session.execute(statement.on_duplicate_key_update(labels=statement.inserted.labels,
                        classified_at=statement.inserted.classified_at))
                # 最后一次锁验证仍在同一事务/物理连接；异常和取消由begin回滚。
                await self._assert_topic_held(lease)
        return {"saved": len(values)}

    async def distribution(self, scope="pending"):
        """统计完整池，而非第一页未归类候选。参数：pending或all；返回：分布/覆盖；异常：scope/标签/数据库错误。"""
        return aggregate_distribution(await read_pool(self.factory), await self.classified_rows(), scope)
