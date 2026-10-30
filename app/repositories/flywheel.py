"""知识缺口持久归并：跨进程命名锁、精确轮次落点、同连接短事务与原子计数。"""

from contextlib import asynccontextmanager

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.lease import MySqlWriteLease
from app.models.customer_service import Message
from app.models.flywheel import ReviewQueue
from app.models.rag_quality import LowConfidenceQuestion
from app.schemas.flywheel import NormalizedGap, ReviewCandidate, TurnPoolState


class CandidateChanged(RuntimeError):
    """模型等待时待审候选被审核或冻结；必须重新分页判定。"""


class FlywheelRepository:
    """按低置信池主键扫描并将同轮三来源归并到一个计数落点。"""

    def __init__(self, factory, lease_engine, *, lease=None):
        """参数：独立会话工厂、NullPool锁引擎、内部已持租约；返回：None；异常：无。"""
        self.factory, self.lease_engine, self.lease = factory, lease_engine, lease

    @asynccontextmanager
    async def locked(self):
        """参数：无；返回：持同一专用连接的仓储；异常：锁等待/失锁/取消。"""
        lease = await MySqlWriteLease.acquire(self.lease_engine, lock_name="mewhelp_ch09_flywheel_merge")
        try:
            yield FlywheelRepository(self.factory, self.lease_engine, lease=lease)
        finally:
            await lease.release()

    async def assert_held(self):
        """核锁后结束隐式事务，模型等待不持行事务。参数：无；返回：None；异常：失锁或连接错误。"""
        if self.lease is None:
            raise RuntimeError("归并写入必须持有数据库租约")
        await self.lease.assert_held()
        await self.lease.connection.commit()

    async def pending_page(self, after_id: int, limit: int) -> list[dict]:
        """参数：非负游标和正数页长；返回：未归并原件；异常：参数或数据库错误。"""
        if after_id < 0 or limit < 1:
            raise ValueError("分页边界非法")
        async with self.factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.id > after_id, LowConfidenceQuestion.matched_review_id.is_(None))
                .order_by(LowConfidenceQuestion.id).limit(limit))).all()
            return [{"id": row.id, "raw_question": row.raw_question} for row in rows]

    async def candidate_page(self, after_id: int, limit: int) -> list[ReviewCandidate]:
        """参数：游标/页长；返回：仅未冻结待审页；异常：失锁、参数或数据库错误。"""
        if after_id < 0 or limit < 1:
            raise ValueError("分页边界非法")
        await self.assert_held()
        async with AsyncSession(bind=self.lease.connection, expire_on_commit=False) as session:
            rows = (await session.scalars(select(ReviewQueue).where(ReviewQueue.id > after_id,
                ReviewQueue.review_status == "待审", ReviewQueue.approved_answer.is_(None))
                .order_by(ReviewQueue.id).limit(limit))).all()
            return [ReviewCandidate(id=row.id, normalized_question=row.normalized_question) for row in rows]

    async def _source_id(self, pool_id):
        """只按原消息pool_ids定位精确轮次，不按同文本猜测。参数：池主键；返回：用户消息ID或None；异常：落点损坏/数据库错误。"""
        async with self.factory() as session:
            pool = await session.get(LowConfidenceQuestion, pool_id)
            if pool is None:
                raise ValueError("问题池原件不存在")
            if pool.conversation_id is None:
                return None
            # 老数据没有轮次列；只遍历同会话用户行中的精确主键映射。
            sources = (await session.scalars(select(Message).where(Message.conversation_id == pool.conversation_id,
                Message.role == "user"))).all()
            identifiers = [row.id for row in sources
                if (row.rag_metadata or {}).get("knowledge_pool", {}).get("pool_ids", {}).get(pool.source) == pool_id]
            if len(identifiers) > 1:
                raise ValueError("池主键指向多个原轮")
            return identifiers[0] if identifiers else None

    async def commit_match(self, pool_id: int, normalized: NormalizedGap, matched_review_id: int | None) -> dict:
        """同连接原子锁定用户→池→队列，历史计数落点优先。

        参数：池ID、严格标准结果、当前完整查重判定。
        返回：processed/merged和最终review_id；异常：CandidateChanged要求重判，其余错误完整回滚。
        """
        source_id = await self._source_id(pool_id)
        await self.assert_held()
        async with AsyncSession(bind=self.lease.connection, expire_on_commit=False) as session:
            async with session.begin():
                source = await session.scalar(select(Message).where(Message.id == source_id).with_for_update()) if source_id else None
                metadata = dict(source.rag_metadata or {}) if source is not None else {}
                state = TurnPoolState.model_validate(metadata.get("knowledge_pool") or {})
                identifiers = sorted({pool_id, *state.pool_ids.values()})
                pools = (await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.id.in_(identifiers))
                    .order_by(LowConfidenceQuestion.id).with_for_update())).all()
                pool = next((row for row in pools if row.id == pool_id), None)
                if pool is None or len(pools) != len(identifiers):
                    raise ValueError("原轮问题池落点损坏")
                if source is not None and (state.pool_ids.get(pool.source) != pool_id or any(
                        row.conversation_id != source.conversation_id or row.raw_question != source.content for row in pools)):
                    raise ValueError("原轮问题池身份不一致")
                if pool.matched_review_id is not None:
                    return {"processed": False, "merged": False, "review_id": pool.matched_review_id}
                prior = {row.matched_review_id for row in pools if row.matched_review_id is not None}
                prior.update(value for value in (state.counted_review_id, state.matched_review_id) if value is not None)
                if len(prior) > 1:
                    raise ValueError("同轮存在冲突的历史计数落点")
                target = next(iter(prior)) if prior else matched_review_id
                review = await session.scalar(select(ReviewQueue).where(ReviewQueue.id == target).with_for_update()) if target else None
                if target is not None and review is None:
                    raise CandidateChanged("历史或候选缺口已消失")
                if not prior and review is not None and (review.review_status != "待审" or review.approved_answer is not None):
                    raise CandidateChanged("候选已审核或冻结，重新查重")
                created = review is None
                if created:
                    review = ReviewQueue(normalized_question=normalized.normalized_question,
                        ai_suggested_answer=normalized.ai_suggested_answer, occurrence_count=1)
                    session.add(review)
                    await session.flush()
                elif state.counted_review_id != review.id:
                    review.occurrence_count += 1
                pool.matched_review_id = review.id
                if source is not None:
                    state.matched_review_id = state.counted_review_id = review.id
                    source.rag_metadata = {**metadata, "knowledge_pool": state.model_dump(mode="json")}
                return {"processed": True, "merged": not created, "review_id": review.id}
