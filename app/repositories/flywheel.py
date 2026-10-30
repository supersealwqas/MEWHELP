"""知识缺口持久归并与人工审核：原子计数、历史原件读取和冻结/完成短事务。"""

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


class ReviewNotFound(LookupError):
    """指定审核缺口不存在，HTTP边界映射为404。"""


class ReviewConflict(RuntimeError):
    """审核已驳回、正文已冻结或服务关闭，HTTP边界映射为409。"""


def _review_payload(row) -> dict:
    """投影固定审核字段并派生发布状态。参数：队列实体；返回：含字符串主键的dict；异常：无。"""
    publication_status = "published" if row.review_status == "通过" else "pending" if row.approved_answer is not None else "not_started"
    return {"id": str(row.id), "normalized_question": row.normalized_question,
        "ai_suggested_answer": row.ai_suggested_answer, "approved_answer": row.approved_answer,
        "occurrence_count": row.occurrence_count, "review_status": row.review_status,
        "publication_status": publication_status, "created_at": row.created_at, "updated_at": row.updated_at}


class FlywheelRepository:
    """管理问题池归并与审核队列，短事务冻结人工正文并支持持久发布恢复。"""

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

    async def list_reviews(self, status: str | None = None, after_id: int = 0, limit: int = 50) -> dict:
        """按稳定主键分页待审队列。参数：可选中文状态、非负游标、1–200页长；返回：items/next_after_id；异常：参数或数据库错误。"""
        if status not in (None, "待审", "通过", "驳回") or after_id < 0 or not 1 <= limit <= 200:
            raise ValueError("审核分页参数非法")
        query = select(ReviewQueue).where(ReviewQueue.id > after_id)
        if status is not None:
            query = query.where(ReviewQueue.review_status == status)
        async with self.factory() as session:
            rows = (await session.scalars(query.order_by(ReviewQueue.id).limit(limit + 1))).all()
            return {"items": [_review_payload(row) for row in rows[:limit]],
                "next_after_id": str(rows[limit - 1].id) if len(rows) > limit else None}

    async def review_detail(self, review_id: int) -> dict:
        """读取审核和所有历史原件，绝不重新检索补快照。参数：正审核主键；返回：字段及originals；异常：ReviewNotFound或数据库错误。"""
        async with self.factory() as session:
            row = await session.get(ReviewQueue, review_id)
            if row is None:
                raise ReviewNotFound("审核缺口不存在")
            originals = (await session.scalars(select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.matched_review_id == review_id).order_by(LowConfidenceQuestion.id))).all()
            return {**_review_payload(row), "originals": [{"id": str(item.id), "raw_question": item.raw_question,
                "source": item.source, "reason": item.reason, "created_at": item.created_at,
                "retrieved_chunks": item.retrieved_chunks} for item in originals]}

    async def freeze_approval(self, review_id: int, answer: str) -> dict:
        """短行锁冻结人工答案；相同正文幂等，不同正文拒绝。参数：审核主键、非空原样答案；返回：审核字段；异常：不存在/冲突/输入或数据库错误。"""
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("人工答案不能为空")
        async with self.factory.begin() as session:
            row = await session.scalar(select(ReviewQueue).where(ReviewQueue.id == review_id).with_for_update())
            if row is None:
                raise ReviewNotFound("审核缺口不存在")
            if row.review_status == "驳回" or (row.approved_answer is not None and row.approved_answer != answer):
                raise ReviewConflict("审核已驳回或人工答案已冻结，不能覆盖")
            if row.approved_answer is None:
                if row.review_status != "待审":
                    raise ReviewConflict("当前审核状态不能冻结")
                row.approved_answer = answer
                await session.flush()
                # MySQL更新时间由服务器生成，显式回读避免异步隐式IO。
                await session.refresh(row)
            return _review_payload(row)

    async def reject(self, review_id: int) -> dict:
        """短行锁驳回未冻结项；重复驳回幂等。参数：审核主键；返回：审核字段；异常：不存在/冲突或数据库错误。"""
        async with self.factory.begin() as session:
            row = await session.scalar(select(ReviewQueue).where(ReviewQueue.id == review_id).with_for_update())
            if row is None:
                raise ReviewNotFound("审核缺口不存在")
            if row.approved_answer is not None or row.review_status == "通过":
                raise ReviewConflict("人工答案已冻结，不能驳回")
            if row.review_status != "驳回":
                row.review_status = "驳回"
                await session.flush()
                await session.refresh(row)
            return _review_payload(row)

    async def pending_publications(self, after_id: int = 0, limit: int = 50) -> list[dict]:
        """读取持久冻结但尚未完成的发布。参数：非负游标、1–200页长；返回：按ID排序审核字段；异常：参数/数据库错误。"""
        if after_id < 0 or not 1 <= limit <= 200:
            raise ValueError("发布分页参数非法")
        async with self.factory() as session:
            rows = (await session.scalars(select(ReviewQueue).where(ReviewQueue.id > after_id,
                ReviewQueue.review_status == "待审", ReviewQueue.approved_answer.is_not(None))
                .order_by(ReviewQueue.id).limit(limit))).all()
            return [_review_payload(row) for row in rows]

    async def complete_publication(self, review_id: int, expected_answer: str) -> dict:
        """可见性已确认后短行锁完成审核，并再次比较冻结正文。参数：主键、预期正文；返回：审核字段；异常：不存在/冲突或数据库错误。"""
        async with self.factory.begin() as session:
            row = await session.scalar(select(ReviewQueue).where(ReviewQueue.id == review_id).with_for_update())
            if row is None:
                raise ReviewNotFound("审核缺口不存在")
            if row.review_status not in ("待审", "通过") or row.approved_answer is None or row.approved_answer != expected_answer:
                raise ReviewConflict("冻结答案或审核状态已变化")
            if row.review_status != "通过":
                row.review_status = "通过"
                await session.flush()
                await session.refresh(row)
            return _review_payload(row)
