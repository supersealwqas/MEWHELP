"""人工答案冻结、CH03稳定发布和持久断点恢复；不调用抽取或审核模型。"""

import asyncio
from contextlib import asynccontextmanager
from contextvars import Context
import logging

from app.knowledge.limits import knowledge_overflow
from app.knowledge.schemas import ChunkDraft, KnowledgeText, KnowledgeUnavailable, Publication
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint
from app.repositories.flywheel import ReviewConflict

logger = logging.getLogger(__name__)
CATEGORY = "商品与购物 FAQ"


class PublicationPending(KnowledgeUnavailable):
    """人工答案已持久冻结，发布请求未完整确认；完成提交后的错误须回读持久状态。"""


class ReviewPublisher:
    """复用在线编码器、索引和CH03存储；数据库冻结状态是恢复依据。"""

    def __init__(self, repository, store, encoder, index, *, page_size=50,
                 interval_seconds=30.0, timeout_seconds=180.0, close_seconds=5.0):
        """保存共享依赖和恢复限制。参数：仓储/存储/编码器/索引及页长、间隔、超时/关闭秒数；返回：None；异常：ValueError非法限制。"""
        if not 1 <= page_size <= 200 or min(interval_seconds, timeout_seconds) <= 0 or close_seconds < 0:
            raise ValueError("审核发布分页和时间限制非法")
        self.repository, self.store, self.encoder, self.index = repository, store, encoder, index
        self.synchronizer = VectorSynchronizer(store, encoder, index)
        self.page_size, self.interval_seconds = page_size, interval_seconds
        self.timeout_seconds, self.close_seconds = timeout_seconds, close_seconds
        self._background, self._closing = None, False
        self._active, self._serial = set(), asyncio.Lock()

    @asynccontextmanager
    async def _operation(self):
        """登记手动与后台调用直到清理结束。参数：无；返回：上下文；异常：关闭冲突、调用异常/取消。"""
        if self._closing:
            raise ReviewConflict("审核发布服务正在关闭")
        task = asyncio.current_task()
        owns = task not in self._active
        self._active.add(task)
        try:
            yield
        finally:
            if owns:
                self._active.discard(task)

    async def approve(self, review_id: int, answer: str) -> dict:
        """用实际编码器校验人工正文后才冻结并发布。参数：审核主键、原样人工答案；返回：审核字段；异常：输入/状态/依赖错误或取消。"""
        async with self._operation():
            async with asyncio.timeout(self.timeout_seconds):
                row = await self.repository.review_detail(review_id)
                if row["review_status"] == "驳回" or (row["approved_answer"] is not None and row["approved_answer"] != answer):
                    raise ReviewConflict("审核已驳回或人工答案已冻结，不能覆盖")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("人工答案不能为空")
                text = KnowledgeText(category=CATEGORY, questions=row["normalized_question"], answer=answer)
                if knowledge_overflow(text, self.encoder.count_tokens):
                    raise ValueError("知识三格合计超过1600字符或格式化输入超过2048 token，请修改人工答案")
                await self.repository.freeze_approval(review_id, answer)
                return await self._publish(review_id)

    async def publish(self, review_id: int) -> dict:
        """重试已冻结项，不自动选用模型示例。参数：审核主键；返回：审核字段；异常：不存在/冲突/发布未完成或取消。"""
        async with self._operation():
            async with asyncio.timeout(self.timeout_seconds):
                return await self._publish(review_id)

    async def _publish(self, review_id: int) -> dict:
        """按稳定来源原样写入并核对当前可见指纹。参数：审核主键；返回：已完成字段；异常：状态冲突或PublicationPending。"""
        async with self._serial:
            if self._closing:
                raise ReviewConflict("审核发布服务正在关闭")
            row = await self.repository.review_detail(review_id)
            if row["approved_answer"] is None or row["review_status"] == "驳回":
                raise ReviewConflict("尚未提交人工核准答案，不能重试发布")
            if row["review_status"] == "通过":
                return {key: value for key, value in row.items() if key != "originals"}
            answer = row["approved_answer"]
            text = KnowledgeText(category=CATEGORY, questions=row["normalized_question"], answer=answer)
            source = f"ch09/review/{review_id}"
            try:
                # 队列冻结事务已经结束，外部编码/索引等待不持有审核行锁。
                async with self.store.writer() as writer:
                    frame = await writer.frame()
                    await writer.publish(Publication(expected_token=frame.token, embedding_revision=self.encoder.revision,
                        drafts=[ChunkDraft(source_key=source, ordinal=0, text=text, content_type="faq")]))
                    # CH03同步处理全库pending，任何其他条目失败同样不能标本条通过。
                    await self.synchronizer.run(writer)
                    await writer.assert_held()
                    chunks = await writer.chunks_by_sources([source])
                    target = self._confirmed_chunk(chunks, text)
                    fingerprints = await self.index.get_fingerprints([target.id])
                    await writer.assert_held()
                    # Strong回读期间再次核对MySQL当前版本，不能用先前快照掩盖变化。
                    current = self._confirmed_chunk(await writer.chunks_by_sources([source]), text)
                    if current.id != target.id or fingerprints.get(current.id) != (current.content_hash, current.embedding_revision):
                        raise PublicationPending("当前知识版本尚未在索引中确认可读")
                    await writer.assert_held()
                    return await self.repository.complete_publication(review_id, answer, writer=writer)
            except Exception as error:
                logger.warning("review publication unconfirmed review_id=%s error_type=%s", review_id, type(error).__name__)
                raise PublicationPending("人工答案已保存，请回读审核状态后重试未完成发布") from error

    def _confirmed_chunk(self, chunks, text):
        """核对本来源唯一active块及全部同步事实。参数：当前快照/人工三格；返回：已完成目标；异常：PublicationPending失配。"""
        active = [chunk for chunk in chunks if chunk.is_active]
        if len(active) != 1:
            raise PublicationPending("审核来源的当前知识块不唯一或不存在")
        row = active[0]
        if (row.ordinal != 0 or row.text != text or row.content_type != "faq"
                or row.content_hash != content_fingerprint(text) or row.embedding_revision != self.encoder.revision
                or row.vectorize_status != "done" or row.vector_id != str(row.id)):
            raise PublicationPending("当前知识正文或向量版本尚未完成同步")
        return row

    async def recover_pending(self) -> dict:
        """分页恢复全部冻结待审项，单项失败保留并继续后页。参数：无；返回：published/failed计数；异常：扫描失败或取消。"""
        result, after = {"published": 0, "failed": 0}, 0
        async with self._operation():
            while not self._closing:
                page = await self.repository.pending_publications(after, self.page_size)
                if not page:
                    break
                for row in page:
                    if self._closing:
                        break
                    try:
                        await self.publish(int(row["id"]))
                        result["published"] += 1
                    except Exception as error:
                        result["failed"] += 1
                        logger.warning("review recovery failed review_id=%s error_type=%s", row["id"], type(error).__name__)
                    after = int(row["id"])
        return result

    def start(self):
        """立即启动持久恢复扫描，不阻塞应用启动。参数：无；返回：None；异常：无运行循环。"""
        if not self._closing and (self._background is None or self._background.done()):
            self._background = asyncio.create_task(self._run(), context=Context())

    async def _run(self):
        """循环恢复未完成发布。参数：无；返回：None；异常：取消传播，扫描错误记录后下轮恢复。"""
        while not self._closing:
            try:
                await self.recover_pending()
            except Exception as error:
                logger.warning("review recovery scan failed error_type=%s", type(error).__name__)
            await asyncio.sleep(self.interval_seconds)

    async def close(self):
        """停止接单并在共享编码器/索引释放前收回工作。参数：无；返回：None；异常：清理失败/超时/取消。"""
        from app.knowledge.lease import finish_cleanup
        self._closing = True
        tasks = self._active | ({self._background} if self._background is not None else set())
        tasks.discard(asyncio.current_task())
        if not tasks:
            return
        async def settle():
            """等待宽限后取消并收回实际任务。参数：无；返回：None；异常：清理超时。"""
            _, pending = await asyncio.wait(tasks, timeout=self.close_seconds)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await finish_cleanup(settle(), timeout_seconds=self.close_seconds + 15.0)
