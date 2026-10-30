"""持久问题池后台恢复worker，完整候选分页、失败保留原件和有限关闭。"""

import asyncio
from contextlib import nullcontext
from contextvars import Context
import logging
from uuid import uuid4

from langchain_core.runnables import RunnableLambda

from app.observability.runtime import TraceIdentity
from app.repositories.flywheel import CandidateChanged

logger = logging.getLogger(__name__)


class FlywheelService:
    """后台扫描和手动触发共用同一持久输入处理方法，数据库命名锁负责跨进程串行。"""

    def __init__(self, repository, model, *, page_size=50, interval_seconds=30.0,
                 timeout_seconds=180.0, close_seconds=5.0, observability=None):
        """参数：仓储/模型、候选页长、扫描/单原件/关闭秒数及观测运行时；返回：None；异常：参数错误。"""
        if page_size < 1 or min(interval_seconds, timeout_seconds) <= 0 or close_seconds < 0:
            raise ValueError("飞轮分页和时间限制非法")
        self.repository, self.model = repository, model
        self.page_size, self.interval_seconds = page_size, interval_seconds
        self.timeout_seconds, self.close_seconds = timeout_seconds, close_seconds
        self.observability = observability
        self._closing, self._background = False, None
        self._after_id = 0
        self._active, self._serial = set(), asyncio.Lock()
        self._invoke = RunnableLambda(self._process_one).with_config(
            callbacks=[observability.callback_handler] if observability is not None else [])

    async def _process_one(self, row):
        """参数：持久原件；返回：原子归并结果；异常：模型/失锁/数据库错误，不丢原件。"""
        async with self.repository.locked() as repo:
            normalized = await self.model.normalize(row["raw_question"])
            await repo.assert_held()
            for attempt in range(3):
                after, matched = 0, None
                while True:
                    page = await repo.candidate_page(after, self.page_size)
                    if not page:
                        break
                    decision = await self.model.match(normalized.normalized_question, page)
                    await repo.assert_held()
                    if decision.matched_review_id is not None:
                        if decision.matched_review_id not in {item.id for item in page}:
                            raise ValueError("模型伪造当前页候选ID")
                        matched = decision.matched_review_id
                        break
                    after = page[-1].id
                try:
                    return await repo.commit_match(row["id"], normalized, matched)
                except CandidateChanged:
                    # 状态变化不能把旧判定改为无重复；从第一页重新读取和调用模型。
                    if attempt == 2:
                        raise

    async def process_pending(self, limit: int = 50) -> dict:
        """参数：本次最多原件数；返回：processed/merged/failed计数；异常：扫描失败或取消传播。"""
        if limit < 1:
            raise ValueError("处理上限必须为正数")
        result = {"processed": 0, "merged": 0, "failed": 0}
        task = asyncio.current_task()
        self._active.add(task)
        try:
            async with self._serial:
                if self._closing:
                    return result
                pending = await self.repository.pending_page(self._after_id, limit)
                if not pending and self._after_id:
                    self._after_id = 0
                    pending = await self.repository.pending_page(0, limit)
                for row in pending:
                    if self._closing:
                        break
                    try:
                        identity = TraceIdentity(uuid4().hex, f"pool:{row['id']}", f"flywheel:{row['id']}", "background")
                        boundary = self.observability.execution(identity, kind="flywheel") if self.observability else nullcontext()
                        with boundary:
                            async with asyncio.timeout(self.timeout_seconds):
                                saved = await self._invoke.ainvoke(row)
                        result["processed"] += int(saved["processed"])
                        result["merged"] += int(saved["merged"])
                    except Exception as error:
                        result["failed"] += 1
                        logger.warning("flywheel failed pool_id=%s error_type=%s", row["id"], type(error).__name__)
                    self._after_id = row["id"]
                return result
        finally:
            self._active.discard(task)

    def start(self):
        """立即登记恢复扫描，不等待模型。参数：无；返回：None；异常：缺少事件循环。"""
        if not self._closing and (self._background is None or self._background.done()):
            self._background = asyncio.create_task(self._run(), context=Context())

    async def _run(self):
        """参数：无；返回：None；异常：取消传播，扫描失败记录类型后下轮重试。"""
        while not self._closing:
            try:
                await self.process_pending()
            except Exception as error:
                logger.warning("flywheel scan failed error_type=%s", type(error).__name__)
            await asyncio.sleep(self.interval_seconds)

    async def close(self):
        """停止接单，宽限后取消工作并等待租约清理。参数：无；返回：None；异常：清理/外部取消传播。"""
        from app.knowledge.lease import finish_cleanup
        self._closing = True
        tasks = self._active | ({self._background} if self._background is not None else set())
        tasks.discard(asyncio.current_task())
        if not tasks:
            return
        async def settle():
            """参数：无；返回：None；异常：内部清理超时。"""
            _, pending = await asyncio.wait(tasks, timeout=self.close_seconds)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await finish_cleanup(settle(), timeout_seconds=self.close_seconds + 15.0)
