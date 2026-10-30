"""会话摘要后台管理：用量触发、冻结区间、追加保存、有限并发与停机/重启恢复。"""

import asyncio
import logging
from contextvars import Context
from contextlib import nullcontext
from time import monotonic
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from app.observability.runtime import TraceIdentity

from app.core.context_budget import TokenEstimator
from app.core.context_layers import build_context_snapshot, history_blocks, project_summaries
from app.repositories.context import ContextRepository, _identifier
from app.workflow.context import shared_budget
from app.workflow.summarization import validate_summary

logger = logging.getLogger(__name__)


class SummaryManager:
    """应用级单实例任务池；不持聊天锁、请求session或checkpoint引用。"""

    def __init__(self, repository: ContextRepository, generator, settings, *, retry_seconds=60.0,
                 close_seconds=5.0, timeout_seconds=90.0, concurrency=2, max_pending=32, observability=None):
        """参数：独立仓储/生成器/窗口配置、退避/关闭/单批超时秒数、并发/排队上限、可选观测运行时；返回：None；异常：ValueError。"""
        if min(retry_seconds, timeout_seconds) <= 0 or close_seconds < 0 or not 1 <= concurrency <= max_pending:
            raise ValueError("摘要任务池需合法的有界并发与时间限制")
        self.repository, self.generator = repository, generator
        self.budget = shared_budget(settings)
        self.estimator = TokenEstimator(settings.context_non_ascii_tokens)
        self.retry_seconds, self.close_seconds, self.timeout_seconds = retry_seconds, close_seconds, timeout_seconds
        self.max_pending, self._slots = max_pending, asyncio.Semaphore(concurrency)
        self._tasks, self._retry_after, self._dirty = {}, {}, set()
        self._closing, self._recovery = False, None
        self.observability = observability
        self._generation = RunnableLambda(self._generate_batch).with_config(
            callbacks=[observability.callback_handler] if observability is not None else [])

    async def _generate_batch(self, batch):
        """让后台模型继承唯一回调。参数：含messages/background的本批输入；返回：摘要草稿；异常：模型错误原样传播。"""
        return await self.generator.generate(batch["messages"], batch["background"])

    @property
    def active_count(self) -> int:
        """返回尚未收回任务数。参数：无；返回：int，含恢复扫描；异常：无。"""
        return len(self._tasks) + int(self._recovery is not None and not self._recovery.done())

    async def maybe_schedule(self, conversation_id: int) -> bool:
        """立即登记后台工作，不等待数据库或模型。参数：可信会话ID；返回：是否接受；异常：非法ID。"""
        _identifier(conversation_id)
        reason = None
        if self._closing:
            reason = "closing"
        elif conversation_id in self._tasks:
            self._dirty.add(conversation_id)
            reason = "in_flight"
        elif monotonic() < self._retry_after.get(conversation_id, 0):
            reason = "backoff"
        elif len(self._tasks) >= self.max_pending:
            reason = "queue_full"
        if reason:
            logger.info("summary skip conversation=%s range=unloaded reason=%s elapsed_ms=0", conversation_id, reason)
            return False
        # 此前没有await，同事件循环内检查与登记不可被另一聊天请求插入。
        # 后台工作不能继承触发它的客服trace或LangChain回调上下文。
        self._tasks[conversation_id] = asyncio.create_task(self._run(conversation_id), context=Context())
        return True

    async def _run(self, conversation_id: int) -> None:
        """在全局槽位内处理一次冻结区间。参数：会话ID；返回：None；异常：失败记录后退避，取消原样传播。"""
        started, first, upto = monotonic(), 0, 0
        try:
            async with self._slots:
                stored = await self.repository.load_context(conversation_id)
                anchor, upto = stored["summary_upto_msg_id"], stored["layer1_from_msg_id"]
                blocks = history_blocks(stored["turns"])
                if any(block.first_id <= boundary < block.last_id for block in blocks for boundary in (anchor, upto)):
                    raise ValueError("摘要冻结边界不能切开相交轮组")
                blocks = [block for block in blocks if anchor < block.last_id <= upto]
                turns = [turn for block in blocks for turn in block.turns]
                snapshot = build_context_snapshot(turns, summary_upto_msg_id=anchor, layer1_from_msg_id=upto,
                    summaries=stored["summaries"], budget=self.budget, estimator=self.estimator)
                if not snapshot.summary_required:
                    logger.info("summary skip conversation=%s from=%s upto=%s reason=within_budget tokens=%s budget=%s elapsed_ms=%.1f",
                        conversation_id, anchor, upto, snapshot.layer2_tokens, self.budget.layer2, (monotonic()-started)*1000)
                    return
                first = blocks[0].first_id
                logger.info("summary trigger conversation=%s from=%s upto=%s 层2约%s token > 预算%s elapsed_ms=%.1f",
                    conversation_id, first, upto, snapshot.layer2_tokens, self.budget.layer2, (monotonic()-started)*1000)
                segments, offset = list(stored["summaries"]), 0
                while offset < len(blocks):
                    background, _ = project_summaries(segments, self.budget.summary_tokens, self.estimator)
                    batch, end = [], offset
                    # 按相交轮组装批，块内原文按主键排序；延迟回执与中间新问不能分开摘要。
                    while end < len(blocks):
                        candidate = [*batch, *blocks[end].messages]
                        if not self.generator.fits(candidate, background):
                            break
                        batch, end = candidate, end + 1
                    if not batch:
                        self._retry_after[conversation_id] = monotonic() + self.retry_seconds
                        logger.warning("summary skip conversation=%s from=%s upto=%s reason=oversized_turn elapsed_ms=%.1f",
                            conversation_id, blocks[offset].first_id, blocks[offset].last_id,
                            (monotonic()-started)*1000)
                        return
                    first, last = batch[0].id, batch[-1].id
                    logger.info("summary start conversation=%s from=%s upto=%s elapsed_ms=%.1f",
                                conversation_id, first, last, (monotonic()-started)*1000)
                    async with asyncio.timeout(self.timeout_seconds):
                        identity = TraceIdentity(uuid4().hex, f"conversation:{conversation_id}",
                                                 f"summary:{first}:{last}", "background")
                        boundary = self.observability.execution(identity, kind="summary") if self.observability else nullcontext()
                        with boundary:
                            draft = await self._generation.ainvoke({"messages": batch, "background": background})
                    validate_summary(draft, batch)
                    saved = await self.repository.append_summary(conversation_id, expected_summary_upto=anchor,
                        from_msg_id=first, upto_msg_id=last, content=draft.content,
                        projection_max_tokens=self.budget.summary_tokens)
                    if saved is None:
                        logger.info("summary skip conversation=%s from=%s upto=%s reason=stale_range elapsed_ms=%.1f",
                                    conversation_id, first, last, (monotonic()-started)*1000)
                        return
                    logger.info("summary done 第%s段 conversation=%s from=%s upto=%s no_facts=%s elapsed_ms=%.1f",
                        saved["seq"], conversation_id, first, last, not saved["content"], (monotonic()-started)*1000)
                    segments.append(saved)
                    anchor, offset = last, end
                self._retry_after.pop(conversation_id, None)
        except asyncio.CancelledError:
            logger.info("summary skip conversation=%s from=%s upto=%s reason=cancelled elapsed_ms=%.1f",
                        conversation_id, first, upto, (monotonic()-started)*1000)
            raise
        except Exception as error:
            self._retry_after[conversation_id] = monotonic() + self.retry_seconds
            logger.warning("summary fail conversation=%s from=%s upto=%s type=%s elapsed_ms=%.1f",
                           conversation_id, first, upto, type(error).__name__, (monotonic()-started)*1000)
        finally:
            self._tasks.pop(conversation_id, None)
            if conversation_id in self._dirty:
                self._dirty.discard(conversation_id)
                await self.maybe_schedule(conversation_id)

    async def recover(self) -> None:
        """登记异步积压扫描，启动过程不等待模型。参数：无；返回：None；异常：无。"""
        if not self._closing and (self._recovery is None or self._recovery.done()):
            self._recovery = asyncio.create_task(self._recover())

    async def _recover(self) -> None:
        """分页重建待办并服从队列容量。参数：无；返回：None；异常：读取失败留痕，取消传播。"""
        try:
            after = 0
            while not self._closing:
                identifiers = await self.repository.pending_summary_ids(after_id=after, limit=self.max_pending)
                if not identifiers:
                    return
                for identifier in identifiers:
                    while len(self._tasks) >= self.max_pending and not self._closing:
                        await asyncio.wait(list(self._tasks.values()), return_when=asyncio.FIRST_COMPLETED)
                    if self._closing:
                        return
                    await self.maybe_schedule(identifier)
                after = identifiers[-1]
        except Exception as error:
            logger.warning("summary fail reason=recovery type=%s", type(error).__name__)

    async def close(self) -> None:
        """停止接受、有限等待后取消并收回任务。参数：无；返回：None；异常：外部取消由生命周期屏蔽。"""
        self._closing = True
        tasks = [*self._tasks.values(), *([self._recovery] if self._recovery is not None else [])]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=self.close_seconds)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._dirty.clear()
        self._retry_after.clear()
