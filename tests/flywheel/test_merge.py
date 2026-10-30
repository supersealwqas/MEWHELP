"""归并生产仓储与worker的事务边界测试；SQLite只验证写入语义，命名锁另做实库验证。"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import event, select

from app.models.customer_service import Message
from app.models.flywheel import ReviewQueue
from app.models.rag_quality import LowConfidenceQuestion
from app.schemas.flywheel import PoolSignal
from tests.flywheel.test_pool import start
from tests.workflow.service_fakes import service_runtime


class Model:
    """只替代外部语义模型，业务写入和分页由正式代码执行。"""

    def __init__(self, decisions=()):
        """参数：顺序判定或异常；返回：None；异常：无。"""
        self.decisions = list(decisions)
        self.pages = []

    async def normalize(self, question):
        """参数：原问；返回：标准问；异常：无。"""
        from app.schemas.flywheel import NormalizedGap
        return NormalizedGap(normalized_question="签收7天后未拆封商品可以退货吗？", ai_suggested_answer=None)

    async def match(self, question, candidates):
        """参数：问和当前页；返回：指定匹配；异常：显式脚本故障。"""
        from app.schemas.flywheel import MatchDecision
        self.pages.append([candidate.id for candidate in candidates])
        value = self.decisions.pop(0) if self.decisions else candidates[0].id
        if isinstance(value, Exception):
            raise value
        return MatchDecision(matched_review_id=value)


@asynccontextmanager
async def runtime(tmp_path, monkeypatch, model=None, **options):
    """参数：独占目录/补丁/模型/worker配置；返回：真实chat、仓储、worker、工厂；异常：数据库错误。"""
    from app.knowledge.lease import MySqlWriteLease
    from app.repositories.flywheel import FlywheelRepository
    from app.services.flywheel import FlywheelService
    async with service_runtime(tmp_path) as (_, deps, factory):
        engine = factory.kw["bind"]
        async with engine.begin() as connection:
            await connection.exec_driver_sql("CREATE TABLE review_queue (id INTEGER PRIMARY KEY, normalized_question TEXT NOT NULL, ai_suggested_answer TEXT, occurrence_count INTEGER NOT NULL DEFAULT 1, review_status TEXT DEFAULT '待审', approved_answer TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        lock = asyncio.Lock()

        class Lease:
            """SQLite专用连接和本地互斥替身，不宣称证明MySQL会话锁。"""
            async def assert_held(self):
                """参数：无；返回：None；异常：锁已释放时断言失败。"""
                assert lock.locked()

            async def release(self):
                """参数：无；返回：None；异常：连接错误。"""
                await self.connection.close()
                lock.release()

        async def acquire(cls, engine, **kwargs):
            """参数：租约工厂签名；返回：测试连接租约；异常：取消。"""
            await lock.acquire()
            lease = Lease()
            lease.connection = await engine.connect()
            return lease

        monkeypatch.setattr(MySqlWriteLease, "acquire", classmethod(acquire))
        repo = FlywheelRepository(factory, engine)
        worker = FlywheelService(repo, model or Model(), **options)
        try:
            yield deps.chat, repo, worker, factory
        finally:
            await worker.close()


async def seed(chat, turn="first", sources=("self_check",)):
    """参数：仓储、轮次、来源；返回：上下文；异常：数据库错误。"""
    ctx = await start(chat, turn=turn)
    from app.workflow.routing import fixed_reply
    await chat.finish_turn(ctx, fixed_reply("fallback")["answer"],
        pool_signals=[PoolSignal(source=source, reason="真实理由") for source in sources if source != "user_feedback"])
    if "user_feedback" in sources:
        await chat.record_unresolved_feedback("s", "u", turn)
    return ctx


async def rows(factory):
    """参数：工厂；返回：池/队列/原用户消息；异常：数据库错误。"""
    async with factory() as session:
        return ((await session.scalars(select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id))).all(),
                (await session.scalars(select(ReviewQueue).order_by(ReviewQueue.id))).all(),
                (await session.scalars(select(Message).where(Message.role == "user").order_by(Message.id))).all())


async def test_same_turn_three_sources_keep_three_rows_and_count_once(tmp_path, monkeypatch):
    """同轮重复计数或覆盖其他JSON会失败。参数：测试资源；返回：None；异常：断言失败。"""
    async with runtime(tmp_path, monkeypatch) as (chat, repo, worker, factory):
        await seed(chat, sources=("retrieval_low_conf", "self_check", "user_feedback"))
        async with factory.begin() as session:
            source = await session.scalar(select(Message).where(Message.role == "user"))
            source.rag_metadata = {**source.rag_metadata, "trace_identity": {"trace_id": "保留"}, "ticket_intent": {"status": "confirmed"}}
            original = source.rag_metadata
        result = await worker.process_pending()
        pool, queue, users = await rows(factory)
        assert result == {"processed": 3, "merged": 2, "failed": 0}
        assert len(pool) == 3 and len(queue) == 1 and queue[0].occurrence_count == 1
        assert {row.matched_review_id for row in pool} == {queue[0].id}
        assert users[0].rag_metadata["knowledge_pool"]["counted_review_id"] == queue[0].id
        for key in ("trace_identity", "ticket_intent", "completion_inputs"):
            assert users[0].rag_metadata[key] == original[key]


async def test_identical_question_in_two_turns_counts_twice(tmp_path, monkeypatch):
    """按文本错误推断同轮将少计数。参数：测试资源；返回：None；异常：断言失败。"""
    async with runtime(tmp_path, monkeypatch) as (chat, _, worker, factory):
        await seed(chat)
        await seed(chat, "second")
        assert (await worker.process_pending())["failed"] == 0
        pool, queue, _ = await rows(factory)
        assert len(pool) == 2 and len(queue) == 1 and queue[0].occurrence_count == 2


async def test_late_feedback_reuses_reviewed_history_without_increment(tmp_path, monkeypatch):
    """历史落点被审核后晚到来源不应新建或累加。参数：测试资源；返回：None；异常：断言失败。"""
    async with runtime(tmp_path, monkeypatch) as (chat, _, worker, factory):
        await seed(chat, sources=("self_check", "retrieval_low_conf"))
        await worker.process_pending(limit=1)
        async with factory.begin() as session:
            review = await session.scalar(select(ReviewQueue))
            review.review_status, review.approved_answer = "通过", "人工核准原答"
        await chat.record_unresolved_feedback("s", "u", "first")
        await worker.process_pending()
        pool, queue, _ = await rows(factory)
        assert len(queue) == 1 and queue[0].occurrence_count == 1 and len(pool) == 3
        assert {row.matched_review_id for row in pool} == {queue[0].id}


async def test_first_feedback_worker_then_repeat_receipt(tmp_path, monkeypatch):
    """回执重试需稳定pool_id且反映worker新归并ID。参数：测试资源；返回：None；异常：断言失败。"""
    async with runtime(tmp_path, monkeypatch) as (chat, _, worker, factory):
        await chat.finish_turn(await start(chat), "原答")
        first = await chat.record_unresolved_feedback("s", "u", "first")
        assert first.matched_review_id is None
        await worker.process_pending()
        repeat = await chat.record_unresolved_feedback("s", "u", "first")
        pool, queue, _ = await rows(factory)
        assert repeat.pool_id == first.pool_id and repeat.matched_review_id == str(queue[0].id)
        assert len(pool) == 1 and queue[0].occurrence_count == 1


@pytest.mark.parametrize("decision", [RuntimeError("第二页故障"), 999])
async def test_unchecked_candidate_page_cannot_create_new_gap(tmp_path, monkeypatch, decision):
    """后页故障或伪造ID不能被当成无重复。参数：测试资源/错误判定；返回：None；异常：断言失败。"""
    model = Model([None, decision])
    async with runtime(tmp_path, monkeypatch, model, page_size=1) as (chat, _, worker, factory):
        await seed(chat)
        async with factory.begin() as session:
            session.add_all([ReviewQueue(normalized_question="候选一"), ReviewQueue(normalized_question="候选二")])
        assert await worker.process_pending() == {"processed": 0, "merged": 0, "failed": 1}
        pool, queue, _ = await rows(factory)
        assert len(queue) == 2 and pool[0].matched_review_id is None
        assert [row.occurrence_count for row in queue] == [1, 1]


async def test_cross_page_match(tmp_path, monkeypatch):
    """只读第一页会错误新建；本例命中第二页。参数：测试资源；返回：None；异常：断言失败。"""
    async with runtime(tmp_path, monkeypatch, Model([None, 2]), page_size=1) as (chat, _, worker, factory):
        await seed(chat)
        async with factory.begin() as session:
            session.add_all([ReviewQueue(normalized_question="候选一"), ReviewQueue(normalized_question="候选二")])
        assert (await worker.process_pending())["merged"] == 1
        pool, queue, _ = await rows(factory)
        assert pool[0].matched_review_id == 2 and [r.occurrence_count for r in queue] == [1, 2]


async def test_candidate_approved_during_model_call_restarts_match(tmp_path, monkeypatch):
    """模型等待时冻结候选，旧ID不得继续写入。参数：测试资源；返回：None；异常：断言失败。"""
    model = Model([1, 2])
    async with runtime(tmp_path, monkeypatch, model, page_size=1) as (chat, _, worker, factory):
        await seed(chat)
        async with factory.begin() as session:
            session.add_all([ReviewQueue(normalized_question="候选一"), ReviewQueue(normalized_question="候选二")])
        original = model.match
        async def match(question, candidates):
            """参数：当前候选；返回：旧模型判定；异常：数据库错误。"""
            result = await original(question, candidates)
            if result.matched_review_id == 1:
                async with factory.begin() as session:
                    row = await session.get(ReviewQueue, 1)
                    row.approved_answer = "正在发布的人工原答"
            return result
        model.match = match
        assert (await worker.process_pending())["failed"] == 0
        pool, queue, _ = await rows(factory)
        assert pool[0].matched_review_id == 2 and [r.occurrence_count for r in queue] == [1, 2]
        assert model.pages == [[1], [2]]


async def test_transaction_failure_keeps_pool_and_count_unchanged(tmp_path, monkeypatch):
    """提交中的池更新失败必须撤销新建和JSON修改。参数：测试资源；返回：None；异常：断言失败。"""
    async with runtime(tmp_path, monkeypatch) as (chat, _, worker, factory):
        await seed(chat)
        def reject(connection, cursor, statement, parameters, context, executemany):
            """参数：SQL事件；返回：None；异常：模拟写失败。"""
            if statement.startswith("UPDATE low_confidence_questions"):
                raise RuntimeError("atomic rollback")
        engine = factory.kw["bind"].sync_engine
        event.listen(engine, "before_cursor_execute", reject)
        try:
            assert (await worker.process_pending())["failed"] == 1
        finally:
            event.remove(engine, "before_cursor_execute", reject)
        pool, queue, users = await rows(factory)
        assert not queue and pool[0].matched_review_id is None
        assert users[0].rag_metadata["knowledge_pool"]["counted_review_id"] is None


async def test_legacy_same_text_is_independent_history(tmp_path, monkeypatch):
    """没有pool_ids精确落点的旧池不能按同文本少计数。参数：测试资源；返回：None；异常：断言失败。"""
    async with runtime(tmp_path, monkeypatch) as (chat, _, worker, factory):
        await seed(chat)
        async with factory.begin() as session:
            session.add(LowConfidenceQuestion(raw_question="原轮问题", source="self_check"))
        await worker.process_pending()
        _, queue, _ = await rows(factory)
        assert len(queue) == 1 and queue[0].occurrence_count == 2


async def test_worker_start_recovers_and_close_cancels_model(tmp_path, monkeypatch):
    """启动不扫描或取消后锁未释放会失败。参数：测试资源；返回：None；异常：断言失败。"""
    entered = asyncio.Event()
    model = Model()
    async def normalize(question):
        """参数：原问；返回：不返回；异常：取消。"""
        entered.set()
        await asyncio.Event().wait()
    model.normalize = normalize
    async with runtime(tmp_path, monkeypatch, model, close_seconds=0.01) as (chat, _, worker, factory):
        await seed(chat)
        worker.start()
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(worker.close(), 3)
        pool, queue, _ = await rows(factory)
        assert not queue and pool[0].matched_review_id is None


async def test_failed_first_page_does_not_starve_later_pool_rows(tmp_path, monkeypatch):
    """固定从零扫描会让首批坏原件永久阻塞后续问题。参数：测试资源；返回：None；异常：断言失败。"""
    model = Model()
    normal = model.normalize
    async def normalize(question):
        """参数：原问；返回：标准问；异常：指定坏原件失败。"""
        if question == "坏原件":
            raise ValueError("坏原件")
        return await normal(question)
    model.normalize = normalize
    async with runtime(tmp_path, monkeypatch, model) as (_, _, worker, factory):
        async with factory.begin() as session:
            session.add_all([LowConfidenceQuestion(raw_question="坏原件", source="self_check"),
                LowConfidenceQuestion(raw_question="可处理原件", source="self_check")])
        assert (await worker.process_pending(limit=1))["failed"] == 1
        assert (await worker.process_pending(limit=1))["processed"] == 1
        pool, queue, _ = await rows(factory)
        assert pool[0].matched_review_id is None and pool[1].matched_review_id == queue[0].id
        assert (await worker.process_pending(limit=1))["failed"] == 1


@pytest.mark.parametrize("close_fails", [False, True])
async def test_database_runtime_starts_and_closes_production_worker(tmp_path, monkeypatch, close_fails):
    """生产装配漏启动或关闭顺序错误时失败。参数：独占资源；返回：None；异常：断言失败。"""
    from app import main
    async with runtime(tmp_path, monkeypatch) as (chat, _, _, factory):
        await seed(chat)
        engine = factory.kw["bind"]
        @asynccontextmanager
        async def search(*args):
            """参数：搜索装配；返回：外部模型占位；异常：无。"""
            from types import SimpleNamespace
            from tests.knowledge.fakes import FakeEncoder, MemoryVectorIndex
            yield SimpleNamespace(encoder=FakeEncoder(), index=MemoryVectorIndex())
        async def schema(config):
            """参数：配置；返回：None；异常：无，实库结构另验。"""
        monkeypatch.setattr(main, "check_ch09_tables", schema)
        monkeypatch.setattr(main, "create_database", lambda: (engine, factory))
        monkeypatch.setattr(main, "create_async_engine", lambda *args, **kwargs: engine)
        monkeypatch.setattr(main, "online_search_runtime", search)
        monkeypatch.setattr(main, "settings", main.settings.model_copy(update={
            "workflow_checkpoint_path": (tmp_path / "main.sqlite3").as_posix(), "observability_enabled": False}))
        from app.knowledge import flywheel
        monkeypatch.setattr(flywheel, "FlywheelModel", Model)
        disposed = []
        def disposed_engine(engine):
            """参数：已释放引擎；返回：None；异常：无。"""
            disposed.append(engine)
        event.listen(engine.sync_engine, "engine_disposed", disposed_engine)
        from contextlib import nullcontext
        with pytest.raises(RuntimeError, match="cleanup failed") if close_fails else nullcontext():
            async with main.database_runtime() as service:
                async with asyncio.timeout(5):
                    while not (await rows(factory))[1]:
                        await asyncio.sleep(0.01)
                assert await service.flywheel.process_pending() == {"processed": 0, "merged": 0, "failed": 0}
                if close_fails:
                    original_close = service.flywheel.close
                    async def close():
                        """参数：无；返回：不返回；异常：收回worker后模拟清理错误。"""
                        await original_close()
                        raise RuntimeError("cleanup failed")
                    monkeypatch.setattr(service.flywheel, "close", close)
        assert service.flywheel._background.done()
        assert len(disposed) == 2, "飞轮清理失败也必须释放锁引擎和业务引擎"


async def test_forged_candidate_id_rejected():
    """真实适配器不能信任结构合法但本页不存在的ID。参数：无；返回：None；异常：断言失败。"""
    from langchain_core.messages import AIMessageChunk
    from app.knowledge.flywheel import FlywheelModel
    from app.schemas.flywheel import ReviewCandidate
    class Answer:
        """外部模型流的固定结构替身，仍经过正式严格解析。"""
        def bind_tools(self, *args, **kwargs):
            """参数：工具契约；返回：自身；异常：无。"""
            return self
        async def astream(self, messages):
            """参数：消息；返回：伪造ID分片；异常：无。"""
            yield AIMessageChunk(content="", tool_call_chunks=[{"index": 0, "name": "submit_match_decision",
                "id": "decision", "args": '{"matched_review_id":999}'}], response_metadata={"finish_reason": "tool_calls"})
    with pytest.raises(ValueError, match="本页不存在"):
        await FlywheelModel(lambda **kwargs: Answer()).match("运费", [ReviewCandidate(id=1, normalized_question="运费")])
