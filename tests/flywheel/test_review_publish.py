"""人工审核冻结与 CH03 稳定发布的断点恢复合同；外部向量使用内存替身。"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.models.flywheel import ReviewQueue
from app.models.rag_quality import LowConfidenceQuestion
from tests.flywheel.test_merge import runtime
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, MemoryVectorIndex


@asynccontextmanager
async def review_runtime(tmp_path, monkeypatch, **options):
    """组装真实审核仓储和发布器。参数：临时路径/补丁/配置；返回：依赖；异常：数据库错误。"""
    from app.services.review_publish import ReviewPublisher
    async with runtime(tmp_path, monkeypatch) as (_, repo, _, factory):
        async with factory.begin() as session:
            row = ReviewQueue(normalized_question="运费如何核实？", ai_suggested_answer="模型示例，不得自动发布")
            session.add(row)
            await session.flush()
        store, encoder, index = MemoryKnowledgeStore(), FakeEncoder(), MemoryVectorIndex()
        from tests.knowledge.fakes import MemoryKnowledgeWriter
        @asynccontextmanager
        async def transaction(writer):
            """为内存知识替身提供真实审核事务。参数：写入器；返回：审核会话；异常：数据库错误。"""
            async with factory.begin() as session:
                yield session
        monkeypatch.setattr(MemoryKnowledgeWriter, "transaction", transaction, raising=False)
        publisher = ReviewPublisher(repo, store, encoder, index, **options)
        try:
            yield SimpleNamespace(repo=repo, factory=factory, store=store, encoder=encoder,
                index=index, publisher=publisher, review_id=row.id)
        finally:
            await publisher.close()


@pytest.mark.parametrize("kind", ["characters", "tokens"])
async def test_approval_rejects_overflow_before_freezing(tmp_path, monkeypatch, kind):
    """超字符或实际编码计数超限均不可先冻结。参数：测试资源/限额；返回：None；异常：断言失败。"""
    async with review_runtime(tmp_path, monkeypatch) as env:
        answer = "字" * 1600 if kind == "characters" else "短答案"
        if kind == "tokens":
            monkeypatch.setattr(env.encoder, "count_tokens", lambda value: 2049)
        with pytest.raises(ValueError, match="超"):
            await env.publisher.approve(env.review_id, answer)
        assert (await env.repo.review_detail(env.review_id))["approved_answer"] is None
        assert not env.store._snapshots
        monkeypatch.setattr(env.encoder, "count_tokens", len)
        assert (await env.publisher.approve(env.review_id, "可修正的人工答复"))["review_status"] == "通过"


async def test_approval_freezes_human_answer_without_model_rewrite(tmp_path, monkeypatch):
    """人工正文和稳定来源必须逐字保存。参数：测试资源；返回：None；异常：断言失败。"""
    async with review_runtime(tmp_path, monkeypatch) as env:
        human = "  人工核准：先核对地区。\n不承诺免费，保留 199 元条件。  "
        result = await env.publisher.approve(env.review_id, human)
        assert result["approved_answer"] == human and result["publication_status"] == "published"
        row = next(iter(env.store._snapshots.values()))
        assert row.source_key == f"ch09/review/{env.review_id}" and row.ordinal == 0
        assert row.text.answer == human and row.text.category == "商品与购物 FAQ" and row.content_type == "faq"
        assert (await env.publisher.approve(env.review_id, human)) == result
        assert len(env.store._snapshots) == 1


@pytest.mark.parametrize("point", ["mysql", "vector", "review"])
async def test_crash_recovers_same_chunk(tmp_path, monkeypatch, point):
    """MySQL写后、向量写后、审核完成前中断均复用同一主键。参数：资源/断点；返回：None；异常：断言失败。"""
    from app.services.review_publish import PublicationPending
    async with review_runtime(tmp_path, monkeypatch) as env:
        original_encode, original_complete = env.encoder.encode, env.repo.complete_publication
        async def crash(*args, **kwargs):
            """模拟进程在明确阶段中断。参数：被替代接口参数；返回：不返回；异常：RuntimeError。"""
            raise RuntimeError("断点")
        if point == "mysql":
            env.encoder.encode = crash
        elif point == "vector":
            env.index.fail_after_upsert = True
        else:
            env.repo.complete_publication = crash
        with pytest.raises(PublicationPending):
            await env.publisher.approve(env.review_id, "人工答案")
        row = (await env.repo.review_detail(env.review_id))
        assert row["review_status"] == "待审" and row["approved_answer"] == "人工答案"
        first_id = next(iter(env.store._snapshots))
        env.encoder.encode, env.repo.complete_publication = original_encode, original_complete
        env.index.fail_after_upsert = False
        from app.services.review_publish import ReviewPublisher
        restarted = ReviewPublisher(env.repo, env.store, env.encoder, env.index)
        try:
            assert await restarted.recover_pending() == {"published": 1, "failed": 0}
        finally:
            await restarted.close()
        assert set(env.store._snapshots) == {first_id} and env.index.ids == {first_id}
        assert sum(row.is_active for row in env.store._snapshots.values()) == 1
        assert (await env.repo.review_detail(env.review_id))["review_status"] == "通过"


async def test_frozen_publication_cannot_be_rejected_or_overwritten(tmp_path, monkeypatch):
    """已冻结人工正文不能改写或驳回；同答案冻结可重试。参数：资源；返回：None；异常：断言失败。"""
    from app.repositories.flywheel import ReviewConflict
    async with review_runtime(tmp_path, monkeypatch) as env:
        first = await env.repo.freeze_approval(env.review_id, "冻结人工答")
        assert await env.repo.freeze_approval(env.review_id, "冻结人工答") == first
        with pytest.raises(ReviewConflict):
            await env.repo.freeze_approval(env.review_id, "另一个答案")
        with pytest.raises(ReviewConflict):
            await env.repo.reject(env.review_id)
        assert (await env.repo.review_detail(env.review_id))["publication_status"] == "pending"


@pytest.mark.parametrize("fault", ["missing", "hash", "revision", "database", "lease"])
async def test_review_passes_only_after_current_version_is_searchable(tmp_path, monkeypatch, fault):
    """同步返回后目标指纹/数据库/租约不符仍保持待审。参数：资源/故障；返回：None；异常：断言失败。"""
    from app.services.review_publish import PublicationPending
    async with review_runtime(tmp_path, monkeypatch) as env:
        normal = env.index.get_fingerprints
        async def fingerprints(ids):
            """在最终目标回读注入不可见或失配。参数：主键；返回：指纹；异常：模拟失锁。"""
            if fault == "lease":
                raise RuntimeError("lease lost")
            values = await normal(ids)
            if fault == "database":
                row = env.store._snapshots[ids[0]]
                env.store._snapshots[row.id] = row.model_copy(update={"vector_id": "wrong"})
            if fault == "missing":
                return {}
            if fault in {"hash", "revision"}:
                return {i: ("bad" if fault == "hash" else value[0], "bad" if fault == "revision" else value[1]) for i, value in values.items()}
            return values
        env.index.get_fingerprints = fingerprints
        with pytest.raises(PublicationPending):
            await env.publisher.approve(env.review_id, "人工核准")
        assert (await env.repo.review_detail(env.review_id))["review_status"] == "待审"


async def test_other_pending_sync_failure_keeps_frozen_review_pending(tmp_path, monkeypatch):
    """全库其他pending失败不能只凭本条done改通过。参数：资源；返回：None；异常：断言失败。"""
    from app.knowledge.schemas import ChunkSnapshot, KnowledgeText
    from app.knowledge.text import content_fingerprint
    from app.services.review_publish import PublicationPending
    async with review_runtime(tmp_path, monkeypatch) as env:
        text = KnowledgeText(category="其他", questions="超长旧条目", answer="长" * 1600)
        env.store._snapshots[99] = ChunkSnapshot(id=99, source_key="qa:99", ordinal=0, text=text,
            content_type="faq", content_hash=content_fingerprint(text), embedding_revision=env.encoder.revision)
        with pytest.raises(PublicationPending):
            await env.publisher.approve(env.review_id, "人工核准")
        assert (await env.repo.review_detail(env.review_id))["publication_status"] == "pending"


async def test_detail_preserves_every_original_and_snapshot(tmp_path, monkeypatch):
    """详情保留各来源原问、理由、时间、null或真实快照。参数：资源；返回：None；异常：断言失败。"""
    async with review_runtime(tmp_path, monkeypatch) as env:
        snapshot = {"schema_version": 1, "status": "empty", "attempts": []}
        async with env.factory.begin() as session:
            session.add_all([LowConfidenceQuestion(raw_question="原话一", source="self_check", reason="理由一", matched_review_id=env.review_id),
                LowConfidenceQuestion(raw_question="原话二", source="user_feedback", reason="理由二", retrieved_chunks=snapshot, matched_review_id=env.review_id)])
        detail = await env.repo.review_detail(env.review_id)
        assert [(r["raw_question"], r["source"], r["reason"], r["retrieved_chunks"]) for r in detail["originals"]] == [
            ("原话一", "self_check", "理由一", None), ("原话二", "user_feedback", "理由二", snapshot)]
        assert all(r["created_at"] for r in detail["originals"])
        assert [r["retrieval_status"] for r in detail["originals"]] == ["legacy_unavailable", "empty"]
        assert detail["originals"][0]["retrieval_snapshot"]["status"] == "legacy_unavailable"
        assert detail["originals"][1]["retrieval_snapshot"] == snapshot


async def test_recovery_start_and_close_collects_inflight_publish(tmp_path, monkeypatch):
    """启动恢复冻结项，关闭收回编码等待并保留冻结答案。参数：资源；返回：None；异常：断言失败。"""
    async with review_runtime(tmp_path, monkeypatch, close_seconds=0.01) as env:
        await env.repo.freeze_approval(env.review_id, "待恢复")
        entered = asyncio.Event()
        async def encode(texts):
            """模拟编码器等待。参数：文本；返回：不返回；异常：取消。"""
            entered.set()
            await asyncio.Event().wait()
        env.encoder.encode = encode
        env.publisher.start()
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(env.publisher.close(), 3)
        assert env.publisher._background.done() and not env.publisher._active
        assert (await env.repo.review_detail(env.review_id))["publication_status"] == "pending"
        from app.repositories.flywheel import ReviewConflict
        with pytest.raises(ReviewConflict):
            await env.publisher.approve(env.review_id, "新答案")


def test_review_source_is_published_qa_snapshot():
    """审核知识引用可识别，且继续保存回答时正文。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.sources import answer_metadata
    class Citation:
        """最小引用替身。"""
        source_key, answer = "ch09/review/12", "历史人工正文"
        def model_dump(self):
            """参数：无；返回：快照；异常：无。"""
            return {"source_key": self.source_key, "answer": self.answer}
    result = answer_metadata(SimpleNamespace(session_id="s", turn_id="t"), [Citation()])
    assert result["citations"][0]["source_type"] == "published_qa"
    assert result["citations"][0]["answer"] == "历史人工正文"


@pytest.mark.parametrize("close_fails", [False, True])
async def test_database_runtime_reuses_search_and_closes_publisher_first(tmp_path, monkeypatch, close_fails):
    """生产必须复用当前搜索依赖并先收回发布任务，清理错误不跳过依赖释放。参数：资源/失败开关；返回：None；异常：断言失败。"""
    from contextlib import nullcontext
    from app import main
    from app.knowledge import flywheel
    from tests.flywheel.test_merge import Model
    async with runtime(tmp_path, monkeypatch) as (_, _, _, factory):
        engine = factory.kw["bind"]
        encoder, index, events = FakeEncoder(), MemoryVectorIndex(), []
        @asynccontextmanager
        async def search(*args):
            """只替代外部资源；记录退出顺序。参数：装配签名；返回：搜索依赖；异常：无。"""
            try:
                yield SimpleNamespace(encoder=encoder, index=index)
            finally:
                if publisher is not None:
                    assert publisher._background.done() and not publisher._active
                events.append("search close")
                await encoder.aclose()
                await index.aclose()
        async def schema(config):
            """SQLite不执行生产DDL检查。参数：配置；返回：None；异常：无。"""
        monkeypatch.setattr(main, "check_ch09_tables", schema)
        monkeypatch.setattr(main, "create_database", lambda: (engine, factory))
        monkeypatch.setattr(main, "create_async_engine", lambda *args, **kwargs: engine)
        monkeypatch.setattr(main, "online_search_runtime", search)
        monkeypatch.setattr(main, "settings", main.settings.model_copy(update={
            "workflow_checkpoint_path": (tmp_path / "main.sqlite3").as_posix(), "observability_enabled": False,
            "flywheel_close_seconds": 0.01}))
        monkeypatch.setattr(flywheel, "FlywheelModel", Model)
        publisher = None
        with pytest.raises(RuntimeError, match="publisher cleanup") if close_fails else nullcontext():
            async with main.database_runtime() as service:
                assert hasattr(service, "review_publisher"), "主运行时未装配审核发布器"
                publisher = service.review_publisher
                assert publisher.encoder is encoder and publisher.index is index
                assert publisher.store is publisher.synchronizer.store
                original_close = publisher.close
                async def close():
                    """记录发布任务已真正收回。参数：无；返回：None；异常：可选模拟清理错误。"""
                    await original_close()
                    events.append("publisher close")
                    if close_fails:
                        raise RuntimeError("publisher cleanup")
                monkeypatch.setattr(publisher, "close", close)
        assert events == ["publisher close", "search close"] and encoder.closed and index.closed


def test_main_registers_review_routes():
    """六个审核/处理入口必须实际注册。参数：无；返回：None；异常：断言失败。"""
    from app.main import create_app
    paths = create_app().openapi()["paths"]
    assert "/api/flywheel/reviews" in paths
    assert "/api/flywheel/reviews/{review_id}" in paths
    assert all(f"/api/flywheel/reviews/{{review_id}}/{operation}" in paths for operation in ("approve", "reject", "retry"))
    assert "/api/flywheel/process" in paths


async def test_staging_failure_keeps_frozen_human_answer(tmp_path, monkeypatch):
    """旧暂存原件补证失败也必须保留已冻结人工正文。参数：资源；返回：None；异常：断言失败。"""
    from app.knowledge.schemas import StagedQA
    from app.services.review_publish import PublicationPending
    async with review_runtime(tmp_path, monkeypatch) as env:
        env.store._staged[1] = StagedQA(id=1, source_ids=[999], question="旧暂存", answer="无原件", category="旧", status="extracted")
        with pytest.raises(PublicationPending):
            await env.publisher.approve(env.review_id, "人工新答案")
        assert not env.store._snapshots
        assert (await env.repo.review_detail(env.review_id))["approved_answer"] == "人工新答案"


async def test_complete_publication_rechecks_frozen_answer(tmp_path, monkeypatch):
    """完成短事务不能通过与冻结正文不同的预期。参数：资源；返回：None；异常：断言失败。"""
    from app.repositories.flywheel import ReviewConflict
    async with review_runtime(tmp_path, monkeypatch) as env:
        await env.repo.freeze_approval(env.review_id, "人工原答")
        async with env.store.writer() as writer:
            with pytest.raises(ReviewConflict):
                await env.repo.complete_publication(env.review_id, "另一个答案", writer=writer)
        assert (await env.repo.review_detail(env.review_id))["review_status"] == "待审"


async def test_recovery_failure_does_not_starve_next_page(tmp_path, monkeypatch):
    """首条失败后仍恢复下一页，失败项仍持久保留。参数：资源；返回：None；异常：断言失败。"""
    async with review_runtime(tmp_path, monkeypatch, page_size=1) as env:
        await env.repo.freeze_approval(env.review_id, "失败原件")
        async with env.factory.begin() as session:
            session.add(ReviewQueue(normalized_question="下一条", approved_answer="人工答案"))
        original = env.publisher.publish
        async def publish(identifier):
            """首条注入失败，下一条执行正式发布。参数：ID；返回：审核字段；异常：首条RuntimeError。"""
            if identifier == env.review_id:
                raise RuntimeError("first fails")
            return await original(identifier)
        env.publisher.publish = publish
        assert await env.publisher.recover_pending() == {"published": 1, "failed": 1}
        assert (await env.repo.review_detail(env.review_id))["publication_status"] == "pending"
        assert (await env.repo.review_detail(2))["review_status"] == "通过"


async def test_lease_loss_after_fingerprint_prevents_completion(tmp_path, monkeypatch):
    """最终Strong回读之后失锁不能更新审核通过。参数：资源；返回：None；异常：断言失败。"""
    from app.knowledge.errors import LeaseLost
    from app.services.review_publish import PublicationPending
    from tests.knowledge.fakes import MemoryKnowledgeWriter
    async with review_runtime(tmp_path, monkeypatch) as env:
        read = False
        normal = env.index.get_fingerprints
        async def fingerprints(ids):
            """记录目标回读完成。参数：主键；返回：指纹；异常：无。"""
            nonlocal read
            result = await normal(ids)
            read = True
            return result
        async def held(writer):
            """只在最终回读后模拟真实租约判定失锁。参数：写入器；返回：None；异常：LeaseLost。"""
            if read:
                raise LeaseLost("最终核锁失败")
        env.index.get_fingerprints = fingerprints
        monkeypatch.setattr(MemoryKnowledgeWriter, "assert_held", held)
        with pytest.raises(PublicationPending):
            await env.publisher.approve(env.review_id, "人工答案")
        assert read and (await env.repo.review_detail(env.review_id))["review_status"] == "待审"


async def test_completion_loses_writer_connection_after_successful_check(tmp_path, monkeypatch):
    """最终核锁成功后完成SELECT期间断线，不得借普通工厂提交通过。参数：资源；返回：None；异常：断言失败。"""
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.knowledge.errors import LeaseLost
    from app.knowledge.store import KnowledgeWriter
    from app.services.review_publish import PublicationPending
    from tests.knowledge.fakes import MemoryKnowledgeWriter
    async with review_runtime(tmp_path, monkeypatch) as env:
        connection = await env.factory.kw["bind"].connect()
        class Lease:
            """仅模拟失锁通知，SQLite连接关闭是实际动作。"""
            async def assert_held(self):
                """核验专用连接。参数：无；返回：None；异常：LeaseLost已关闭。"""
                if connection.closed:
                    raise LeaseLost("持锁连接已关闭")
        lease = Lease()
        lease.connection = connection
        actual_writer = KnowledgeWriter(env.factory, lease)
        completed_select = []
        armed = False
        original_scalar, original_fingerprints = AsyncSession.scalar, env.index.get_fingerprints
        async def scalar(session, statement, *args, **kwargs):
            """在最终审核SELECT之前真实关闭专用连接。参数：原会话查询；返回：原查询结果；异常：连接错误。"""
            nonlocal armed
            if armed and getattr(statement, "_for_update_arg", None) is not None:
                armed = False
                completed_select.append(session.bind is connection)
                await connection.close()
            return await original_scalar(session, statement, *args, **kwargs)
        async def fingerprints(ids):
            """最后指纹成功回读之后打开断点。参数：ID；返回：实际内存指纹；异常：无。"""
            nonlocal armed
            result = await original_fingerprints(ids)
            armed = True
            return result
        @asynccontextmanager
        async def writer():
            """内存知识读写配正式持锁事务底座。参数：无；返回：知识写入器；异常：原事务错误。"""
            memory = MemoryKnowledgeWriter(env.store)
            @asynccontextmanager
            async def transaction():
                """只把完成审核事务交给正式CH03底座。参数：无；返回：专用会话；异常：连接错误。"""
                async with actual_writer.transaction() as session:
                    yield session
            memory.transaction = transaction
            memory.assert_held = actual_writer.assert_held
            yield memory
        env.store.writer = writer
        env.index.get_fingerprints = fingerprints
        monkeypatch.setattr(AsyncSession, "scalar", scalar)
        try:
            with pytest.raises(PublicationPending):
                await env.publisher.approve(env.review_id, "仍需持锁完成的人工答案")
            assert completed_select == [True]
            assert (await env.repo.review_detail(env.review_id))["review_status"] == "待审"
        finally:
            await connection.close()


@pytest.mark.parametrize("source", ["retrieval_low_conf", "self_check"])
async def test_legacy_null_without_pool_identity_is_unknown_even_with_same_latest_question(tmp_path, monkeypatch, source):
    """旧NULL不能按同文新轮的明确未检索标记推断。参数：资源/池来源；返回：None；异常：断言失败。"""
    from app.models.customer_service import Conversation, Message
    async with review_runtime(tmp_path, monkeypatch) as env:
        async with env.factory.begin() as session:
            conversation = Conversation(session_id="same", user_id="u")
            session.add(conversation)
            await session.flush()
            pool = LowConfidenceQuestion(conversation_id=conversation.id, raw_question="相同问题", source=source, matched_review_id=env.review_id)
            session.add_all([pool, Message(conversation_id=conversation.id, turn_id="latest", role="user", content="相同问题",
                turn_status="completed", rag_metadata={"retrieval_snapshot": None})])
        original = (await env.repo.review_detail(env.review_id))["originals"][0]
        assert original["retrieved_chunks"] is None
        assert original["retrieval_status"] == "legacy_unavailable"
        assert original["retrieval_snapshot"] == {"schema_version": 1, "status": "legacy_unavailable", "attempts": []}
        async with env.factory() as session:
            assert (await session.get(LowConfidenceQuestion, pool.id)).retrieved_chunks is None


async def test_review_original_explicit_no_retrieval_is_proven_by_exact_pool_identity(tmp_path, monkeypatch):
    """真实完成轮显式None且pool_ids精确匹配时才能展示未检索。参数：资源；返回：None；异常：断言失败。"""
    from app.repositories.chat import ChatRepository
    from tests.flywheel.test_pool import start
    async with review_runtime(tmp_path, monkeypatch) as env:
        chat = ChatRepository(env.factory)
        await chat.finish_turn(await start(chat), "不需要知识检索的原答")
        receipt = await chat.record_unresolved_feedback("s", "u", "first")
        async with env.factory.begin() as session:
            pool = await session.get(LowConfidenceQuestion, int(receipt.pool_id))
            pool.matched_review_id = env.review_id
        original = (await env.repo.review_detail(env.review_id))["originals"][0]
        assert original["retrieved_chunks"] is None
        assert original["retrieval_status"] == "not_retrieved" and original["retrieval_snapshot"] is None


@pytest.mark.parametrize("mode", ["modern", "legacy", "missing", "corrupt", "wrong_identity"])
async def test_review_snapshot_recovers_only_exact_original_turn_without_writes(tmp_path, monkeypatch, mode):
    """精确旧轮可只读恢复，不能取同文新轮；损坏身份和证据保持legacy。参数：资源/历史形式；返回：None；异常：断言失败。"""
    from app.models.customer_service import Conversation, Message
    from sqlalchemy import select
    async with review_runtime(tmp_path, monkeypatch) as env:
        old_snapshot = {"schema_version": 1, "status": "empty", "attempts": [{"attempt_id": "old", "query": "真正旧查询",
            "status": "empty", "candidate_count": 0, "top_k": 5, "chunks": [], "fault": None}]}
        async with env.factory.begin() as session:
            conversation = Conversation(session_id="history", user_id="u")
            session.add(conversation)
            await session.flush()
            pool = LowConfidenceQuestion(conversation_id=conversation.id, raw_question="相同原话", source="self_check", matched_review_id=env.review_id)
            session.add(pool)
            await session.flush()
            metadata = {"knowledge_pool": {"pool_ids": {"self_check": pool.id}}}
            if mode == "modern":
                metadata["retrieval_snapshot"] = old_snapshot
            elif mode == "corrupt":
                metadata["retrieval_snapshot"] = {"破损": True}
            elif mode == "wrong_identity":
                metadata["retrieval_snapshot"] = None
            session.add_all([Message(conversation_id=conversation.id, turn_id="old", role="user",
                content="不属于池的原文" if mode == "wrong_identity" else "相同原话", turn_status="completed", rag_metadata=metadata),
                Message(conversation_id=conversation.id, turn_id="new", role="user", content="相同原话",
                    turn_status="completed", rag_metadata={"retrieval_snapshot": None})])
            if mode == "legacy":
                session.add(Message(conversation_id=conversation.id, turn_id="old", role="assistant", content="历史回答",
                    turn_status="completed", rag_metadata={"citations": [{"question": "旧问法", "answer": "当时原文", "content_hash": "不能当块指纹"}]}))
        async with env.factory() as session:
            before = [(row.id, row.rag_metadata) for row in (await session.scalars(select(Message))).all()]
        original = (await env.repo.review_detail(env.review_id))["originals"][0]
        assert original["retrieved_chunks"] is None
        if mode == "modern":
            assert original["retrieval_status"] == "empty" and original["retrieval_snapshot"] == old_snapshot
        elif mode == "legacy":
            assert original["retrieval_status"] == "legacy_incomplete"
            chunk = original["retrieval_snapshot"]["attempts"][0]["chunks"][0]
            assert chunk["answer"] == "当时原文" and chunk["content_hash"] is None
        else:
            assert original["retrieval_status"] == ("legacy_unavailable" if mode == "missing" else "legacy_incomplete")
        async with env.factory() as session:
            assert before == [(row.id, row.rag_metadata) for row in (await session.scalars(select(Message))).all()]
            assert (await session.get(LowConfidenceQuestion, pool.id)).retrieved_chunks is None


async def test_review_detail_reuses_read_session_for_original_identity(tmp_path, monkeypatch):
    """详情不能持有一个连接再等待第二个连接查原轮，以免并发耗尽连接池。参数：资源；返回：None；异常：断言失败。"""
    async with review_runtime(tmp_path, monkeypatch) as env:
        async with env.factory.begin() as session:
            session.add(LowConfidenceQuestion(raw_question="旧原件", source="self_check", matched_review_id=env.review_id))
        active = False
        @asynccontextmanager
        async def factory():
            """限制单请求只占一个读连接。参数：无；返回：原会话；异常：嵌套连接断言失败。"""
            nonlocal active
            assert not active, "详情读取不应嵌套申请第二个连接"
            active = True
            try:
                async with env.factory() as session:
                    yield session
            finally:
                active = False
        env.repo.factory = factory
        assert (await env.repo.review_detail(env.review_id))["originals"][0]["retrieval_status"] == "legacy_unavailable"
