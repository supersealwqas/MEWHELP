"""CH10真实MySQL专库集成，显式门禁、空库保护与本轮白名单清理。"""

import asyncio
from contextlib import asynccontextmanager
import os

import httpx
import pytest
from sqlalchemy import event, inspect, select, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db.session import create_database
from app.db.migrate_ch10 import migrate_connection, validate_ch10_schema, migrate
from app.db.migrate_rag_quality import QualitySchemaError
from app.knowledge.errors import ConcurrentKnowledgeWrite
from app.knowledge.lease import MySqlWriteLease
from app.models.customer_service import Conversation, Message
from app.models.flywheel import ReviewQueue
from app.models.rag_quality import LowConfidenceQuestion
from app.models.topics import TopicClassification
from app.repositories.topics import TopicRepository, TOPIC_EXECUTION_LOCK
from app.services.topics import TopicBatchService
from tests.topics.test_batch import response

pytestmark = [pytest.mark.mysql, pytest.mark.skipif(os.getenv("MEWHELP_RUN_CH10_MYSQL") != "1",
              reason="必须显式MEWHELP_RUN_CH10_MYSQL=1，且只允许专库")]
TEST_DATABASE = "mewhelp_ch10_test"
OWNED_TABLES = ("topic_classifications", "messages", "low_confidence_questions", "review_queue", "conversations")


@pytest.fixture
async def mysql_env(tmp_path):
    """只进入具名空专库，登记成功CREATE再反序清理。

    参数：tmp_path隔离报告目录；返回：配置/引擎/租约引擎/仓储/工厂。
    异常：库名不符、已有表或连接错误直接失败；只删除本轮成功创建白名单。
    """
    config = Settings(_env_file=os.getenv("MEWHELP_TEST_ENV_FILE", ".env")).model_copy(update={
        "mysql_database": TEST_DATABASE, "topic_enabled": True, "topic_reports_dir": str(tmp_path), "topic_batch_size": 200})
    engine, factory = create_database(config, database=TEST_DATABASE)
    lease_engine = create_async_engine(engine.url, poolclass=NullPool,
        connect_args={"connect_timeout": 5, "init_command": "SET time_zone = '+00:00'"})
    owned = set()
    def remember(connection, cursor, statement, parameters, context, executemany):
        """仅成功CREATE登记本轮所属表。参数：SQLAlchemy事件；返回：None；异常：无。"""
        source = statement.strip().replace("`", "")
        for name in OWNED_TABLES:
            if source.startswith(f"CREATE TABLE {name} "):
                owned.add(name)
    try:
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
            assert await connection.run_sync(lambda sync: inspect(sync).get_table_names()) == [], "专库非空，拒绝修改"
        event.listen(engine.sync_engine, "after_cursor_execute", remember)
        async with engine.begin() as connection:
            for model in (Conversation, ReviewQueue, LowConfidenceQuestion, Message):
                await connection.run_sync(lambda sync, model=model: model.__table__.create(sync))
        yield config, engine, lease_engine, TopicRepository(factory, lease_engine), factory
    finally:
        await lease_engine.dispose()
        try:
            if owned:
                async with engine.begin() as connection:
                    assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
                    for name in OWNED_TABLES:
                        if name in owned:
                            await connection.exec_driver_sql(f"DROP TABLE `{name}`")
        finally:
            await engine.dispose()


async def apply(engine):
    """参数：已门禁引擎；返回：None；异常：DDL或核验错误。"""
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))


async def add_pool(factory, count=1):
    """填充本轮专库池。参数：会话工厂/条数；返回：None；异常：数据库错误。"""
    async with factory.begin() as session:
        session.add_all([LowConfidenceQuestion(id=i, raw_question=f"物流问题{i}", source="self_check")
                         for i in range(1, count + 1)])


async def test_mysql_fixed_ddl_preview_reentry_comments_and_wrong_structure(mysql_env):
    """只读预览、四字段中文注释/约束及重复apply可重入。参数：空专库；返回：None；异常：断言失败。"""
    config, engine, _, _, _ = mysql_env
    async with engine.begin() as connection:
        assert len(await connection.run_sync(migrate_connection)) == 1
        assert not await connection.run_sync(lambda sync: inspect(sync).has_table("topic_classifications"))
    await apply(engine)
    assert await migrate(config, TEST_DATABASE, apply=True) == []
    with pytest.raises(ValueError):
        await migrate(config, "mewhelp_test", apply=True)
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: validate_ch10_schema(inspect(sync)))
        reflected = await connection.run_sync(lambda sync: inspect(sync).get_columns("topic_classifications"))
        assert [(r["name"], r.get("comment")) for r in reflected] == [
            ("id", "主键"), ("question_id", "归类的问题,指向 low_confidence_questions.id"),
            ("labels", "多标签,17 类权威类目里命中的若干个"), ("classified_at", "归类时间")]
        await connection.exec_driver_sql("ALTER TABLE topic_classifications COMMENT='错误结构'")
        with pytest.raises(QualitySchemaError):
            await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))


async def test_mysql_group_union_review_scope_big_id_and_utc_upsert(mysql_env):
    """真实多来源同轮并集/不同轮同文/历史按ID和UTC覆盖。参数：专库；返回：None；异常：断言失败。"""
    _, engine, _, repo, factory = mysql_env
    await apply(engine)
    async with factory.begin() as session:
        session.add(Conversation(id=1, session_id="demo-ch10", user_id="test"))
        session.add_all([ReviewQueue(id=1, normalized_question="通过", review_status="通过"),
                         ReviewQueue(id=2, normalized_question="驳回", review_status="驳回")])
        await session.flush()
        session.add_all([LowConfidenceQuestion(id=i, conversation_id=1, raw_question="同样问题", source=source,
            matched_review_id=review) for i, source, review in [(1, "self_check", None), (2, "user_feedback", None),
             (3, "self_check", None), (4, "self_check", None), (5, "self_check", 1), (6, "self_check", 2)]])
        session.add(LowConfidenceQuestion(id=9007199254740993, raw_question="超大ID", source="self_check"))
        session.add_all([Message(id=10, conversation_id=1, role="user", content="同样问题", turn_id="t1",
            rag_metadata={"knowledge_pool": {"pool_ids": {"self_check": 1, "user_feedback": 2}}}),
            Message(id=11, conversation_id=1, role="user", content="同样问题", turn_id="t2",
            rag_metadata={"knowledge_pool": {"pool_ids": {"self_check": 3}}})])
    async with repo.execution() as lease:
        assert await lease.connection.scalar(text("SELECT @@session.time_zone")) == "+00:00"
        await repo.save([dict(question_id=str(i), labels=labels, classified_at="2026-10-30T00:00:00")
            for i, labels in [(1, ["物流"]), (2, ["物流", "运费"]), (3, ["物流"])]], lease=lease)
        await repo.save([dict(question_id="1", labels=["发票"], classified_at="2026-10-31T00:00:00")], lease=lease)
    rows = await repo.pending(200)
    assert rows[-1]["question_id"] == "9007199254740993"
    result = await repo.distribution()
    assert result["total_questions"] == 4 and result["classified_questions"] == 2
    assert result["pool_rows"] == 5 and result["history_uncertain_pool_rows"] == 2
    counts = {r["label"]: r["count"] for r in result["labels"]}
    assert counts["物流"] == 2 and counts["发票"] == counts["运费"] == 1
    assert (await repo.distribution("all"))["total_questions"] == 6
    assert (await repo.classified_rows())[0]["classified_at"] == "2026-10-31T00:00:00"


async def test_mysql_real_shared_lock_and_disconnect_rollback(mysql_env, monkeypatch):
    """真实跨连接排它与写后KILL原租约回滚，不能换连接提交。参数：专库/替换器；返回：None；异常：断言失败。"""
    _, engine, _, repo, factory = mysql_env
    await apply(engine)
    await add_pool(factory)
    original = MySqlWriteLease.assert_held
    async with repo.execution() as lease:
        with pytest.raises(ConcurrentKnowledgeWrite):
            async with repo.execution():
                pytest.fail("第二连接不能取得相同锁")
        owner = await lease.connection.scalar(text("SELECT CONNECTION_ID()"))
        calls, killed = 0, []
        async def disconnect(self):
            """仅KILL已登记且再次核对专库与锁owner的本轮连接。参数：lease；返回：None；异常：断线。"""
            nonlocal calls
            calls += 1
            if calls == 2:
                async with factory() as control:
                    assert await control.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
                    assert await control.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": TOPIC_EXECUTION_LOCK}) == owner
                    assert await control.scalar(text("SELECT DB FROM information_schema.PROCESSLIST WHERE ID=:id"), {"id": owner}) == TEST_DATABASE
                    await control.execute(text(f"KILL CONNECTION {int(owner)}"))
                    killed.append(owner)
            await original(self)
        monkeypatch.setattr(MySqlWriteLease, "assert_held", disconnect)
        with pytest.raises(Exception):
            await repo.save([dict(question_id="1", labels=["物流"])], lease=lease)
    monkeypatch.setattr(MySqlWriteLease, "assert_held", original)
    assert killed == [owner]
    assert await repo.classified_rows() == []
    async with repo.execution() as fresh:
        assert await fresh.connection.scalar(text("SELECT CONNECTION_ID()")) != owner
        await repo.save([dict(question_id="1", labels=["物流"])], lease=fresh)
    assert len(await repo.classified_rows()) == 1


async def test_mysql_uncertain_201st_reclassification_and_report_resume(mysql_env):
    """真实SQL分页不能让200个未确定挡住第201条，重跑覆盖全部。参数：专库；返回：None；异常：断言失败。"""
    config, engine, _, repo, factory = mysql_env
    await apply(engine)
    await add_pool(factory, 201)
    def handler(request):
        """参数：请求；返回：仅第201条确定；异常：无。"""
        import json
        texts = json.loads(request.content).get("texts", []) if request.content else []
        return response(request, uncertain=bool(texts and texts[0] != "物流问题201"))
    service = TopicBatchService(repo, config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    report = await service.process()
    assert report["counts"] == dict(checked=201, saved=1, uncertain=200)
    assert [r["question_id"] for r in await repo.classified_rows()] == ["201"]
    await service.close()
    rerun = TopicBatchService(repo, config, httpx.AsyncClient(transport=httpx.MockTransport(response)))
    result = await rerun.process(reclassify=True)
    assert result["status"] == "complete" and result["counts"]["saved"] == 201
    assert len(await repo.classified_rows()) == 201
    await rerun.close()


async def test_mysql_close_during_http_has_no_row_transaction_or_commit(mysql_env):
    """网络等待不持池行锁；停机取消释放真实命名锁且不落半批。参数：专库；返回：None；异常：断言失败。"""
    config, engine, _, repo, factory = mysql_env
    await apply(engine)
    await add_pool(factory, 2)
    entered = asyncio.Event()
    async def handler(request):
        """参数：请求；返回：响应；异常：分类时等待取消。"""
        if request.url.path == "/classify":
            entered.set()
            await asyncio.Future()
        return response(request)
    service = TopicBatchService(repo, config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    task = asyncio.create_task(service.process())
    await asyncio.wait_for(entered.wait(), 5)
    try:
        async with factory.begin() as session:
            # 网络等待期间可立即取池行锁，旁路没有跨网络持行事务。
            async with asyncio.timeout(3):
                assert await session.scalar(select(LowConfidenceQuestion.id).where(
                    LowConfidenceQuestion.id == 1).with_for_update()) == 1
        await service.close()
        assert task.cancelled() and await repo.classified_rows() == []
        async with repo.execution() as lease:
            await lease.assert_held()
    finally:
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_mysql_save_rejects_lease_for_other_lock(mysql_env):
    """其他业务锁不能被冒充主题共同执行锁。参数：专库；返回：None；异常：断言失败。"""
    _, engine, lease_engine, repo, factory = mysql_env
    await apply(engine)
    await add_pool(factory)
    lease = await MySqlWriteLease.acquire(lease_engine, lock_name="mewhelp_ch10_test_wrong_lock")
    try:
        with pytest.raises(Exception):
            await repo.save([dict(question_id="1", labels=["物流"])], lease=lease)
        assert await repo.classified_rows() == []
    finally:
        await lease.release()


@pytest.mark.parametrize("zero_watermark", [False, True])
async def test_mysql_lost_lock_at_empty_scan_end_never_confirms_complete(mysql_env, monkeypatch, zero_watermark):
    """真实空页/零上界读取中释放自有锁，不得推进完成。参数：专库/替换器/零上界；返回：None；异常：断言失败。"""
    config, engine, _, repo, factory = mysql_env
    await apply(engine)
    if not zero_watermark:
        await add_pool(factory)
        async with repo.execution() as lease:
            await repo.save([dict(question_id="1", labels=["物流"])], lease=lease)
    current = []
    original_execution = repo.execution
    @asynccontextmanager
    async def captured():
        """参数：无；返回：真实执行租约；异常：连接或锁错误。"""
        async with original_execution() as lease:
            current.append(lease)
            yield lease
    monkeypatch.setattr(repo, "execution", captured)
    original_read = repo.high_watermark if zero_watermark else repo.pending
    async def release_after_read(*args, **kwargs):
        """只在门禁专库的原持锁连接释放本轮主题锁。参数：读分页参数；返回：真实读结果；异常：断言或连接错误。"""
        result = await original_read(*args, **kwargs)
        connection = current[0].connection
        assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
        owner = await connection.scalar(text("SELECT CONNECTION_ID()"))
        assert await connection.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": TOPIC_EXECUTION_LOCK}) == owner
        assert await connection.scalar(text("SELECT RELEASE_LOCK(:name)"), {"name": TOPIC_EXECUTION_LOCK}) == 1
        return result
    monkeypatch.setattr(repo, "high_watermark" if zero_watermark else "pending", release_after_read)
    service = TopicBatchService(repo, config, httpx.AsyncClient(transport=httpx.MockTransport(response)))
    try:
        report = await service.process()
        assert report["status"] == "failed" and report["cursor"] == "0"
        assert report["error"]["type"] == "LeaseLost"
        assert not await repo.execution_busy()
    finally:
        await service.close()
