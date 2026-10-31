"""仅在显式开启的空 mewhelp_ch09_test 验证真实 MySQL 迁移断点与 JSON。

不操作应用库。若专库存在任何表则拒绝运行；只清理本轮成功登记创建的白名单表。
运行：MEWHELP_RUN_CH09_MYSQL=1 uv run pytest tests/flywheel/test_mysql.py -m mysql -q。
"""

import os

import pytest
from sqlalchemy import JSON, event, inspect, text

from app.config import Settings
from app.db.session import create_database
from app.models.customer_service import Conversation


pytestmark = [pytest.mark.mysql, pytest.mark.skipif(
    os.getenv("MEWHELP_RUN_CH09_MYSQL") != "1", reason="需要显式 MEWHELP_RUN_CH09_MYSQL=1")]
TEST_DATABASE = "mewhelp_ch09_test"
OWNED_TABLES = ("qa_extraction_staging", "qa_extraction_batches", "knowledge_chunks", "messages", "low_confidence_questions", "faith_cases", "eval_runs", "review_queue", "conversations")


def _save_eval_artifact_process(config, artifact):
    """子进程独立连接专库保存同一产物，不调用模型。参数：专库配置/目录；返回：保存结果；异常：数据库错误。"""
    import asyncio
    from pathlib import Path
    from app.repositories.evaluations import evaluation_repository_runtime

    async def save():
        """只连接显式专库。参数：无；返回：最少结果；异常：配置/数据库错误。"""
        assert config.mysql_database == TEST_DATABASE
        async with evaluation_repository_runtime(config) as repository:
            result = await repository.save_completed_run(Path(artifact), "手动")
            return {"id": result["id"], "saved": result["saved"]}

    return asyncio.run(save())


async def test_same_artifact_is_saved_once_across_processes(isolated_mysql, tmp_path):
    """两个独立进程保存相同artifact必须恰好一行，新连接可见且重复内容冲突拒绝。参数：空专库/目录；返回：None；异常：断言失败。"""
    import asyncio
    import json
    from concurrent.futures import ProcessPoolExecutor
    from sqlalchemy import select
    from app.models.flywheel import EvalRun
    from app.repositories.evaluations import evaluation_repository_runtime
    from app.evaluation.runner import atomic_json
    from app.evaluation.policy import digest_json
    from tests.flywheel.test_eval_runs import completed_artifact
    config, engine, factory, _ = isolated_mysql
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: EvalRun.__table__.create(sync))
    artifact = completed_artifact(tmp_path / "run")
    with ProcessPoolExecutor(max_workers=2) as pool:
        loop = asyncio.get_running_loop()
        outcomes = await asyncio.gather(*(loop.run_in_executor(pool, _save_eval_artifact_process, config, str(artifact)) for _ in range(2)))
    assert outcomes[0]["id"] == outcomes[1]["id"] and sorted(row["saved"] for row in outcomes) == [False, True]
    async with factory() as session:
        rows = (await session.scalars(select(EvalRun))).all()
        assert len(rows) == 1 and rows[0].dataset_size == 48
        assert rows[0].metrics["faithfulness"] is None and rows[0].metrics["aggregate"]["recall@10"]["denominator"] == 40
    # 仅更改诊断耗时仍是一份自洽完整产物，但相同artifact_id不能覆盖旧内容。
    result_rows = [json.loads(line) for line in (artifact / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    result_rows[0]["elapsed_seconds"] = 2.0
    (artifact / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in result_rows), encoding="utf-8")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["results_digest"] = digest_json(result_rows)
    atomic_json(artifact / "manifest.json", manifest)
    async with evaluation_repository_runtime(config) as repository:
        with pytest.raises(ValueError, match="不同内容"):
            await repository.save_completed_run(artifact, "手动")
        assert len((await repository.list_runs(0, 1))["items"]) == 1
    print("两个真实进程相同artifact仅一行；新连接可见；48/40原分母和null保持；同ID不同内容拒绝")


async def test_cli_and_manager_share_database_execution_lock(isolated_mysql, tmp_path, monkeypatch):
    """独立CLI与manager子进程均在资源/付费前因同一真实锁退出busy，释放后可重取。参数：空专库/目录/环境；返回：None；异常：断言失败。"""
    import asyncio
    import subprocess
    import sys
    from pathlib import Path
    from app.evaluation.jobs import EvaluationJobManager
    from app.repositories.evaluations import evaluation_repository_runtime
    config, _, _, _ = isolated_mysql
    values = {"MYSQL_DATABASE": TEST_DATABASE, "MYSQL_HOST": config.mysql_host, "MYSQL_PORT": str(config.mysql_port),
        "MYSQL_USER": config.mysql_user, "MYSQL_PASSWORD": config.mysql_password.get_secret_value(),
        "PYTHONIOENCODING": "utf-8", "KNOWLEDGE_MODEL_PATH": str(tmp_path / "no-model")}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    root = Path(__file__).resolve().parents[2]
    manager = EvaluationJobManager(root, tmp_path / "reports", timeout_seconds=20)
    try:
        async with evaluation_repository_runtime(config) as repository:
            async with repository.execution():
                assert await repository.execution_busy() is True
                manager.start(root / "data/evaluation/ch04.json", [], profile="production",
                    policy_path=root / "data/knowledge/evidence-policy.json")
                process = await asyncio.create_subprocess_exec(sys.executable, "-m", "app.evaluation.cli", "evaluate-production",
                    "--dataset", "data/evaluation/ch04.json", "--policy", "data/knowledge/evidence-policy.json",
                    "--target", "application-readonly", "--output-dir", str(tmp_path / "cli-output"), cwd=root,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                assert process.returncode == 3, (stdout.decode(), stderr.decode())
                async with asyncio.timeout(20):
                    while manager.status()["status"] == "running":
                        await asyncio.sleep(0.02)
                assert manager.status()["status"] == "busy" and manager.status()["admission"] == "busy"
                assert not (tmp_path / "cli-output").exists()
                assert not list((tmp_path / "reports").glob("*-production-*"))
            assert await repository.execution_busy() is False
            async with repository.execution() as lease:
                await lease.assert_held()
        print("独立CLI与真实manager子进程共用MySQL执行锁，均退出3/busy且没有产物/模型资源；释放后重新获取成功")
    finally:
        await manager.close()


async def test_eval_save_lost_owned_connection_rolls_back(isolated_mysql, tmp_path, monkeypatch):
    """INSERT后断开本轮自有锁连接不得提交，新执行者可重取保存锁完成一次。参数：专库/目录/替身；返回：None；异常：断言失败。"""
    from sqlalchemy import select
    from app.models.flywheel import EvalRun
    from app.repositories.evaluations import evaluation_repository_runtime, MySqlWriteLease, EVALUATION_SAVE_LOCK
    from tests.flywheel.test_eval_runs import completed_artifact
    config, engine, factory, _ = isolated_mysql
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: EvalRun.__table__.create(sync))
    artifact = completed_artifact(tmp_path / "run")
    original = MySqlWriteLease.assert_held
    checks = 0

    async def disconnect_owned(lease):
        """在当前自有保存连接第二次核锁前真实断开它。参数：本轮lease；返回：None；异常：连接故障。"""
        nonlocal checks
        checks += 1
        if checks == 2:
            identifier = await lease.connection.scalar(text("SELECT CONNECTION_ID()"))
            assert await lease.connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
            async with factory() as session:
                assert await session.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": EVALUATION_SAVE_LOCK}) == identifier
                assert await session.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
                await session.execute(text(f"KILL CONNECTION {int(identifier)}"))
        await original(lease)

    async with evaluation_repository_runtime(config) as repository:
        monkeypatch.setattr(MySqlWriteLease, "assert_held", disconnect_owned)
        with pytest.raises(Exception):
            await repository.save_completed_run(artifact, "手动")
        monkeypatch.setattr(MySqlWriteLease, "assert_held", original)
        async with factory() as session:
            assert (await session.scalars(select(EvalRun))).all() == []
        assert (await repository.save_completed_run(artifact, "手动"))["saved"] is True
    print("真实断开本轮专庫自有保存连接：未提交零行；释放后新连接成功补存一行")


@pytest.fixture
async def isolated_mysql():
    """连接已准备专库并保护已存在表，只删除本轮创建表。

    参数：无，环境只提供凭据路径，库名不可覆盖。
    返回：配置/引擎/会话工厂/成功建表登记集合。
    异常：连接错误、非空库或实际目标不符时立即拒绝；测试错误清理后传播。
    """
    config = Settings(_env_file=os.getenv("MEWHELP_TEST_ENV_FILE", ".env"))
    config = config.model_copy(update={"mysql_database": TEST_DATABASE})
    assert config.mysql_database == TEST_DATABASE
    engine, factory = create_database(config, database=TEST_DATABASE)
    owned = set()

    def remember_table(connection, cursor, statement, parameters, context, executemany):
        """仅成功 CREATE 后记录所属表。参数：SQLAlchemy 执行事件参数；返回：None；异常：无。"""
        normalized = statement.strip().replace("`", "")
        for name in OWNED_TABLES:
            if normalized.startswith(f"CREATE TABLE {name} ") or normalized.startswith(f"CREATE TABLE {name} ("):
                owned.add(name)

    try:
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
            assert await connection.run_sync(lambda sync: inspect(sync).get_table_names()) == [], "拒绝覆盖任何预先存在的测试库表"
        event.listen(engine.sync_engine, "after_cursor_execute", remember_table)
        yield config, engine, factory, owned
    finally:
        if owned:
            async with engine.begin() as connection:
                assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
                for name in OWNED_TABLES:
                    if name in owned:
                        await connection.execute(text(f"DROP TABLE `{name}`"))
        await engine.dispose()


async def prepare_ch04(engine):
    """按旧章顺序创建最少依赖。参数：专库引擎；返回：None；异常：DDL/结构错误。"""
    from app.db.migrate_rag_quality import _migrate

    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: Conversation.__table__.create(sync))
        await connection.run_sync(_migrate)


async def prepare_feedback(engine):
    """在具名空专库准备原轮反馈依赖。参数：引擎；返回：None；异常：DDL错误。"""
    from app.db.migrate_ch09 import migrate_connection
    from app.models.customer_service import Message
    await prepare_ch04(engine)
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
        await connection.run_sync(lambda sync: Message.__table__.create(sync))


async def test_mysql_score_roundtrip_append_restore_and_finish_retry(isolated_mysql):
    """真实MySQL末位变化不能阻断累计保存、恢复和完成回放。参数：空专库；返回：None；异常：断言失败。"""
    from sqlalchemy import select
    from app.models.customer_service import Message
    from app.repositories.chat import ChatRepository, PersistenceUncertain
    from app.schemas.flywheel import RetrievalAttempt, RetrievalSnapshot
    from app.tools.knowledge import restore_turn_knowledge
    from app.workflow.state import new_turn_state
    from tests.flywheel.test_pool import start
    from tests.flywheel.test_snapshots import scored_snapshot
    _, engine, factory, _ = isolated_mysql
    await prepare_feedback(engine)
    chat = ChatRepository(factory)
    ctx = await start(chat)
    original = scored_snapshot()
    await chat.save_retrieval_snapshot(ctx, original)
    first = (await chat.get_workflow_turn("s", "u", "first"))["retrieval_snapshot"]
    assert first["attempts"][0]["chunks"][0]["rerank_score"] == 0.99353688955307
    original.attempts.append(RetrievalAttempt(attempt_id="second", query="退款时限", status="empty", top_k=5, candidate_count=0))
    await chat.save_retrieval_snapshot(ctx, original)
    stored = (await chat.get_workflow_turn("s", "u", "first"))["retrieval_snapshot"]
    assert [a["attempt_id"] for a in stored["attempts"]] == ["first", "second"]
    assert stored["attempts"][0] == first["attempts"][0]
    restored = await restore_turn_knowledge(new_turn_state(ctx, [], 0), chat)
    assert restored["retrieval_snapshot"] == stored
    completed = await chat.finish_turn(ctx, "政策原答", retrieval_snapshot=original)
    assert await chat.finish_turn(ctx, "政策原答", retrieval_snapshot=original) == completed
    await chat.save_retrieval_snapshot(ctx, original)
    assert (await chat.get_workflow_turn("s", "u", "first"))["retrieval_snapshot"] == stored
    # 相对数据库原件0.99353688955307正好差两步；0699/0698实际都只差一步。
    for field, value in (("rerank_score", 0.9935368895530697), ("answer", "不同原文"),
                         ("chunk_id", 99), ("rank", 2)):
        changed = original.model_copy(deep=True)
        setattr(changed.attempts[0].chunks[0], field, value)
        with pytest.raises(ValueError, match="历史快照"):
            await chat.save_retrieval_snapshot(ctx, changed)
        with pytest.raises(PersistenceUncertain):
            await chat.finish_turn(ctx, "政策原答", retrieval_snapshot=changed)
    with pytest.raises(PersistenceUncertain):
        await chat.finish_turn(ctx, "政策原答", retrieval_snapshot=RetrievalSnapshot(status="ok", attempts=original.attempts[:1]))
    async with factory() as session:
        assert len((await session.scalars(select(Message).where(Message.role == "assistant"))).all()) == 1
    async with engine.connect() as connection:
        version = await connection.scalar(text("SELECT VERSION()"))
        assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
    print(f"MySQL {version}：真实分数首存→追加→恢复→完成重试通过；旧原件不变；真实分数/正文/身份/排名变化拒绝")


async def test_same_source_concurrent_retry_keeps_one_pool_row(isolated_mysql):
    """真实MySQL行锁必须串行化同一原轮的并发反馈。参数：空专库；返回：None；异常：断言失败。"""
    import asyncio
    from sqlalchemy import select
    from app.repositories.chat import ChatRepository
    from app.models.rag_quality import LowConfidenceQuestion
    from tests.flywheel.test_pool import start
    _, engine, factory, _ = isolated_mysql
    await prepare_feedback(engine)
    chat = ChatRepository(factory)
    ctx = await start(chat)
    await chat.finish_turn(ctx, "服务端原答")
    receipts = await asyncio.gather(*(chat.record_unresolved_feedback("s", "u", "first") for _ in range(8)))
    assert all(receipt == receipts[0] for receipt in receipts)
    async with factory() as session:
        rows = (await session.scalars(select(LowConfidenceQuestion))).all()
        assert len(rows) == 1 and receipts[0].pool_id == str(rows[0].id)
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT retrieved_chunks IS NULL FROM low_confidence_questions")) == 1
    print("8个独立session并发反馈：池中1行，回执完全一致，确认无检索保存SQL NULL")


async def test_mysql_pool_failure_rolls_back_and_lost_response_is_idempotent(isolated_mysql):
    """真实提交前后故障分别要求完整回滚及重发幂等。参数：空专库；返回：None；异常：断言失败。"""
    from contextlib import asynccontextmanager
    from sqlalchemy import select
    from app.models.customer_service import Message
    from app.models.rag_quality import LowConfidenceQuestion
    from app.repositories.chat import ChatRepository
    from app.schemas.flywheel import PoolSignal
    from app.workflow.routing import fixed_reply
    from tests.flywheel.test_pool import start
    _, engine, factory, _ = isolated_mysql
    await prepare_feedback(engine)
    chat = ChatRepository(factory)
    ctx = await start(chat)
    def fail(connection, cursor, statement, parameters, context, executemany):
        """在真实入池SQL前模拟失败。参数：数据库事件；返回：None；异常：RuntimeError。"""
        if statement.startswith("INSERT INTO low_confidence_questions"):
            raise RuntimeError("pool failed")
    event.listen(engine.sync_engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="pool failed"):
            await chat.finish_turn(ctx, fixed_reply("fallback")["answer"],
                pool_signals=[PoolSignal(source="retrieval_low_conf", reason="证据不足")])
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail)
    async with factory() as session:
        rows = (await session.scalars(select(Message))).all()
        assert len(rows) == 1 and rows[0].turn_status == "running"
        assert (await session.scalars(select(LowConfidenceQuestion))).all() == []
    await chat.finish_turn(ctx, "完成答复")
    class LostResponse:
        """在数据库成功提交后模拟HTTP返回中断。"""
        @asynccontextmanager
        async def begin(self):
            """参数：无；返回：事务；异常：ConnectionError。"""
            async with factory.begin() as session:
                yield session
            raise ConnectionError("lost response")
    with pytest.raises(ConnectionError):
        await ChatRepository(LostResponse()).record_unresolved_feedback("s", "u", "first")
    receipt = await chat.record_unresolved_feedback("s", "u", "first")
    async with factory() as session:
        rows = (await session.scalars(select(LowConfidenceQuestion))).all()
        assert len(rows) == 1 and receipt.pool_id == str(rows[0].id)
    print("真实MySQL：入池异常完整回滚；提交响应丢失重发仍为1条同ID回执")


async def test_multiple_sources_share_counted_review_and_preserve_source_metadata(isolated_mysql):
    """同轮新入口不可重复计数，下一轮命中同问题只加一次。参数：专库；返回：None；异常：断言失败。"""
    import asyncio
    from sqlalchemy import select
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL, SelfCheckFailure
    from app.models.customer_service import Message
    from app.models.flywheel import ReviewQueue
    from app.models.rag_quality import LowConfidenceQuestion
    from app.repositories.chat import ChatRepository
    from tests.flywheel.test_pool import start
    _, engine, factory, _ = isolated_mysql
    await prepare_feedback(engine)
    chat = ChatRepository(factory)
    first = await start(chat)
    await chat.finish_turn(first, FIXED_KNOWLEDGE_REFUSAL, low_confidence=SelfCheckFailure(reason="原轮自评不足"))
    async with factory.begin() as session:
        review = ReviewQueue(normalized_question="原问题", occurrence_count=1)
        session.add(review)
        await session.flush()
        source = await session.scalar(select(Message).where(Message.role == "user", Message.turn_id == "first"))
        pool = await session.scalar(select(LowConfidenceQuestion))
        pool.matched_review_id = review.id
        source.rag_metadata = {**source.rag_metadata, "trace_identity": {"trace_id": "保留"},
            "ticket_intent": {"status": "confirmed"}, "knowledge_pool": {**source.rag_metadata["knowledge_pool"],
                "matched_review_id": review.id, "counted_review_id": review.id}}
    receipts = await asyncio.gather(*(chat.record_unresolved_feedback("s", "u", "first") for _ in range(5)))
    async with factory() as session:
        assert (await session.get(ReviewQueue, review.id)).occurrence_count == 1
        source = await session.scalar(select(Message).where(Message.role == "user", Message.turn_id == "first"))
        assert set(source.rag_metadata["knowledge_pool"]["pool_ids"]) == {"self_check", "user_feedback"}
        assert source.rag_metadata["trace_identity"] == {"trace_id": "保留"}
        assert source.rag_metadata["ticket_intent"] == {"status": "confirmed"}
        assert all(r == receipts[0] and r.matched_review_id == str(review.id) for r in receipts)
    second = await start(chat, turn="second", question="同义的第二轮问题")
    await chat.finish_turn(second, "原答")
    async with factory.begin() as session:
        source = await session.scalar(select(Message).where(Message.role == "user", Message.turn_id == "second"))
        source.rag_metadata = {**source.rag_metadata, "knowledge_pool": {"matched_review_id": review.id}}
    await asyncio.gather(*(chat.record_unresolved_feedback("s", "u", "second") for _ in range(5)))
    async with factory() as session:
        assert (await session.get(ReviewQueue, review.id)).occurrence_count == 2
        assert len((await session.scalars(select(LowConfidenceQuestion))).all()) == 3
    print("同轮self_check+feedback共2条池记录但只计1次；新轮同义命中并发重试只加1次；原工单/trace字段保留")


async def test_empty_database_ch04_then_ch09_succeeds(isolated_mysql):
    """首次 CH04 隐含依赖新表、预览写入或新启动接受旧表时失败。参数：专库夹具；返回：None；异常：断言失败。"""
    from app.db.migrate_ch09 import check_ch09_tables, migrate_connection, validate_ch09_schema
    from app.db.migrate_rag_quality import QualitySchemaError, validate_quality_schema

    config, engine, _, _ = isolated_mysql
    await prepare_ch04(engine)
    with pytest.raises(QualitySchemaError):
        await check_ch09_tables(config)
    async with engine.begin() as connection:
        before = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        old_columns = await connection.run_sync(lambda sync: [c["name"] for c in inspect(sync).get_columns("low_confidence_questions")])
        preview = await connection.run_sync(migrate_connection)
        after = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        assert preview and after == before
        changes = await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
        assert len(changes) == 6
        upgraded = await connection.run_sync(lambda sync: [c["name"] for c in inspect(sync).get_columns("low_confidence_questions")])
        assert set(upgraded) - set(old_columns) == {"retrieved_chunks", "matched_review_id"}
        await connection.run_sync(lambda sync: validate_quality_schema(inspect(sync)))
        await connection.run_sync(lambda sync: validate_ch09_schema(inspect(sync)))
    await check_ch09_tables(config)


@pytest.mark.parametrize("breakpoint", range(1, 7))
async def test_ch09_partial_migration_resumes_without_data_loss(isolated_mysql, breakpoint):
    """任意真实 DDL 已提交后中断不能丢旧行或重复修改。参数：专库/断点；返回：None；异常：断言失败。"""
    from app.db.migrate_ch09 import migrate_connection, validate_ch09_schema

    _, engine, _, _ = isolated_mysql
    await prepare_ch04(engine)
    async with engine.begin() as connection:
        await connection.execute(text("INSERT INTO low_confidence_questions (raw_question, source, reason) VALUES ('旧问题中文', 'self_check', '原始理由')"))
        await connection.execute(text("INSERT INTO faith_cases (eval_id,bucket,query,answer,reason) VALUES ('CH09-legacy','A_policy','旧问','旧答','旧理由')"))
        original = (await connection.execute(text("SELECT id,raw_question,source,reason,created_at FROM low_confidence_questions"))).all()
        original_faith = (await connection.execute(text("SELECT * FROM faith_cases"))).all()
    applied = 0

    def interrupt_after_ddl(connection, cursor, statement, parameters, context, executemany):
        """在服务器成功提交指定步后模拟进程中断。参数：执行事件；返回：None；异常：RuntimeError。"""
        nonlocal applied
        if statement.startswith(("CREATE TABLE", "ALTER TABLE")):
            applied += 1
            if applied == breakpoint:
                raise RuntimeError("simulated CH09 interruption")

    event.listen(engine.sync_engine, "after_cursor_execute", interrupt_after_ddl)
    try:
        with pytest.raises(RuntimeError, match="simulated CH09 interruption"):
            async with engine.begin() as connection:
                await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
    finally:
        event.remove(engine.sync_engine, "after_cursor_execute", interrupt_after_ddl)
    async with engine.begin() as connection:
        remaining = await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
        assert len(remaining) == 6 - breakpoint
        repeated = await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
        rows = (await connection.execute(text("SELECT id,raw_question,source,reason,created_at FROM low_confidence_questions"))).all()
        assert rows == original and repeated == []
        assert (await connection.execute(text("SELECT * FROM faith_cases"))).all() == original_faith
        await connection.run_sync(lambda sync: validate_ch09_schema(inspect(sync)))
        enums = await connection.run_sync(lambda sync: {
            "review_status": next(c["type"].enums for c in inspect(sync).get_columns("review_queue") if c["name"] == "review_status"),
            "triggered_by": next(c["type"].enums for c in inspect(sync).get_columns("eval_runs") if c["name"] == "triggered_by")})
        assert enums == {"review_status": ["待审", "通过", "驳回"], "triggered_by": ["定时", "手动"]}
    print(f"断点 {breakpoint}/6：真实 DDL 中断后续迁 {len(remaining)} 步，旧行/中文 ENUM 保留，重复迁移为空")


async def test_mysql_sql_null_json_null_and_whole_snapshot_replacement(isolated_mysql):
    """Python None 被存成 JSON null、整体快照赋值未持久化或删除外键行为错误时失败。参数：专库；返回：None；异常：断言失败。"""
    from app.db.migrate_ch09 import migrate_connection
    from app.models.flywheel import ReviewQueue
    from app.models.rag_quality import LowConfidenceQuestion

    _, engine, factory, _ = isolated_mysql
    await prepare_ch04(engine)
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
    async with factory.begin() as session:
        review = ReviewQueue(normalized_question="标准问")
        session.add(review)
        await session.flush()
        absent = LowConfidenceQuestion(raw_question="未检索", source="self_check", retrieved_chunks=None, matched_review_id=review.id)
        json_null = LowConfidenceQuestion(raw_question="JSON空", source="user_feedback", retrieved_chunks=JSON.NULL)
        session.add_all([absent, json_null])
        await session.flush()
    async with engine.connect() as connection:
        rows = (await connection.execute(text("SELECT raw_question, retrieved_chunks IS NULL, JSON_TYPE(retrieved_chunks) FROM low_confidence_questions ORDER BY id"))).all()
        assert rows == [("未检索", 1, None), ("JSON空", 0, "NULL")]
    async with factory.begin() as session:
        saved = await session.get(LowConfidenceQuestion, absent.id)
        saved.retrieved_chunks = {"schema_version": 1, "status": "empty", "attempts": []}
        review = await session.get(ReviewQueue, review.id)
        await session.delete(review)
    async with factory() as session:
        saved = await session.get(LowConfidenceQuestion, absent.id)
        assert saved.retrieved_chunks["status"] == "empty" and saved.matched_review_id is None


@pytest.mark.parametrize("changes", [
    ("ALTER TABLE review_queue MODIFY review_status ENUM('待审','通过','乱码') NOT NULL DEFAULT '待审'",),
    ("ALTER TABLE review_queue DROP INDEX idx_review_status, ADD INDEX idx_review_status (created_at)",),
    ("ALTER TABLE low_confidence_questions DROP FOREIGN KEY fk_lcq_review",
     "ALTER TABLE low_confidence_questions ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue(id) ON DELETE CASCADE"),
    ("ALTER TABLE review_queue MODIFY updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",),
], ids=["enum", "index", "foreign_key", "on_update"])
async def test_ch09_rejects_changed_enum_index_or_foreign_key(isolated_mysql, changes):
    """真实反射必须拒绝枚举、索引、外键或自动更新时间漂移。参数：专库/变更 SQL；返回：None；异常：断言失败。"""
    from app.db.migrate_ch09 import migrate_connection
    from app.db.migrate_rag_quality import QualitySchemaError

    _, engine, _, _ = isolated_mysql
    await prepare_ch04(engine)
    async with engine.begin() as connection:
        await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
        for statement in changes:
            await connection.execute(text(statement))
        with pytest.raises(QualitySchemaError):
            await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))


@pytest.mark.parametrize("default", ["hybrid_()rerank", "HYBRID_RERANK", " hybrid_rerank "])
async def test_changed_varchar_default_refuses_upgrade_without_writes(isolated_mysql, default):
    """真实 VARCHAR 默认漂移必须在任何新章写入前被拒绝。参数：专库/错误默认值；返回：None；异常：断言失败。"""
    from app.db.migrate_ch09 import migrate_connection
    from app.db.migrate_rag_quality import QualitySchemaError

    _, engine, _, _ = isolated_mysql
    await prepare_ch04(engine)
    async with engine.begin() as connection:
        await connection.execute(text("ALTER TABLE faith_cases ALTER COLUMN strategy SET DEFAULT :value"), {"value": default})
        await connection.execute(text("INSERT INTO faith_cases (eval_id,bucket,query,answer,reason) VALUES ('default-drift','A_policy','测试问','测试答','测试理由')"))
        reflected = await connection.run_sync(lambda sync: next(
            column["default"] for column in inspect(sync).get_columns("faith_cases") if column["name"] == "strategy"))
        assert reflected == "'" + default + "'"
        before_tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        before_rows = (await connection.execute(text("SELECT * FROM faith_cases"))).all()
        assert (await connection.execute(text("SELECT strategy FROM faith_cases"))).scalar_one() == default
        writes = []

        def remember_write(connection, cursor, statement, parameters, context, executemany):
            """记录迁移阶段实际发往驱动的写操作。参数：游标执行事件；返回：None；异常：无。"""
            if statement.lstrip().upper().startswith(("CREATE ", "ALTER ", "DROP ", "INSERT ", "UPDATE ", "DELETE ", "SET ")):
                writes.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", remember_write)
        try:
            with pytest.raises(QualitySchemaError, match="strategy.*默认值"):
                await connection.run_sync(lambda sync: migrate_connection(sync, apply=True))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", remember_write)
        assert writes == []
        assert await connection.run_sync(lambda sync: inspect(sync).get_table_names()) == before_tables
        assert (await connection.execute(text("SELECT * FROM faith_cases"))).all() == before_rows
    print(f"VARCHAR default {reflected!r}：真实反射及默认写入保留原值；预检拒绝，迁移写操作 0")


@pytest.fixture
async def merge_mysql(isolated_mysql):
    """参数：显式空专库；返回：真实chat/归并仓储/工厂/锁引擎；异常：专库初始化失败。"""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool
    from app.repositories.chat import ChatRepository
    from app.repositories.flywheel import FlywheelRepository
    _, engine, factory, _ = isolated_mysql
    await prepare_feedback(engine)
    lease_engine = create_async_engine(engine.url, poolclass=NullPool, connect_args={"connect_timeout": 5})
    try:
        yield ChatRepository(factory), FlywheelRepository(factory, lease_engine), factory, lease_engine
    finally:
        await lease_engine.dispose()


async def test_two_workers_empty_queue_create_one_semantic_gap(merge_mysql):
    """两真实命名锁连接不能为同义两轮各建一条。参数：专库；返回：None；异常：断言失败。"""
    import asyncio
    from app.services.flywheel import FlywheelService
    from tests.flywheel.test_merge import Model, seed, rows
    chat, repo, factory, _ = merge_mysql
    await seed(chat, sources=("retrieval_low_conf", "self_check", "user_feedback"))
    await seed(chat, turn="second")
    workers = [FlywheelService(repo, Model()) for _ in range(2)]
    try:
        results = await asyncio.gather(*(worker.process_pending() for worker in workers))
        pool, queue, users = await rows(factory)
        assert sum(row["processed"] for row in results) == 4
        assert sum(row["failed"] for row in results) == 0
        assert len(pool) == 4 and len(queue) == 1 and queue[0].occurrence_count == 2
        assert {row.matched_review_id for row in pool} == {queue[0].id}
        assert all(user.rag_metadata["knowledge_pool"]["counted_review_id"] == queue[0].id for user in users)
        print("真实MySQL双worker：三来源第一轮+第二轮=4原件，1待审缺口，count=2；无重复写入")
    finally:
        await asyncio.gather(*(worker.close() for worker in workers))


async def test_mysql_feedback_worker_then_repeat_receipt_and_rollback(merge_mysql):
    """真实生产worker更新反馈回执，提交失败不得部分增加计数。参数：专库；返回：None；异常：断言失败。"""
    from app.services.flywheel import FlywheelService
    from tests.flywheel.test_merge import Model, rows, seed
    from tests.flywheel.test_pool import start
    chat, repo, factory, lease_engine = merge_mysql
    await chat.finish_turn(await start(chat), "原答")
    before = await chat.record_unresolved_feedback("s", "u", "first")
    worker = FlywheelService(repo, Model())
    try:
        assert (await worker.process_pending())["processed"] == 1
        after = await chat.record_unresolved_feedback("s", "u", "first")
        pool, queue, _ = await rows(factory)
        assert before.pool_id == after.pool_id and before.matched_review_id is None
        assert after.matched_review_id == str(queue[0].id) and queue[0].occurrence_count == 1 and len(pool) == 1
        await seed(chat, turn="second")
        def reject(connection, cursor, statement, parameters, context, executemany):
            """参数：持锁连接SQL事件；返回：None；异常：注入提交前写入故障。"""
            if statement.startswith("UPDATE low_confidence_questions"):
                raise RuntimeError("atomic test")
        event.listen(lease_engine.sync_engine, "before_cursor_execute", reject)
        try:
            assert (await worker.process_pending())["failed"] == 1
        finally:
            event.remove(lease_engine.sync_engine, "before_cursor_execute", reject)
        pool, queue, users = await rows(factory)
        assert queue[0].occurrence_count == 1 and pool[1].matched_review_id is None
        assert users[1].rag_metadata["knowledge_pool"]["counted_review_id"] is None
        print("真实MySQL反馈→生产worker→重复反馈：pool_id稳定、归并ID更新；故障后count/池/JSON完整回滚")
    finally:
        await worker.close()


async def test_mysql_candidate_frozen_during_model_restarts_without_row_transaction(merge_mysql):
    """真实审核连接在模型中途可提交；worker须重读候选并保持核准答案。参数：专库；返回：None；异常：断言失败。"""
    from sqlalchemy import select
    from app.models.flywheel import ReviewQueue
    from app.services.flywheel import FlywheelService
    from tests.flywheel.test_merge import Model, seed, rows
    from app.repositories.flywheel import FlywheelRepository
    chat, repo, factory, lease_engine = merge_mysql
    connections = []
    class TrackedEngine:
        """保留本测试自有真实连接以检查事务状态。"""
        async def connect(self):
            """参数：无；返回：真实lease连接；异常：数据库错误。"""
            connection = await lease_engine.connect()
            connections.append(connection)
            return connection
    repo = FlywheelRepository(factory, TrackedEngine())
    await seed(chat)
    async with factory.begin() as session:
        session.add_all([ReviewQueue(normalized_question="候选一"), ReviewQueue(normalized_question="候选二")])
    model = Model([1, 2])
    match = model.match
    async def approve(question, candidates):
        """参数：当前页；返回：原模型判定；异常：审核阻塞超时或数据库错误。"""
        import asyncio
        decision = await match(question, candidates)
        assert connections and not connections[-1].in_transaction(), "模型期间仍有事务"
        if decision.matched_review_id == 1:
            async with asyncio.timeout(2):
                async with factory.begin() as session:
                    first = await session.scalar(select(ReviewQueue).where(ReviewQueue.id == 1).with_for_update())
                    first.approved_answer = "冻结人工答案"
        return decision
    model.match = approve
    worker = FlywheelService(repo, model, page_size=1)
    try:
        assert (await worker.process_pending())["processed"] == 1
        pool, queue, _ = await rows(factory)
        assert model.pages == [[1], [2]] and pool[0].matched_review_id == 2
        assert queue[0].approved_answer == "冻结人工答案" and [row.occurrence_count for row in queue] == [1, 2]
        print("模型中途真实审核行锁事务2秒内独立提交；自有lease.in_transaction=False；重新分页命中候选2")
    finally:
        await worker.close()


async def test_mysql_lost_connection_discards_old_model_decision(merge_mysql):
    """真实断开本轮自有lease后新worker成功，旧模型结果绝不可累加。参数：专库；返回：None；异常：断言失败。"""
    import asyncio
    from app.models.flywheel import ReviewQueue
    from app.services.flywheel import FlywheelService
    from tests.flywheel.test_merge import Model, seed, rows
    from app.repositories.flywheel import FlywheelRepository
    chat, repo, factory, lease_engine = merge_mysql
    owned_ids = []
    class TrackedEngine:
        """仅登记本次旧worker自己创建且已核对专库的连接身份。"""
        async def connect(self):
            """参数：无；返回：真实自有连接；异常：身份不符立即拒绝。"""
            connection = await lease_engine.connect()
            assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
            owned_ids.append(await connection.scalar(text("SELECT CONNECTION_ID()")))
            await connection.commit()
            return connection
    await seed(chat)
    async with factory.begin() as session:
        session.add(ReviewQueue(normalized_question="已存在等价缺口"))
    entered, resume = asyncio.Event(), asyncio.Event()
    model = Model()
    async def slow_match(question, candidates):
        """参数：当前候选；返回：失锁前模型结果；异常：取消。"""
        from app.schemas.flywheel import MatchDecision
        entered.set()
        await resume.wait()
        return MatchDecision(matched_review_id=candidates[0].id)
    model.match = slow_match
    old, fresh = FlywheelService(FlywheelRepository(factory, TrackedEngine()), model), FlywheelService(repo, Model())
    operation = asyncio.create_task(old.process_pending())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with factory() as session:
            identifier = await session.scalar(text("SELECT IS_USED_LOCK('mewhelp_ch09_flywheel_merge')"))
            assert isinstance(identifier, int) and identifier > 0
            assert owned_ids == [identifier]
            assert await session.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
            # 该锁名仅本测试worker持有；再次确认归属后只断开此轮创建的专用连接。
            assert await session.scalar(text("SELECT IS_USED_LOCK('mewhelp_ch09_flywheel_merge')")) == identifier
            await session.execute(text(f"KILL CONNECTION {identifier}"))
        assert (await fresh.process_pending())["processed"] == 1
        resume.set()
        assert (await asyncio.wait_for(operation, 10))["failed"] == 1
        pool, queue, _ = await rows(factory)
        assert queue[0].occurrence_count == 2 and pool[0].matched_review_id == queue[0].id
        print("真实KILL自有lease连接→新worker取得锁并提交→旧判定失败；唯一计数从1到2，未到3")
    finally:
        resume.set()
        await asyncio.gather(old.close(), fresh.close())
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.parametrize("phase", ["acquire", "model", "release_before", "release_after"])
async def test_mysql_worker_cancel_releases_owned_lock_and_connection(merge_mysql, phase, monkeypatch):
    """真实等待锁/模型/释放SQL中取消均必须物理关闭，随后连接可获锁。参数：专库/阶段；返回：None；异常：断言失败。"""
    import asyncio
    from app.knowledge.lease import MySqlWriteLease
    from sqlalchemy.ext.asyncio import AsyncConnection
    from app.repositories.flywheel import FlywheelRepository
    from app.services.flywheel import FlywheelService
    from tests.flywheel.test_merge import Model, seed, rows
    chat, repo, factory, lease_engine = merge_mysql
    await seed(chat)
    entered = asyncio.Event()
    identifiers, connections = [], []
    blocker = None

    class TrackedEngine:
        """记录本测试实际创建连接，释放阶段以真实可取消SQL延长等待。"""
        async def connect(self):
            """参数：无；返回：真实专用连接；异常：数据库错误。"""
            connection = await lease_engine.connect()
            identifiers.append(await connection.scalar(text("SELECT CONNECTION_ID()")))
            await connection.commit()
            connections.append(connection)
            if phase == "acquire":
                entered.set()
            return connection

    if phase == "acquire":
        blocker = await MySqlWriteLease.acquire(lease_engine, lock_name="mewhelp_ch09_flywheel_merge")
    model = Model()
    if phase == "model":
        async def wait_model(question):
            """参数：原问；返回：不返回；异常：取消。"""
            entered.set()
            await asyncio.Event().wait()
        model.normalize = wait_model
    scalar = AsyncConnection.scalar
    async def barrier(connection, statement, *args, **kwargs):
        """在真实RELEASE_LOCK发送前或响应后暂停，不替换SQL。参数：真实调用；返回：原结果；异常：取消。"""
        target = connection in connections and "RELEASE_LOCK" in str(statement)
        if target and phase == "release_before":
            entered.set()
            await asyncio.Event().wait()
        value = await scalar(connection, statement, *args, **kwargs)
        if target and phase == "release_after":
            entered.set()
            await asyncio.Event().wait()
        return value
    monkeypatch.setattr(AsyncConnection, "scalar", barrier)
    worker = FlywheelService(FlywheelRepository(factory, TrackedEngine()), model, close_seconds=0.01)
    operation = asyncio.create_task(worker.process_pending())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # 等待事件循环进入实际数据库/模型await，再走正式close而非仅取消fake。
        await asyncio.sleep(0.05)
        await asyncio.wait_for(worker.close(), 12)
        assert operation.done() and operation.cancelled()
        assert connections and all(connection.closed for connection in connections)
        if blocker is not None:
            await blocker.release()
            blocker = None
        async with factory() as session:
            assert await session.scalar(text("SELECT IS_USED_LOCK('mewhelp_ch09_flywheel_merge')")) is None
            for identifier in identifiers:
                assert await session.scalar(text("SELECT COUNT(*) FROM information_schema.PROCESSLIST WHERE ID=:id"), {"id": identifier}) == 0
        check = await MySqlWriteLease.acquire(lease_engine, lock_name="mewhelp_ch09_flywheel_merge", timeout_seconds=0)
        await check.release()
        pool, queue, _ = await rows(factory)
        assert (len(queue) == 1 and pool[0].matched_review_id is not None) if phase.startswith("release") else (not queue and pool[0].matched_review_id is None)
        print(f"真实MySQL {phase}取消：worker已收回、专用连接关闭且PROCESSLIST无该ID、命名锁可重新获取")
    finally:
        if blocker is not None:
            await blocker.release()
        await worker.close()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.fixture
async def publication_mysql(isolated_mysql):
    """在显式空专库组装真实审核与CH03存储，向量仍是可断点替身。参数：安全夹具；返回：发布依赖；异常：建表/连接错误。"""
    from types import SimpleNamespace
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool
    from app.models.knowledge import KnowledgeBase
    from app.models.flywheel import ReviewQueue
    from app.repositories.flywheel import FlywheelRepository
    from app.knowledge.store import KnowledgeStore
    from app.services.review_publish import ReviewPublisher
    from tests.knowledge.fakes import FakeEncoder, MemoryVectorIndex
    config, engine, factory, _ = isolated_mysql
    await prepare_feedback(engine)
    async with engine.begin() as connection:
        assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
        for table in KnowledgeBase.metadata.sorted_tables:
            await connection.run_sync(table.create)
            warnings = (await connection.execute(text("SHOW WARNINGS"))).all()
            if warnings:
                print(f"既有CH03建表警告 {table.name}: {warnings}")
    lease_engine = create_async_engine(engine.url, poolclass=NullPool, connect_args={"connect_timeout": 5})
    repo, store = FlywheelRepository(factory, lease_engine), KnowledgeStore(factory, lease_engine)
    encoder, index = FakeEncoder(), MemoryVectorIndex()
    publisher = ReviewPublisher(repo, store, encoder, index, close_seconds=0.01)
    async with factory.begin() as session:
        row = ReviewQueue(normalized_question="实际审核的问题", ai_suggested_answer="模型建议不能自动通过")
        session.add(row)
        await session.flush()
    try:
        yield SimpleNamespace(config=config, engine=engine, factory=factory, repo=repo, store=store,
            encoder=encoder, index=index, publisher=publisher, review_id=row.id, lease_engine=lease_engine)
    finally:
        try:
            await publisher.close()
        finally:
            await lease_engine.dispose()


@pytest.mark.parametrize("point", ["mysql", "vector", "review"])
async def test_mysql_review_publication_breakpoints_recover_same_identity(publication_mysql, point):
    """真实MySQL持久写后断点恢复不得重复块或丢失人工正文。参数：专库依赖/断点；返回：None；异常：断言失败。"""
    from sqlalchemy import select
    from app.models.knowledge import KnowledgeChunk
    from app.services.review_publish import PublicationPending, ReviewPublisher
    env = publication_mysql
    original_encode = env.encoder.encode
    async def crash(texts):
        """数据库发布已提交后拒绝编码。参数：文本；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("after mysql publication")
    def reject_status(connection, cursor, statement, parameters, context, executemany):
        """在真实审核完成UPDATE前断开业务路径。参数：SQL事件；返回：None；异常：RuntimeError。"""
        if statement.startswith("UPDATE review_queue SET review_status="):
            raise RuntimeError("before review complete")
    if point == "mysql":
        env.encoder.encode = crash
    elif point == "vector":
        env.index.fail_after_upsert = True
    else:
        event.listen(env.lease_engine.sync_engine, "before_cursor_execute", reject_status)
    try:
        with pytest.raises(PublicationPending):
            await env.publisher.approve(env.review_id, "  人工原样答复：保留199元、否定和空白。  ")
    finally:
        if point == "review":
            event.remove(env.lease_engine.sync_engine, "before_cursor_execute", reject_status)
    first = await env.repo.review_detail(env.review_id)
    assert first["review_status"] == "待审" and first["publication_status"] == "pending"
    async with env.factory() as session:
        chunks = (await session.scalars(select(KnowledgeChunk))).all()
        assert len(chunks) == 1
        chunk_id = chunks[0].id
        assert chunks[0].vectorize_status == ("done" if point == "review" else "pending")
        assert chunks[0].answer == first["approved_answer"]
    env.encoder.encode, env.index.fail_after_upsert = original_encode, False
    restarted = ReviewPublisher(env.repo, env.store, env.encoder, env.index)
    try:
        assert await restarted.recover_pending() == {"published": 1, "failed": 0}
    finally:
        await restarted.close()
    async with env.factory() as session:
        chunks = (await session.scalars(select(KnowledgeChunk))).all()
        assert len(chunks) == 1 and chunks[0].id == chunk_id and chunks[0].is_active
        assert chunks[0].source_key == f"ch09/review/{env.review_id}" and chunks[0].ordinal == 0
        assert chunks[0].vectorize_status == "done" and chunks[0].vector_id == str(chunk_id)
        assert await env.index.get_fingerprints([chunk_id]) == {chunk_id: (chunks[0].content_hash, chunks[0].embedding_revision)}
    assert (await env.repo.review_detail(env.review_id))["review_status"] == "通过"
    print(f"真实MySQL断点 {point}：审核冻结待审→新发布器恢复；知识主键{chunk_id}不变且仅1 active块；索引替身指纹一致")


async def test_mysql_review_concurrent_freeze_and_reject_are_atomic(publication_mysql):
    """多个真实事务只能冻结同一正文，审核和驳回只有一方成功。参数：专库；返回：None；异常：断言失败。"""
    import asyncio
    from app.repositories.flywheel import ReviewConflict
    from app.models.flywheel import ReviewQueue
    env = publication_mysql
    saved = await asyncio.gather(*(env.repo.freeze_approval(env.review_id, "人工唯一原答") for _ in range(8)))
    assert all(row["approved_answer"] == "人工唯一原答" and row["publication_status"] == "pending" for row in saved)
    with pytest.raises(ReviewConflict):
        await env.repo.freeze_approval(env.review_id, "另一个答案")
    with pytest.raises(ReviewConflict):
        await env.repo.reject(env.review_id)
    async with env.factory.begin() as session:
        other = ReviewQueue(normalized_question="竞争审核")
        session.add(other)
        await session.flush()
    outcomes = await asyncio.gather(env.repo.freeze_approval(other.id, "人审答"), env.repo.reject(other.id), return_exceptions=True)
    assert sum(isinstance(outcome, ReviewConflict) for outcome in outcomes) == 1
    detail = await env.repo.review_detail(other.id)
    assert (detail["review_status"], detail["approved_answer"]) in [("待审", "人审答"), ("驳回", None)]
    print("真实MySQL：8会话同答案冻结稳定；不同正文/冻结后驳回409；冻结与驳回并发只允许一方")


async def test_mysql_review_real_tokenizer_overflow_precedes_freeze(publication_mysql):
    """使用正式BGE tokenizer验证字符未超但token超限时仍未冻结。参数：专库配置；返回：None；异常：断言/分词器错误。"""
    from app.knowledge.embedding import BgeM3Encoder
    from app.knowledge.schemas import KnowledgeText
    from app.knowledge.text import format_knowledge
    from app.services.review_publish import ReviewPublisher
    env = publication_mysql
    encoder = BgeM3Encoder(env.config)
    publisher = ReviewPublisher(env.repo, env.store, encoder, env.index)
    # NFKC会把此兼容字符展开成多个日文片段；真实计数高于原字符串字符数。
    answer = "㌖" * 1200
    knowledge = KnowledgeText(category="商品与购物 FAQ", questions="实际审核的问题", answer=answer)
    try:
        count = encoder.count_tokens(format_knowledge(knowledge))
        assert len(knowledge.category) + len(knowledge.questions) + len(answer) <= 1600 and count > 2048
        with pytest.raises(ValueError, match="超过"):
            await publisher.approve(env.review_id, answer)
        assert (await env.repo.review_detail(env.review_id))["approved_answer"] is None
        assert encoder._model is None
        print(f"真实BGE tokenizer：三格字符{len(knowledge.category) + len(knowledge.questions) + len(answer)}，含特殊token共{count}；拒绝前未冻结且未加载推理模型")
    finally:
        await publisher.close()
        await encoder.aclose()


async def test_mysql_review_close_releases_real_writer_before_dependency_close(publication_mysql, monkeypatch):
    """关闭正式发布器后真实CH03命名锁连接已回收，冻结答案可恢复。参数：专库/补丁；返回：None；异常：断言失败。"""
    import asyncio
    from app.knowledge.lease import MySqlWriteLease, KNOWLEDGE_WRITE_LOCK
    env = publication_mysql
    entered, captured = asyncio.Event(), []
    original_acquire = MySqlWriteLease.acquire
    async def acquire(cls, engine, **kwargs):
        """登记本轮专库真实租约连接。参数：正式参数；返回：原租约；异常：原样传播。"""
        lease = await original_acquire(engine, **kwargs)
        assert await lease.connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
        captured.append((lease, await lease.connection.scalar(text("SELECT CONNECTION_ID()"))))
        await lease.connection.commit()
        return lease
    monkeypatch.setattr(MySqlWriteLease, "acquire", classmethod(acquire))
    async def encode(texts):
        """保持编码等待以验证取消。参数：文本；返回：不返回；异常：取消。"""
        entered.set()
        await asyncio.Event().wait()
    env.encoder.encode = encode
    task = asyncio.create_task(env.publisher.approve(env.review_id, "关闭后保留答案"))
    await asyncio.wait_for(entered.wait(), 5)
    await asyncio.wait_for(env.publisher.close(), 5)
    assert task.done() and task.cancelled() and not env.publisher._active
    assert captured[0][0].connection.closed
    async with env.engine.connect() as connection:
        assert await connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
        assert await connection.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": KNOWLEDGE_WRITE_LOCK}) is None
        assert await connection.scalar(text("SELECT COUNT(*) FROM information_schema.PROCESSLIST WHERE ID=:id"), {"id": captured[0][1]}) == 0
    assert (await env.repo.review_detail(env.review_id))["publication_status"] == "pending"
    print("真实MySQL：发布器关闭已取消并收回编码等待，专用物理连接closed且PROCESSLIST无登记ID，CH03命名锁无owner；人工答案保留")


async def test_mysql_review_completion_wait_loses_owned_writer_connection(publication_mysql, monkeypatch):
    """最终核锁后在审核行锁等待中KILL本轮专用连接，旧完成事务不得提交。参数：空专库/补丁；返回：None；异常：断言失败。"""
    import asyncio
    from sqlalchemy import select
    from app.knowledge.lease import MySqlWriteLease, KNOWLEDGE_WRITE_LOCK
    from app.models.flywheel import ReviewQueue
    from app.services.review_publish import PublicationPending
    env = publication_mysql
    acquired, completing = [], []
    entered = asyncio.Event()
    blocker = env.factory()
    armed = False
    original_acquire, original_fingerprints = MySqlWriteLease.acquire, env.index.get_fingerprints
    async def acquire(cls, engine, **kwargs):
        """仅登记本轮自行取得的专库CH03租约。参数：租约参数；返回：真实租约；异常：身份错误。"""
        lease = await original_acquire(engine, **kwargs)
        assert await lease.connection.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
        acquired.append(await lease.connection.scalar(text("SELECT CONNECTION_ID()")))
        await lease.connection.commit()
        return lease
    async def fingerprints(ids):
        """最终Strong替身回读后由另一真实事务锁住审核行。参数：知识ID；返回：原指纹；异常：数据库错误。"""
        nonlocal armed
        result = await original_fingerprints(ids)
        await blocker.begin()
        await blocker.scalar(select(ReviewQueue).where(ReviewQueue.id == env.review_id).with_for_update())
        armed = True
        return result
    def waiting(connection, cursor, statement, parameters, context, executemany):
        """记录已进入最终完成SELECT的真实连接，后续核对服务器活动SQL。参数：SQL事件；返回：None；异常：数据库错误。"""
        if armed and statement.startswith("SELECT review_queue.") and "FOR UPDATE" in statement:
            completing.append(connection.scalar(text("SELECT CONNECTION_ID()")))
            entered.set()
    monkeypatch.setattr(MySqlWriteLease, "acquire", classmethod(acquire))
    env.index.get_fingerprints = fingerprints
    event.listen(env.engine.sync_engine, "before_cursor_execute", waiting)
    event.listen(env.lease_engine.sync_engine, "before_cursor_execute", waiting)
    operation = asyncio.create_task(env.publisher.approve(env.review_id, "最终事务必须仍受原租约保护"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert len(acquired) == len(completing) == 1
        async with env.factory() as control:
            assert await control.scalar(text("SELECT DATABASE()")) == TEST_DATABASE
            # 确认完成SELECT已经送达服务器且被另一事务持有的审核行阻塞。
            async with asyncio.timeout(5):
                while True:
                    info = await control.scalar(text("SELECT INFO FROM information_schema.PROCESSLIST WHERE ID=:id"), {"id": completing[0]})
                    if info and "review_queue" in info and "FOR UPDATE" in info:
                        break
                    await asyncio.sleep(0.01)
            identifier = await control.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": KNOWLEDGE_WRITE_LOCK})
            assert type(identifier) is int and identifier > 0 and identifier == acquired[0]
            assert await control.scalar(text("SELECT DB FROM information_schema.PROCESSLIST WHERE ID=:id"), {"id": identifier}) == TEST_DATABASE
            # 只KILL本轮登记、库名/CH03命名锁owner均再次核对过的精确ID。
            await control.execute(text(f"KILL CONNECTION {identifier}"))
            await control.commit()
        async with env.store.writer() as fresh:
            current = (await fresh.chunks_by_sources([f"ch09/review/{env.review_id}"]))[0]
            assert await fresh.mark_pending(current)
            await env.index.delete([current.id])
        await blocker.rollback()
        with pytest.raises(PublicationPending):
            await asyncio.wait_for(operation, 5)
        assert completing[0] == acquired[0], "完成事务必须绑定原writer连接"
        assert (await env.repo.review_detail(env.review_id))["review_status"] == "待审"
        print(f"真实MySQL：完成SELECT在连接{completing[0]}等待审核行锁；仅KILL本轮CH03 owner {acquired[0]}；新writer可改知识，旧完成仍待审")
    finally:
        event.remove(env.engine.sync_engine, "before_cursor_execute", waiting)
        event.remove(env.lease_engine.sync_engine, "before_cursor_execute", waiting)
        await blocker.rollback()
        await blocker.close()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


async def test_mysql_review_detail_preserves_raw_null_and_distinguishes_history(merge_mysql):
    """真实SQLNULL原样保留，只有精确现代原轮才能展示明确未检索。参数：专库；返回：None；异常：断言失败。"""
    from sqlalchemy import select
    from app.models.flywheel import ReviewQueue
    from app.models.customer_service import Message
    from app.models.rag_quality import LowConfidenceQuestion
    from tests.flywheel.test_pool import start
    chat, repo, factory, _ = merge_mysql
    ctx = await start(chat, question="相同问题")
    await chat.finish_turn(ctx, "不需要检索的回答")
    receipt = await chat.record_unresolved_feedback("s", "u", "first")
    async with factory.begin() as session:
        review = ReviewQueue(normalized_question="待审问题")
        session.add(review)
        await session.flush()
        current = await session.get(LowConfidenceQuestion, int(receipt.pool_id))
        current.matched_review_id = review.id
        session.add(LowConfidenceQuestion(conversation_id=ctx.conversation_id, raw_question="相同问题", source="retrieval_low_conf", matched_review_id=review.id))
    async with factory() as session:
        before = [(row.id, row.rag_metadata) for row in (await session.scalars(select(Message))).all()]
        assert (await session.execute(text("SELECT retrieved_chunks IS NULL FROM low_confidence_questions ORDER BY id"))).scalars().all() == [1, 1]
    current, legacy = (await repo.review_detail(review.id))["originals"]
    assert current["retrieved_chunks"] is legacy["retrieved_chunks"] is None
    assert current["retrieval_status"] == "not_retrieved" and current["retrieval_snapshot"] is None
    assert legacy["retrieval_status"] == "legacy_unavailable" and legacy["retrieval_snapshot"]["status"] == "legacy_unavailable"
    async with factory() as session:
        assert (await session.execute(text("SELECT retrieved_chunks IS NULL FROM low_confidence_questions ORDER BY id"))).scalars().all() == [1, 1]
        assert before == [(row.id, row.rag_metadata) for row in (await session.scalars(select(Message))).all()]
    print("真实MySQL：两条SQLNULL未改写；按pool_ids精确原轮展示not_retrieved，旧同文retrieval_low_conf原件展示legacy_unavailable；用户JSON未写入")
