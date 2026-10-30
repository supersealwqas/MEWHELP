"""仅在显式开启的空 mewhelp_ch09_test 验证真实 MySQL 迁移断点与 JSON。

不操作应用库。若专库存在任何表则拒绝运行；只清理本轮成功登记创建的五张表。
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
OWNED_TABLES = ("low_confidence_questions", "faith_cases", "eval_runs", "review_queue", "conversations")


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
