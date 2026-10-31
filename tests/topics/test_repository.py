"""主题统计与存储边界：精确轮次并集、全池覆盖和非法写入。"""

import importlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session


def module():
    """加载待实现仓储。参数：无；返回：模块；异常：断言失败。"""
    assert importlib.util.find_spec("app.repositories.topics"), "缺少主题仓储"
    return importlib.import_module("app.repositories.topics")


def test_distribution_counts_turn_union_and_legacy_ids():
    """按行或相同文本合并会误计。参数：无；返回：None；异常：断言失败。"""
    rows = [dict(question_id=str(i), group_id=g, history_uncertain=h, review_status=s, text="同文")
            for i, g, h, s in [(1, "turn:a", False, None), (2, "turn:a", False, "待审"),
                               (3, "turn:b", False, None), (4, "legacy-pool:4", True, None),
                               (5, "turn:c", False, "通过"), (6, "turn:d", False, "驳回")]]
    classes = [dict(question_id=str(i), labels=labels) for i, labels in
               [(1, ["物流"]), (2, ["物流", "运费"]), (3, ["物流"]), (5, ["其他"])]]
    result = module().aggregate_distribution(rows, classes, "pending")
    assert result["total_questions"] == 3
    assert result["classified_questions"] == 2 and result["unclassified_questions"] == 1
    assert result["pool_rows"] == 4 and result["classified_pool_rows"] == 3
    assert result["known_original_turns"] == 2 and result["history_uncertain_pool_rows"] == 1
    counts = {row["label"]: row["count"] for row in result["labels"]}
    assert len(counts) == 17 and counts["物流"] == 2 and counts["运费"] == 1 and counts["其他"] == 0
    assert module().aggregate_distribution(rows, classes, "all")["total_questions"] == 5


def test_distribution_includes_all_pages():
    """统计截断到一页会漏掉最后组。参数：无；返回：None；异常：断言失败。"""
    rows = [dict(question_id=str(i), group_id=f"legacy-pool:{i}", history_uncertain=True, review_status=None)
            for i in range(1, 402)]
    result = module().aggregate_distribution(rows, [dict(question_id="401", labels=["发票"])], "all")
    assert result["total_questions"] == 401 and result["classified_questions"] == 1


@pytest.mark.parametrize("rows", [[{"question_id": "1", "labels": ["其他", "物流"]}],
    [{"question_id": "1", "labels": ["物流", "物流"]}], [{"question_id": "1", "labels": ["未知"]}],
    [{"question_id": "1", "labels": []}], [{"question_id": "1", "labels": ["物流"]}] * 2])
async def test_save_rejects_invalid_or_duplicate_before_connection(rows):
    """非法或重复输入不能触达数据库。参数：rows错误输入；返回：None；异常：断言失败。"""
    repo = module().TopicRepository(None, None)
    with pytest.raises(ValueError):
        await repo.save(rows, lease=NS())


def test_source_counts_are_pool_rows_with_scope_filter():
    """来源按池行统计，同组两入口不能合并成一条；审核范围必须过滤。参数：无；返回：None；异常：断言失败。"""
    rows = [dict(question_id=str(i), group_id=group, history_uncertain=history, review_status=review, source=source)
        for i, group, history, review, source in [
            (1, "turn:a", False, None, "application_demo"),
            (2, "turn:a", False, "待审", "application_demo"),
            (3, "legacy-pool:3", True, None, "application_history"),
            (4, "turn:b", False, "通过", "application_demo"),
            (5, "legacy-pool:5", True, "驳回", "application_history")]]
    pending = module().aggregate_distribution(rows, [], "pending")
    assert pending["source_counts"] == {"application_demo": 2, "application_history": 1}
    assert pending["total_questions"] == 2 and pending["pool_rows"] == 3
    assert "演示记录2条" in pending["source_note"] and "来源未核实历史1条" in pending["source_note"]
    assert module().aggregate_distribution(rows, [], "all")["source_counts"] == {
        "application_demo": 3, "application_history": 2}


@pytest.mark.parametrize("reclassify,expected", [(False, ["3"]), (True, ["2"])])
async def test_pending_limits_candidates_and_reads_only_their_metadata(reclassify, expected):
    """以真实SQL验证分页和关联范围，批外歧义不能阻塞。参数：reclassify布尔重跑开关、expected预期候选ID；返回：None；异常：断言失败。"""
    engine = create_engine("sqlite://")
    queries = []

    def record(connection, cursor, statement, parameters, context, many):
        """记录实际查询及绑定值。参数：SQLAlchemy游标执行事件六参数；返回：None；异常：无。"""
        queries.append((statement, parameters))

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE low_confidence_questions (id INTEGER PRIMARY KEY, conversation_id INTEGER, raw_question TEXT, source TEXT, reason TEXT, retrieved_chunks JSON, matched_review_id INTEGER, created_at DATETIME)"))
        connection.execute(text("CREATE TABLE topic_classifications (id INTEGER PRIMARY KEY, question_id INTEGER, labels JSON, classified_at DATETIME)"))
        connection.execute(text("CREATE TABLE conversations (id INTEGER PRIMARY KEY, session_id TEXT, user_id TEXT, status TEXT, summary TEXT, summary_upto_msg_id INTEGER, layer1_from_msg_id INTEGER, created_at DATETIME, updated_at DATETIME)"))
        connection.execute(text("CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, role TEXT, content TEXT, tool_calls JSON, rag_metadata JSON, tool_call_id TEXT, turn_id TEXT, turn_status TEXT, created_at DATETIME)"))
        connection.execute(text("CREATE TABLE review_queue (id INTEGER PRIMARY KEY, normalized_question TEXT, ai_suggested_answer TEXT, occurrence_count INTEGER, review_status TEXT, approved_answer TEXT, created_at DATETIME, updated_at DATETIME)"))
        for identifier in range(1, 5):
            connection.execute(text("INSERT INTO low_confidence_questions (id, conversation_id, raw_question, source, matched_review_id) VALUES (:id,:id,'尺寸是多少','self_check',:id)"), {"id": identifier})
            connection.execute(text("INSERT INTO conversations (id,session_id,user_id) VALUES (:id,'demo-test','test')"), {"id": identifier})
            connection.execute(text("INSERT INTO review_queue (id,review_status) VALUES (:id,'待审')"), {"id": identifier})
            connection.execute(text("INSERT INTO messages (id,conversation_id,role,rag_metadata) VALUES (:id,:id,'user',:metadata)"), {"id": identifier, "metadata": json.dumps({"knowledge_pool": {"pool_ids": {"self_check": identifier}}})})
        # 第四条有两条原轮证据，但它不属于当前一条候选。
        connection.execute(text("INSERT INTO messages (id,conversation_id,role,rag_metadata) VALUES (40,4,'user',:metadata)"), {"metadata": json.dumps({"knowledge_pool": {"pool_ids": {"self_check": 4}}})})
        connection.execute(text("INSERT INTO topic_classifications (id,question_id,labels,classified_at) VALUES (2,2,'[\"物流\"]','2026-11-01 00:00:00')"))

    event.listen(engine, "before_cursor_execute", record)

    @asynccontextmanager
    async def factory():
        """提供真实只读SQL的异步调用适配。参数：无；返回：异步上下文Session适配器；异常：数据库错误。"""
        with Session(engine) as session:
            async def scalars(statement):
                """执行实际查询。参数：statement为只读Select；返回：ScalarResult；异常：数据库错误。"""
                return session.scalars(statement)
            yield NS(scalars=scalars)

    try:
        rows = await module().TopicRepository(factory, None).pending(1, reclassify, after_id=1)
        assert [row["question_id"] for row in rows] == expected
        assert rows[0]["review_status"] == "待审" and not rows[0]["history_uncertain"]
        candidate = queries[0][0]
        assert "LIMIT" in candidate and "low_confidence_questions.id >" in candidate
        assert ("EXISTS" in candidate) != reclassify
        assert len(queries) == 4
        for statement, parameters in queries[1:]:
            assert "WHERE" in statement and int(expected[0]) in parameters
            assert "messages.content" not in statement and "conversations.summary" not in statement
            assert "review_queue.ai_suggested_answer" not in statement
        queries.clear()
        assert await module().TopicRepository(factory, None).pending(1, after_id=100) == []
        assert len(queries) == 1
    finally:
        engine.dispose()
