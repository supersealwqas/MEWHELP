"""CH09 共享数据契约、固定 DDL 与跨元数据引用的边界测试。"""

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable


def test_flywheel_orm_matches_fixed_ddl_and_sql_null() -> None:
    """字段越界、中文枚举或自动更新时间缺失应失败。参数：无；返回：None；异常：断言失败。"""
    from app.models.flywheel import EvalRun, ReviewQueue
    from app.models.rag_quality import LowConfidenceQuestion
    from app.models.customer_service import Base

    review, evaluation = ReviewQueue.__table__, EvalRun.__table__
    assert set(review.c.keys()) == {"id", "normalized_question", "ai_suggested_answer", "occurrence_count",
                                   "review_status", "approved_answer", "created_at", "updated_at"}
    assert set(evaluation.c.keys()) == {"id", "triggered_by", "dataset_size", "metrics", "created_at"}
    assert review.c.review_status.type.enums == ["待审", "通过", "驳回"]
    assert evaluation.c.triggered_by.type.enums == ["定时", "手动"]
    compiled = str(CreateTable(review).compile(dialect=mysql.dialect()))
    updated_definition = next(line for line in compiled.splitlines() if "updated_at" in line)
    assert "DATETIME NOT NULL" in updated_definition
    assert "DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP" in updated_definition
    assert LowConfidenceQuestion.__table__.c.retrieved_chunks.type.none_as_null is True
    foreign = next(key for key in LowConfidenceQuestion.__table__.foreign_keys if key.name == "fk_lcq_review")
    assert foreign.column is review.c.id and foreign.ondelete == "SET NULL"
    assert review.metadata is not Base.metadata
    assert "REFERENCES review_queue (id) ON DELETE SET NULL" in str(
        CreateTable(LowConfidenceQuestion.__table__).compile(dialect=mysql.dialect()))


def test_retrieval_contract_preserves_real_scores_and_all_statuses() -> None:
    """缺失分数被伪造或历史状态被合并时失败。参数：无；返回：None；异常：断言或校验错误。"""
    from app.schemas.flywheel import RetrievalAttempt, RetrievalChunkSnapshot, RetrievalSnapshot

    chunk = RetrievalChunkSnapshot(chunk_id=9007199254740993, source_key="ch09/review/2",
        content_hash="a" * 64, embedding_revision="revision", question="能退吗？", answer="核准答案", rank=1)
    assert chunk.recall_score is None and chunk.rerank_score is None
    attempt = RetrievalAttempt(attempt_id="attempt-1", query="能退吗？", status="ok", candidate_count=8,
                               top_k=5, chunks=[chunk], fault=None)
    snapshot = RetrievalSnapshot(status="ok", attempts=[attempt])
    assert snapshot.schema_version == 1 and snapshot.attempts[0].chunks[0].chunk_id == 9007199254740993
    for status in ("ok", "empty", "fault", "legacy_incomplete", "legacy_unavailable"):
        assert RetrievalSnapshot(status=status, attempts=[]).status == status
    with pytest.raises(ValidationError):
        RetrievalSnapshot(status="not_retrieved", attempts=[])
    with pytest.raises(ValidationError):
        RetrievalSnapshot(schema_version=2, status="empty", attempts=[])


def test_pool_state_is_internal_integer_and_receipt_is_decimal_string() -> None:
    """三入口限制、内部整数或前端十进制字符串边界丢失时失败。参数：无；返回：None；异常：断言或校验错误。"""
    from app.schemas.flywheel import FeedbackReceipt, PoolSignal, TurnPoolState

    for source in ("retrieval_low_conf", "self_check", "user_feedback"):
        assert PoolSignal(source=source, reason="证据不足").source == source
    with pytest.raises(ValidationError):
        PoolSignal(source="other", reason="证据不足")
    state = TurnPoolState(pool_ids={"user_feedback": 9007199254740993})
    assert state.schema_version == 1 and state.matched_review_id is None and state.counted_review_id is None
    assert state.pool_ids["user_feedback"] == 9007199254740993
    receipt = FeedbackReceipt(pool_id="9007199254740993", matched_review_id=None)
    assert receipt.model_dump() == {"pool_id": "9007199254740993", "matched_review_id": None}
    for value in (9007199254740993, "1e3", "-1", "1.1", "01", "0"):
        with pytest.raises(ValidationError):
            FeedbackReceipt(pool_id=value, matched_review_id=None)
    with pytest.raises(ValidationError):
        PoolSignal(source="self_check", reason="不足", raw_question="客户端伪造")


def test_legacy_and_fault_snapshots_do_not_invent_missing_history() -> None:
    """不允许未知候选数/旧指纹保持缺失时失败。参数：无；返回：None；异常：校验或断言失败。"""
    from app.schemas.flywheel import RetrievalAttempt, RetrievalChunkSnapshot

    legacy_chunk = RetrievalChunkSnapshot(chunk_id=None, source_key=None, content_hash=None,
        embedding_revision=None, question="旧问", answer="旧原文", rank=1)
    legacy = RetrievalAttempt(attempt_id="legacy-1", query=None, status="legacy_incomplete",
                              candidate_count=None, top_k=None, chunks=[legacy_chunk], fault=None)
    assert legacy.candidate_count is None and legacy.chunks[0].content_hash is None
    fault = RetrievalAttempt(attempt_id="failed-1", query="真实检索", status="fault",
                             candidate_count=None, top_k=5, chunks=[], fault="连接失败")
    assert fault.candidate_count is None
    for status in ("ok", "empty"):
        with pytest.raises(ValidationError):
            RetrievalAttempt(attempt_id="new-1", query=None, status=status, candidate_count=None,
                             top_k=None, chunks=[], fault=None)
    with pytest.raises(ValidationError):
        RetrievalAttempt(attempt_id="new-1", query="新问", status="ok", candidate_count=1,
                         top_k=5, chunks=[legacy_chunk], fault=None)
