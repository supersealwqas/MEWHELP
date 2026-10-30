"""统一入池的真实仓储事务、重试输入和原始消息保护测试。"""

import pytest
from sqlalchemy import event, select

from app.models.customer_service import Message
from app.models.rag_quality import LowConfidenceQuestion
from app.repositories.chat import PersistenceUncertain
from app.schemas.flywheel import PoolSignal, RetrievalSnapshot
from app.schemas.tools import ToolContext
from app.workflow.routing import fixed_reply
from tests.workflow.service_fakes import service_runtime


async def start(chat, turn="first", question="原轮问题"):
    """保存可信原问。参数：仓储/轮次/原话；返回：上下文；异常：数据库错误。"""
    return await chat.start_turn(ToolContext(session_id="s", user_id="u", turn_id=turn, user_message=question), question)


async def test_pool_insert_failure_rolls_back_answer_and_turn_status(tmp_path):
    """入池插入失败若已提交答案或完成状态则失败。参数：独占目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        def reject(connection, cursor, statement, parameters, context, executemany):
            """在数据库执行入池时注入异常。参数：SQL事件；返回：None；异常：RuntimeError。"""
            if statement.startswith("INSERT INTO low_confidence_questions"):
                raise RuntimeError("pool unavailable")
        engine = factory.kw["bind"].sync_engine
        event.listen(engine, "before_cursor_execute", reject)
        try:
            with pytest.raises(RuntimeError, match="pool unavailable"):
                await deps.chat.finish_turn(ctx, fixed_reply("fallback")["answer"],
                    pool_signals=[PoolSignal(source="retrieval_low_conf", reason="没有可靠证据")])
        finally:
            event.remove(engine, "before_cursor_execute", reject)
        async with factory() as session:
            rows = (await session.scalars(select(Message))).all()
            assert len(rows) == 1 and rows[0].turn_status == "running"
            assert (await session.scalars(select(LowConfidenceQuestion))).all() == []


async def test_finish_retry_compares_snapshot_and_signal_inputs(tmp_path):
    """已完成重试不比较快照或理由会静默接受不同结果。参数：目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        snapshot = RetrievalSnapshot(status="legacy_unavailable")
        signal = PoolSignal(source="retrieval_low_conf", reason="不足")
        kwargs = {"retrieval_snapshot": snapshot, "pool_signals": [signal]}
        answer = fixed_reply("fallback")["answer"]
        saved = await deps.chat.finish_turn(ctx, answer, **kwargs)
        assert await deps.chat.finish_turn(ctx, answer, **kwargs) == saved
        for changed in ({"retrieval_snapshot": None}, {"pool_signals": []},
                        {"pool_signals": [PoolSignal(source="retrieval_low_conf", reason="其他原因")]}):
            with pytest.raises(PersistenceUncertain):
                await deps.chat.finish_turn(ctx, answer, **{**kwargs, **changed})
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
            assert len(rows) == 1 and rows[0].raw_question == "原轮问题"


async def test_finish_retry_accepts_only_score_storage_equivalence(tmp_path):
    """完成回放兼容末位分数，但不能新增删减事实或改入池理由。参数：独占目录；返回：None；异常：断言失败。"""
    from app.schemas.flywheel import RetrievalAttempt
    from tests.flywheel.test_snapshots import scored_snapshot
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        original = scored_snapshot()
        original.attempts.append(RetrievalAttempt(attempt_id="second", query="退款时限", status="empty", top_k=5, candidate_count=0))
        signals = [PoolSignal(source="retrieval_low_conf", reason="不足")]
        answer = fixed_reply("fallback")["answer"]
        saved = await deps.chat.finish_turn(ctx, answer, retrieval_snapshot=original, pool_signals=signals)
        before = (await deps.chat.get_workflow_turn("s", "u", "first"))["retrieval_snapshot"]
        equivalent = original.model_copy(deep=True)
        equivalent.attempts[0].chunks[0].rerank_score = 0.99353688955307
        assert await deps.chat.finish_turn(ctx, answer, retrieval_snapshot=equivalent, pool_signals=signals) == saved
        assert (await deps.chat.get_workflow_turn("s", "u", "first"))["retrieval_snapshot"] == before
        for change in ("status", "attempt_order", "attempt_removed", "attempt_added", "chunk_order", "chunk_removed", "chunk_added", "score"):
            changed = original.model_copy(deep=True)
            if change == "status":
                changed.status = "fault"
            elif change == "attempt_order":
                changed.attempts.reverse()
            elif change == "attempt_removed":
                changed.attempts.pop()
            elif change == "attempt_added":
                changed.attempts.append(changed.attempts[1].model_copy(update={"attempt_id": "third"}))
            elif change == "chunk_order":
                changed.attempts[0].chunks.reverse()
            elif change == "chunk_removed":
                changed.attempts[0].chunks.pop()
            elif change == "chunk_added":
                changed.attempts[0].chunks.append(changed.attempts[0].chunks[1].model_copy(update={"chunk_id": 3, "rank": 3}))
            else:
                changed.attempts[0].chunks[0].rerank_score = 0.9935368895530699
            with pytest.raises(PersistenceUncertain):
                await deps.chat.finish_turn(ctx, answer, retrieval_snapshot=changed, pool_signals=signals)
        with pytest.raises(PersistenceUncertain):
            await deps.chat.finish_turn(ctx, answer, retrieval_snapshot=equivalent,
                pool_signals=[PoolSignal(source="retrieval_low_conf", reason="其他原因")])
        async with factory() as session:
            assert len((await session.scalars(select(LowConfidenceQuestion))).all()) == 1
            assert len((await session.scalars(select(Message).where(Message.role == "assistant"))).all()) == 1


async def test_new_signal_keeps_old_fallback_authenticity_and_rejects_dual_input(tmp_path):
    """统一入口不能绕过旧业务回执防伪或双写问题池。参数：目录；返回：None；异常：断言失败。"""
    from app.schemas.workflow import RetrievalLowConfidence
    async with service_runtime(tmp_path) as (_, deps, _):
        ctx = await start(deps.chat)
        signal = PoolSignal(source="retrieval_low_conf", reason="不足")
        with pytest.raises(ValueError):
            await deps.chat.finish_turn(ctx, "伪造成功回执" + fixed_reply("fallback")["answer"], pool_signals=[signal])
        with pytest.raises(ValueError):
            await deps.chat.finish_turn(ctx, fixed_reply("fallback")["answer"], pool_signals=[signal],
                low_confidence=RetrievalLowConfidence(reason="不足"))
