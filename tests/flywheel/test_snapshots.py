"""检索快照真实性测试：低分、裁剪、故障和同轮恢复不能丢失原始证据。"""

import importlib
import json

import pytest

from app.core.model_budget import BudgetLedger
from app.schemas.tools import ToolContext
from app.tools.knowledge import retrieve_verified_knowledge
from app.workflow.state import new_turn_state
from tests.rag.test_retrieval import _chunk, _search


def snapshots():
    """加载待实现接口。参数：无；返回：快照模块；异常：实现不存在时断言失败。"""
    assert importlib.util.find_spec("app.knowledge.snapshots"), "缺少原轮快照实现"
    return importlib.import_module("app.knowledge.snapshots")


async def test_low_score_snapshot_preserves_top_five_before_gate():
    """先按低分清空证据会丢失审核原文。参数：无；返回：None；异常：断言失败。"""
    snapshots()
    search, _, _, _ = _search([_chunk(i) for i in range(1, 9)], scores={i: .1 for i in range(1, 9)})
    state = new_turn_state(ToolContext(conversation_id=1, session_id="s", user_id="u", turn_id="first", user_message="退货政策"), [], 0)
    outcome = await retrieve_verified_knowledge(state, BudgetLedger(), search=search, settings=search.config,
        query_expander=None, query="退货政策", prepared_queries=["退货政策"])
    assert outcome.evidence == []
    assert len(outcome.retrieval_snapshot.attempts[0].chunks) == 5
    assert outcome.retrieval_snapshot.attempts[0].candidate_count == 8


async def test_snapshot_pairs_scores_by_chunk_id_after_sort_and_trim():
    """按列表位置配分会把第八块的分错给第一块。参数：无；返回：None；异常：断言失败。"""
    capture = snapshots().RetrievalCapture("first", "sort")
    scores = {i: i / 10 for i in range(1, 9)}
    search, _, _, _ = _search([_chunk(i) for i in range(1, 9)], scores=scores)
    result = await search.retrieve("退货政策", top_k=2, capture=capture)
    assert [item.chunk.id for item in result] == [8, 7]
    assert [item.chunk_id for item in capture.snapshot.attempts[0].chunks] == [8, 7, 6, 5, 4]
    for item in capture.snapshot.attempts[0].chunks:
        assert item.rerank_score == scores[item.chunk_id] and item.recall_score == .01
        assert item.answer == f"权威答案{item.chunk_id}。"


async def test_no_retrieval_empty_and_fault_are_distinct():
    """未知或故障不能伪装成零结果或没有检索。参数：无；返回：None；异常：断言失败。"""
    module = snapshots()
    capture = module.RetrievalCapture("turn", "empty")
    assert capture.snapshot is None
    search, encoder, _, _ = _search([])
    await search.retrieve("空库问题", capture=capture)
    assert capture.snapshot.status == "empty" and capture.snapshot.attempts[0].candidate_count == 0
    capture = module.RetrievalCapture("turn", "fault")
    async def fail(texts):
        """模拟真实外部编码故障。参数：文本；返回：无；异常：ConnectionError。"""
        raise ConnectionError("upstream unavailable")
    encoder.encode = fail
    with pytest.raises(ConnectionError):
        await search.retrieve("失败问题", capture=capture)
    assert capture.snapshot.status == "fault" and capture.snapshot.attempts[0].candidate_count is None
    assert module.recover_legacy_snapshot([]).status == "legacy_unavailable"
    assert module.recover_legacy_snapshot([{"role": "user", "rag_metadata": {"retrieval_snapshot": None}}]) is None


async def test_multiple_retrieval_attempts_survive_confirmation_resume():
    """恢复或同轮缓存不能覆盖暂停前检索。参数：无；返回：None；异常：断言失败。"""
    module = snapshots()
    search, _, index, _ = _search([_chunk(1)])
    state = new_turn_state(ToolContext(conversation_id=1, session_id="s", user_id="u", turn_id="first", user_message="退货政策"), [], 0)
    options = dict(search=search, settings=search.config, query_expander=None)
    await retrieve_verified_knowledge(state, BudgetLedger(), query="退货政策", prepared_queries=["退货政策"], **options)
    resumed = json.loads(json.dumps(state))
    await retrieve_verified_knowledge(resumed, BudgetLedger(), query="退款时限", prepared_queries=["退款时限"], **options)
    repeated = await retrieve_verified_knowledge(resumed, BudgetLedger(), query="退货政策", prepared_queries=["退货政策"], **options)
    assert len(index.calls) == 2
    assert [a.query for a in repeated.retrieval_snapshot.attempts] == ["退货政策", "退款时限"]
    assert module.merge_snapshots(repeated.retrieval_snapshot, repeated.retrieval_snapshot) == repeated.retrieval_snapshot
    assert new_turn_state(ToolContext(session_id="s", user_id="u", turn_id="next", user_message="新轮"), [], 0)["retrieval_snapshot"] is None


async def test_partial_failure_is_persisted_before_retry_without_replacing_history(tmp_path):
    """后续子查询失败或重试不能抹掉已保存原文和失败事实。参数：目录；返回：None；异常：断言失败。"""
    from tests.flywheel.test_pool import start
    from tests.workflow.service_fakes import service_runtime
    search, encoder, _, _ = _search([_chunk(1)])
    original = encoder.encode
    async def fail_second(texts):
        """第二个真实编码失败。参数：查询文本；返回：向量；异常：ConnectionError。"""
        if "二" in texts[0]:
            raise ConnectionError("unavailable")
        return await original(texts)
    encoder.encode = fail_second
    async with service_runtime(tmp_path) as (_, deps, _):
        ctx = await start(deps.chat, question="退款规则")
        state = new_turn_state(ctx, [], 0)
        kwargs = dict(search=search, settings=search.config, query_expander=None, query="退款规则",
                      prepared_queries=["退款规则一", "退款规则二", "退款规则三"], policy=True, chat=deps.chat)
        with pytest.raises(ConnectionError):
            await retrieve_verified_knowledge(state, BudgetLedger(), **kwargs)
        saved = (await deps.chat.get_workflow_turn("s", "u", "first"))["retrieval_snapshot"]
        assert saved["status"] == "fault" and [a["status"] for a in saved["attempts"]] == ["ok", "fault"]
        encoder.encode = original
        outcome = await retrieve_verified_knowledge(state, BudgetLedger(), **kwargs)
        assert outcome.retrieval_snapshot.status == "fault"
        assert len(outcome.retrieval_snapshot.attempts) == 5


def test_legacy_recovery_uses_original_citations_and_marks_missing_results_unknown():
    """历史最终引用不能丢失，缺工具回执也不代表没检索。参数：无；返回：None；异常：断言失败。"""
    module = snapshots()
    saved = module.recover_legacy_snapshot([{"role": "user", "content": "原问"},
        {"role": "assistant", "rag_metadata": {"citations": [{"question": "政策问法", "answer": "当时原文", "content_hash": "旧引用的答案指纹"}]}}])
    assert saved.status == "legacy_incomplete"
    attempt = saved.attempts[0]
    assert attempt.query is None and attempt.candidate_count is None and attempt.top_k is None
    assert attempt.chunks[0].chunk_id is None and attempt.chunks[0].answer == "当时原文"
    assert attempt.chunks[0].content_hash is None
    missing = module.recover_legacy_snapshot([{"role": "assistant", "tool_calls": [
        {"id": "lost", "name": "query_faq", "args": {"keyword": "真实查询"}}]}])
    assert missing.status == "legacy_incomplete" and missing.attempts[0].query == "真实查询"


async def test_real_confirmation_pause_and_resume_keep_both_knowledge_attempts(tmp_path):
    """真实图暂停/节点重放必须保留先前两个检索且不重新查知识。参数：目录；返回：None；异常：断言失败。"""
    from langgraph.types import Command
    from app.schemas.workflow import AgentDecision
    from tests.workflow.ch08_ticket_fakes import saved_decision, ticket_runtime
    from tests.workflow.test_ch08_ticket_interrupt import begin, ticket_call
    calls = [{"id": "faq-1", "name": "query_faq", "args": {"keyword": "退货条件"}},
             {"id": "faq-2", "name": "query_faq", "args": {"keyword": "退款时限"}}, ticket_call()]
    async with ticket_runtime(tmp_path, intents=["订单"], decisions=[AgentDecision(kind="tools", tool_calls=calls),
            AgentDecision(kind="answer")], answers=[["工单已创建。"]]) as (service, deps, factory, _):
        context, config, paused = await begin(service.graph, deps.chat)
        assert paused.interrupts
        original = (await deps.chat.get_workflow_turn(context.session_id, context.user_id, context.turn_id))["retrieval_snapshot"]
        assert len(original["attempts"]) == 2
        final = await service.graph.ainvoke(Command(resume=await saved_decision(factory, context)), config)
        assert len(deps.search.calls) == 2
        assert final["retrieval_snapshot"] == original


async def test_filter_storage_failure_is_fault_before_any_candidate_is_known():
    """过滤权威存储提前失败不能当作未检索。参数：无；返回：None；异常：断言失败。"""
    capture = snapshots().RetrievalCapture("turn", "filter")
    search, _, _, _ = _search([_chunk(1)])
    async def fail(**kwargs):
        """模拟权威范围读取故障。参数：分页参数；返回：无；异常：ConnectionError。"""
        raise ConnectionError("scope unavailable")
    search.store.active_page = fail
    with pytest.raises(ConnectionError):
        await search.retrieve("退货政策", filters={"category": "售后"}, capture=capture)
    assert capture.snapshot is not None and capture.snapshot.status == "fault"
    assert capture.snapshot.attempts[0].candidate_count is None


async def test_rerank_failure_keeps_only_verified_originals_without_invented_scores():
    """精排故障仍应保存已核验召回原文，不能虚构精排分。参数：无；返回：None；异常：断言失败。"""
    capture = snapshots().RetrievalCapture("turn", "rerank-fault")
    search, _, _, reranker = _search([_chunk(i) for i in range(1, 9)])
    async def fail(*args, **kwargs):
        """模拟模型推理故障。参数：查询/块/数量；返回：无；异常：ConnectionError。"""
        raise ConnectionError("model unavailable")
    reranker.rank = fail
    with pytest.raises(ConnectionError):
        await search.retrieve("退货政策", capture=capture)
    attempt = capture.snapshot.attempts[0]
    assert attempt.status == "fault" and attempt.candidate_count == 8
    assert len(attempt.chunks) == 5 and all(item.recall_score == .01 and item.rerank_score is None for item in attempt.chunks)
