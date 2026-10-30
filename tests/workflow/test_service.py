"""真实在线图入口、原子完成快照和低证据问题池的行为验证。"""

import pytest
from sqlalchemy import select

from app.models.customer_service import Message
from app.models.rag_quality import LowConfidenceQuestion
from app.repositories.chat import PersistenceUncertain
from app.schemas.tools import ToolContext
from app.schemas.workflow import RetrievalLowConfidence, SuggestedAction
from app.schemas.flywheel import PoolSignal, RetrievalSnapshot
from app.schemas.chat import public_rag_metadata
from tests.workflow.service_fakes import collect, service_runtime


async def test_real_graph_persists_suggestions_before_metadata_and_done(tmp_path):
    """服务绕过图或提前发完成帧会失败。参数：tmp_path独占目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["投诉"]) as (service, deps, factory):
        events = await collect(service, "  服务太差了  ")
        assert [event.event for event in events][-2:] == ["answer_metadata", "done"]
        metadata = events[-2].data
        assert metadata["suggested_actions"][0]["source_turn_id"] == metadata["turn_id"]
        assert metadata["suggested_actions"][1]["question"] == "  服务太差了  "
        async with factory() as session:
            rows = (await session.scalars(select(Message).order_by(Message.id))).all()
        assert len(rows) == 2 and all(row.turn_status == "completed" for row in rows)
        assert public_rag_metadata(rows[-1].rag_metadata) == metadata and metadata["workflow"]["intent"] == "投诉"
        assert {"retrieval_snapshot", "knowledge_pool", "trace_identity", "completion_inputs"}.isdisjoint(metadata)
        assert deps.executor.calls == []


async def test_low_evidence_finish_retry_keeps_one_snapshot_and_one_raw_question(tmp_path):
    """重复完成入池或新快照不同却接受重试会失败。参数：tmp_path目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["商品咨询"]) as (service, deps, factory):
        events = await collect(service, "  这个到底能不能用？  ")
        assert events[-1].event == "done"
        snapshot = (await service.graph.aget_state({"configurable": {"thread_id": "mewhelp:ch05:1"}})).values
        context = ToolContext.model_validate(snapshot["context"])
        kwargs = {"pool_signals": [PoolSignal.model_validate(item) for item in snapshot["pool_signals"]],
                  "retrieval_snapshot": RetrievalSnapshot.model_validate(snapshot["retrieval_snapshot"]),
                  "suggested_actions": [SuggestedAction.model_validate(item) for item in snapshot["suggested_actions"]],
                  "workflow_metadata": events[-2].data["workflow"]}
        assert await deps.chat.finish_turn(context, snapshot["answer"], **kwargs) == events[-2].data
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
            answers = (await session.scalars(select(Message).where(Message.role == "assistant"))).all()
        assert len(rows) == len(answers) == 1
        assert rows[0].source == "retrieval_low_conf" and rows[0].raw_question == context.user_message
        from uuid import NAMESPACE_URL, uuid5
        changed_action = SuggestedAction(action_id=str(uuid5(NAMESPACE_URL,
            f"mewhelp:ch05:{context.conversation_id}:{context.turn_id}:handoff")), kind="handoff",
            source_turn_id=context.turn_id, question=context.user_message, ticket_type=None)
        with pytest.raises(PersistenceUncertain):
            await deps.chat.finish_turn(context, snapshot["answer"], **{**kwargs, "suggested_actions": [changed_action]})
        with pytest.raises(PersistenceUncertain):
            await deps.chat.finish_turn(context, snapshot["answer"], **{**kwargs, "workflow_metadata": {"intent": "订单"}})
        with pytest.raises(PersistenceUncertain):
            await deps.chat.finish_turn(context, snapshot["answer"], **{**kwargs, "pool_signals": []})


async def test_complete_snapshot_retry_compares_body_citations_and_suggestions(tmp_path):
    """完成重试遗漏正文/完整引用/建议比对时失败。参数：tmp_path目录；返回：None；异常：断言失败。"""
    from app.knowledge.answering import EvidenceCitation
    from app.schemas.workflow import AgentDecision
    from tests.workflow.graph_fakes import ranked
    evidence = ranked()
    evidence = evidence.model_copy(update={"chunk": evidence.chunk.model_copy(update={"section_path": "退货/七天"})})
    async with service_runtime(tmp_path, intents=["商品咨询"], ranked=[evidence],
            decisions=[AgentDecision(kind="answer", suggested_actions=["handoff"])], answers=[["答复[1]"]]) as (service, deps, _):
        events = await collect(service, "政策")
        assert events[-1].event == "done"
        state = (await service.graph.aget_state({"configurable": {"thread_id": "mewhelp:ch05:1"}})).values
        context = ToolContext.model_validate(state["context"])
        citations = [EvidenceCitation.model_validate(item) for item in state["evidence"]]
        actions = [SuggestedAction.model_validate(item) for item in state["suggested_actions"]]
        kwargs = {"citations": citations, "suggested_actions": actions, "workflow_metadata": events[-2].data["workflow"],
                  "pool_signals": [], "retrieval_snapshot": RetrievalSnapshot.model_validate(state["retrieval_snapshot"])}
        assert await deps.chat.finish_turn(context, "答复[1]", **kwargs) == events[-2].data
        for content, override in [("另一个答复", {}), ("答复[1]", {"citations": []}),
                                  ("答复[1]", {"suggested_actions": []}),
                                  ("答复[1]", {"citations": [citations[0].model_copy(update={"section_path": None})]}),
                                  ("答复[1]", {"citations": [citations[0].model_copy(update={"answer": "被替换"})]})]:
            with pytest.raises(PersistenceUncertain):
                await deps.chat.finish_turn(context, content, **{**kwargs, **override})


@pytest.mark.parametrize("field", ["source_key", "source_revision"])
async def test_finish_retry_rejects_erasing_original_nullable_source_fields(tmp_path, field):
    """原始来源字段非空改None却被补全兼容规则忽略时失败。参数：tmp_path目录、field字段；返回：None；异常：断言失败。"""
    from app.knowledge.answering import EvidenceCitation
    async with service_runtime(tmp_path) as (_, deps, _):
        context = await deps.chat.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="政策"), "政策")
        citation = EvidenceCitation(n=1, chunk_id=1, question="条件", answer="完整原文", section_path="退货/七天",
                                    source_key="doc:policy", source_revision="a" * 64)
        saved = await deps.chat.finish_turn(context, "答复[1]", citations=[citation])
        assert await deps.chat.finish_turn(context, "答复[1]", citations=[citation]) == saved
        with pytest.raises(PersistenceUncertain):
            await deps.chat.finish_turn(context, "答复[1]", citations=[citation.model_copy(update={field: None})])


async def test_low_confidence_insert_failure_rolls_back_final_message_and_completed_status(tmp_path):
    """入池失败仍提交回答或completed时失败。参数：tmp_path目录；返回：None；异常：断言失败。"""
    from sqlalchemy import text
    async with service_runtime(tmp_path, intents=["商品咨询"]) as (service, deps, factory):
        async with factory.begin() as session:
            await session.execute(text("CREATE TRIGGER reject_low_confidence BEFORE INSERT ON low_confidence_questions BEGIN SELECT RAISE(ABORT, 'test failure'); END"))
        events = await collect(service, "原问")
        assert events[-1].event == "error" and all(event.event not in {"answer_metadata", "done"} for event in events)
        turns = await deps.chat.load_turns(1)
        assert turns[0].turn_status == "failed" and len(turns[0].messages) == 1


async def test_node_logs_keep_safe_summary_and_drop_internal_custom_payloads(tmp_path, caplog):
    """缺语义节点日志或内部custom事件混入正文时失败。参数：tmp_path/caplog；返回：None；异常：断言失败。"""
    from langgraph.config import get_stream_writer
    async with service_runtime(tmp_path, intents=["闲聊"]) as (service, deps, _):
        original = deps.model.classify

        async def classify(messages, ledger):
            """注入禁止公开的内部事件。参数：模型契约；返回：分类；异常：脚本错误。"""
            get_stream_writer()({"event": "intent", "data": {"intent": "SECRET_JSON"}})
            get_stream_writer()({"event": "answer_metadata", "data": {"secret": "SECRET_JSON"}})
            return await original(messages, ledger)

        deps.model.classify = classify
        with caplog.at_level("INFO"):
            events = await collect(service, "SECRET_RAW_原话")
        assert events[-1].event == "done" and "SECRET_JSON" not in str(events)
        assert "SECRET_RAW_原话" not in caplog.text
        for node in ["resolve_reference", "classify_intent", "chitchat_reply", "log_turn"]:
            assert f"node={node} phase=enter" in caplog.text and f"node={node} phase=exit" in caplog.text
        assert "budget=" in caplog.text and "elapsed_ms=" in caplog.text and "thread=mewhelp:ch05:1" in caplog.text


async def test_failed_classification_keeps_diagnostic_without_exposing_it_to_user(tmp_path):
    """分类最终失败时原始工具诊断必须落库，不能作为正文或可执行工具回灌。

    参数：tmp_path独占测试库目录；返回：None；异常：断言失败或资源错误。
    """
    from app.schemas.chat import InvalidToolDecision

    async with service_runtime(tmp_path) as (service, deps, factory):
        async def classify(messages, ledger):
            """模拟最终协议失败。参数：messages与ledger契约；返回：无；异常：InvalidToolDecision。"""
            raise InvalidToolDecision(raw_output="SECRET_BAD_CLASSIFICATION", validation_category="intent_structure")

        deps.model.classify = classify
        events = await collect(service, "他是什么物品？从哪里到哪里")
        assert events[-1].event == "error" and "SECRET_BAD_CLASSIFICATION" not in str(events)
        async with factory() as session:
            rows = (await session.scalars(select(Message).order_by(Message.id))).all()
        audits = [row for row in rows if row.tool_calls]
        assert len(audits) == 1
        assert audits[0].tool_calls[0]["terminal_decision"]["raw_output"] == "SECRET_BAD_CLASSIFICATION"
        assert all(row.turn_status == "failed" for row in rows)


async def test_recovered_classification_diagnostic_stays_out_of_operational_log(tmp_path, caplog):
    """补救后账本诊断可查，但节点安全日志不能输出模型原文。参数：tmp_path/caplog；返回：None；异常：断言失败。"""
    from app.schemas.chat import InvalidToolDecision
    from app.schemas.workflow import IntentResult, AgentDecision

    async with service_runtime(tmp_path, decisions=[AgentDecision(kind="answer")]) as (service, deps, _):
        async def classify(messages, ledger):
            """记录结构失败后成功分类。参数：messages与ledger契约；返回：闲聊分类；异常：预算错误。"""
            ledger.reserve("classify", 50, 128)
            ledger.record_failure("classify", InvalidToolDecision(raw_output="SECRET_CLASSIFICATION_RAW").audit_record)
            return IntentResult(intent="闲聊", confidence=0.9)

        deps.model.classify = classify
        with caplog.at_level("INFO"):
            events = await collect(service)
        assert events[-1].event == "done"
        state = (await service.graph.aget_state({"configurable": {"thread_id": "mewhelp:ch05:1"}})).values
        failure = next(record["failure"] for record in state["budget"]["records"] if "failure" in record)
        assert failure["terminal_decision"]["raw_output"] == "SECRET_CLASSIFICATION_RAW"
        assert "SECRET_CLASSIFICATION_RAW" not in caplog.text
        assert "SECRET_CLASSIFICATION_RAW" not in str(events)


async def test_diagnostic_storage_error_does_not_hide_the_safe_error_frame(tmp_path):
    """附加审计写失败不能中断原本给用户的安全错误提示。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    from app.schemas.chat import InvalidToolDecision

    async with service_runtime(tmp_path) as (service, deps, _):
        async def classify(messages, ledger):
            """产生协议失败。参数：模型契约；返回：无；异常：InvalidToolDecision。"""
            raise InvalidToolDecision(raw_output="SECRET_DIAGNOSTIC")

        async def save_tool_calls(context, calls):
            """模拟审计落盘失败。参数：审计契约；返回：无；异常：ConnectionError。"""
            raise ConnectionError("审计连接不可用")

        deps.model.classify = classify
        deps.chat.save_tool_calls = save_tool_calls
        events = await collect(service)
        assert events[-1].event == "error" and "SECRET_DIAGNOSTIC" not in str(events)
        assert (await deps.chat.load_turns(1))[0].turn_status == "failed"
