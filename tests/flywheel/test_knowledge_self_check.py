"""CH09默认生产图的私有自评、真实事务前缀与共享生成边界测试。"""

import json

import pytest
from sqlalchemy import select

from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
from app.knowledge.answering import EvidenceCitation, FIXED_KNOWLEDGE_REFUSAL
from app.knowledge.pipeline import ProductionKnowledgePipeline
from app.models.rag_quality import LowConfidenceQuestion
from app.schemas.flywheel import PoolSignal
from app.schemas.tools import ToolResult
from app.schemas.workflow import AgentDecision
from app.workflow.model import WorkflowModelAdapter
from tests.workflow.graph_fakes import ranked, turn
from tests.workflow.service_fakes import collect, service_runtime
from tests.workflow.test_model import sdk_runtime, call_response


def response(*, useful=False, answer="不应展示的内部草稿", citations=()):
    """构造实际SDK流协议。参数：自评、草稿、引用；返回：SSE块；异常：序列化错误。"""
    return call_response("submit_knowledge_answer", json.dumps({"useful": useful,
        "reason": "缺少当前问题的必要依据" if not useful else "", "answer": answer,
        "citation_numbers": list(citations)}, ensure_ascii=False), usage=True)


def install_answer_adapter(deps, sdk_runtime):
    """仅替代模型HTTP，保留真实结构化适配器与默认图。参数：依赖/SDK夹具；返回：None；异常：无。"""
    adapter = WorkflowModelAdapter(model_factory=sdk_runtime.factory, stream_usage=True)
    deps.model.answer_knowledge = adapter.answer_knowledge


async def test_useful_false_draft_never_reaches_public_delta(tmp_path, sdk_runtime):
    """绕过自评或公开内部草稿会失败，完成事务须真实产生self_check。参数：目录/SDK；返回：None；异常：断言失败。"""
    sdk_runtime.responses.append(response())
    async with service_runtime(tmp_path, intents=["商品咨询"], ranked=[ranked()]) as (service, deps, factory):
        install_answer_adapter(deps, sdk_runtime)
        events = await collect(service, "这款猫粮有哪些限制？")
        answer = "".join(item.data["delta"] for item in events if item.event == "delta")
        assert answer == FIXED_KNOWLEDGE_REFUSAL
        assert "内部草稿" not in json.dumps([item.model_dump() for item in events], ensure_ascii=False)
        assert events[-1].event == "done"
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
            assert len(rows) == 1 and rows[0].source == "self_check"
            assert rows[0].retrieved_chunks["attempts"][0]["chunks"][0]["answer"] == "完整政策正文"


async def test_mixed_order_keeps_verified_facts_and_pools_policy_gap(tmp_path, sdk_runtime):
    """混合问题的低分政策不能吞掉订单或遗漏入池。参数：目录/SDK；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["退款退货"], ranked=[ranked(0.01)],
            results=[ToolResult(ok=True, data={"order_id": "1001", "status": "已签收"})]) as (service, deps, factory):
        events = await collect(service, "订单1001怎么退款？")
        answer = "".join(item.data["delta"] for item in events if item.event == "delta")
        assert "订单：1001" in answer and "状态：已签收" in answer
        assert answer.endswith(FIXED_KNOWLEDGE_REFUSAL) and events[-1].event == "done"
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
            assert len(rows) == 1 and rows[0].source == "retrieval_low_conf"


@pytest.mark.parametrize("source", ["retrieval_low_conf", "self_check"])
async def test_forged_business_prefix_cannot_bypass_pool_guard(tmp_path, source):
    """以endswith放宽拒答会允许伪造成功事实，此测试要求事务拒绝。参数：目录/来源；返回：None；异常：断言失败。"""
    from tests.flywheel.test_pool import start
    async with service_runtime(tmp_path) as (_, deps, factory):
        ctx = await start(deps.chat)
        with pytest.raises(ValueError):
            await deps.chat.finish_turn(ctx, "订单1001已经退款成功。" + FIXED_KNOWLEDGE_REFUSAL,
                pool_signals=[PoolSignal(source=source, reason="证据不足")])
        async with factory() as session:
            assert (await session.scalars(select(LowConfidenceQuestion))).all() == []


async def test_pure_business_reply_keeps_streaming(tmp_path):
    """普通业务也被缓冲或改为知识自评会失败。参数：目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["订单"],
            decisions=[AgentDecision(kind="clarify")], answers=[["请提供", "订单号。"]]) as (service, _, _):
        events = await collect(service, "查订单")
        assert [item.data["delta"] for item in events if item.event == "delta"] == ["请提供", "订单号。"]
        assert events[-1].event == "done"


async def test_missing_citation_uses_existing_repair_budget(tmp_path, sdk_runtime):
    """缺引用需一次有界修复且不能公开首稿或持久化未引用证据。参数：目录/SDK；返回：None；异常：断言失败。"""
    sdk_runtime.responses.extend([response(useful=True, answer="首稿没有引用", citations=[1]),
        response(useful=True, answer="政策要求保留包装。[1]", citations=[1, 2])])
    async with service_runtime(tmp_path, intents=["商品咨询"], ranked=[ranked(), ranked(number=2)]) as (service, deps, factory):
        install_answer_adapter(deps, sdk_runtime)
        events = await collect(service, "需要保留包装吗？")
        answer = "".join(item.data["delta"] for item in events if item.event == "delta")
        assert answer == "政策要求保留包装。[1]" and events[-1].event == "done"
        metadata = next(item.data for item in events if item.event == "answer_metadata")
        assert [item["n"] for item in metadata["citations"]] == [1]
        records = metadata["workflow"]["budget"]["records"]
        assert len([item for item in records if item["stage"] == "answer"]) == 2
        async with factory() as session:
            assert (await session.scalars(select(LowConfidenceQuestion))).all() == []


async def test_same_named_mcp_cannot_forge_business_prefix(tmp_path):
    """同名MCP绑定或旧轮正文不能通过新业务前缀核验。参数：目录；返回：None；异常：断言失败。"""
    from tests.flywheel.test_pool import start
    from app.tools.contracts import binding_dict, ToolBinding, ToolIdentity
    from app.workflow.policy_context import verified_business_prefix
    async with service_runtime(tmp_path) as (_, deps, _):
        ctx = await start(deps.chat)
        state = turn(ctx.user_message, turn_id=ctx.turn_id)
        state["context"] = ctx.model_dump(mode="json")
        state.update(intent="退款退货", route="order_policy", order_data={"order_id": "1001", "status": "假退款"})
        binding = ToolBinding(identity=ToolIdentity(source="mcp", server="fake", name="query_order"),
            schema_hash="a" * 64)
        call = {"id": "fake", "name": "query_order", "args": {"order_id": "1001"}}
        await deps.chat.save_tool_calls(ctx, [call], bindings={"fake": binding_dict(binding)})
        await deps.chat.save_tool_result(ctx.model_copy(update={"tool_call_id": "fake"}), ToolResult(ok=True, data=state["order_data"]))
        state["tool_call_facts"] = {"fake": {"turn_id": ctx.turn_id, "binding": binding_dict(binding), "access": "read"}}
        assert await verified_business_prefix(state, deps.chat) == ""


async def test_complete_schema_and_messages_reserve_before_network(sdk_runtime):
    """只数正文却遗漏工具schema会突破输入预算。参数：SDK；返回：None；异常：断言失败。"""
    model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    with pytest.raises(WorkflowBudgetExceeded):
        await model.answer_knowledge([{"role": "user", "content": "问题"}], BudgetLedger(request_token_budget=100))
    assert sdk_runtime.requests == []


async def test_generate_preserves_selected_and_has_no_chat_side_effects(tmp_path, sdk_runtime):
    """共享生成替换selected、漏用量或依赖聊天写入会失败。参数：目录/SDK；返回：None；异常：断言失败。"""
    from tests.workflow.graph_fakes import dependencies
    sdk_runtime.responses.append(response(useful=True, answer="完整说明。[7]", citations=[7]))
    deps = dependencies()
    state = turn("需要什么条件？")
    state.update(resolved_query=state["raw_query"], standard_query=state["raw_query"])
    selected = [EvidenceCitation(n=7, chunk_id=4, question="条件", answer="完整说明")]
    ledger = BudgetLedger.from_settings(deps.settings)
    pipeline = ProductionKnowledgePipeline(None, deps.settings, query_expander=None, state=state,
        model=WorkflowModelAdapter(model_factory=sdk_runtime.factory))
    result = await pipeline.generate({"ledger": ledger}, {"selected": selected, "gate": {"passed": True}})
    assert result["answer"] == "完整说明。[7]" and not result["refused"]
    assert result["citations"] == selected and result["usage"]["actual_tokens"] == 112
    assert "完整说明" in json.dumps(sdk_runtime.requests[0]["messages"], ensure_ascii=False)


async def test_retrieval_fault_does_not_become_knowledge_gap(tmp_path):
    """检索故障若入池则污染真实缺口。参数：目录；返回：None；异常：断言失败。"""
    async with service_runtime(tmp_path, intents=["商品咨询"]) as (service, deps, factory):
        deps.search.error = OSError("offline")
        events = await collect(service, "退款到账要多久？")
        assert events[-1].event == "done"
        async with factory() as session:
            assert (await session.scalars(select(LowConfidenceQuestion))).all() == []


@pytest.mark.parametrize("source", ["optional_miss", "optional_pass", "explicit_miss"])
async def test_business_probe_and_explicit_knowledge_have_distinct_responsibility(tmp_path, sdk_runtime, source):
    """可选未命中仍业务流，主动知识低分和可选命中自评拒绝均入池。参数：目录/SDK/路径；返回：None；异常：断言失败。"""
    from tests.workflow.ch08_ticket_fakes import ticket_runtime
    query = "查订单"
    explicit = source == "explicit_miss"
    decisions = ([AgentDecision(kind="tools", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": query}}])]
                 if explicit else []) + [AgentDecision(kind="answer")]
    async with ticket_runtime(tmp_path, intents=["订单"], decisions=decisions,
            ranked=[ranked()] if source == "optional_pass" else [], answers=[["请提供", "订单号。"]]) as (service, deps, factory, _):
        if source == "optional_pass":
            sdk_runtime.responses.append(response())
            install_answer_adapter(deps, sdk_runtime)
        events = await collect(service, query)
        deltas = [item.data["delta"] for item in events if item.event == "delta"]
        assert events[-1].event == "done"
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
        if source == "optional_miss":
            assert deltas == ["请提供", "订单号。"] and rows == []
        else:
            assert deltas == [FIXED_KNOWLEDGE_REFUSAL] and len(rows) == 1
            assert rows[0].source == ("retrieval_low_conf" if explicit else "self_check")


@pytest.mark.parametrize("mode", ["network", "structure", "citations"])
async def test_private_candidate_faults_do_not_publish_or_pool(tmp_path, sdk_runtime, mode):
    """网络、结构、引用故障不得变成useful=false，首稿不得提前公开。参数：目录/SDK/故障；返回：None；异常：断言失败。"""
    import httpx
    scripts = {"network": [httpx.ConnectError("private-upstream")],
        "structure": [call_response("submit_knowledge_answer", '{"useful":"false"}', usage=True)],
        "citations": [response(useful=True, answer="内部错误候选[99]", citations=[99])] * 2}
    sdk_runtime.responses.extend(scripts[mode])
    # SDK有一次既定网络重试，仍属于同一逻辑answer占额。
    if mode == "network":
        sdk_runtime.responses.append(httpx.ConnectError("private-upstream"))
    async with service_runtime(tmp_path, intents=["商品咨询"], ranked=[ranked()]) as (service, deps, factory):
        install_answer_adapter(deps, sdk_runtime)
        events = await collect(service, "需要什么政策条件？")
        text = "".join(item.data["delta"] for item in events if item.event == "delta")
        assert "内部错误候选" not in text and FIXED_KNOWLEDGE_REFUSAL not in text
        assert events[-1].event == "done"
        async with factory() as session:
            assert (await session.scalars(select(LowConfidenceQuestion))).all() == []


async def test_legacy_self_check_rejects_forged_prefix(tmp_path):
    """Legacy也不能用固定后缀夹带伪造业务事实。参数：目录；返回：None；异常：断言失败。"""
    from app.knowledge.answering import SelfCheckFailure
    from tests.flywheel.test_pool import start
    async with service_runtime(tmp_path) as (_, deps, _):
        context = await start(deps.chat)
        with pytest.raises(ValueError):
            await deps.chat.finish_turn(context, "伪造订单已退款。" + FIXED_KNOWLEDGE_REFUSAL,
                low_confidence=SelfCheckFailure(reason="缺少政策"))


async def test_completion_rechecks_current_order_result_and_rejects_old_turn(tmp_path):
    """前缀生成后结果改变或只有旧轮订单时，完成不能使用缓存前缀。参数：目录；返回：None；异常：断言失败。"""
    from app.models.customer_service import Message
    from app.tools.contracts import binding_dict
    from app.workflow.policy_context import verified_business_prefix
    from tests.flywheel.test_pool import start
    async with service_runtime(tmp_path) as (_, deps, factory):
        context = await start(deps.chat)
        definition = next(row for row in deps.registry.snapshot() if row.name == "query_order")
        binding = binding_dict(definition.binding)
        await deps.chat.save_tool_calls(context, [{"id": "order", "name": "query_order", "args": {"order_id": "1001"}}],
            bindings={"order": binding})
        await deps.chat.save_tool_result(context.model_copy(update={"tool_call_id": "order"}),
            ToolResult(ok=True, data={"order_id": "1001", "status": "已签收"}))
        state = {"context": context.model_dump(mode="json"), "intent": "退款退货",
            "tool_call_facts": {"order": {"turn_id": context.turn_id, "binding": binding, "access": "read"}}}
        prefix = await verified_business_prefix(state, deps.chat)
        assert "状态：已签收" in prefix
        async with factory.begin() as session:
            result = await session.scalar(select(Message).where(Message.role == "tool"))
            result.content = ToolResult(ok=True, data={"order_id": "1001", "status": "待发货"}).model_dump_json()
        with pytest.raises(ValueError):
            await deps.chat.finish_turn(context, prefix + FIXED_KNOWLEDGE_REFUSAL,
                workflow_metadata={"intent": "退款退货"}, pool_signals=[PoolSignal(source="self_check", reason="不足")])
        new_context = await start(deps.chat, turn="next")
        assert await verified_business_prefix({**state, "context": new_context.model_dump(mode="json")}, deps.chat) == ""


async def test_generate_rejects_selected_shrink_before_model(sdk_runtime):
    """预算缩小后不能悄悄删证据而沿用原闸。参数：SDK；返回：None；异常：断言失败。"""
    from app.core.memory import ContextBudgetExceeded
    from tests.workflow.graph_fakes import dependencies
    deps = dependencies()
    state = turn("问题")
    selected = [EvidenceCitation(n=1, chunk_id=1, question="条件", answer="规则" * 10000)]
    pipeline = ProductionKnowledgePipeline(None, deps.settings, query_expander=None, state=state,
        model=WorkflowModelAdapter(model_factory=sdk_runtime.factory))
    with pytest.raises(ContextBudgetExceeded):
        await pipeline.generate({"ledger": BudgetLedger.from_settings(deps.settings)}, {"selected": selected, "gate": {"passed": True}})
    assert sdk_runtime.requests == []


async def test_retrieval_gap_survives_terminal_reason_change(tmp_path):
    """检索缺口后停止于重复调用仍须固定政策兜底并原子入池。参数：目录；返回：None；异常：断言失败。"""
    from tests.workflow.ch08_ticket_fakes import ticket_runtime
    calls = [{"id": name, "name": "query_faq", "args": {"keyword": "会员条件"}} for name in ("one", "two")]
    async with ticket_runtime(tmp_path, intents=["其他"], decisions=[
            AgentDecision(kind="tools", tool_calls=[call]) for call in calls]) as (service, _, factory, _):
        events = await collect(service, "会员条件是什么？")
        assert events[-1].event == "done"
        assert "".join(item.data["delta"] for item in events if item.event == "delta") == FIXED_KNOWLEDGE_REFUSAL
        metadata = next(item.data for item in events if item.event == "answer_metadata")
        assert metadata["workflow"]["stop_reason"] == "duplicate_tool_call"
        async with factory() as session:
            rows = (await session.scalars(select(LowConfidenceQuestion))).all()
            assert len(rows) == 1 and rows[0].source == "retrieval_low_conf"


async def test_failed_second_knowledge_query_cannot_become_self_check(tmp_path, sdk_runtime):
    """一条成功证据不能掩盖本轮另一政策查询故障并污染自评池。参数：目录/SDK；返回：None；异常：断言失败。"""
    from tests.workflow.ch08_ticket_fakes import ticket_runtime
    sdk_runtime.responses.append(response())
    calls = [{"id": str(i), "name": "query_faq", "args": {"keyword": query}}
             for i, query in enumerate(("会员条件", "退款时限"))]
    async with ticket_runtime(tmp_path, intents=["其他"], ranked=[ranked()], decisions=[
            AgentDecision(kind="tools", tool_calls=calls), AgentDecision(kind="answer")]) as (service, deps, factory, _):
        original = deps.search.retrieve
        async def retrieve(query, **kwargs):
            """第二条实际查询故障仍经原快照捕获。参数：原检索输入；返回：排名；异常：OSError。"""
            if query == "退款时限":
                deps.search.error = OSError("检索网络失败")
            return await original(query, **kwargs)
        deps.search.retrieve = retrieve
        install_answer_adapter(deps, sdk_runtime)
        events = await collect(service, "会员能退货吗，退款时限多久？")
        assert events[-1].event == "done" and sdk_runtime.requests == []
        async with factory() as session:
            assert (await session.scalars(select(LowConfidenceQuestion))).all() == []


@pytest.mark.parametrize("retry_kind", ["same", "different_query", "different_filters"])
async def test_retrieval_recovery_only_clears_same_request_responsibility(tmp_path, sdk_runtime, retry_kind):
    """同查询同范围成功可恢复；其他查询或同字不同范围不能掩盖故障。参数：目录/SDK/重试类型；返回：None；异常：断言失败。"""
    from app.tools.knowledge import retrieve_verified_knowledge
    from app.workflow.agent import build_agent_graph
    from app.workflow.policy_context import knowledge_retrieval_fault
    from tests.flywheel.test_pool import start
    async with service_runtime(tmp_path, ranked=[ranked()]) as (_, deps, factory):
        context = await start(deps.chat)
        state = turn(context.user_message, turn_id=context.turn_id)
        state.update(context=context.model_dump(mode="json"), resolved_query=context.user_message,
            standard_query=context.user_message, intent="商品咨询", route="knowledge", route_ready=True, stop_reason="answer")
        ledger = BudgetLedger.from_settings(deps.settings)
        deps.search.error = OSError("真实检索故障")
        with pytest.raises(OSError):
            await retrieve_verified_knowledge(state, ledger, search=deps.search, settings=deps.settings,
                query_expander=deps.query_expander, query="会员规则", filters={"category": "规则"}, chat=deps.chat)
        deps.search.error = None
        outcome = await retrieve_verified_knowledge(state, ledger, search=deps.search, settings=deps.settings,
            query_expander=deps.query_expander, query="其他规则" if retry_kind == "different_query" else "会员规则",
            filters={"category": "其他" if retry_kind == "different_filters" else "规则"}, chat=deps.chat)
        state.update(outcome.model_dump(mode="json"), budget=ledger.snapshot())
        assert [item["status"] for item in state["retrieval_snapshot"]["attempts"]] == ["fault", "ok"]
        assert knowledge_retrieval_fault(state) is (retry_kind != "same")
        if retry_kind == "same":
            sdk_runtime.responses.append(response())
            install_answer_adapter(deps, sdk_runtime)
            result = await build_agent_graph(deps).ainvoke(state)
            assert result["answer"] == FIXED_KNOWLEDGE_REFUSAL
            assert result["pool_signals"][0]["source"] == "self_check"
            await deps.chat.finish_turn(context, result["answer"], pool_signals=[PoolSignal.model_validate(item)
                for item in result["pool_signals"]], retrieval_snapshot=outcome.retrieval_snapshot)
            async with factory() as session:
                assert len((await session.scalars(select(LowConfidenceQuestion))).all()) == 1


async def test_old_checkpoint_recovers_fault_before_tool_subgraph(tmp_path):
    """旧子图缺快照字段时须先回读故障，不能先建空字典让恢复误认为已清除。参数：目录；返回：None；异常：断言失败。"""
    from app.schemas.flywheel import RetrievalAttempt, RetrievalSnapshot
    from app.tools.contracts import binding_dict
    from app.workflow.agent import build_agent_graph
    from tests.flywheel.test_pool import start
    async with service_runtime(tmp_path, decisions=[AgentDecision(kind="answer")],
            results=[ToolResult(ok=True, data={"order_id": "1001", "status": "已签收"})]) as (_, deps, _):
        context = await start(deps.chat)
        await deps.chat.save_retrieval_snapshot(context, RetrievalSnapshot(status="fault", attempts=[
            RetrievalAttempt(attempt_id="old-fault", query="政策", status="fault", top_k=5, fault="OSError")]))
        state = turn(context.user_message, turn_id=context.turn_id)
        state.update(context=context.model_dump(mode="json"), route="business", intent="订单")
        definition = next(item for item in deps.registry.snapshot() if item.name == "query_order")
        call = {"id": "order", "name": "query_order", "args": {"order_id": "1001"}}
        binding = binding_dict(definition.binding)
        state.update(pending_calls=[call], pending_bindings={"order": binding}, pending_tool_bindings={"order": binding},
            tool_call_facts={"order": {"turn_id": context.turn_id, "binding": binding, "access": "read"}})
        for key in ("knowledge_faults", "pool_signals", "retrieval_snapshot"):
            state.pop(key)
        result = await build_agent_graph(deps).ainvoke(state)
        assert result["stop_reason"] == "retrieval_unavailable" and result["knowledge_faults"]
        assert result["pool_signals"] == []


async def test_parent_callbacks_keep_private_raw_output_outside_public_stream(tmp_path, sdk_runtime):
    """父回调可追踪内部模型流，但公共custom流不能包含未通过的草稿。参数：目录/SDK；返回：None；异常：断言失败。"""
    from langchain_core.callbacks import BaseCallbackHandler
    from app.workflow.graph import build_workflow
    from app.workflow.state import new_turn_state
    from tests.flywheel.test_pool import start
    observed = []
    class PrivateObserver(BaseCallbackHandler):
        """仅记录测试中的内部模型事件，不成为用户输出消费者。"""
        def on_llm_new_token(self, token, **kwargs):
            """捕获内部原片。参数：框架token/chunk；返回：None；异常：无。"""
            if kwargs.get("chunk") is not None:
                observed.append(kwargs["chunk"].message.model_dump(mode="json"))
    sdk_runtime.responses.append(response())
    async with service_runtime(tmp_path, intents=["商品咨询"], ranked=[ranked()]) as (_, deps, _):
        context = await start(deps.chat, question="退款到账多久？")
        install_answer_adapter(deps, sdk_runtime)
        public = [item[1] async for item in build_workflow(deps, checkpointer=None).astream(
            new_turn_state(context, [], 0), {"callbacks": [PrivateObserver()]}, stream_mode="custom", subgraphs=True)]
        assert "内部草稿" in json.dumps(observed, ensure_ascii=False)
        assert "内部草稿" not in json.dumps(public, ensure_ascii=False)
        assert "".join(item["data"]["delta"] for item in public if item["event"] == "delta") == FIXED_KNOWLEDGE_REFUSAL
