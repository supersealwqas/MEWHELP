"""真实客服根图的固定路由、知识门闸、建议快照与完成契约测试。"""
import json
import pytest
from app.schemas.workflow import AgentDecision
from app.schemas.tools import ToolResult
from app.workflow.graph import build_workflow
from tests.workflow.graph_fakes import NodeTrace, dependencies, turn, ranked, call

@pytest.mark.parametrize("intent,entry", [("商品咨询", "retrieve"), ("退款退货", "retrieve_policy")])
async def test_knowledge_nodes_precede_agent_and_keep_full_evidence(intent, entry):
    """删检索/闸、混淆退款入口或裁短证据时失败。参数：intent/entry路径；返回：None；异常：断言失败。"""
    evidence = [ranked(number=n, answer=f"第{n}块完整原文") for n in range(1, 31 if intent == "退款退货" else 11)]
    deps = dependencies(intents=[intent], ranked=evidence, decisions=[AgentDecision(kind="answer")], answers=[["政策[1]"]])
    trace = NodeTrace()
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("  它还能退吗？ "), {"callbacks": [trace]})
    assert trace.nodes.index("retrieve_knowledge") < trace.nodes.index("confidence_check") < trace.nodes.index("main_agent")
    assert deps.search.calls == [(entry, "  它还能退吗？ ")]
    assert len(state["evidence"]) == len(evidence)
    decision_input = deps.model.inputs[1]["messages"]
    assert all(item.chunk.text.answer in json.dumps(decision_input, ensure_ascii=False) for item in evidence)
    assert deps.chat.finished[0]["low_confidence"] is None
    assert state["completion_metadata"]["turn_id"] == state["context"]["turn_id"]
    json.dumps(state, allow_nan=False)

@pytest.mark.parametrize("intent", ["物流", "订单", "售后"])
async def test_business_routes_skip_retrieval_and_gate(intent):
    """业务被错误地预检索或门闸阻塞时失败。参数：intent业务意图；返回：None；异常：断言失败。"""
    deps = dependencies(intents=[intent], decisions=[AgentDecision(kind="clarify")], answers=[["请给订单号"]])
    trace = NodeTrace()
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn(), {"callbacks": [trace]})
    assert not {"retrieve_knowledge", "confidence_check"}.intersection(trace.nodes)
    assert state["answer"] == "请给订单号" and deps.search.calls == []

@pytest.mark.parametrize("scores", [[], [0.2]])
async def test_weak_evidence_finishes_without_main_model_and_records_only_low_confidence(scores):
    """低证据仍调用Agent或入池原因错误时失败。参数：scores分数；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["商品咨询"], ranked=[ranked(s) for s in scores])
    trace = NodeTrace()
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn(), {"callbacks": [trace]})
    assert "main_agent" not in trace.nodes and "log_turn" in trace.nodes
    assert state["stop_reason"] == "retrieval_low_conf"
    assert deps.chat.finished[0]["low_confidence"].source == "retrieval_low_conf"

@pytest.mark.parametrize("budget_env", ["TOKEN_BUDGET", "CHAT_TOKEN_BUDGET"])
async def test_overlong_knowledge_stops_without_retrieval_or_low_confidence(monkeypatch, budget_env):
    """宿主低预算不得干扰2001字知识输入的业务长度检查。

    参数：monkeypatch隔离环境变更；budget_env为当前或旧版预算环境变量名。
    返回：None，断言仍以input_limit结束且不检索、不执行Agent或入池。
    异常：AssertionError或ContextBudgetExceeded表示业务边界或测试隔离失效。
    """
    # 模拟第三方SDK导入.env后遗留的配置，不改变本例明确使用的离线预算。
    monkeypatch.delenv("TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("CHAT_TOKEN_BUDGET", raising=False)
    monkeypatch.setenv(budget_env, "31")
    deps = dependencies(intents=["退款退货"])
    trace = NodeTrace()
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("问" * 2001), {"callbacks": [trace]})
    assert state["stop_reason"] == "input_limit"
    assert not {"confidence_check", "main_agent"}.intersection(trace.nodes)
    assert deps.search.calls == [] and deps.chat.finished[0]["low_confidence"] is None

@pytest.mark.parametrize("error", [TimeoutError("慢"), ConnectionError("不可用")])
async def test_retrieval_failure_is_not_completed_as_low_confidence(error):
    """故障被伪装低证据并完成时失败。参数：error依赖故障；返回：None；异常：受控错误。"""
    deps = dependencies(intents=["商品咨询"])
    deps.search.error = error
    with pytest.raises(type(error)):
        await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert deps.chat.finished == []

@pytest.mark.parametrize("intent,want", [("投诉", ["handoff", "create_ticket"]), ("闲聊", [])])
async def test_fixed_exits_create_stable_suggestions_without_agent(intent, want):
    """投诉/闲聊调用主力或建议快照身份变化时失败。参数：intent/want；返回：None；异常：断言失败。"""
    deps = dependencies(intents=[intent, intent])
    graph = build_workflow(deps, checkpointer=None)
    original = turn("  服务很差  ", turn_id="t")
    first, second = await graph.ainvoke(original), await graph.ainvoke(original)
    assert [item["kind"] for item in first["suggested_actions"]] == want
    assert first["suggested_actions"] == second["suggested_actions"]
    assert all(x["question"] == "  服务很差  " and x["source_turn_id"] == "t" for x in first["suggested_actions"])
    assert all(x["ticket_type"] == ("投诉" if x["kind"] == "create_ticket" else None) for x in first["suggested_actions"])
    assert [i["stage"] for i in deps.model.inputs] == ["classify", "classify"]

async def test_refund_can_query_order_after_policy_gate():
    """退款知识路径禁止后续业务工具时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["退款退货"], ranked=[ranked()], results=[ToolResult.success({"status": "已发货"})],
        decisions=[AgentDecision(kind="tools", tool_calls=[call("1")]), AgentDecision(kind="answer")], answers=[["可以申请[1]"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert [c["name"] for c in deps.executor.calls] == ["query_order"]
    assert state["tool_count"] == 1 and state["gate"]["passed"]

@pytest.mark.parametrize("settings", [{"workflow_max_model_calls": 1}, {"workflow_total_token_budget": 1000}])
async def test_configured_budget_stops_before_extra_model_request(settings):
    """配置被默认State账本覆盖或预算外补救时失败。参数：settings低限额；返回：None；异常：断言失败。"""
    deps = dependencies(**settings)
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert state["stop_reason"] == "budget_exceeded"
    assert len(deps.model.inputs) == 1
    assert deps.chat.finished[0]["low_confidence"] is None

async def test_stored_history_is_visible_but_current_question_stays_unchanged():
    """丢掉已追问订单号的历史或改写原问时失败。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="clarify")], answers=[["正在查询"]])
    state = turn(" 1001 ")
    state["history_turns"] = [{"turn_id": "old", "turn_status": "completed", "messages": [
        {"role": "user", "content": "查物流"}, {"role": "assistant", "content": "请给订单号"}]}]
    result = await build_workflow(deps, checkpointer=None).ainvoke(state)
    assert deps.model.inputs[0]["messages"] == [{"role": "user", "content": "查物流"},
        {"role": "assistant", "content": "请给订单号"}, {"role": "user", "content": " 1001 "}]
    assert result["resolved_query"] == result["raw_query"] == " 1001 "

async def test_suggestion_snapshot_survives_json_roundtrip_and_rejects_malformed_payload():
    """JSON保存后无法重建或接受伪造字段/动作类型时失败。参数：无；返回：None；异常：断言失败。"""
    from app.schemas.workflow import SuggestedAction
    deps = dependencies(intents=["投诉"])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("原话" * 2000))
    for payload in json.loads(json.dumps(state["suggested_actions"])):
        assert SuggestedAction.model_validate(payload).model_dump(mode="json") == payload
        for override in ({"url": "https://example.invalid"}, {"kind": "pay"}, {"action_id": "bad"},
                         {"question": " "}, {"question": "长" * 4001}, {"ticket_type": 1}):
            with pytest.raises(ValueError):
                SuggestedAction.model_validate({**payload, **override})
        with pytest.raises(ValueError):
            SuggestedAction.model_validate({**payload, "ticket_type": "投诉" if payload["kind"] == "handoff" else None})
