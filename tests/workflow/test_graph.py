"""真实客服根图的固定路由、知识门闸、建议快照与完成契约测试。"""
import json
import pytest
from app.core.model_budget import WorkflowBudgetExceeded
from app.schemas.workflow import AgentDecision, UnderstandingResult
from app.schemas.tools import ToolResult
from app.workflow.graph import build_workflow
from tests.workflow.graph_fakes import NodeTrace, dependencies, turn, ranked, call


@pytest.mark.parametrize("question", ["支持哪些付款方式？", "商品的清洁注意事项有哪些？", "设备耗材的兼容规则在哪里？"])
@pytest.mark.parametrize("decision", ["clarify", "answer"])
async def test_misclassified_clarification_can_recover_with_knowledge(question, decision):
    """业务分类不能封死无具体对象的知识问题。参数：question不同领域问题、decision拟澄清或回答；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["订单"], decisions=[AgentDecision(kind=decision)],
                        ranked=[ranked(answer="本轮可用说明")], answers=[["依据资料回答[1]。"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn(question))
    assert deps.search.calls == [("retrieve", question)]
    assert state["retrieval_attempted"] and state["evidence"] and state["stop_reason"] == "answer"
    assert "本轮可用说明" in json.dumps(deps.model.inputs[-1]["messages"], ensure_ascii=False)
    assert len([row for row in deps.model.inputs if row["stage"] == "decide"]) == 1
    assert deps.executor.calls == []


@pytest.mark.parametrize("stage", ["classify", "decide", "answer", "retrieve", "empty_answer"])
async def test_dependency_failure_has_a_persisted_reply_instead_of_crashing(stage):
    """外部依赖异常或空正文不能留下空回答，异常原文也不得泄露。参数：stage故障阶段；返回：None；异常：断言失败。"""
    from app.knowledge.search import KnowledgeSearchUnavailable
    error = ConnectionError("private-upstream-token")
    deps = dependencies(intents=[error if stage == "classify" else "商品咨询" if stage == "retrieve" else "订单"],
                        decisions=[error if stage == "decide" else AgentDecision(kind="answer")],
                        answers=[[error]] if stage == "answer" else [[]] if stage == "empty_answer" else [["正常正文"]])
    if stage == "retrieve":
        deps.search.error = KnowledgeSearchUnavailable("private-upstream-token")
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert state["answer"].strip() and "private-upstream-token" not in state["answer"]
    assert len(deps.chat.finished) == 1 and deps.chat.finished[0]["content"] == state["answer"]
    assert state["degradations"] and state["completion_metadata"]


async def test_partial_answer_failure_keeps_stream_and_persistence_equal():
    """流式输出中断后不能丢失已发文字或让数据库只保存错误提示。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(decisions=[AgentDecision(kind="answer")], answers=[["已查到部分信息。", ConnectionError("私密端点")]])
    fragments, final = [], None
    async for namespace, mode, payload in build_workflow(deps, checkpointer=None).astream(
            turn(), stream_mode=["custom", "values"], subgraphs=True):
        if mode == "values" and not namespace:
            final = payload
        elif mode == "custom" and payload["event"] == "delta":
            fragments.append(payload["data"]["delta"])
    assert final["answer"].startswith("已查到部分信息。")
    assert final["answer"] == "".join(fragments) == deps.chat.finished[0]["content"]
    assert final["stop_reason"] == "answer_unavailable" and "私密端点" not in final["answer"]


async def test_policy_retrieval_failure_can_still_answer_verified_order():
    """检索服务不可用时不能丢失已核实订单或将故障伪装低分入池。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.search import KnowledgeSearchUnavailable
    deps = dependencies(intents=["退款退货"], results=[ToolResult.success({"order_id": "1001", "status": "已发货"})],
                        answers=[["暂时无法查询政策，可稍后重试。"]])
    deps.search.error = KnowledgeSearchUnavailable("服务中断")
    initial = turn("订单1001的售后怎么处理")
    initial["context"]["user_id"] = "demo-user"
    state = await build_workflow(deps, checkpointer=None).ainvoke(initial)
    assert state["order_id"] == "1001" and "已发货" in state["answer"]
    assert not state["evidence"] and state["degradations"]
    assert deps.chat.finished[0]["low_confidence"] is None


@pytest.mark.parametrize("stage", ["understand", "recover_knowledge", "answer"])
async def test_other_dependency_boundaries_keep_safe_completion(stage):
    """补全失败保留原话，补救检索失败或纯空白回答也可靠结束。参数：stage边界；返回：None；异常：断言失败。"""
    deps = dependencies(understandings=[ConnectionError("敏感信息")] if stage == "understand" else [],
                        decisions=[AgentDecision(kind="clarify" if stage == "recover_knowledge" else "answer")],
                        answers=[[" \n "]] if stage == "answer" else [["有效回复"]])
    if stage == "recover_knowledge":
        deps.search.error = TimeoutError("敏感信息")
    query = "请说明商品的清洁方法"
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn(query))
    assert state["raw_query"] == state["resolved_query"] == query
    assert state["answer"].strip() and "敏感信息" not in json.dumps(state, ensure_ascii=False)
    assert state["degradations"][0]["stage"] == stage
    assert deps.chat.finished[0]["content"] == state["answer"]


@pytest.mark.parametrize("error", [ConnectionError("断流"), WorkflowBudgetExceeded("额度")])
async def test_policy_answer_interruption_keeps_verified_prefix(error):
    """政策说明中断仍保存程序事实及已发出的说明，预算耗尽同样不能丢字。参数：error故障；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["售后"], results=[ToolResult.success({"order_id": "1001", "total_amount": 89})],
                        answers=[["先说明已查到的部分。", error]])
    initial = turn("订单1001能否维修")
    initial["context"]["user_id"] = "demo-user"
    pieces = []
    async for _, event in build_workflow(deps, checkpointer=None).astream(initial, stream_mode="custom", subgraphs=True):
        if event["event"] == "delta":
            pieces.append(event["data"]["delta"])
    assert "商品总金额：89" in "".join(pieces) and "先说明已查到的部分。" in "".join(pieces)
    assert deps.chat.finished[0]["content"] == "".join(pieces)


@pytest.mark.parametrize("query,previous,reply", [
    ("他是什么物品？从哪里到哪里", "1001包邮吗", "1001是猫粮，从杭州发往宁波。"),
    ("？", "退换货政策是怎样的", "您是想了解退货条件、运费，还是申请流程？"),
])
async def test_contextual_followup_without_policy_need_reaches_agent(query, previous, reply):
    """已有事实或澄清被裸问检索的低分门闸拦住时失败。

    参数：query当前追问；previous上轮原问；reply回答脚本。返回：None。
    异常：AssertionError，发生无关检索、历史丢失或未生成正文。
    """
    from types import SimpleNamespace

    classification = "订单" if "物品" in query else "其他"
    deps = dependencies(intents=[classification], decisions=[AgentDecision(kind="answer")], answers=[[reply]])
    original = turn(query)
    original["history_turns"] = [{"turn_id": "previous", "turn_status": "completed", "messages": [
        {"role": "user", "content": previous}, {"role": "assistant", "content": "1001是猫粮，从杭州发往宁波。"}]}]
    result = await build_workflow(deps, checkpointer=None).ainvoke(original)
    assert result["answer"] == reply and result["stop_reason"] == "answer"
    assert result["route"] == "business" and not deps.search.calls
    assert previous in json.dumps(deps.model.inputs[-1]["messages"], ensure_ascii=False)
    assert result["raw_query"] == query


async def test_policy_followup_retrieves_resolved_question_without_replacing_user_words():
    """退货运费追问仍只检索无对象短句，或改写覆盖原话时失败。参数：无；返回：None；异常：断言失败。"""
    from types import SimpleNamespace

    resolved = "七天无理由退货的运费由谁承担？"
    understanding = UnderstandingResult(resolved_query=resolved, standard_query=resolved, needs_resolution=True, order_id=None)
    deps = dependencies(intents=["商品咨询"], understandings=[understanding], ranked=[ranked()],
        decisions=[AgentDecision(kind="answer")], answers=[["退回运费由买家承担[1]。"]])
    original = turn("那运费呢？")
    original["history_turns"] = [{"turn_id": "previous", "turn_status": "completed", "messages": [
        {"role": "user", "content": "七天无理由退货有什么要求？"}, {"role": "assistant", "content": "商品需完好。"}]}]
    result = await build_workflow(deps, checkpointer=None).ainvoke(original)
    assert deps.search.calls == [("retrieve", resolved)]
    assert result["raw_query"] == result["context"]["user_message"] == "那运费呢？"
    assert result["messages"][0] == {"role": "user", "content": "那运费呢？"}
    assert result["gate"]["passed"]


@pytest.mark.parametrize("standard_query", ["问" * 2001, "退款政策"])
async def test_classification_cannot_shorten_away_the_original_knowledge_input_limit(standard_query):
    """分类输出短问或空值不能让超长知识原问绕过输入上限。参数：knowledge_query模型输出；返回：None；异常：断言失败。"""
    from types import SimpleNamespace

    deps = dependencies(intents=["退款退货"], understandings=[UnderstandingResult(resolved_query="问" * 2001,
        standard_query=standard_query, needs_resolution=False, order_id=None)])
    result = await build_workflow(deps, checkpointer=None).ainvoke(turn("问" * 2001))
    assert result["stop_reason"] == "input_limit" and deps.search.calls == []


async def test_order_with_shipping_policy_need_still_retrieves_knowledge():
    """订单标签不能让金额和邮费混合问题漏掉政策半问。参数：无；返回：None；异常：断言失败。"""
    from types import SimpleNamespace

    classification = "商品咨询"
    deps = dependencies(intents=[classification], ranked=[ranked()],
        decisions=[AgentDecision(kind="answer")], answers=[["商品89元，实际运费未知；若实付未满99元且无减免，基础运费6元[1]。"]])
    result = await build_workflow(deps, checkpointer=None).ainvoke(turn("多少钱，需要付邮费吗"))
    assert result["route"] == "knowledge" and result["gate"]["passed"]
    assert deps.search.calls == [("retrieve", "多少钱，需要付邮费吗")]

@pytest.mark.parametrize("intent,entry", [("商品咨询", "retrieve"), ("退款退货", "retrieve_policy")])
async def test_knowledge_nodes_precede_agent_and_keep_full_evidence(intent, entry):
    """删检索/闸、混淆退款入口或裁短证据时失败。参数：intent/entry路径；返回：None；异常：断言失败。"""
    evidence = [ranked(number=n, answer=f"第{n}块完整原文") for n in range(1, 31 if intent == "退款退货" else 11)]
    deps = dependencies(intents=[intent], ranked=evidence, results=[ToolResult.success({"order_id": "1001", "status": "已签收"})],
        decisions=[AgentDecision(kind="answer")], answers=[["政策[1]"]])
    trace = NodeTrace()
    original = turn("订单1001还能退吗？")
    original["context"]["user_id"] = "demo-user"
    state = await build_workflow(deps, checkpointer=None).ainvoke(original, {"callbacks": [trace]})
    retrieval_node = "retrieve_policy" if intent == "退款退货" else "retrieve_knowledge"
    assert trace.nodes.index(retrieval_node) < trace.nodes.index("confidence_check") < trace.nodes.index("main_agent")
    assert len(deps.search.calls) == 1 and deps.search.calls[0][0] == entry
    assert original["raw_query"] in deps.search.calls[0][1]
    if intent == "退款退货":
        assert "当前已确认对象：订单1001" in deps.search.calls[0][1] and "已签收" in deps.search.calls[0][1]
        assert len(state["expanded_queries"]) == 3
    assert len(state["evidence"]) == len(evidence)
    decision_input = deps.model.inputs[-1]["messages"]
    assert all(item.chunk.text.answer in json.dumps(decision_input, ensure_ascii=False) for item in evidence)
    assert deps.chat.finished[0]["low_confidence"] is None
    assert state["completion_metadata"]["turn_id"] == state["context"]["turn_id"]
    json.dumps(state, allow_nan=False)

@pytest.mark.parametrize("intent", ["物流", "订单", "其他"])
async def test_business_routes_skip_retrieval_and_gate(intent):
    """业务被错误地预检索或门闸阻塞时失败。参数：intent业务意图；返回：None；异常：断言失败。"""
    deps = dependencies(intents=[intent], decisions=[AgentDecision(kind="clarify")], answers=[["请给订单号"]])
    trace = NodeTrace()
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn(), {"callbacks": [trace]})
    assert not {"retrieve_knowledge", "confidence_check"}.intersection(trace.nodes)
    assert state["answer"] == "请给订单号" and deps.search.calls == []

@pytest.mark.parametrize("intent", ["商品咨询"])
@pytest.mark.parametrize("scores", [[], [0.2]])
async def test_weak_evidence_finishes_without_main_model_and_records_only_low_confidence(intent, scores):
    """纯知识低证据仍调用Agent或入池原因错误时失败。参数：intent知识意图；scores分数；返回：None；异常：断言失败。"""
    deps = dependencies(intents=[intent], ranked=[ranked(s) for s in scores])
    trace = NodeTrace()
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn(), {"callbacks": [trace]})
    assert "main_agent" not in trace.nodes and "log_turn" in trace.nodes
    assert state["stop_reason"] == "retrieval_low_conf"
    assert state["evidence"] == [] and not deps.executor.calls
    assert [item["stage"] for item in deps.model.inputs] == ["understand", "classify"]
    assert deps.chat.finished[0]["citations"] == []
    assert deps.chat.finished[0]["low_confidence"].source == "retrieval_low_conf"


@pytest.mark.parametrize("intent", ["订单", "物流"])
@pytest.mark.parametrize("scores", [[], [0.04]])
async def test_weak_policy_evidence_does_not_block_readonly_business_queries(intent, scores):
    """知识需求误判把可以核实的业务问题提前拒答时失败。

    参数：intent业务意图；scores无命中或低相关分数。返回：None。
    异常：AssertionError，业务未查询、低质政策泄漏到模型或引用、失败审计丢失。
    """
    from types import SimpleNamespace

    query = "1001订单是什么，从哪里到哪里，是什么东西？现在运到哪里了？什么快递运的"
    classification = intent
    deps = dependencies(intents=[classification],
        ranked=[ranked(score, answer="不应被采用的政策原文") for score in scores],
        results=[ToolResult.success({"order_id": "1001", "product": "猫粮", "tracking_no": "SF123",
                                    "shipping_origin": "浙江省杭州市", "shipping_destination": "浙江省宁波市"}),
                 ToolResult.success({"carrier": "顺丰速运", "latest_node": "杭州转运中心"})],
        decisions=[AgentDecision(kind="tools", tool_calls=[call("order", order_id="1001")]),
                   AgentDecision(kind="tools", tool_calls=[call("logistics", "query_logistics", tracking_no="SF123")]),
                   AgentDecision(kind="answer")], answers=[["1001是猫粮，由顺丰承运，已到杭州转运中心。"]])
    result = await build_workflow(deps, checkpointer=None).ainvoke(turn(query))
    assert [item["name"] for item in deps.executor.calls] == ["query_order", "query_logistics"]
    assert result["tool_count"] == 2 and result["stop_reason"] == "answer"
    assert result["gate"] is None and result["evidence"] == []
    assert result["evidence_scores"] == [] and not deps.search.calls
    inputs = json.dumps([item["messages"] for item in deps.model.inputs[1:]], ensure_ascii=False)
    assert "不应被采用的政策原文" not in inputs
    assert "本轮完整知识证据" not in inputs
    assert "1001" in result["answer"] and "杭州转运中心" in result["answer"]
    assert deps.chat.finished[0]["citations"] == []
    # 业务意图低分后继续查事实并正常作答，不属于弱证据拒答，不能作为低证据问题入池；
    # 否则仓储会要求正文等于固定拒答话术，导致有效回答无法落库。
    assert deps.chat.finished[0]["low_confidence"] is None


async def test_weak_policy_evidence_keeps_business_history_for_followups():
    """低分恢复丢掉可信历史或将旧政策当作本轮证据时失败。参数：无；返回：None；异常：断言失败。"""
    from types import SimpleNamespace

    deps = dependencies(intents=["订单"],
        ranked=[ranked(0.01, answer="低分无关政策")], decisions=[AgentDecision(kind="answer")],
        answers=[["1001的猫粮从杭州发往宁波。"]])
    original = turn("是什么东西，哪里到哪里")
    original["history_turns"] = [{"turn_id": "previous", "turn_status": "completed", "messages": [
        {"role": "user", "content": "1001订单到哪了"},
        {"role": "assistant", "content": "", "tool_calls": [call("known", order_id="1001")]},
        {"role": "tool", "tool_call_id": "known", "content": ToolResult.success(
            {"order_id": "1001", "product": "猫粮", "shipping_origin": "杭州", "shipping_destination": "宁波"}).model_dump_json()},
        {"role": "assistant", "content": "1001已发货。"}]}]
    result = await build_workflow(deps, checkpointer=None).ainvoke(original)
    assert result["stop_reason"] == "answer"
    decision_input = json.dumps(deps.model.inputs[-1]["messages"], ensure_ascii=False)
    assert "猫粮" in decision_input and "宁波" in decision_input
    assert "低分无关政策" not in decision_input
    assert result["answer"] == "1001的猫粮从杭州发往宁波。"
    assert not deps.executor.calls and result["evidence"] == []


async def test_policy_gate_state_does_not_leak_across_high_low_high_checkpoint_turns():
    """同一检查点连续高分、低分业务恢复、高分时，证据或恢复状态串轮即失败。

    参数：无；三个新轮共用内存检查点，并按真实存储形态回灌业务历史。
    返回：None；异常：AssertionError，旧证据或引用权限残留、可信事实丢失或恢复提示污染新轮。
    """
    from types import SimpleNamespace
    from langgraph.checkpoint.memory import InMemorySaver

    deps = dependencies(
        intents=["商品咨询", "订单", "商品咨询"],
        ranked=[ranked(answer="第一轮仅在证据中出现的规则")],
        results=[ToolResult.success({"order_id": "1001", "product": "猫粮",
                                    "shipping_origin": "杭州", "shipping_destination": "宁波"})],
        decisions=[AgentDecision(kind="tools", tool_calls=[call("known", order_id="1001")]),
                   AgentDecision(kind="answer"), AgentDecision(kind="answer"), AgentDecision(kind="answer")],
        answers=[["第一轮政策已核实[1]。"], ["1001的猫粮从杭州发往宁波[1]。"], ["第三轮政策[1]。"]])
    graph = build_workflow(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "policy-gate-three-turns"}}
    first = await graph.ainvoke(turn("1001包邮吗", turn_id="first"), config)
    assert first["gate"]["passed"] and "[1]" in first["answer"]
    assert deps.chat.finished[0]["low_confidence"] is None

    # 仓储只保存用户、助手和工具消息；本轮注入的system证据不能冒充持久化历史。
    first_history = {"turn_id": "first", "turn_status": "completed", "messages": [
        *[message for message in first["messages"] if message["role"] != "system"],
        {"role": "assistant", "content": first["answer"]}]}
    deps.search.ranked = [ranked(0.04, answer="第二轮不得使用的低分规则")]
    second_input = turn("是什么东西，哪里到哪里", turn_id="second")
    second_input["history_turns"] = [first_history]
    second = await graph.ainvoke(second_input, config)
    assert second["gate"] is None and second["evidence"] == []
    assert second["evidence_scores"] == [] and second["stop_reason"] == "answer"
    assert second["answer"] == "1001的猫粮从杭州发往宁波。"
    assert len(deps.executor.calls) == 1
    second_model_input = json.dumps(deps.model.inputs[-1]["messages"], ensure_ascii=False)
    assert "猫粮" in second_model_input and "宁波" in second_model_input
    assert "第一轮仅在证据中出现的规则" not in second_model_input
    assert "第二轮不得使用的低分规则" not in second_model_input
    assert "本轮完整知识证据" not in second_model_input
    assert deps.chat.finished[1]["citations"] == []
    assert deps.chat.finished[1]["low_confidence"] is None

    second_history = {"turn_id": "second", "turn_status": "completed", "messages": [
        *[message for message in second["messages"] if message["role"] != "system"],
        {"role": "assistant", "content": second["answer"]}]}
    deps.search.ranked = [ranked(answer="第三轮独立核实的规则")]
    third_input = turn("退换货政策", turn_id="third")
    third_input["history_turns"] = [first_history, second_history]
    third = await graph.ainvoke(third_input, config)
    assert third["gate"]["passed"] and third["stop_reason"] == "answer"
    assert third["answer"] == "第三轮政策[1]。"
    third_model_input = json.dumps(deps.model.inputs[-1]["messages"], ensure_ascii=False)
    assert "第三轮独立核实的规则" in third_model_input
    assert "本轮没有可用知识政策证据" not in third_model_input
    assert "第二轮不得使用的低分规则" not in third_model_input
    assert [item.answer for item in deps.chat.finished[2]["citations"]] == ["第三轮独立核实的规则"]
    assert deps.chat.finished[2]["low_confidence"] is None


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
    """依赖故障应有明确回复但不能当低证据入池。参数：error依赖故障；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["商品咨询"])
    deps.search.error = error
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn())
    assert state["stop_reason"] == "retrieval_unavailable" and state["answer"]
    assert deps.chat.finished[0]["low_confidence"] is None and not state["evidence"]

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
    assert [i["stage"] for i in deps.model.inputs] == ["understand", "classify", "understand", "classify"]

async def test_refund_queries_order_before_policy_gate():
    """退款必须在政策前确定性查询订单。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["退款退货"], ranked=[ranked()], results=[ToolResult.success({"order_id": "1001", "status": "已发货"})],
        decisions=[AgentDecision(kind="tools", tool_calls=[call("1")]), AgentDecision(kind="answer")], answers=[["可以申请[1]"]])
    original = turn()
    original["context"]["user_id"] = "demo-user"
    state = await build_workflow(deps, checkpointer=None).ainvoke(original)
    assert [c["name"] for c in deps.executor.calls] == ["query_order"]
    assert state["tool_count"] == 1 and state["gate"]["passed"]

@pytest.mark.parametrize("settings", [{"workflow_max_model_calls": 1}, {"workflow_total_token_budget": 9000}])
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
