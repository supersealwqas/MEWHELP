"""CH07最终模型输入验收：完整原话、稳定前缀、来源校验及真实SDK日志。"""

import json
import logging

import pytest
from langchain_core.messages import convert_to_messages

from app.core.context_budget import TokenEstimator
from app.core.model_budget import BudgetLedger
from app.schemas.tools import ToolContext, ToolResult
from app.schemas.workflow import AgentDecision, UnderstandingResult
from app.workflow.agent import model_messages
from app.workflow.graph import build_workflow
from app.workflow.model import WorkflowModelAdapter, understanding_tool_schema
from app.workflow.prompts import UNDERSTANDING_PROMPT
from app.workflow.state import new_turn_state
from tests.test_context_layers import _turn
from tests.workflow.graph_fakes import call, dependencies, turn
from tests.workflow.test_model import sdk_runtime, call_response, intent_response, text_response, understanding_response


@pytest.fixture
def context_logs(caplog):
    """临时捕获app自有handler的内容。参数：caplog；返回：日志夹具；异常：无。"""
    logger = logging.getLogger("app")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


async def test_resolved_query_is_data_and_current_original_words_are_unchanged():
    """理解后的问句只能进资料区，不能替换原问。参数：无；返回：None；异常：断言失败。"""
    raw, resolved = "那件呢？  ", "订单1001的商品是什么？"
    deps = dependencies(understandings=[UnderstandingResult(resolved_query=resolved, standard_query=resolved,
        needs_resolution=True, order_id="1001")], decisions=[AgentDecision(kind="answer")], answers=[["请核实订单。"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn(raw))
    for entry in deps.model.inputs:
        messages = entry["messages"]
        assert all(message["role"] != "system" for message in messages)
        assert messages[0]["content"] == raw
        assert len(messages) >= 2 and messages[1]["role"] == "user"
        if entry["stage"] != "understand":
            assert resolved in messages[1]["content"]
    assert state["raw_query"] == raw


async def test_trailing_data_does_not_become_query_and_assistant_guess_stays_untrusted(sdk_runtime):
    """真实SDK保留助手原文，但不能把猜测编号或资料当作本轮原问。参数：sdk_runtime；返回：None；异常：断言失败。"""
    raw = "那袋猫粮能退吗？"
    history = [_turn(1, user="买过一袋猫粮", answer="您的订单是1007").model_dump(mode="json")]
    state = new_turn_state(ToolContext(conversation_id=1, session_id="s", user_id="u", turn_id="now", user_message=raw), history, 4)
    deps = dependencies()
    messages = model_messages(state, deps.settings, UNDERSTANDING_PROMPT, [understanding_tool_schema()])
    sdk_runtime.responses.append(call_response("submit_understanding", json.dumps({
        "resolved_query":"订单1007的猫粮能退吗？", "standard_query":"订单1007退货", "needs_resolution":True,
        "order_id":"1007"}, ensure_ascii=False)))
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(messages, BudgetLedger())
    sent = sdk_runtime.requests[0]["messages"]
    assert sent[-2]["content"] == raw and "上下文资料" in sent[-1]["content"]
    assert any(row["content"] == "您的订单是1007" for row in sent)
    assert result.resolved_query == raw and result.order_id is None


@pytest.mark.parametrize("window,output_limit", [(131072, 4096), (22000, 2000)])
async def test_sdk_context_log_matches_each_actual_request(sdk_runtime, context_logs, window, output_limit):
    """不能把适配器处理前的消息冒充实际日志；同阶段前缀跨轮稳定。参数：SDK及日志；返回：None；异常：断言失败。"""
    deps = dependencies(model_context_window=window, max_output_tokens=output_limit, max_agent_steps=3)
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    for index in range(2):
        raw = f"第{index}次查询订单1001"
        sdk_runtime.responses.extend([understanding_response(raw), intent_response("订单"),
            call_response("submit_decision", '{"kind":"answer","suggested_actions":[]}'), text_response("请继续补充。")])
        await build_workflow(deps, checkpointer=None).ainvoke(turn(raw))
    records = [json.loads(record.message.split("model_ctx ", 1)[1]) for record in context_logs.records
               if "model_ctx " in record.message]
    assert len(records) == len(sdk_runtime.requests) == 8
    for record, request in zip(records, sdk_runtime.requests):
        assert record["messages"] == request["messages"]
        assert record["tools"] == request.get("tools", [])
        assert record["message_count"] == len(request["messages"])
        assert record["tokens_estimate"] > 0
        assert request["max_completion_tokens"] <= output_limit
        assert sum(message["role"] == "system" for message in record["messages"]) == 1
    for first, second in zip(records[:4], records[4:]):
        assert first["prefix_sha256"] == second["prefix_sha256"]
    assert len([record for record in context_logs.records if "history_ctx " in record.message]) == 2


async def test_fixed_reply_still_logs_one_shared_history_snapshot(context_logs):
    """不经过Agent的闲聊也必须能追查历史。参数：context_logs；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["闲聊"])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("你好"))
    logs = [record.message for record in context_logs.records if "history_ctx " in record.message]
    assert len(logs) == 1 and state["history_context"] is not None
    assert [entry["stage"] for entry in deps.model.inputs] == ["understand", "classify", "decide"]


async def test_large_current_tool_is_explicitly_omitted_only_in_model_view():
    """大结果不能导致无回答，也不能缩写原始落库。参数：无；返回：None；异常：断言失败。"""
    original = "工具原始事实" * 10000
    deps = dependencies(decisions=[AgentDecision(kind="tools", tool_calls=[call("a")]), AgentDecision(kind="answer")],
                        results=[ToolResult.success({"order_id":"1001", "detail":original})], answers=[["查询已完成。"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("请查订单1001"))
    assert state["answer"] == "查询已完成。"
    assert any(original in row.content for row in state["messages"] if row.type == "tool")
    answer_input = next(entry["messages"] for entry in deps.model.inputs if entry["stage"] == "answer")
    tool = next(row for row in answer_input if row["role"] == "tool")
    assert json.loads(tool["content"])["truncated"] and json.loads(tool["content"])["data"]["order_id"] == "1001"
    assert TokenEstimator().message_tokens(convert_to_messages([tool])) <= deps.settings.tool_result_max_tokens


async def test_old_message_caps_do_not_override_configured_model_window():
    """旧12000/16000配置不能成为新窗口背后的隐藏上限。参数：无；返回：None；异常：断言失败。"""
    deps = dependencies(token_budget=1, request_token_budget=1, workflow_total_token_budget=1,
                        decisions=[AgentDecision(kind="answer")], answers=[["新窗口预算正常。"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("订单查询"))
    assert state["answer"] == "新窗口预算正常。"
    assert state["budget"]["request_token_budget"] > 16000


async def test_full_dynamic_schema_cannot_be_omitted_to_fit_small_window(sdk_runtime):
    """完整工具清单已超旧演示预留时，漏计schema继续发包会失败。

    参数：sdk_runtime 外部HTTP替身；返回：None；异常：图或断言失败。
    """
    deps = dependencies(model_context_window=18000, max_output_tokens=2000, max_agent_steps=3)
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    state = await build_workflow(deps, checkpointer=None).ainvoke(turn("查询订单1001"))
    assert state["stop_reason"] == "budget_exceeded" and sdk_runtime.requests == []


@pytest.mark.parametrize("claimed,expected", [("1001", "1001"), ("1007", None)])
async def test_summary_candidate_is_checked_against_original_covered_sources(sdk_runtime, claimed, expected):
    """只能回忆摘要区原文确实出现的订单，梗概自己不能创造可信编号。参数：SDK/声明/预期；返回：None；异常：断言失败。"""
    from app.workflow.context import history_snapshot
    raw = "最开始那个订单后来怎么说"
    history = [_turn(10, user="订单1001想退款，尚未提交申请。", answer="需要确认退款条件。"),
               _turn(20, user="现在查询订单1002", answer="需要核实当前状态。")]
    state = new_turn_state(ToolContext(conversation_id=1, session_id="s", user_id="u", turn_id="now", user_message=raw),
                           [row.model_dump(mode="json") for row in history], 23)
    config = dependencies().settings
    stored = {"turns": history, "summary_upto_msg_id": 13, "layer1_from_msg_id": 13, "summaries": [{
        "seq":1, "from_msg_id":10, "upto_msg_id":13, "content":f"用户曾问订单{claimed}退款，申请尚未提交。"}]}
    state["history_context"] = history_snapshot(state, config, stored)
    sdk_runtime.responses.append(call_response("submit_understanding", json.dumps({"resolved_query": f"订单{claimed}的退款需求后来如何处理？",
        "standard_query":f"订单{claimed}退款", "needs_resolution":True, "order_id":claimed}, ensure_ascii=False)))
    result = await WorkflowModelAdapter(model_factory=sdk_runtime.factory).understand(
        model_messages(state, config, UNDERSTANDING_PROMPT, [understanding_tool_schema()]), BudgetLedger())
    assert result.order_id == expected
    actual = sdk_runtime.requests[0]["messages"]
    assert not any(row["content"] == history[0].messages[0].content for row in actual)
    assert f"用户曾问订单{claimed}" in actual[-1]["content"]


def test_demo_budget_reserves_actual_protocol_retry_and_allows_one_turn():
    """补救消息也占瞬时窗口；空工具集合不能虚增固定成本。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.context import shared_budget
    from app.workflow.prompts import DECISION_NUDGE
    config = dependencies(model_context_window=18000, max_output_tokens=2000, max_agent_steps=3).settings
    budget = shared_budget(config)
    retry = TokenEstimator().message_tokens(convert_to_messages([{"role":"user", "content":DECISION_NUDGE}]))
    assert budget.current_peak_tokens == config.max_user_input_tokens + config.tool_max_calls_per_turn * (
        config.tool_result_max_tokens + config.tool_call_max_tokens) + retry
    assert budget.layer1 >= config.context_steady_turn_tokens


async def test_demo_all_sdk_stages_use_configured_output_limit(sdk_runtime):
    """窗口预留2000时，适配器不能仍申请4096输出。参数：sdk_runtime；返回：None；异常：断言失败。"""
    from app.knowledge.query_rewrite import QueryExpander
    config = dependencies(model_context_window=18000, max_output_tokens=2000, max_agent_steps=3).settings
    ledger = BudgetLedger.from_settings(config)
    sdk_runtime.responses.append(call_response("submit_expanded_queries", '{"queries":["订单政策"]}'))
    result = await QueryExpander(model_factory=sdk_runtime.factory).expand("订单政策", simple_query="订单政策", budget=ledger)
    assert result.queries == ["订单政策"]
    assert sdk_runtime.requests[0]["max_completion_tokens"] == 2000


async def test_default_twenty_one_turns_keep_full_checkpoint_and_do_not_demote(tmp_path):
    """默认配置在完整21轮可容纳时不得主动压缩；真实服务和仓储验证。参数：tmp_path；返回：None；异常：断言失败。"""
    from sqlalchemy import text
    from tests.workflow.service_fakes import collect, service_runtime
    config = {"configurable": {"thread_id": "mewhelp:ch05:1"}}
    async with service_runtime(tmp_path, intents=["闲聊"] * 21) as (service, deps, factory):
        for index in range(21):
            raw = f"第{index}轮：" + "这是保留原文的上下文验证。" * 25
            assert (await collect(service, raw))[-1].event == "done"
        state = (await service.graph.aget_state(config)).values
        assert len(state["messages"]) == 42
        assert state["history_context"]["layer1_tokens"] > 12000
        assert state["history_context"]["layer1_from_msg_id"] == 0
        assert not state["history_context"]["summary_required"]
        async with factory() as session:
            assert (await session.execute(text("SELECT summary_upto_msg_id, layer1_from_msg_id FROM conversations"))).one() == (None, None)
            assert (await session.execute(text("SELECT COUNT(*) FROM conversation_summaries"))).scalar_one() == 0


def test_evidence_selection_never_splits_a_block_or_renumbers_sources():
    """整块超限必须省略，不能截半段条件或把原编号重编。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.context import select_evidence
    config = dependencies(context_evidence_max_tokens=100).settings
    evidence = [{"n":1,"answer":"完整条款" * 200}, {"n":2,"answer":"第二条原文。"}]
    chosen = select_evidence(evidence, config)
    assert chosen == [evidence[1]]
    assert evidence[0]["answer"] == "完整条款" * 200


async def test_graph_loads_only_context_metadata_instead_of_rereading_all_messages(tmp_path, monkeypatch):
    """checkpoint已做增量对账后，快照不能再次全量扫描原始消息。参数：隔离目录/替换器；返回：None；异常：断言失败。"""
    from app.repositories.context import ContextRepository
    from tests.workflow.service_fakes import collect, service_runtime
    observed, original = [], ContextRepository.load_context

    async def record(repository, identifier, **options):
        """保留真实查询仅记录读取选项。参数：仓储/主键/选项；返回：真实结果；异常：数据库错误。"""
        observed.append(options.get("include_turns", True))
        return await original(repository, identifier, **options)

    monkeypatch.setattr(ContextRepository, "load_context", record)
    async with service_runtime(tmp_path, intents=["闲聊"] * 2) as (service, _, _):
        for question in ("第一轮", "第二轮"):
            assert (await collect(service, question))[-1].event == "done"
    assert observed == [False, False]


async def test_prompt_evaluator_also_keeps_raw_question_for_classification(sdk_runtime):
    """评估必须跟线上同形态，不能另把补全句写成用户原话。参数：sdk_runtime；返回：None；异常：断言失败。"""
    from tests.eval_ch06_prompts import evaluate_case
    raw, resolved = "它到哪了？", "订单1001的物流到哪了？"
    sdk_runtime.responses.extend([understanding_response(resolved, needs_resolution=True, order_id="1001"), intent_response("物流")])
    row = await evaluate_case({"id":"ch07-eval", "query":raw, "expected":{"intent":"物流", "order_id":"1001"}},
        [{"role":"user", "content":"查订单1001"}, {"role":"assistant", "content":"需要确认物流。"}], sdk_runtime.factory)
    assert row["correct"]
    assert sdk_runtime.requests[1]["messages"][-2]["content"] == raw
    assert resolved in sdk_runtime.requests[1]["messages"][-1]["content"]


async def test_sdk_logs_preserve_actual_tool_pair_order(sdk_runtime, context_logs):
    """带工具时也须逐项核对实际SDK发包，资料不能插入工具配对中间。参数：SDK与日志；返回：None；异常：断言失败。"""
    raw = "查询订单1001"
    deps = dependencies(results=[ToolResult.success({"order_id":"1001", "status":"已发货"})])
    deps.model = WorkflowModelAdapter(model_factory=sdk_runtime.factory)
    sdk_runtime.responses.extend([understanding_response(raw), intent_response("订单"),
        call_response("query_order", '{"order_id":"1001"}'),
        call_response("submit_decision", '{"kind":"answer","suggested_actions":[]}'), text_response("订单已发货。")])
    await build_workflow(deps, checkpointer=None).ainvoke(turn(raw))
    logs = [json.loads(record.message.split("model_ctx ", 1)[1]) for record in context_logs.records if "model_ctx " in record.message]
    assert len(logs) == len(sdk_runtime.requests) == 5
    for log, request in zip(logs, sdk_runtime.requests):
        assert log["messages"] == request["messages"]
    assert [row["role"] for row in logs[-1]["messages"]] == ["system", "user", "user", "assistant", "tool"]
