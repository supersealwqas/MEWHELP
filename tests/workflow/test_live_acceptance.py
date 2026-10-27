"""真实验收器的保护与失败判定测试；只验证工具本身，不冒充真实模型评估。"""

import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from tests.workflow.test_model import sdk_runtime, understanding_response, intent_response


@pytest.mark.parametrize("large_old_turn", [False, True])
async def test_prompt_evaluation_preserves_recent_history_and_trims_whole_turns(sdk_runtime, large_old_turn):
    """真实SDK发包必须保留近期分类历史，并按线上预算整轮裁剪旧历史。

    参数：sdk_runtime仅替换外部HTTP；large_old_turn是否添加超预算旧轮。
    返回：None；异常：断言失败。
    """
    from tests.eval_ch06_prompts import evaluate_case
    recent = [{"role": "user", "content": "我想投诉你们的服务"},
              {"role": "assistant", "content": "抱歉给您带来不好的体验。"}]
    history = ([{"role": "user", "content": "旧话" * 12000},
                {"role": "assistant", "content": "旧回复"}] if large_old_turn else []) + recent
    query = "运费怎么算"
    sdk_runtime.responses.extend([understanding_response(query), intent_response("商品咨询")])
    row = await evaluate_case({"id": "history-contract", "query": query,
                               "expected": {"intent": "商品咨询", "order_id": None, "preserve_raw": True}},
                              history, sdk_runtime.factory)
    assert row["correct"], (row["failures"], row["error_type"])
    expected_messages = [*recent, {"role": "user", "content": query}]
    assert len(sdk_runtime.requests) == 2
    assert sdk_runtime.requests[0]["messages"][1:] == expected_messages
    assert sdk_runtime.requests[1]["messages"][1:] == expected_messages


@pytest.mark.asyncio
async def test_recording_chat_preserves_bound_tool_protocol():
    """取证按工具名分阶段且保留原始分片。参数：无；返回：None；异常：断言失败。"""
    from langchain_core.messages import AIMessageChunk, HumanMessage
    from tests.ch05_live_runtime import RecordingChat
    calls, rows = [], []

    class Model:
        """不联网的协议探针，检查包装器没有更改工具约束。"""
        def bind_tools(self, tools, **kwargs):
            """记录原绑定参数。参数：工具和选项；返回：自身；异常：无。"""
            calls.append((tools, kwargs))
            return self

        async def astream(self, messages):
            """发出带计量的原工具片。参数：消息；返回：分片；异常：无。"""
            yield AIMessageChunk(content="", tool_call_chunks=[{"name":"submit_intent", "args":"{}", "id":"x", "index":0}],
                response_metadata={"finish_reason":"tool_calls"}, usage_metadata={"input_tokens":1,"output_tokens":2,"total_tokens":3})

    schema = [{"type":"function", "function":{"name":"submit_intent"}}]
    model = RecordingChat(Model(), rows, "answer").bind_tools(schema, tool_choice="required", strict=True)
    async for _ in model.astream([HumanMessage(content="你好")]):
        pass
    assert calls == [(schema, {"tool_choice":"required", "strict":True})]
    assert rows[0]["stage"] == "classify"
    assert rows[0]["chunks"][0]["tool_call_chunks"][0]["args"] == "{}"
    assert rows[0]["chunks"][0]["response_metadata"]["finish_reason"] == "tool_calls"
    assert rows[0]["chunks"][0]["usage_metadata"]["total_tokens"] == 3


def test_confidence_report_separates_fallbacks():
    """回退零分不得混进原始分布。参数：无；返回：None；异常：断言失败。"""
    from tests.eval_ch06_prompts import confidence_report
    data, report = confidence_report([
        {"id":"ok","raw_classification_valid":True,"correct":True,"confidence":.8},
        {"id":"bad","raw_classification_valid":True,"correct":False,"confidence":.95,"expected_intent":"订单","predicted_intent":"物流"},
        {"id":"fallback","raw_classification_valid":False,"correct":False,"confidence":0},
    ])
    assert "错判样本的 confidence 分布" in report
    assert data["valid_correct"] == data["valid_wrong"] == data["fallbacks"] == 1
    assert data["wrong_min"] == data["wrong_median"] == data["wrong_max"] == .95
    assert data["wrong_buckets"] == [0, 0, 0, 1]
    assert data["high_confidence_errors"][0]["predicted_intent"] == "物流"


def test_confidence_report_handles_no_errors():
    """无错判不能伪造校准结论或零分统计。参数：无；返回：None；异常：断言失败。"""
    from tests.eval_ch06_prompts import confidence_report
    data, report = confidence_report([])
    assert "错判样本的 confidence 分布" in report and "本次无错判样本" in report
    assert data["wrong_min"] is data["wrong_median"] is data["wrong_max"] is None


def test_cleanup_manifest_rejects_foreign_refund():
    """新增退款表清理仍须逐条登记身份。参数：无；返回：None；异常：断言失败。"""
    from tests.ch05_live_runtime import validate_cleanup
    entry = {"id":7,"session_id":"s","user_id":"demo-user","turn_ids":["t"],"pool_ids":[],"ticket_numbers":[],"refund_numbers":["RF1"]}
    data = {"conversation":{"id":7,"session_id":"s","user_id":"demo-user"},"messages":[{"turn_id":"t"}],"pool_rows":[],"tickets":[],
        "refunds":[{"refund_no":"RF1","conversation_id":8,"source_turn_id":"t","user_id":"demo-user"}]}
    with pytest.raises(ValueError, match="退款"):
        validate_cleanup(entry, data)


def test_judge_accepts_delivery_progress_synonyms():
    """同义的送达位置不能被词表误判。参数：无；返回：None；异常：断言失败。"""
    from tests.eval_ch06_prompts import inspect_understanding
    assert inspect_understanding("查刚才那笔配送进度", {"resolved_any":["物流","配送","运输"]},
        {"resolved_query":"订单1001的包裹目前送到哪里", "standard_query":"订单配送进度"}) == []


def test_live_judge_reads_resolution_flag_from_model_evidence():
    """needs_resolution不在State契约内，须读取真实模型取证。参数：无；返回：None；异常：断言失败。"""
    from langchain_core.messages import AIMessageChunk
    from tests.verify_ch06_live import judge
    raw = {"resolved_query":"订单1001物流进度", "standard_query":"订单1001物流进度", "order_id":"1001", "needs_resolution":True}
    output = AIMessageChunk(content="",tool_call_chunks=[{"name":"submit_understanding","args":json.dumps(raw),"id":"u","index":0}],
        response_metadata={"finish_reason":"tool_calls"}).model_dump(mode="json")
    row = {"state":{key:value for key,value in raw.items() if key != "needs_resolution"},"events":[{"event":"answer_metadata"},{"event":"done"}],
        "failures":[],"http_status":200,"models":[{"stage":"understand","output":output}],"active_model_streams":0,"active_graph_nodes":0}
    row["state"].update(raw_query="它呢",intent="物流",completion_metadata={"status":"completed"})
    assert judge(row,{"query":"它呢","expected":{"intent":"物流","needs_resolution":True,"order_id":"1001"}})["failures"] == []


@pytest.mark.parametrize("current", ["它呢？", "订单1002的物流进度"])
def test_live_judge_checks_resolved_question_reaches_business_models(current):
    """意图正确不足以通过，业务决策和回答实际输入必须携带完整问题。

    参数：current模拟真实模型输入中的本轮用户消息；返回：None；异常：断言失败。
    """
    from tests.verify_ch06_live import judge
    resolved = "订单1002的物流进度"
    row = {"state": {"raw_query": "它呢？", "resolved_query": resolved, "standard_query": resolved,
        "intent": "物流", "route": "business", "order_id": "1002", "completion_metadata": {"status": "completed"}},
        "events": [{"event": "answer_metadata"}, {"event": "done"}], "failures": [], "http_status": 200,
        "models": [{"stage": stage, "inputs": [{"type": "human", "content": current}]} for stage in ("decide", "answer")],
        "active_model_streams": 0, "active_graph_nodes": 0}
    failures = judge(row, {"query": "它呢？", "expected": {"intent": "物流", "order_id": "1002"}})["failures"]
    assert ("business_resolved_query_not_consumed" in failures) == (current != resolved)


def test_explicit_excluded_order_is_not_ambiguous():
    """明确排除旧号后当前和下一轮均只选择新号；真正并列仍不得任选。参数：无；返回：None；异常：断言失败。"""
    from app.schemas.workflow import UnderstandingResult
    from app.workflow.routing import _understanding_outcome, order_candidate
    raw = "改查订单1002的商品总价，不查1001了。"
    result = UnderstandingResult(resolved_query=raw, standard_query="订单1002商品总价", needs_resolution=False, order_id="1002")
    validated, fallback = _understanding_outcome(raw,result,[])
    assert not fallback and validated.order_id == "1002"
    assert order_candidate({"raw_query":raw,"order_id":"1002"},[{"order_id":"1001"},{"order_id":"1002"}]) == "1002"
    followup = UnderstandingResult(resolved_query="订单1002能退货吗？", standard_query="订单1002退货条件", needs_resolution=True, order_id="1002")
    validated, fallback = _understanding_outcome("这笔能退吗？",followup,[{"messages":[{"role":"user","content":raw}]}])
    assert not fallback and validated.order_id == "1002"
    ambiguous = "订单1001和1002，其中一笔想退货。"
    assert order_candidate({"raw_query":ambiguous,"order_id":"1002"},[{"order_id":"1001"},{"order_id":"1002"}]) is None


@pytest.mark.parametrize("expected_count,actual_count", [(3, 1), (3, 3), (1, 1), (1, 3)])
def test_live_judge_enforces_core_and_faq_query_counts(expected_count, actual_count):
    """核心适用性三查询与单事实FAQ单查询必须进入真实验收门禁。

    参数：expected_count标注条数；actual_count真实扩写条数；返回：None；异常：断言失败。
    """
    from tests.verify_ch06_live import judge
    row = {"state": {"raw_query": "能退吗", "resolved_query": "能退吗", "standard_query": "能退吗",
        "intent": "退款退货", "route": "order_policy", "expanded_queries": [f"侧重点{i}" for i in range(actual_count)],
        "completion_metadata": {"status": "completed"}},
        "events": [{"event": "answer_metadata"}, {"event": "done"}], "failures": [], "http_status": 200,
        "nodes": ["fetch_order", "retrieve_policy", "main_agent"], "models": [],
        "active_model_streams": 0, "active_graph_nodes": 0}
    failures = judge(row, {"query": "能退吗", "expected": {"intent": "退款退货", "query_count": expected_count}})["failures"]
    assert ("policy_query_count" in failures) == (actual_count != expected_count)


def test_agent_cli_refuses_without_test_before_loading_env(tmp_path):
    """缺test必须在读取环境/创建报告前拒绝。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    target = tmp_path / "report"
    result = subprocess.run([sys.executable, "-m", "tests.eval_ch05_prompts", "--suite", "agent",
        "--cases", "tests/fixtures/ch05/agent_cases.json", "--report-dir", str(target),
        "--env-file", str(tmp_path / "missing.env")], capture_output=True, text=True)
    assert result.returncode == 2 and "agent suite requires --test" in result.stderr
    assert not target.exists()


def test_verifier_refuses_without_test_before_creating_report(tmp_path):
    """独立验收缺test不能连接资源。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    target = tmp_path / "report"
    result = subprocess.run([sys.executable, "-m", "tests.verify_ch05_live", "--report-dir", str(target)],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "--test" in result.stderr
    assert not target.exists()


def test_agent_judge_catches_wrong_route_tools_and_unconfirmed_ticket():
    """错节点、猜参数和聊天建票均必须变红。参数：无；返回：None；异常：断言失败。"""
    from tests.ch05_live_runtime import inspect_turn
    expected = {"query":"订单1001", "intent":"物流", "nodes":["main_agent"],
        "tools":[{"name":"query_order","args":{"order_id":"1001"}},
                 {"name":"query_logistics","args":{"tracking_no":"SF213502378238"}}],
        "actions":[], "required":["杭州"], "forbidden":["已签收"], "dependent_tools":True}
    row = {"events":[{"event":"delta","data":{"delta":"杭州运输中"}},{"event":"answer_metadata"},{"event":"done"}],
        "state":{"intent":"物流", "raw_query":"订单1001", "resolved_query":"订单1001", "suggested_actions":[],
                 "completion_metadata":{"status":"completed"}, "budget":{}},
        "nodes":["resolve_reference","classify_intent","main_agent","log_turn"],
        "tools":expected["tools"], "tool_results":[], "decisions":[{"kind":"tools","tool_calls":[expected["tools"][0]]},
            {"kind":"tools","tool_calls":[expected["tools"][1]]}], "pool_rows":[], "tickets":[], "failures":[]}
    assert inspect_turn(expected, row)["failures"] == []
    broken = copy.deepcopy(row)
    broken["nodes"].insert(2, "retrieve_knowledge")
    broken["tools"][1]["args"]["tracking_no"] = "SF1"
    broken["tickets"] = [{"ticket_no":"BAD"}]
    broken["events"][0]["data"]["delta"] = "已签收"
    failures = inspect_turn(expected, broken)["failures"]
    assert {"node_order", "tool_sequence", "unconfirmed_ticket", "missing_fact:杭州", "forbidden_claim:已签收"} <= set(failures)


def test_cleanup_manifest_rejects_foreign_turn_and_pool_row():
    """清理必须拒绝未登记轮次和其他会话问题池。参数：无；返回：None；异常：断言失败。"""
    from tests.ch05_live_runtime import validate_cleanup
    entry = {"id":7,"session_id":"ch05-unique","user_id":"demo-user","turn_ids":["t1"],"pool_ids":[3],"ticket_numbers":[]}
    snapshot = {"conversation":{"id":7,"session_id":"ch05-unique","user_id":"demo-user"},
        "messages":[{"turn_id":"t1"}],"pool_rows":[{"id":3,"conversation_id":7}],"tickets":[]}
    validate_cleanup(entry, snapshot)
    for field, value in [("messages", [{"turn_id":"foreign"}]), ("pool_rows", [{"id":3,"conversation_id":8}])]:
        broken = {**snapshot, field:value}
        with pytest.raises(ValueError):
            validate_cleanup(entry, broken)


def test_invalid_tool_audit_is_reported_without_aborting_other_cases():
    """真实非法工具审计不能令验收器KeyError丢证据。参数：无；返回：None；异常：断言失败。"""
    from tests.ch05_live_runtime import collect_tool_audit
    messages = [{"role":"assistant", "tool_calls":[{"type":"invalid_tool_decision", "tool_calls":[
        {"id":"bad", "name":"query_faq", "args":{"keyword":"猫砂盆开不了机"}}], "invalid_tool_calls":[], "tool_call_chunks":[]}]}]
    result = collect_tool_audit(messages)
    assert result["tools"] == []
    assert result["invalid_tool_decisions"] == messages[0]["tool_calls"]
    assert result["failures"] == ["invalid_tool_decision"]


def test_explicit_acceptance_budget_overrides_legacy_env_without_modifying_it(tmp_path, monkeypatch):
    """验收显式预算必须覆盖旧env但不改原文件。参数：tmp_path临时配置；返回：None；异常：断言失败。"""
    from tests.ch05_live_runtime import build_config
    monkeypatch.setenv("TOKEN_BUDGET", "2000")
    env_file = tmp_path / "test.env"
    original = "TOKEN_BUDGET=2000\nCHAT_BASE_URL=http://127.0.0.1:9/v1\nCHAT_MODEL=offline\nCHAT_API_KEY=offline\n"
    env_file.write_text(original, encoding="utf-8")
    for name in ("bge-m3", "bge-reranker-large"):
        (tmp_path / "models" / name).mkdir(parents=True)
    args = SimpleNamespace(test=True, env_file=env_file, report_dir=tmp_path / "report", token_budget=12000, request_token_budget=16000)
    config = build_config(args)
    assert config.token_budget == 12000 and config.request_token_budget == 16000
    assert env_file.read_text(encoding="utf-8") == original
