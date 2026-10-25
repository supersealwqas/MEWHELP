"""真实验收器的保护与失败判定测试；只验证工具本身，不冒充真实模型评估。"""

import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest


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
