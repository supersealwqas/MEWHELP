"""CH08验收器保护：目标库、真实依赖证据和必验项目缺失时不得报告通过。"""

from copy import deepcopy
import importlib
import json
import hashlib
from types import SimpleNamespace
import struct

import pytest


def complete_report() -> dict:
    """构造独立手写的完整证据样例。

    参数：无；返回：仅供判定器测试的报告字典；异常：无。
    """
    return {
        "database": "mewhelp_test",
        "dependencies": {"model": "real", "mysql": "real", "mcp": "real"},
        "cases": [
            {"id": name, "status": "passed", "failures": [], "evidence": ["observations.json"]}
            for name in (
                "builtin_hot_registration", "mcp_three_tools", "mcp_hot_discovery",
                "ticket_confirm", "ticket_cancel", "read_retry_write_once",
                "recovery_and_faults", "browser_actions", "annotated_dialogues",
            )
        ],
    }


@pytest.mark.parametrize("database", ["mewhelp", "mewhelp_ch04_test", "", "MEWHELP_TEST"])
def test_live_verifier_rejects_application_database(database):
    """只允许明确的本章测试库，防止验收器写入应用数据。

    参数：database为不允许的目标名；返回：None；异常：断言失败。
    """
    module = importlib.import_module("tests.verify_ch08_live")
    with pytest.raises(ValueError, match="mewhelp_test"):
        module.validate_database(database)


@pytest.mark.parametrize("dependency", ["model", "mysql", "mcp"])
def test_missing_real_model_or_mcp_evidence_is_failure(dependency):
    """缺依赖或用替身时整体失败，即使各项目自报成功。

    参数：dependency为故意移除的依赖；返回：None；异常：断言失败。
    """
    module = importlib.import_module("tests.verify_ch08_live")
    report = complete_report()
    assert module.report_failures(report) == []
    report["dependencies"].pop(dependency)
    assert f"dependency:{dependency}" in module.report_failures(report)
    report["dependencies"][dependency] = "fake"
    assert f"dependency:{dependency}" in module.report_failures(report)


@pytest.mark.parametrize("defect", ["skipped", "missing", "no_evidence", "error", "duplicate"])
def test_report_cannot_pass_with_skipped_required_case(defect):
    """必验项目缺失、跳过、无原始证据或仍有错误时不能验收通过。

    参数：defect为单一报告缺陷；返回：None；异常：断言失败。
    """
    module = importlib.import_module("tests.verify_ch08_live")
    report = deepcopy(complete_report())
    if defect == "skipped":
        report["cases"][0]["status"] = "skipped"
    elif defect == "missing":
        report["cases"].pop(0)
    elif defect == "no_evidence":
        report["cases"][0]["evidence"] = []
    elif defect == "error":
        report["cases"][0]["failures"] = ["工具未真正调用"]
    else:
        report["cases"].append(deepcopy(report["cases"][0]))
    assert module.report_failures(report)


def evidence_file(root, name, value):
    """写入测试证据并计算独立摘要。参数：目录、文件名、JSON或字节；返回：引用；异常：文件错误。"""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, default=str).encode()
    path.write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def model_report(root):
    """构造有完整原始形状的离线报告，用于损坏证据测试，不声称真实联网。

    参数：root测试目录；返回：报告目录；异常：固定题集或文件错误。
    """
    from langchain_core.messages import AIMessageChunk, HumanMessage
    from tests import eval_ch08_tools as evaluation
    from app.tools.contracts import json_hash
    cases, fingerprint = evaluation.load_cases()
    catalog = [{"name": name, "identity": {"name": name, "source": origin[0], "server": origin[1]},
                "access": "write" if name == "create_ticket" else "read", "description": "离线协议样例工具",
                "parameters": {"type": "object", "properties": {}}} for name, origin in evaluation.REQUIRED_TOOLS.items()]
    for definition in catalog:
        definition["schema_hash"] = json_hash(definition["parameters"])
    schemas = [{"type": "function", "function": {key: definition[key] for key in ("name", "description", "parameters")}}
               for definition in catalog] + [{"type": "function", "function": {"name": "submit_decision"}}]
    rows = []
    for repetition in (1, 2):
        for case in cases:
            for number, turn in enumerate(case["turns"], 1):
                expected = turn["expected"]
                identifier = f"{case['id']}.{repetition}.{number}"
                calls = []
                for name in expected["required_tools"]:
                    args = ({"description": expected["description"], "ticket_type": expected["ticket_types"][0]}
                            if name == "create_ticket" else {"order_id": "1001"})
                    calls.append({"id": name, "name": name, "args": args, "argument_sources":
                                  {key: [{"kind": "fixture"}] for key in args}})
                raw_calls = [{key: call[key] for key in ("id", "name", "args")} for call in calls]
                if not raw_calls:
                    raw_calls = [{"id": "terminal", "name": "submit_decision",
                                  "args": {"kind": "answer", "suggested_actions": []}}]
                output = AIMessageChunk(content="", tool_calls=raw_calls,
                                        response_metadata={"finish_reason": "tool_calls"}).model_dump(mode="json")
                record = {"stage": "decide", "turn_id": identifier, "started": 1, "ended": 2, "closed": True,
                          "inputs": [HumanMessage(content=turn["query"]).model_dump(mode="json")],
                          "tool_schemas": deepcopy(schemas),
                          "chunks": [{"clock": 1.5, **output}], "output": output}
                records = [record]
                if not expected["preview"]:
                    answer = AIMessageChunk(content="已按原问核对本次事实。",
                        response_metadata={"finish_reason": "stop"}).model_dump(mode="json")
                    records.append({**record, "stage": "answer", "tool_schemas": [],
                                    "chunks": [{"clock": 1.5, **answer}], "output": answer})
                observation = {"id": identifier, "case_id": case["id"], "repetition": repetition, "turn": number,
                    "query": turn["query"], "calls": calls, "errors": [], "failures": [], "models": records,
                    "answer": "" if expected["preview"] else "已按原问核对本次事实。",
                    "preview_candidate": expected["preview"], "trusted_description": expected.get("description"),
                    "trusted_intent": {"status": "collecting"} if expected.get("collecting") else None,
                    "feedback": [], "actual_preview_executed": False, "business_tools_executed": 0, "database_used": False}
                rows.append(evaluation.inspect_turn(expected, observation))
    metadata = {"suite": "ch08_tools_model_evaluation", "model": "review-model", "case_sha256": fingerprint,
        "model_origin": "configured_chat_openai", "feedback_origin": "simulation", "business_execution": "none",
        "actual_interrupt": False, "repetitions": 2, "expected_rows": 46, "suite_errors": [],
        "catalog": catalog,
        "source_sha256": evaluation._source_hashes(), "source_sha256_after": evaluation._source_hashes()}
    reviews = [{"id": row["id"], "evidence_sha256": evaluation.evidence_hash(row), "reviewer": "Codex (AI reviewer)",
                "verdict": "passed", "reason": "仅测试报告协议的离线样例", "violations": []} for row in rows]
    evaluation.write_report(root, metadata, evaluation.apply_semantic_reviews(rows, reviews))
    evidence_file(root, "semantic-review-applied.json", reviews)
    (root / "frozen-cases.json").write_bytes(evaluation.CASES_PATH.read_bytes())
    return root


@pytest.mark.parametrize("defect", ["summary_only", "controlled", "stale_source", "missing_row", "missing_raw",
                         "changed_output", "changed_review", "changed_query", "changed_labels", "missing_answer", "wrong_bound_schema"])
def test_model_import_rejects_unproven_or_stale_success(tmp_path, defect):
    """防止摘要通过掩盖假模型、旧代码、缺采样或错绑审阅。参数：临时目录/损坏点；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch08_live")
    assert callable(getattr(module, "validate_model_evidence", None)), "导入模型证据缺少逐项重验门禁"
    source = model_report(tmp_path)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if defect == "summary_only":
        (source / "results.json").unlink()
    elif defect == "controlled":
        summary["model_origin"] = "controlled_test"
    elif defect == "stale_source":
        summary["source_sha256"]["app/main.py"] = "0" * 64
    elif defect == "missing_row":
        rows.pop()
    elif defect == "missing_raw":
        rows[0]["models"] = []
    elif defect == "changed_output":
        rows[0]["models"][0]["output"]["content"] = "没有原始分片支持的内容"
    elif defect == "changed_review":
        reviews = json.loads((source / "semantic-review-applied.json").read_text(encoding="utf-8"))
        reviews[0]["evidence_sha256"] = "0" * 64
        evidence_file(source, "semantic-review-applied.json", reviews)
    elif defect == "changed_query":
        rows[0]["query"] = "换过的题目"
    elif defect == "changed_labels":
        (source / "frozen-cases.json").write_text("[]", encoding="utf-8")
    else:
        from tests import eval_ch08_tools as evaluation
        if defect == "missing_answer":
            changed = next(row for row in rows if not row["preview_candidate"])
            changed["models"] = [record for record in changed["models"] if record["stage"] != "answer"]
            changed["answer"] = ""
        else:
            rows[0]["models"][0]["tool_schemas"] = [{"type": "function", "function": {"name": "submit_decision"}}]
        reviews = [{"id": row["id"], "evidence_sha256": evaluation.evidence_hash(row), "reviewer": "Codex (AI)",
                    "verdict": "passed", "reason": "故意错误批准缺正文的记录", "violations": []} for row in rows]
        evidence_file(source, "semantic-review-applied.json", reviews)
    evidence_file(source, "summary.json", summary)
    if defect != "summary_only":
        evidence_file(source, "results.json", rows)
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        module.validate_model_evidence(source, "review-model")


def test_model_import_recomputes_complete_evidence_and_accepts_ai_reviewer(tmp_path):
    """防止新门禁误拒完整证据或要求谎称人类审阅。参数：临时目录；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch08_live")
    assert callable(getattr(module, "validate_model_evidence", None)), "导入模型证据缺少逐项重验门禁"
    result = module.validate_model_evidence(model_report(tmp_path), "review-model")
    assert result["passed"] is True and result["verified_rows"] == 46


@pytest.mark.asyncio
async def test_browser_gate_rejects_only_booleans_and_placeholder(tmp_path):
    """防止八个true加任意文本文件冒充浏览器验收。参数：临时目录；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch08_live")
    assert callable(getattr(module, "validate_browser_evidence", None)), "浏览器证据缺少身份与实际数据库门禁"
    evidence_file(tmp_path, "placeholder.txt", b"not a screenshot")
    value = {"runtime_pid": 123, "checks": dict.fromkeys(["missing_description", "confirm", "cancel", "double_click",
        "refresh_preview", "disconnect_resume", "session_switch", "narrow_screen"], True), "artifacts": ["placeholder.txt"]}
    with pytest.raises((ValueError, AssertionError)):
        await module.validate_browser_evidence(SimpleNamespace(), value, tmp_path, {"run_id": "new"})


@pytest.mark.parametrize("defect", ["wrong_code", "wrong_status", "too_fast", "retried"])
def test_write_timeout_requires_timeout_not_any_failure(defect):
    """防止普通错误或超时机制失效通过慢写验收。参数：损坏点；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch08_live")
    assert callable(getattr(module, "validate_write_timeout", None)), "慢写缺少超时状态核验"
    value = {"result": {"ok": False, "error": {"code": "TOOL_TIMEOUT"}}, "database": {
        "tickets": [], "audits": [{"status": "超时", "retry_count": 0, "duration_ms": 55}]}}
    count = 1
    if defect == "wrong_code":
        value["result"]["error"]["code"] = "TOOL_FAILED"
    elif defect == "wrong_status":
        value["database"]["audits"][0]["status"] = "失败"
    elif defect == "too_fast":
        value["database"]["audits"][0]["duration_ms"] = 0
    else:
        count = 2
    with pytest.raises((ValueError, AssertionError)):
        module.validate_write_timeout(value, count, 0.05)


def test_write_timeout_accepts_clock_resolution_but_rejects_early_failure(monkeypatch):
    """防止Windows粗粒度时钟误拒真实超时，同时拒绝过早结束。

    参数：monkeypatch固定时钟精度为15.625毫秒；返回：None；异常：断言失败。
    """
    module = importlib.import_module("tests.verify_ch08_live")
    monkeypatch.setattr(module.time, "get_clock_info", lambda name: SimpleNamespace(resolution=0.015625))
    value = {"result": {"ok": False, "error": {"code": "TOOL_TIMEOUT"}}, "database": {
        "tickets": [], "audits": [{"status": "超时", "retry_count": 0, "duration_ms": 47}]}}
    module.validate_write_timeout(value, 1, 0.05)
    value["database"]["audits"][0]["duration_ms"] = 20
    with pytest.raises(ValueError):
        module.validate_write_timeout(value, 1, 0.05)


def browser_report(root):
    """手写八类数据库与浏览器观察，外部读库由测试边界代替。

    参数：root独立测试目录；返回：报告、当前运行索引和各会话终态；异常：文件错误。
    """
    context = {"run_id": "current-run", "runtime": {"pid": 123, "registry_id": 456, "graph_id": 789},
               "url": "http://127.0.0.1:8765", "started_at": 1, "snapshots": {}}
    checks, databases = [], {}
    names = ["missing_description", "confirm", "cancel", "double_click", "refresh_preview",
             "disconnect_resume", "session_switch", "narrow_screen"]
    image = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 360, 640) + bytes(9)
    for index, name in enumerate(names, 1):
        session_id, action_id, turn_id, tool_id = "session-" + name, "action-" + name, "turn-" + name, "tool-" + name
        action = {"action_id": action_id, "source_turn_id": turn_id, "tool_call_id": tool_id,
                  "ticket_type": "售后", "description": "订单1001收到时破损"}
        user = {"id": 1, "role": "user", "turn_id": turn_id, "content": "帮我建工单，订单1001收到时破损",
                "rag_metadata": {"pending_ticket": {"action": action, "decision": None}}}
        before = {"session_id": session_id, "user_id": "demo-user", "conversation_id": index,
                  "messages": [user], "tickets": [], "audits": []}
        after = deepcopy(before)
        final = name in {"confirm", "cancel", "double_click", "disconnect_resume", "session_switch"}
        if final:
            decision = "cancel" if name == "cancel" else "confirm"
            after["messages"][0]["rag_metadata"]["pending_ticket"]["decision"] = decision
            after["tickets"] = [] if decision == "cancel" else [{"ticket_no": "TK-" + name,
                "description": action["description"], "ticket_type": action["ticket_type"]}]
            after["audits"] = [{"tool_call_id": tool_id, "tool_name": "create_ticket", "tool_source": "builtin",
                                "status": "权限拒绝" if decision == "cancel" else "成功"}]
            result = ({"ok": False, "error": {"code": "USER_CANCELLED"}} if decision == "cancel"
                      else {"ok": True, "data": {"ticket_no": "TK-" + name}})
            after["messages"].append({"id": 2, "role": "tool", "turn_id": turn_id,
                                       "tool_call_id": tool_id, "content": json.dumps(result)})
        if name == "missing_description":
            before["messages"] = [{"id": 0, "role": "user", "turn_id": "missing-first", "content": "帮我建工单"}]
            after["messages"].insert(0, deepcopy(before["messages"][0]))
        row = {"id": name, "session_id": session_id, "user_id": "demo-user", "source_turn_id": turn_id,
               "action_id": action_id, "tool_call_id": tool_id, "steps": ["执行对应页面操作", "保存观察及原始证据"],
               "screenshots": [evidence_file(root, name + "/before.png", image), evidence_file(root, name + "/after.png", image)]}
        for phase, database, captured_at in (("before", before, 2), ("after", after, 5)):
            reference = evidence_file(root, name + "/" + phase + ".json", {
                "run_id": context["run_id"], "runtime": context["runtime"], "captured_at": captured_at, "database": database})
            row[phase] = reference
            context["snapshots"][reference["path"]] = reference["sha256"]
        body = {"type": "confirm_ticket", "session_id": session_id, "user_id": "demo-user", "source_turn_id": turn_id,
                "action_id": action_id, "decision": "cancel" if name == "cancel" else "confirm"}
        entries = [{"method": "POST", "url": context["url"] + "/api/chat/resume", "request_body": body,
                    "started_at": 3, "ended_at": 4, "status": 200, "response_body": "event: done\ndata: {}\n\n"}]
        if name == "missing_description":
            entries = [{**entries[0], "url": context["url"] + "/api/chat", "request_body": {
                "session_id": session_id, "user_id": "demo-user", "message": text}} for text in
                ("帮我建工单", "订单1001收到时破损")]
        if name in {"refresh_preview", "narrow_screen"}:
            entries = [{**entries[0], "method": "GET", "url": context["url"] +
                        "/api/actions/current?session_id=" + session_id + "&user_id=demo-user", "request_body": None,
                        "response_body": json.dumps({"ticket_confirmation": action})}]
        if name == "disconnect_resume":
            entries.insert(0, {**deepcopy(entries[0]), "started_at": 2.5, "ended_at": 2.8,
                               "disconnected": True, "response_body": ""})
        if name == "double_click":
            row["ui"] = {"click_count": 2}
        if name == "session_switch":
            row["other_session_id"] = "other-session"
            row["ui"] = {"visible_session_id": "other-session", "visible_action_ids": [],
                         "visible_ticket_numbers": [], "switched_at": 3.5}
            databases["other-session"] = {"conversation_id": 99, "tickets": []}
        if name == "narrow_screen":
            row["ui"] = {"viewport_width": 360, "description": action["description"], "buttons": ["确认提交", "取消"]}
        row["network"] = evidence_file(root, name + "/network.json", {"run_id": context["run_id"],
            "session_id": session_id, "user_id": "demo-user", "entries": entries})
        databases[session_id] = after
        checks.append(row)
    return {"schema_version": 1, "run_id": context["run_id"], "runtime": context["runtime"], "checks": checks}, context, databases


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", [None, "old_run", "missing_case", "modified_file", "foreign_snapshot", "snapshot_user",
    "same_snapshot", "empty_network", "wrong_action", "duplicate_ticket", "missing_denial_audit", "new_user_on_resume",
    "missing_disconnect", "late_result_in_other_session", "text_screenshot", "new_user_after_capture"])
async def test_browser_gate_checks_original_artifacts_and_current_database(tmp_path, monkeypatch, defect):
    """防止单纯本地记录通过而真实库错误，保留完整证据成功路径。参数：目录/读库边界/损坏点；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch08_live")
    value, context, databases = browser_report(tmp_path)

    async def read(runtime, session_id, user_id):
        """仅代替外部MySQL读取；参数：同一运行时及精确身份；返回：本会话真实形状；异常：未知会话。"""
        assert user_id == "demo-user"
        return deepcopy(databases[session_id])

    monkeypatch.setattr(module, "browser_database", read)
    by_name = {row["id"]: row for row in value["checks"]}
    if defect == "old_run":
        value["run_id"] = "previous-run"
    elif defect == "missing_case":
        value["checks"].pop()
    elif defect == "modified_file":
        (tmp_path / by_name["confirm"]["network"]["path"]).write_text("{}")
    elif defect == "foreign_snapshot":
        context["snapshots"].pop(by_name["confirm"]["before"]["path"])
    elif defect == "snapshot_user":
        reference = by_name["confirm"]["before"]
        data = json.loads((tmp_path / reference["path"]).read_text())
        data["database"]["user_id"] = "other-user"
        by_name["confirm"]["before"] = evidence_file(tmp_path, reference["path"], data)
        context["snapshots"][reference["path"]] = by_name["confirm"]["before"]["sha256"]
    elif defect == "same_snapshot":
        by_name["refresh_preview"]["after"] = by_name["refresh_preview"]["before"]
    elif defect in {"empty_network", "wrong_action", "missing_disconnect"}:
        row = by_name["disconnect_resume" if defect == "missing_disconnect" else "confirm"]
        network = json.loads((tmp_path / row["network"]["path"]).read_text())
        if defect == "empty_network":
            network["entries"] = []
        elif defect == "wrong_action":
            network["entries"][0]["request_body"]["action_id"] = "another-action"
        else:
            network["entries"] = network["entries"][1:]
        row["network"] = evidence_file(tmp_path, row["network"]["path"], network)
    elif defect == "duplicate_ticket":
        databases["session-double_click"]["tickets"].append({"ticket_no": "extra-ticket"})
    elif defect == "missing_denial_audit":
        databases["session-cancel"]["audits"] = []
    elif defect == "new_user_on_resume":
        row = by_name["confirm"]
        saved = json.loads((tmp_path / row["after"]["path"]).read_text())
        saved["database"]["messages"].append({"id": 4, "role": "user", "turn_id": "bad-new", "content": "确认"})
        row["after"] = evidence_file(tmp_path, row["after"]["path"], saved)
        context["snapshots"][row["after"]["path"]] = row["after"]["sha256"]
    elif defect == "new_user_after_capture":
        databases["session-confirm"]["messages"].append({"id": 4, "role": "user", "turn_id": "bad-late", "content": "确认"})
    elif defect == "late_result_in_other_session":
        by_name["session_switch"]["ui"]["visible_ticket_numbers"] = ["TK-session_switch"]
    elif defect == "text_screenshot":
        row = by_name["narrow_screen"]
        row["screenshots"][0] = evidence_file(tmp_path, row["screenshots"][0]["path"], b"placeholder")
    if defect is None:
        result = await module.validate_browser_evidence(SimpleNamespace(), value, tmp_path, context)
        assert result["passed"] and len(result["checks"]) == 8
    else:
        with pytest.raises(ValueError):
            await module.validate_browser_evidence(SimpleNamespace(), value, tmp_path, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("identity,accepted", [(None, True), ("demo-user", True), ("other-user", False)])
async def test_browser_chat_identity_matches_api_default(tmp_path, monkeypatch, identity, accepted):
    """验证普通聊天省略演示用户字段仍是合法原始请求，显式异用户仍拒绝。

    参数：临时目录、数据库替身、请求用户及通过预期；返回：None；异常：断言失败。
    """
    module = importlib.import_module("tests.verify_ch08_live")
    value, context, databases = browser_report(tmp_path)

    async def read(runtime, session_id, user_id):
        """提供对应会话的库形状。参数：运行时、会话及用户；返回：独立快照；异常：未知会话。"""
        return deepcopy(databases[session_id])

    monkeypatch.setattr(module, "browser_database", read)
    row = next(row for row in value["checks"] if row["id"] == "missing_description")
    network = json.loads((tmp_path / row["network"]["path"]).read_text())
    for entry in network["entries"]:
        if identity is None:
            entry["request_body"].pop("user_id")
        else:
            entry["request_body"]["user_id"] = identity
    row["network"] = evidence_file(tmp_path, row["network"]["path"], network)
    if accepted:
        assert (await module.validate_browser_evidence(SimpleNamespace(), value, tmp_path, context))["passed"]
    else:
        with pytest.raises(ValueError):
            await module.validate_browser_evidence(SimpleNamespace(), value, tmp_path, context)


@pytest.mark.asyncio
async def test_snapshot_capture_is_owned_and_never_accepts_preexisting_output(tmp_path, monkeypatch):
    """防止根目录外请求或预造快照进入可信索引。参数：目录/只读数据库边界；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch08_live")
    context = {"run_id": "run", "runtime": {"pid": 1}, "snapshots": {}}

    async def read(runtime, session_id, user_id):
        """替代外部读取。参数：可信身份；返回：对应快照；异常：无。"""
        return {"session_id": session_id, "user_id": user_id, "messages": [], "tickets": [], "audits": []}

    monkeypatch.setattr(module, "browser_database", read)
    evidence_file(tmp_path, "browser-snapshot-requests/first.json", {"session_id": "s", "user_id": "demo-user"})
    await module.capture_browser_requests(SimpleNamespace(), tmp_path, context)
    assert "browser-snapshots/first.json" in context["snapshots"]
    evidence_file(tmp_path, "browser-snapshot-requests/prebuilt.json", {"session_id": "s", "user_id": "demo-user"})
    evidence_file(tmp_path, "browser-snapshots/prebuilt.json", {"fake": True})
    with pytest.raises(ValueError, match="已存在"):
        await module.capture_browser_requests(SimpleNamespace(), tmp_path, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("database,user", [("mewhelp", "demo-user"), ("mewhelp_test", "other-user")])
async def test_browser_database_refuses_wrong_actual_database_or_owner(database, user):
    """防止采集器误读应用库或他人会话。参数：真实连接返回库名/会话归属；返回：None；异常：断言失败。"""
    from contextlib import asynccontextmanager
    module = importlib.import_module("tests.verify_ch08_live")

    class Connection:
        """只代替SQL执行边界，保留采集器的实际库名和用户校验。"""
        async def scalar(self, query):
            """返回连接库名或查询会话。参数：SQL；返回：对应外部数据；异常：无。"""
            return database if str(query) == "SELECT DATABASE()" else SimpleNamespace(id=1, user_id=user)

    @asynccontextmanager
    async def factory():
        """提供独占只读连接。参数：无；返回：外部执行边界；异常：无。"""
        yield Connection()

    with pytest.raises(ValueError):
        await module.browser_database(SimpleNamespace(factory=factory), "session", "demo-user")
