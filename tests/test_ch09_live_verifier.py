"""CH09验收器的拒伪证据测试：真实校验逻辑不能相信摘要、串轮或被改写用量。"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def verifier():
    """取得被测验收器。参数：无；返回：模块；异常：功能缺失时断言失败。"""
    spec = importlib.util.find_spec("tests.verify_ch09_live")
    assert spec is not None, "缺少CH09只读真实证据验收器"
    from tests import verify_ch09_live
    return verify_ch09_live


def sample():
    """构造一轮可手算的完整数据库和观测协议。参数：无；返回：身份/状态/观测；异常：无。"""
    identity = {"id": "knowledge", "session_id": "s", "user_id": "u", "turn_id": "t", "trace_id": "trace"}
    context = {"session_id": "s", "user_id": "u", "turn_id": "t", "conversation_id": 1, "user_message": "问题"}
    trace = {**{key: identity[key] for key in ("session_id", "user_id", "turn_id", "trace_id")}, "root_observation_id": "root"}
    metadata = {"trace_identity": trace, "retrieval_snapshot": None, "knowledge_pool": {"pool_ids": {}}}
    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    state = {"workflow": {"context": context, "trace_identity": trace, "turn_status": "completed",
        "answer": "答复", "retrieval_snapshot": None, "completion_metadata": {"workflow": {"intent": "商品咨询",
        "budget": {"model_calls": 1, "actual_tokens": 12, "records": [{"stage": "answer", "usage": usage}]}}}},
        "messages": [{"id": 1, "role": "user", "turn_id": "t", "content": "问题", "turn_status": "completed", "tool_calls": None, "rag_metadata": metadata},
                     {"id": 2, "role": "assistant", "turn_id": "t", "content": "答复", "turn_status": "completed", "tool_calls": None, "rag_metadata": {}}], "pool": []}
    observations = [{"id": "root", "traceId": "trace", "type": "SPAN", "name": "execution.chat",
        "startTime": "2026-10-31T00:00:00Z", "endTime": "2026-10-31T00:00:03Z",
        "metadata": {"turn_id": "t", "intent": "商品咨询", "task_kind": "chat"}},
        {"id": "gen", "traceId": "trace", "type": "GENERATION", "name": "ChatOpenAI", "parentObservationId": "root",
         "startTime": "2026-10-31T00:00:01Z", "endTime": "2026-10-31T00:00:02Z",
         "metadata": {"turn_id": "t", "intent": "unclassified", "task_kind": "chat", "usage_origin": "model_reported",
         "input_usage_present": True, "output_usage_present": True, "reported_input_tokens": 10, "reported_output_tokens": 2}}]
    return identity, state, observations


def test_actual_turn_and_raw_usage_are_reconciled():
    """丢失分类前调用或把调用数当请求数会失败。参数：无；返回：None；异常：断言失败。"""
    identity, state, observations = sample()
    result = verifier().verify_turn(identity, state, observations)
    assert result["model_call_count"] == 1 and result["known_tokens"] == 12


@pytest.mark.parametrize("damage", ["turn", "session", "trace", "final_answer", "budget", "input", "bool_token", "missing_generation", "duplicate_conflict", "open_root"])
def test_turn_rejects_wrong_identity_or_forged_numbers(damage):
    """串轮、删调用、伪数字或未结束根都不能过关。参数：破坏类型；返回：None；异常：断言失败。"""
    identity, state, observations = sample()
    if damage in {"turn", "session"}:
        state["workflow"]["context"][f"{damage}_id"] = "wrong"
    elif damage == "trace":
        observations[-1]["traceId"] = "another"
    elif damage == "final_answer":
        state["messages"][-1]["content"] = "伪报答复"
    elif damage == "budget":
        state["workflow"]["completion_metadata"]["workflow"]["budget"]["model_calls"] = 9
    elif damage in {"input", "bool_token"}:
        observations[-1]["metadata"]["reported_input_tokens"] = True if damage == "bool_token" else 999
    elif damage == "missing_generation":
        observations.pop()
    elif damage == "duplicate_conflict":
        duplicate = copy.deepcopy(observations[-1])
        duplicate["metadata"]["reported_output_tokens"] = 99
        observations.append(duplicate)
    else:
        observations[0]["endTime"] = None
    with pytest.raises(ValueError):
        verifier().verify_turn(identity, state, observations)


@pytest.mark.parametrize("damage", ["added_id", "changed_usage", "changed_origin", "wrong_trace", "late_export", "before_empty"])
def test_replay_requires_same_ids_raw_usage_and_time(damage):
    """只比较次数/总数或忽略迟到导出，会把真实新增消费误判成零。参数：破坏类型；返回：None；异常：断言失败。"""
    identity, _, rows = sample()
    before, after, current = copy.deepcopy(rows), copy.deepcopy(rows), copy.deepcopy(rows)
    timing = {"started": "2026-10-31T00:00:04Z", "ended": "2026-10-31T00:00:05Z"}
    if damage == "added_id":
        after[-1]["id"] = "new"
    elif damage == "changed_usage":
        after[-1]["metadata"]["reported_input_tokens"] = 11
    elif damage == "changed_origin":
        after[-1]["metadata"]["usage_origin"] = "inferred_or_unknown"
    elif damage == "wrong_trace":
        before[-1]["traceId"] = "other"
    elif damage == "late_export":
        timing["started"] = "2026-10-31T00:00:01Z"
    else:
        before = []
    with pytest.raises(ValueError):
        verifier().verify_replay(identity, before, after, current, timing)


def test_replay_validates_zero_new_consumption():
    """完整稳定前后原件应得出零新增。参数：无；返回：None；异常：断言失败。"""
    identity, _, rows = sample()
    result = verifier().verify_replay(identity, rows, rows, rows,
        {"started": "2026-10-31T00:00:04Z", "ended": "2026-10-31T00:00:05Z"})
    assert result == {"generation_count": 1, "new_model_calls": 0, "new_known_tokens": 0}


@pytest.mark.parametrize("damage", ["source", "question", "conversation", "snapshot", "missing"])
def test_pool_cannot_borrow_another_turn_or_snapshot(damage):
    """只检查池行存在会漏过串轮和历史伪造。参数：破坏类型；返回：None；异常：断言失败。"""
    _, state, _ = sample()
    state["messages"][0]["rag_metadata"]["knowledge_pool"]["pool_ids"] = {"self_check": 7}
    row = {"id": 7, "conversation_id": 1, "raw_question": "问题", "source": "self_check", "retrieved_chunks": None}
    state["pool"] = [row]
    if damage == "source":
        row["source"] = "user_feedback"
    elif damage == "question":
        row["raw_question"] = "另一个问题"
    elif damage == "conversation":
        row["conversation_id"] = 9
    elif damage == "snapshot":
        row["retrieved_chunks"] = {"schema_version": 1, "status": "empty", "attempts": []}
    else:
        state["pool"] = []
    with pytest.raises(ValueError):
        verifier().verify_pool(state, "self_check")


def test_reference_rejects_changed_file_and_directory_escape(tmp_path):
    """文件存在不足以当证据，必须验SHA和许可路径。参数：目录；返回：None；异常：断言失败。"""
    target = tmp_path / "proof.json"
    target.write_text('{"number":1}', encoding="utf-8")
    reference = {"path": "proof.json", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    assert verifier().read_reference(reference, tmp_path, tmp_path) == {"number": 1}
    target.write_text('{"number":9}', encoding="utf-8")
    with pytest.raises(ValueError):
        verifier().read_reference(reference, tmp_path, tmp_path)
    reference["path"] = "../outside.json"
    with pytest.raises(ValueError):
        verifier().read_reference(reference, tmp_path, tmp_path)


def test_missing_cases_or_manual_claim_cannot_pass():
    """自报passed与截图存在不等于完整机器/人工核验。参数：无；返回：None；异常：断言失败。"""
    report = verifier().finish_report({"cases": [{"id": "knowledge_loop", "status": "passed"}], "manual": {"status": "passed"}})
    assert report["passed"] is False and report["pending"]
    assert report["manual"]["machine_verified"] is False


def test_eval_rejects_forged_summary_before_database_comparison(tmp_path):
    """不能绕过Task9完整产物验证只信DB和摘要相等。参数：目录；返回：None；异常：断言失败。"""
    from tests.flywheel.test_eval_runs import completed_artifact
    path = completed_artifact(tmp_path / "first")
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["faithfulness"] = 1.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        verifier().verify_evaluation_pair([path, path], [{"id": "1", "dataset_size": 48, "metrics": summary}] * 2)


def loop_sample():
    """创建可手算发布闭环，快照来自同一已知来源。参数：无；返回：六个证据参数；异常：无。"""
    from app.knowledge.answering import EvidenceCitation, FIXED_KNOWLEDGE_REFUSAL
    from app.knowledge.schemas import KnowledgeText
    from app.knowledge.sources import answer_metadata
    from app.knowledge.text import content_fingerprint
    from app.schemas.tools import ToolContext
    _, before, _ = sample()
    after = copy.deepcopy(before)
    before["workflow"]["answer"] = FIXED_KNOWLEDGE_REFUSAL
    old_snapshot = {"schema_version": 1, "status": "empty", "attempts": [{"attempt_id": "old", "query": "问题", "status": "empty", "candidate_count": 0, "top_k": 5, "chunks": []}]}
    before["workflow"]["retrieval_snapshot"] = old_snapshot
    before["messages"][0]["rag_metadata"]["knowledge_pool"]["pool_ids"] = {"retrieval_low_conf": 7}
    before["pool"] = [{"id": 7, "conversation_id": 1, "source": "retrieval_low_conf", "raw_question": "问题", "retrieved_chunks": old_snapshot, "matched_review_id": 8}]
    text = KnowledgeText(category="商品与购物 FAQ", questions="问题", answer="核准事实")
    chunk = {"id": 9, "source_key": "ch09/review/8", "category": text.category, "questions": "问题", "answer": text.answer,
             "content_type": "faq", "content_hash": content_fingerprint(text), "embedding_revision": "emb", "vectorize_status": "done", "vector_id": "9", "is_active": True}
    # 合法引用必须走生产合同：公开指纹只覆盖答案，知识/向量/检索快照覆盖完整三格。
    after["workflow"]["completion_metadata"].update(answer_metadata(
        ToolContext(**after["workflow"]["context"]),
        [EvidenceCitation(n=1, chunk_id=9, source_key=chunk["source_key"], question="问题", answer=chunk["answer"])]))
    after["workflow"]["retrieval_snapshot"] = {"schema_version": 1, "status": "ok", "attempts": [{"attempt_id": "a", "query": "问题", "status": "ok", "candidate_count": 1, "top_k": 5,
        "chunks": [{"chunk_id": 9, "source_key": chunk["source_key"], "content_hash": chunk["content_hash"], "embedding_revision": "emb", "question": "问题", "answer": "核准事实", "rank": 1}]}]}
    review = {"id": "8", "review_status": "通过", "publication_status": "published", "approved_answer": "核准事实", "normalized_question": "问题"}
    return before, after, review, [chunk], {9: (chunk["content_hash"], "emb")}, {"review_id": 8, "answer": "核准事实"}


@pytest.mark.parametrize("damage", [None, "question", "approval", "vector", "citation", "pool", "pending", "missing_snapshot"])
def test_knowledge_loop_checks_actual_source_and_publication(damage):
    """不同原问、核准正文、向量、引用或未发布都不能拼成闭环。参数：破坏类型；返回：None；异常：断言失败。"""
    args = loop_sample()
    before, after, review, chunks, fingerprints, approved = args
    if damage == "question":
        after["workflow"]["context"]["user_message"] = "其他问题"
    elif damage == "approval":
        approved["answer"] = "模型猜测"
    elif damage == "vector":
        fingerprints.clear()
    elif damage == "citation":
        after["workflow"]["completion_metadata"]["citations"][0]["source_key"] = "doc:another"
    elif damage == "pool":
        before["pool"][0]["matched_review_id"] = 99
    elif damage == "pending":
        review["publication_status"] = "pending"
    elif damage == "missing_snapshot":
        before["workflow"]["retrieval_snapshot"] = None
        before["pool"][0]["retrieved_chunks"] = None
    if damage:
        with pytest.raises(ValueError):
            verifier().verify_knowledge_loop(*args)
    else:
        assert verifier().verify_knowledge_loop(*args)["chunk_id"] == 9


@pytest.mark.parametrize("target,field,value,message", [
    ("chunk", "content_hash", "f" * 64, "MySQL或强一致向量指纹不符"),
    ("vector", "content_hash", "f" * 64, "MySQL或强一致向量指纹不符"),
    ("vector", "embedding_revision", "another", "MySQL或强一致向量指纹不符"),
    ("citation", "content_hash", "f" * 64, "发布后答案没有引用本条核准事实"),
    ("citation", "content_hash", "364c834e4c5eef9671172b9ad5905bac50769b3776b72dc045f097acc647ab83", "发布后答案没有引用本条核准事实"),
    ("citation", "chunk_id", 10, "发布后答案没有引用本条核准事实"),
    ("citation", "answer", "另一条事实", "发布后答案没有引用本条核准事实"),
    ("snapshot", "content_hash", "f" * 64, "发布后原轮快照未记录核准来源"),
    ("snapshot", "content_hash", "c286e99465ef23045ea3b0659ae037cfc1bedf6619867eecd58f0b3063514d34", "发布后原轮快照未记录核准来源"),
    ("snapshot", "chunk_id", 10, "发布后原轮快照未记录核准来源"),
    ("snapshot", "source_key", "ch09/review/99", "发布后原轮快照未记录核准来源"),
    ("snapshot", "answer", "另一条事实", "发布后原轮快照未记录核准来源"),
])
def test_knowledge_loop_rejects_damaged_fingerprints_and_evidence_identity(target, field, value, message):
    """两类指纹互换、单独损坏或来源身份串用都必须在对应边界拒绝。

    参数：target 为知识/向量/公开引用/检索快照；field 为被损坏字段；value 为非法替换值；message 为预期拒绝边界。
    返回：None；异常：AssertionError，坏证据通过或在错误边界失败。
    """
    args = loop_sample()
    _, after, _, chunks, fingerprints, _ = args
    targets = {"chunk": chunks[0], "citation": after["workflow"]["completion_metadata"]["citations"][0],
               "snapshot": after["workflow"]["retrieval_snapshot"]["attempts"][0]["chunks"][0]}
    if target == "vector":
        content_hash, revision = fingerprints[9]
        fingerprints[9] = (value, revision) if field == "content_hash" else (content_hash, value)
    else:
        targets[target][field] = value
    with pytest.raises(ValueError, match=message):
        verifier().verify_knowledge_loop(*args)


def test_feedback_rejects_repeated_new_pool_id():
    """重复反馈另建池行不能被摘要掩盖。参数：无；返回：None；异常：断言失败。"""
    _, before, _ = sample()
    after = copy.deepcopy(before)
    after["messages"][0]["rag_metadata"]["knowledge_pool"]["pool_ids"] = {"user_feedback": 7}
    after["pool"] = [{"id": 7, "conversation_id": 1, "source": "user_feedback", "raw_question": "问题", "retrieved_chunks": None}]
    repeat = copy.deepcopy(after)
    before["captured_at"], after["captured_at"], repeat["captured_at"] = [f"2026-10-31T00:00:0{n}Z" for n in (1, 2, 3)]
    for row in (before, after, repeat):
        row["review_counts"] = {}
    repeat["messages"][0]["rag_metadata"]["knowledge_pool"]["pool_ids"]["user_feedback"] = 8
    with pytest.raises(ValueError):
        verifier().verify_feedback(before, after, repeat, after)


@pytest.mark.parametrize("damage", [None, "serial", "same_trace"])
def test_concurrency_requires_distinct_overlapping_roots(damage):
    """两条串行或同trace记录不等于并发。参数：破坏类型；返回：None；异常：断言失败。"""
    _, _, first = sample()
    second = copy.deepcopy(first)
    second[0]["traceId"] = "second"
    second[0]["startTime"] = "2026-10-31T00:00:02Z"
    second[0]["endTime"] = "2026-10-31T00:00:04Z"
    if damage == "serial":
        second[0]["startTime"] = "2026-10-31T00:00:03Z"
    elif damage == "same_trace":
        second[0]["traceId"] = "trace"
    if damage:
        with pytest.raises(ValueError):
            verifier().verify_concurrency([first, second])
    else:
        assert verifier().verify_concurrency([first, second])["overlap"] is True


def test_missing_manifest_cli_records_pending_and_nonzero(tmp_path):
    """无清单时不能静默成功或调用付费依赖。参数：输出目录；返回：None；异常：断言失败。"""
    code = verifier().main(["--base-url", "http://127.0.0.1:18000", "--output-dir", str(tmp_path / "run")])
    assert code == 1
    report = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and "knowledge_loop" in report["pending"]


async def test_observation_reader_accepts_current_server_raw_io(tmp_path, monkeypatch):
    """当前服务端拒绝parseIoAsJson=true，本地解析raw IO才能完成只读分页。参数：目录/替换HTTP传输；返回：None；异常：断言失败。"""
    import httpx
    from app.config import Settings
    identity, _, rows = sample()
    rows[-1]["output"] = json.dumps({"tool_calls": [{"name": "submit_knowledge_answer", "args": {"useful": False}}]})
    original = httpx.AsyncClient

    def server(request):
        """重现服务端已核实协议。参数：HTTP请求；返回：响应；异常：无。"""
        if request.url.params.get("parseIoAsJson") == "true":
            return httpx.Response(400, json={"message": "parseIoAsJson=true is no longer supported"})
        return httpx.Response(200, json={"data": rows, "meta": {"cursor": None}})

    def client(**kwargs):
        """只替换外部HTTP传输。参数：正常客户端配置；返回：httpx客户端；异常：配置错误。"""
        return original(**kwargs, transport=httpx.MockTransport(server))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    config = Settings(mysql_password="test", langfuse_public_key="public", langfuse_secret_key="secret")
    reader = verifier().LiveReader(config, "http://127.0.0.1:18000", tmp_path)
    try:
        actual = await reader.observations(identity)
        assert verifier().knowledge_decisions(actual[-1]["output"]) == [{"useful": False}]
    finally:
        await reader.close()


@pytest.mark.parametrize("damage", [None, "success", "unrelated"])
def test_structured_tool_fault_is_evidence_even_with_default_level(damage):
    """真实工具故障可为DEFAULT，但必须与本轮实库故障回执相符。参数：破坏类型；返回：None；异常：断言失败。"""
    _, state, rows = sample()
    failure = {"ok": False, "data": None, "error": {"code": "TOOL_FAILED", "message": "失败", "category": "fault"}, "is_mock": False}
    state["messages"].append({"id": 3, "role": "tool", "turn_id": "t", "content": json.dumps(failure), "tool_call_id": "call"})
    rows.append({"id": "tool", "traceId": "trace", "type": "TOOL", "name": "tool.execute", "level": "DEFAULT",
        "metadata": {"result_ok": False}, "output": json.dumps(failure)})
    if damage == "success":
        rows[-1]["output"] = json.dumps({"ok": True, "data": {"evidence_count": 0}, "error": None})
    elif damage == "unrelated":
        state["messages"].pop()
    if damage:
        with pytest.raises(ValueError):
            verifier().verify_error(state, rows)
    else:
        assert verifier().verify_error(state, rows)["observations"] == ["tool"]


async def test_partial_observation_failure_keeps_received_pages(tmp_path, monkeypatch):
    """中间HTTP失败仍须保留已收到页，不能输出完成聚合件。参数：目录/HTTP替换；返回：None；异常：断言失败。"""
    import httpx
    from app.config import Settings
    identity, _, rows = sample()
    original = httpx.AsyncClient

    def server(request):
        """模拟第二页失败。参数：请求；返回：响应；异常：无。"""
        if request.url.params.get("cursor"):
            return httpx.Response(503, json={"message": "unavailable"})
        return httpx.Response(200, json={"data": rows, "meta": {"cursor": "next"}})

    def client(**kwargs):
        """仅替换远程传输。参数：配置；返回：客户端；异常：无。"""
        return original(**kwargs, transport=httpx.MockTransport(server))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    reader = verifier().LiveReader(Settings(mysql_password="test"), "http://127.0.0.1:18000", tmp_path)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await reader.observations(identity)
        assert json.loads((tmp_path / "trace-knowledge-page-1.json").read_text(encoding="utf-8"))["data"][0]["id"] == "root"
        assert not (tmp_path / "trace-knowledge.json").exists()
    finally:
        await reader.close()


@pytest.mark.parametrize("damage", ["root_turn", "source_trace", "answer_turn"])
def test_turn_requires_message_and_root_identity_not_only_summary(damage):
    """工作流摘要正确时，底层消息或根串轮仍必须拒绝。参数：破坏类型；返回：None；异常：断言失败。"""
    identity, state, rows = sample()
    if damage == "root_turn":
        rows[0]["metadata"]["turn_id"] = "other"
    elif damage == "source_trace":
        state["messages"][0]["rag_metadata"]["trace_identity"] = {"trace_id": "other"}
    else:
        state["messages"][-1]["turn_id"] = "other"
    with pytest.raises(ValueError):
        verifier().verify_turn(identity, state, rows)


@pytest.mark.parametrize("inputs,outputs,want", [(0, 0, (0, 0)), (None, 0, (None, 0)), (3, None, (3, None)), (None, None, (None, None))])
def test_usage_distinguishes_real_zero_from_missing(inputs, outputs, want):
    """缺失方向不可被转换成已知零。参数：原始方向/手算结果；返回：None；异常：断言失败。"""
    row = {"metadata": {"usage_origin": "model_reported" if inputs is not None and outputs is not None else "inferred_or_unknown",
        "input_usage_present": inputs is not None, "output_usage_present": outputs is not None,
        "reported_input_tokens": inputs, "reported_output_tokens": outputs}}
    assert verifier().raw_usage(row) == want


@pytest.mark.parametrize("damage", [None, "db_number", "db_size", "same_id"])
def test_eval_pair_rechecks_each_artifact_and_database(tmp_path, damage):
    """完整两轮可复核，伪造DB数字或复用身份必须失败。参数：目录/破坏类型；返回：None；异常：断言失败。"""
    from app.evaluation.runner import validate_completed_artifact, production_summary
    from app.evaluation.policy import digest_json
    from tests.flywheel.test_eval_runs import completed_artifact
    first, second = [completed_artifact(tmp_path / name) for name in ("first", "second")]
    manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (second / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["artifact_id"] = "second-artifact"
    summary = production_summary(rows, manifest["metadata"], "second-artifact")
    manifest.update(artifact_id="second-artifact", summary_digest=digest_json(summary), results_digest=digest_json(rows))
    (second / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for name, value in (("manifest.json", manifest), ("summary.json", summary)):
        (second / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    database = [{"id": str(number), "dataset_size": 48, "metrics": validate_completed_artifact(path)} for number, path in enumerate((first, second), 1)]
    if damage == "db_number":
        database[1]["metrics"]["faithfulness"] = 0.99
    elif damage == "db_size":
        database[1]["dataset_size"] = 72
    elif damage == "same_id":
        database[1]["id"] = "1"
    if damage:
        with pytest.raises(ValueError):
            verifier().verify_evaluation_pair([first, second], database)
    else:
        actual = verifier().verify_evaluation_pair([first, second], database)
        assert actual["artifacts"] == ["test-artifact", "second-artifact"]
        assert actual["error_counts"] == [48, 48] and actual["differences"]["faithfulness"] is None
