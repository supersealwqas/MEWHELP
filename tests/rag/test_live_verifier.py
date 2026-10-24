"""真实验收断言器的离线反例：防止缺事件、坏快照和错误入池被算通过。"""

from copy import deepcopy
import hashlib

import pytest

from tests.verify_ch04_live import inspect_exchange


def _observation() -> tuple[dict, dict]:
    """手工给定最小问候证据。参数：无；返回：案例和采样；异常：无。"""
    metadata = {"session_id": "live-one", "turn_id": "turn-one", "status": "completed", "citations": []}
    return {"id": "greeting", "query": "你好", "kind": "greeting", "tools": []}, {
        "session_id": "live-one", "user_id": "demo-user", "http_status": 200,
        "content_type": "text/event-stream; charset=utf-8", "failures": [],
        "events": [{"event": "delta", "data": {"delta": "您好。"}},
                   {"event": "answer_metadata", "data": deepcopy(metadata)}, {"event": "done", "data": "[DONE]"}],
        "messages": [
            {"id": 1, "role": "user", "content": "你好", "turn_id": "turn-one", "turn_status": "completed"},
            {"id": 2, "role": "assistant", "content": "您好。", "turn_id": "turn-one", "turn_status": "completed",
             "rag_metadata": deepcopy(metadata)}],
        "model_checks": [], "pool_rows": [], "tickets": [], "citation_responses": [], "access_checks": [],
        "tool_max_calls_per_turn": 4,
    }


def _knowledge_observation() -> tuple[dict, dict]:
    """手工构造两个模型证据且只引用第一条。参数：无；返回：案例和采样；异常：无。"""
    case, observed = _observation()
    case.update(id="knowledge", kind="knowledge", tools=["query_faq"])
    body = "# 手册\n\n## 保修\n\n保修一年。\n\n## 寄修\n\n联系客服寄修。\n"
    revision = hashlib.sha256(body.encode()).hexdigest()
    citations = [{"n": 1, "chunk_id": 10, "question": "保修", "answer": "保修一年。", "section_path": "手册 / 保修",
                  "source_key": "doc:manual.md", "source_revision": revision, "source_type": "document",
                  "content_hash": hashlib.sha256("保修一年。".encode()).hexdigest()},
                 {"n": 2, "chunk_id": 11, "question": "寄修", "answer": "联系客服寄修。", "section_path": "手册 / 寄修",
                  "source_key": "doc:manual.md", "source_revision": revision, "source_type": "document",
                  "content_hash": hashlib.sha256("联系客服寄修。".encode()).hexdigest()}]
    for metadata in (observed["events"][1]["data"], observed["messages"][-1]["rag_metadata"]):
        metadata["citations"] = deepcopy(citations)
    observed["events"][0]["data"]["delta"] = observed["messages"][-1]["content"] = "保修一年。[1]"
    observed["messages"][1:1] = [
        {"id": 3, "role": "assistant", "content": "", "turn_id": "turn-one", "turn_status": "completed",
         "tool_calls": [{"id": "call-one", "name": "query_faq", "args": {"keyword": "你好"}}]},
        {"id": 4, "role": "tool", "content": '{"ok":true,"is_mock":false,"data":{"hits":[]},"error":null}',
         "turn_id": "turn-one", "turn_status": "completed", "tool_call_id": "call-one"}]
    observed["events"].insert(0, {"event": "tool_status", "data": {
        "tool_call_id": "call-one", "name": "query_faq", "status": "succeeded", "is_mock": False}})
    observed["model_checks"] = [{"decision": {"useful": True, "reason": "", "answer": "保修一年。[1]", "citation_numbers": [1]},
        "evidence": [{key: value for key, value in citation.items() if key not in {"content_hash", "source_type"}}
                     for citation in citations]}]
    observed["citation_responses"] = [{"n": citation["n"], "http_status": 200, "snapshot": deepcopy(citation),
        "source_status": 200, "source": {"content": body, "source_revision": revision, "source_changed": False,
            "section_path": citation["section_path"], "section_line": 3 if citation["n"] == 1 else 7},
        "expected_source": {"content": body, "revision": revision, "section_line": 3 if citation["n"] == 1 else 7}}
        for citation in citations]
    observed["access_checks"] = [{"kind": "other_user", "http_status": 404},
                                {"kind": "other_user_source", "http_status": 404}]
    return case, observed


def test_completed_turn_requires_metadata_even_when_body_and_done_exist() -> None:
    """缺失metadata不能被完整文本及DONE掩盖。参数：无；返回：None；异常：断言失败。"""
    case, observed = _observation()
    assert inspect_exchange(case, observed)["failures"] == []
    observed["events"].pop(1)
    assert "missing_or_misordered_metadata" in inspect_exchange(case, observed)["failures"]


@pytest.mark.parametrize("field,replacement,expected", [
    ("content", "另一份回答", "persisted_answer_mismatch"),
    ("rag_metadata", {"status": "completed", "citations": []}, "persisted_metadata_mismatch"),
])
def test_database_body_and_metadata_must_match_stream(field, replacement, expected) -> None:
    """数据库正文和JSON分别核验，不能只核对状态。参数：变异字段/值/错误；返回：None；异常：断言失败。"""
    case, observed = _observation()
    observed["messages"][-1][field] = replacement
    assert expected in inspect_exchange(case, observed)["failures"]


@pytest.mark.parametrize("mutation,expected", [
    ("snapshot", "citation_snapshot_mismatch"), ("source", "source_mismatch"),
    ("unused_evidence", "model_evidence_mismatch"), ("access", "citation_access_not_404"),
])
def test_citations_compare_api_full_source_and_all_model_evidence(mutation, expected) -> None:
    """快照、全文、未引用证据或跨用户泄露任一损坏均须失败。参数：变异/错误；返回：None；异常：断言失败。"""
    case, observed = _knowledge_observation()
    assert inspect_exchange(case, observed)["failures"] == []
    if mutation == "snapshot":
        observed["citation_responses"][0]["snapshot"]["answer"] = "修改后的正文"
    elif mutation == "source":
        observed["citation_responses"][0]["source"]["content"] = "另一份文档"
    elif mutation == "unused_evidence":
        observed["model_checks"][0]["evidence"].append({"n": 3, "chunk_id": 12, "question": "遗漏", "answer": "未保存证据"})
    else:
        observed["access_checks"][0]["http_status"] = 200
    assert expected in inspect_exchange(case, observed)["failures"]


def test_pool_requires_real_useful_false_and_exact_question_reason() -> None:
    """仅检索低分或普通回复入池应失败，自评false必须对应原话和理由。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL

    case, observed = _knowledge_observation()
    observed["pool_rows"] = [{"raw_question": "你好", "source": "self_check", "reason": "证据不足"}]
    assert "pool_without_useful_false" in inspect_exchange(case, observed)["failures"]
    observed["model_checks"][0]["decision"] = {"useful": False, "reason": "证据不足"}
    next(event for event in observed["events"] if event["event"] == "delta")["data"]["delta"] = FIXED_KNOWLEDGE_REFUSAL
    observed["messages"][-1]["content"] = FIXED_KNOWLEDGE_REFUSAL
    case["kind"] = "unknown_model"
    assert inspect_exchange(case, observed)["failures"] == []
    observed["pool_rows"][0]["raw_question"] = "改写原话"
    assert "self_check_pool_mismatch" in inspect_exchange(case, observed)["failures"]


def test_resource_guard_and_explicit_cli_override_env(tmp_path, monkeypatch) -> None:
    """显式端口/模型路径须覆盖env，任何半边应用资源配置均拒绝。参数：临时目录/环境替换；返回：None；异常：断言失败。"""
    from tests.verify_ch04_live import build_config, parser, validate_resources

    environment = tmp_path / ".env"
    environment.write_text("CHAT_BASE_URL=http://invalid.test/v1\nCHAT_MODEL=offline\nCHAT_API_KEY=offline\nMYSQL_PORT=3307\n", encoding="utf-8")
    for name, value in {"CHAT_BASE_URL": "http://invalid.test/v1", "CHAT_MODEL": "offline", "CHAT_API_KEY": "offline"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MYSQL_PORT", "3333")
    args = parser().parse_args(["--test", "--env-file", str(environment), "--report-dir", str(tmp_path / "new"),
                               "--mysql-port", "13307", "--knowledge-model-path", "D:/models/bge-m3"])
    config = build_config(args)
    assert config.mysql_port == 13307 and config.knowledge_model_path == "D:/models/bge-m3"
    validate_resources(config, test=True)
    with pytest.raises(ValueError):
        validate_resources(config, test=False)
    for name in ("mysql_database", "milvus_database", "milvus_hybrid_collection"):
        with pytest.raises(ValueError):
            validate_resources(config.model_copy(update={name: "mewhelp"}), test=True)


def test_existing_corpus_and_vectors_cannot_be_silently_replaced() -> None:
    """异构知识、额外向量及同ID异文向量均在同步删除/覆盖前拒绝。参数：无；返回：None；异常：断言失败。"""
    from tests.rag.test_retrieval import _chunk
    from tests.verify_ch04_live import verify_corpus, verify_vector_subset

    chunk = _chunk(1)
    verify_corpus([], [chunk])
    verify_corpus([chunk], [chunk])
    with pytest.raises(ValueError):
        verify_corpus([chunk.model_copy(update={"source_key": "doc:foreign.md"})], [chunk])
    with pytest.raises(ValueError):
        verify_corpus([chunk.model_copy(update={"is_active": False})], [chunk])
    verify_vector_subset([chunk], {1: (chunk.content_hash, chunk.embedding_revision)})
    with pytest.raises(ValueError):
        verify_vector_subset([chunk], {7: (chunk.content_hash, chunk.embedding_revision)})
    with pytest.raises(ValueError):
        verify_vector_subset([chunk], {1: ("f" * 64, chunk.embedding_revision)})


@pytest.mark.parametrize("kind", ["knowledge", "unknown_model"])
@pytest.mark.parametrize("names,accepted", [
    (["query_faq"], True), (["query_faq", "query_faq"], True),
    ([], False), (["query_faq", "query_product"], False), (["query_faq"] * 3, False),
])
def test_faq_path_accepts_repeated_calls_only_within_actual_limit(kind, names, accepted) -> None:
    """知识路径可执行不同关键词的多次FAQ，但禁止零次、异类工具及超配置上限。参数：案例类型/工具/预期；返回：None；异常：断言失败。"""
    import json
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL

    case, observed = _knowledge_observation()
    case["kind"] = kind
    observed["tool_max_calls_per_turn"] = 2
    if kind == "unknown_model":
        observed["model_checks"][0]["decision"] = {"useful": False, "reason": "没有该型号资料", "answer": "", "citation_numbers": []}
        observed["pool_rows"] = [{"raw_question": case["query"], "source": "self_check", "reason": "没有该型号资料"}]
        observed["messages"][-1]["content"] = FIXED_KNOWLEDGE_REFUSAL
        next(event for event in observed["events"] if event["event"] == "delta")["data"]["delta"] = FIXED_KNOWLEDGE_REFUSAL
    messages, statuses = [], []
    for index, name in enumerate(names):
        call_id = f"call-{index}"
        messages.extend([
            {"id": 10 + index * 2, "role": "assistant", "content": "", "turn_id": "turn-one", "turn_status": "completed",
             "tool_calls": [{"id": call_id, "name": name, "args": {"keyword": f"型号容量{index}"}}]},
            {"id": 11 + index * 2, "role": "tool", "turn_id": "turn-one", "turn_status": "completed", "tool_call_id": call_id,
             "content": json.dumps({"ok": True, "is_mock": name == "query_product", "data": {"hits": []}, "error": None})}])
        statuses.append({"event": "tool_status", "data": {"name": name, "tool_call_id": call_id, "status": "succeeded"}})
    observed["messages"] = [observed["messages"][0], *messages, observed["messages"][-1]]
    observed["events"] = [*statuses, *(event for event in observed["events"] if event["event"] != "tool_status")]
    failures = inspect_exchange(case, observed)["failures"]
    assert ("unexpected_tool_path" not in failures) is accepted
    if accepted:
        assert failures == []
        # 放宽工具数量不能绕过后续工具结果配对、证据全集或单条问题池检查。
        observed["messages"].pop(-2)
        assert "tool_result_pairing_mismatch" in inspect_exchange(case, observed)["failures"]


def test_explicit_budgets_override_legacy_env_alias(tmp_path, monkeypatch) -> None:
    """显式预算必须覆盖旧CHAT_TOKEN_BUDGET和请求预算，省略时尊重env。参数：临时目录/环境替换；返回：None；异常：断言失败。"""
    from tests.verify_ch04_live import build_config, parser

    environment = tmp_path / ".env"
    environment.write_text("CHAT_TOKEN_BUDGET=2000\nREQUEST_TOKEN_BUDGET=4000\n", encoding="utf-8")
    for name, value in {"CHAT_BASE_URL": "http://invalid.test/v1", "CHAT_MODEL": "offline", "CHAT_API_KEY": "offline",
                        "CHAT_TOKEN_BUDGET": "2000", "REQUEST_TOKEN_BUDGET": "4000"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TOKEN_BUDGET", raising=False)
    arguments = ["--test", "--env-file", str(environment), "--report-dir", str(tmp_path / "new")]
    before = build_config(parser().parse_args(arguments))
    assert (before.token_budget, before.request_token_budget) == (2000, 4000)
    try:
        args = parser().parse_args([*arguments, "--token-budget", "12000", "--request-token-budget", "16000"])
    except SystemExit:
        pytest.fail("真实验收缺少显式预算参数")
    after = build_config(args)
    assert (after.token_budget, after.request_token_budget) == (12000, 16000)


def test_prompt_positions_do_not_change_complete_evidence_identity() -> None:
    """首尾位置调整不能把相同完整证据判为不一致，也不能改写原始采样。

    参数：无；用手工三条引用模拟实际1、3、2的Prompt顺序。
    返回：None，核验成功时正常结束。
    异常：AssertionError，完整证据误判或采样顺序被修改。
    """
    case, observed = _knowledge_observation()
    third = {**deepcopy(observed["events"][-2]["data"]["citations"][1]), "n": 3, "chunk_id": 12}
    for metadata in (observed["events"][-2]["data"], observed["messages"][-1]["rag_metadata"]):
        metadata["citations"].append(deepcopy(third))
    response = deepcopy(observed["citation_responses"][1])
    response.update(n=3, snapshot=deepcopy(third))
    observed["citation_responses"].append(response)
    actual = observed["model_checks"][0]["evidence"]
    actual.insert(1, {key: value for key, value in third.items() if key not in {"content_hash", "source_type"}})
    assert [item["n"] for item in actual] == [1, 3, 2]
    assert inspect_exchange(case, observed)["failures"] == []
    assert [item["n"] for item in actual] == [1, 3, 2]


@pytest.mark.parametrize("field,replacement", [
    ("n", True), ("n", 2), ("chunk_id", 999), ("answer", "保证两天到账"),
])
def test_model_evidence_rejects_invalid_identity_or_changed_facts(field, replacement) -> None:
    """按编号匹配仍须拒绝布尔编号、重复编号、错块及正文篡改。

    参数：field 待破坏的证据字段；replacement 独立给定的非法值。
    返回：None，非法证据被准确标记时正常结束。
    异常：AssertionError，非法或变化的证据被验收放行。
    """
    case, observed = _knowledge_observation()
    observed["model_checks"][0]["evidence"][0][field] = replacement
    assert "model_evidence_mismatch" in inspect_exchange(case, observed)["failures"]

