"""RAG 评估看板接口测试：产物读取、缺失降级与参数校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.evaluation.policy import digest_json


def _summary(mrr: float = 0.88) -> dict:
    """构造最小可用的比较运行汇总。参数：mrr 目标值；返回：dict；异常：无。"""
    metric = lambda v: {"value": v, "numerator": v, "denominator": 1}
    strategy = {
        "sample_count": 8, "answerable_count": 8, "unanswerable_count": 0,
        "error_count": 0, "error_stages": {},
        "coverage": metric(0.5), "false_refusal_rate": metric(0.1),
        "unanswerable_correct_refusal": metric(1.0), "citation_validity": metric(1.0),
        "faithfulness_macro": metric(0.9), "faithfulness_micro": metric(0.9),
        "retrieval_success_count": 8,
        "recall@10": metric(1.0), "all_evidence@10": metric(0.6), "mrr@10": metric(mrr),
    }
    return {
        "status": "completed", "expected_sample_count": 8, "elapsed_seconds": 12.0,
        "metadata": {"corpus_digest": "c" * 64, "chunk_count": 52, "model": "m",
                     "judge_model": "j", "dataset_digest": "d" * 64, "source_revision": "abc123",
                     "policy_digest": digest_json({"thresholds": {"dense": {"minimum": 0.66}}})},
        "strategies": {"dense": strategy, "bm25": strategy, "hybrid": strategy,
                       "hybrid_rerank": {**strategy, "mrr@10": metric(mrr)}},
        "buckets": {name: {"A_policy": strategy} for name in
                    ("dense", "bm25", "hybrid", "hybrid_rerank")},
        "difficulties": {name: {"easy": strategy} for name in
                         ("dense", "bm25", "hybrid", "hybrid_rerank")},
    }


def _client(monkeypatch, tmp_path: Path, runs: dict[str, dict]) -> TestClient:
    """把看板的产物根目录指到临时目录。参数：monkeypatch/tmp_path/runs；返回：TestClient；异常：无。"""
    from app.api import rageval
    from app.main import create_app

    root = tmp_path / "ch04"
    root.mkdir()
    for name, payload in runs.items():
        directory = root / name
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        if "calibration" in name:
            (directory / "policy.json").write_text(
                json.dumps({"thresholds": {"dense": {"minimum": 0.66}}}), encoding="utf-8")
    monkeypatch.setattr(rageval, "REPORTS_ROOT", root)
    monkeypatch.setattr(rageval, "JOB_STATE", root / "job.json")
    monkeypatch.setattr(rageval, "JOB_LOG", root / "job.log")
    return TestClient(create_app())


def test_overview_reads_latest_comparison(monkeypatch, tmp_path) -> None:
    """有产物时返回四策略矩阵与最佳策略。参数：monkeypatch/tmp_path；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {
        "20261024-calibration-01": _summary(),
        "20261024-evaluation-01": _summary(0.9),
    })
    body = client.get("/api/rag-eval/overview").json()
    assert body["present"] is True
    assert body["best"]["name"] == "hybrid_rerank"
    assert body["best"]["mrr"] == 0.9
    assert set(body["by_bucket"]) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    # 库外桶不参与召回图，只保留五个可作答桶。
    assert "F_unanswerable" not in body["buckets"]
    # _summary 里 expected_sample_count=8 且 4 个策略，真实题数是 8//4=2
    assert body["calibration"]["case_count"] == 2
    assert body["meta"]["sample_count"] == 2
    assert body["calibration"]["thresholds"]["dense"]["minimum"] == 0.66


def test_overview_without_artifacts_degrades(monkeypatch, tmp_path) -> None:
    """没有任何运行目录时返回 present=False 并给出提示。参数：monkeypatch/tmp_path；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {})
    body = client.get("/api/rag-eval/overview").json()
    assert body["present"] is False
    assert "先跑一次评估" in body["hint"]


def test_overview_pairs_comparison_with_its_actual_policy(monkeypatch, tmp_path) -> None:
    """新校准尚未完成比较时，旧成绩只能显示旧校准门槛。

    参数：monkeypatch 隔离接口目录；tmp_path 本例独占报告目录。
    返回：None。
    异常：AssertionError，接口把不同运行的门槛与成绩混用。
    """
    client = _client(monkeypatch, tmp_path, {
        "20261024-calibration-01": _summary(),
        "20261024-evaluation-01": _summary(),
        "20261025-calibration-01": {**_summary(), "expected_sample_count": 416},
    })
    (tmp_path / "ch04/20261025-calibration-01/policy.json").write_text(
        json.dumps({"thresholds": {"dense": {"minimum": 0.99}}}), encoding="utf-8")
    body = client.get("/api/rag-eval/overview").json()
    assert body["calibration"]["thresholds"]["dense"]["minimum"] == 0.66
    assert body["calibration"]["case_count"] == 2


@pytest.mark.parametrize("missing_binding", [True, False])
def test_overview_does_not_guess_unmatched_calibration(monkeypatch, tmp_path, missing_binding) -> None:
    """没有绑定摘要或摘要不匹配时，保留成绩而不猜校准门槛。

    参数：monkeypatch/tmp_path 隔离报告；missing_binding 是否缺失绑定。
    返回：None。
    异常：AssertionError，接口为无法证明匹配的成绩展示门槛。
    """
    comparison = _summary()
    if missing_binding:
        comparison["metadata"].pop("policy_digest")
    else:
        comparison["metadata"]["policy_digest"] = "0" * 64
    client = _client(monkeypatch, tmp_path, {
        "20261024-calibration-01": _summary(),
        "20261024-evaluation-01": comparison,
    })
    body = client.get("/api/rag-eval/overview").json()
    assert body["present"] is True
    assert body["calibration"]["thresholds"] == {}
    assert body["calibration"]["case_count"] is None


def test_overview_without_comparison_still_reports_hint(monkeypatch, tmp_path) -> None:
    """只有校准没有比较时不能假装有成绩。参数：monkeypatch/tmp_path；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {"20261024-calibration-01": _summary()})
    body = client.get("/api/rag-eval/overview").json()
    assert body["present"] is False
    assert "比较" in body["hint"]


def test_rerun_rejects_invalid_strategy(monkeypatch, tmp_path) -> None:
    """非法策略名必须在启动子进程前拒绝。参数：monkeypatch/tmp_path；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {})
    response = client.post("/api/rag-eval/rerun", params={"strategies": "dense,nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_rerun_rejects_missing_dataset(monkeypatch, tmp_path) -> None:
    """题集不存在时不启动进程。参数：monkeypatch/tmp_path；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {})
    response = client.post("/api/rag-eval/rerun", params={"dataset": "no/such/file.json"})
    assert response.status_code == 400


def test_job_endpoint_reports_idle_without_state(monkeypatch, tmp_path) -> None:
    """没有任务状态文件时按 idle 返回，不报错。参数：monkeypatch/tmp_path；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {})
    body = client.get("/api/rag-eval/job").json()
    assert body["status"] == "idle"


def test_rag_eval_page_is_served(monkeypatch, tmp_path) -> None:
    """看板页面与聊天页同源提供。参数：monkeypatch/tmp_path；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {})
    response = client.get("/rag-eval")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/static/rag-eval.mjs" in response.text
    script = client.get("/static/rag-eval.mjs")
    assert script.status_code == 200 and "production/runs" in script.text


def test_case_count_divides_by_strategy_count() -> None:
    """看板展示的是题数，不是「题数×策略数」的总样本量。

    参数：无。
    返回：None。
    异常：AssertionError，把样本量当成题数展示。
    """
    from app.api.rageval import _case_count

    # 24 题 × 4 策略 = 96 样本
    assert _case_count({"expected_sample_count": 96,
                        "strategies": {k: {} for k in ("dense", "bm25", "hybrid", "hybrid_rerank")}}) == 24
    # 48 题 × 4 策略 = 192 样本
    assert _case_count({"expected_sample_count": 192,
                        "strategies": {k: {} for k in ("dense", "bm25", "hybrid", "hybrid_rerank")}}) == 48
    # 只跑两个策略时按两个除
    assert _case_count({"expected_sample_count": 48,
                        "strategies": {"dense": {}, "bm25": {}}}) == 24
    # 缺字段时不猜
    assert _case_count(None) is None
    assert _case_count({}) is None
    assert _case_count({"expected_sample_count": 96}) is None


def test_overview_reports_error_total_for_completed_with_errors(monkeypatch, tmp_path) -> None:
    """有出错样本时要把总数暴露出来，避免「跑完了但有错」被当成全绿。

    参数：monkeypatch/tmp_path。
    返回：None。
    异常：AssertionError，出错数未统计。
    """
    summary = _summary()
    summary["status"] = "completed_with_errors"
    # _summary 里四个策略共用同一个 dict 对象，先各自复制再改，避免互相污染。
    summary["strategies"] = {name: dict(record) for name, record in summary["strategies"].items()}
    summary["strategies"]["dense"]["error_count"] = 2
    summary["strategies"]["hybrid"]["error_count"] = 3
    client = _client(monkeypatch, tmp_path, {"20261024-evaluation-01": summary})
    body = client.get("/api/rag-eval/overview").json()
    assert body["meta"]["status"] == "completed_with_errors"
    assert body["meta"]["error_total"] == 5


def test_cli_treats_completed_with_errors_as_success() -> None:
    """部分样本出错仍属完成，退出码不能是失败，否则看板会把正常结果标成失败。

    参数：无。
    返回：None。
    异常：AssertionError，completed_with_errors 被判失败。
    """
    from app.evaluation.cli import run  # noqa: F401  # 导入即验证模块可用
    import inspect
    source = inspect.getsource(run)
    assert '"completed", "completed_with_errors"' in source


def test_mixed_cli_and_legacy_reports_are_sorted_and_paired(monkeypatch, tmp_path) -> None:
    """遗漏CLI命名或仅按字符串排序、猜测校准时失败。参数：隔离目录/替换器；返回：None；异常：断言失败。"""
    names = ["20261024-evaluation-235959", "20261025T000000Z-calibrate-ab123456",
             "20261025T010000Z-compare-ab123456", "20261026-calibration-01"]
    client = _client(monkeypatch, tmp_path, {name: _summary(0.95 if "compare" in name else 0.5) for name in names})
    root = tmp_path / "ch04"
    policy = {"thresholds": {"dense": {"minimum": 0.77}}}
    (root / names[1] / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    summary = _summary(0.95)
    summary["metadata"]["policy_digest"] = digest_json(policy)
    (root / names[2] / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    body = client.get("/api/rag-eval/overview").json()
    assert body["meta"]["run"] == names[2]
    assert body["best"]["mrr"] == 0.95
    assert body["calibration"]["thresholds"]["dense"]["minimum"] == 0.77
    assert [row["name"] for row in client.get("/api/rag-eval/runs").json()["runs"]] == list(reversed(names))


@pytest.mark.parametrize("bad", [[], {}, {**_summary(), "metadata": []},
                                  {**_summary(), "strategies": {"dense": []}},
                                  {**_summary(), "status": []},
                                  {**_summary(), "metadata": {"dataset_digest": "d" * 64, "corpus_digest": "c" * 64, "source_revision": 42}},
                                  {**_summary(), "strategies": {"dense": {"error_count": "broken"}}},
                                  {**_summary(), "status": "interrupted"}])
def test_invalid_reports_are_ignored_by_overview_and_runs(monkeypatch, tmp_path, bad) -> None:
    """损坏JSON对象被列出或覆盖有效成绩时失败。参数：隔离目录/坏摘要；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {"20261024-evaluation-01": _summary(),
                                          "20261026-evaluation-01": bad,
                                          "unrelated-report": _summary()})
    body = client.get("/api/rag-eval/overview").json()
    assert body["meta"]["run"] == "20261024-evaluation-01"
    assert client.get("/api/rag-eval/runs").json()["runs"] == [{"name": "20261024-evaluation-01"}]


def test_cli_comparison_never_pairs_unmatched_policy(monkeypatch, tmp_path) -> None:
    """支持CLI目录后失去policy身份核对时失败。参数：隔离目录/替换器；返回：None；异常：断言失败。"""
    client = _client(monkeypatch, tmp_path, {"20261026T000000Z-compare-123456ab": _summary(),
                                          "20261026T000000Z-calibrate-123456ab": _summary()})
    (tmp_path / "ch04/20261026T000000Z-calibrate-123456ab/policy.json").write_text(
        json.dumps({"thresholds": {"dense": {"minimum": 0.9}}}), encoding="utf-8")
    body = client.get("/api/rag-eval/overview").json()
    assert body["present"] is True
    assert body["calibration"]["present"] is False


def test_same_day_reports_sort_by_time_instead_of_name(monkeypatch, tmp_path) -> None:
    """同日CLI字符串排序盖过更晚网页报告时失败。参数：隔离目录；返回：None；异常：断言失败。"""
    names = ["20261026-evaluation-235959", "20261026T000000Z-compare-123456ab"]
    client = _client(monkeypatch, tmp_path, {name: _summary() for name in names})
    assert client.get("/api/rag-eval/overview").json()["meta"]["run"] == names[0]
    assert [row["name"] for row in client.get("/api/rag-eval/runs").json()["runs"]] == names


def test_manifest_type_and_calibration_metadata_must_match(monkeypatch, tmp_path):
    """改名报告或不同校准摘要携带相同policy时被误用会失败。参数：隔离目录；返回：None；异常：断言失败。"""
    comparison = _summary()
    policy = {"metadata": {**comparison["metadata"], "dataset_digest": "e" * 64},
              "thresholds": {"dense": {"minimum": 0.8}}}
    comparison["metadata"]["policy_digest"] = digest_json(policy)
    client = _client(monkeypatch, tmp_path, {"20261024-evaluation-01": comparison,
        "20261024-calibration-01": _summary(), "20261026T000000Z-compare-123456ab": _summary()})
    root = tmp_path / "ch04"
    (root / "20261024-calibration-01/policy.json").write_text(json.dumps(policy), encoding="utf-8")
    (root / "20261026T000000Z-compare-123456ab/manifest.json").write_text(
        json.dumps({"retrieval_only": True}), encoding="utf-8")
    body = client.get("/api/rag-eval/overview").json()
    assert body["meta"]["run"] == "20261024-evaluation-01"
    assert body["calibration"]["present"] is False
    assert len(client.get("/api/rag-eval/runs").json()["runs"]) == 2


def test_nullable_source_revision_and_unreadable_policy_keep_comparison(monkeypatch, tmp_path):
    """正式元信息允许无Git版本且损坏policy不能遮蔽成绩。参数：隔离目录；返回：None；异常：断言失败。"""
    summary = _summary()
    summary["metadata"]["source_revision"] = None
    client = _client(monkeypatch, tmp_path, {"20261024-evaluation-01": summary,
                                          "20261024-calibration-01": _summary()})
    (tmp_path / "ch04/20261024-calibration-01/policy.json").write_bytes(b"\xff")
    body = client.get("/api/rag-eval/overview").json()
    assert body["present"] is True and body["meta"]["source_revision"] == ""
    assert body["calibration"]["present"] is False


@pytest.mark.parametrize("status", ["invalidated", "failed", "interrupted", "running", None, [], "completed", "completed_with_errors"])
def test_manifest_state_controls_whether_completed_summary_is_available(monkeypatch, tmp_path, status):
    """现有清单未有效完成时不能采用残留完成摘要。参数：替换器/目录/清单状态；返回：None；异常：断言失败。"""
    name = "20261101T000000Z-compare-123456ab"
    client = _client(monkeypatch, tmp_path, {name: _summary()})
    manifest = tmp_path / "ch04" / name / "manifest.json"
    manifest.write_text(json.dumps({"status": status, "retrieval_only": False}), encoding="utf-8")
    try:
        expected = status in ("completed", "completed_with_errors")
        assert client.get("/api/rag-eval/overview").json()["present"] is expected
        assert client.get("/api/rag-eval/runs").json()["runs"] == ([{"name": name}] if expected else [])
    finally:
        client.close()


def test_invalidation_hides_run_even_when_summary_cannot_be_replaced(monkeypatch, tmp_path):
    """真实撤销写入清单后，摘要替换失败也不能展示旧成绩。参数：替换器/目录；返回：None；异常：断言失败。"""
    from app.evaluation import runner
    name = "20261101T000000Z-compare-123456ab"
    client = _client(monkeypatch, tmp_path, {name: _summary()})
    directory = tmp_path / "ch04" / name
    (directory / "manifest.json").write_text(json.dumps({"status": "completed", "retrieval_only": False}), encoding="utf-8")
    original_replace = runner.os.replace

    def blocked_summary_replace(source, target):
        """只模拟摘要的文件系统替换失败，其他写入真实执行。参数：源与目标路径；返回：Path；异常：PermissionError/IO错误。"""
        if Path(target) == directory / "summary.json":
            raise PermissionError("测试隔离摘要替换失败")
        return original_replace(source, target)

    monkeypatch.setattr(runner.os, "replace", blocked_summary_replace)
    try:
        runner.invalidate_evaluation_artifacts(directory)
        assert json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["status"] == "invalidated"
        assert json.loads((directory / "summary.json").read_text(encoding="utf-8"))["status"] == "completed"
        assert client.get("/api/rag-eval/overview").json()["present"] is False
        assert client.get("/api/rag-eval/runs").json()["runs"] == []
    finally:
        client.close()
