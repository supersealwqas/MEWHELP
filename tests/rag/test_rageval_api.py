"""RAG 评估看板接口测试：产物读取、缺失降级与参数校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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
                     "judge_model": "j", "dataset_digest": "d" * 64, "source_revision": "abc123"},
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
    assert "RAG 评估看板" in response.text


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
