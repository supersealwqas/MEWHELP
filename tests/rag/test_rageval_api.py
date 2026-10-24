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
    assert body["calibration"]["case_count"] == 8
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
