"""RAG 评估看板接口：读取四策略评估产物并提供受控重跑入口。

本模块只读 `reports/ch04/` 下已有的评估运行目录，不在请求线程里跑模型或数据库。
重跑走独立的子进程任务，状态写入 `reports/ch04/rageval-job.json`，前端轮询读取。

对外接口：
- GET  /api/rag-eval/overview   返回最近一次运行的四策略汇总、分桶与分难度指标
- GET  /api/rag-eval/runs       列出所有可用的运行目录
- POST /api/rag-eval/rerun      启动一次校准+比较（后台子进程）
- GET  /api/rag-eval/job        查询重跑任务状态与日志尾部
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/rag-eval", tags=["rag-eval"])

# 评估产物根目录与任务状态文件；与 CLI 的默认输出位置保持一致。
REPORTS_ROOT = Path(__file__).resolve().parents[2] / "reports" / "ch04"
JOB_STATE = REPORTS_ROOT / "rageval-job.json"
JOB_LOG = REPORTS_ROOT / "rageval-job.log"
# 单次评估最多跑多久，超时由子进程自身终止，避免僵尸进程长期占用 GPU。
JOB_TIMEOUT_SECONDS = 3600

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
STRATEGY_LABELS = {
    "dense": "纯向量",
    "bm25": "纯 BM25",
    "hybrid": "混合",
    "hybrid_rerank": "混合 + 重排",
}
# 只有可作答桶参与检索召回计分；F_unanswerable 只看拒答是否正确。
BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_filter", "E_multi", "F_unanswerable")
BUCKET_LABELS = {
    "A_policy": "政策类",
    "B_model": "型号类",
    "C_colloquial": "口语类",
    "D_filter": "筛选类",
    "E_multi": "多证据类",
    "F_unanswerable": "库外问题",
}
_RUN_NAME = re.compile(r"^\d{8}-(?:calibration|evaluation)-\d+$")


def _run_dirs() -> list[Path]:
    """列出形如 `20261024-evaluation-01` 的运行目录，按名称倒序（最新在前）。

    参数：无。
    返回：list[Path]，只包含含 summary.json 的目录。
    异常：无；目录不可读时返回空列表。
    """
    if not REPORTS_ROOT.is_dir():
        return []
    found = [p for p in REPORTS_ROOT.iterdir()
             if p.is_dir() and _RUN_NAME.match(p.name) and (p / "summary.json").is_file()]
    return sorted(found, key=lambda p: p.name, reverse=True)


def _read_summary(directory: Path) -> dict | None:
    """读取并解析运行目录的 summary.json。

    参数：directory 运行目录。
    返回：dict，解析结果；文件缺失或损坏时为 None。
    异常：无；解析失败按缺失处理，不让看板整体白屏。
    """
    path = directory / "summary.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pick_pair() -> tuple[dict | None, dict | None, str | None]:
    """挑出最近一次校准与比较运行，供看板展示。

    参数：无。
    返回：tuple(校准 summary, 比较 summary, 比较运行目录名)，缺失时为 None。
    异常：无。
    """
    calibration = comparison = None
    comparison_name = None
    for directory in _run_dirs():
        summary = _read_summary(directory)
        if summary is None:
            continue
        if "calibration" in directory.name and calibration is None:
            # 阈值不在 summary 里，而是同目录 policy.json 的产物。
            summary = {**summary, "thresholds": _read_policy(directory)}
            calibration = summary
        elif "evaluation" in directory.name and comparison is None:
            comparison, comparison_name = summary, directory.name
        if calibration is not None and comparison is not None:
            break
    if comparison is None:
        return calibration, None, "还没有固定集的四策略比较结果，先跑一次评估。"
    return calibration, comparison, comparison_name


def _read_policy(directory: Path) -> dict:
    """读取校准目录下的 policy.json 中的 thresholds 字段。

    参数：directory 校准运行目录。
    返回：dict，策略到阈值记录；文件缺失或损坏时为空字典。
    异常：无。
    """
    try:
        policy = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    thresholds = policy.get("thresholds")
    return thresholds if isinstance(thresholds, dict) else {}


def _best_strategy(summary: dict) -> dict:
    """按 MRR@10 选出最佳策略，同分时优先覆盖率更高者。

    参数：summary 比较运行汇总。
    返回：dict，含 name/label/mrr/coverage；无可用数据时为空字典。
    异常：无。
    """
    strategies = summary.get("strategies") or {}
    ranked = []
    for name, record in strategies.items():
        # strategies 里的指标都是 {value, numerator, denominator} 结构，先解包。
        mrr = _metric(record, "mrr@10")
        if mrr is None:
            continue
        coverage = _metric(record, "coverage")
        ranked.append((mrr, coverage if coverage is not None else -1.0, name))
    if not ranked:
        return {}
    mrr, coverage, name = max(ranked)
    return {"name": name, "label": STRATEGY_LABELS.get(name, name),
            "mrr": mrr, "coverage": None if coverage < 0 else coverage}


def _metric(record: dict, key: str) -> float | None:
    """从带分子分母的指标结构里取出数值。

    参数：record 策略或分桶记录；key 指标名。
    返回：float 或 None（缺值/不可用）。
    异常：无。
    """
    value = record.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    return value if isinstance(value, (int, float)) else None


def _series(container: dict, section: str) -> dict:
    """把 `strategies` 或 `buckets` 结构整理成前端绘图用的矩阵。

    参数：container summary 全文；section "strategies" | "buckets" | "difficulties"。
    返回：dict，形如 {策略: {分桶: {指标: 值}}}。
    异常：无。
    """
    block = container.get(section) or {}
    if section == "strategies":
        # strategies 是扁平结构：策略 -> 指标
        return {name: record for name, record in block.items() if name in STRATEGIES}
    return {name: record for name, record in block.items() if name in STRATEGIES}


def _job_state() -> dict:
    """读取重跑任务状态文件。

    参数：无。
    返回：dict，无任务时返回 idle 状态。
    异常：无。
    """
    try:
        state = json.loads(JOB_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "idle", "message": "尚未运行"}
    # 进程已消失但状态仍是 running，说明被外部终止，如实标注而不是假装还在跑。
    pid = state.get("pid")
    if state.get("status") == "running" and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except OSError:
            state["status"] = "interrupted"
            state["message"] = "进程已退出，任务未正常结束"
    return state


def _log_tail(lines: int = 40) -> str:
    """读取重跑日志尾部，便于前端直接展示进度。

    参数：lines 最多返回的行数。
    返回：str，日志尾部文本；文件不存在为空串。
    异常：无。
    """
    try:
        content = JOB_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


@router.get("/overview")
async def overview() -> dict:
    """返回看板所需的全部数据：元信息、KPI、分桶矩阵与重跑状态。

    参数：无。
    返回：dict，present=False 表示还没有可展示的比较结果。
    异常：HTTPException，产物读取出现不可恢复错误。
    """
    calibration, comparison, note = _pick_pair()
    job = _job_state()
    runs = [{"name": p.name} for p in _run_dirs()[:20]]
    if comparison is None:
        return {"present": False, "hint": note, "job": job, "runs": runs,
                "strategies": list(STRATEGIES), "strategy_labels": STRATEGY_LABELS,
                "buckets": list(BUCKETS), "bucket_labels": BUCKET_LABELS}

    metadata = comparison.get("metadata") or {}
    buckets = comparison.get("buckets") or {}
    # 只保留可作答桶参与召回图，库外桶单独用拒答率展示。
    graded = [b for b in BUCKETS if b != "F_unanswerable"]
    return {
        "present": True,
        "meta": {
            "run": note,
            "dataset_digest": (metadata.get("dataset_digest") or "")[:16],
            "corpus_digest": (metadata.get("corpus_digest") or "")[:16],
            "chunk_count": metadata.get("chunk_count"),
            "model": metadata.get("model"),
            "judge_model": metadata.get("judge_model"),
            "source_revision": (metadata.get("source_revision") or "")[:8],
            "elapsed_seconds": comparison.get("elapsed_seconds"),
            "status": comparison.get("status"),
            "sample_count": comparison.get("expected_sample_count"),
        },
        "calibration": {
            "present": calibration is not None,
            "case_count": (calibration or {}).get("expected_sample_count"),
            "thresholds": ((calibration or {}).get("thresholds") or {}),
        },        "strategies": list(STRATEGIES),
        "strategy_labels": STRATEGY_LABELS,
        "buckets": graded,
        "bucket_labels": BUCKET_LABELS,
        "overall": _series(comparison, "strategies"),
        "by_bucket": {name: buckets.get(name) or {} for name in STRATEGIES},
        "by_difficulty": _series(comparison, "difficulties"),
        "best": _best_strategy(comparison),
        "runs": runs,
        "job": job,
    }


@router.get("/runs")
async def runs() -> dict:
    """列出可用的评估运行目录。

    参数：无。
    返回：dict，包含目录名列表。
    异常：无。
    """
    return {"runs": [{"name": p.name} for p in _run_dirs()]}


@router.get("/job")
async def job() -> dict:
    """返回重跑任务状态与日志尾部。

    参数：无。
    返回：dict，status/log 两个字段。
    异常：无。
    """
    return {**_job_state(), "log": _log_tail()}


@router.post("/rerun")
async def rerun(
    dataset: str = Query(default="data/evaluation/ch04.json", description="题集路径，相对项目根目录"),
    strategies: str = Query(default="dense,bm25,hybrid,hybrid_rerank", description="逗号分隔的策略列表"),
) -> dict:
    """在后台启动一次「校准 + 固定集比较」，不阻塞请求。

    参数：dataset 题集路径；strategies 逗号分隔的策略名。
    返回：dict，含 pid 与启动信息。
    异常：HTTPException 409，已有任务在跑；400，参数非法。
    """
    current = _job_state()
    if current.get("status") == "running":
        raise HTTPException(status_code=409, detail="已有评估任务在运行，请等待结束")

    chosen = [item.strip() for item in strategies.split(",") if item.strip()]
    invalid = [item for item in chosen if item not in STRATEGIES]
    if not chosen or invalid:
        raise HTTPException(status_code=400, detail=f"策略名非法：{invalid or '空'}")

    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[2] / dataset_path
    if not dataset_path.is_file():
        raise HTTPException(status_code=400, detail=f"题集不存在：{dataset}")

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    calibration_dir = REPORTS_ROOT / f"{stamp[:8]}-calibration-{stamp[-6:]}"
    comparison_dir = REPORTS_ROOT / f"{stamp[:8]}-evaluation-{stamp[-6:]}"

    # 两段串行执行：先校准出 policy.json，再用它跑固定集比较。
    script = (
        "set -e\n"
        f'python -m app.evaluation.cli calibrate --split calibration --dataset "{dataset_path}" '
        f'--strategies {" ".join(chosen)} --output-dir "{calibration_dir}"\n'
        f'python -m app.evaluation.cli compare --split evaluation --dataset "{dataset_path}" '
        f'--strategies {" ".join(chosen)} --policy "{calibration_dir / "policy.json"}" '
        f'--output-dir "{comparison_dir}"\n'
    )
    JOB_LOG.write_text("", encoding="utf-8")
    log_handle = open(JOB_LOG, "a", encoding="utf-8", buffering=1)
    try:
        process = subprocess.Popen(
            ["bash", "-lc", script], cwd=str(Path(__file__).resolve().parents[2]),
            stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except OSError as error:
        log_handle.close()
        raise HTTPException(status_code=500, detail=f"无法启动评估进程：{error}") from error

    state = {
        "status": "running",
        "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path.name),
        "strategies": chosen,
        "calibration_dir": calibration_dir.name,
        "comparison_dir": comparison_dir.name,
        "message": "正在校准并比较四策略，完成后刷新页面查看",
    }
    JOB_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"started": True, **state}
