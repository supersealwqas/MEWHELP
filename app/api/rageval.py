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
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from app.evaluation.jobs import EvaluationJobManager
from app.evaluation.policy import digest_json

router = APIRouter(prefix="/api/rag-eval", tags=["rag-eval"])

# 评估产物根目录与任务状态文件；与 CLI 的默认输出位置保持一致。
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = REPO_ROOT / "reports" / "ch04"
JOB_STATE = REPORTS_ROOT / "rageval-job.json"
JOB_LOG = REPORTS_ROOT / "rageval-job.log"
# 校准与比较共享整体上限，由应用生命周期拥有的管理器负责超时清理。
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
_CLI_RUN_NAME = re.compile(r"^(\d{8}T\d{6}Z)-(calibrate|compare)-[0-9a-f]{8}$")


def _run_identity(directory: Path) -> tuple[str, float, int] | None:
    """统一解析CLI和历史网页目录的类型与时间。参数：directory目录；返回：类型/时间/序号或None；异常：无。"""
    try:
        if match := _CLI_RUN_NAME.fullmatch(directory.name):
            stamp, command = match.groups()
            instant = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return ("calibration" if command == "calibrate" else "evaluation", instant.timestamp(), 0)
        if _RUN_NAME.fullmatch(directory.name):
            date, kind, number = directory.name.split("-")
            # 网页历史命名使用主机本地时间；更早的短后缀是当日序号。
            stamp = datetime.strptime(date + (number if len(number) == 6 else "000000"), "%Y%m%d%H%M%S")
            return kind, stamp.timestamp(), int(number)
    except ValueError:
        pass
    return None


def _run_dirs() -> list[Path]:
    """列出CLI与历史网页的有效运行目录，按解析时间倒序（最新在前）。

    参数：无。
    返回：list[Path]，只包含有效完成摘要的受约束运行目录。
    异常：无；目录不可读时返回空列表。
    """
    if not REPORTS_ROOT.is_dir():
        return []
    try:
        found = [(p, identity) for p in REPORTS_ROOT.iterdir()
                 if p.is_dir() and (identity := _run_identity(p)) and _read_summary(p) is not None]
    except OSError:
        return []
    return [p for p, _ in sorted(found, key=lambda item: (*item[1][1:], item[0].name), reverse=True)]


def _read_summary(directory: Path) -> dict | None:
    """读取并解析运行目录的 summary.json。

    参数：directory 运行目录。
    返回：dict，解析结果；文件缺失或损坏时为 None。
    异常：无；解析失败按缺失处理，不让看板整体白屏。
    """
    path = directory / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        # 目录名仅决定来源类型；缺元信息、未完成或结构损坏的内容不能成为成绩。
        if not isinstance(summary, dict) or summary.get("status") not in ("completed", "completed_with_errors"):
            return None
        metadata, strategies = summary.get("metadata"), summary.get("strategies")
        if not isinstance(metadata, dict) or not isinstance(strategies, dict) or not strategies:
            return None
        if any(not isinstance(metadata.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[key])
               for key in ("dataset_digest", "corpus_digest")):
            return None
        if metadata.get("source_revision") is not None and not isinstance(metadata["source_revision"], str):
            return None
        if any(name not in STRATEGIES or not isinstance(record, dict) for name, record in strategies.items()):
            return None
        if any(type(record.get("error_count", 0)) is not int or record.get("error_count", 0) < 0
               for record in strategies.values()):
            return None
        # JSON标准不接受NaN/Infinity，避免坏指标拖垮HTTP序列化。
        json.dumps(summary, allow_nan=False)
        if any(not isinstance(summary.get(key, {}), dict) for key in ("buckets", "difficulties")):
            return None
        # 有运行清单时核对明确的校准标志，拒绝被改名的异类报告。
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = _run_identity(directory)
            if (not isinstance(manifest, dict) or identity is None
                    or manifest.get("retrieval_only") is not (identity[0] == "calibration")):
                return None
        return summary
    except (OSError, ValueError, UnicodeError):
        return None


def _pick_pair() -> tuple[dict | None, dict | None, str | None]:
    """选择最近比较及其实际绑定的校准，避免跨轮次拼接成绩。

    参数：无。
    返回：tuple(校准 summary, 比较 summary, 比较运行目录名)，缺失时为 None。
    异常：无。
    """
    directories = _run_dirs()
    comparison = None
    comparison_name = None
    for directory in directories:
        if _run_identity(directory)[0] != "evaluation":
            continue
        summary = _read_summary(directory)
        if summary is not None:
            comparison, comparison_name = summary, directory.name
            break
    if comparison is None:
        return None, None, "还没有固定集的四策略比较结果，先跑一次评估。"

    policy_digest = (comparison.get("metadata") or {}).get("policy_digest")
    # 旧报告缺少绑定时保留成绩，但不能猜测它使用了哪个门槛。
    if not isinstance(policy_digest, str) or not policy_digest:
        return None, comparison, comparison_name
    for directory in directories:
        if _run_identity(directory)[0] != "calibration":
            continue
        policy = _read_policy(directory)
        try:
            matched = bool(policy) and digest_json(policy) == policy_digest
        except ValueError:
            # 损坏产物中的非有限数字不是有效JSON策略，不能拖垮已有成绩展示。
            continue
        if matched and (summary := _read_summary(directory)) is not None:
            if "metadata" in policy and policy["metadata"] != summary["metadata"]:
                continue
            return {**summary, "thresholds": policy["thresholds"]}, comparison, comparison_name
    return None, comparison, comparison_name


def _read_policy(directory: Path) -> dict:
    """读取完整校准策略，供摘要匹配后展示其中的门槛。

    参数：directory 校准运行目录。
    返回：dict，完整策略记录；文件缺失或门槛结构损坏时为空字典。
    异常：无。
    """
    try:
        policy = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return {}
    return policy if isinstance(policy, dict) and isinstance(policy.get("thresholds"), dict) else {}


def _case_count(summary: dict | None) -> int | None:
    """从运行汇总反推真实题数。

    `expected_sample_count` 是「题数 × 策略数」的总样本量，不是题数；
    看板要展示的是题数，因此除以本次实际运行的策略数。

    参数：summary 运行汇总，可为 None。
    返回：int 题数；无法推断时为 None。
    异常：无。
    """
    if not summary:
        return None
    total = summary.get("expected_sample_count")
    strategies = summary.get("strategies")
    if not isinstance(total, int) or not isinstance(strategies, dict) or not strategies:
        return None
    return total // len(strategies) if len(strategies) else None


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


def create_job_manager() -> EvaluationJobManager:
    """建立本应用的评估所有者。参数：无；返回：管理器；异常：无。"""
    return EvaluationJobManager(REPO_ROOT, REPORTS_ROOT, timeout_seconds=JOB_TIMEOUT_SECONDS,
                                state_path=JOB_STATE, log_path=JOB_LOG)


def _job_state(request: Request) -> dict:
    """读取生命周期所有者或降级读取历史状态。参数：request请求；返回：状态；异常：无。"""
    manager = getattr(request.app.state, "evaluation_jobs", None)
    # 未进入lifespan的只读请求仍可看历史报告，但临时读取者永远不能启动作业。
    return (manager or create_job_manager()).status()


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
async def overview(request: Request) -> dict:
    """返回看板所需的全部数据：元信息、KPI、分桶矩阵与重跑状态。

    参数：request 当前请求，用于读取应用持有的评估状态。
    返回：dict，present=False 表示还没有可展示的比较结果。
    异常：HTTPException，产物读取出现不可恢复错误。
    """
    calibration, comparison, note = _pick_pair()
    job = _job_state(request)
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
            # 出错样本数单独统计，让「跑完了但有错」不被当成全绿。
            "error_total": sum((r.get("error_count") or 0)
                               for r in (comparison.get("strategies") or {}).values()),
            "sample_count": _case_count(comparison),
        },
        "calibration": {
            "present": calibration is not None,
            "case_count": _case_count(calibration),
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
async def job(request: Request) -> dict:
    """返回重跑任务状态与日志尾部。

    参数：request 当前请求，用于读取应用持有的评估状态。
    返回：dict，status/log 两个字段。
    异常：无。
    """
    return {**_job_state(request), "log": _log_tail()}


@router.post("/rerun")
async def rerun(
    request: Request,
    dataset: str = Query(default="data/evaluation/ch04.json", description="题集路径，相对项目根目录"),
    strategies: str = Query(default="dense,bm25,hybrid,hybrid_rerank", description="逗号分隔的策略列表"),
) -> dict:
    """在后台启动一次「校准 + 固定集比较」，不阻塞请求。

    参数：request 当前请求；dataset 题集路径；strategies 逗号分隔的策略名。
    返回：dict，含 pid 与启动信息。
    异常：HTTPException 409，已有任务或清理未完成；400，参数非法；503，未进入应用生命周期。
    """
    chosen = [item.strip() for item in strategies.split(",") if item.strip()]
    invalid = [item for item in chosen if item not in STRATEGIES]
    if not chosen or invalid:
        raise HTTPException(status_code=400, detail=f"策略名非法：{invalid or '空'}")

    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[2] / dataset_path
    if not dataset_path.is_file():
        raise HTTPException(status_code=400, detail=f"题集不存在：{dataset}")

    manager = getattr(request.app.state, "evaluation_jobs", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="评估运行时尚未启动")
    try:
        return manager.start(dataset_path, chosen)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _production_dependencies(request: Request):
    """取得生命周期提供的共同所有者和应用仓储。参数：请求；返回：manager/repository；异常：HTTP503未启动。"""
    manager = getattr(request.app.state, "evaluation_jobs", None)
    service = getattr(request.app.state, "chat_service", None)
    repository = getattr(service, "evaluations", None)
    if manager is None or repository is None:
        raise HTTPException(status_code=503, detail="生产评估运行时尚未启动")
    return manager, repository


@router.post("/production/rerun")
async def production_rerun(request: Request) -> dict:
    """接受一次配置题集和冻结策略评估，数据库准入由子进程明确报告。

    参数：请求；返回：本实例started/admission=pending，不能表示全库已取得执行锁。
    异常：HTTP503资源缺失、409本实例忙或关闭；不接受浏览器任意文件路径。
    """
    from app.config import settings
    manager, _ = _production_dependencies(request)
    dataset, policy = Path(settings.evaluation_dataset_path).resolve(), Path(settings.evidence_policy_path).resolve()
    if not dataset.is_file() or not policy.is_file():
        raise HTTPException(status_code=503, detail="生产题集或冻结策略文件不可用")
    try:
        return manager.start(dataset, [], profile="production", policy_path=policy, triggered_by="手动")
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/production/job")
async def production_job(request: Request) -> dict:
    """同时报告本实例任务与数据库实际执行锁，不以本机idle代替全库空闲。

    参数：请求；返回：共同job状态、database_busy与下次时间；异常：HTTP503资源或数据库不可用。
    """
    manager, repository = _production_dependencies(request)
    try:
        busy = await repository.execution_busy()
    except Exception as error:
        raise HTTPException(status_code=503, detail="无法核验数据库评估占用状态") from error
    scheduler = getattr(request.app.state, "evaluation_scheduler", None)
    return {**manager.status(), "database_busy": busy,
        "next_run_at": scheduler.next_at.isoformat() if scheduler is not None else None}


@router.get("/production/runs")
async def production_runs(request: Request,
        after_id: int = Query(default=0, ge=0, description="上一页最后主键，首次为0"),
        limit: int = Query(default=50, ge=1, le=200, description="每页条数")) -> dict:
    """分页读取已联合核验并保存的生产评估，缺失指标保持null。

    参数：请求、非负游标、1–200页长；返回：items/next_after_id；异常：HTTP503数据库或生命周期不可用。
    """
    _, repository = _production_dependencies(request)
    try:
        return await repository.list_runs(after_id, limit)
    except Exception as error:
        raise HTTPException(status_code=503, detail="生产评估历史暂不可用") from error
