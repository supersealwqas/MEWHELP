"""四策略逐题评估与可靠本地报告，不写聊天问题池或编造台账。"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable
from uuid import uuid4

from app.evaluation.faithfulness import FaithfulnessError
from app.evaluation.metrics import aggregate_results, retrieval_metrics
from app.evaluation.policy import STRATEGIES
from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL, decide_answer
from app.knowledge.retrieval import prepare_query


def atomic_json(path: Path, value: object) -> None:
    """在同目录完整写临时文件后原子替换，避免半份摘要。

    参数：path 输出文件；value JSON内容，不包含密钥。
    返回：None。
    异常：OSError/TypeError/ValueError，写盘失败明确上抛。
    """
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(rows: list[dict], strategies: list[str]) -> dict:
    """按策略、题桶和难度聚合。参数：rows 全部记录、strategies 策略顺序；返回：dict；异常：无。"""
    return {"strategies": {strategy: aggregate_results([row for row in rows if row["strategy"] == strategy])
                            for strategy in strategies},
            "buckets": {strategy: {bucket: aggregate_results([row for row in rows if row["strategy"] == strategy
                                                               and row["bucket"] == bucket])
                                    for bucket in sorted({row["bucket"] for row in rows})}
                        for strategy in strategies},
            "difficulties": {strategy: {difficulty: aggregate_results([row for row in rows if row["strategy"] == strategy
                                                                        and row["difficulty"] == difficulty])
                                         for difficulty in ("easy", "medium", "hard")}
                             for strategy in strategies}}


def markdown_report(report: dict) -> str:
    """生成与JSON同源的可读摘要。参数：report 汇总；返回：Markdown；异常：无。"""
    lines = ["# CH04 四策略评估", "", f"运行状态：{report['status']}。原始证据与错误见 results.jsonl。", "",
             "| 策略 | 样本 | Recall@10 | MRR@10 | 覆盖率 | 误拒率 | 无答案正确拒答 | 忠实度 | 错误 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["strategies"].items():
        values = []
        for metric in ("recall@10", "mrr@10", "coverage", "false_refusal_rate", "unanswerable_correct_refusal", "faithfulness_macro"):
            value = row[metric]["value"]
            values.append("null" if value is None else f"{value:.4f}")
        lines.append(f"| {name} | {row['sample_count']} | " + " | ".join(values) + f" | {row['error_count']} |")
    lines.extend(["", "难度分组：", "", "| 策略 | 难度 | 样本 | Recall@10（分母） | MRR@10（分母） | 错误 |",
                  "|---|---|---:|---:|---:|---:|"])
    for strategy, difficulties in report["difficulties"].items():
        for difficulty, row in difficulties.items():
            values = []
            for metric in ("recall@10", "mrr@10"):
                value = row[metric]["value"]
                values.append(("null" if value is None else f"{value:.4f}") + f"（{row[metric]['denominator']}）")
            lines.append(f"| {strategy} | {difficulty} | {row['sample_count']} | " + " | ".join(values) + f" | {row['error_count']} |")
    lines.extend(["", "所有比率的分子、分母与分桶/难度完整指标见 summary.json。检索失败在可答题召回分母中按零计；拒答不算忠实度满分。",
                  "", "排名深度：dense、bm25、hybrid最多50；hybrid_rerank每子句重排10，普通最多10、退款最多30。其Recall@50按实际排名深度计算，不表示所有策略都有50条重排结果。",
                  "", "这是同一模型作为生成者和独立调用裁判的自动评估，不能代替人工事实核查。unsupported 判定均保留逐事实理由。", ""])
    return "\n".join(lines)


async def evaluate_dataset(cases: list, runtime, output_dir: Path, *,
                           strategies: list[str] | None = None, thresholds: dict[str, float | None] | None = None,
                           evidence_resolver: Callable | None = None, retrieval_only: bool = False,
                           progress: Callable[[dict], None] | None = None) -> dict:
    """运行选定题目，每完成一题一策略即持久化，失败不抹掉其他样本。

    参数：cases 严格加载题；runtime 冻结资源和模型；output_dir 不存在的独占目录；strategies 四策略子集；
        thresholds 各策略校准门槛或None；evidence_resolver 默认dataset解析器；retrieval_only 校准仅检索；progress 可选状态回调。
    返回：dict，状态、逐策略/分桶统计与版本信息。
    异常：ValueError，参数非法；FileExistsError，目录已存在；OSError，报告失败；取消保留已落盘样本后传播。
    """
    strategies = list(STRATEGIES) if strategies is None else strategies
    if not cases or not strategies or len(set(strategies)) != len(strategies) or set(strategies) - set(STRATEGIES):
        raise ValueError("评估需要非空题目与不重复的合法策略")
    thresholds = thresholds or {}
    if set(thresholds) - set(STRATEGIES) or any(value is not None and not math.isfinite(value) for value in thresholds.values()):
        raise ValueError("策略门槛必须是有限值或null")
    if evidence_resolver is None:
        from app.evaluation.dataset import resolve_evidence
        evidence_resolver = resolve_evidence
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"status": "running", "expected_sample_count": len(cases) * len(strategies),
                "strategies": strategies, "thresholds": thresholds, "retrieval_only": retrieval_only,
                "metadata": runtime.metadata}
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(output_dir / "corpus.json", [chunk.model_dump(mode="json") for chunk in runtime.chunks])
    rows, status = [], "interrupted"
    try:
        with (output_dir / "results.jsonl").open("x", encoding="utf-8", newline="\n") as log:
            for case in cases:
                # 规则只读取原问，不能以标注bucket/answerable改变检索路径。
                prepared = None
                for strategy in strategies:
                    begin, stage = time.perf_counter(), "dataset"
                    row = {"id": case.id, "split": case.split, "bucket": case.bucket, "difficulty": case.difficulty,
                           "strategy": strategy, "query": case.query, "answerable": case.answerable,
                           "refused": None, "refusal_reason": None, "answer": None,
                           "error": None, "faithfulness": None, "supported_count": 0, "total_count": 0,
                           "retrieval": None, "ranked_ids": [], "top_score": None, "citations": [],
                           "citation_valid": None, "timings": {}}
                    try:
                        if prepared is None:
                            stage = "query"
                            prepared = prepare_query(case.query, policy=any(term in case.query for term in ("退款", "退货", "退换")))
                        stage = "dataset"
                        groups = evidence_resolver(case, runtime.chunks)
                        row["evidence_groups"] = [sorted(group) for group in groups]
                        row["prepared_queries"] = prepared.groups
                        stage, tick = "retrieval", time.perf_counter()
                        result = await runtime.retrieve(prepared, strategy, case.filters)
                        row["timings"]["retrieval_seconds"] = time.perf_counter() - tick
                        row["ranked_ids"] = [item.chunk.id for item in result.ranked]
                        row["ranking"] = [{"chunk_id": item.chunk.id, "source_key": item.chunk.source_key,
                                           "ordinal": item.chunk.ordinal, "score": item.score} for item in result.ranked]
                        row["top_score"] = max((item.score for item in result.ranked), default=None)
                        row["retrieval"] = retrieval_metrics(row["ranked_ids"], groups)
                        if not retrieval_only:
                            minimum = thresholds.get(strategy)
                            if not result.selected or (minimum is not None and row["top_score"] < minimum):
                                row.update(refused=True, answer=FIXED_KNOWLEDGE_REFUSAL,
                                           refusal_reason="empty" if not result.selected else "threshold")
                            else:
                                stage, tick = "generation", time.perf_counter()
                                decision, citations = await runtime.generate(case.query, result.selected)
                                row["timings"]["generation_seconds"] = time.perf_counter() - tick
                                row["citations"] = [item.model_dump() for item in citations]
                                row["decision"] = decision.model_dump()
                                row["refused"] = not decision.useful
                                row["citation_valid"] = False if decision.useful else None
                                stage = "citation"
                                row["answer"], failure = decide_answer(decision, citations)
                                row["refusal_reason"] = "self_check" if failure else None
                                row["citation_valid"] = True if decision.useful else None
                                if decision.useful:
                                    stage, tick = "judge", time.perf_counter()
                                    judged = await runtime.judge(case.query, row["answer"], citations)
                                    if not judged.facts:
                                        raise FaithfulnessError("非拒答答案裁判没有可核对事实")
                                    numbers = {item.n for item in citations}
                                    if any(not set(fact.evidence_numbers).issubset(numbers)
                                           or (fact.supported and not fact.evidence_numbers) for fact in judged.facts):
                                        raise FaithfulnessError("裁判事实引用不合法")
                                    row["judgment"] = judged.model_dump()
                                    row["supported_count"] = sum(fact.supported for fact in judged.facts)
                                    row["total_count"] = len(judged.facts)
                                    row["faithfulness"] = row["supported_count"] / row["total_count"]
                                    row["timings"]["judge_seconds"] = time.perf_counter() - tick
                    except Exception as error:
                        # 不把依赖失败冒充拒答；异常文字可能带端点凭据，日志仅保留类型和阶段。
                        row["error"] = {"stage": stage, "type": type(error).__name__}
                    row["elapsed_seconds"] = time.perf_counter() - begin
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                    rows.append(row)
                    if progress:
                        progress(row)
        status = "completed_with_errors" if any(row["error"] for row in rows) else "completed"
    finally:
        report = {**summarize(rows, strategies), "status": status, "expected_sample_count": len(cases) * len(strategies),
                  "elapsed_seconds": time.perf_counter() - started, "metadata": runtime.metadata}
        atomic_json(output_dir / "summary.json", report)
        atomic_json(output_dir / "manifest.json", {**manifest, "status": status, "completed_sample_count": len(rows)})
        # Markdown是同源展示，JSON摘要和逐题日志为权威数据。
        (output_dir / "summary.md").write_text(markdown_report(report), encoding="utf-8")
    return report
