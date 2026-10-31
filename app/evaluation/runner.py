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

PRODUCTION_SCOPE = "production_knowledge_subpipeline"
PRODUCTION_SCHEMA_VERSION = 1


def production_comparison_key(metadata: dict) -> str:
    """只将不可混用的数据/模型/策略/指标协议组成比较键，知识版本独立展示。

    参数：完整生产元信息；返回：SHA256；异常：缺少必需协议时KeyError。
    """
    from app.evaluation.policy import digest_json
    fields = ("dataset_digest", "split", "case_ids", "policy_sha256", "protocols", "models",
              "generation_protocol", "judge_protocol", "budget_config", "retrieval_config")
    return digest_json({"schema_version": PRODUCTION_SCHEMA_VERSION, "scope": PRODUCTION_SCOPE,
        **{key: metadata[key] for key in fields}})


def production_summary(rows: list[dict], metadata: dict, artifact_id: str, *, status: str | None = None) -> dict:
    """按全部逐题记录复用CH04指标，并独立统计线上路径和测量裁判用量。

    参数：rows含错误的完整逐题、metadata冻结口径、artifact_id本轮身份、status可覆盖失效/中断状态。
    返回：可展示且带原分母的指标；异常：错误数据结构或不可序列化值。
    """
    aggregate = aggregate_results(rows)
    usage = {}
    for name in ("production", "judge"):
        records = [record for row in rows for record in ((row.get("budget") or {}).get("records", [])
            if name == "production" else ([row["judge_usage"]] if row.get("judge_usage") is not None else []))]
        values = [(record.get("usage") or {}).get("total_tokens") for record in records]
        known = [value for value in values if type(value) is int]
        usage[name] = {"known_actual_tokens": sum(known), "unknown_usage_records": len(values) - len(known),
            "total_tokens": sum(known) if len(known) == len(values) else None, "request_count": len(records)}
    return {"schema_version": PRODUCTION_SCHEMA_VERSION, "artifact_id": artifact_id,
        "scope": PRODUCTION_SCOPE, "status": status or ("completed_with_errors" if aggregate["error_count"] else "completed"),
        "dataset_size": len(rows), "error_count": aggregate["error_count"],
        "refusal_count": sum(row.get("refused") is True for row in rows),
        "recall_at_k": aggregate["recall@10"]["value"], "recall_k": 10,
        "mrr": aggregate["mrr@10"]["value"], "faithfulness": aggregate["faithfulness_macro"]["value"],
        "aggregate": aggregate, "usage": usage, "metadata": metadata,
        "comparison_key": production_comparison_key(metadata),
        "limitation": "CH04原72题（24校准/48evaluation）；本轮为冻结策略重复监测，仅覆盖生产知识子流程，无订单标签，不代表整张客服图。新增等价知识的证据标签仍需人工复核。"}


def validate_completed_artifact(artifact_dir: Path) -> dict:
    """离线联合核验完成标记、冻结题集/语料/策略、逐题身份及重算指标。

    参数：artifact_dir已完成独占目录；返回：可保存的summary及内容指纹。
    异常：ValueError，任何损坏、未完成、混版本、指标不符；OSError，文件无法读取。
    """
    from app.evaluation.dataset import load_dataset, resolve_evidence
    from app.evaluation.policy import digest_json
    from app.knowledge.schemas import ChunkSnapshot
    from app.knowledge.confidence import EvidencePolicy
    from app.knowledge.answering import EvidenceCitation
    from app.knowledge.rerank import RankedChunk
    from app.evaluation.faithfulness import FaithfulnessDecision
    from hashlib import sha256
    try:
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
        raw = (artifact_dir / "results.jsonl").read_text(encoding="utf-8")
        if not raw.endswith("\n"):
            raise ValueError("逐题文件末行未完成")
        rows = [json.loads(line) for line in raw.splitlines()]
        corpus = json.loads((artifact_dir / "corpus.json").read_text(encoding="utf-8"))
        policy = json.loads((artifact_dir / "policy.json").read_text(encoding="utf-8"))
        EvidencePolicy.model_validate(policy)
        chunks = [ChunkSnapshot.model_validate(row) for row in corpus]
        frozen_chunks = {chunk.id: chunk for chunk in chunks}
        cases = load_dataset(artifact_dir / "dataset.json", chunks=chunks)
        dataset = [case.model_dump(mode="json") for case in cases]
        metadata = manifest["metadata"]
        expected = [case for case in cases if case.split == "evaluation"]
        identity = manifest["artifact_id"]
        if not isinstance(identity, str) or not identity or len(identity) > 128:
            raise ValueError("产物身份非法")
        if (manifest["status"] not in {"completed", "completed_with_errors"}
                or manifest["schema_version"] != PRODUCTION_SCHEMA_VERSION or manifest["scope"] != PRODUCTION_SCOPE
                or manifest["expected_sample_count"] != 48 or manifest["completed_sample_count"] != 48
                or len(rows) != 48 or metadata["split"] != "evaluation"
                or metadata["case_ids"] != [case.id for case in expected]
                or [row["id"] for row in rows] != metadata["case_ids"]):
            raise ValueError("产物未完整完成或题目身份不一致")
        before, after = metadata["versions_before"], metadata["versions_after"]
        if (before != after or before["knowledge"] != digest_json(corpus)
                or before["knowledge"] != metadata["corpus_version"] or before["policy"] != digest_json(policy)
                or before["policy"] != metadata["policy_sha256"] or before["protocols"] != metadata["protocols"]
                or metadata["dataset_digest"] != digest_json(dataset)
                or metadata["dataset_sha256"] != sha256((artifact_dir / "dataset.json").read_bytes()).hexdigest()
                or metadata["triggered_by"] not in {"定时", "手动"}):
            raise ValueError("产物版本失效或冻结文件不符")
        for case, row in zip(expected, rows, strict=True):
            if row["artifact_id"] != identity or any(row[key] != getattr(case, key) for key in
                    ("split", "query", "filters", "bucket", "difficulty", "answerable")):
                raise ValueError("逐题身份或标注串线")
            if row["refused"] is not None and type(row["refused"]) is not bool:
                raise ValueError("拒答状态非法")
            if row["error"] is None and (row["refused"] is None or row["retrieval"] is None):
                raise ValueError("无错误样本缺少实际执行结果")
            if row["error"] is None and row["refused"] is False and row["faithfulness"] is None:
                raise ValueError("成功非拒答样本必须包含裁判结果，否则应记录真实错误")
            ranked = [RankedChunk.model_validate(item) for item in row["ranked"]]
            # 同一主键不足以证明同一知识版本；逐项比较冻结快照的原文及全部身份字段。
            if any(item.chunk != frozen_chunks.get(item.chunk.id) for item in ranked):
                raise ValueError("ranked原文或身份与冻结corpus不符")
            selected = [EvidenceCitation.model_validate(item) for item in row["selected"]]
            citations = [EvidenceCitation.model_validate(item) for item in row["citations"]]
            known = {item.chunk.id: item.chunk for item in ranked}
            if len(known) != len(ranked) or len({item.n for item in selected}) != len(selected):
                raise ValueError("排名或selected编号重复")
            for item in selected:
                chunk = known.get(item.chunk_id)
                if (chunk is None or item.answer != chunk.text.answer or item.question != chunk.text.questions
                        or item.section_path != chunk.section_path or item.source_key != chunk.source_key):
                    raise ValueError("selected与本轮ranked原文身份不符")
            if any(item not in selected for item in citations):
                raise ValueError("公开引用不是本轮完整selected的子集")
            if row["retrieval"] is not None:
                groups = resolve_evidence(case, chunks)
                if row["retrieval"] != retrieval_metrics(row["ranked_ids"], groups):
                    raise ValueError("逐题召回指标与实际排名不符")
                if row["ranked_ids"] != [item["chunk"]["id"] for item in row["ranked"]]:
                    raise ValueError("排名身份与原始检索不符")
            # persist-run独立读取文件，必须重新执行在线裁判的严格类型/字段校验。
            judgment = FaithfulnessDecision.model_validate(row["judgment"], strict=True) if row.get("judgment") is not None else None
            if (type(row["supported_count"]) is not int or type(row["total_count"]) is not int
                    or not 0 <= row["supported_count"] <= row["total_count"]):
                raise ValueError("裁判事实计数必须为非负整数且支持数不超过总数")
            if row["faithfulness"] is not None:
                if type(row["faithfulness"]) not in (int, float) or not math.isfinite(row["faithfulness"]) or not 0 <= row["faithfulness"] <= 1:
                    raise ValueError("忠实度必须是0到1的有限数值，不能用布尔值代替")
                facts = judgment.facts if judgment is not None else []
                numbers = {item["n"] for item in row["selected"]}
                if (not facts or row["refused"] is not False
                        or any(not set(fact.evidence_numbers).issubset(numbers)
                            or (fact.supported and not fact.evidence_numbers) for fact in facts)
                        or row["total_count"] != len(facts) or row["supported_count"] != sum(fact.supported for fact in facts)
                        or row["faithfulness"] != row["supported_count"] / len(facts)):
                    raise ValueError("逐题忠实度与完整裁判不符")
            elif row["total_count"] != 0 or row["supported_count"] != 0:
                raise ValueError("缺失裁判不得贡献事实分母")
        rebuilt = production_summary(rows, metadata, identity)
        if (summary != rebuilt or manifest["status"] != summary["status"]
                or manifest["summary_digest"] != digest_json(summary) or manifest["results_digest"] != digest_json(rows)):
            raise ValueError("汇总/完成标记与逐题重算结果不符")
        json.dumps({"manifest": manifest, "summary": summary, "rows": rows}, allow_nan=False)
        return {**summary, "artifact_digest": digest_json({"manifest": manifest, "results": rows})}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("产物结构不完整") from error


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
                preparation_attempted, preparation_error = False, None
                for strategy in strategies:
                    begin, stage = time.perf_counter(), "dataset"
                    row = {"id": case.id, "split": case.split, "bucket": case.bucket, "difficulty": case.difficulty,
                           "strategy": strategy, "query": case.query, "answerable": case.answerable,
                           "refused": None, "refusal_reason": None, "answer": None,
                           "error": None, "faithfulness": None, "supported_count": 0, "total_count": 0,
                           "retrieval": None, "ranked_ids": [], "top_score": None, "citations": [],
                           "citation_valid": None, "timings": {}}
                    try:
                        stage = "query"
                        if not preparation_attempted:
                            preparation_attempted = True
                            try:
                                prepared = await runtime.prepare(case.query)
                            except Exception as error:
                                # 失败也共用同一准备结果，不能每策略重新调用模型改变比较条件。
                                preparation_error = error
                        if preparation_error is not None:
                            raise preparation_error
                        stage = "dataset"
                        groups = evidence_resolver(case, runtime.chunks)
                        row["evidence_groups"] = [sorted(group) for group in groups]
                        row["prepared_queries"] = prepared.groups
                        row["prepared_bm25_queries"] = prepared.bm25_queries
                        row["prepared_rerank_queries"] = prepared.rerank_queries
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
