"""完整生产评估产物与固定指标合同；真实跨进程保存另在专库测试验证。"""

import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from app.evaluation.policy import digest_json
from app.evaluation.runner import atomic_json


def runner_module():
    """取得评估产物接口。参数：无；返回：模块；异常：缺失时断言失败。"""
    from app.evaluation import runner
    assert hasattr(runner, "production_summary"), "缺少完整生产评估指标与产物协议"
    return runner


def completed_artifact(path, triggered_by="手动"):
    """构造含48真实题身份和真实源块的离线错误产物，不调用模型。参数：目录；返回：目录；异常：文件错误。"""
    from app.evaluation.dataset import _source_chunks, load_dataset
    module = runner_module()
    chunks = _source_chunks()
    cases = load_dataset(Path("data/evaluation/ch04.json"), chunks=chunks)
    dataset = [case.model_dump(mode="json") for case in cases]
    corpus = [chunk.model_dump(mode="json") for chunk in chunks]
    policy = json.loads(Path("data/knowledge/evidence-policy.json").read_text(encoding="utf-8"))
    versions = {"knowledge": digest_json(corpus), "policy": digest_json(policy),
        "protocols": {"query": "fixed-query", "rerank": "fixed-rerank", "budget": "fixed-budget"}}
    metadata = {"dataset_digest": digest_json(dataset), "dataset_sha256": "a" * 64, "split": "evaluation", "triggered_by": triggered_by,
        "case_ids": [case.id for case in cases if case.split == "evaluation"],
        "corpus_version": versions["knowledge"], "policy_sha256": versions["policy"],
        "protocols": versions["protocols"], "models": {"generation": "test-model", "judge": "test-judge"},
        "generation_protocol": "test-generation", "judge_protocol": "test-judge-protocol",
        "budget_config": {"max_model_calls": 8}, "retrieval_config": {"top_k": 10},
        "versions_before": versions, "versions_after": versions}
    rows = [{"id": case.id, "split": case.split, "query": case.query, "filters": case.filters,
        "bucket": case.bucket, "difficulty": case.difficulty, "answerable": case.answerable,
        "artifact_id": "test-artifact", "refused": None, "answer": None,
        "error": {"stage": "understand", "type": "ConnectionError"}, "faithfulness": None,
        "supported_count": 0, "total_count": 0, "retrieval": None, "ranked_ids": [],
        "ranked": [], "selected": [], "citations": [], "citation_valid": None,
        "budget": None, "judge_usage": None} for case in cases if case.split == "evaluation"]
    path.mkdir()
    atomic_json(path / "dataset.json", dataset)
    from hashlib import sha256
    metadata["dataset_sha256"] = sha256((path / "dataset.json").read_bytes()).hexdigest()
    atomic_json(path / "corpus.json", corpus)
    atomic_json(path / "policy.json", policy)
    (path / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = module.production_summary(rows, metadata, "test-artifact")
    atomic_json(path / "summary.json", summary)
    atomic_json(path / "manifest.json", {"schema_version": 1, "artifact_id": "test-artifact",
        "scope": "production_knowledge_subpipeline", "status": summary["status"],
        "expected_sample_count": 48, "completed_sample_count": 48, "metadata": metadata,
        "summary_digest": digest_json(summary), "results_digest": digest_json(rows)})
    return path


def rewrite_artifact(path, *, rows=None, metadata=None):
    """为深层校验测试重算外层摘要，确保不是仅由摘要不符拦截。参数：目录/可选数据；返回：None；异常：文件错误。"""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if rows is None:
        rows = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = runner_module().production_summary(rows, metadata or manifest["metadata"], manifest["artifact_id"])
    manifest.update(metadata=summary["metadata"], status=summary["status"],
        summary_digest=digest_json(summary), results_digest=digest_json(rows))
    (path / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    atomic_json(path / "manifest.json", manifest)
    atomic_json(path / "summary.json", summary)


def test_original_dataset_bytes_must_match_recorded_identity(tmp_path):
    """只改原文件身份且重算外层摘要不能通过本地完整验证。参数：目录；返回：None；异常：断言失败。"""
    path = completed_artifact(tmp_path / "run")
    metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))["metadata"]
    metadata["dataset_sha256"] = "b" * 64
    rewrite_artifact(path, metadata=metadata)
    with pytest.raises(ValueError):
        runner_module().validate_completed_artifact(path)


@pytest.mark.parametrize("damage", ["missing_judge", "selected_identity", "published_identity"])
def test_answer_rows_require_complete_judge_and_same_evidence(tmp_path, damage):
    """成功答复缺裁判或证据串线即使摘要自洽也必须拒存。参数：目录/损坏方式；返回：None；异常：断言失败。"""
    from app.evaluation.dataset import load_dataset, resolve_evidence
    from app.evaluation.metrics import retrieval_metrics
    from app.knowledge.schemas import ChunkSnapshot
    from app.knowledge.rerank import RankedChunk
    from app.workflow.routing import citations_from_ranked
    path = completed_artifact(tmp_path / "run")
    chunks = [ChunkSnapshot.model_validate(row) for row in json.loads((path / "corpus.json").read_text(encoding="utf-8"))]
    case = next(case for case in load_dataset(path / "dataset.json", chunks=chunks) if case.split == "evaluation")
    groups = resolve_evidence(case, chunks)
    ranked = [RankedChunk(chunk=next(chunk for chunk in chunks if chunk.id == next(iter(groups[0]))), score=0.9)]
    selected = [item.model_dump(mode="json") for item in citations_from_ranked(ranked)]
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0].update(error=None, answer="事实[1]", refused=False, citation_valid=True,
        ranked=[item.model_dump(mode="json") for item in ranked], ranked_ids=[ranked[0].chunk.id],
        retrieval=retrieval_metrics([ranked[0].chunk.id], groups), selected=copy.deepcopy(selected), citations=copy.deepcopy(selected))
    if damage != "missing_judge":
        rows[0].update(faithfulness=1.0, total_count=1, supported_count=1,
            judgment={"facts": [{"claim": "事实", "supported": True, "reason": "原文", "evidence_numbers": [1]}]})
        rows[0]["selected" if damage == "selected_identity" else "citations"][0]["chunk_id"] = 999999
    rewrite_artifact(path, rows=rows)
    with pytest.raises(ValueError):
        runner_module().validate_completed_artifact(path)


def test_partial_errors_keep_denominators_and_status(tmp_path):
    """48题全故障仍保留40可答题召回分母，缺失裁判不可变成零/满分。参数：目录；返回：None；异常：断言失败。"""
    path = completed_artifact(tmp_path / "run")
    summary = runner_module().validate_completed_artifact(path)
    assert summary["dataset_size"] == 48 and summary["status"] == "completed_with_errors"
    assert summary["error_count"] == 48 and summary["refusal_count"] == 0
    assert summary["aggregate"]["recall@10"] == {"value": 0.0, "numerator": 0, "denominator": 40}
    assert summary["aggregate"]["faithfulness_macro"]["denominator"] == 0
    assert summary["faithfulness"] is None


@pytest.mark.parametrize("damage", ["running", "invalidated", "truncated", "duplicate", "identity", "summary", "mixed_version", "metric"])
def test_unfinished_run_is_not_recorded(tmp_path, damage):
    """任何未完成/混版本/身份串线/汇总篡改均不得进入数据库。参数：目录/损坏方式；返回：None；异常：断言失败。"""
    module = runner_module()
    path = completed_artifact(tmp_path / "run")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    if damage in {"running", "invalidated"}:
        manifest["status"] = damage
    elif damage == "mixed_version":
        manifest["metadata"]["versions_after"]["knowledge"] = "changed"
    elif damage == "truncated":
        rows.pop()
    elif damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "identity":
        rows[0]["query"] = "被替换的问题"
    elif damage == "metric":
        rows[0]["retrieval"] = {"recall@10": 1.0}
    else:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        summary["recall_at_k"] = 1.0
        atomic_json(path / "summary.json", summary)
        manifest["summary_digest"] = digest_json(summary)
    (path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest["results_digest"] = digest_json(rows)
    atomic_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        module.validate_completed_artifact(path)


def test_knowledge_revision_changes_are_comparable_but_dataset_changes_are_not(tmp_path):
    """知识升级应允许趋势比较，题目/生成协议变化必须分组。参数：目录；返回：None；异常：断言失败。"""
    module = runner_module()
    metadata = json.loads((completed_artifact(tmp_path / "run") / "summary.json").read_text(encoding="utf-8"))["metadata"]
    newer = copy.deepcopy(metadata)
    newer["corpus_version"] = "new-knowledge"
    newer["versions_before"]["knowledge"] = newer["versions_after"]["knowledge"] = "new-knowledge"
    assert module.production_comparison_key(metadata) == module.production_comparison_key(newer)
    for field in ("dataset_digest", "policy_sha256", "generation_protocol", "judge_protocol"):
        changed = {**newer, field: "different"}
        assert module.production_comparison_key(metadata) != module.production_comparison_key(changed)


async def test_invalid_artifact_never_acquires_database_lease(tmp_path, monkeypatch):
    """本地核验失败后不能取得写锁或写入数据库。参数：目录/替身；返回：None；异常：断言失败。"""
    assert importlib.util.find_spec("app.repositories.evaluations"), "缺少评估仓储"
    repository = importlib.import_module("app.repositories.evaluations")
    path = completed_artifact(tmp_path / "run")
    (path / "manifest.json").write_text("{}", encoding="utf-8")

    async def forbidden(*args, **kwargs):
        """任何数据库动作都说明顺序错误。参数：任意；返回：无；异常：AssertionError。"""
        raise AssertionError("不完整产物不得触碰数据库")

    monkeypatch.setattr(repository.MySqlWriteLease, "acquire", forbidden)
    with pytest.raises(ValueError):
        await repository.EvaluationRepository(None, None).save_completed_run(path, "手动")
