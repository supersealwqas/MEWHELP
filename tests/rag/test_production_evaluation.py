"""生产共享评估的证据标签隔离、故障传播与冻结校准边界。"""

import importlib
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core.model_budget import BudgetLedger
from app.knowledge.schemas import KnowledgeUnavailable


def production():
    """加载生产评估入口。参数：无；返回：模块；异常：未实现时测试失败。"""
    try:
        return importlib.import_module("app.evaluation.production")
    except ModuleNotFoundError:
        pytest.fail("生产共享评估尚未实现")


def test_sufficient_label_uses_selected_evidence_groups():
    """裁掉第二必需组后为不充分，空组也不能放行。参数：无；返回：None；异常：断言失败。"""
    label = production().selected_evidence_sufficient
    assert label([{1, 2}, {3}], [SimpleNamespace(chunk_id=1)]) is False
    assert label([{1, 2}, {3}], [SimpleNamespace(chunk_id=2), SimpleNamespace(chunk_id=3)]) is True
    assert label([], []) is False


@pytest.mark.asyncio
async def test_fault_never_becomes_low_confidence():
    """真实外部故障穿过共享管线，不能返回低分/空结果。参数：无；返回：None；异常：断言失败。"""
    production()
    from app.knowledge.pipeline import ProductionKnowledgePipeline

    class FailedSearch:
        """只在外部检索边界产生故障。"""
        async def retrieve(self, query, **kwargs):
            """模拟依赖故障。参数：查询与选项；返回：无；异常：KnowledgeUnavailable。"""
            raise KnowledgeUnavailable("检索依赖不可用")

    settings = Settings()
    pipeline = ProductionKnowledgePipeline(FailedSearch(), settings, query_expander=None)
    prepared = await pipeline.prepare("测试问题", {}, messages=[], budget=BudgetLedger.from_settings(settings))
    with pytest.raises(KnowledgeUnavailable):
        await pipeline.retrieve(prepared)


def test_runtime_refuses_answer_labels():
    """模型运行入口不接受参考答案或标签。参数：无；返回：None；异常：断言失败。"""
    runtime = production().ProductionEvaluationRuntime(None, Settings())
    with pytest.raises(TypeError):
        runtime.execute("问题", {}, retrieval_only=True, answerable=True)


def test_calibration_requires_both_actual_label_classes():
    """没有实际正负标签不能产出拍定制品。参数：无；返回：None；异常：断言失败。"""
    production()
    from app.evaluation.calibration import calibrate_evidence_policy
    result = calibrate_evidence_policy([], {"protocols": {}})
    assert result["policy"] is None and result["status"] == "insufficient_labels"


@pytest.mark.asyncio
async def test_full_ranked_precedes_cross_query_cap_and_audit_topk():
    """跨查询30条业务精排不能被10条裁剪或15条审计扩容改变。参数：无；返回：None；异常：断言失败。"""
    from tests.rag.test_retrieval import _chunk, _search
    from app.knowledge.snapshots import RetrievalCapture
    from app.knowledge.confidence import extract_features
    from app.workflow.routing import citations_from_ranked
    chunks = [_chunk(i) for i in range(1, 36)]
    queries = ["商品保修期限", "商品退换条件", "商品维修流程"]
    service, _, _, _ = _search(chunks, groups={query: list(range(start, start + 15)) for query, start in zip(queries, [1, 11, 21])})
    features = []
    for audit_top_k in (5, 15):
        full = []
        selected = await service.retrieve("商品保修期限和商品退换条件及维修流程", prepared_queries=queries,
            ranked_capture=full, capture=RetrievalCapture("test", str(audit_top_k), top_k=audit_top_k))
        assert len(selected) == 10 and len(full) == 30
        features.append(extract_features(full, citations_from_ranked(selected), minimum_score=0.5))
    assert features[0] == features[1]


def test_merge_cannot_hide_same_identity_conflicting_text():
    """跨查询取高分不能吞原文冲突。参数：无；返回：None；异常：断言失败。"""
    from tests.flywheel.test_confidence import ranked
    from app.knowledge.retrieval import merge_ranked_chunks
    with pytest.raises(ValueError, match="身份|原文"):
        merge_ranked_chunks([[ranked(score=0.1)], [ranked(score=0.9, answer="被替换原文")]])


def test_finite_calibration_compares_same_selected_labels_with_baseline():
    """最高分完全同分时，实际两组证据数量应能区分充分性。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.calibration import calibrate_evidence_policy
    rows = [{"id": str(i), "split": "calibration", "query": "测试问题", "error": None,
             "sufficient": i < 12, "selected_scores": [0.8, 0.75] if i < 12 else [0.8],
             "features": {"top1_score": 0.8, "top2_score": 0.75, "score_gap": 0.05,
                          "candidate_count": 2, "effective_count": 2 if i < 12 else 1}} for i in range(24)]
    result = calibrate_evidence_policy(rows, {"protocols": {"query": "test", "rerank": "test", "budget": "test"}})
    assert result["baseline"]["balanced_accuracy"] == 0.5
    assert result["chosen"]["balanced_accuracy"] == 1.0
    assert result["policy"] is not None and len(result["comparisons"]) == 4


@pytest.mark.asyncio
async def test_generation_is_rejected_before_paid_work():
    """未接Task5时拒绝伪装完整生成。参数：无；返回：None；异常：断言失败。"""
    with pytest.raises(NotImplementedError):
        await production().ProductionEvaluationRuntime(None, Settings()).execute("问题", {}, retrieval_only=False)


def test_threshold_selection_counts_mandatory_empty_refusal():
    """零候选拒答是硬规则，不能在阈值选优时被错计为误放行。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.calibration import _best_threshold
    threshold, key = _best_threshold([0.0, 0.0, 0.5], [
        {"sufficient": False, "features": {"candidate_count": 0}},
        {"sufficient": True, "features": {"candidate_count": 1}},
        {"sufficient": True, "features": {"candidate_count": 1}}])
    assert threshold == 0.0 and key[:2] == (4, 0)


@pytest.mark.asyncio
async def test_online_cache_binds_policy_version_and_keeps_single_gate(tmp_path):
    """同策略缓存复用、换策略重判必须保留新事实且不重复原检索。参数：临时目录；返回：None；异常：断言失败。"""
    import json
    from tests.rag.test_retrieval import _chunk, _search
    from tests.workflow.graph_fakes import write_test_evidence_policy, turn
    from app.tools.knowledge import retrieve_verified_knowledge
    search, _, index, _ = _search([_chunk(1)])
    path = tmp_path / "policy.json"
    settings = write_test_evidence_policy(search.config, path)
    state = turn("退货政策")
    options = dict(search=search, settings=settings, query_expander=None, query="退货政策", prepared_queries=["退货政策"])
    first = await retrieve_verified_knowledge(state, BudgetLedger.from_settings(settings), **options)
    repeated = await retrieve_verified_knowledge(state, BudgetLedger.from_settings(settings), **options)
    assert first.gate == repeated.gate and first.gate["schema_version"] == 2 and len(index.calls) == 1
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["version"] = "new-frozen-test-version"
    path.write_text(json.dumps(changed), encoding="utf-8")
    latest = await retrieve_verified_knowledge(state, BudgetLedger.from_settings(settings), **options)
    assert latest.gate["policy_version"] == "new-frozen-test-version" and len(index.calls) == 2
    assert len(latest.retrieval_snapshot.attempts) == 2


def test_policy_compatibility_uses_protocol_not_training_corpus(tmp_path):
    """知识更新不会强制重拟合，但预算协议变动必须拒绝。参数：临时目录；返回：None；异常：断言失败。"""
    from tests.flywheel.test_confidence import policy
    from app.knowledge.confidence import load_evidence_policy
    item = policy()
    path = tmp_path / "policy.json"
    path.write_text(item.model_dump_json(), encoding="utf-8")
    assert load_evidence_policy(path, protocols=item.protocols).version == item.version
    with pytest.raises(ValueError, match="不兼容"):
        load_evidence_policy(path, protocols={**item.protocols, "budget": "changed"})


@pytest.mark.asyncio
async def test_runner_preserves_model_budget_failure_kind(tmp_path, monkeypatch):
    """模型预算停止不能冒充检索故障或不充分标签。参数：临时目录/外部依赖替身；返回：None；异常：断言失败。"""
    import json
    from contextlib import asynccontextmanager
    from pathlib import Path
    from app.core.model_budget import WorkflowBudgetExceeded
    from app.evaluation.dataset import _source_chunks
    import app.evaluation.runtime as legacy_runtime
    module, chunks = production(), _source_chunks()

    class BudgetStoppedRuntime:
        """只在付费前的预算边界停止。"""
        last_attempt = {"stage": "understand", "ledger": BudgetLedger()}

        async def execute(self, query, filters, *, retrieval_only):
            """模拟预算限制。参数：原问题/过滤/模式；返回：无；异常：WorkflowBudgetExceeded。"""
            raise WorkflowBudgetExceeded("模型预算不足")

    class ReadOnlyStore:
        """提供不发生变化的独立完整语料。"""
        async def active_page(self, *, after_id, limit):
            """只读测试快照。参数：分页游标与数量；返回：候选页；异常：无。"""
            return [row for row in chunks if row.id > after_id][:limit]

    @asynccontextmanager
    async def local_runtime(settings, *, target):
        """替换模型/数据库外部边界。参数：设置/目标；返回：独立运行依赖；异常：无。"""
        yield BudgetStoppedRuntime(), ReadOnlyStore(), chunks, None

    async def verified(index, chunks):
        """代替外部索引指纹读取。参数：索引/语料；返回：None；异常：无。"""
        return None

    monkeypatch.setattr(module, "production_runtime", local_runtime)
    monkeypatch.setattr(legacy_runtime, "_verify_vectors", verified)
    args = SimpleNamespace(command="calibrate-production", output_dir=tmp_path / "budget-report",
        dataset=Path("data/evaluation/ch04.json"), target="application-readonly")
    assert await module.run_production(args, Settings()) == 2
    rows = [json.loads(line) for line in (args.output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 24 and all(row["error"]["kind"] == "budget" and row["sufficient"] is None for row in rows)
