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


@pytest.mark.parametrize("change", ["endpoint", "weights"])
def test_model_identity_binds_endpoint_and_actual_local_weights(monkeypatch, change):
    """模型同名但端点或本地权重变化必须进入不同比较组。参数：替身/变化种类；返回：None；异常：断言失败。"""
    module = production()
    fingerprint = "a" * 64
    monkeypatch.setattr("app.evaluation.runtime.model_digest", lambda path: fingerprint)
    settings = Settings()
    before = module._production_signature(settings)
    if change == "endpoint":
        settings = settings.model_copy(update={"chat_base_url": "https://other.invalid/v1"})
    else:
        fingerprint = "b" * 64
    assert module._production_signature(settings)["models"] != before["models"]


@pytest.mark.asyncio
async def test_explicit_frozen_policy_controls_generation_and_keeps_partial_retrieval():
    """CLI指定策略不能被settings覆盖，生成故障不能抹掉已成功检索。参数：无；返回：None；异常：断言失败。"""
    from tests.workflow.graph_fakes import dependencies, ranked
    from tests.flywheel.test_confidence import policy
    deps = dependencies(intents=["商品咨询"], ranked=[ranked()], answers=[["依据政策。[1]"]])
    runtime = production().ProductionEvaluationRuntime(deps.search, deps.settings,
        model=deps.model, query_expander=deps.query_expander)
    runtime.frozen_policy = policy().model_copy(update={"minimum_score": 0.99, "threshold": 0.99})
    result = await runtime.execute("有什么条件？", {}, retrieval_only=False)
    assert result["refused"] is True
    assert result["gate"]["policy_version"] == runtime.frozen_policy.version
    assert runtime.last_attempt["result"]["ranked"] == result["ranked"]


@pytest.mark.parametrize("failure", [None, "protocol", "stream"])
async def test_judge_optional_observer_keeps_real_usage_on_all_exits(failure):
    """成功、协议失败和部分流失败都保留真实usage，不改变裁判返回类型。参数：错误类型；返回：None；异常：断言失败。"""
    from langchain_core.messages import AIMessageChunk
    from app.evaluation.faithfulness import FaithfulnessJudge, FaithfulnessDecision
    from tests.rag.test_faith_cases import _citation
    records = []

    class Model:
        """替换远程流，运行真实结构化裁判。"""
        def bind_tools(self, *args, **kwargs):
            """绑定控制工具。参数：工具选项；返回：自身；异常：无。"""
            return self

        async def astream(self, messages):
            """产生实际用量分片及可控错误。参数：消息；返回：分片流；异常：ConnectionError。"""
            yield AIMessageChunk(content="", tool_calls=[{"name": "bad" if failure == "protocol" else "submit_faithfulness",
                "id": "j1", "args": {"facts": [{"claim": "事实", "supported": True, "reason": "原文", "evidence_numbers": [1]}]}}],
                usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                response_metadata={"finish_reason": "tool_calls"})
            if failure == "stream":
                raise ConnectionError("远程断流")

    judge = FaithfulnessJudge(lambda **options: Model(), usage_observer=records.append)
    if failure:
        with pytest.raises(Exception):
            await judge.judge("问题", "事实[1]", [_citation()])
    else:
        assert isinstance(await judge.judge("问题", "事实[1]", [_citation()]), FaithfulnessDecision)
    assert len(records) == 1 and records[0]["usage"]["total_tokens"] == 14


@pytest.mark.parametrize("drift", [False, True])
async def test_complete_runner_preserves_generation_and_judge_errors(tmp_path, monkeypatch, drift):
    """成功检索不可被后续故障抹掉，裁判必须拿两条selected，版本变化不能完成。参数：目录/外部替身/漂移；返回：None；异常：断言失败。"""
    import json
    from contextlib import asynccontextmanager
    from pathlib import Path
    from app.evaluation.dataset import _source_chunks, load_dataset, resolve_evidence
    from app.knowledge.rerank import RankedChunk
    from app.workflow.routing import citations_from_ranked
    from app.evaluation.runner import validate_completed_artifact
    from app.evaluation.faithfulness import FaithfulnessDecision, FactJudgment
    module = production()
    assert hasattr(module, "evaluate_production"), "缺少完整生产执行入口"
    chunks = _source_chunks()
    cases = [case for case in load_dataset(Path("data/evaluation/ch04.json"), chunks=chunks) if case.split == "evaluation"]
    by_query = {case.query: case for case in cases}

    class Runtime:
        """仅替换付费模型与外部检索资源。"""
        last_attempt = {}
        last_judge_usage = None

        async def execute(self, query, filters, *, retrieval_only):
            """给真实汇总器提供同轮排名和部分生成结果。参数：真实查询参数；返回：字典；异常：首题生成故障。"""
            assert not retrieval_only
            case = by_query[query]
            groups = resolve_evidence(case, chunks)
            ids = list(dict.fromkeys([identifier for group in groups for identifier in group] + [chunks[-1].id]))
            ranked = [RankedChunk(chunk=next(chunk for chunk in chunks if chunk.id == identifier), score=0.9) for identifier in ids]
            selected = citations_from_ranked(ranked)
            result = {"ranked": ranked, "selected": selected, "snapshot": None, "budget": {"records": []}}
            self.last_attempt = {"stage": "generate", "result": result}
            if case.id == cases[0].id:
                raise ConnectionError("生成失败")
            if not case.answerable:
                result.update(answer="缺少足够证据", refused=True, citations=[], usage={"records": []})
                return result
            result.update(answer="事实[1]", refused=False, citations=selected[:1], usage={"records": []})
            return result

        async def judge(self, query, answer, selected):
            """检查完整证据并模拟独立裁判失败。参数：问答及全集；返回：裁判；异常：第二题故障。"""
            self.last_judge_usage = {"usage": {"total_tokens": 9}}
            assert by_query[query].answerable, "拒答不得调用额外裁判"
            if query == cases[1].query:
                assert len(selected) >= 2
                raise ConnectionError("裁判失败")
            return FaithfulnessDecision(facts=[FactJudgment(claim="事实", supported=True, reason="证据", evidence_numbers=[1])])

    class Store:
        """读当前完整知识，按参数模拟运行后变化。"""
        async def active_page(self, *, after_id, limit):
            """读取实际分页。参数：游标/数量；返回：快照页；异常：无。"""
            return [chunk.model_copy(update={"content_hash": "f" * 64}) if drift else chunk for chunk in chunks if chunk.id > after_id][:limit]

    @asynccontextmanager
    async def resources(settings, *, target):
        """代替真实外部资源入口。参数：配置/目标；返回：依赖；异常：无。"""
        yield Runtime(), Store(), chunks, None

    async def verified(*args):
        """外部索引验证替身。参数：索引/语料；返回：None；异常：无。"""

    monkeypatch.setattr(module, "production_runtime", resources)
    monkeypatch.setattr("app.evaluation.runtime.model_digest", lambda path: "a" * 64)
    monkeypatch.setattr("app.evaluation.runtime._verify_vectors", verified)
    args = SimpleNamespace(output_dir=tmp_path / "run", policy=Path("data/knowledge/evidence-policy.json"),
        dataset=Path("data/evaluation/ch04.json"), target="application-readonly", triggered_by="手动")
    if drift:
        with pytest.raises(ValueError):
            await module.evaluate_production(args, Settings(), checkpoint=verified)
        assert json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "invalidated"
    else:
        await module.evaluate_production(args, Settings(), checkpoint=verified)
        summary = validate_completed_artifact(args.output_dir)
        rows = [json.loads(line) for line in (args.output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        assert summary["error_count"] == 2 and summary["aggregate"]["recall@10"]["denominator"] == 40
        assert summary["refusal_count"] == 8 and summary["aggregate"]["faithfulness_macro"]["denominator"] == 38
        assert rows[0]["retrieval"]["recall@10"] == 1.0 and rows[0]["refused"] is None
        assert rows[1]["answer"] == "事实[1]" and rows[1]["error"]["stage"] == "judge"
        assert rows[1]["judge_usage"]["usage"]["total_tokens"] == 9


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
async def test_generation_reuses_shared_pipeline_without_chat_side_effects():
    """完整模式必须生成真实结果且无聊天写入，保留实际selected。参数：无；返回：None；异常：断言失败。"""
    from tests.workflow.graph_fakes import dependencies, ranked
    deps = dependencies(intents=["商品咨询"], ranked=[ranked()], answers=[["依据当前政策。[1]"]])
    runtime = production().ProductionEvaluationRuntime(deps.search, deps.settings,
        model=deps.model, query_expander=deps.query_expander)
    result = await runtime.execute("有什么条件？", {}, retrieval_only=False)
    assert result["answer"] == "依据当前政策。[1]" and result["refused"] is False
    assert result["citations"] == result["selected"] and len(result["selected"]) == 1
    assert result["usage"]["model_calls"] == 3 and result["budget"] == result["usage"]
    assert not deps.chat.finished and not deps.chat.calls and not deps.chat.results


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
