"""离线评估运行器：完整证据、门禁前指标、原子报告及裁判失败分母。"""

import json
from types import SimpleNamespace

import pytest

from app.knowledge.answering import EvidenceCitation, KnowledgeAnswer
from app.knowledge.retrieval import EvaluationRetrieval, RetrievalCandidate
from tests.rag.test_retrieval import _chunk


class RuntimeBoundary:
    """替换数据库与模型边界，运行真实统计与日志实现。"""

    def __init__(self, *, refuse=False, judge_error=False):
        """配置外部响应。参数：refuse 拒答、judge_error 裁判故障；返回：None；异常：无。"""
        self.chunks = [_chunk(1)]
        self.refuse, self.judge_error, self.inputs = refuse, judge_error, []
        self.metadata = {"corpus_digest": "fixture", "model": "test", "prompt_digest": "fixture"}

    async def assert_held(self):
        """提供稳定的外部持锁边界。参数：无；返回：None；异常：无。"""

    async def prepare(self, query):
        """替换外部改写模型，保留确定准备。参数：原问；返回：准备查询；异常：ValueError。"""
        from app.knowledge.retrieval import prepare_query

        return prepare_query(query, policy=any(term in query for term in ("退款", "退货", "退换")))

    async def retrieve(self, prepared, strategy, filters):
        """返回门禁前低分命中。参数：查询/策略/过滤；返回：结果；异常：无。"""
        item = RetrievalCandidate(chunk=self.chunks[0], score=0.01)
        return EvaluationRetrieval(ranked=[item], selected=[item])

    async def generate(self, query, selected):
        """只接收原问及实际候选，不接受标注答案。参数：原问/候选；返回：自评和快照；异常：无。"""
        self.inputs.append((query, selected))
        citation = EvidenceCitation(n=1, chunk_id=1, question="问题", answer="权威答案1。")
        return KnowledgeAnswer(useful=not self.refuse, reason="证据不足" if self.refuse else "",
                               answer="" if self.refuse else "权威答案1。[1]",
                               citation_numbers=[] if self.refuse else [1]), [citation]

    async def judge(self, query, answer, citations):
        """返回真实裁判协议或外部故障。参数：问答/证据；返回：Decision；异常：RuntimeError。"""
        from app.evaluation.faithfulness import FaithfulnessDecision

        if self.judge_error:
            raise RuntimeError("upstream failure")
        return FaithfulnessDecision(facts=[])


async def test_one_rewrite_is_shared_by_four_strategies(tmp_path):
    """逐策略重做prepare会增加费用并污染比较。参数：临时目录；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset
    from app.knowledge.retrieval import prepare_query

    runtime = RuntimeBoundary()
    calls, prepared_objects = [], []
    retrieve = runtime.retrieve

    async def prepare(query):
        """记录唯一模型准备入口。参数：原问；返回：准备结果；异常：无。"""
        calls.append(query)
        return prepare_query(query)

    async def record(prepared, strategy, filters):
        """记录传给召回的实际准备对象。参数：查询/策略/过滤；返回：候选；异常：无。"""
        prepared_objects.append(prepared)
        return await retrieve(prepared, strategy, filters)

    runtime.prepare, runtime.retrieve = prepare, record
    directory = tmp_path / "shared"
    await evaluate_dataset([case()], runtime, directory, retrieval_only=True,
                           evidence_resolver=lambda case, chunks: [{1}])
    assert calls == ["退货申请条件"]
    assert len(prepared_objects) == 4 and all(item is prepared_objects[0] for item in prepared_objects)
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["prepared_bm25_queries"] == rows[0]["prepared_bm25_queries"] for row in rows)


async def test_failed_rewrite_is_not_retried_for_each_strategy(tmp_path):
    """改写失败仍保留四个异常分母，但不能重复发模型请求。参数：临时目录；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset

    runtime, calls = RuntimeBoundary(), []

    async def fail(query):
        """模拟网络准备失败。参数：原问；返回：不返回；异常：RuntimeError。"""
        calls.append(query)
        raise RuntimeError("rewrite failed")

    runtime.prepare = fail
    report = await evaluate_dataset([case()], runtime, tmp_path / "failed", retrieval_only=True,
                                     evidence_resolver=lambda case, chunks: [{1}])
    assert calls == ["退货申请条件"]
    assert all(value["error_count"] == 1 for value in report["strategies"].values())


def case():
    """构造仅运行器关心的题字段。参数：无；返回：题；异常：无。"""
    return SimpleNamespace(id="A-1", split="evaluation", fact_group="f", bucket="A_policy",
                           query="退货申请条件", filters={}, answerable=True,
                           difficulty="easy",
                           reference_answer="此标注不得传给模型", required_facts=["不得传模型"])


@pytest.mark.parametrize("judge_error", [False, True])
async def test_judge_errors_remain_samples_and_do_not_erase_retrieval(tmp_path, judge_error) -> None:
    """网络故障或非拒答空事实均记错误，已生成正文和排名保留。参数：tmp_path/judge_error；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset

    runtime = RuntimeBoundary(judge_error=judge_error)
    directory = tmp_path / "unique"
    report = await evaluate_dataset([case()], runtime, directory, strategies=["bm25"],
                                     evidence_resolver=lambda case, chunks: [{1}])
    row = json.loads((directory / "results.jsonl").read_text(encoding="utf-8"))
    assert report["strategies"]["bm25"]["sample_count"] == 1
    assert report["strategies"]["bm25"]["error_count"] == 1
    assert row["retrieval"]["recall@1"] == 1
    assert row["error"]["stage"] == "judge" and row["faithfulness"] is None
    assert row["answer"] == "权威答案1。[1]"
    assert runtime.inputs[0][0] == "退货申请条件"
    with pytest.raises(FileExistsError):
        await evaluate_dataset([case()], runtime, directory, strategies=["bm25"], evidence_resolver=lambda c, s: [{1}])


async def test_threshold_gate_scores_original_ranking_and_refusal_is_null(tmp_path) -> None:
    """低分拒答依然命中检索指标且不调用生成/裁判。参数：tmp_path；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset

    runtime = RuntimeBoundary()
    report = await evaluate_dataset([case()], runtime, tmp_path / "gate", strategies=["bm25"],
                                     thresholds={"bm25": 2.0}, evidence_resolver=lambda c, s: [{1}])
    result = report["strategies"]["bm25"]
    assert result["recall@1"]["value"] == 1
    assert result["false_refusal_rate"]["value"] == 1
    assert result["faithfulness_macro"]["value"] is None
    assert runtime.inputs == []


async def test_invalid_query_is_recorded_per_strategy_instead_of_losing_samples(tmp_path) -> None:
    """超长原问是逐题错误，不能截断整个运行且漏掉分母。参数：tmp_path；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset

    invalid = case()
    invalid.query = "长" * 2001
    report = await evaluate_dataset([invalid, case()], RuntimeBoundary(refuse=True), tmp_path / "invalid",
                                     strategies=["bm25"], evidence_resolver=lambda c, s: [{1}])
    assert report["strategies"]["bm25"]["sample_count"] == 2
    assert report["strategies"]["bm25"]["error_count"] == 1


async def test_cancel_preserves_prior_record_and_interrupted_manifest(tmp_path) -> None:
    """取消不伪装成功或覆盖已落盘行。参数：tmp_path；返回：None；异常：断言失败。"""
    import asyncio
    from app.evaluation.runner import evaluate_dataset

    runtime = RuntimeBoundary(refuse=True)
    original = runtime.retrieve

    async def second_cancel(prepared, strategy, filters):
        """第一策略完成后模拟取消。参数：检索参数；返回：结果；异常：CancelledError。"""
        if strategy == "dense":
            raise asyncio.CancelledError()
        return await original(prepared, strategy, filters)

    runtime.retrieve = second_cancel
    path = tmp_path / "cancel"
    with pytest.raises(asyncio.CancelledError):
        await evaluate_dataset([case()], runtime, path, strategies=["bm25", "dense"], evidence_resolver=lambda c, s: [{1}])
    assert len((path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads((path / "manifest.json").read_text(encoding="utf-8"))["status"] == "interrupted"


async def test_invalid_citation_not_counted_as_delivered_answer(tmp_path) -> None:
    """引用失败不能增加有效回答覆盖率，但必须占引用校验分母并保留模型输出。参数：tmp_path；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset

    runtime = RuntimeBoundary()

    async def invalid_output(query, selected):
        """只替换模型输出边界。参数：原问/候选；返回：越界引用；异常：无。"""
        return KnowledgeAnswer(useful=True, reason="", answer="越界断言[99]", citation_numbers=[99]), [
            EvidenceCitation(n=1, chunk_id=1, question="问", answer="答")]

    runtime.generate = invalid_output
    directory = tmp_path / "citation"
    report = await evaluate_dataset([case()], runtime, directory, strategies=["bm25"], evidence_resolver=lambda c, s: [{1}])
    result = report["strategies"]["bm25"]
    assert result["coverage"]["value"] == 0
    assert result["citation_validity"] == {"value": 0, "numerator": 0, "denominator": 1}
    row = json.loads((directory / "results.jsonl").read_text(encoding="utf-8"))
    assert row["decision"]["answer"] == "越界断言[99]"


async def test_report_retains_frozen_corpus_for_later_audit(tmp_path) -> None:
    """后续数据库ID或正文变化不影响本次报告证据审阅。参数：tmp_path；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset

    directory = tmp_path / "audit"
    await evaluate_dataset([case()], RuntimeBoundary(refuse=True), directory, strategies=["bm25"], evidence_resolver=lambda c, s: [{1}])
    saved = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    assert saved[0]["text"]["answer"] == "权威答案1。"


async def test_difficulty_summaries_keep_independent_denominators(tmp_path) -> None:
    """难题和易题单独统计样本及指标分母，不能用总题数相除。参数：tmp_path；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset

    easy, hard = case(), case()
    hard.id, hard.difficulty = "A-2", "hard"
    directory = tmp_path / "difficulty"
    result = await evaluate_dataset([easy, hard], RuntimeBoundary(refuse=True), directory, strategies=["bm25"], evidence_resolver=lambda c, s: [{1}])
    assert result["difficulties"]["bm25"]["hard"]["sample_count"] == 1
    assert result["difficulties"]["bm25"]["easy"]["recall@1"]["denominator"] == 1
    assert [json.loads(line)["difficulty"] for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()] == ["easy", "hard"]
    assert "普通最多10" in (directory / "summary.md").read_text(encoding="utf-8")
