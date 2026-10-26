"""编造台账与忠实度裁判的真实仓储 SQL/服务边界测试。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
import asyncio
import json

import pytest
from sqlalchemy.dialects import mysql
from langchain_core.messages import AIMessageChunk

from app.knowledge.answering import EvidenceCitation
from tests.workflow.test_model import sdk_runtime, call_response, text_response


class RecordingSession:
    """只替换外部数据库执行，保留真实 SQLAlchemy upsert 与处置逻辑。"""

    def __init__(self, row=None) -> None:
        """保存现有记录。参数：row 行；返回：None；异常：无。"""
        self.row = row
        self.statements = []
        self.added = []

    async def execute(self, statement):
        """记录真实 SQL。参数：表达式；返回：None；异常：无。"""
        self.statements.append(statement)

    async def scalar(self, statement):
        """读取现有台账行。参数：真实查询；返回：行；异常：无。"""
        self.statements.append(statement)
        return self.row


class SessionFactory:
    """短事务边界替身。"""

    def __init__(self, session) -> None:
        """保存会话。参数：session 记录器；返回：None；异常：无。"""
        self.session = session

    @asynccontextmanager
    async def begin(self):
        """提供事务。参数：无；返回：会话；异常：透传。"""
        yield self.session


def _citation(number=1):
    """构造实际模型证据。参数：编号；返回：EvidenceCitation；异常：校验错误。"""
    return EvidenceCitation(n=number, chunk_id=number, section_path="售后 / 政策", question=f"问法{number}", answer=f"原证据{number}")


def _draft():
    """构造全量证据台账草案。参数：无；返回：FaithCaseDraft；异常：校验错误。"""
    from app.repositories.faith_cases import FaithCaseDraft

    return FaithCaseDraft(eval_id="A43", bucket="A_policy", query="退款到账几天", strategy="hybrid_rerank",
                           answer="保证两天到账[1]", reason="证据没有两天到账承诺", citations=[_citation(1), _citation(2)], judge_model="judge-test")


async def test_false_self_check_cannot_create_pool_record_with_non_refusal_content() -> None:
    """有入池草案也必须与固定拒答相配，不能让普通成功答案因错误调用污染池。参数：无；返回：None；异常：断言失败。"""
    from app.repositories.chat import ChatRepository
    from app.knowledge.answering import SelfCheckFailure
    from app.schemas.tools import ToolContext

    repository = ChatRepository(None)
    with pytest.raises(ValueError, match="拒答"):
        await repository.finish_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="问"),
                                     "保证两天退款到账", low_confidence=SelfCheckFailure(reason="不足"))


async def test_record_emits_atomic_unique_eval_upsert_and_preserves_recurrence_fields() -> None:
    """SQL 必须按 eval_id 原子累计，不覆盖 first_seen_at/resolved_at，并重置处置。参数：无；返回：None；异常：断言失败。"""
    from app.repositories.faith_cases import FaithCaseRepository

    session = RecordingSession()
    await FaithCaseRepository(SessionFactory(session)).record(_draft())
    compiled = session.statements[0].compile(dialect=mysql.dialect())
    sql = str(compiled)
    updates = sql.split("ON DUPLICATE KEY UPDATE", 1)[1]
    assert "seen_count = (faith_cases.seen_count +" in updates
    assert "first_seen_at =" not in updates and "resolved_at =" not in updates
    assert "resolution =" in updates and "status =" in updates and "last_seen_at =" in updates
    params = compiled.params
    assert params["eval_id"] == "A43" and len(params["citations"]) == 2
    assert params["citations"][1]["answer"] == "原证据2"


@pytest.mark.parametrize("status", ["已解决", "无需解决"])
async def test_resolution_requires_nonempty_explanation(status: str) -> None:
    """空处置说明必须在数据库调用前拒绝。参数：处置状态；返回：None；异常：断言失败。"""
    from app.repositories.faith_cases import FaithCaseRepository

    session = RecordingSession()
    repository = FaithCaseRepository(SessionFactory(session))
    with pytest.raises(ValueError):
        await repository.resolve("A43", status, "   ")
    assert not session.statements


async def test_resolution_changes_status_and_reopening_preserves_resolved_time() -> None:
    """解决必须记录说明/时间；退回未解决清说明但保留时间用于复发。参数：无；返回：None；异常：断言失败。"""
    from app.repositories.faith_cases import FaithCaseRepository

    row = SimpleNamespace(status="未解决", resolution=None, resolved_at=None)
    session = RecordingSession(row)
    repository = FaithCaseRepository(SessionFactory(session))
    await repository.resolve("A43", "已解决", "已补充权威退款政策")
    assert row.status == "已解决" and row.resolution == "已补充权威退款政策" and row.resolved_at is not None
    resolved_time = row.resolved_at
    await repository.resolve("A43", "未解决", None)
    assert row.status == "未解决" and row.resolution is None and row.resolved_at == resolved_time


class JudgeBoundary:
    """仅替换外部裁判模型。"""

    def __init__(self, result) -> None:
        """保存评判。参数：result 结果或异常；返回：None；异常：无。"""
        self.result = result
        self.inputs = []
        self.model_name = "judge-test"

    async def judge(self, query, answer, citations):
        """记录实际证据。参数：题问/答案/证据全集；返回：决定；异常：预设错误。"""
        self.inputs.append((query, answer, citations))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FaithStore:
    """裁判服务的外部台账仓储记录器。"""

    def __init__(self) -> None:
        """创建记录。参数：无；返回：None；异常：无。"""
        self.recorded = []

    async def record(self, draft):
        """记录服务交给持久化的真实草案。参数：draft；返回：None；异常：无。"""
        self.recorded.append(draft)


async def test_only_unsupported_facts_record_case_with_complete_citations() -> None:
    """裁判指出无证据事实才入台账，保存全集而非其引用子集。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FactJudgment, FaithfulnessDecision, FaithfulnessService

    decision = FaithfulnessDecision(facts=[FactJudgment(claim="保证两天到账", supported=False, reason="证据未给到账承诺", evidence_numbers=[])])
    judge, store = JudgeBoundary(decision), FaithStore()
    citations = [_citation(1), _citation(2)]
    result = await FaithfulnessService(judge, store).evaluate(eval_id="A43", bucket="A_policy", query="退款多久到账",
        strategy="hybrid_rerank", answer="保证两天到账[1]", citations=citations)
    assert result.faithfulness == 0 and len(store.recorded) == 1
    assert len(store.recorded[0].citations) == 2 and judge.inputs[0][2] == citations


async def test_supported_answer_and_judge_failure_never_record_false_case() -> None:
    """全部支持、裁判故障不能形成编造台账。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FactJudgment, FaithfulnessDecision, FaithfulnessService

    decision = FaithfulnessDecision(facts=[FactJudgment(claim="原证据1", supported=True, reason="证据支持", evidence_numbers=[1])])
    store = FaithStore()
    service = FaithfulnessService(JudgeBoundary(decision), store)
    result = await service.evaluate(eval_id="A43", bucket="A_policy", query="政策", strategy="hybrid_rerank", answer="原证据1[1]", citations=[_citation(1)])
    assert result.faithfulness == 1 and not store.recorded
    service = FaithfulnessService(JudgeBoundary(ConnectionError("judge unavailable")), store)
    with pytest.raises(ConnectionError):
        await service.evaluate(eval_id="A43", bucket="A_policy", query="政策", strategy="hybrid_rerank", answer="答案", citations=[_citation(1)])
    assert not store.recorded


async def test_refusal_has_no_facts_and_is_not_perfect_faithfulness() -> None:
    """无事实拒答不得记忠实度1或生成台账，需显式空指标。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessDecision, FaithfulnessService

    store = FaithStore()
    result = await FaithfulnessService(JudgeBoundary(FaithfulnessDecision(facts=[])), store).evaluate(
        eval_id="A43", bucket="A_policy", query="政策", strategy="hybrid_rerank", answer="证据不足", citations=[], refused=True)
    assert result.faithfulness is None and not store.recorded


async def test_judge_adapter_aggregates_strict_output_with_full_evidence() -> None:
    """真实裁判适配器要传全集且关闭流，不只处理预置决定。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessJudge

    class Model:
        """替代外部模型流，保留工具参数校验逻辑。"""

        closed = False
        inputs = []

        def bind_tools(self, tools, **kwargs):
            """提供受控绑定流。参数：tools工具定义、kwargs绑定参数；返回：自身；异常：无。"""
            return self

        async def astream(self, messages):
            """提供裁判工具参数。参数：实际输入；返回：分片；异常：无。"""
            self.inputs.append(messages)
            try:
                yield AIMessageChunk(content="", tool_call_chunks=[{"name": "submit_faithfulness", "id": "judge-1", "index": 0,
                    "args": '{"facts":[{"claim":"两天到账","supported":false,"reason":"没有依据","evidence_numbers":[]}]}'}],
                    response_metadata={"finish_reason": "tool_calls"})
            finally:
                self.closed = True

    model = Model()
    judge = FaithfulnessJudge(lambda **kwargs: model, model_name="judge-test")
    result = await judge.judge("退款时间", "两天到账[1]", [_citation(1), _citation(2)])
    assert not result.facts[0].supported and model.closed
    payload = json.loads(model.inputs[0][1].content)
    assert len(payload["citations"]) == 2 and payload["citations"][-1]["chunk_id"] == 2


@pytest.mark.parametrize("payload", [
    '{"facts":[{"claim":"事实","supported":"false","reason":"原因","evidence_numbers":[]}]}',
    '{"facts":[],"facts":[]}',
    '{"facts":[],"extra":1}',
    '{"facts":[{"claim":"事实","supported":false,"reason":"原因","evidence_numbers":[NaN]}]}',
])
async def test_judge_adapter_rejects_bad_json_and_types(payload: str) -> None:
    """格式非法不能伪造编造或满分。参数：输出；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessJudge, FaithfulnessError

    class Model:
        """非法上游输出边界。"""

        def bind_tools(self, tools, **kwargs):
            """提供受控绑定流。参数：tools工具定义、kwargs绑定参数；返回：自身；异常：无。"""
            return self

        async def astream(self, messages):
            """返回非法工具参数。参数：输入；返回：分片；异常：无。"""
            yield AIMessageChunk(content="", tool_call_chunks=[
                {"name": "submit_faithfulness", "args": payload, "id": "judge-1", "index": 0}],
                response_metadata={"finish_reason": "tool_calls"})

    with pytest.raises(FaithfulnessError):
        await FaithfulnessJudge(lambda **kwargs: Model(), model_name="judge-test").judge("问题", "答案", [_citation()])


async def test_faithfulness_judge_uses_strict_named_tool_and_complete_evidence(sdk_runtime):
    """裁判没有强制工具输出或漏传实际证据时失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessJudge

    payload = {"facts": [{"claim": "两天到账", "supported": False, "reason": "证据未给承诺", "evidence_numbers": []}]}
    sdk_runtime.responses.append(call_response("submit_faithfulness", json.dumps(payload)))

    def factory(**kwargs):
        """将显式裁判模型名交给受控SDK。参数：kwargs模型选项；返回：真实模型客户端；异常：配置错误。"""
        assert kwargs.pop("model") == "judge-test"
        return sdk_runtime.factory(**kwargs)

    result = await FaithfulnessJudge(factory, model_name="judge-test").judge("退款多久到账", "两天到账", [_citation(1), _citation(2)])
    assert result.model_dump() == payload and len(sdk_runtime.requests) == 1
    request = sdk_runtime.requests[0]
    assert request["tool_choice"] == {"type": "function", "function": {"name": "submit_faithfulness"}}
    assert request["parallel_tool_calls"] is False
    function = request["tools"][0]["function"]
    assert len(request["tools"]) == 1 and function["strict"] is True
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["required"] == ["facts"]
    sent = json.loads(request["messages"][1]["content"])
    assert [item["n"] for item in sent["citations"]] == [1, 2]
    assert "ground_truth" not in sent and "strategy" not in sent


async def test_faithfulness_judge_rejects_valid_free_text_json(sdk_runtime):
    """裁判把合法自由文本JSON当成结构化提交时失败。参数：HTTP夹具；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessJudge, FaithfulnessError

    sdk_runtime.responses.append(text_response('{"facts":[]}'))

    def factory(**kwargs):
        """保留真实SDK并移除夹具内已固定的模型名。参数：kwargs模型选项；返回：模型客户端；异常：配置错误。"""
        kwargs.pop("model")
        return sdk_runtime.factory(**kwargs)

    with pytest.raises(FaithfulnessError):
        await FaithfulnessJudge(factory, model_name="judge-test").judge("问题", "答案", [_citation()])
    assert len(sdk_runtime.requests) == 1


async def test_faithfulness_input_limit_includes_control_schema_before_model_request():
    """只够正文的临界输入遗漏工具定义仍发模型请求时失败。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessJudge, FaithfulnessError
    from tests.rag.test_quality_answer import ResponseModel

    citations = [_citation()]
    base = json.dumps({"query": "问题", "answer": "", "citations": [item.model_dump() for item in citations]}, ensure_ascii=False)
    answer = "答" * (64999 - len(base))
    calls = []

    def factory(**kwargs):
        """记录本不该发生的模型构造。参数：kwargs模型选项；返回：外部模型流；异常：无。"""
        calls.append(kwargs)
        return ResponseModel([AIMessageChunk(content='{"facts":[]}')])

    with pytest.raises(FaithfulnessError):
        await FaithfulnessJudge(factory).judge("问题", answer, citations)
    assert calls == []


@pytest.mark.parametrize("failure", ["length", "network", "cancel"])
async def test_faithfulness_tool_stream_limits_and_errors_close_without_false_result(failure):
    """裁判超长参数继续读取、网络故障或取消被吞掉时失败。参数：故障类型；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessJudge, FaithfulnessError
    from tests.rag.test_quality_answer import ResponseModel

    if failure == "length":
        raw = json.dumps({"facts": [{"claim": "事实", "supported": False,
            "reason": "证" * 20001, "evidence_numbers": []}]}, ensure_ascii=False)
        frames = [AIMessageChunk(content="", tool_call_chunks=[
            {"name": "submit_faithfulness", "args": raw, "id": "judge-1", "index": 0}]),
            RuntimeError("超长之后不能继续读取")]
        expected = FaithfulnessError
    elif failure == "network":
        frames, expected = [ConnectionError("裁判网络中断")], ConnectionError
    else:
        frames, expected = [asyncio.CancelledError()], asyncio.CancelledError
    upstream = ResponseModel(frames)
    with pytest.raises(expected):
        await FaithfulnessJudge(lambda **kwargs: upstream).judge("问题", "答案", [_citation()])
    assert upstream.closed


async def test_true_answer_empty_judge_facts_is_error_not_full_score() -> None:
    """非拒答却裁判未给事实不能美化为满分。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.faithfulness import FaithfulnessDecision, FaithfulnessService, FaithfulnessError

    with pytest.raises(FaithfulnessError):
        await FaithfulnessService(JudgeBoundary(FaithfulnessDecision(facts=[])), FaithStore()).evaluate(
            eval_id="A43", bucket="A_policy", query="政策", strategy="hybrid_rerank", answer="三天到账", citations=[_citation(1)])
