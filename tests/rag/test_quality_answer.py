"""生成 useful 自评的严格输出、全集证据和拒答入池接入行为测试。"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from app.schemas.chat import ChatRequest
from tests.chat_fakes import make_chat_service
from tests.rag.test_retrieval import _chunk, _search
from app.repositories.support import SupportRepository


class ResponseModel:
    """仅替换上游模型流，保留真实适配器的 JSON 聚合与关闭。"""

    def __init__(self, frames: list) -> None:
        """保存响应。参数：frames 模型流分片；返回：None；异常：无。"""
        self.frames = frames
        self.closed = False

    async def astream(self, messages):
        """产生外部模型输出。参数：实际提示；返回：异步分片；异常：预设网络异常。"""
        try:
            for frame in self.frames:
                if isinstance(frame, Exception):
                    raise frame
                yield frame
        finally:
            self.closed = True


async def test_adapter_strict_json_self_check_and_closes_stream(monkeypatch) -> None:
    """useful 正常布尔与完整 JSON 分片须聚合，资源立即关闭。参数：替换器；返回：None；异常：断言失败。"""
    from app.core import llm

    upstream = ResponseModel([AIMessageChunk(content='{"useful":false,"reason":"证据不足",'),
                              AIMessageChunk(content='"answer":"","citation_numbers":[]}')])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: upstream)
    result = await llm.ChatModelAdapter().answer_knowledge([HumanMessage(content="能退吗")])
    assert result.useful is False and result.reason == "证据不足" and upstream.closed


@pytest.mark.parametrize("payload", [
    '{"useful":"false","reason":"不足","answer":"","citation_numbers":[]}',
    '{"useful":false,"useful":true,"reason":"不足","answer":"","citation_numbers":[]}',
    '{"useful":false,"reason":"不足","answer":"","citation_numbers":[],"conversation_id":1}',
    '{"useful":false,"reason":"不足","answer":"","citation_numbers":[NaN]}',
    '{"useful":false,"reason":"不足","answer":"","citation_numbers":[1.5]}',
    '{"useful":false,"reason":"不足","answer":"","citation_numbers":[]',
])
async def test_adapter_rejects_malformed_self_check_without_coercion(monkeypatch, payload: str) -> None:
    """坏模型格式不能被修成 useful=false 然后错误入池。参数：替换器/非法输出；返回：None；异常：断言失败。"""
    from app.core import llm
    from app.knowledge.answering import KnowledgeAnswerError

    upstream = ResponseModel([AIMessageChunk(content=payload)])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: upstream)
    with pytest.raises(KnowledgeAnswerError):
        await llm.ChatModelAdapter().answer_knowledge([])
    assert upstream.closed


async def test_adapter_rejects_tool_call_in_self_check(monkeypatch) -> None:
    """生成阶段工具申请不得当成成功结构化答案。参数：替换器；返回：None；异常：断言失败。"""
    from app.core import llm
    from app.knowledge.answering import KnowledgeAnswerError

    upstream = ResponseModel([AIMessageChunk(content="", tool_call_chunks=[{"name": "query_faq", "args": "{}", "id": "bad", "index": 0}])])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: upstream)
    with pytest.raises(KnowledgeAnswerError):
        await llm.ChatModelAdapter().answer_knowledge([])
    assert upstream.closed


def test_citations_collects_all_hits_not_only_answer_reference_subset() -> None:
    """答案只引两条也须保留全部本轮十条证据，并跨工具按 chunk id 去重。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import collect_citations

    hits = [{"id": number, "question": f"问{number}", "answer": f"答{number}", "category": "售后", "section_path": "政策 / 条件"}
            for number in range(1, 11)]
    current = [AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "政策"}},
                                                {"id": "refund", "name": "query_refund_policy", "args": {"query": "退款"}}]),
               ToolMessage(content=json.dumps({"ok": True, "data": {"hits": hits}}), tool_call_id="faq"),
               ToolMessage(content=json.dumps({"ok": True, "data": {"hits": [hits[0]]}}), tool_call_id="refund")]
    citations = collect_citations(current)
    assert len(citations) == 10
    assert [item.n for item in citations] == list(range(1, 11))
    assert citations[-1].chunk_id == 10 and citations[-1].section_path == "政策 / 条件"


class SelfCheckModel:
    """为真实聊天服务提供受约束自评边界，记录实际模型证据。"""

    def __init__(self, decision) -> None:
        """保存生成决定。参数：决定或异常；返回：None；异常：无。"""
        self.decision = decision
        self.inputs = []

    async def decide(self, messages, tools):
        """先申请 FAQ 再结束工具阶段。参数：当前消息/工具；返回：调用决定；异常：无。"""
        if not any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "政策"}}])
        return AIMessage(content="")

    async def answer_knowledge(self, messages):
        """返回自评。参数：真实模型提示；返回：KnowledgeAnswer；异常：预设故障。"""
        self.inputs.append(messages)
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


def _service(decision, *, low_score: bool = False):
    """组装真实聊天及混合知识，数据库/上游为测试边界。参数：决定/低分开关；返回：服务/仓储/模型；异常：无。"""
    service, store, _, _ = make_chat_service()
    search, _, _, _ = _search([_chunk(1)], scores={1: 0.1} if low_score else None)
    service.executor.support = SupportRepository(None, search)
    service.model = SelfCheckModel(decision)
    return service, store, service.model


async def test_useful_false_refuses_and_pools_raw_question_without_guess() -> None:
    """模型否定有用性时不能展示猜测，入池只保存真实原话。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import KnowledgeAnswer, FIXED_KNOWLEDGE_REFUSAL

    service, repository, model = _service(KnowledgeAnswer(useful=False, reason="现有政策没写", answer="猜测三天到账", citation_numbers=[]))
    request = ChatRequest(session_id="self-check-false", message="  你们到底能不能给我退啊？！  ")
    events = [event async for event in service.stream_turn(request)]
    shown = "".join(event.data["delta"] for event in events if event.event == "delta")
    assert shown == FIXED_KNOWLEDGE_REFUSAL and "三天到账" not in shown
    assert events[-1].event == "done" and len(repository.low_confidence) == 1
    assert repository.low_confidence[0]["raw_question"] == request.message
    assert repository.low_confidence[0]["source"] == "self_check"
    assert len(model.inputs) == 1


async def test_useful_true_publishes_after_commit_and_never_pools() -> None:
    """有用答案要先保存再输出，不能误入问题池。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import KnowledgeAnswer

    service, repository, model = _service(KnowledgeAnswer(useful=True, reason="", answer="权威答案1。[1]", citation_numbers=[1]))
    stream = service.stream_turn(ChatRequest(session_id="self-check-true", message="政策条件"))
    events = []
    async for event in stream:
        if event.event == "delta":
            assert next(iter(repository.turns.values())).turn_status == "completed"
        events.append(event)
    assert events[-1].event == "done" and not repository.low_confidence
    evidence = [json.loads(message.content)["data"]["hits"] for message in model.inputs[0] if isinstance(message, ToolMessage)]
    assert evidence[0][0]["n"] == 1 and evidence[0][0]["answer"] == "权威答案1。"


async def test_low_retrieval_score_still_requires_false_self_check_to_pool() -> None:
    """检索低分本身不入池，必须经过生成 useful=false。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import KnowledgeAnswer

    service, repository, model = _service(KnowledgeAnswer(useful=False, reason="无证据", answer="", citation_numbers=[]), low_score=True)
    events = [event async for event in service.stream_turn(ChatRequest(session_id="low-score", message="政策条件"))]
    assert len(model.inputs) == 1 and len(repository.low_confidence) == 1
    assert repository.low_confidence[0]["source"] == "self_check" and events[-1].event == "done"


@pytest.mark.parametrize("decision", [
    {"useful": True, "reason": "", "answer": "无证据猜测", "citation_numbers": []},
    {"useful": True, "reason": "", "answer": "答案[2]", "citation_numbers": [2]},
    {"useful": True, "reason": "", "answer": "答案[99]", "citation_numbers": [1]},
])
async def test_invalid_knowledge_citations_are_system_error_not_pool(decision: dict) -> None:
    """知识答案缺引用或越界不能凭模型 useful=true 放行，也不记自评问题。参数：决定；返回：None；异常：断言失败。"""
    from app.knowledge.answering import KnowledgeAnswer

    service, repository, _ = _service(KnowledgeAnswer(**decision))
    events = [event async for event in service.stream_turn(ChatRequest(session_id="bad-cite", message="政策条件"))]
    assert any(event.event == "error" for event in events)
    assert not any(event.event in {"delta", "done"} for event in events) and not repository.low_confidence


async def test_backend_failed_knowledge_call_does_not_run_self_check_or_pool() -> None:
    """全部知识依赖故障不等于证据不足，不能让 useful=false 把运行错误入池。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.vector import VectorIndexUnavailable

    decision = AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "政策"}}])
    service, repository, _, model = make_chat_service(decision, chunks=["查询暂时不可用，请稍后再试。"])
    search, _, index, _ = _search([_chunk(1)])

    async def unavailable(*args, **kwargs):
        """模拟索引故障。参数：请求；返回：不返回；异常：VectorIndexUnavailable。"""
        raise VectorIndexUnavailable("依赖故障")

    index.hybrid_search = unavailable
    service.executor.support = SupportRepository(None, search)
    calls = []
    original = model.answer_knowledge

    async def self_check(messages):
        """观察自评是否错误执行。参数：提示；返回：决定；异常：原错误。"""
        calls.append(messages)
        return await original(messages)

    model.answer_knowledge = self_check
    events = [event async for event in service.stream_turn(ChatRequest(session_id="backend-fail", message="政策条件"))]
    assert not calls and not repository.low_confidence
    assert next(iter(repository.results.values())).ok is False
    assert events[-1].event == "done"


async def test_ticket_receipt_is_preserved_when_later_knowledge_self_check_refuses() -> None:
    """已提交工单不能因后续知识拒答被隐藏或重复创建。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL, KnowledgeAnswer
    from app.schemas.tools import ToolResult

    service, repository, _ = _service(KnowledgeAnswer(useful=False, reason="政策证据不足", answer="", citation_numbers=[]))
    decisions = [AIMessage(content="", tool_calls=[{"id": "ticket", "name": "create_ticket", "args": {"description": "请人工处理", "ticket_type": "售后"}}]),
                 AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "政策"}}]), AIMessage(content="")]
    counter = []

    async def decide(messages, tools):
        """返回三步外部决策。参数：消息/工具；返回：决定；异常：无。"""
        counter.append(True)
        return decisions[len(counter) - 1]

    async def create_ticket(context, description, ticket_type):
        """替换外部真实工单事务，成功事实由真实执行器保存。参数：身份/诉求/类型；返回：结果；异常：无。"""
        return ToolResult.success({"ticket_no": "T-confirmed", "ticket_type": ticket_type, "status": "待处理"})

    service.model.decide = decide
    service.executor.support.create_ticket = create_ticket
    events = [event async for event in service.stream_turn(ChatRequest(session_id="ticket-refusal", message="请创建售后工单，并告诉我政策"))]
    visible = "".join(event.data["delta"] for event in events if event.event == "delta")
    assert "T-confirmed" in visible and FIXED_KNOWLEDGE_REFUSAL in visible
    assert events[-1].event == "done" and len(repository.low_confidence) == 1
    final = next(iter(repository.turns.values())).messages[-1].content
    assert "T-confirmed" in final and FIXED_KNOWLEDGE_REFUSAL in final


async def test_self_check_network_failure_and_commit_failure_never_pool_or_publish() -> None:
    """模型故障和数据库失败不能伪装成 useful=false 正常拒答。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import KnowledgeAnswer

    service, repository, _ = _service(ConnectionError("internal secret"))
    events = [event async for event in service.stream_turn(ChatRequest(session_id="model-fail", message="政策条件"))]
    assert events[-1].event == "error" and not repository.low_confidence
    service, repository, _ = _service(KnowledgeAnswer(useful=False, reason="不足", answer="", citation_numbers=[]))
    repository.fail_finish = True
    events = [event async for event in service.stream_turn(ChatRequest(session_id="pool-fail", message="政策条件"))]
    assert events[-1].event == "error" and not repository.low_confidence
    assert not any(event.event == "delta" for event in events)
