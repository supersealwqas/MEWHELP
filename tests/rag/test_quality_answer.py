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


async def test_knowledge_generation_handoff_preserves_raw_query_and_completes_refusal(monkeypatch) -> None:
    """原问题与工具结束指令须一起进入最终请求，不能把工具尾帧空输出算拒答。

    参数：monkeypatch 仅替换上游网络边界。
    返回：None，真实聊天服务通过自评后原子完成并按原话入池。
    异常：AssertionError，阶段未交接、原话改变、失败误入池或流未关闭。
    """
    from app.core import llm
    from app.knowledge.answering import KnowledgeAnswer

    raw = '  MH-W99水箱容量是多少？\n不能用MH-W20代替。  '

    class EndingAwareGateway(ResponseModel):
        """模拟无工具定义请求以工具结果结束时空输出、明确新任务后正常自评的网关。"""

        async def astream(self, messages):
            """响应真实请求末帧。参数：messages 实际输入；返回：模型流；异常：原测试异常。"""
            ending = json.loads(messages[-1].content)
            if isinstance(messages[-1], HumanMessage) and ending.get("query") == raw and ending.get("instruction"):
                self.frames = [AIMessageChunk(content='{"useful":false,"reason":"无该型号原文","answer":"","citation_numbers":[]}')]
            async for chunk in super().astream(messages):
                yield chunk

    upstream = EndingAwareGateway([AIMessageChunk(content="")])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: upstream)
    service, repository, _ = _service(KnowledgeAnswer(useful=False, reason="不足", answer="", citation_numbers=[]), low_score=True)
    service.model.answer_knowledge = llm.ChatModelAdapter().answer_knowledge
    events = [event async for event in service.stream_turn(ChatRequest(session_id="handoff-refusal", message=raw))]
    assert events[-1].event == "done" and upstream.closed
    assert len(repository.low_confidence) == 1
    assert repository.low_confidence[0]["raw_question"] == raw
    assert repository.low_confidence[0]["source"] == "self_check"


async def test_adapter_network_error_is_not_converted_to_refusal(monkeypatch) -> None:
    """模型网络故障必须传播，不能本地制造useful=false或遗漏流关闭。

    参数：monkeypatch 上游替换器。
    返回：None，故障保持原异常时正常结束。
    异常：AssertionError，故障被吞或流未关闭。
    """
    from app.core import llm

    upstream = ResponseModel([ConnectionError("private upstream failure")])
    monkeypatch.setattr(llm, "get_chat_model", lambda **kwargs: upstream)
    with pytest.raises(ConnectionError):
        await llm.ChatModelAdapter().answer_knowledge([])
    assert upstream.closed


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


@pytest.mark.parametrize("ranks,expected", [
    ([], []), ([1], [1]), ([1, 2], [1, 2]), ([1, 2, 3], [1, 3, 2]),
    ([1, 2, 3, 4, 5], [1, 3, 5, 4, 2]),
    (list(range(1, 11)), [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]),
])
def test_edge_order_places_ranked_evidence_at_both_ends_without_mutation(ranks, expected) -> None:
    """偶数、奇数和小集合按原排名交替放首尾，不改原列表。参数：原排名/预期位置；返回：None；异常：断言失败。"""
    from app.knowledge.answering import edge_order

    items = [{"rank": rank, "answer": f"完整原文{rank}"} for rank in ranks]
    original = [dict(item) for item in items]
    ordered = edge_order(items)
    assert [item["rank"] for item in ordered] == expected
    assert items == original and ordered is not items
    assert {id(item) for item in ordered} == {id(item) for item in items}


def test_numbered_knowledge_groups_keep_citation_ids_and_saved_facts_unchanged() -> None:
    """每个工具结果独立首尾排列，稳定编号和原始持久化正文不变。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import collect_citations, number_knowledge_results

    hits = [{"id": rank, "question": f"问{rank}", "answer": f"完整原文{rank}", "section_path": "政策 / 条件"}
            for rank in range(1, 14)]
    current = [AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "政策"}},
                                                {"id": "refund", "name": "query_refund_policy", "args": {"query": "退款"}}]),
               ToolMessage(content=json.dumps({"ok": True, "data": {"hits": hits[:10]}}), tool_call_id="faq"),
               ToolMessage(content=json.dumps({"ok": True, "data": {"hits": hits[10:]}}), tool_call_id="refund")]
    original = [message.model_dump() for message in current]
    citations = collect_citations(current)
    numbered = number_knowledge_results(current, citations)
    groups = [json.loads(message.content)["data"]["hits"] for message in numbered if isinstance(message, ToolMessage)]
    # 模型输入只带 n；首尾排列按 n 校验，主键 id 不再暴露给模型。
    assert [hit["n"] for hit in groups[0]] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert [hit["n"] for hit in groups[1]] == [11, 13, 12]
    assert all("id" not in hit for group in groups for hit in group)
    assert all(hit["answer"] == f"完整原文{hit['n']}" for group in groups for hit in group)
    assert [item.n for item in citations] == list(range(1, 14))
    # 持久化的工具事实（原消息）仍保留主键，不被模型输入裁剪影响。
    assert [message.model_dump() for message in current] == original
    assert [hit["id"] for hit in hits] == list(range(1, 14))


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


@pytest.mark.parametrize("tool_name,data,expected", [
    ("query_order", {"order_id": "1001", "status": "已发货"}, "订单 1001"),
    ("query_logistics", {"tracking_no": "SF001", "status": "运输中", "latest_node": "杭州转运中心"}, "杭州转运中心"),
    ("query_product", {"name": "猫粮", "price": 39.9, "currency": "CNY", "stock": 2}, "39.9"),
])
def test_refusal_business_summary_uses_only_successful_current_facts(tool_name: str, data: dict, expected: str) -> None:
    """政策不足时仍能展示本轮可信业务事实，失败结果不得进入摘要。参数：工具/事实/期望；返回：None；异常：断言失败。"""
    from app.knowledge.answering import business_facts_summary

    messages = [AIMessage(content="", tool_calls=[{"id": "business", "name": tool_name, "args": {}}]),
                ToolMessage(content=json.dumps({"ok": True, "data": data}), tool_call_id="business")]
    assert expected in business_facts_summary(messages)
    failed = [messages[0], ToolMessage(content=json.dumps({"ok": False, "data": data}), tool_call_id="business")]
    assert business_facts_summary(failed) == ""


async def test_useful_false_cannot_override_valid_business_order_fact() -> None:
    """FAQ没回答政策时仍须保留已查到的订单状态，不能整轮仅拒答。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.answering import KnowledgeAnswer, FIXED_KNOWLEDGE_REFUSAL

    service, repository, model = _service(KnowledgeAnswer(useful=False, reason="没有退款政策", answer="", citation_numbers=[]))
    decisions = [AIMessage(content="", tool_calls=[{"id": "order", "name": "query_order", "args": {"order_id": "1001"}}]),
                 AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "退货"}}]), AIMessage(content="")]
    calls = []

    async def decide(messages, tools):
        """模拟工具决策边界。参数：消息/工具；返回：申请；异常：无。"""
        calls.append(messages)
        return decisions[len(calls) - 1]

    service.model.decide = decide
    events = [event async for event in service.stream_turn(ChatRequest(session_id="partial-order", message="订单1001的状态是什么，还能退货吗？"))]
    shown = "".join(event.data["delta"] for event in events if event.event == "delta")
    assert "已发货" in shown and FIXED_KNOWLEDGE_REFUSAL in shown
    assert events[-1].event == "done" and len(repository.low_confidence) == 1


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
    assert events[-1].event == "error"


@pytest.mark.parametrize("ticket", [False, True])
async def test_knowledge_backend_failure_is_not_completed_or_pooled(ticket):
    """知识故障禁止普通生成成功，已提交工单仍保留。参数：ticket 先建工单；返回：None；异常：断言失败。"""
    from app.knowledge.vector import VectorIndexUnavailable
    from tests.test_multi_tool_chat import call, decision, install_ticket_store

    decisions = [decision(call("create_ticket", "ticket", description="请人工处理", ticket_type="售后"))] if ticket else []
    decisions.append(decision(call("query_faq", "faq", keyword="政策")))
    service, repository, _, model = make_chat_service(decisions=decisions, chunks=["我编造的政策答案"])
    search, _, index, _ = _search([_chunk(1)])

    async def unavailable(*args, **kwargs):
        """模拟真实索引依赖异常。参数：请求；返回：不返回；异常：VectorIndexUnavailable。"""
        raise VectorIndexUnavailable("broken")

    index.hybrid_search = unavailable
    service.executor.support = SupportRepository(None, search)
    writes = install_ticket_store(repository, service.executor.support)
    events = [event async for event in service.stream_turn(ChatRequest(session_id="dependency-failure", message="请创建售后工单并查询政策"))]
    assert events[-1].event == "error" and not any(event.event == "done" for event in events)
    assert not repository.low_confidence
    assert next(iter(repository.turns.values())).turn_status == "failed"
    assert not any(phase == "answer" for phase, _ in model.inputs)
    if ticket:
        assert len(writes) == 1 and "T-1" in events[-1].data["error"]
        assert "T-1" in "".join(event.data["delta"] for event in events if event.event == "delta")
        assert events[-1].data["retry_safe"] is False
    else:
        assert not writes and not any(event.event == "delta" for event in events)


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


async def test_merged_citation_brackets_are_accepted_not_system_error() -> None:
    """模型把多个编号合并在一个方括号内（[1, 4, 6]）是合法引用写法，不能被当成系统错误。

    参数：无。
    返回：None，完成回答并保留引用。
    异常：AssertionError，合并写法被判失败。
    """
    from app.knowledge.answering import KnowledgeAnswer

    # 夹具只提供 n=1 的权威证据，因此合并写法引用同一组已存在的编号。
    service, repository, _ = _service(KnowledgeAnswer(
        useful=True, reason="",
        answer="支持无理由退货[1]。运费由平台承担[1, 1]。",
        citation_numbers=[1],
    ))
    events = [event async for event in service.stream_turn(ChatRequest(session_id="merged", message="退货政策"))]
    assert not any(event.event == "error" for event in events), "合并角标不应被判系统错误"
    metadata = next(event.data for event in events if event.event == "answer_metadata")
    assert [item["n"] for item in metadata["citations"]] == [1]


@pytest.mark.parametrize("answer,declared", [
    ("答案[1][4]", [1, 4]),
    ("答案[1, 4]", [1, 4]),
    ("答案[1，4]", [1, 4]),
    ("答案[1,4,6]", [1, 4, 6]),
    ("答案[1, 4]和[10]", [1, 4, 10]),
])
def test_inline_citation_formats_are_equivalent(answer: str, declared: list[int]) -> None:
    """单编号、逗号合并、中文逗号、混合写法都代表同一组引用，校验必须一致通过。

    参数：answer 模型正文；declared 模型声明的引用编号。
    返回：None。
    异常：AssertionError，任一合法写法被拒。
    """
    from app.knowledge.answering import EvidenceCitation, decide_answer, KnowledgeAnswer

    citations = [EvidenceCitation(n=n, chunk_id=n, question="问", answer="答") for n in declared]
    text, failure = decide_answer(
        KnowledgeAnswer(useful=True, reason="", answer=answer, citation_numbers=declared), citations)
    assert failure is None and text == answer


@pytest.mark.parametrize("answer,declared", [
    ("答案[1, 99]", [1, 99]),
    ("答案[1, 2]", [1]),
])
def test_merged_citation_still_rejects_unknown_or_mismatched_numbers(answer: str, declared: list[int]) -> None:
    """放宽合并写法不能放过越界或声明缺失的编号。

    参数：answer 模型正文；declared 模型声明编号。
    返回：None。
    异常：AssertionError，非法引用被放行。
    """
    from app.knowledge.answering import EvidenceCitation, KnowledgeAnswer, KnowledgeAnswerError, decide_answer

    citations = [EvidenceCitation(n=1, chunk_id=1, question="问", answer="答")]
    with pytest.raises(KnowledgeAnswerError):
        decide_answer(KnowledgeAnswer(useful=True, reason="", answer=answer, citation_numbers=declared), citations)


@pytest.mark.parametrize("answer,declared", [
    ("答案[1, 2, 3]", [1, 2, 3, 4, 5, 6]),
    ("答案[1]", [1, 2, 3]),
    ("答案[1][2]", [1, 2, 3, 4]),
])
def test_declared_superset_of_inline_is_accepted(answer: str, declared: list[int]) -> None:
    """模型在 citation_numbers 多列参考过的证据是常见行为，正文角标是声明的子集应放行。

    参数：answer 模型正文；declared 模型声明编号（宽于正文）。
    返回：None。
    异常：AssertionError，合理行为被判系统错误。
    """
    from app.knowledge.answering import EvidenceCitation, KnowledgeAnswer, decide_answer

    citations = [EvidenceCitation(n=n, chunk_id=n, question="问", answer="答") for n in declared]
    text, failure = decide_answer(
        KnowledgeAnswer(useful=True, reason="", answer=answer, citation_numbers=declared), citations)
    assert failure is None and text == answer


@pytest.mark.parametrize("answer,declared", [
    ("答案[1, 9]", [1, 2]),
    ("答案[9]", [1, 2]),
    ("答案[1][7][8]", [1, 2, 3]),
])
def test_inline_outside_declared_is_still_rejected(answer: str, declared: list[int]) -> None:
    """正文出现声明之外的编号仍属伪造引用，必须拒绝。

    参数：answer 模型正文；declared 模型声明编号。
    返回：None。
    异常：AssertionError，越界角标被放行。
    """
    from app.knowledge.answering import EvidenceCitation, KnowledgeAnswer, KnowledgeAnswerError, decide_answer

    citations = [EvidenceCitation(n=n, chunk_id=n, question="问", answer="答") for n in declared]
    with pytest.raises(KnowledgeAnswerError):
        decide_answer(KnowledgeAnswer(useful=True, reason="", answer=answer, citation_numbers=declared), citations)


def test_model_evidence_hides_database_id_from_model_input() -> None:
    """模型输入只能看到引用编号 n，不能看到数据库主键 id，否则模型会误把主键当角标。

    参数：无。
    返回：None。
    异常：AssertionError，模型输入里仍暴露 id 字段。
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from app.knowledge.answering import EvidenceCitation, number_knowledge_results

    hit = {"id": 16, "question": "退货政策是什么", "answer": "支持 7 天无理由退货", "category": "FAQ"}
    current = [
        AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "退货"}}]),
        ToolMessage(content=json.dumps({"ok": True, "data": {"hits": [hit]}}), tool_call_id="faq"),
    ]
    citations = [EvidenceCitation(n=1, chunk_id=16, question=hit["question"], answer=hit["answer"])]
    numbered = number_knowledge_results(current, citations)
    payload = json.loads(numbered[1].content)["data"]["hits"][0]
    assert payload["n"] == 1, "模型必须看到引用编号 n"
    assert "id" not in payload, f"模型输入不应出现数据库主键：{payload}"


def test_collect_citations_still_reads_database_id_from_tool_result() -> None:
    """隐藏模型输入里的 id 不能破坏服务端从工具事实读取主键。

    参数：无。
    返回：None。
    异常：AssertionError，服务端拿不到 chunk_id。
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from app.knowledge.answering import collect_citations

    hit = {"id": 16, "question": "退货政策是什么", "answer": "支持 7 天无理由退货", "category": "FAQ"}
    current = [
        AIMessage(content="", tool_calls=[{"id": "faq", "name": "query_faq", "args": {"keyword": "退货"}}]),
        ToolMessage(content=json.dumps({"ok": True, "data": {"hits": [hit]}}), tool_call_id="faq"),
    ]
    citations = collect_citations(current)
    assert citations[0].chunk_id == 16 and citations[0].n == 1
