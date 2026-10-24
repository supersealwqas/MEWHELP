"""正文引用必须是可展示文本，不能借代码、链接或HTML编号满足门槛。"""

import pytest

from app.knowledge.answering import EvidenceCitation, KnowledgeAnswer, KnowledgeAnswerError, decide_answer
from app.schemas.chat import ChatRequest
from tests.rag.test_quality_answer import _service


def citations():
    """构造完整引用集合。参数：无；返回：四条快照；异常：校验错误。"""
    return [EvidenceCitation(n=n, chunk_id=n, question="政策", answer="原文") for n in range(1, 5)]


def test_declared_citation_without_inline_marker_is_rejected():
    """只声明编号不能发布答案。参数：无；返回：None；异常：断言失败。"""
    with pytest.raises(KnowledgeAnswerError):
        decide_answer(KnowledgeAnswer(useful=True, reason="", answer="没有角标的答案", citation_numbers=[1]), citations())


@pytest.mark.parametrize("answer", [
    "说明 `答案[1]`", "```text\n答案[1]\n```", "    答案[1]\n",
    "[[1]](https://example.com)", "[查看](https://example.com/[1])",
    "![说明[1]](https://example.com/a.png)", "[[1]][ref]\n\n[ref]: https://example.com",
    "<div>答案[1]</div>", "说明 <span>答案[1]</span>", "<!-- [1] -->",
])
async def test_code_or_link_only_marker_is_not_a_citation(answer):
    """删掉token过滤会错误放行代码/链接编号。参数：非正文编号；返回：None；异常：断言失败。"""
    decision = KnowledgeAnswer(useful=True, reason="", answer=answer, citation_numbers=[1])
    with pytest.raises(KnowledgeAnswerError):
        decide_answer(decision, citations())
    service, repository, _ = _service(decision)
    events = [event async for event in service.stream_turn(ChatRequest(session_id="non-body", message="政策条件"))]
    assert any(event.event == "error" for event in events)
    assert not any(event.event in {"delta", "done"} for event in events)
    assert not repository.low_confidence


@pytest.mark.parametrize("answer", ["答案[1,4]", "答案[1，4]", "答案[1][4]", "**答案[1]**", "> 答案[1]",
                                     "答案[1]，`代码[99]`", "答案[1] <br> 换行", "<span>HTML[99]</span> 答案[1]"])
def test_merged_brackets_and_declared_superset_remain_valid(answer):
    """合法角标与声明超集继续兼容，不因代码中其他编号误拒。参数：正文；返回：None；异常：断言失败。"""
    assert decide_answer(KnowledgeAnswer(useful=True, reason="", answer=answer, citation_numbers=[1, 2, 4]), citations()) == (answer, None)
