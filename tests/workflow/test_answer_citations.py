"""真实工作流正文出口的引用边界回归：逐片发布与完成落库必须使用同一份合法正文。"""

import asyncio

import pytest

from app.schemas.workflow import AgentDecision
from app.workflow.graph import build_workflow
from tests.workflow.graph_fakes import dependencies, ranked, turn


@pytest.mark.parametrize("chunk_size", [1, 3, 1000])
@pytest.mark.parametrize("has_evidence,raw,expected", [
    (False, "支持七天退货[1]，订单金额为89元[query_order]。", "支持七天退货，订单金额为89元。"),
    (True, "本轮政策[1]，旧编号[9]；合并[1，9]，连续[1][9]。", "本轮政策[1]，旧编号；合并[1]，连续[1]。"),
    (False, "物流[query_logistics]和商品[query_product]均已查询。", "物流和商品均已查询。"),
    (False, "政策[query_faq][query_refund_policy]，操作[create_ticket][submit_decision][submit_intent]。", "政策，操作。"),
    (False, "状态[待处理]，范围[1-3]，备注[套餐 A]和[query_order_extra]。", "状态[待处理]，范围[1-3]，备注[套餐 A]和[query_order_extra]。"),
    (False, "[1](https://example.com/[2])、[[3]](https://example.com)和[query_order](/help)。", "[1](https://example.com/[2])、[[3]](https://example.com)和[query_order](/help)。"),
    (False, "![图片[1]](/a.png)和[[2]][帮助]\n\n[帮助]: /help", "![图片[1]](/a.png)和[[2]][帮助]\n\n[帮助]: /help"),
    (False, "普通`代码[1]`、``代码`[query_order]``，正文[9]。", "普通`代码[1]`、``代码`[query_order]``，正文。"),
    (False, "```text\n数组[1]和[query_order]\n```\n正文[9]。", "```text\n数组[1]和[query_order]\n```\n正文。"),
    (False, "~~~text\n数组[1]\n~~~\n正文[9]。", "~~~text\n数组[1]\n~~~\n正文。"),
    (False, "    数组[1]和[query_order]\n\n正文[9]。", "    数组[1]和[query_order]\n\n正文。"),
    (False, "[1](/help \"标题[2]\")。\n\n[2]: /help/[3]", "[1](/help \"标题[2]\")。\n\n[2]: /help/[3]"),
    (True, "政策结尾[1]", "政策结尾[1]"),
    (False, "旧引用结尾[1]", "旧引用结尾"),
    (False, "转义\\[1]与未闭合[说明", "转义\\[1]与未闭合[说明"),
])
async def test_answer_filters_only_unavailable_citations_before_stream_and_storage(chunk_size, has_evidence, raw, expected):
    """恢复原样转发会泄漏失效标记，宽泛替换会破坏链接与普通正文。

    参数：chunk_size 模拟任意模型分片；has_evidence 是否注入本轮编号1；raw 原始输出；expected 人工核对结果。
    返回：None，检查实时文本每一步都为最终合法正文前缀且完成状态、仓储内容一致。
    异常：AssertionError，出现瞬时泄漏、误删、分片丢字或落库不一致。
    """
    deps = dependencies(intents=["商品咨询" if has_evidence else "订单"],
        decisions=[AgentDecision(kind="answer")], ranked=[ranked()] if has_evidence else [],
        answers=[[raw[index:index + chunk_size] for index in range(0, len(raw), chunk_size)]])
    state = turn("这笔订单呢")
    state["history_turns"] = [{"turn_id": "previous-policy", "turn_status": "completed", "messages": [
        {"role": "user", "content": "可以退货吗"},
        {"role": "assistant", "content": "满足条件可七天退货[1]。"},
    ]}]
    emitted, completed = "", None
    async for namespace, mode, payload in build_workflow(deps, checkpointer=None).astream(
            state, stream_mode=["custom", "values"], subgraphs=True):
        if mode == "custom" and payload["event"] == "delta":
            emitted += payload["data"]["delta"]
            assert expected.startswith(emitted)
        elif mode == "values" and not namespace:
            completed = payload
    assert emitted == completed["answer"] == deps.chat.finished[0]["content"] == expected
    assert [item.n for item in deps.chat.finished[0]["citations"]] == ([1] if has_evidence else [])


async def test_pending_marker_does_not_hold_already_available_plain_text():
    """把过滤做成整答缓冲或把标记前文字一起缓存时失败。

    参数：无；返回：None，观测模型等待后续片段时页面已收到首段文字。
    异常：AssertionError 或 TimeoutError，正文流被阻塞。
    """
    deps = dependencies(decisions=[AgentDecision(kind="answer")])
    received = asyncio.Event()

    async def stream(messages, ledger):
        """生成待闭合伪引用。参数：messages、ledger 本轮依赖；返回：异步正文；异常：等待超时。"""
        ledger.reserve("answer", 10, 4096)
        yield "订单金额为89元[query_"
        await asyncio.wait_for(received.wait(), 2)
        yield "order]。"

    deps.model.stream_answer = stream
    pieces = []
    async for _, event in build_workflow(deps, checkpointer=None).astream(turn(), stream_mode="custom", subgraphs=True):
        if event["event"] == "delta":
            pieces.append(event["data"]["delta"])
            if not received.is_set():
                assert pieces == ["订单金额为89元"]
                received.set()
    assert "".join(pieces) == "订单金额为89元。"


@pytest.mark.parametrize("pending", ["[", "[1", "[query_order"])
async def test_interrupted_answer_never_flushes_an_incomplete_marker(pending):
    """在异常清理时错误刷新缓冲会让半个标记漏到页面。

    参数：pending 模型中断前的半个角标；返回：None，核对可见前缀和未完成落库。
    异常：AssertionError，损坏标记被发布或失败答案被保存。
    """
    deps = dependencies(decisions=[AgentDecision(kind="answer")],
        answers=[["订单金额为89元" + pending, ConnectionError("模拟正文断流")]])
    pieces = []
    with pytest.raises(ConnectionError, match="模拟正文断流"):
        async for _, event in build_workflow(deps, checkpointer=None).astream(turn(), stream_mode="custom", subgraphs=True):
            if event["event"] == "delta":
                pieces.append(event["data"]["delta"])
    assert "".join(pieces) == "订单金额为89元"
    assert deps.chat.finished == []
