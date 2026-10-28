"""三层历史视图行为测试：防止改写原文、截断用户句子、越过未完成轮或伪造摘要边界。"""

import importlib
import json

import pytest
from langchain_core.messages import ToolMessage

from app.core.context_budget import ContextBudgetExceeded, TokenEstimator, derive_history_budget
from app.schemas.history import StoredMessage, StoredTurn


def _turn(start, *, user="猫" * 100, answer="答" * 100, status="completed"):
    """构造有主键间隙的轮。参数：start起始ID及原文/状态；返回：StoredTurn；异常：数据校验失败。"""
    return StoredTurn(turn_id=f"t-{start}", turn_status=status, messages=[
        StoredMessage(id=start, role="user", content=user),
        StoredMessage(id=start + 3, role="assistant", content=answer),
    ])


def _budget(history=2000, *, summary=500):
    """构造独立容量。参数：history历史容量、summary摘要预留；返回：预算；异常：预算非法。"""
    return derive_history_budget(window=history + summary, fixed_tokens=0, evidence_tokens=0,
        summary_tokens=summary, output_tokens=0, safety_tokens=0, current_peak_tokens=0,
        target_turns=100, steady_turn_tokens=1000)


def _snapshot(turns, *, summary_upto=0, layer1_from=0, summaries=None, history=2000, summary=500):
    """调用真实投影。参数：历史、锚点及预算；返回：快照；异常：被测校验错误。"""
    module = importlib.import_module("app.core.context_layers")
    return module.build_context_snapshot(turns, summary_upto_msg_id=summary_upto,
        layer1_from_msg_id=layer1_from, summaries=summaries or [],
        budget=_budget(history, summary=summary), estimator=TokenEstimator())


def test_twenty_one_turns_that_fit_are_unchanged_and_never_demoted():
    """按条数压缩、改掉空格或标点都会失败。参数：无；返回：None；异常：断言失败。"""
    turns = [_turn(10 * i + 1, user=f"第{i}轮：猫\n  ", answer=f"答{i}。  ") for i in range(21)]
    snapshot = _snapshot(turns, history=32768)
    assert [message.content for message in snapshot.layer1] == [row.content for turn in turns for row in turn.messages]
    assert snapshot.layer1_from_msg_id == snapshot.summary_upto_msg_id == 0
    assert snapshot.layer2 == [] and not snapshot.summary_required


def test_overflow_demotes_only_oldest_complete_turn_using_real_ids():
    """拿数组下标当锚点、移多一轮或拆轮都会失败。参数：无；返回：None；异常：断言失败。"""
    turns = [_turn(start) for start in (10, 100, 1000, 10000)]
    snapshot = _snapshot(turns)
    assert snapshot.layer1_from_msg_id == 13
    assert [m.additional_kwargs["db_message_id"] for m in snapshot.layer1] == [100, 103, 1000, 1003, 10000, 10003]
    assert snapshot.layer2[0].content == "猫" * 100
    assert snapshot.layer2[1].content.startswith("答" * 60)
    assert snapshot.layer2[1].content.count("答") == 60
    assert snapshot.layer2[1].content != "答" * 60  # 必须明确显示截短。
    assert snapshot.summary_upto_msg_id == 0
    assert not snapshot.summary_required


def test_layer2_user_text_is_never_cut_and_snapshot_does_not_mutate_sources():
    """只压客服原文，快照内容修改也不能改到原始轮。参数：无；返回：None；异常：断言失败。"""
    turns = [_turn(5, user="这是我完整的诉求\n" * 20), _turn(100)]
    before = [turn.model_dump() for turn in turns]
    snapshot = _snapshot(turns, layer1_from=8, history=5000)
    assert snapshot.layer2[0].content == turns[0].messages[0].content
    snapshot.layer1[0].content = "外部改动"
    snapshot.layer2[0].content = "外部改动"
    assert [turn.model_dump() for turn in turns] == before


def _tool_turn(*, paired=True, ok=True):
    """构造大工具结果。参数：paired是否配对、ok真实状态；返回：轮；异常：校验失败。"""
    rows = [StoredMessage(id=11, role="user", content="查订单"),
        StoredMessage(id=17, role="assistant", content="", tool_calls=[
            {"id": "c1", "name": "query_order", "args": {"order_id": "ORD-812"}}])]
    if paired:
        rows.append(StoredMessage(id=22, role="tool", tool_call_id="c1", content=json.dumps(
            {"ok": ok, "data": {"order_id": "ORD-812", "detail": "内容\n" * 1000}}, ensure_ascii=False)))
    rows.append(StoredMessage(id=29, role="assistant", content="查询结束"))
    return StoredTurn(turn_id="tools", turn_status="completed", messages=rows)


@pytest.mark.parametrize("ok,label", [(True, "成功"), (False, "失败")])
def test_large_tool_result_becomes_one_line_with_pair_and_source(ok, label):
    """摘要式工具标识不能丢来源、抹去失败或破坏申请配对。参数：ok/label；返回：None；异常：断言失败。"""
    turn = _tool_turn(ok=ok)
    original = turn.model_dump()
    snapshot = _snapshot([turn, _turn(40)], layer1_from=29, history=5000)
    result = next(m for m in snapshot.layer2 if isinstance(m, ToolMessage))
    assert all(part in result.content for part in ["query_order", "ORD-812", label, "22"])
    assert "\n" not in result.content and len(result.content) < 300
    assert snapshot.layer2[1].tool_calls[0]["id"] == result.tool_call_id == "c1"
    assert turn.model_dump() == original


def test_incomplete_tool_pair_is_not_summarized_or_shown_as_a_success():
    """缺结果的轮不可被锚点跨过。参数：无；返回：None；异常：断言失败。"""
    snapshot = _snapshot([_tool_turn(paired=False)])
    assert snapshot.layer1_from_msg_id == 0 and not snapshot.summary_required
    assert not any(getattr(m, "tool_calls", []) or isinstance(m, ToolMessage) for m in snapshot.layer1)
    with pytest.raises(ValueError, match="未完成|配对"):
        _snapshot([_tool_turn(paired=False)], layer1_from=29)


@pytest.mark.parametrize("status", ["running", "awaiting_input", "uncertain"])
def test_boundary_cannot_cross_unfinished_turn(status):
    """终态之外的轮不能被移动到摘要候选区。参数：status状态；返回：None；异常：断言失败。"""
    turns = [_turn(1, status=status), *[_turn(i) for i in (100, 200, 300)]]
    with pytest.raises(ContextBudgetExceeded):
        _snapshot(turns)


def test_pending_summary_omits_whole_old_layer2_turns_without_advancing_summary():
    """后台没完成前不能假装已摘要，也不能截开超长用户原话。参数：无；返回：None；异常：断言失败。"""
    turns = [_turn(10, user="用户长问题" * 200), _turn(1000, user="新", answer="答"), _turn(2000)]
    before = [turn.model_dump() for turn in turns]
    snapshot = _snapshot(turns, layer1_from=1003)
    assert snapshot.summary_required and snapshot.layer2_tokens > snapshot.budget.layer2
    assert snapshot.summary_upto_msg_id == 0 and snapshot.layer1_from_msg_id == 1003
    assert snapshot.omitted_ranges == [(10, 13)]
    assert [m.content for m in snapshot.layer2] == ["新", "答"]
    assert [turn.model_dump() for turn in turns] == before


def test_summary_uses_whole_recent_segments_and_skips_valid_empty_segments():
    """拼接时不能重写旧段、截断编号或注入空段标记。参数：无；返回：None；异常：断言失败。"""
    summaries = [dict(seq=1, from_msg_id=1, upto_msg_id=4, content="甲" * 40),
        dict(seq=2, from_msg_id=10, upto_msg_id=13, content=""),
        dict(seq=3, from_msg_id=20, upto_msg_id=23, content="订单ORD-812待查")]
    snapshot = _snapshot([_turn(100)], summary_upto=23, layer1_from=23, summaries=summaries, summary=40)
    assert snapshot.summary == "订单ORD-812待查" and snapshot.summary_seqs == [3]


@pytest.mark.parametrize("anchor", [11, 12])
def test_anchor_may_not_split_a_turn(anchor):
    """边界位于轮内时必须拒绝，不能静默丢半轮。参数：anchor非法锚点；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        _snapshot([_turn(10)], layer1_from=anchor)


def test_missing_ids_cannot_be_replaced_with_message_counts():
    """旧数据没主键时不得自动按条数伪造边界。参数：无；返回：None；异常：断言失败。"""
    turn = StoredTurn(turn_id="legacy", turn_status="completed", messages=[StoredMessage(role="user", content="原话")])
    with pytest.raises(ValueError, match="ID|主键"):
        _snapshot([turn])


def test_orphan_result_without_call_id_blocks_demotion_without_type_error():
    """坏历史应明确阻止推进，不能因比较None与字符串崩溃。参数：无；返回：None；异常：断言失败。"""
    turn = _tool_turn()
    turn.messages.insert(3, StoredMessage(id=24, role="tool", content="孤立结果"))
    with pytest.raises(ValueError, match="未完成|配对"):
        _snapshot([turn], layer1_from=29)


def test_small_tool_result_and_recent_tool_calls_keep_original_text():
    """小结果及最近申请正文不能被不必要压缩或丢字。参数：无；返回：None；异常：断言失败。"""
    turn = _tool_turn()
    turn.messages[1].content = "查询前说明  "
    turn.messages[2].content = '{"ok":true,"data":{"order_id":"ORD-812"}}'
    recent = _snapshot([turn])
    assert recent.layer1[1].content == "查询前说明  "
    assert recent.layer1[2].content == turn.messages[2].content
    half = _snapshot([turn], layer1_from=29)
    assert half.layer2[2].content == turn.messages[2].content


def interleaved_turns(*, late_result=False, large=False):
    """构造动作与新问交错的原文。参数：结果是否延迟、是否超原文预算；返回：两轮；异常：字段校验错误。"""
    action = _tool_turn()
    action.messages[-1].id = 35
    action.messages[2].content = '{"ok":true,"data":{"order_id":"ORD-812"}}'
    if late_result:
        action.messages[2].id = 32
    middle = _turn(26, user="中间新问题", answer="中间答复")
    if large:
        action.messages[0].content = "查询订单" * 150
        middle.messages[0].content = "中间新问题" * 100
    return [action, middle]


@pytest.mark.parametrize("late_result,expected", [
    (False, [11, 17, 22, 26, 29, 35]), (True, [11, 17, 32, 26, 29, 35])])
def test_interleaved_originals_render_in_time_order_with_atomic_tool_pairs(late_result, expected):
    """交错回执不能误判乱序或让用户消息插入工具配对。参数：结果时序及手算顺序；返回：None；异常：断言失败。"""
    turns = interleaved_turns(late_result=late_result)
    originals = [turn.model_dump() for turn in turns]
    snapshot = _snapshot(turns, history=5000)
    assert [message.additional_kwargs["db_message_id"] for message in snapshot.layer1] == expected
    assert [turn.model_dump() for turn in turns] == originals
    assert snapshot.layer1_from_msg_id == 0


def test_interleaved_turns_demote_and_omit_as_one_complete_range():
    """按单轮切分会使层1/层2区间重叠；相交轮必须整体移动。参数：无；返回：None；异常：断言失败。"""
    snapshot = _snapshot([*interleaved_turns(large=True), _turn(100)])
    assert snapshot.layer1_from_msg_id == 35
    assert [message.additional_kwargs["db_message_id"] for message in snapshot.layer1] == [100, 103]
    assert snapshot.omitted_ranges == [(11, 35)] and snapshot.summary_required


@pytest.mark.parametrize("boundary", [22, 29])
def test_interleaved_turn_end_is_not_a_complete_block_boundary(boundary):
    """某一轮结束仍可能切开另一轮，不允许推进到块内。参数：块内主键；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        _snapshot(interleaved_turns(), layer1_from=boundary)


def test_interleaving_support_still_rejects_duplicate_message_ids():
    """支持相交范围不能放松真实主键唯一性。参数：无；返回：None；异常：断言失败。"""
    turns = interleaved_turns()
    turns[1].messages[0].id = 22
    with pytest.raises(ValueError, match="ID|主键"):
        _snapshot(turns)


def test_transitive_interleaving_moves_all_three_turns_without_merging_next_turn():
    """第二轮延伸到第三轮时也须整组推进，紧随其后的独立轮应保留。参数：无；返回：None；异常：断言失败。"""
    turns = [_turn(1, user="长原话" * 200), _turn(10), _turn(25)]
    for turn, end in zip(turns, [20, 30, 40]):
        turn.messages[-1].id = end
    latest = _turn(41)
    snapshot = _snapshot([turns[2], turns[0], latest, turns[1]])
    assert snapshot.layer1_from_msg_id == 40 and snapshot.omitted_ranges == [(1, 40)]
    assert [message.additional_kwargs["db_message_id"] for message in snapshot.layer1] == [41, 44]
    assert (snapshot.original_from_msg_id, snapshot.original_upto_msg_id) == (1, 44)
    with pytest.raises(ValueError):
        _snapshot(turns, layer1_from=30)


@pytest.mark.parametrize("unfinished", ["running", "awaiting_input", "unpaired", "receipt_pending"])
def test_one_unsafe_turn_blocks_demotion_of_entire_interleaved_group(unfinished):
    """不能用已完成邻轮掩盖未完成或工具未配对的轮。参数：不安全状态；返回：None；异常：断言失败。"""
    turns = interleaved_turns(large=True)
    if unfinished == "unpaired":
        turns[0].messages.pop(2)
    elif unfinished == "receipt_pending":
        turns[0].turn_status, turns[0].receipt_pending = "cancelled", True
    else:
        turns[1].turn_status = unfinished
    with pytest.raises(ContextBudgetExceeded):
        _snapshot([*turns, _turn(100)])
    with pytest.raises(ValueError, match="未完成|配对"):
        _snapshot(turns, layer1_from=35)


def test_interleaved_block_can_be_fully_summarized_without_hiding_new_originals():
    """共同末尾摘要后仅隐藏该组，下一完整轮仍应原样可见。参数：无；返回：None；异常：断言失败。"""
    snapshot = _snapshot([*interleaved_turns(), _turn(100)], summary_upto=35, layer1_from=35,
        summaries=[dict(seq=1, from_msg_id=11, upto_msg_id=35, content="查询已完成")])
    assert snapshot.summary == "查询已完成" and snapshot.layer2 == []
    assert [message.additional_kwargs["db_message_id"] for message in snapshot.layer1] == [100, 103]


def test_interleaving_support_still_rejects_disordered_ids_inside_a_turn():
    """跨轮允许交错不代表可把轮内坏顺序偷偷排序修好。参数：无；返回：None；异常：断言失败。"""
    turns = interleaved_turns()
    turns[0].messages.reverse()
    with pytest.raises(ValueError, match="ID|主键"):
        _snapshot(turns)
