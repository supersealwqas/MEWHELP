"""CH06真实根图的订单先行、可信编号、原轮暂停与受限政策回答回归。"""

import json
import pytest
from langgraph.types import Command
from app.schemas.tools import ToolResult
from app.schemas.workflow import UnderstandingResult
from app.workflow.runtime import workflow_runtime
from app.workflow.state import checkpoint_thread_id
from tests.workflow.graph_fakes import dependencies, ranked, turn, NodeTrace


@pytest.fixture(autouse=True)
def offline_chat_runtime():
    """本模块直接组装图，无需全局HTTP服务夹具。参数：无；返回：None；异常：无。"""


def owned_turn(query, *, user_id="demo-user"):
    """构造有限演示用户的原轮。参数：query原问、user_id身份；返回：State；异常：输入非法。"""
    state = turn(query)
    state["context"]["user_id"] = user_id
    return state


def order_result(order_id="1001"):
    """构造成功订单事实。参数：order_id完整主键；返回：ToolResult；异常：数据非法。"""
    return ToolResult.success({"order_id": order_id, "product": "猫粮", "status": "已签收", "created_at": "2026-01-01"})


@pytest.mark.parametrize("intent", ["退款退货", "售后"])
async def test_order_precedes_policy_and_answer_without_agent_decision(tmp_path, intent):
    """政策必须先核实订单且直接回答。参数：tmp_path隔离目录、intent政策类；返回：None；异常：断言失败。"""
    deps = dependencies(intents=[intent], results=[order_result()], ranked=[ranked()], answers=[["订单已签收，政策说明[1]。"]])
    trace = NodeTrace()
    async with workflow_runtime(deps, tmp_path / "graph.sqlite") as graph:
        state = await graph.ainvoke(owned_turn("订单1001能退吗"), {"configurable": {"thread_id": "test"}, "callbacks": [trace]})
    positions = [trace.nodes.index(name) for name in ["resolve_reference", "classify_intent", "fetch_order", "retrieve_policy", "main_agent"]]
    assert positions == sorted(positions)
    assert [row["stage"] for row in deps.model.inputs] == ["understand", "classify", "answer"]
    assert state["order_data"]["order_id"] == "1001" and state["decision_count"] == 0
    assert len(deps.chat.calls) == len(deps.chat.results) == 1
    assert deps.query_expander.calls == [("订单1001能退吗 已签收", "订单1001能退吗")]


@pytest.mark.parametrize("query,candidate", [
    ("我要退货", None), ("订单9999退款", "9999"), ("订单2003退款", "2003"),
    ("SF213502378238退款", "213502378238"), ("213502378238退款", "213502378238"),
    ("快递单号1001退款", "1001"), ("运单号1001退款", "1001"),
    ("订单1001和1002退款", "1001"), ("A-1001退款", "1001"),
])
async def test_tracking_number_never_used_as_order_id(tmp_path, query, candidate):
    """未知、他人、快递或歧义编号只能选本人订单。参数：隔离目录/原问/模型候选；返回：None；异常：断言失败。"""
    understood = UnderstandingResult(resolved_query=query, standard_query=query, needs_resolution=False, order_id=candidate)
    deps = dependencies(intents=["退款退货"], understandings=[understood])
    async with workflow_runtime(deps, tmp_path / "wait.sqlite") as graph:
        state = await graph.ainvoke(owned_turn(query), {"configurable": {"thread_id": "test"}})
    action = state["__interrupt__"][0].value
    assert action["type"] == "select_order" and action["orders"]
    assert "2003" not in {row["order_id"] for row in action["orders"]}
    assert not deps.executor.calls and not deps.chat.calls and not deps.search.calls
    assert [row["stage"] for row in deps.model.inputs] == ["understand", "classify"]


async def test_selection_resume_keeps_turn_and_budget_and_rechecks_owner(tmp_path):
    """原保存器恢复保留理解分类账本并检查所选订单。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["退款退货"], results=[order_result()], ranked=[ranked()], answers=[["已查询政策[1]。"]])
    state = owned_turn("我要退货")
    config = {"configurable": {"thread_id": checkpoint_thread_id(1)}}
    path = tmp_path / "resume.sqlite"
    async with workflow_runtime(deps, path) as graph:
        paused = await graph.ainvoke(state, config)
        before = paused["budget"]["records"]
    async with workflow_runtime(deps, path) as graph:
        done = await graph.ainvoke(Command(resume="1001"), config)
    assert done["context"]["turn_id"] == state["context"]["turn_id"]
    assert done["budget"]["records"][:len(before)] == before
    assert done["order_data"]["order_id"] == "1001"
    assert [row["stage"] for row in deps.model.inputs].count("classify") == 1


async def test_no_owned_orders_finishes_without_interrupt(tmp_path):
    """本人没有订单时明确结束。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["退款退货"])
    async with workflow_runtime(deps, tmp_path / "empty.sqlite") as graph:
        state = await graph.ainvoke(owned_turn("我要退款", user_id="nobody"), {"configurable": {"thread_id": "test"}})
    assert "__interrupt__" not in state and "订单" in state["answer"]
    assert not deps.executor.calls and not deps.search.calls


async def test_resume_rechecks_ownership_before_writing(tmp_path):
    """篡改恢复值不执行查询。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["退款退货"])
    config = {"configurable": {"thread_id": "test"}}
    async with workflow_runtime(deps, tmp_path / "ownership.sqlite") as graph:
        await graph.ainvoke(owned_turn("我要退款"), config)
        with pytest.raises(ValueError, match="不属于"):
            await graph.ainvoke(Command(resume="2003"), config)
    assert not deps.chat.calls and not deps.executor.calls


async def test_prefixed_order_uses_full_identifier(tmp_path):
    """理解成功保留带前缀订单全号。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    query = "A-1001退款"
    understood = UnderstandingResult(resolved_query=query, standard_query=query, needs_resolution=False, order_id="A-1001")
    deps = dependencies(intents=["退款退货"], understandings=[understood], results=[order_result("A-1001")],
                        ranked=[ranked()], answers=[["已核实订单[1]。"]])
    async with workflow_runtime(deps, tmp_path / "prefix.sqlite") as graph:
        state = await graph.ainvoke(owned_turn(query), {"configurable": {"thread_id": "test"}})
    assert state["order_id"] == "A-1001" and deps.executor.calls[0]["args"] == {"order_id": "A-1001"}


async def test_general_returns_policy_does_not_fetch_order(tmp_path):
    """通用规则不要求选订单。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    from app.schemas.workflow import AgentDecision
    deps = dependencies(intents=["商品咨询"], ranked=[ranked()], decisions=[AgentDecision(kind="answer")], answers=[["通用退货规则[1]。"]])
    async with workflow_runtime(deps, tmp_path / "general.sqlite") as graph:
        state = await graph.ainvoke(owned_turn("退货运费谁承担"), {"configurable": {"thread_id": "test"}})
    assert not deps.executor.calls and not deps.query_expander.calls
    assert deps.search.calls == [("retrieve", "退货运费谁承担")]
    assert state["order_id"] is None


async def test_policy_answer_uses_current_facts_without_old_assistant_claims(tmp_path):
    """缺签收时间不把创建时间当签收，旧助手结论不作证据。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["退款退货"], results=[order_result()], ranked=[ranked(0.1, answer="低分政策")], answers=[["已签收，签收时间未知，暂不能确认退货条件。"]])
    initial = owned_turn("订单1001退款")
    initial["history_turns"] = [{"turn_id": "old", "turn_status": "completed", "messages": [
        {"role": "user", "content": "订单1001"}, {"role": "assistant", "content": "肯定已批准退款，签收时间2026-01-01"}]}]
    async with workflow_runtime(deps, tmp_path / "facts.sqlite") as graph:
        state = await graph.ainvoke(initial, {"configurable": {"thread_id": "test"}})
    answer_input = json.dumps(deps.model.inputs[-1]["messages"], ensure_ascii=False)
    assert "肯定已批准退款" not in answer_input and "低分政策" not in answer_input
    assert "created_at" in answer_input and "签收" in answer_input and "原因" in answer_input
    assert state["evidence"] == [] and deps.chat.finished[0]["low_confidence"] is None
