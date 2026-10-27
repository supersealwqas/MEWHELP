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


def context_data(messages):
    """按资料标记读取动态区，不能再取最后一条工具消息。参数：messages模型输入；返回：字典；异常：标记缺失。"""
    content = next(row["content"] for row in messages if row["content"].startswith("上下文资料"))
    return json.loads(content.split("\n", 1)[1])


async def test_selected_scope_reaches_policy_and_model_as_confirmed_input(tmp_path):
    """已选对象丢在旧歧义原话里时失败，恢复后必须显式携带确认事实。参数：tmp_path保存器目录；返回：None；异常：断言失败。"""
    deps = dependencies(intents=["退款退货"], results=[order_result("1002")],
                        ranked=[ranked()], answers=[["请按页面申请。"]])
    original = owned_turn("其中一笔想退，未拆封，但我还没决定哪笔。")
    config = {"configurable": {"thread_id": "selected-scope"}}
    async with workflow_runtime(deps, tmp_path / "scope.sqlite") as graph:
        await graph.ainvoke(original, config)
        state = await graph.ainvoke(Command(resume="1002"), config)
    assert state["raw_query"] == original["raw_query"]
    assert state["selection_action"]["accepted_order_id"] == "1002"
    assert len(state["expanded_queries"]) == 3
    assert "1002" in deps.search.calls[0][1] and "未拆封" in deps.search.calls[0][1]
    inputs = deps.model.inputs[-1]["messages"]
    context = context_data(inputs)
    assert context["已确认对象"] == {"订单号": "1002", "来源": "用户选单"}
    assert context["后续表单收集"] == ["退款原因"]


async def test_verified_order_section_never_promotes_user_amount(tmp_path):
    """用户自报金额进入程序生成的核实事实区时失败。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    data = {"order_id": "1001", "product": "滤芯", "status": "已发货", "total_amount": 129.0}
    deps = dependencies(intents=["退款退货"], results=[ToolResult.success(data)],
                        ranked=[ranked()], answers=[["仅根据已提供政策判断。"]])
    async with workflow_runtime(deps, tmp_path / "provenance.sqlite") as graph:
        state = await graph.ainvoke(owned_turn("订单1001实付9876元，想退货"), {"configurable": {"thread_id": "fact"}})
    facts, explanation = state["answer"].split("**政策与办理说明**", 1)
    assert "129" in facts and "9876" not in facts
    assert "商品总金额" in facts and "仅根据已提供政策判断" in explanation
    context = context_data(deps.model.inputs[-1]["messages"])
    assert "9876" in context["用户陈述（未经核实）"]
    assert context["订单事实"] == data


@pytest.mark.parametrize("stage", ["decide", "answer"])
async def test_business_followup_uses_resolved_question_after_history_trim(tmp_path, stage):
    """理解后的完整问题必须到达业务模型，即使旧历史被整轮裁剪，原始输入仍保留。

    参数：tmp_path独立检查点；stage检查决策或回答入口。
    返回：None；异常：断言失败。
    """
    from app.schemas.workflow import AgentDecision
    resolved = "订单1002的包裹运输到哪里了？"
    understood = UnderstandingResult(resolved_query=resolved, standard_query="订单1002物流进度",
                                     needs_resolution=True, order_id="1002")
    deps = dependencies(intents=["物流"], understandings=[understood],
                        decisions=[AgentDecision(kind="clarify")], answers=[["已明确查询订单1002的物流。"]])
    initial = owned_turn("它呢？")
    initial["history_turns"] = [{"turn_id": "old", "turn_status": "completed", "messages": [
        {"role": "user", "content": "查询订单1002的物流。" + "旧内容" * 12000},
        {"role": "assistant", "content": "包裹运输中。"}]}]
    async with workflow_runtime(deps, tmp_path / "resolved-business.sqlite") as graph:
        state = await graph.ainvoke(initial, {"configurable": {"thread_id": "test"}})
    observed = next(row for row in deps.model.inputs if row["stage"] == stage)
    assert observed["messages"][0]["content"] == "它呢？"
    assert context_data(observed["messages"])["完整问题"] == resolved
    assert state["raw_query"] == state["context"]["user_message"] == "它呢？"
    current = next(message for message in state["messages"] if message.id == state["current_message_id"])
    assert current.type == "human" and current.content == "它呢？"
    assert [row["stage"] for row in deps.model.inputs] == ["understand", "classify", "decide", "answer"]


async def test_policy_answer_receives_unsubmitted_action_state():
    """生成申请表单前没有提交回执，正文输入与程序展示必须保留未提交状态。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.graph import build_workflow
    deps = dependencies(intents=["退款退货"], results=[order_result()], answers=[["请阅读政策说明。"]])
    state = await build_workflow(deps, checkpointer=None).ainvoke(owned_turn("订单1001能退吗"))
    model_input = json.dumps(deps.model.inputs[-1]["messages"], ensure_ascii=False)
    assert "尚未提交" in model_input and "执行回执" in model_input
    assert "申请保存为待处理" not in model_input
    assert "退款申请：尚未提交" in state["answer"]


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
    assert len(deps.query_expander.calls) == 1
    assert "订单1001" in deps.query_expander.calls[0][0] and "猫粮" in deps.query_expander.calls[0][0]
    assert "已签收" in deps.query_expander.calls[0][0] and len(state["expanded_queries"]) == 3


@pytest.mark.parametrize("query,candidate", [
    ("我要退货", None), ("订单9999退款", "9999"), ("订单2003退款", "2003"),
    ("SF213502378238退款", "213502378238"), ("213502378238退款", "213502378238"),
    ("快递单号1001退款", "1001"), ("运单号1001退款", "1001"),
    ("退款金额是1001元，想问能不能退", None),
    ("退款金额是1001元，想问能不能退", "1001"),
    ("1001是快递单号，我想退货", None),
    ("1001是快递单号，我想退货", "1001"),
    ("1001是我的快递单号，我想退货", None),
    ("1001是我的快递单号，我想退货", "1001"),
    ("1001是本次退货的物流单号，我想退货", None),
    ("1001（我的快递单号）对应的东西想退", None),
    ("1001是实际支付金额，我想退款", None),
    ("运单号是1001，我想退款", "1001"),
    ("我在1001年买的，想退货", None),
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


@pytest.mark.parametrize("query", ["1001退款", "订单1001能退吗", "订单1001实付1002元，能退吗",
                                  "订单1001是我买的，运费1002元，能退吗"])
async def test_explicit_order_with_non_order_numbers_can_be_queried(tmp_path, query):
    """排除金额不能误伤明确订单，金额也不能制造多订单歧义。

    参数：tmp_path隔离保存器；query含明确订单与可选金额。
    返回：None；异常：断言失败。
    """
    deps = dependencies(intents=["退款退货"], results=[order_result()], ranked=[ranked()], answers=[["请核实签收时间。"]])
    async with workflow_runtime(deps, tmp_path / "explicit.sqlite") as graph:
        state = await graph.ainvoke(owned_turn(query), {"configurable": {"thread_id": "test"}})
    assert "__interrupt__" not in state
    assert state["order_id"] == "1001" and deps.executor.calls[0]["args"] == {"order_id": "1001"}


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
    # 近期助手原话保留，但不能进入本轮核实事实或执行回执。
    facts = context_data(deps.model.inputs[-1]["messages"])
    assert "肯定已批准退款" in answer_input and "低分政策" not in answer_input
    assert "肯定已批准退款" not in json.dumps(facts, ensure_ascii=False)
    assert facts["办理进度"]["当前阶段"] == "尚未提交" and facts["办理进度"]["执行回执"] is None
    assert "signed_at" not in facts["订单事实"]
    assert "created_at" in answer_input and "签收" in answer_input and "原因" in answer_input
    assert state["evidence"] == [] and deps.chat.finished[0]["low_confidence"] is None


@pytest.mark.parametrize("intent", ["退款退货", "售后"])
@pytest.mark.parametrize("scope_case", ["mixed", "empty", "ignored"])
async def test_order_policy_enforces_document_scope(tmp_path, intent, scope_case):
    """实际根图必须过滤非政策来源，空范围不放宽，索引越界时安全降级。

    参数：tmp_path独立保存器；intent订单政策意图；scope_case混合语料/空范围/索引忽略过滤。
    返回：None；异常：断言失败。
    """
    from app.knowledge.retrieval import HybridKnowledgeSearch
    from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, make_settings
    from tests.rag.test_online_filters import FilterIndex
    from tests.rag.test_retrieval import _chunk, RerankBoundary

    sources = ["doc:returns-policy.md", "doc:after-sales-manual.md", "doc:billing-shipping.md", "doc:member-benefits.md"]
    allowed = [_chunk(index).model_copy(update={"source_key": source}) for index, source in enumerate(sources, 1)]
    outside = _chunk(5).model_copy(update={"source_key": "doc:product-specs.md", "content_type": "manual"})
    chunks = [outside] if scope_case == "empty" else [*allowed, outside]
    index = FilterIndex(chunks, ignore_filter=scope_case == "ignored")
    search = HybridKnowledgeSearch(MemoryKnowledgeStore(snapshots=chunks), FakeEncoder(),
                                   index, RerankBoundary({5: 0.99}), make_settings())
    deps = dependencies(intents=[intent], results=[order_result()], answers=[["需核对适用条件。"]])
    deps.search = search
    async with workflow_runtime(deps, tmp_path / "policy-scope.sqlite") as graph:
        state = await graph.ainvoke(owned_turn("订单1001能退吗"), {"configurable": {"thread_id": "test"}})
    if scope_case == "ignored":
        assert not state["evidence"] and state["stop_reason"] == "retrieval_unavailable"
        assert state["degradations"] and "1001" in state["answer"]
        assert deps.chat.finished[0]["low_confidence"] is None
    elif scope_case == "empty":
        assert state["evidence"] == [] and index.calls == []
    else:
        assert {item["source_key"] for item in state["evidence"]} == set(sources)
        assert index.filters == [[1, 2, 3, 4]] * 3
