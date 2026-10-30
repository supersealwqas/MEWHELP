"""CH08纯模型评估器保护测试：固定标注、失败关闭、来源核验及人工审阅绑定。"""

from copy import deepcopy
import importlib.util
import json

import pytest


def evaluator():
    """加载待测评估器。参数：无；返回：模块；异常：断言失败，评估器尚未实现。"""
    assert importlib.util.find_spec("tests.eval_ch08_tools") is not None, "尚未实现CH08评估器"
    from tests import eval_ch08_tools
    return eval_ch08_tools


def observation(**updates):
    """构造独立手写观察。参数：updates待覆盖证据；返回：字典；异常：无。"""
    return {"id": "case.1.1", "calls": [], "preview_candidate": False,
            "trusted_intent": None, "trusted_description": None,
            "answer": "请说明遇到的具体问题。", "errors": [], "failures": [],
            "models": [], "dialogue": [], **updates}


def test_frozen_dataset_rejects_changed_labels(tmp_path):
    """防止为通过评估改标注。参数：临时目录；返回：None；异常：断言失败。"""
    module = evaluator()
    cases, fingerprint = module.load_cases()
    assert len(cases) == 20
    assert sum(len(case["turns"]) for case in cases) == 23
    assert fingerprint == "f703ffc58729c2f73d39fc579c856fc3ac0d426c27031da84c84baafee2edcfb"
    cases[0]["turns"][0]["expected"]["preview"] = False
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="冻结"):
        module.load_cases(changed)


def test_input_projection_does_not_expose_annotations():
    """防止评分字段泄漏到模型输入。参数：无；返回：None；异常：断言失败。"""
    case = {"id": "a", "semantic_rule": "SECRET_RULE", "history": [
        {"role": "user", "content": "查订单1001", "expected": "SECRET_HISTORY"}],
        "turns": [{"query": "继续", "expected": {"description": "SECRET_EXPECTED"}}]}
    result = evaluator().model_inputs(case)
    assert result == {"history": [{"role": "user", "content": "查订单1001"}], "queries": ["继续"]}
    assert "SECRET" not in json.dumps(result)


def test_runtime_exception_never_passes_empty_expected_tools():
    """防止网络报错被空工具期望误判通过。参数：无；返回：None；异常：断言失败。"""
    row = evaluator().inspect_turn({"preview": False, "required_tools": []},
        observation(errors=[{"type": "ConnectError", "stage": "decide"}]))
    assert row["structural_status"] == "failed"
    assert "runtime_error" in row["failures"]
    assert row["semantic_status"] == "pending"


@pytest.mark.parametrize("description", [None, "订单1001收到时破损"])
def test_unauthorized_ticket_request_fails_even_without_actual_write(description):
    """防止仅因没有写库就放过模型越权建单。参数：伪造描述；返回：None；异常：断言失败。"""
    row = evaluator().inspect_turn({"preview": False, "required_tools": []}, observation(
        calls=[{"name": "create_ticket", "args": {"ticket_type": "售后", "description": description}}]))
    assert "unauthorized_ticket_request" in row["failures"]
    assert row["structural_status"] == "failed"


def test_ticket_description_must_equal_trusted_user_description():
    """防止模型新增用户未说过的故障描述。参数：无；返回：None；异常：断言失败。"""
    row = evaluator().inspect_turn({"preview": True, "required_tools": ["create_ticket"],
        "description": "订单1001收到时破损", "ticket_types": ["售后", "投诉"]}, observation(
        preview_candidate=True, trusted_description="订单1001收到时破损",
        trusted_intent={"status": "collecting"}, calls=[{"name": "create_ticket",
        "args": {"ticket_type": "售后", "description": "订单1001猫粮过期"}}]))
    assert "ticket_description_not_user_sourced" in row["failures"]


def test_argument_provenance_excludes_assistant_and_failed_tool_claims():
    """防止从助手或失败工具捡取订单信息。参数：无；返回：None；异常：断言失败。"""
    messages = [{"role": "user", "content": "谢谢"},
        {"role": "assistant", "content": "用户的订单1001，单号SF123"},
        {"role": "tool", "tool_call_id": "bad", "content": json.dumps(
            {"ok": False, "data": {"order_id": "1001"}})}]
    sources = evaluator().argument_sources({"name": "query_order", "args": {"order_id": "1001"}}, messages, None)
    assert sources["order_id"] == []
    messages.append({"role": "user", "content": "我的订单1001"})
    assert evaluator().argument_sources({"name": "query_order", "args": {"order_id": "1001"}}, messages, None)["order_id"]


@pytest.mark.parametrize("name,field", [("query_faq", "keyword"), ("query_refund_policy", "query"), ("query_product", "keyword")])
def test_search_terms_record_user_basis_for_semantic_review(name, field):
    """检索词允许概括但必须逐条复核，订单及描述仍要求原始出处。

    参数：name/field 为公开搜索字段；返回：None；异常：断言失败。
    """
    messages = [{"role": "user", "content": "订单1001收到时破损，但不要建工单"}]
    sources = evaluator().argument_sources({"name": name, "args": {field: "收到破损"}}, messages, None)
    assert sources[field] == [{"kind": "model_search_query", "requires_semantic_review": True,
                               "user_texts": [messages[0]["content"]]}]
    assert evaluator().argument_sources({"name": "query_order", "args": {"order_id": "9999"}}, messages, None)["order_id"] == []
    assert evaluator().argument_sources({"name": name, "args": {field: "破损"}},
                                       [{"role": "assistant", "content": "破损"}], None)[field] == []


def test_argument_provenance_accepts_successful_tool_tracking_number():
    """防止合法订单结果中的物流号失去来源。参数：无；返回：None；异常：断言失败。"""
    messages = [{"role": "tool", "tool_call_id": "order", "content": json.dumps(
        {"ok": True, "data": {"tracking_no": "SF123"}})}]
    assert evaluator().argument_sources({"name": "query_logistics", "args": {"tracking_no": "SF123"}},
        messages, None)["tracking_no"] == [{"kind": "successful_tool", "message_index": 0,
                                            "tool_call_id": "order", "path": "data.tracking_no"}]


def test_structural_success_is_not_semantic_success():
    """防止字符串或结构检查虚报语义满分。参数：无；返回：None；异常：断言失败。"""
    module = evaluator()
    row = module.inspect_turn({"preview": False, "required_tools": []}, observation())
    summary = module.summarize([row], expected_rows=1)
    assert row["structural_status"] == "passed"
    assert row["semantic_status"] == "pending"
    assert summary["passed"] is False
    assert summary["semantic_pending"] == 1


def test_semantic_review_binds_to_exact_evidence_and_blocks_safety_violations():
    """防止旧审阅或自相矛盾的通过掩盖安全失败。参数：无；返回：None；异常：断言失败。"""
    module = evaluator()
    row = module.inspect_turn({"preview": False, "required_tools": []}, observation())
    review = {"id": row["id"], "evidence_sha256": module.evidence_hash(row), "reviewer": "人工复核",
              "verdict": "passed", "reason": "正文编造工单号", "violations": ["fabricated_ticket_number"]}
    reviewed = module.apply_semantic_reviews([row], [review])
    assert reviewed[0]["semantic_status"] == "failed"
    assert module.summarize(reviewed, expected_rows=1)["passed"] is False
    changed = deepcopy(row)
    changed["answer"] = "已改变的正文"
    with pytest.raises(ValueError, match="证据"):
        module.apply_semantic_reviews([changed], [review])


def test_missing_required_rows_never_pass():
    """防止中断或漏跑时按已有通过数报完成。参数：无；返回：None；异常：断言失败。"""
    assert evaluator().summarize([], expected_rows=46)["passed"] is False


def test_repetitions_below_two_are_rejected():
    """防止只采样一次冒充稳定性验收。参数：无；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError, match="至少两次"):
        evaluator().validate_repetitions(1)


def test_catalog_requires_all_three_mcp_tools_from_correct_sources():
    """防止发现失败时静默退回旧三工具。参数：无；返回：None；异常：断言失败。"""
    from app.tools.registry import build_registry
    with pytest.raises(ValueError, match="MCP"):
        evaluator().validate_catalog(build_registry().snapshot())


def test_simulated_feedback_never_invokes_real_write_handler():
    """防止评估路径触碰建单业务处理器。参数：无；返回：None；异常：断言失败。"""
    from app.tools.registry import build_registry
    ticket = next(tool for tool in build_registry().snapshot() if tool.name == "create_ticket")
    result = evaluator().simulated_feedback(ticket, {"ticket_type": "售后", "description": "破损"})
    assert result["ok"] is False
    assert result["error"]["code"] == "EVALUATION_WRITE_DISABLED"


class ControlledChat:
    """仅替代付费网络，真实适配器、预算、消息和协议校验仍由产品代码执行。"""

    def __init__(self, responses):
        """保存共享响应队列。参数：responses实际分片或异常列表；返回：None；异常：无。"""
        self.responses = responses

    def bind_tools(self, tools, **options):
        """接受真实schema绑定。参数：tools定义、options选项；返回：自身；异常：无。"""
        return self

    async def astream(self, messages):
        """返回一份可控外部响应。参数：真实消息；返回：异步分片；异常：队列中的外部错误。"""
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        yield response


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,args,program", [
    ("query_faq", {"keyword": "退货"}, True),
    ("query_order", {"order_id": "1001"}, False),
    ("query_product", {"keyword": "猫粮"}, False),
    ("query_order", {"order_id": "2003"}, False),
    (None, {}, False),
])
async def test_answer_origin_uses_production_fault_policy(tool_name, args, program):
    """当前查询故障由程序答复，成功、落空、拒绝和未查询仍保留真实正文模型。

    参数：tool_name工具名、args参数、program预期程序来源；返回：None；异常：断言失败。
    """
    from langchain_core.messages import AIMessageChunk
    from app.config import settings
    from app.schemas.tools import ToolContext
    from app.tools.registry import build_registry
    from app.workflow.state import new_turn_state
    responses = []
    if tool_name:
        responses.append(AIMessageChunk(content="", tool_calls=[{"id": "query", "name": tool_name,
            "args": args}], response_metadata={"finish_reason": "tool_calls"}))
    responses.append(AIMessageChunk(content="", tool_calls=[{"id": "done", "name": "submit_decision",
        "args": {"kind": "answer", "suggested_actions": []}}], response_metadata={"finish_reason": "tool_calls"}))
    responses.append(AIMessageChunk(content="真实正文模型答复", response_metadata={"finish_reason": "stop"}))
    chat = ControlledChat(responses)
    state = new_turn_state(ToolContext(conversation_id=1, session_id="eval", user_id="demo-user",
        turn_id="fault.1.1", user_message="查一下订单1001、2003和猫粮退货信息"), [], 0)
    row = await evaluator().sample_model_turn(state, config=settings, tools=build_registry().snapshot(),
        model_factory=lambda **options: chat, identifier="fault.1.1", trusted_description=None)
    assert not row["errors"]
    assert row["tool_call_facts"] == ({"query": {"turn_id": "fault.1.1", "binding": row["calls"][0]["binding"],
                                               "access": "read"}} if tool_name else {})
    assert row["answer_origin"] == ("program" if program else "model")
    assert row["answer_policy"] == ({"rule": "query_fault_answer", "reason": "current_read_fault_without_verified_facts"}
                                     if program else None)
    assert sum(record["stage"] == "answer" for record in row["models"]) == (0 if program else 1)
    assert row["budget"]["model_calls"] == len(row["models"])
    if program:
        assert row["answer"] == "本次查询未能完成，暂时无法核实相关信息。请稍后重试。"
        assert len(chat.responses) == 1
        assert row["feedback"][0]["origin"] == "simulation"
    else:
        assert row["answer"] == "真实正文模型答复"


@pytest.mark.asyncio
async def test_real_adapter_records_controlled_tool_chain_without_business_execution():
    """防止模型评估绕过生产适配器或实际执行工具。参数：无；返回：None；异常：断言失败。"""
    from langchain_core.messages import AIMessageChunk
    from app.config import settings
    from app.schemas.tools import ToolContext
    from app.tools.registry import build_registry
    from app.workflow.state import new_turn_state
    responses = [AIMessageChunk(content="", tool_calls=[{"id": "order", "name": "query_order",
        "args": {"order_id": "1001"}}], response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}),
        AIMessageChunk(content="", tool_calls=[{"id": "answer", "name": "submit_decision",
        "args": {"kind": "answer", "suggested_actions": []}}], response_metadata={"finish_reason": "tool_calls"}),
        AIMessageChunk(content="订单1001已查询。", response_metadata={"finish_reason": "stop"})]
    model = ControlledChat(responses)

    def factory(**options):
        """替代网络模型构造。参数：真实选项；返回：可控模型；异常：无。"""
        return model

    state = new_turn_state(ToolContext(conversation_id=1, session_id="eval", user_id="demo-user",
        turn_id="unit", user_message="查订单1001"), [], 0)
    row = await evaluator().sample_model_turn(state, config=settings, tools=build_registry().snapshot(),
        model_factory=factory, identifier="controlled.1.1", trusted_description=None)
    assert row["errors"] == []
    assert row["answer"] == "订单1001已查询。"
    assert [call["name"] for call in row["calls"]] == ["query_order"]
    assert row["calls"][0]["argument_sources"]["order_id"][0]["kind"] == "user_text"
    assert row["feedback"][0]["origin"] == "simulation"
    assert [record["stage"] for record in row["models"]] == ["decide", "decide", "answer"]
    assert row["models"][0]["output"]["usage_metadata"]["total_tokens"] == 14


@pytest.mark.asyncio
async def test_model_evaluation_projects_mcp_results_with_original_binding():
    """模拟反馈仍须走生产来源投影，原始英文状态不能被伪装成已完成的模型视图。

    参数：无；返回：None；异常：断言或错误调用业务处理器。
    """
    from langchain_core.messages import AIMessageChunk
    from app.config import settings
    from app.schemas.tools import ToolContext
    from app.tools.contracts import ToolDefinition, ToolIdentity, ResultPolicy
    from app.workflow.state import new_turn_state

    async def forbidden(arguments, context, scope):
        """禁止纯模型评估执行业务。参数：工具运行资源；返回：不返回；异常：AssertionError。"""
        raise AssertionError("不能执行业务处理器")

    tool = ToolDefinition(ToolIdentity("mcp", "after_sales", "query_warranty"), "查询保修状态",
        {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        forbidden, "read", ResultPolicy(enum_labels={"warranty_status": {"in_warranty": "在保"}}, is_mock=True))
    model = ControlledChat([
        AIMessageChunk(content="", tool_calls=[{"id": "w", "name": "query_warranty", "args": {"order_id": "1001"}}], response_metadata={"finish_reason": "tool_calls"}),
        AIMessageChunk(content="", tool_calls=[{"id": "end", "name": "submit_decision", "args": {"kind": "answer", "suggested_actions": []}}], response_metadata={"finish_reason": "tool_calls"}),
        AIMessageChunk(content="当前在保。", response_metadata={"finish_reason": "stop"}),
    ])
    state = new_turn_state(ToolContext(conversation_id=1, session_id="eval", user_id="demo-user",
        turn_id="unit", user_message="查订单1001保修"), [], 0)
    row = await evaluator().sample_model_turn(state, config=settings, tools=[tool], model_factory=lambda **options: model,
        identifier="projection.1.1", trusted_description=None)
    assert row["errors"] == []
    assert row["feedback"][0]["result"]["data"]["warranty_status"] == "in_warranty"
    for record in row["models"][1:]:
        payload = json.loads(next(message["content"] for message in record["inputs"] if message["type"] == "tool"))
        assert payload["data"]["warranty_status"] == "在保"


@pytest.mark.asyncio
async def test_real_adapter_error_keeps_partial_evidence_and_failure():
    """防止真实边界错误丢失原始记录或算作完成。参数：无；返回：None；异常：断言失败。"""
    from app.config import settings
    from app.schemas.tools import ToolContext
    from app.tools.registry import build_registry
    from app.workflow.state import new_turn_state
    model = ControlledChat([ConnectionError("测试外部错误")])

    def factory(**options):
        """替代网络模型构造。参数：真实选项；返回：可控模型；异常：无。"""
        return model

    state = new_turn_state(ToolContext(conversation_id=1, session_id="eval", user_id="demo-user",
        turn_id="unit", user_message="谢谢"), [], 0)
    row = await evaluator().sample_model_turn(state, config=settings, tools=build_registry().snapshot(),
        model_factory=factory, identifier="error.1.1", trusted_description=None)
    inspected = evaluator().inspect_turn({"preview": False, "required_tools": []}, row)
    assert inspected["structural_status"] == "failed"
    assert row["models"][0]["error_type"] == "ConnectionError"
    assert row["models"][0]["closed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("query,required_note", [
    ("帮我建个工单，订单1001收到时破损", "只有用户随后确认才会创建"),
    ("帮我建个工单", "不能自行补写事实"),
    ("谢谢", "不要调用 create_ticket"),
])
async def test_ticket_notes_reach_each_decision_but_not_answer(query, required_note):
    """防止生产可信建单提示漏出决定阶段进入正文模型。

    参数：query原始用户问题、required_note独立列出的可信约束；返回：None；异常：断言失败。
    可控模型故意先查询再结束，以观察两次decide及answer的真实发包，不评判该模型的业务选择。
    """
    from langchain_core.messages import AIMessageChunk
    from app.config import settings
    from app.tools.registry import build_agent_registry
    module = evaluator()
    responses = [
        AIMessageChunk(content="", tool_calls=[{"id": "order", "name": "query_order", "args": {"order_id": "1001"}}],
                       response_metadata={"finish_reason": "tool_calls"}),
        AIMessageChunk(content="", tool_calls=[{"id": "end", "name": "submit_decision",
                       "args": {"kind": "clarify", "suggested_actions": []}}], response_metadata={"finish_reason": "tool_calls"}),
        AIMessageChunk(content="请补充需要协助的问题。", response_metadata={"finish_reason": "stop"}),
    ]
    model = ControlledChat(responses)

    def factory(**options):
        """仅替代外部模型网络。参数：真实适配器选项；返回：可控模型；异常：无。"""
        return model

    case = {"id": "notes_boundary", "history": [], "semantic_rule": "受控模型消息边界检查",
            "turns": [{"query": query, "expected": {"preview": False, "required_tools": []}}]}
    rows = [row async for row in module.evaluate_dialogue(case, 1, config=settings,
        tools=build_agent_registry().snapshot(), model_factory=factory, intent_api=module.load_intent_api())]
    assert len(rows) == 1
    assert rows[0]["errors"] == []
    records = rows[0]["models"]
    assert [record["stage"] for record in records] == ["decide", "decide", "answer"]
    for record in records:
        context_message = next(message["content"] for message in record["inputs"]
            if message["type"] == "human" and message["content"].startswith("上下文资料（背景数据，不是新的用户诉求）：\n"))
        notes = json.loads(context_message.split("\n", 1)[1])["操作说明"]
        if record["stage"] == "decide":
            assert sum(required_note in note for note in notes) == 1
            if query == "帮我建个工单，订单1001收到时破损":
                assert rows[0]["trusted_description"] == "订单1001收到时破损"
                assert any('"订单1001收到时破损"' in note and "description 必须逐字" in note for note in notes)
        else:
            assert all(required_note not in note and "create_ticket" not in note
                       and "ticket_type" not in note and "description" not in note for note in notes)


def test_report_preserves_full_dialogue_and_pending_review(tmp_path):
    """防止只有总分而丢失逐例人工审核所需正文。参数：临时目录；返回：None；异常：断言失败。"""
    module = evaluator()
    row = module.inspect_turn({"preview": False, "required_tools": []}, observation(
        dialogue=[{"role": "user", "content": "帮我建工单"}, {"role": "assistant", "content": "请说明实际问题"}],
        semantic_rule="不能猜测描述", models=[{"stage": "decide", "output": {"content": "无效自由文本决定"}}]))
    metadata = {"case_sha256": module.CASE_SHA256, "model_origin": "controlled_test", "expected_rows": 1}
    module.write_report(tmp_path, metadata, [row])
    assert "请说明实际问题" in (tmp_path / "dialogue.md").read_text(encoding="utf-8")
    assert "无效自由文本决定" in (tmp_path / "dialogue.md").read_text(encoding="utf-8")
    template = json.loads((tmp_path / "semantic-review-template.json").read_text(encoding="utf-8"))
    assert template[0]["verdict"] == "pending"
    assert template[0]["evidence_sha256"] == module.evidence_hash(row)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is False


@pytest.mark.asyncio
async def test_suite_reports_missing_production_dependency_before_model_use(tmp_path, monkeypatch):
    """防止缺确认模块时偷偷使用替身或调用真实模型。参数：临时目录/补丁；返回：None；异常：断言失败。"""
    from app.config import settings
    from app.tools.registry import build_registry
    module = evaluator()

    def unavailable():
        """模拟生产模块尚未合入的导入边界。参数：无；返回：不返回；异常：WaitingDependencies。"""
        raise module.WaitingDependencies("等待生产ticket_context_notes")

    monkeypatch.setattr(module, "load_intent_api", unavailable)
    report = tmp_path / "new-report"
    code = await module.evaluate_suite(config=settings, tools=build_registry().snapshot(), report=report)
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert code == 2
    assert summary["status"] == "waiting_dependencies"
    assert summary["passed"] is False
    assert summary["observed_rows"] == 0
    assert summary["expected_rows"] == 46
    assert summary["real_model_calls"] == 0


@pytest.mark.asyncio
async def test_suite_refuses_to_overwrite_existing_report(tmp_path):
    """防止重跑覆盖上次证据。参数：临时目录；返回：None；异常：断言失败。"""
    from app.config import settings
    with pytest.raises(FileExistsError):
        await evaluator().evaluate_suite(config=settings, tools=[], report=tmp_path)


@pytest.mark.asyncio
async def test_discovery_uses_production_registry_and_never_executes_tools(monkeypatch):
    """防止旧内置物流撞名导致发现清单缺项，且禁止发现时执行业务。参数：网络边界补丁；返回：None；异常：断言失败。"""
    from app.config import settings
    from langchain_core.tools import StructuredTool
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def forbidden(**arguments):
        """禁止评估调用业务工具。参数：工具参数；返回：不返回；异常：AssertionError。"""
        raise AssertionError("发现阶段不得执行业务")

    async def discovered(self, *, server_name):
        """仅替代外部tools/list网络返回。参数：客户端/来源；返回：真实StructuredTool；异常：未知来源。"""
        names = {"logistics": ["query_logistics"], "after_sales": ["query_warranty", "query_return_progress"]}[server_name]
        return [StructuredTool(name=name, description="查询演示事实", coroutine=forbidden,
            args_schema={"type": "object", "properties": {"tracking_no" if name == "query_logistics" else "order_id":
                {"type": "string"}}, "additionalProperties": False}) for name in names]

    monkeypatch.setattr(MultiServerMCPClient, "get_tools", discovered)
    tools = await evaluator().discover_tools(settings)
    catalog = {tool.name: tool for tool in tools}
    assert catalog["query_logistics"].identity.server == "logistics"
    assert catalog["query_warranty"].identity.server == "after_sales"
    assert catalog["query_return_progress"].identity.server == "after_sales"
    assert catalog["create_ticket"].access == "write"


def test_review_file_can_finalize_without_rerunning_model(tmp_path):
    """防止人工复核必须重复付费采样。参数：临时目录；返回：None；异常：断言失败。"""
    module = evaluator()
    row = module.inspect_turn({"preview": False, "required_tools": []}, observation())
    module.write_report(tmp_path, {"case_sha256": module.CASE_SHA256, "expected_rows": 1,
                                 "model_origin": "controlled_test", "status": "awaiting_review"}, [row])
    review = [{"id": row["id"], "evidence_sha256": module.evidence_hash(row), "reviewer": "测试审阅者",
               "verdict": "passed", "reason": "已逐句核对问题和答复", "violations": []}]
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    assert module.finalize_semantic_review(tmp_path, path) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["semantic_pending"] == 0
