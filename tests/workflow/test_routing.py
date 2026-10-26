"""验证七意图四出口、原话透传和固定回复的数据边界。"""

import pytest


@pytest.mark.parametrize("intent,want", [
    ("物流", "business"), ("订单", "business"), ("售后", "business"),
    ("商品咨询", "knowledge"), ("退款退货", "knowledge"),
    ("投诉", "complaint"), ("闲聊", "chitchat"),
])
def test_routes_read_intent_from_state(intent, want):
    """错误映射或读取旧 route 时失败。参数：intent 七类意图、want 固定出口；返回：None；异常：断言失败。"""
    from app.workflow.routing import route_by_intent

    assert route_by_intent({"intent": intent, "route": "old-route",
        "knowledge_query": "商城政策" if want == "knowledge" else None}) == want


@pytest.mark.parametrize("state", [
    {}, {"intent": None}, {"intent": "knowledge"}, {"intent": "退款"},
    {"intent": " 物流 "}, {"intent": 1}, {"intent": ["物流"]}, "物流",
])
def test_routes_reject_missing_unknown_or_non_state_input(state):
    """未知分类或直接字符串误入默认出口时失败。参数：state 非法输入；返回：None；异常：断言失败。"""
    from app.workflow.routing import route_by_intent

    with pytest.raises(ValueError):
        route_by_intent(state)


@pytest.mark.parametrize("query", ["  它还能退吗？ ", " \n" + "问" * 3996 + "\n "], ids=["whitespace", "4000-chars"])
def test_reference_resolution_preserves_the_complete_original_query(query):
    """去首尾空白、改写指代或截断4000字原话时失败。参数：query 合法原话；返回：None；异常：断言失败。"""
    from app.workflow.routing import resolve_reference

    assert resolve_reference(query) == query


@pytest.mark.parametrize("size,want", [(2000, None), (2001, "input_limit"), (4000, "input_limit")])
def test_knowledge_input_limit_is_a_separate_stop_reason(size, want):
    """超长知识原话被放行或错标弱证据时失败。参数：size 字数、want 停止原因；返回：None；异常：断言失败。"""
    from app.workflow.routing import knowledge_input_stop_reason

    assert knowledge_input_stop_reason("问" * size) == want


@pytest.mark.parametrize("kind,want_answer,want_actions", [
    ("chitchat", "您好，我是小猫客服，可以帮您查询商品、订单、物流和售后问题。", []),
    ("complaint", "抱歉给您带来不好的体验。您可以选择转人工或创建工单，也可以继续告诉我具体情况。",
     ["handoff", "create_ticket"]),
    ("fallback", "抱歉，当前知识库证据不足以可靠回答这个问题。请补充必要信息，或联系官方人工客服核实。", []),
    ("input_limit", "问题超过知识检索的 2000 字上限，请拆分后再提问。", []),
])
def test_fixed_reply_returns_only_text_and_unexecuted_suggestions(kind, want_answer, want_actions):
    """固定出口遗留旧建议或生成执行载荷时失败。参数：kind 出口、want_answer 文案、want_actions 建议；返回：None；异常：断言失败。"""
    from app.workflow.routing import fixed_reply

    assert fixed_reply(kind) == {"answer": want_answer, "suggested_kinds": want_actions}


def test_fixed_reply_rejects_agent_routes_and_returns_independent_suggestions():
    """业务请求误走固定回复或两轮共享建议列表时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.routing import fixed_reply

    with pytest.raises(ValueError):
        fixed_reply("business")
    first = fixed_reply("complaint")
    first["suggested_kinds"].clear()
    assert fixed_reply("complaint")["suggested_kinds"] == ["handoff", "create_ticket"]


@pytest.mark.parametrize("query", ["null", " \tnull\n", "None", "nil"], ids=["literal-null", "whitespace-null", "none", "nil"])
def test_routes_normalize_string_null_placeholder(query):
    """完整字符串null占位被当知识问题送去检索时失败。

    参数：query为模型常见的空值字面量。返回：None。
    异常：断言失败，占位被当成知识检索问题。
    """
    from app.workflow.routing import route_by_intent

    # 严格schema已在IntentResult层确定性还原为None，路由不应再把它当知识问法送去检索。
    assert route_by_intent({"intent": "商品咨询", "knowledge_query": query}) == "business"


@pytest.mark.parametrize("query,expected", [(None, "business"),
    ("商品说明中null字段是什么意思？", "knowledge")], ids=["real-null", "null-in-question"])
def test_routes_preserve_real_null_and_normal_null_word(query, expected):
    """真实None或含null词语的正常知识问题被空值保护误伤时失败。参数：query合法值、expected出口；返回：None；异常：断言失败。"""
    from app.workflow.routing import route_by_intent

    assert route_by_intent({"intent": "商品咨询", "knowledge_query": query}) == expected
