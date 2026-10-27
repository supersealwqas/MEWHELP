"""验证八意图五出口、可信理解及固定回复的数据边界。"""

import pytest


@pytest.mark.parametrize("intent,want", [
    ("物流", "business"), ("订单", "business"), ("售后", "order_policy"),
    ("商品咨询", "knowledge"), ("退款退货", "order_policy"),
    ("投诉", "complaint"), ("闲聊", "chitchat"), ("其他", "business"),
])
def test_routes_read_intent_from_state(intent, want):
    """错误映射或读取旧 route 时失败。参数：intent 七类意图、want 固定出口；返回：None；异常：断言失败。"""
    from app.workflow.routing import route_by_intent

    assert route_by_intent({"intent": intent, "route": "old-route",
        "intent_confidence": 0.9}) == want


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


def _understanding(**overrides):
    """构造严格理解候选。参数：overrides覆盖字段；返回：UnderstandingResult；异常：候选非法时校验错误。"""
    from app.schemas.workflow import UnderstandingResult

    return UnderstandingResult(**{"resolved_query": "运费怎么算", "standard_query": "运费怎么算",
                                  "needs_resolution": False, "order_id": None, **overrides})


def _history(*messages):
    """构造保留整轮边界的历史。参数：messages可信存储消息；返回：单轮字典列表；异常：无。"""
    return [{"turn_id": "old", "turn_status": "completed", "messages": list(messages)}]


def test_complete_query_preserves_whitespace():
    """无需消解的完整问题被模型格式化时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    raw = "  运费怎么算？  "
    result = validate_understanding(raw, _understanding(resolved_query="运费怎么算？"), [])
    assert result.resolved_query == raw
    assert result.standard_query == "运费怎么算"


@pytest.mark.parametrize("override", [
    {"standard_query": "订单1001的运费怎么算"},
    {"order_id": "1001"},
    {"resolved_query": "订单1001的运费怎么算"},
])
def test_generic_policy_rejects_old_order(override):
    """完整通用问的任一字段夹带旧订单时必须整组透传。参数：override失真字段；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    raw = "运费怎么算"
    result = validate_understanding(raw, _understanding(**override),
        _history({"role": "user", "content": "订单1001到哪了"}))
    assert result.model_dump() == {"resolved_query": raw, "standard_query": raw,
                                  "needs_resolution": False, "order_id": None}


def test_generic_policy_rejects_known_product_binding():
    """无需消解时标准问法偷偷带入旧商品名也必须回退。参数：无；返回：None；异常：断言失败。"""
    import json
    from app.workflow.routing import validate_understanding

    history = _history({"role": "tool", "content": json.dumps(
        {"ok": True, "data": {"order_id": "1001", "product_name": "智能猫砂盆"}})})
    result = validate_understanding("运费怎么算", _understanding(standard_query="智能猫砂盆的运费规则"), history)
    assert result.standard_query == "运费怎么算" and result.order_id is None


@pytest.mark.parametrize("order_id", ["1001", "1002"])
def test_new_explicit_order_wins(order_id):
    """当前1002被旧1001替换或有效新号被拒时失败。参数：order_id模型候选；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    raw = "订单1002能退吗"
    result = validate_understanding(raw, _understanding(resolved_query=raw, standard_query=f"订单{order_id}能退吗",
        order_id=order_id), _history({"role": "user", "content": "订单1001到哪了"}))
    if order_id == "1002":
        assert result.order_id == "1002" and result.resolved_query == raw
    else:
        assert result.order_id is None and result.standard_query == raw


@pytest.mark.parametrize("raw,resolved,order_id", [
    ("订单1002能退吗", "订单1002和1001能退吗", "1002"),
    ("1001和1002哪单能退", "订单1001能退吗", "1001"),
    ("A-1001能退吗", "订单1001能退吗", "1001"),
])
def test_understanding_rejects_old_additions_ambiguity_and_truncated_ids(raw, resolved, order_id):
    """候选自证、多个对象猜单或截取前缀号时失败。参数：原问/候选/订单；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    result = validate_understanding(raw, _understanding(resolved_query=resolved, standard_query=resolved,
        needs_resolution=True, order_id=order_id), _history({"role": "user", "content": "订单1001"}))
    assert result.order_id is None and result.resolved_query == result.standard_query == raw


def test_understanding_retains_complete_prefixed_order_id():
    """可信带前缀订单被截断或抹掉时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    result = validate_understanding("A-1001能退吗", _understanding(resolved_query="A-1001能退吗",
        standard_query="A-1001订单能否退货", order_id="A-1001"), [])
    assert result.order_id == "A-1001"


def test_understanding_multiple_history_candidates_does_not_guess():
    """上一轮并列两单却随意挑选一个时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    raw = "它能退吗"
    result = validate_understanding(raw, _understanding(resolved_query="订单1001能退吗",
        standard_query="订单1001能退吗", needs_resolution=True, order_id="1001"),
        _history({"role": "user", "content": "1001和1002都是同一商品"}))
    assert result.order_id is None and result.resolved_query == raw


def test_understanding_latest_trusted_object_can_resolve_reference():
    """早前换过订单却阻止最近明确对象的合法指代时失败。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    history = [*_history({"role": "user", "content": "订单1001到哪里了"}),
               {"turn_id": "recent", "turn_status": "completed",
                "messages": [{"role": "user", "content": "订单1002呢"}]}]
    result = validate_understanding("它能退吗", _understanding(resolved_query="订单1002能退吗",
        standard_query="订单1002能否退货", needs_resolution=True, order_id="1002"), history)
    assert result.order_id == "1002" and result.needs_resolution


@pytest.mark.parametrize("derived", ["订单1001能退吗", "订单1001和1002能退吗"])
def test_null_order_field_cannot_hide_stale_reference(derived):
    """留空订单字段不能掩盖完整问或标准问混入旧号。参数：derived旧号派生问；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding

    history = [*_history({"role": "user", "content": "订单1001到哪里了"}),
               {"turn_id": "recent", "turn_status": "completed",
                "messages": [{"role": "user", "content": "订单1002呢"}]}]
    result = validate_understanding("它能退吗", _understanding(resolved_query=derived,
        standard_query=derived, needs_resolution=True, order_id=None), history)
    assert result.resolved_query == result.standard_query == "它能退吗"
    assert result.order_id is None and not result.needs_resolution


@pytest.mark.parametrize("query", [None, "旧知识问题", "null"])
def test_routes_ignore_legacy_checkpoint_knowledge_query(query):
    """旧检查点检索字段影响新八类映射时失败。参数：query旧字段；返回：None；异常：断言失败。"""
    from app.workflow.routing import route_by_intent

    assert route_by_intent({"intent": "订单", "knowledge_query": query}) == "business"
    assert route_by_intent({"intent": "商品咨询", "knowledge_query": query}) == "knowledge"


@pytest.mark.parametrize("raw", ["金额1001", "1001元退款", "1001是物流单号", "2026年买的", "价格￥1001.50",
                                 "1001是我的快递单号", "1001（我的快递单号）"])
def test_non_order_numbers_cannot_become_current_or_historical_order(raw):
    """用途排除同时保护本轮查询和下轮可信对象。

    参数：raw含金额、日期或物流用途的数字；返回：None；异常：断言失败。
    """
    from app.workflow.routing import order_candidate, _trusted_history
    orders = [{"order_id": "1001"}, {"order_id": "2026"}]
    assert order_candidate({"raw_query": raw, "order_id": None}, orders) is None
    assert _trusted_history([{"messages": [{"role": "user", "content": raw}]}])[1] == set()


@pytest.mark.parametrize("identifier", ["SO-4801", "ABC90022"])
def test_order_identifier_can_be_omitted_from_search_without_losing_target(identifier):
    """检索问法省略已单独保存的订单号不能清空可信目标。参数：identifier完整订单号；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding
    raw = f"查订单{identifier}的状态和金额。"
    result = validate_understanding(raw, _understanding(resolved_query=raw,
        standard_query="查询订单状态与金额", order_id=identifier), [])
    assert result.order_id == identifier and result.resolved_query == raw
    assert result.standard_query == "查询订单状态与金额"


def test_excluded_prefixed_order_does_not_become_required_product_model():
    """被明确排除的订单号不能强制出现在新订单检索中。参数：无；返回：None；异常：断言失败。"""
    from app.workflow.routing import validate_understanding
    raw = "不要查订单SO-4801了，只查订单9205的物流。"
    result = validate_understanding(raw, _understanding(resolved_query=raw,
        standard_query="查询订单9205物流", order_id="9205"), [])
    assert result.order_id == "9205" and result.standard_query == "查询订单9205物流"
