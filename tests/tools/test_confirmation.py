"""可信建单意图与描述的行为保护，否定、引用、假设和旧诉求不能变成新写入许可。"""

import importlib

import pytest

from app.schemas.tools import ToolContext


def context(text: str, turn: str = "turn-1") -> ToolContext:
    """构造仅含本轮真实用户原话的服务端测试上下文。

    参数：text为用户原话，turn为原轮身份；返回：ToolContext；异常：字段校验失败。
    """
    return ToolContext(conversation_id=1, session_id="confirmation-test", user_id="demo-user",
                       turn_id=turn, user_message=text)


@pytest.mark.parametrize("text", [
    "不要建工单", "订单1001坏了，但别给我建工单", "如果坏了再建工单",
    "客服说‘帮我建个工单’是什么意思", "有人让我说\"帮我建个工单\"，这是干什么",
    "建工单有什么用", "转人工", "我想投诉一下", "先不用建工单",
    "建工单要收费吗", "建工单需要什么资料", "建工单前要准备什么",
])
def test_denial_quote_hypothesis_or_complaint_does_not_authorize(text):
    """把否定、他人引用或仅投诉当作建单申请会失败。

    参数：text为独立反例；返回：None；异常：断言或缺模块。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context(text), "user-message-1")
    assert intent is None or intent.status in {"closed", "cancelled"}


def test_missing_description_collects_only_current_user_source():
    """无实际问题的建单命令不能成为可提交描述。

    参数：无；返回：None；异常：断言或缺模块。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context("帮我建个工单"), "user-message-1")
    assert intent.status == "collecting"
    assert intent.source_turn_id == "turn-1" and intent.source_message_ids == ["user-message-1"]
    assert module.ticket_description(intent) is None


def test_explicit_request_preserves_user_problem_without_command():
    """工单描述必须来自实质原话，不能把建单命令或模型补充混进去。

    参数：无；返回：None；异常：断言或缺模块。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context("帮我建个工单，订单1001收到时破损"), "user-message-1")
    assert module.ticket_description(intent) == "订单1001收到时破损"


@pytest.mark.parametrize("problem", [
    "之前买的订单1001现在坏了", "上次买的商品现在无法开机", "以前购买的商品出现裂口",
    "曾经购买的商品现在漏水", "之前反馈过订单1001破损，现在还没解决",
])
def test_current_request_can_include_historical_problem_facts(problem):
    """历史词修饰购买或问题事实时不能否定当前明确申请。

    参数：problem 历史业务事实与当前问题；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context(problem + "，帮我建个工单"), "user-message-1")
    assert intent is not None and intent.status == "collecting"
    assert module.ticket_description(intent) == problem


@pytest.mark.parametrize("problem", ["订单1001取消后退款未到账", "设备停止工作"])
def test_current_request_can_include_business_cancellation_or_stoppage(problem):
    """把业务取消或设备停止误判为撤销建单时，本轮实际问题收集会失败。

    参数：problem 含取消或停止的业务原话；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    text = problem + "，帮我建个工单"
    intent = module.advance_ticket_intent(None, context(text), "message-1")
    assert intent is not None and intent.status == "collecting"
    assert intent.user_texts == [text]
    assert module.ticket_description(intent) == problem


@pytest.mark.parametrize("problem", ["订单1001取消后退款未到账", "设备停止工作"])
def test_collecting_accepts_business_cancellation_or_stoppage_followup(problem):
    """把待补问题里的业务取消或设备停止误判为撤销时，相邻原话收集会失败。

    参数：problem 紧接建单申请补充的业务事实；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    prior = module.advance_ticket_intent(None, context("帮我建个工单"), "message-1")
    current = module.advance_ticket_intent(prior, context(problem, "turn-2"), "message-2")
    assert current.status == "collecting"
    assert current.source_message_ids == ["message-1", "message-2"]
    assert current.user_texts == ["帮我建个工单", problem]
    assert module.ticket_description(current) == problem


@pytest.mark.parametrize("text", [
    "上次已经建了工单", "之前我说过，帮我建个工单", "我以前提过，帮我建个工单",
    "之前客服说，帮我建个工单", "之前买的订单坏了，但不要建工单",
    "如果之前买的东西坏了，帮我建个工单", "之前我说过‘帮我建个工单’",
    "之前已经说过，帮我建个工单", "我之前就说了，帮我建个工单",
])
def test_historical_permission_or_noncurrent_request_still_cannot_authorize(text):
    """放行历史事实不能连带放行历史同意、他人指令、否定或假设。

    参数：text 无当前明确授权的原话；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context(text), "user-message-1")
    assert intent is None or intent.status in {"closed", "cancelled"}


@pytest.mark.parametrize("problem", ["订单1001收到时包装破损，猫粮撒漏了。", "商品有裂口！", "商品无法开机？"])
def test_description_preserves_problem_sentence_punctuation(problem):
    """完整问题保留用户句末标点，避免模型照抄原话却被拒绝。

    参数：problem 为包含真实问题及标点的用户原文；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    direct = module.advance_ticket_intent(None, context("帮我建个工单，" + problem), "message-1")
    assert module.ticket_description(direct) == problem
    prior = module.advance_ticket_intent(None, context("帮我建个工单。"), "message-1")
    followup = module.advance_ticket_intent(prior, context(problem, "turn-2"), "message-2")
    assert module.ticket_description(followup) == problem


def test_next_user_turn_can_complete_collecting_intent():
    """尚未结束的待补流程应承接紧接的用户事实并保留两条来源身份。

    参数：无；返回：None；异常：断言或缺模块。
    """
    module = importlib.import_module("app.tools.confirmation")
    prior = module.advance_ticket_intent(None, context("帮我建个工单"), "user-message-1")
    current = module.advance_ticket_intent(prior, context("订单1001收到时破损", "turn-2"), "user-message-2")
    assert module.ticket_description(current) == "订单1001收到时破损"
    assert current.source_message_ids == ["user-message-1", "user-message-2"]


@pytest.mark.parametrize("text", ["算了，不建了", "先查订单1002在不在保", "查一下订单1001的物流",
                                 "取消工单", "算了", "停止建单", "不要建工单", "取消", "停止"])
def test_cancel_or_independent_query_closes_previous_collecting(text):
    """取消或开始独立查询之后不能继续沿用旧建单许可。

    参数：text为后续取消或新查询；返回：None；异常：断言或缺模块。
    """
    module = importlib.import_module("app.tools.confirmation")
    prior = module.advance_ticket_intent(None, context("帮我建个工单"), "user-message-1")
    current = module.advance_ticket_intent(prior, context(text, "turn-2"), "user-message-2")
    assert current is None or current.status in {"closed", "cancelled"}


@pytest.mark.parametrize("withdrawal", ["取消工单", "算了", "停止建单", "不要建工单",
                                       "取消工单，先查订单1002的物流"])
def test_withdrawal_ends_permission_before_later_business_facts(withdrawal):
    """收窄业务取消误拒不能让明确撤销后的新问题重新获得旧建单许可。

    参数：withdrawal 用户明确撤销当前建单的原话；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    prior = module.advance_ticket_intent(None, context("帮我建个工单"), "message-1")
    withdrawn = module.advance_ticket_intent(prior, context(withdrawal, "turn-2"), "message-2")
    current = module.advance_ticket_intent(withdrawn, context("订单1001取消后退款未到账", "turn-3"), "message-3")
    assert current.status == "cancelled"
    assert current.source_message_ids == ["message-1"]
    assert module.ticket_description(current) is None


@pytest.mark.parametrize("withdrawal", [
    "取消吧，问题已经解决了", "请取消，退款已到账", "先取消吧，订单问题已解决",
    "请停止，退款我自己解决",
])
def test_polite_withdrawal_with_business_facts_cannot_keep_ticket_permission(withdrawal):
    """漏掉带礼貌词或语气词的撤销时，同条及后续业务事实会沿用旧建单许可。

    参数：withdrawal 明确撤销且附带业务事实的用户原话；返回：None；异常：断言失败。
    """
    from app.tools.confirmation import advance_ticket_intent, ticket_description, ticket_write_rejection
    prior = advance_ticket_intent(None, context("帮我建个工单"), "message-1")
    withdrawn = advance_ticket_intent(prior, context(withdrawal, "turn-2"), "message-2")
    assert withdrawn.status == "cancelled"
    assert ticket_description(withdrawn) is None
    followup_context = context("订单1001取消后退款未到账", "turn-3")
    current = advance_ticket_intent(withdrawn, followup_context, "message-3")
    assert current.status == "cancelled" and current.source_message_ids == ["message-1"]
    rejected = ticket_write_rejection(current, {"description": "订单1001取消后退款未到账"}, followup_context, None)
    assert rejected is not None and rejected.error.code == "NOT_CONFIRMED"


def test_description_over_limit_is_not_silently_truncated():
    """过长问题不能悄悄截断为另一份可提交描述。

    参数：无；返回：None；异常：断言或缺模块。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context("帮我建个工单，" + "问题" * 1001), "user-message-1")
    assert module.ticket_description(intent) is None


@pytest.mark.parametrize("text", ["帮我建个工单，谢谢", "请创建一个工单，订单1001", "帮我提交工单，好的",
                                 "帮我建个工单，请尽快处理", "帮我建个工单，订单1001，麻烦解决一下"])
def test_command_acknowledgement_or_order_number_is_not_a_problem(text):
    """命令附带礼貌话或仅订单号仍须补充实际问题。

    参数：text为缺少问题的申请；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context(text), "user-message-1")
    assert intent.status == "collecting"
    assert module.ticket_description(intent) is None


@pytest.mark.parametrize("text", ["你能帮我建工单吗", "我上次说帮我建工单", "假如故障了，请提交工单", "他让我创建工单"])
def test_ambiguous_historical_or_third_party_requests_are_not_current_authority(text):
    """把能力询问、过去事件或他人指令误当本轮申请会失败。

    参数：text为非当前直接申请；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context(text), "user-message-1")
    assert intent is None or intent.status in {"closed", "cancelled"}


def test_replayed_message_id_does_not_duplicate_or_replace_source():
    """同一来源消息重放不能新增描述或通过改文替换已保存的原话。

    参数：无；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    prior = module.advance_ticket_intent(None, context("帮我建个工单，订单1001收到时破损"), "message-1")
    current = module.advance_ticket_intent(prior, context("客服态度很差"), "message-1")
    assert current.source_message_ids == ["message-1"]
    assert module.ticket_description(current) == "订单1001收到时破损"


def test_acknowledgement_ends_continuation_and_later_facts_do_not_revive_it():
    """待补流程被无关致谢打断后，后续问题不能复用旧的申请。

    参数：无；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    prior = module.advance_ticket_intent(None, context("帮我建个工单"), "message-1")
    interrupted = module.advance_ticket_intent(prior, context("谢谢", "turn-2"), "message-2")
    current = module.advance_ticket_intent(interrupted, context("订单1001坏了", "turn-3"), "message-3")
    assert current is None or current.status == "closed"


def test_followup_order_number_stays_collecting_until_problem_is_supplied():
    """紧接补充订单号可继续收集，但不能单独形成可提交描述。

    参数：无；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    prior = module.advance_ticket_intent(None, context("帮我建个工单"), "message-1")
    number = module.advance_ticket_intent(prior, context("订单号是1001", "turn-2"), "message-2")
    assert number.status == "collecting" and module.ticket_description(number) is None
    current = module.advance_ticket_intent(number, context("收到时外壳破损", "turn-3"), "message-3")
    assert module.ticket_description(current) == "订单号是1001\n收到时外壳破损"


def test_intent_state_is_bounded_and_overflow_cannot_be_submitted():
    """超长原话只保留有界来源片段，并且不能冒充完整工单描述。

    参数：无；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context("帮我建个工单，" + "破损" * 50000), "message-1")
    assert sum(map(len, intent.user_texts)) <= 8192
    assert module.ticket_description(intent) is None
    for number in range(2, 15):
        intent = module.advance_ticket_intent(intent, context("补充：还有划痕", f"turn-{number}"), f"message-{number}")
    assert len(intent.source_message_ids) <= 8 and len(intent.user_texts) <= 8
    assert sum(map(len, intent.user_texts)) <= 8192
    assert module.ticket_description(intent) is None


def test_context_notes_offer_trusted_description_without_declaring_confirmation():
    """模型提示只提供服务端原话描述和待确认状态，不能声称已获用户确认。

    参数：无；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    intent = module.advance_ticket_intent(None, context("帮我建个工单，订单1001收到时破损"), "message-1")
    notes = module.ticket_context_notes(intent)
    assert isinstance(notes, list) and all(isinstance(note, str) for note in notes)
    assert "订单1001收到时破损" in "\n".join(notes)
    assert "待确认" in "\n".join(notes)
    assert "expected" not in "\n".join(notes)


def test_context_notes_request_problem_for_missing_or_overlong_description():
    """缺少问题与超长描述的提示均要求补充整理，且不将截断问题交模型填写。

    参数：无；返回：None；异常：断言失败。
    """
    module = importlib.import_module("app.tools.confirmation")
    missing = module.advance_ticket_intent(None, context("帮我建个工单"), "message-1")
    long = module.advance_ticket_intent(None, context("帮我建个工单，" + "坏了" * 1100), "message-2")
    assert "实际问题" in "\n".join(module.ticket_context_notes(missing))
    assert "2000" in "\n".join(module.ticket_context_notes(long))
    assert "坏了" * 100 not in "\n".join(module.ticket_context_notes(long))
    assert module.ticket_context_notes(None)
