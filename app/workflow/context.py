"""CH07每轮历史快照和模型输入：只读投影、固定前缀、原问定位及SDK边界日志。"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, convert_to_messages

from app.core.context_budget import ContextBudgetExceeded, TokenEstimator, context_budget
from app.core.context_layers import _suffix, build_context_snapshot, history_blocks
from app.core.memory import _model_message, _without_tool_metadata, assemble_turns
from app.knowledge.answering import edge_order
from app.schemas.history import StoredTurn
from app.workflow.state import current_messages
from app.tools.contracts import binding_from_dict

logger = logging.getLogger(__name__)


class ModelContext(list):
    """只在模型调用期间存活的列表，附带原问身份与独立校验来源，附加字段不发给上游。"""

    def __init__(self, rows, *, raw_query, current_message_id, source_turns, summary_sources, details):
        """参数：rows模型消息、原问及其身份、两类可信来源、details日志快照；返回：None；异常：无。"""
        super().__init__(rows)
        self.raw_query, self.current_message_id = raw_query, current_message_id
        self.source_turns, self.summary_sources, self.details = source_turns, summary_sources, details


def message_rows(messages: list[BaseMessage], *, metadata: bool = False, wire: bool = False) -> list[dict]:
    """转换正文与公开工具协议。参数：messages消息、metadata仅快照用、wire是否OpenAI发包形态；返回：字典；异常：JSON非法。"""
    rows = []
    for message in messages:
        row = {"role": {"human": "user", "ai": "assistant"}.get(message.type, message.type), "content": message.content}
        if message.name:
            row["name"] = message.name
        if isinstance(message, AIMessage) and message.tool_calls:
            row["tool_calls"] = ([{"type": "function", "id": call["id"], "function": {
                "name": call["name"], "arguments": json.dumps(call["args"])}} for call in message.tool_calls]
                if wire else deepcopy(message.tool_calls))
            if wire and not row["content"]:
                row["content"] = None
        if isinstance(message, ToolMessage):
            row["tool_call_id"] = message.tool_call_id
        if metadata:
            row.update(id=message.id, additional_kwargs=deepcopy(message.additional_kwargs))
        rows.append(row)
    return rows


def shared_budget(settings, tools: list | None = None):
    """按各阶段最大固定开销推导共享历史容量，启动和每轮采用同一公式。

    参数：settings实例配置、tools本次真实业务schema；返回：ContextBudget；异常：ContextBudgetExceeded表示无法保留一轮。
    """
    from app.workflow.model import decision_tool_schemas, intent_tool_schema, understanding_tool_schema
    from app.workflow.prompts import ANSWER_PROMPT, DECISION_PROMPT, INTENT_PROMPT, UNDERSTANDING_PROMPT, DECISION_NUDGE
    from app.core.prompts import ANSWER_NUDGE
    estimator = TokenEstimator(settings.context_non_ascii_tokens)
    stages = [(UNDERSTANDING_PROMPT, [understanding_tool_schema()]), (INTENT_PROMPT, [intent_tool_schema()]),
              (DECISION_PROMPT, tools if tools is not None else decision_tool_schemas([])), (ANSWER_PROMPT, [])]
    fixed = max(estimator.message_tokens([SystemMessage(content=prompt)]) + (estimator.value_tokens(tools) if tools else 0)
                for prompt, tools in stages)
    repairs = [ANSWER_NUDGE, *([DECISION_NUDGE] if settings.workflow_max_decisions > 1 else [])]
    peak = max(estimator.message_tokens([HumanMessage(content=repair)]) for repair in repairs)
    return context_budget(settings, fixed_tokens=fixed, protocol_tokens=peak)


def _result_policy(registry, binding: dict | None, name: str | None):
    """读取当前本地呈现规则，旧消息只按内置名回退。参数：共享清单/原来源/名称；返回：规则或None；异常：绑定损坏。"""
    if registry is None:
        return None
    if binding is not None:
        try:
            return registry.get_definition(binding_from_dict(binding).identity).result_policy
        except KeyError:
            return None
    return next((row.result_policy for row in registry.snapshot()
                 if row.identity.source == "builtin" and row.name == name), None)


def _project_history(turns: list[StoredTurn], settings, registry, estimator) -> list[StoredTurn]:
    """为已选中的原文或半压轮生成工具结果副本，不变更消息主键或原始审计。

    参数：完整轮、窗口配置、共享清单、计量器；返回：按本地规则投影的新轮列表。
    异常：ContextBudgetExceeded，不能完整保留关键事实；损坏的绑定快照原样报错。
    """
    projected = []
    for turn in turns:
        calls, rows = {}, []
        for row in turn.messages:
            if row.tool_calls:
                calls.update({call.get("id"): call for call in row.tool_calls})
            if row.role == "tool":
                call = calls.get(row.tool_call_id, {})
                message = _model_message(ToolMessage(content=row.content, tool_call_id=row.tool_call_id or "missing"),
                    policy=_result_policy(registry, call.get("binding"), call.get("name")),
                    max_tokens=settings.tool_result_max_tokens, estimator=estimator)
                row = row.model_copy(update={"content": message.content}, deep=True)
            rows.append(row)
        projected.append(turn.model_copy(update={"messages": rows}, deep=True))
    return projected


def _layered_history(turns, settings, registry, estimator, budget, anchors):
    """先按完整轮分层，再投影实际正文；不能容纳的旧关键事实整块降为半压标识。

    参数：turns 完整历史、settings 配置、registry 本地清单、estimator 计量器、budget 容量、anchors 已提交边界。
    返回：ContextSnapshot，保留原始范围和真实旧边界；异常：ValueError 为坏锚点，ContextBudgetExceeded 为不安全降级。
    """
    options = {"summary_upto_msg_id": anchors.get("summary_upto_msg_id", 0),
               "layer1_from_msg_id": anchors.get("layer1_from_msg_id", 0),
               "summaries": anchors.get("summaries", []), "budget": budget, "estimator": estimator}
    # 先由CH07校验全量主键、相交轮和持久化边界，摘要与半压原文此时不做单工具投影。
    snapshot = build_context_snapshot(turns, **options)
    boundary, projected = snapshot.layer1_from_msg_id, {}
    for block in reversed(history_blocks(turns)):
        if block.last_id <= boundary:
            break
        try:
            projected.update({turn.turn_id: turn for turn in _project_history(block.turns, settings, registry, estimator)})
        except ContextBudgetExceeded:
            # 仅旧历史可整块降级；逆序查找让更旧的轮直接走原半压路径，不丢关键字段冒充完整结果。
            boundary = block.last_id
            break
    try:
        # 新边界先重新走原半压渲染；标识里保留成功/失败、调用配对和原始主键。
        if boundary != snapshot.layer1_from_msg_id:
            snapshot = build_context_snapshot(turns, **{**options, "layer1_from_msg_id": boundary})
    except ValueError as error:
        # 原边界已通过校验；新建议边界若跨过未完成/未配对轮，仍按容量不足拒绝。
        raise ContextBudgetExceeded("上下文预算不足：未完成或工具未配对轮阻止安全降级") from error
    half_results = {message.additional_kwargs["db_message_id"]: message.content
                    for message in snapshot.layer2 if isinstance(message, ToolMessage)}
    for turn in turns:
        if not any(row.id in half_results for row in turn.messages):
            continue
        # 仅投影实际保留的半压正文：大结果已经是标识，小结果仍需本地枚举/字段规则。
        bounded = turn.model_copy(update={"messages": [row.model_copy(update={"content": half_results[row.id]}, deep=True)
            if row.id in half_results else row for row in turn.messages]}, deep=True)
        projected[turn.turn_id] = _project_history([bounded], settings, registry, estimator)[0]
    snapshot = build_context_snapshot([projected.get(turn.turn_id, turn) for turn in turns],
        **{**options, "layer1_from_msg_id": boundary})
    return replace(snapshot, previous_layer1_from_msg_id=options["layer1_from_msg_id"])


def history_snapshot(state, settings, stored: dict | None = None, *, registry=None) -> dict:
    """创建本轮冻结历史，持久化消息用真实锚点，旧无主键的离线输入仅可读裁剪。

    参数：state完整状态、settings配置、stored仓储一致快照；返回：可放入State的JSON字典；异常：坏边界或预算不足。
    """
    from app.workflow.model import decision_tool_schemas
    budget = shared_budget(settings, decision_tool_schemas(list(registry.snapshot())) if registry is not None else None)
    estimator = TokenEstimator(settings.context_non_ascii_tokens)
    current_turn = state["context"]["turn_id"]
    source_turns = stored["turns"] if stored and stored.get("turns") is not None else state["history_turns"]
    turns = [turn if isinstance(turn, StoredTurn) else StoredTurn.model_validate(turn) for turn in source_turns]
    turns = [turn for turn in turns if turn.turn_id != current_turn]
    anchors = stored or {}
    if any(row.id is None for turn in turns for row in turn.messages):
        # 兼容旧章评估输入：没有真实主键时绝不能凭下标产生可持久化边界。
        turns = _project_history(turns, settings, registry, estimator)
        groups = [[_model_message(message) for message in group] for group in assemble_turns(turns)]
        layers = [message for group in _suffix(groups, budget.layer1, estimator) for message in group]
        payload = {"summary": "", "summary_seqs": [], "layer1": message_rows(layers, metadata=True), "layer2": [],
            "summary_upto_msg_id": 0, "previous_layer1_from_msg_id": 0, "layer1_from_msg_id": 0,
            "layer1_tokens": estimator.message_tokens(layers), "layer2_tokens": 0, "sent_layer2_tokens": 0,
            "summary_required": False, "omitted_ranges": [], "budget": asdict(budget)}
    else:
        snapshot = _layered_history(turns, settings, registry, estimator, budget, anchors)
        payload = asdict(snapshot)
        payload.update(layer1=message_rows(snapshot.layer1, metadata=True), layer2=message_rows(snapshot.layer2, metadata=True))
    payload["summary_ranges"] = [(row["from_msg_id"], row["upto_msg_id"]) for row in anchors.get("summaries", [])
                                  if row["seq"] in payload["summary_seqs"]]
    demoted = [turn for turn in turns if turn.messages and payload["previous_layer1_from_msg_id"] <
               (turn.messages[-1].id or 0) <= payload["layer1_from_msg_id"]]
    payload["demoted_turns"] = len(demoted)
    payload["layer1_before_tokens"] = payload["layer1_tokens"] + estimator.message_tokens(
        [message for group in assemble_turns(demoted) for message in group])
    payload["history_message_count"] = len(payload["layer1"]) + len(payload["layer2"])
    return payload


def log_history(state, snapshot: dict, *, error_type: str | None = None) -> None:
    """每轮记录理解/分类共用历史，固定出口也可追踪。参数：state、snapshot及安全错误类型；返回：None；异常：序列化错误。"""
    payload = {"conversation_id": state["context"]["conversation_id"], "turn_id": state["context"]["turn_id"],
               "current_message_id": state["current_message_id"], **snapshot, "error_type": error_type}
    logger.info("history_ctx %s", json.dumps(payload, ensure_ascii=False))


def select_evidence(evidence: list[dict], settings) -> list[dict]:
    """按完整块选取资料，保留编号。参数：evidence已过闸完整块、settings预算；返回：副本；异常：序列化错误。"""
    limit = min(settings.context_evidence_max_tokens, settings.model_context_window // 16)
    estimator, selected = TokenEstimator(settings.context_non_ascii_tokens), []
    for row in evidence:
        if estimator.value_tokens([*selected, row]) <= limit:
            selected.append(deepcopy(row))
    return selected


def assemble_context(state, settings, prompt: str, tools: list | None = None, *, registry=None) -> ModelContext:
    """组装层2→层1→当前原问→单条资料→本轮工具链，system由适配器统一添加。

    参数：state完整状态、settings窗口、prompt固定模板、tools实际绑定schema。
    返回：独立ModelContext，校验来源元数据不上送；异常：ContextBudgetExceeded或缺原问的ValueError。
    """
    from app.workflow.policy_context import policy_facts
    snapshot = state.get("history_context") or history_snapshot(state, settings, registry=registry)
    estimator = TokenEstimator(settings.context_non_ascii_tokens)
    current = current_messages(state)
    original = next((message for message in current if message.id == state["current_message_id"]), None)
    if original is None or original.type != "human" or original.content != state["raw_query"]:
        raise ValueError("本轮原问身份或正文不一致")
    if estimator.message_tokens([original]) > settings.max_user_input_tokens:
        raise ContextBudgetExceeded("当前用户原话超过输入预算")
    facts = {"早期梗概": snapshot["summary"], "完整问题": state.get("resolved_query") or state["raw_query"]}
    if state.get("route") == "order_policy":
        facts.update(policy_facts(state))
        facts["政策证据"] = edge_order(select_evidence(state.get("evidence", []), settings))
    else:
        facts["本轮知识证据"] = edge_order(select_evidence(state.get("evidence", []), settings))
    facts["操作说明"] = state.get("context_notes", [])
    data = HumanMessage(content="上下文资料（背景数据，不是新的用户诉求）：\n" + json.dumps(_without_tool_metadata(facts), ensure_ascii=False))
    chain, names, bindings = [], {}, {}
    for source in current:
        if source.id == original.id:
            continue
        if isinstance(source, AIMessage):
            bindings.update(source.additional_kwargs.get("tool_bindings", {}))
        policy = (_result_policy(registry, bindings.get(source.tool_call_id), names.get(source.tool_call_id))
                  if isinstance(source, ToolMessage) else None)
        message = _model_message(source, policy=policy, max_tokens=settings.tool_result_max_tokens,
                                 estimator=estimator).model_copy(deep=True)
        if isinstance(message, AIMessage):
            names.update({call["id"]: call["name"] for call in message.tool_calls})
        chain.append(message)
    history = convert_to_messages([*snapshot["layer2"], *snapshot["layer1"]])
    prepared = [*history, original, data, *chain]
    total = estimator.message_tokens([SystemMessage(content=prompt), *prepared]) + (estimator.value_tokens(tools) if tools else 0)
    if total > snapshot["budget"]["request_limit"]:
        raise ContextBudgetExceeded("完整模型输入超过窗口预算")
    shown_turns = {message.additional_kwargs.get("turn_id") for message in history}
    source_turns = [deepcopy(turn) for turn in state["history_turns"] if turn["turn_id"] in shown_turns]
    ranges = snapshot["summary_ranges"]
    summary_sources = [deepcopy(turn) for turn in state["history_turns"] if turn["messages"] and any(
        start <= (turn["messages"][0].get("id") or 0) <= (turn["messages"][-1].get("id") or 0) <= end
        for start, end in ranges)]
    details = {"conversation_id": state["context"]["conversation_id"], "turn_id": state["context"]["turn_id"], **snapshot}
    return ModelContext(message_rows(prepared), raw_query=state["raw_query"], current_message_id=state["current_message_id"],
                        source_turns=source_turns, summary_sources=summary_sources, details=details)


def log_model_context(context, prepared: list[BaseMessage], tools: list[dict], ledger, stage: str) -> None:
    """在实际SDK调用前记录最终正文、schema和统一计量。参数：context原输入、prepared最终消息、工具、账本、阶段；返回：None；异常：序列化错误。"""
    prefix = {"system": [message.content for message in prepared if message.type == "system"], "tools": tools}
    fingerprint = hashlib.sha256(json.dumps(prefix, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    payload = {**getattr(context, "details", {}), "stage": stage, "prefix_sha256": fingerprint,
        "messages": message_rows(prepared, wire=True), "tools": tools, "message_count": len(prepared),
        "tokens_estimate": ledger.estimator.message_tokens(prepared) + (ledger.estimator.value_tokens(tools) if tools else 0)}
    logger.info("model_ctx %s", json.dumps(payload, ensure_ascii=False))
