"""工作流纯规则：可信理解、八意图五出口、输入上限、证据闸与固定回复。"""

import json
import math
import re
from typing import Literal

from app.knowledge.answering import EvidenceCitation, FIXED_KNOWLEDGE_REFUSAL
from app.knowledge.rerank import RankedChunk
from app.knowledge.query_rewrite import QueryRewrite, QueryRewriteError, validate_rewrite
from app.schemas.workflow import GateDecision, IntentResult, RetrievalLowConfidence, UnderstandingResult
from app.workflow.state import ConversationState

_INTENT_ROUTES = {
    "物流": "business", "订单": "business", "售后": "order_policy",
    "商品咨询": "knowledge", "退款退货": "order_policy", "其他": "business",
    "投诉": "complaint", "闲聊": "chitchat",
}
_FIXED_REPLIES = {
    "chitchat": "您好，我是小猫客服，可以帮您查询商品、订单、物流和售后问题。",
    "complaint": "抱歉给您带来不好的体验。您可以选择转人工或创建工单，也可以继续告诉我具体情况。",
    "fallback": FIXED_KNOWLEDGE_REFUSAL,
    "input_limit": "问题超过知识检索的 2000 字上限，请拆分后再提问。",
}


def resolve_reference(query: str) -> str:
    """原样透传当前问题，不改写指代。参数：query为入口校验后的完整原话；返回：相同字符串；异常：无。"""
    return query


def route_by_intent(state: ConversationState) -> str:
    """只根据八类意图映射固定出口，忽略旧检查点的派生问法。

    参数：state包含intent的工作流字典，不能由模型指定节点。
    返回：knowledge/business/complaint/chitchat/order_policy之一。
    异常：ValueError，State形态、缺失意图或未知枚举非法。
    """
    if not isinstance(state, dict):
        raise ValueError("意图路由必须接收State字典")
    result = IntentResult.model_validate({"intent": state.get("intent"), "confidence": 0.0})
    return _INTENT_ROUTES[result.intent]


# 完整字母前缀与数字视作一个候选，避免A-1001被截成1001；用途/归属仍由fetch_order确定。
_ORDER_CANDIDATE = re.compile(r"(?<![A-Za-z0-9_-])(?:[A-Za-z]+(?:-[A-Za-z]+)*-?)?\d{4,}[A-Za-z0-9-]*(?![A-Za-z0-9_-])")


def original_understanding(raw_query: str) -> UnderstandingResult:
    """构造不修改原字符的理解回退。参数：raw_query合法1–4000字原话；返回：空订单的透传结果；异常：字段校验错误。"""
    return UnderstandingResult(resolved_query=raw_query, standard_query=raw_query,
                               needs_resolution=False, order_id=None)


def _trusted_history(history_turns: list[dict]) -> tuple[str, set[str], set[str]]:
    """提取可信历史、最近一轮订单候选与已知商品名，助手正文不作事实来源。

    参数：history_turns服务端读取的整轮历史；仅用户原话及成功工具data参与事实校验。
    返回：历史文本、最近含订单的一轮完整候选集合、成功工具给出的商品名称集合。
    异常：无；畸形或失败工具结果不提升为可信事实。
    """
    parts, latest_orders, products = [], set(), set()
    for turn in history_turns:
        orders = set()
        for message in turn.get("messages", []):
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if message.get("role") == "user":
                parts.append(content)
                orders.update(_ORDER_CANDIDATE.findall(content))
            elif message.get("role") == "tool":
                try:
                    result = json.loads(content)
                except (ValueError, TypeError):
                    continue
                if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("data"), dict):
                    continue
                data = result["data"]
                parts.append(json.dumps(data, ensure_ascii=False))
                # 物流号也可能很长，历史工具只把明确order_id字段作为订单候选。
                if isinstance(data.get("order_id"), str):
                    orders.add(data["order_id"])
                if isinstance(data.get("product_name"), str) and data["product_name"]:
                    products.add(data["product_name"])
        selection = turn.get("order_selection")
        if isinstance(selection, dict) and selection.get("status") == "accepted":
            selected = selection.get("accepted_order_id")
            if isinstance(selected, str) and selected:
                parts.append(selected)
                orders.add(selected)
        if orders:
            latest_orders = orders
    return "\n".join(parts), latest_orders, products


def validate_understanding(raw_query: str, result: UnderstandingResult,
                           history_turns: list[dict]) -> UnderstandingResult:
    """核对一次理解的来源，非法组合整体回退，不进行关键词意图分类或额外模型修复。

    参数：raw_query完整用户原话；result严格模型结果；history_turns当前会话的可信整轮历史。
    返回：可信UnderstandingResult；无需消解时完整问强制原样，来源失真则四字段整体透传。
    异常：原问不满足UnderstandingResult基本契约时抛校验错误；语义是否需消解由Prompt及真实评估验证。
    """
    return _understanding_outcome(raw_query, result, history_turns)[0]


def _understanding_outcome(raw_query: str, result: UnderstandingResult,
                           history_turns: list[dict]) -> tuple[UnderstandingResult, bool]:
    """复用唯一来源校验并标记真实回退，区别正常空白恢复和来源失真。

    参数：raw_query原话、result严格候选、history_turns可信历史。
    返回：理解结果和是否因来源失真回退的布尔值，供模型边界准确记录失败。
    异常：原问契约非法时抛字段校验错误。
    """
    fallback = original_understanding(raw_query)
    history, recent_orders, products = _trusted_history(history_turns)
    current_orders = set(_ORDER_CANDIDATE.findall(raw_query))
    allowed_orders = current_orders or (recent_orders if result.needs_resolution else set())
    source = history if result.needs_resolution else ""
    try:
        for candidate in (result.resolved_query, result.standard_query):
            # UnderstandingResult已做严格类型与4000字校验；这里只复用事实护栏，
            # 不借QueryRewrite的2000字检索限制裁短合法原话，检索节点稍后独立停止。
            rewrite = QueryRewrite.model_construct(standard_query=candidate, synonyms=[])
            validate_rewrite(raw_query, rewrite, context=source)
            candidate_orders = set(_ORDER_CANDIDATE.findall(candidate))
            if (current_orders or result.needs_resolution) and not candidate_orders.issubset(allowed_orders):
                # 同时约束完整问和标准问，不能靠order_id=null绕过最近对象保护。
                raise QueryRewriteError("理解引入了当前或最近明确对象之外的旧编号")
            if not result.needs_resolution and any(name in candidate and name not in raw_query for name in products):
                raise QueryRewriteError("无需消解的问题不能绑定历史商品")
        if result.order_id is not None:
            if len(allowed_orders) != 1 or result.order_id not in allowed_orders:
                raise QueryRewriteError("订单候选没有唯一可信来源")
            if result.needs_resolution and result.order_id not in set(_ORDER_CANDIDATE.findall(result.resolved_query)):
                raise QueryRewriteError("具体订单指代必须在完整问题中保留全号")
        # 无订单字段也不能在完整问里悄悄替用户从并列对象中选一个。
        if result.needs_resolution and len(allowed_orders) > 1:
            for candidate in (result.resolved_query, result.standard_query):
                selected = set(_ORDER_CANDIDATE.findall(candidate)) & allowed_orders
                if selected and selected != allowed_orders:
                    raise QueryRewriteError("多个订单对象不能任意选择")
    except QueryRewriteError:
        return fallback, True
    normalized = result if result.needs_resolution else result.model_copy(update={"resolved_query": raw_query})
    return normalized, False


def knowledge_input_stop_reason(query: str) -> Literal["input_limit"] | None:
    """在知识检索前检查原话长度。参数：query为入口校验后的1–4000字原话；返回：超2000字为input_limit，否则None；异常：无。"""
    return "input_limit" if len(query) > 2000 else None


def _validate_score(value: float) -> None:
    """拒绝非归一化或非有限分数。参数：value为0–1有限数值，不接受布尔/字符串；返回：None；异常：ValueError，坏分数属于故障。"""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("重排分数和门槛必须为0–1的有限数值")


def assess_evidence(scores: list[float], threshold: float) -> GateDecision:
    """仅对成功检索的全部分数执行最高分证据闸。

    参数：scores为完整归一化重排分列表，可为空；threshold为已有rerank_min_score的0–1有限门槛。
    返回：GateDecision，空结果/低分携带retrieval_low_conf草案，通过则不创建草案。
    异常：ValueError，任一分数或门槛非法；调用方必须保留检索异常，不能转成空列表传入本函数。
    """
    _validate_score(threshold)
    for score in scores:
        _validate_score(score)
    # 必须先检查每一分数，再取最高分；首尾排列后最高分不保证在首项。
    max_score = max(scores, default=None)
    passed = max_score is not None and max_score >= threshold
    low_confidence = None
    if not passed:
        reason = ("检索成功但未找到可用证据" if max_score is None
                  else f"最高重排相关度{max_score:g}低于门槛{threshold:g}")
        low_confidence = RetrievalLowConfidence(reason=reason)
    return GateDecision(passed=passed, max_score=max_score, threshold=threshold, low_confidence=low_confidence)


def citations_from_ranked(ranked: list[RankedChunk]) -> list[EvidenceCitation]:
    """直接从权威重排快照构造完整引用，不伪造模型工具调用。

    参数：ranked为检索器返回的原序完整块；普通/退款数量由检索器控制，本函数不截断。
    返回：list[EvidenceCitation]，保持输入顺序和稳定编号；显示/模型输入由调用方沿用edge_order。
    异常：Pydantic校验错误，权威块内容不符合引用契约。
    """
    return [EvidenceCitation(n=number, chunk_id=item.chunk.id, section_path=item.chunk.section_path,
                             question=item.chunk.text.questions, answer=item.chunk.text.answer,
                             source_key=item.chunk.source_key)
            # content_hash是知识三格摘要，不是原文文件版本，不能冒充source_revision。
            for number, item in enumerate(ranked, start=1)]


def fixed_reply(kind: Literal["chitchat", "complaint", "fallback", "input_limit"]) -> ConversationState:
    """生成固定出口的最小State更新，不执行转人工或建单。

    参数：kind为闲聊、投诉、弱证据兜底或知识输入超限出口。
    返回：ConversationState部分更新，只含正文和独占建议类型列表；仅投诉建议两项。
    异常：ValueError，出口非法；本函数不创建低证据草案或稳定动作快照。
    """
    if kind not in _FIXED_REPLIES:
        raise ValueError("该出口没有固定回复")
    return {"answer": _FIXED_REPLIES[kind],
            "suggested_kinds": ["handoff", "create_ticket"] if kind == "complaint" else []}
