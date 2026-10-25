"""工作流纯规则：原话透传、七意图四出口、输入上限、证据闸与固定回复。"""

import math
from typing import Literal

from app.knowledge.answering import EvidenceCitation, FIXED_KNOWLEDGE_REFUSAL
from app.knowledge.rerank import RankedChunk
from app.schemas.workflow import GateDecision, IntentResult, RetrievalLowConfidence
from app.workflow.state import ConversationState

_INTENT_ROUTES = {
    "物流": "business", "订单": "business", "售后": "business",
    "商品咨询": "knowledge", "退款退货": "knowledge",
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
    """将State内本轮意图映射到固定出口，可直接作为条件边回调。

    参数：state为包含intent的工作流字典，不接受直接字符串或模型指定节点名。
    返回：str，knowledge/business/complaint/chitchat之一。
    异常：ValueError（包含Pydantic校验错误），State形态、缺失意图或未知枚举非法。
    """
    if not isinstance(state, dict):
        raise ValueError("意图路由必须接收State字典")
    # 不去空白、不猜分类；分类失败不能悄悄变成闲聊或知识兜底。
    intent = IntentResult.model_validate({"intent": state.get("intent")}).intent
    return _INTENT_ROUTES[intent]


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
