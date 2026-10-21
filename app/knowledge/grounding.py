"""核验抽取摘要、完整原句引用与来源指纹，生成可信的持久审计证据。

本模块只做纯校验，不调用模型。允许保序组合完整原句；句内空白和数字原样
保留，业务条件与否定句不得遗漏。模型引用与代码生成的审计字段分离。
"""

from __future__ import annotations

import hashlib
import json
import re

from app.knowledge.schemas import AnswerExcerpt, CompletedTurn, ExtractedQA, KnowledgeUnavailable

GROUNDING_VERSION = "ch03-grounding-v1"
_SENTENCE = re.compile(r"[^。！？!?\n]+(?:[。！？!?]+[”’\"）)]*|(?=\n|$))|[。！？!?]+[”’\"）)]*")
# 语义关键词不能证明条件完整：只准删明确寒暄，其余完整业务句都须保留。
_GREETING = re.compile(r"(?:您好|你好|欢迎咨询|谢谢|感谢您的咨询|祝您生活愉快)[。！？!?]*")


class GroundingError(KnowledgeUnavailable):
    """摘要缺少可信原句或漏掉关键业务条件。"""


def turns_digest(turns: list[CompletedTurn]) -> str:
    """计算完整轮次输入指纹。

    参数：turns 为按消息ID排序的完整轮次，不修改句内字符。
    返回：str，兼容旧批次算法的SHA256。异常：序列化错误。
    """
    payload = [turn.model_dump(mode="json") for turn in turns]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _sentences(text: str) -> list[tuple[str, int, int]]:
    """按句号、问叹号和换行获取原句及偏移。

    参数：text 为原文，句间空白可忽略，句内空白保留。
    返回：list，元素为原句/起点/终点。异常：GroundingError，非空原文未完整覆盖。
    """
    result = []
    for match in _SENTENCE.finditer(text):
        raw = match.group()
        quote = raw.strip()
        if quote:
            start = match.start() + len(raw) - len(raw.lstrip())
            result.append((quote, start, start + len(quote)))
    # 句间空白可变；任何未覆盖非空字符都失败关闭，不能静默漏掉业务行。
    covered = {i for _, start, end in result for i in range(start, end)}
    if any(not char.isspace() and i not in covered for i, char in enumerate(text)):
        raise GroundingError("原句切分未覆盖完整原文")
    return result


def validate_grounding(
    qa: ExtractedQA, turns: list[CompletedTurn], *, require_excerpts: bool = False
) -> tuple[ExtractedQA, list[dict]]:
    """纯校验摘要并生成代码回核证据。

    参数：qa 为候选（实例也重新校验）；turns 为持久来源完整轮次；
        require_excerpts=True要求模型显式引用，False仅用于旧记录补证。
    返回：tuple，规范候选与包含message_id/quote/offset/hash/版本的审计列表。
    异常：GroundingError，未知来源、问题改写、伪原句、乱序或关键条件遗漏；
        pydantic.ValidationError，字段非法。
    """
    payload = qa.model_dump(mode="json")
    # 按抽取契约投影并重验，避免依赖StagedQA字段名或model_copy绕过字段约束。
    qa = ExtractedQA.model_validate({key: payload[key] for key in ExtractedQA.model_fields})
    sources = {turn.message_id: turn for turn in turns}
    if len(set(qa.source_ids)) != len(qa.source_ids) or not set(qa.source_ids).issubset(sources):
        raise GroundingError("抽取包含未知或重复来源")
    selected = [sources[i] for i in sorted(qa.source_ids)]
    if qa.question not in {turn.question for turn in selected}:
        raise GroundingError("抽取问题不是来源用户原话")
    if len({turn.conversation_id for turn in selected}) != 1:
        raise GroundingError("抽取结果跨越会话边界")
    available = [(turn, quote, start, end) for turn in selected
                 for quote, start, end in _sentences(turn.answer)]
    excerpts = qa.answer_excerpts
    if not excerpts:
        if require_excerpts:
            raise GroundingError("抽取缺少原句证据")
        # 旧候选仅从原文精确定位补证，不能把无证据状态当合法。
        excerpts = []
        cursor = -1
        for quote, _, _ in _sentences(qa.answer):
            found = next((i for i, (_, original, _, _) in enumerate(available)
                          if i > cursor and original == quote), None)
            if found is None:
                raise GroundingError("答案不是来源完整原句")
            cursor = found
            excerpts.append(AnswerExcerpt(message_id=available[found][0].message_id, quote=quote))
    if [quote for quote, _, _ in _sentences(qa.answer)] != [ref.quote for ref in excerpts]:
        raise GroundingError("答案与原句证据不一致")
    audit = []
    selected_indices = set()
    cursor = -1
    for ref in excerpts:
        found = next((i for i, (turn, quote, _, _) in enumerate(available)
                      if i > cursor and turn.message_id == ref.message_id and quote == ref.quote), None)
        if found is None:
            raise GroundingError("证据不是保序完整原句")
        cursor = found
        selected_indices.add(found)
        turn, quote, start, end = available[found]
        audit.append({"kind": "source", "message_id": turn.message_id, "quote": quote,
                      "start": start, "end": end,
                      "answer_hash": hashlib.sha256(turn.answer.encode("utf-8")).hexdigest(),
                      "quote_hash": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
                      "verification_version": GROUNDING_VERSION})
    if {ref.message_id for ref in excerpts} != set(qa.source_ids):
        raise GroundingError("来源没有对应原句证据")
    if any(not _GREETING.fullmatch(quote) and i not in selected_indices
           for i, (_, quote, _, _) in enumerate(available)):
        raise GroundingError("摘要遗漏关键业务条件原句")
    return qa.model_copy(update={"answer_excerpts": excerpts}), audit
