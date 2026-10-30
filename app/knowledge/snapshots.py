"""原轮检索证据收集、幂等合并和历史只读恢复，不用当前知识补造旧事实。"""

import json
import math
from hashlib import sha256

from app.schemas.flywheel import RetrievalAttempt, RetrievalChunkSnapshot, RetrievalSnapshot


def _scores_equivalent(saved: float | None, incoming: float | None) -> bool:
    """仅容纳MySQL JSON已证实的单步浮点表示差异。

    参数：saved/incoming 为已验证的可空有限分数；零不等价于非零。
    返回：bool，相同值或相邻一个可表示浮点值；异常：无。
    """
    if saved is None or incoming is None:
        return saved is incoming
    if type(saved) is not float or type(incoming) is not float or not math.isfinite(saved) or not math.isfinite(incoming):
        return False
    if saved == incoming:
        return True
    # 不使用相对/绝对容差；缺失、零和非零事实不能因值很小而混同。
    return saved != 0.0 and incoming != 0.0 and math.nextafter(saved, incoming) == incoming


def _attempts_equivalent(saved: RetrievalAttempt, incoming: RetrievalAttempt) -> bool:
    """精确比较尝试身份和原文，只有召回/重排分采用存储等价。

    参数：saved/incoming 为经过严格schema验证的尝试。
    返回：bool，全部非分数字段、块顺序及数量一致；异常：无。
    """
    if (saved.model_dump(exclude={"chunks"}) != incoming.model_dump(exclude={"chunks"})
            or len(saved.chunks) != len(incoming.chunks)):
        return False
    scores = {"recall_score", "rerank_score"}
    return all(old.model_dump(exclude=scores) == new.model_dump(exclude=scores)
        and _scores_equivalent(old.recall_score, new.recall_score)
        and _scores_equivalent(old.rerank_score, new.rerank_score)
        for old, new in zip(saved.chunks, incoming.chunks))


def snapshots_equivalent(saved: RetrievalSnapshot | None, incoming: RetrievalSnapshot | None) -> bool:
    """判断完成重试是否提供同一份完整、有序的快照事实，不以可合并代替相同。

    参数：saved/incoming 为严格验证的快照，None仅代表明确未检索。
    返回：bool，版本/状态/尝试顺序与数量精确相同，两个分数字段允许单步存储差异；异常：无。
    """
    if saved is None or incoming is None:
        return saved is incoming
    return (saved.model_dump(exclude={"attempts"}) == incoming.model_dump(exclude={"attempts"})
        and len(saved.attempts) == len(incoming.attempts)
        and all(_attempts_equivalent(old, new) for old, new in zip(saved.attempts, incoming.attempts)))


def merge_snapshots(saved: RetrievalSnapshot | None, incoming: RetrievalSnapshot) -> RetrievalSnapshot:
    """按稳定尝试标识合并，已有不同事实拒绝覆盖。

    参数：saved 原轮旧记录或确认未检索；incoming 新记录。
    返回：独立合并快照；异常：ValueError，同尝试标识出现冲突事实。
    """
    attempts = {a.attempt_id: a.model_copy(deep=True) for a in saved.attempts} if saved else {}
    for attempt in incoming.attempts:
        previous = attempts.get(attempt.attempt_id)
        if previous is not None:
            if not _attempts_equivalent(previous, attempt):
                raise ValueError("同一次检索的历史快照不能被替换")
            # 等价也保留原件，不能反复写入相邻分数造成累计漂移。
            continue
        attempts[attempt.attempt_id] = attempt.model_copy(deep=True)
    statuses = {a.status for a in attempts.values()}
    statuses.update(s.status for s in (saved, incoming) if s is not None)
    status = next((s for s in ("fault", "legacy_unavailable", "legacy_incomplete", "ok", "empty") if s in statuses), "empty")
    return RetrievalSnapshot(status=status, attempts=list(attempts.values()))


class RetrievalCapture:
    """请求局部收集器，保留闸前 Top5 与部分故障证据；不挂在共享检索器上。"""

    def __init__(self, turn_id: str, request_key: str, top_k: int = 5):
        """初始化本次请求。参数：原轮ID、稳定请求键、正整数快照上限；返回：None；异常：ValueError，身份/上限非法。"""
        if not turn_id or not request_key or type(top_k) is not int or top_k < 1:
            raise ValueError("检索收集器需要原轮身份、请求键和正整数上限")
        self.turn_id, self.request_key, self.top_k = turn_id, request_key, top_k
        self.snapshot: RetrievalSnapshot | None = None
        self.on_update = None  # 可选异步持久化回调，只存在于当前请求内。

    def attempt_id(self, query: str, *, scope=None, bm25_query=None, rerank_query=None) -> str:
        """派生同轮稳定子查询身份。参数：实际查询及范围/阶段文本；返回：SHA256；异常：输入不可JSON序列化。"""
        value = [self.turn_id, self.request_key, query, scope, bm25_query, rerank_query]
        return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    async def record(self, attempt: RetrievalAttempt) -> None:
        """保留一次完整或失败尝试并及时保存。参数：真实尝试；返回：None；异常：冲突事实或持久化错误。"""
        self.snapshot = merge_snapshots(self.snapshot, RetrievalSnapshot(status=attempt.status, attempts=[attempt]))
        if self.on_update is not None:
            await self.on_update(self.snapshot)


def chunk_snapshot(chunk, rank: int, *, recall_score=None, rerank_score=None) -> RetrievalChunkSnapshot:
    """将权威块按主键配分保存。参数：原块/名次/实际召回和重排分；返回：片段快照；异常：契约错误。"""
    return RetrievalChunkSnapshot(chunk_id=chunk.id, source_key=chunk.source_key, content_hash=chunk.content_hash,
        embedding_revision=chunk.embedding_revision, question=chunk.text.questions, answer=chunk.text.answer,
        rank=rank, recall_score=recall_score, rerank_score=rerank_score)


def _legacy_chunks(hits, *, citation: bool = False) -> list[RetrievalChunkSnapshot]:
    """恢复旧原文，不推断身份/分数。参数：旧条目、citation表示旧引用而非原始召回块；返回：最多五块；异常：契约错误。"""
    chunks = []
    for hit in hits if isinstance(hits, list) else []:
        # 未记录问答正文的条目无法满足原文契约，保守跳过而不填造空答案。
        if not isinstance(hit, dict) or not isinstance(hit.get("question"), str) or not isinstance(hit.get("answer"), str):
            continue
        identifier = hit.get("id", hit.get("chunk_id"))
        chunks.append(RetrievalChunkSnapshot(chunk_id=identifier if type(identifier) is int and identifier > 0 else None,
            # 旧引用的content_hash仅为答案正文指纹，不能充当知识三格的原块指纹。
            source_key=hit.get("source_key"), content_hash=None if citation else hit.get("content_hash"),
            embedding_revision=hit.get("embedding_revision"), question=hit["question"], answer=hit["answer"],
            rank=len(chunks) + 1, recall_score=hit.get("recall_score"), rerank_score=hit.get("rerank_score")))
    return chunks[:5]


def recover_legacy_snapshot(messages: list[dict]) -> RetrievalSnapshot | None:
    """只读取调用方给出的单个原轮消息，优先已保存快照再恢复旧工具结果。

    参数：messages 同一原轮的用户/工具/助手记录，不接受多轮混合输入。
    返回：快照；仅明确保存未检索标记时为None，无记录时legacy_unavailable。
    异常：ValueError，多轮混入或已保存的现代快照损坏；不调用检索器。
    """
    if len({m["turn_id"] for m in messages if m.get("turn_id")}) > 1:
        raise ValueError("历史快照只能恢复指定原轮")
    for message in messages:
        metadata = message.get("rag_metadata") or {}
        if message.get("role") == "user" and "retrieval_snapshot" in metadata:
            raw = metadata["retrieval_snapshot"]
            return RetrievalSnapshot.model_validate(raw) if raw is not None else None
    calls, attempts, saved, completed = {}, [], None, set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            if (isinstance(call, dict) and call.get("name") in {"query_faq", "query_refund_policy"}
                    and (call.get("binding") or {}).get("identity", {}).get("source", "builtin") == "builtin"):
                calls[call.get("id")] = call
        if message.get("role") != "tool" or message.get("tool_call_id") not in calls:
            continue
        call = calls[message["tool_call_id"]]
        completed.add(message["tool_call_id"])
        try:
            payload = json.loads(message.get("content") or "{}")
        except (ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        outcome = payload.get("verified_knowledge") or {}
        if outcome.get("retrieval_snapshot") is not None:
            saved = merge_snapshots(saved, RetrievalSnapshot.model_validate(outcome["retrieval_snapshot"]))
            continue
        hits = (payload.get("data") or {}).get("hits", outcome.get("evidence", []))
        args = call.get("args") or {}
        attempts.append(RetrievalAttempt(attempt_id=f"legacy:{message['tool_call_id']}",
            query=args.get("query", args.get("keyword")), status="legacy_incomplete", chunks=_legacy_chunks(hits)))
    for identifier, call in calls.items():
        if identifier not in completed:
            args = call.get("args") or {}
            attempts.append(RetrievalAttempt(attempt_id=f"legacy:{identifier}",
                query=args.get("query", args.get("keyword")), status="legacy_incomplete"))
    if not calls and saved is None:
        for i, message in enumerate(messages):
            chunks = _legacy_chunks((message.get("rag_metadata") or {}).get("citations"), citation=True)
            if message.get("role") == "assistant" and chunks:
                attempts.append(RetrievalAttempt(attempt_id=f"legacy:citations:{i}", status="legacy_incomplete", chunks=chunks))
    if attempts:
        saved = merge_snapshots(saved, RetrievalSnapshot(status="legacy_incomplete", attempts=attempts))
    return saved or RetrievalSnapshot(status="legacy_unavailable")
