"""当前在线混合集合的只读浏览、同步核对与分阶段检索调试。

复用在线search的编码器、索引、原文存储和重排器，不调用改写、回答生成或数据写入。
Milvus原始记录与MySQL原始18字段并列展示，失配只影响重排资格，召回始终保留。
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from time import monotonic
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.knowledge.hybrid_index import MilvusHybridIndex
from app.knowledge.limits import valid_dense_vector
from app.knowledge.retrieval import HybridKnowledgeSearch
from app.knowledge.rerank import RerankUnavailable
from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, KnowledgeUnavailable, VectorHit
from app.knowledge.text import content_fingerprint, format_knowledge

_MYSQL_TIMEOUT_SECONDS = 3.0
_MYSQL_WARNING = "MySQL原文读取暂时不可用，本次只展示Milvus记录，无法核对同步状态"


def _utc_now() -> str:
    """参数：无；返回：带UTC偏移的ISO读取时间；异常：无。"""
    return datetime.now(timezone.utc).isoformat()


def _remaining_seconds(deadline: float) -> float:
    """检查整次预算，供实际业务等待使用，不向消费者发送任务挂载计时器。

    参数：deadline为monotonic时钟上的整次结束期限。
    返回：float，剩余正秒数；异常：TimeoutError，发送或业务执行已耗尽总预算。
    """
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("知识调试请求预算已耗尽")
    return remaining


class RetrievalDebugRequest(BaseModel):
    """调试专用参数，严格拒绝布尔数值、非有限权重及隐式整数转换。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str = Field(strict=True, min_length=1, max_length=2000, description="去首尾空白的完整原问，1–2000字符")
    strategy: Literal["weighted", "rrf", "dense", "bm25"] = Field(default="weighted", description="原生加权、固定60的RRF或单路策略")
    dense_weight: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False, description="Dense权重，有限0–1，weighted自动归一化")
    sparse_weight: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False, description="BM25权重，有限0–1，weighted自动归一化")
    dense_limit: int = Field(default=50, strict=True, ge=1, le=50, description="Dense原始召回上限，1–50严格整数")
    sparse_limit: int = Field(default=50, strict=True, ge=1, le=50, description="BM25原始召回上限，1–50严格整数")
    candidate_limit: int = Field(default=50, strict=True, ge=1, le=50, description="融合候选上限，1–50严格整数")
    rerank_top_k: int = Field(default=10, strict=True, ge=1, le=50, description="重排最终返回上限，全部合法候选均评分")
    rerank: bool = Field(default=True, strict=True, description="是否对已校验完整原文进行共享模型重排")

    @field_validator("query", mode="before")
    @classmethod
    def clean_query(cls, value: Any) -> Any:
        """参数：用户query原值；返回：仅去首尾空白的文本；异常：字段校验负责非法类型。"""
        return value.strip() if isinstance(value, str) else value

    @field_validator("dense_weight", "sparse_weight", mode="before")
    @classmethod
    def numeric_weight(cls, value: Any) -> Any:
        """参数：权重原值；返回：真实数字；异常：ValueError，bool或字符串等隐式数字。"""
        if type(value) not in (int, float):
            raise ValueError("权重必须为真实数字，不能使用布尔值或字符串")
        return value

    @model_validator(mode="after")
    def positive_total(self) -> RetrievalDebugRequest:
        """参数：当前请求；返回：已校验请求；异常：ValueError，加权策略全零。"""
        if self.strategy == "weighted" and self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("加权策略至少需要一个正权重")
        return self

    def parameters(self) -> dict:
        """参数：无；返回：实际执行参数，weighted回显归一化权重；异常：无。"""
        result = self.model_dump(exclude={"query"})
        if self.strategy == "weighted":
            total = self.dense_weight + self.sparse_weight
            result["dense_weight"], result["sparse_weight"] = self.dense_weight / total, self.sparse_weight / total
        return result


def _raw_snapshot(row: dict) -> ChunkSnapshot:
    """将原始18字段转换为现有模型，保留共享content_type空值的unknown兼容规则。

    参数：row为JSON兼容MySQL原始记录；返回：完整共享快照。
    异常：Pydantic字段校验错误；原始记录本身不做修改。
    """
    return ChunkSnapshot(id=int(row["id"]), source_key=row["source_key"], ordinal=row["ordinal"],
        text=KnowledgeText(category=row["category"], questions=row["questions"], answer=row["answer"]),
        section_path=row["section_path"], content_type=row["content_type"] or "unknown",
        is_key_clause=bool(row["is_key_clause"]), content_hash=row["content_hash"],
        embedding_revision=row["embedding_revision"], is_active=bool(row["is_active"]),
        vector_id=row["vector_id"], vectorize_status=row["vectorize_status"],
        prev_chunk_id=int(row["prev_chunk_id"]) if row["prev_chunk_id"] is not None else None,
        next_chunk_id=int(row["next_chunk_id"]) if row["next_chunk_id"] is not None else None)


class KnowledgeInspector:
    """包装当前在线共享混合search，以请求局部状态执行只读诊断。"""

    def __init__(self, search: HybridKnowledgeSearch) -> None:
        """参数：已启动的共享search；返回：None；异常：无，不加载任何资源。"""
        self.search = search

    def preflight(self) -> None:
        """开流前核验当前hybrid运行时及必要共享依赖，不尝试初始化。

        参数：无；返回：None；异常：KnowledgeUnavailable，依赖缺失/已关闭/类型不匹配。
        """
        search = self.search
        if not isinstance(search, HybridKnowledgeSearch) or not isinstance(search.index, MilvusHybridIndex):
            raise KnowledgeUnavailable("当前在线混合知识检索尚未就绪")
        if search.config.knowledge_retrieval_mode != "hybrid":
            raise KnowledgeUnavailable("当前运行时不是混合检索")
        if (not callable(getattr(search.store, "get_raw_chunks", None))
                or not callable(getattr(search.encoder, "encode", None))
                or not isinstance(getattr(search.encoder, "revision", None), str)
                or not callable(getattr(search.reranker, "rank", None))
                or getattr(search.encoder, "_closed", False) or getattr(search.reranker, "_closed", False)):
            raise KnowledgeUnavailable("知识检索共享依赖暂时不可用")
        search.index._require_client()

    def _identity(self) -> dict:
        """参数：无；返回：共享索引公开库名与集合名；异常：无。"""
        config = self.search.index._config
        return {"database": config.milvus_database, "collection": config.milvus_collection}

    async def collection(self) -> dict:
        """参数：无；返回：实时集合结构/count和UTC读取时间；异常：Milvus不可用。"""
        result = await self.search.index.inspection_collection()
        return {**result, "read_at": _utc_now()}

    async def _mysql(self, ids: list[int]) -> tuple[dict[int, dict], list[str]]:
        """用三秒预算批量读取原始MySQL字段，故障明确降级。

        参数：ids为本批主键并集；返回：主键记录字典与安全warnings。
        异常：取消向上传播；其他数据库故障返回告警而不当缺字段。
        """
        try:
            rows = await asyncio.wait_for(self.search.store.get_raw_chunks(ids), _MYSQL_TIMEOUT_SECONDS)
            identifiers = [int(row["id"]) for row in rows]
            if len(identifiers) != len(set(identifiers)) or any(value not in ids for value in identifiers):
                raise ValueError("MySQL回读主键重复或越界")
            return dict(zip(identifiers, rows)), []
        except Exception:
            return {}, [_MYSQL_WARNING]

    def _record(self, identifier: int, milvus: dict | None, mysql: dict | None, *,
                read_at: str, mysql_failed: bool, fingerprints: list[VectorHit] | None = None
                ) -> tuple[dict, ChunkSnapshot | None]:
        """展示原始双库字段并核对真实正文与所有搜索快照，不把新正文配给旧命中。

        参数：identifier正主键；双库原始记录；read_at本批UTC时间；故障标记；可选各路搜索指纹。
        返回：展示行及仅同步完全正确时的共享重排快照。
        异常：无，非法原文字段形成同步失配原因。
        """
        snapshot, reasons = None, []
        changed = bool(fingerprints and (milvus is None or any(
            hit.content_hash != milvus.get("content_hash") or hit.embedding_revision != milvus.get("embedding_revision")
            for hit in fingerprints)))
        if mysql_failed:
            status, label, reasons = "mysql_unavailable", "原文读取失败", [_MYSQL_WARNING]
        elif mysql is None:
            status, label, reasons = "missing_mysql", "缺少MySQL原文", ["Milvus主键没有对应MySQL记录"]
        elif milvus is None:
            status, label, reasons = "missing_milvus", "Milvus记录已不存在", ["搜索后Milvus记录已被移除"]
        else:
            try:
                snapshot = _raw_snapshot(mysql)
                if not snapshot.is_active:
                    reasons.append("MySQL记录已撤回")
                if snapshot.vectorize_status != "done":
                    reasons.append("MySQL向量同步状态尚未完成")
                if snapshot.vector_id != str(identifier):
                    reasons.append("MySQL vector_id与主键不一致")
                actual_hash = content_fingerprint(snapshot.text)
                if actual_hash != snapshot.content_hash:
                    reasons.append("MySQL正文实际指纹与保存指纹不一致")
                if actual_hash != milvus["content_hash"]:
                    reasons.append("MySQL正文实际指纹与Milvus指纹不一致")
                if snapshot.content_hash != milvus["content_hash"]:
                    reasons.append("两库保存的正文指纹不一致")
                if format_knowledge(snapshot.text) != milvus["text"]:
                    reasons.append("Milvus正文与MySQL三格原文不一致")
                if snapshot.embedding_revision != milvus["embedding_revision"]:
                    reasons.append("两库编码版本不一致")
                if snapshot.embedding_revision != self.search.encoder.revision or milvus["embedding_revision"] != self.search.encoder.revision:
                    reasons.append("记录编码版本与当前共享编码器不一致")
            except Exception:
                reasons.append("MySQL原始记录字段不能形成合法知识快照")
            status = "changed" if changed else "mismatch" if reasons else "ok"
            label = {"ok": "同步一致", "mismatch": "同步不一致", "changed": "读取期间已变化"}[status]
        # 主状态保持稳定，但两库事实独立累积，SQL故障不能遮蔽Milvus变化或移除。
        if milvus is None and status != "missing_milvus":
            reasons.append("搜索后Milvus记录已被移除")
        elif milvus is not None and changed:
            reasons.append("搜索指纹与回读指纹不同，读取期间记录已变化")
        if reasons:
            snapshot = None
        raw_milvus = {**milvus, "id": str(identifier)} if milvus is not None else None
        return {"id": str(identifier), "milvus": raw_milvus, "mysql": mysql,
                "sync": {"status": status, "label": label, "reasons": reasons}, "read_at": read_at}, snapshot

    async def records(self, *, after_id: int = 0, limit: int = 20) -> dict:
        """参数：after_id非负INT64游标，limit为1–100；返回：原始记录页；异常：参数/Milvus错误。"""
        rows = await self.search.index.inspection_page(after_id, limit)
        selected, read_at = rows[:limit], _utc_now()
        metadata, warnings = await self._mysql([row["id"] for row in selected])
        records = [self._record(row["id"], row, metadata.get(row["id"]), read_at=read_at,
                    mysql_failed=bool(warnings))[0] for row in selected]
        return {**self._identity(), "read_at": read_at, "rows": records,
                "next_after_id": str(selected[-1]["id"] if selected else after_id),
                "has_more": len(rows) > limit, "warnings": warnings}

    async def record(self, identifier: int) -> dict | None:
        """参数：identifier正signed INT64；返回：含vector的单行或None；异常：参数/Milvus错误。"""
        rows = await self.search.index.inspection_records([identifier], vector=True)
        if not rows:
            return None
        metadata, warnings = await self._mysql([identifier])
        return self._record(identifier, rows[0], metadata.get(identifier), read_at=_utc_now(),
                            mysql_failed=bool(warnings))[0]

    async def _recall(self, request: RetrievalDebugRequest, vector: list[float] | None) -> dict[str, list[VectorHit]]:
        """独立读取单路原始分数并按策略调用原生融合。

        参数：已校验请求与一次编码向量，纯BM25为None。
        返回：dense/sparse/fused命中组；异常：索引或命中身份/数量不合法。
        """
        index, params = self.search.index, request.parameters()
        dense, sparse = [], []
        if request.strategy != "bm25":
            dense = await index.search(vector, request.dense_limit)
        if request.strategy != "dense":
            sparse = await index.search_bm25(request.query, request.sparse_limit)
        if request.strategy == "dense":
            fused = dense[:request.candidate_limit]
        elif request.strategy == "bm25":
            fused = sparse[:request.candidate_limit]
        elif request.strategy == "rrf":
            fused = await index.hybrid_search(vector, request.query, dense_limit=request.dense_limit,
                sparse_limit=request.sparse_limit, limit=request.candidate_limit)
        else:
            fused = await index.weighted_search(vector, request.query, dense_limit=request.dense_limit,
                sparse_limit=request.sparse_limit, limit=request.candidate_limit,
                dense_weight=params["dense_weight"], sparse_weight=params["sparse_weight"])
        groups = {"dense": dense, "sparse": sparse, "fused": fused}
        for name, size in (("dense", request.dense_limit), ("sparse", request.sparse_limit), ("fused", request.candidate_limit)):
            hits = groups[name]
            if len(hits) > size or len({hit.id for hit in hits}) != len(hits):
                raise KnowledgeUnavailable("索引候选重复或超出请求上限")
        return groups

    async def _recall_rows(self, groups: dict[str, list[VectorHit]], request: RetrievalDebugRequest
                           ) -> tuple[dict, dict[int, ChunkSnapshot], dict[int, dict]]:
        """回读命中并集一次，保留原始分数、缺路null及每路指纹校验。

        参数：三路命中与请求；返回：recall事件内容、合法快照、主键展示记录。
        异常：Milvus回读错误；MySQL故障降级为warnings。
        """
        fingerprints: dict[int, list[VectorHit]] = {}
        for hits in groups.values():
            for hit in hits:
                fingerprints.setdefault(hit.id, []).append(hit)
        ids = sorted(fingerprints)
        milvus = {row["id"]: row for row in await self.search.index.inspection_records(ids)}
        mysql, warnings = await self._mysql(ids)
        read_at, records, chunks = _utc_now(), {}, {}
        for identifier in ids:
            record, chunk = self._record(identifier, milvus.get(identifier), mysql.get(identifier),
                read_at=read_at, mysql_failed=bool(warnings), fingerprints=fingerprints[identifier])
            records[identifier] = record
            if chunk is not None:
                chunks[identifier] = chunk
        dense = {hit.id: (rank, hit.score) for rank, hit in enumerate(groups["dense"], 1)}
        sparse = {hit.id: (rank, hit.score) for rank, hit in enumerate(groups["sparse"], 1)}
        fused = {hit.id: hit.score for hit in groups["fused"]}
        rendered = {}
        for name, hits in groups.items():
            rendered[name] = [{**records[hit.id], "rank": rank,
                "dense_rank": dense.get(hit.id, (None, None))[0], "sparse_rank": sparse.get(hit.id, (None, None))[0],
                "cosine_score": dense.get(hit.id, (None, None))[1], "bm25_score": sparse.get(hit.id, (None, None))[1],
                "weighted_score": fused.get(hit.id) if request.strategy == "weighted" else None,
                "rrf_score": fused.get(hit.id) if request.strategy == "rrf" else None}
                for rank, hit in enumerate(hits, 1)]
        return {"query": request.query, "parameters": request.parameters(), "read_at": read_at, **rendered,
                "counts": {**{name: len(hits) for name, hits in groups.items()}, "union": len(ids),
                           "eligible": sum(hit.id in chunks for hit in groups["fused"])},
                "warnings": warnings}, chunks, records

    async def _rerank(self, request: RetrievalDebugRequest, fused: list[VectorHit],
                      chunks: dict[int, ChunkSnapshot], records: dict[int, dict], *, mysql_failed: bool) -> dict:
        """评分全部合法融合候选并验证共享模型的完整快照、主键与分数。

        参数：已验证请求、融合组、合法快照及展示记录；mysql_failed表示本次原文读取故障。
        返回：重排事件含Top-K、全部scored_ids及明确skipped原因。
        异常：重排依赖/查询长度/模型输出错误；MySQL故障启用重排时失败关闭。
        """
        if request.rerank and mysql_failed:
            raise RerankUnavailable("MySQL原文读取不可用，无法安全重排")
        candidates = [chunks[hit.id] for hit in fused if hit.id in chunks]
        scored_ids = [str(chunk.id) for chunk in candidates] if request.rerank else []
        skipped = [{"id": str(hit.id), "reason": "同步核对未通过，未送入重排模型"}
                   for hit in fused if hit.id not in chunks]
        ranked = []
        if request.rerank and candidates:
            ranked = await self.search.reranker.rank(request.query, candidates, top_n=request.rerank_top_k)
            expected = {chunk.id: chunk for chunk in candidates}
            ids = [item.chunk.id for item in ranked]
            if (len(ids) != min(request.rerank_top_k, len(candidates)) or len(ids) != len(set(ids))
                    or any(item.chunk != expected.get(item.chunk.id) or not math.isfinite(item.score)
                           or not 0 <= item.score <= 1 for item in ranked)):
                raise RerankUnavailable("重排结果缺项、重复、替换原文或分数非法")
            skipped.extend({"id": str(chunk.id), "reason": "已重排，未进入重排Top-K"}
                           for chunk in candidates if chunk.id not in ids)
        elif not request.rerank:
            skipped = [{"id": str(hit.id), "reason": "本次未启用重排"} for hit in fused]
        recall_ranks = {hit.id: rank for rank, hit in enumerate(fused, 1)}
        rows = [{**records[item.chunk.id], "rank": rank, "recall_rank": recall_ranks[item.chunk.id],
                 "rerank_score": item.score, "rank_change": recall_ranks[item.chunk.id] - rank}
                for rank, item in enumerate(ranked, 1)]
        return {"query": request.query, "parameters": request.parameters(), "read_at": _utc_now(),
                "rows": rows, "scored_ids": scored_ids, "skipped": skipped}

    async def debug(self, request: RetrievalDebugRequest) -> AsyncIterator[dict]:
        """依次发布编码/召回/重排阶段，整次沿用线上超时，故障保留已发布召回。

        参数：request为已验证调试请求；返回：status/recall/rerank/complete或唯一error事件。
        异常：取消原样传播，其他故障转换安全终止事件；不关闭共享模型。
        """
        started, stage = monotonic(), "encoding" if request.strategy != "bm25" else "recall"
        deadline = started + self.search.config.knowledge_timeout_seconds
        timings = {"encoding_ms": 0., "recall_ms": 0., "rerank_ms": 0.}
        try:
            vector = None
            if request.strategy != "bm25":
                yield {"event": "status", "data": {"stage": "encoding", "message": "正在编码完整原问"}}
                before = monotonic()
                async with asyncio.timeout(_remaining_seconds(deadline)):
                    vectors = await self.search.encoder.encode([request.query])
                if len(vectors) != 1 or not valid_dense_vector(vectors[0]):
                    raise KnowledgeUnavailable("查询编码结果不合法")
                vector = vectors[0]
                timings["encoding_ms"] = (monotonic() - before) * 1000
            stage, before = "recall", monotonic()
            _remaining_seconds(deadline)
            yield {"event": "status", "data": {"stage": "recall", "message": "正在读取原始召回与双库记录"}}
            # 每个实际await使用同一deadline的剩余预算；超时上下文不能跨越任何yield。
            async with asyncio.timeout(_remaining_seconds(deadline)):
                groups = await self._recall(request, vector)
            async with asyncio.timeout(_remaining_seconds(deadline)):
                recall, chunks, records = await self._recall_rows(groups, request)
            timings["recall_ms"] = (monotonic() - before) * 1000
            _remaining_seconds(deadline)
            # 先发布完整召回，后续超长、模型故障或MySQL故障都不能抹掉这一阶段。
            yield {"event": "recall", "data": recall}
            stage, before = "rerank", monotonic()
            # 慢发送已超过预算时，从yield返回后发送安全error，不取消仍连接的发送任务。
            _remaining_seconds(deadline)
            if request.rerank:
                yield {"event": "status", "data": {"stage": "rerank", "message": "正在对全部合法候选评分"}}
            async with asyncio.timeout(_remaining_seconds(deadline)):
                ranked = await self._rerank(request, groups["fused"], chunks, records, mysql_failed=bool(recall["warnings"]))
            timings["rerank_ms"] = (monotonic() - before) * 1000
            _remaining_seconds(deadline)
            yield {"event": "rerank", "data": ranked}
            _remaining_seconds(deadline)
            timings["total_ms"] = (monotonic() - started) * 1000
            yield {"event": "complete", "data": {"query": request.query, "parameters": request.parameters(), "timings": timings}}
        except Exception as error:
            if stage == "rerank" and isinstance(error, ValueError):
                message = "重排问题或原文超出模型长度限制，请缩短问题；本次召回已保留"
            else:
                message = {"encoding": "问题编码暂时不可用，请稍后重试",
                    "recall": "知识召回暂时不可用，请稍后重试",
                    "rerank": "知识重排暂时不可用；本次召回已保留"}[stage]
            yield {"event": "error", "data": {"stage": stage, "message": message}}
