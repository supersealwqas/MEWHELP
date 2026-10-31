"""CH09只读真实验收：清单只指定身份，结论来自数据库、完整观测与冻结评估原件。

CLI：--base-url --output-dir --evidence-manifest。缺项保持pending并退出1；
仅GET/SELECT，不发聊天、反馈、审核、发布或评估请求，不创建/清理数据库。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REQUIRED = {"knowledge_loop", "self_check", "mixed", "feedback", "replay", "concurrency", "error", "costs", "evaluations"}


def require(condition, message):
    """优化模式亦执行门禁。参数：条件/固定说明；返回：None；异常：ValueError，证据不足。"""
    if not condition:
        raise ValueError(message)


def canonical(value):
    """统一ORM日期/元组为JSON结构。参数：可序列化值；返回：JSON对象；异常：非法数字/类型。"""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False,
        default=lambda item: item.isoformat() if isinstance(item, datetime) else str(item)))


def read_reference(reference, base: Path, allowed: Path):
    """核验原件路径与内容摘要。参数：path/sha256引用、清单目录、许可根；返回：JSON；异常：ValueError/OSError。"""
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "证据引用须只有path和sha256")
    require(isinstance(reference["path"], str) and isinstance(reference["sha256"], str), "证据路径或摘要类型非法")
    path = (base / reference["path"]).resolve()
    require(path.is_relative_to(allowed.resolve()) and path.is_file(), "证据缺失或越出许可目录")
    raw = path.read_bytes()
    require(bool(raw) and hashlib.sha256(raw).hexdigest() == reference["sha256"], "证据SHA256不符")
    return json.loads(raw)


def observation_rows(value):
    """展开已有完整分页或聚合原件。参数：data/pages对象；返回：观测列表；异常：ValueError。"""
    if isinstance(value, dict) and isinstance(value.get("pages"), list):
        pages = value["pages"]
        require(bool(pages) and all(isinstance(page.get("data"), list) and isinstance(page.get("meta"), dict) for page in pages), "观测分页结构错误")
        require(not pages[-1]["meta"].get("cursor"), "最后一页仍有游标")
        cursors = [page["meta"].get("cursor") for page in pages[:-1]]
        require(all(cursors) and len(set(cursors)) == len(cursors), "中间分页缺失或游标重复")
        return [row for page in pages for row in page["data"]]
    require(isinstance(value, dict) and isinstance(value.get("data"), list), "缺少观测原始数据")
    return value["data"]


def unique_observations(rows, trace_id):
    """拒绝串trace和同ID矛盾数据，允许分页边界相同记录。参数：观测/trace；返回：去重列表；异常：ValueError。"""
    result = {}
    for row in rows:
        require(isinstance(row, dict) and row.get("traceId") == trace_id and isinstance(row.get("id"), str), "观测身份串线")
        previous = result.get(row["id"])
        require(previous is None or previous == row, "同一observation ID内容矛盾")
        result[row["id"]] = row
    return list(result.values())


def raw_usage(row):
    """读取原始方向用量，缺失保持None。参数：generation；返回：输入/输出；异常：ValueError，伪类型或来源。"""
    metadata = row.get("metadata") or {}
    require(metadata.get("usage_origin") in {"model_reported", "inferred_or_unknown"}, "缺少原始用量来源")
    values = []
    for direction in ("input", "output"):
        present, value = metadata.get(f"{direction}_usage_present"), metadata.get(f"reported_{direction}_tokens")
        require(type(present) is bool, "用量presence必须是布尔")
        require(type(value) is int and value >= 0 if present else value is None, "原始用量非法或缺失被伪装为零")
        values.append(value if present else None)
    return tuple(values)


def source_message(evidence):
    """取得唯一原用户消息。参数：数据库快照；返回：来源行；异常：ValueError，来源不唯一。"""
    rows = [row for row in evidence["messages"] if row["role"] == "user"]
    require(len(rows) == 1, "缺少唯一原用户消息")
    return rows[0]


def verify_turn(identity, evidence, observations):
    """核对原轮身份、完成答案与每次原始用量。参数：清单身份/实库快照/实机观测；返回：用量摘要；异常：ValueError。"""
    from app.schemas.flywheel import RetrievalSnapshot
    workflow = evidence["workflow"]
    context, trace = workflow["context"], workflow["trace_identity"]
    require(all(context.get(key) == identity[key] for key in ("session_id", "user_id", "turn_id")), "数据库原轮身份不符")
    require(all(trace.get(key) == identity[key] for key in ("session_id", "user_id", "turn_id", "trace_id")), "数据库trace绑定不符")
    source = source_message(evidence)
    require(source["content"] == context["user_message"] and source.get("turn_id") == identity["turn_id"], "原问或消息轮次不符")
    require(all(row.get("turn_id") == identity["turn_id"] for row in evidence["messages"]), "底层消息混入其他原轮")
    require((source.get("rag_metadata") or {}).get("trace_identity") == trace, "原用户行trace绑定与工作流摘要不符")
    require(source["turn_status"] == workflow["turn_status"], "原问与工作流状态不符")
    snapshot = workflow.get("retrieval_snapshot")
    if snapshot is not None:
        parsed = RetrievalSnapshot.model_validate(snapshot)
        require(parsed.status not in {"legacy_unavailable", "legacy_incomplete"}, "当前验收缺少完整历史快照")
        require(all(attempt.top_k == 5 and len(attempt.chunks) <= 5 for attempt in parsed.attempts), "快照不是约定Top5")
    rows = unique_observations(observations, identity["trace_id"])
    roots = [row for row in rows if row["id"] == trace["root_observation_id"] and row.get("name") == "execution.chat"]
    require(len(roots) == 1 and bool(roots[0].get("endTime")), "原执行根尚未结束或不存在")
    require((roots[0].get("metadata") or {}).get("turn_id") == identity["turn_id"], "原根归属其他轮次")
    generations = sorted((row for row in rows if row["type"] == "GENERATION"), key=lambda row: (row["startTime"], row["id"]))
    require(all((row.get("metadata") or {}).get("turn_id") == identity["turn_id"] and row.get("endTime") for row in generations), "模型调用轮次或完成状态错误")
    usages = [raw_usage(row) for row in generations]
    if workflow["turn_status"] == "completed":
        answers = [row for row in evidence["messages"] if row["role"] == "assistant" and not row.get("tool_calls") and row["turn_status"] == "completed"]
        require(len(answers) == 1 and answers[0]["content"] == workflow["answer"] and bool(workflow["answer"]), "最终答案缺失、重复或与实库不符")
        budget = workflow["completion_metadata"]["workflow"]["budget"]
        require(type(budget["model_calls"]) is int and budget["model_calls"] == len(generations) == len(budget["records"]), "模型调用数量与完成预算不符")
        for record, usage in zip(budget["records"], usages, strict=True):
            expected = record.get("usage") or {}
            require(usage == (expected.get("input_tokens"), expected.get("output_tokens")), "逐generation原始用量与实库不符")
        if all(None not in usage for usage in usages):
            require(budget["actual_tokens"] == sum(sum(usage) for usage in usages), "模型实际总用量不符")
    return {"model_call_count": len(generations), "known_tokens": sum(sum(value for value in usage if value is not None) for usage in usages),
        "unknown_usage_count": sum(None in usage for usage in usages), "turn_status": workflow["turn_status"], "trace_id": identity["trace_id"]}


def verify_replay(identity, before, after, current, timing):
    """前后及当前ID/原始方向消费必须相同，迟到导出不能算回放。参数：身份/三组观测/HTTP时间；返回：零新增证明；异常：ValueError。"""
    start, end = datetime.fromisoformat(timing["started"]), datetime.fromisoformat(timing["ended"])
    require(start.utcoffset() is not None and end >= start, "回放时序非法")
    signatures = []
    for rows in (before, after, current):
        rows = unique_observations(rows, identity["trace_id"])
        generations = [row for row in rows if row["type"] == "GENERATION"]
        require(bool(generations), "回放基线没有完整模型观测")
        require(all(row.get("endTime") and datetime.fromisoformat(row["endTime"]) < start for row in generations), "基线含未完成或回放后迟到调用")
        signatures.append({row["id"]: (row["traceId"], row["startTime"], row["endTime"],
            row["metadata"].get("usage_origin"), raw_usage(row)) for row in generations})
    require(signatures[0] == signatures[1] == signatures[2], "回放前后generation或原始用量变化")
    return {"generation_count": len(signatures[0]), "new_model_calls": 0, "new_known_tokens": 0}


def verify_pool(evidence, source):
    """核对原轮入口主键、原文与完整历史快照。参数：实库快照/固定source；返回：池行；异常：ValueError。"""
    from app.knowledge.snapshots import snapshots_equivalent
    from app.schemas.flywheel import RetrievalSnapshot, TurnPoolState
    message = source_message(evidence)
    state = TurnPoolState.model_validate(message.get("rag_metadata", {}).get("knowledge_pool") or {})
    identifier = state.pool_ids.get(source)
    rows = [row for row in evidence["pool"] if row["id"] == identifier]
    require(identifier is not None and len(rows) == 1, "原轮入口没有唯一落池行")
    row, workflow = rows[0], evidence["workflow"]
    require(row["source"] == source and row["conversation_id"] == workflow["context"]["conversation_id"] and row["raw_question"] == message["content"], "问题池来源串轮")
    snapshots = [RetrievalSnapshot.model_validate(value) if value is not None else None
                 for value in (row["retrieved_chunks"], workflow.get("retrieval_snapshot"))]
    require(snapshots_equivalent(*snapshots), "问题池历史快照与原轮不符")
    return row


def verify_evaluation_pair(directories, database_rows):
    """复用Task9完整产物验证并核对DB和比较协议。参数：两个产物路径/对应实库行；返回：分数差值；异常：ValueError。"""
    from app.evaluation.runner import validate_completed_artifact
    require(len(directories) == len(database_rows) == 2, "需要两轮实际评估")
    summaries = [validate_completed_artifact(Path(path)) for path in directories]
    require(summaries[0]["artifact_id"] != summaries[1]["artifact_id"] and str(database_rows[0]["id"]) != str(database_rows[1]["id"]), "两轮评估不能复用同一产物或DB行")
    for summary, row in zip(summaries, database_rows, strict=True):
        require(row["dataset_size"] == summary["dataset_size"] == 48 and row["metrics"] == summary, "评估完整产物与DB行不符")
        require(summary["metadata"]["split"] == "evaluation", "校准题混入独立验收")
    require(summaries[0]["comparison_key"] == summaries[1]["comparison_key"], "评估协议不同，不能合并趋势")
    return {"artifacts": [row["artifact_id"] for row in summaries], "knowledge_versions": [row["metadata"]["corpus_version"] for row in summaries],
        "error_counts": [row["error_count"] for row in summaries], "differences": {key: summaries[1][key] - summaries[0][key]
            if all(row[key] is not None for row in summaries) else None for key in ("recall_at_k", "mrr", "faithfulness")}}


def verify_knowledge_loop(before, after, review, chunks, fingerprints, approved):
    """核验原问缺口到同问引用核准正文的完整链。参数：前后实库/审核/知识/向量/核准原件；返回：身份；异常：ValueError。"""
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL
    from app.knowledge.schemas import KnowledgeText
    from app.knowledge.text import content_fingerprint
    old, new = before["workflow"], after["workflow"]
    require(old.get("retrieval_snapshot") is not None and bool(old["retrieval_snapshot"].get("attempts")), "知识缺口缺少当轮实际检索快照")
    require(old["context"]["user_message"] == new["context"]["user_message"], "发布前后不是同一个问题")
    require(old["turn_status"] == new["turn_status"] == "completed" and old["answer"].endswith(FIXED_KNOWLEDGE_REFUSAL)
            and not new["answer"].endswith(FIXED_KNOWLEDGE_REFUSAL), "缺少真实兜底到回答的变化")
    require(review["review_status"] == "通过" and review["publication_status"] == "published", "人工审核尚未完成发布")
    require(str(approved["review_id"]) == str(review["id"]) and approved["answer"] == review["approved_answer"], "核准原件与审核正文不符")
    sources = (source_message(before).get("rag_metadata") or {}).get("knowledge_pool", {}).get("pool_ids", {})
    pools = [verify_pool(before, source) for source in sources if source in {"retrieval_low_conf", "self_check"}]
    require(bool(pools) and all(str(row.get("matched_review_id")) == str(review["id"]) for row in pools), "原缺口未归并到本审核")
    key = f"ch09/review/{review['id']}"
    published = [row for row in chunks if row["source_key"] == key and row["is_active"]]
    require(len(published) == 1, "发布知识不存在或重复")
    chunk = published[0]
    require(chunk["answer"] == approved["answer"] and chunk["questions"] == review["normalized_question"]
            and chunk["category"] == "商品与购物 FAQ" and chunk["content_type"] == "faq", "发布正文或固定分类不符")
    expected_hash = content_fingerprint(KnowledgeText(category=chunk["category"], questions=chunk["questions"], answer=chunk["answer"]))
    require(chunk["content_hash"] == expected_hash and chunk["vectorize_status"] == "done" and chunk["vector_id"] == str(chunk["id"])
            and fingerprints.get(chunk["id"]) == (expected_hash, chunk["embedding_revision"]), "MySQL或强一致向量指纹不符")
    citations = new["completion_metadata"].get("citations", [])
    require(any(row.get("chunk_id") == chunk["id"] and row.get("source_key") == key and row.get("content_hash") == expected_hash
                and row.get("answer") == approved["answer"] for row in citations), "发布后答案没有引用本条核准事实")
    snapshot = new.get("retrieval_snapshot") or {}
    require(any(row.get("chunk_id") == chunk["id"] and row.get("source_key") == key and row.get("content_hash") == expected_hash
                and row.get("answer") == approved["answer"] for attempt in snapshot.get("attempts", []) for row in attempt["chunks"]), "发布后原轮快照未记录核准来源")
    return {"review_id": str(review["id"]), "chunk_id": chunk["id"], "source_key": key, "content_hash": expected_hash}


def verify_feedback(before, after, repeated, current):
    """核验反馈新落点及重复不增，当前实库须仍保留同一原件。参数：三阶段/当前快照；返回：池身份；异常：ValueError。"""
    states = (before, after, repeated, current)
    require(all(row["workflow"]["context"] == before["workflow"]["context"] for row in states), "反馈证据串轮")
    times = [datetime.fromisoformat(row["captured_at"]) for row in states[:3]]
    require(times[0] < times[1] < times[2], "反馈前后采集时序错误")
    mappings = [(source_message(row).get("rag_metadata") or {}).get("knowledge_pool", {}).get("pool_ids", {}) for row in states]
    require("user_feedback" not in mappings[0] and "user_feedback" in mappings[1], "未证明首次反馈新增落点")
    require(mappings[1] == mappings[2] == mappings[3], "重复反馈新增问题池行")
    pools = [verify_pool(row, "user_feedback") for row in states[1:]]
    keys = ("id", "source", "conversation_id", "raw_question", "retrieved_chunks")
    require(all({key: row[key] for key in keys} == {key: pools[0][key] for key in keys} for row in pools), "重复反馈修改历史原件")
    require(after.get("review_counts") == repeated.get("review_counts"), "重复反馈增加审核出现次数")
    return {"pool_id": pools[0]["id"], "repeated_new_rows": 0}


def verify_concurrency(groups):
    """从真实根时间证明不同trace重叠。参数：各轮完整观测；返回：并发事实；异常：ValueError。"""
    roots = [next((row for row in rows if row.get("name") == "execution.chat"), None) for rows in groups]
    require(len(roots) >= 2 and all(roots), "缺少并发原根")
    require(len({row["traceId"] for row in roots}) == len(roots), "并发不能复用同一trace")
    require(max(datetime.fromisoformat(row["startTime"]) for row in roots) < min(datetime.fromisoformat(row["endTime"]) for row in roots), "请求没有实际重叠")
    return {"traces": [row["traceId"] for row in roots], "overlap": True}


def verify_error(evidence, observations):
    """承认真实异常观测或与实库配对的结构化工具故障，不把低分当错误。参数：实库/观测；返回：错误身份；异常：ValueError。"""
    failures = []
    for message in evidence["messages"]:
        if message["role"] != "tool":
            continue
        try:
            result = json.loads(message["content"])
        except (TypeError, ValueError):
            continue
        if result.get("ok") is False and (result.get("error") or {}).get("category") == "fault":
            failures.append(result)
    found = []
    for row in observations:
        if row.get("level") == "ERROR":
            found.append(row["id"])
        elif row.get("type") == "TOOL" and (row.get("metadata") or {}).get("result_ok") is False:
            output = row.get("output")
            try:
                output = json.loads(output) if isinstance(output, str) else output
            except ValueError:
                continue
            if output in failures:
                found.append(row["id"])
    require(bool(found), "缺少实际错误观测及匹配的实库故障")
    return {"observations": found, "persistent_tool_faults": len(failures)}


def knowledge_decisions(value):
    """只解析模型输出中的结构化知识自评。参数：输出JSON；返回：决定列表；异常：非法嵌套JSON视为空。"""
    if isinstance(value, str):
        try:
            return knowledge_decisions(json.loads(value))
        except ValueError:
            return []
    if isinstance(value, list):
        return [item for child in value for item in knowledge_decisions(child)]
    if not isinstance(value, dict):
        return []
    if value.get("name") == "submit_knowledge_answer":
        args = value.get("args", value.get("arguments"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = None
        if isinstance(args, dict) and type(args.get("useful")) is bool:
            return [args]
    return [item for child in value.values() for item in knowledge_decisions(child)]


def write_json(path, value):
    """独占保存去密钥证据。参数：新路径/数据；返回：None；异常：文件已存在或序列化失败。"""
    with path.open("x", encoding="utf-8") as output:
        json.dump(canonical(value), output, ensure_ascii=False, indent=2, allow_nan=False)


class LiveReader:
    """只持有本轮数据库/HTTP资源；所有查询原件保存在独占输出目录。"""

    def __init__(self, config, base_url, output):
        """设置连接和输出。参数：Settings/本机API/独占目录；返回：None；异常：数据库配置错误。"""
        from app.db.session import create_database
        self.config, self.base_url, self.output = config, base_url, output
        self.engine, self.factory = create_database(config)

    async def close(self):
        """释放本轮连接。参数：无；返回：None；异常：资源关闭错误。"""
        await self.engine.dispose()

    async def api(self, path, *, params=None):
        """只允许应用GET。参数：路由/查询；返回：真实JSON；异常：HTTP错误。"""
        import httpx
        async with httpx.AsyncClient(base_url=self.base_url, timeout=70, trust_env=False, follow_redirects=False) as client:
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()

    async def turn(self, identity):
        """按显式本人原轮SELECT消息/问题池/计数。参数：身份；返回：带采集时间的快照；异常：不存在/数据库错误。"""
        from sqlalchemy import select
        from app.models.customer_service import Message
        from app.models.rag_quality import LowConfidenceQuestion
        from app.models.flywheel import ReviewQueue
        from app.repositories.chat import ChatRepository
        repository = ChatRepository(self.factory)
        saved = await repository.get_workflow_turn(identity["session_id"], identity["user_id"], identity["turn_id"])
        require(saved is not None, "原轮不存在")
        context = saved["context"]
        async with self.factory() as session:
            messages = (await session.scalars(select(Message).where(Message.conversation_id == context.conversation_id,
                Message.turn_id == context.turn_id).order_by(Message.id))).all()
            records = [{"id": row.id, "role": row.role, "turn_id": row.turn_id, "content": row.content,
                "turn_status": row.turn_status, "tool_calls": row.tool_calls, "tool_call_id": row.tool_call_id, "rag_metadata": row.rag_metadata} for row in messages]
            ids = {number for row in records if row["role"] == "user" for number in (row.get("rag_metadata") or {}).get("knowledge_pool", {}).get("pool_ids", {}).values()}
            pools = (await session.scalars(select(LowConfidenceQuestion).where(LowConfidenceQuestion.id.in_(ids)))).all() if ids else []
            pool = [{"id": row.id, "conversation_id": row.conversation_id, "raw_question": row.raw_question, "source": row.source,
                "reason": row.reason, "retrieved_chunks": row.retrieved_chunks, "matched_review_id": row.matched_review_id} for row in pools]
            review_ids = {row.matched_review_id for row in pools if row.matched_review_id is not None}
            reviews = (await session.scalars(select(ReviewQueue).where(ReviewQueue.id.in_(review_ids)))).all() if review_ids else []
        saved["context"] = context.model_dump(mode="json")
        return canonical({"captured_at": datetime.now(timezone.utc).isoformat(), "workflow": saved, "messages": records,
            "pool": pool, "review_counts": {str(row.id): row.occurrence_count for row in reviews}})

    async def observations(self, identity):
        """完整两条分页读含IO观测，保留每页且不忽略中间失败。参数：原轮身份；返回：观测列表；异常：超时/HTTP/游标错误。"""
        import httpx
        pages, cursors = [], set()
        # 当前v2服务端已移除parseIoAsJson=true；保留raw IO并在核验时本地解析。
        params = {"traceId": identity["trace_id"], "limit": 2,
            "fields": "core,basic,time,io,metadata,model,usage,metrics,trace_context"}
        auth = (self.config.langfuse_public_key, self.config.langfuse_secret_key.get_secret_value())
        async with asyncio.timeout(90), httpx.AsyncClient(base_url=self.config.langfuse_base_url, auth=auth,
                timeout=15, trust_env=False, follow_redirects=False) as client:
            while True:
                response = await client.get("/api/public/v2/observations", params=params)
                if response.is_error:
                    write_json(self.output / f"trace-{identity['id']}-http-error.json",
                        {"http_status": response.status_code, "received_pages": len(pages)})
                response.raise_for_status()
                page = response.json()
                for row in page["data"]:
                    (row.get("metadata") or {}).pop("scope.attributes.public_key", None)
                pages.append(page)
                # 每页收到即保存；后续页失败不会丢掉已经核实的前缀，但也不产出完整聚合件。
                write_json(self.output / f"trace-{identity['id']}-page-{len(pages)}.json", page)
                cursor = page["meta"].get("cursor")
                if not cursor:
                    break
                require(cursor not in cursors, "观测分页游标重复")
                cursors.add(cursor)
                params["cursor"] = cursor
        write_json(self.output / f"trace-{identity['id']}.json", {"pages": pages})
        return observation_rows({"pages": pages})

    async def publication(self, review_id):
        """回读审核API/实库与强一致向量指纹，不创建资源。参数：审核主键；返回：审核/知识/指纹；异常：不符或依赖失败。"""
        from sqlalchemy import select
        from app.models.knowledge import KnowledgeChunk
        from app.repositories.flywheel import FlywheelRepository
        from app.knowledge.hybrid_index import MilvusHybridIndex
        detail = canonical(await FlywheelRepository(self.factory, None).review_detail(review_id))
        api = await self.api(f"/api/flywheel/reviews/{review_id}")
        require(detail == api, "审核API与实库不符")
        async with self.factory() as session:
            rows = (await session.scalars(select(KnowledgeChunk).where(KnowledgeChunk.source_key == f"ch09/review/{review_id}"))).all()
            chunks = [{column.name: getattr(row, column.name) for column in KnowledgeChunk.__table__.columns} for row in rows]
        index = MilvusHybridIndex(self.config)
        try:
            await index.ensure_ready(create=False)
            fingerprints = await index.get_fingerprints([row["id"] for row in chunks])
        finally:
            await index.aclose()
        write_json(self.output / "publication.json", {"review": detail, "api": api, "chunks": chunks, "fingerprints": fingerprints})
        return detail, chunks, fingerprints

    async def evaluations(self, identifiers):
        """完整仓储分页读取指定评估行。参数：两个主键；返回：对应行；异常：缺失/重复分页。"""
        from app.repositories.evaluations import EvaluationRepository
        repository = EvaluationRepository(self.factory, None)
        all_rows, after = [], 0
        while True:
            page = await repository.list_runs(after_id=after, limit=50)
            all_rows.extend(page["items"])
            cursor = page["next_after_id"]
            if cursor is None:
                break
            require(int(cursor) > after, "评估分页游标未前进")
            after = int(cursor)
        write_json(self.output / "evaluation-database.json", all_rows)
        selected = [[row for row in all_rows if str(row["id"]) == str(identifier)] for identifier in identifiers]
        require(all(len(rows) == 1 for rows in selected), "评估实库行缺失或不唯一")
        return [rows[0] for rows in selected]


def finish_report(report):
    """按真实检查结果汇总，人工结论单列且不冒称机器确认。参数：报告；返回：补全报告；异常：无。"""
    cases = report.get("cases", [])
    complete = {row.get("id") for row in cases if row.get("status") == "passed" and isinstance(row.get("verified"), dict) and row["verified"]}
    pending = sorted(REQUIRED - complete)
    manual = report.get("manual") or {}
    manual["machine_verified"] = False
    if not manual.get("evidence_checked") or not manual.get("accepted"):
        pending.append("manual_browser_and_semantics")
    report.update(passed=not pending and not report.get("failures"), pending=pending, manual=manual)
    return report


def validate_local_url(value):
    """限制验收网络为本机服务。参数：URL；返回：去末尾斜杠URL；异常：ValueError。"""
    parsed = urlparse(value)
    require(parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
            and parsed.path in {"", "/"}, "验收端点必须是无凭据的本机根URL")
    return value.rstrip("/")


def manual_evidence(value, base, identities):
    """仅记录具名人工/浏览器观察，校验截图摘要与原轮绑定。参数：人工记录/目录/身份表；返回：独立结论；异常：ValueError。"""
    require(isinstance(value.get("reviewer"), str) and value["reviewer"].strip(), "人工观察缺少具名复核者")
    checks = value.get("checks", [])
    expected = {"langfuse", "knowledge_semantics", "feedback_ui", "review_ui", "eval_trend"}
    require(len(checks) == len(expected) and {row["id"] for row in checks} == expected, "人工观察项目缺失或重复")
    for row in checks:
        require(row.get("decision") in {"pass", "fail"} and bool(row.get("observation")) and bool(row.get("steps")), "缺少实际观察与操作步骤")
        require(isinstance(row.get("turns"), list) and all(name in identities for name in row["turns"]), "人工观察原轮身份未绑定")
        require(bool(row.get("screenshots")), "人工观察缺少截图")
        for reference in row["screenshots"]:
            require(set(reference) == {"path", "sha256"}, "截图引用字段不符")
            path = (base / reference["path"]).resolve()
            require(path.is_relative_to(REPO) and path.is_file(), "截图缺失或越界")
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == reference["sha256"] and raw.startswith(b"\x89PNG\r\n\x1a\n"), "截图指纹或PNG格式不符")
    return {"reviewer": value["reviewer"], "checks": checks, "evidence_checked": True,
        "accepted": all(row["decision"] == "pass" for row in checks), "machine_verified": False,
        "limitation": "仅核验人工记录身份/原件完整性；页面视觉和答案语义由具名复核者判定，API不能证明。"}


async def run_manifest(args, manifest, report):
    """只读取清单指定的真实身份并逐项计算结论。参数：CLI/清单/报告；返回：None；异常：初始配置或读取失败。"""
    from dotenv import dotenv_values
    from sqlalchemy import text
    from app.config import Settings
    from app.observability.costs import LangfuseObservationReader, get_intent_usage
    from app.knowledge.answering import FIXED_KNOWLEDGE_REFUSAL
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1, "清单版本非法")
    require(set(manifest).issubset({"schema_version", "database", "turns", "manual", *REQUIRED}), "清单含未知字段")
    turns = manifest.get("turns", [])
    identities = {}
    for identity in turns:
        require(set(identity) == {"id", "session_id", "user_id", "turn_id", "trace_id"}
                and all(isinstance(value, str) and value for value in identity.values()), "原轮身份字段非法")
        require(re.fullmatch(r"[a-z0-9_-]{1,40}", identity["id"]) and identity["id"] not in identities, "原轮别名重复或非法")
        identities[identity["id"]] = identity
    values = dotenv_values(REPO / ".env.langfuse")
    overrides = {"observability_enabled": True}
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        if values.get(name):
            overrides[name.lower()] = values[name]
    config = Settings(_env_file=REPO / ".env", **overrides)
    validate_local_url(config.langfuse_base_url)
    reader = LiveReader(config, args.base_url, args.output_dir)
    base = args.evidence_manifest.resolve().parent
    database, observations = {}, {}

    def imported(reference):
        """校验并复制原件，不覆盖已有摘要同一原件。参数：引用；返回：JSON；异常：引用错误。"""
        value = read_reference(reference, base, REPO)
        path = args.output_dir / f"imported-{reference['sha256']}.json"
        if not path.exists():
            path.write_bytes((base / reference["path"]).resolve().read_bytes())
        return value

    async def record(name, operation):
        """保留每项计算结果或失败，不把异常吞成通过。参数：项目/协程；返回：None；异常：控制取消透传。"""
        try:
            result = await operation()
            row = {"id": name, "status": "passed", "verified": canonical(result)}
        except Exception as error:
            # 连接错误可能带连接信息，仅安全的校验失败使用本工具固定消息。
            row = {"id": name, "status": "failed", "error_type": type(error).__name__,
                "reason": str(error) if isinstance(error, ValueError) else "真实依赖读取或原件解析失败"}
            report["failures"].append(name)
        report["cases"].append(row)
        write_json(args.output_dir / f"check-{name}.json", row)

    async def check(name):
        """执行单个具名证据合同。参数：必验项目；返回：计算结果；异常：缺项/不符/真实依赖失败。"""
        entry = manifest[name]
        if name == "knowledge_loop":
            review, chunks, fingerprints = await reader.publication(int(entry["review_id"]))
            return verify_knowledge_loop(database[entry["before"]], database[entry["after"]], review, chunks, fingerprints,
                imported(entry["approved_answer"]))
        if name in {"self_check", "mixed"}:
            state = database[entry]
            require(state["workflow"]["turn_status"] == "completed", "知识不足原轮未完成")
            sources = (source_message(state).get("rag_metadata") or {}).get("knowledge_pool", {}).get("pool_ids", {})
            source = "self_check" if name == "self_check" else next((key for key in sources if key in {"self_check", "retrieval_low_conf"}), None)
            pool = verify_pool(state, source)
            answer = state["workflow"]["answer"]
            require(answer.endswith(FIXED_KNOWLEDGE_REFUSAL), "知识不足未展示固定兜底")
            if name == "self_check":
                decisions = [decision for row in observations[entry] if row["type"] == "GENERATION" for decision in knowledge_decisions(row.get("output"))]
                require(any(item["useful"] is False for item in decisions), "缺少生产模型useful=false原始输出")
            else:
                from app.repositories.chat import ChatRepository
                state_input = {**state["workflow"]["completion_metadata"]["workflow"], "context": state["workflow"]["context"]}
                prefix = await ChatRepository(reader.factory).verified_business_prefix(state_input)
                require(bool(prefix) and answer == prefix + FIXED_KNOWLEDGE_REFUSAL, "业务前缀与原轮可信回执不符")
            return {"pool_id": pool["id"], "source": source, "turn_id": identities[entry]["turn_id"]}
        if name == "feedback":
            return verify_feedback(*(imported(entry[phase]) for phase in ("before", "after", "repeated")), database[entry["turn"]])
        if name == "replay":
            alias = entry["turn"]
            resumes = [row for row in observations[alias] if row.get("name") == "execution.resume" and row.get("endTime")]
            root = database[alias]["workflow"]["trace_identity"]["root_observation_id"]
            require(any(row.get("parentObservationId") == root for row in resumes), "恢复根未正确衔接原根")
            require(database[alias]["workflow"]["turn_status"] == "completed", "回放原轮未完成")
            return verify_replay(identities[alias], observation_rows(imported(entry["before"])), observation_rows(imported(entry["after"])),
                observations[alias], imported(entry["timing"]))
        if name == "concurrency":
            return verify_concurrency([observations[alias] for alias in entry])
        if name == "error":
            return verify_error(database[entry], observations[entry])
        if name == "costs":
            start, end = datetime.fromisoformat(entry["from"]), datetime.fromisoformat(entry["to"])
            paged = await get_intent_usage(start, end, reader=LangfuseObservationReader(config.model_copy(update={"observability_cost_page_size": 2})))
            api = await reader.api("/api/observability/costs", params={"from": start.isoformat(), "to": end.isoformat()})
            raw_reader = LangfuseObservationReader(config.model_copy(update={"observability_cost_page_size": 2}))
            raw = []
            async for row in raw_reader.iter_observations(start=start, end=end, observation_type="GENERATION"):
                (row.get("metadata") or {}).pop("scope.attributes.public_key", None)
                raw.append(row)
            write_json(args.output_dir / "cost-generations.json", {"data": raw, "reader_page_size": 2})
            write_json(args.output_dir / "costs.json", {"api": api, "page_size_2": paged})
            require(paged["coverage"]["complete"] and api == paged, "费用API与完整分页不符")
            require(api["coverage"]["model_call_count"] > 0 and all(row["cost"] is None for row in api["groups"]), "费用窗口为空或伪造金额")
            unique = {row["id"]: row for row in raw}
            known = sum(value for row in unique.values() for value in raw_usage(row) if value is not None)
            require(len(unique) == api["coverage"]["model_call_count"] and known == api["coverage"]["known_tokens"], "API数字与原始generation不符")
            return api
        if name == "evaluations":
            directories = [(base / item["artifact_dir"]).resolve() for item in entry]
            require(all(path.is_relative_to(REPO) for path in directories), "评估产物越出项目")
            rows = await reader.evaluations([item["id"] for item in entry])
            return verify_evaluation_pair(directories, rows)
        raise ValueError("未知验收项目")

    try:
        async with reader.factory() as session:
            db_name, db_version = (await session.execute(text("SELECT DATABASE(), VERSION()"))).one()
        require(db_name == config.mysql_database and ("database" not in manifest or manifest["database"] == db_name), "实际应用数据库与清单/配置不符")
        api = await reader.api("/openapi.json")
        report["runtime"] = {"database": db_name, "mysql_version": db_version, "application": api["info"], "base_url": args.base_url,
            "langfuse_base_url": config.langfuse_base_url, "mode": "GET/SELECT，仅复核既有真实请求"}
        for alias, identity in identities.items():
            async def collect(identity=identity, alias=alias):
                """采集当前别名的真实状态与观测。参数：本轮身份/别名；返回：核验结果；异常：读取/校验失败。"""
                evidence = await reader.turn(identity)
                write_json(args.output_dir / f"turn-{alias}.json", evidence)
                rows = await reader.observations(identity)
                result = verify_turn(identity, evidence, rows)
                database[alias], observations[alias] = evidence, rows
                return result
            await record(f"turn-{alias}", collect)
        for name in sorted(REQUIRED):
            if name in manifest:
                await record(name, lambda name=name: check(name))
        if "manual" in manifest:
            report["manual"] = manual_evidence(imported(manifest["manual"]), base, identities)
    finally:
        await reader.close()


def main(argv=None):
    """运行只读CLI；缺证据退出1，成功不代表机器替代人工语义判断。参数：可选argv；返回：退出码；异常：输出目录已存在。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(), "cases": [], "failures": []}
    try:
        args.base_url = validate_local_url(args.base_url)
        if args.evidence_manifest is None:
            report["failures"].append("缺少--evidence-manifest，未执行任何真实网络/数据库操作")
        else:
            manifest = json.loads(args.evidence_manifest.read_text(encoding="utf-8"))
            write_json(args.output_dir / "evidence-manifest.json", manifest)
            asyncio.run(run_manifest(args, manifest, report))
    except Exception as error:
        report["failures"].append({"type": type(error).__name__, "reason": str(error) if isinstance(error, ValueError) else "真实依赖或原件读取失败"})
    report["ended_at"] = datetime.now(timezone.utc).isoformat()
    report["evidence"] = [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in sorted(args.output_dir.iterdir()) if path.is_file()]
    write_json(args.output_dir / "summary.json", finish_report(report))
    print(json.dumps({"passed": report["passed"], "pending": report["pending"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
