"""生产知识只读运行时与真实校准/验收报告；运行入口永不接收标注或写问题池。"""

from __future__ import annotations

from contextlib import asynccontextmanager, nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic
from uuid import uuid4

import anyio

from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
from app.core.memory import ContextBudgetExceeded
from app.knowledge.confidence import extract_features, load_evidence_policy, assess_evidence, evidence_policy_digest
from app.knowledge.pipeline import ProductionKnowledgePipeline, production_protocols, understand_query
from app.knowledge.snapshots import RetrievalCapture


def selected_evidence_sufficient(evidence_groups: list[set[int]], selected: list) -> bool:
    """仅runner按实际预算证据判断全部必需组覆盖，空组代表无充分证据。

    参数：完整语料解析后的必需组、实际selected。返回：严格充分性标签；异常：非法引用属性错误。
    """
    identifiers = {item.chunk_id for item in selected}
    return bool(evidence_groups) and all(bool(group & identifiers) for group in evidence_groups)


class ProductionEvaluationRuntime:
    """只接问题和过滤的单轮生产知识运行时，禁止读取答案或题目标注。"""

    def __init__(self, search, settings, *, model=None, query_expander=None, observability=None):
        """保存只读依赖。参数：搜索/配置/正式模型/扩写/可观测；返回：None；异常：无。"""
        from app.workflow.model import WorkflowModelAdapter
        from app.knowledge.query_rewrite import QueryExpander
        self.search, self.settings, self.observability = search, settings, observability
        self.model = model or WorkflowModelAdapter(stream_usage=True)
        self.query_expander = query_expander or QueryExpander(request_budget=settings.request_token_budget)
        self.last_attempt = {}

    async def execute(self, query: str, filters: dict, *, retrieval_only: bool) -> dict:
        """执行真实理解、分类、共享检索及可选知识生成。

        参数：query完整原问、filters真实筛选、retrieval_only是否只采样检索。
        返回：实际检索结果；完整模式追加answer/refused/citations/usage，不写聊天。
        异常：预算/依赖/参数错误原样传播。
        """
        from app.schemas.tools import ToolContext
        from app.workflow.state import new_turn_state
        from app.workflow.agent import model_messages
        from app.workflow.prompts import UNDERSTANDING_PROMPT, INTENT_PROMPT
        from app.workflow.model import understanding_tool_schema, intent_tool_schema
        from app.workflow.routing import route_by_intent
        from app.tools.knowledge import ORDER_POLICY_SOURCES
        turn_id = str(uuid4())
        context = ToolContext(session_id="production-evaluation", user_id="evaluation", conversation_id=1,
                              turn_id=turn_id, user_message=query)
        state, ledger = new_turn_state(context, [], 0), BudgetLedger.from_settings(self.settings)
        capture = RetrievalCapture(turn_id, sha256(query.encode()).hexdigest())
        self.last_attempt = {"stage": "understand", "ledger": ledger, "capture": capture}
        observed = self.observability.execution(self.observability.identity_for(context), kind="evaluation") if self.observability else nullcontext()
        with observed:
            understood = await understand_query(self.model, model_messages(state, self.settings, UNDERSTANDING_PROMPT,
                [understanding_tool_schema()]), ledger)
            state.update(understood.model_dump())
            self.last_attempt["stage"] = "classify"
            intent = await self.model.classify(model_messages(state, self.settings, INTENT_PROMPT, [intent_tool_schema()]), ledger)
            route = route_by_intent({**state, "intent": intent.intent})
            state.update(intent=intent.intent, route=route)
            policy = route == "order_policy"
            actual_filters = dict(filters)
            if policy:
                wanted = actual_filters.get("source_key", list(ORDER_POLICY_SOURCES))
                wanted = wanted if isinstance(wanted, list) else [wanted]
                actual_filters["source_key"] = [key for key in wanted if key in ORDER_POLICY_SOURCES] or ["evaluation:no-policy-source"]
            pipeline = ProductionKnowledgePipeline(self.search, self.settings, query_expander=self.query_expander,
                state=state, policy=policy, model=self.model)
            self.last_attempt["stage"] = "prepare"
            prepared = await pipeline.prepare(understood.resolved_query, actual_filters, messages=[], budget=ledger)
            self.last_attempt.update(stage="retrieve", prepared=prepared)
            result = await pipeline.retrieve(prepared, capture=capture)
            if not retrieval_only:
                self.last_attempt["stage"] = "generate"
                result.update(await pipeline.generate(prepared, result))
                result["budget"] = ledger.snapshot()
            return {**result, "route": route, "filters": actual_filters,
                    "intent": intent.intent, "intent_confidence": intent.confidence}


async def read_corpus(store) -> list:
    """分页只读当前全部有效块。参数：知识仓储；返回：快照；异常：数据库故障。"""
    chunks = []
    while page := await store.active_page(after_id=chunks[-1].id if chunks else 0, limit=500):
        chunks.extend(page)
    return chunks


def corpus_version(chunks: list) -> str:
    """绑定本次完整原文与真实主键/向量状态，独立于策略兼容性。

    参数：当前完整块；返回：SHA256；异常：序列化错误。
    """
    return sha256(json.dumps([row.model_dump(mode="json") for row in chunks], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def summary_markdown(summary: dict) -> str:
    """把同源报告转换为可核对的中文摘要，完整细节留在JSON及逐题原件。

    参数：校准或验收summary；返回：Markdown；异常：缺失必需指标时KeyError。
    """
    metadata = summary.get("metadata", {})
    lines = ["# CH09 生产证据策略", "", f"状态：`{summary['status']}`。", "",
        f"数据：{metadata.get('split')} / {metadata.get('case_count')} 题；知识版本 `{metadata.get('corpus_version')}`。", "",
        "|口径|平衡准确率|充分证据题|不充分题|误拦截|误放行|", "|---|---:|---:|---:|---:|---:|"]
    metrics = ([("候选策略", summary["chosen"]), ("最高分基线", summary["baseline"])] if "chosen" in summary else
               [("冻结策略独立验收", summary["metrics"])] if "metrics" in summary else [])
    for name, values in metrics:
        accuracy = values["balanced_accuracy"]
        lines.append(f"|{name}|{accuracy:.2%}|{values['positive_count']}|{values['negative_count']}|"
                     f"{len(values['false_block'])}|{len(values['false_pass'])}|" if accuracy is not None else f"|{name}|无足够标签|||||")
    lines += ["", "误拦截表示已有已标注充分证据却拒答；误放行表示本次实际证据没有覆盖全部必需组却通过。"]
    for name, values in metrics:
        lines += ["", f"{name}误拦截：{', '.join(values['false_block']) or '无'}。",
                  f"{name}误放行：{', '.join(values['false_pass']) or '无'}。"]
    if "stability" in summary:
        stability = summary["stability"]
        lines += ["", f"候选减基线的分层重采样95%区间：{stability['lower_95']:.2%} 至 {stability['upper_95']:.2%}。",
                  "", stability["limitation"], "", "规则差异样例："]
        lines += [f"- {row['id']}：{row['query']}；充分={row['sufficient']}，候选通过={row['candidate_passed']}，基线通过={row['baseline_passed']}。"
                  for row in summary["disagreements"]]
    diagnostics = summary.get("execution_diagnostics")
    if diagnostics:
        lines += ["", f"已知实际用量：{diagnostics['known_actual_tokens']} tokens；用量缺失请求：{diagnostics['unknown_usage_records']}。",
                  f"理解/查询优化回退记录：{len(diagnostics['model_fallbacks'])}；详见summary.json执行诊断，不能将回退掩盖为全程无模型故障。"]
    lines += ["", "本次只验证真实检索与实际预算证据，不包含最终答案生成和忠实度。标签只验证已标注证据覆盖，新增等价证据须人工复核。",
              "", "逐题原文、查询、Top5快照、预算及真实用量见 results.jsonl；参数、版本、误判与比较详情见 summary.json。", ""]
    return "\n".join(lines)


def execution_diagnostics(rows: list[dict]) -> dict:
    """汇总真实usage和已保存优化回退，不让成功检索掩盖先前模型故障。

    参数：本轮原始逐题记录；返回：已知用量、未知请求和回退清单；异常：坏记录结构错误。
    """
    known, unknown, fallbacks = 0, 0, []
    for row in rows:
        for record in (row.get("budget") or {}).get("records", []):
            value = (record.get("usage") or {}).get("total_tokens")
            if type(value) is int:
                known += value
            else:
                unknown += 1
            failure = record.get("failure")
            if failure:
                fallbacks.append({"id": row["id"], "stage": record["stage"], "failure": failure})
    return {"known_actual_tokens": known, "unknown_usage_records": unknown, "model_fallbacks": fallbacks}


@asynccontextmanager
async def production_runtime(settings, *, target: str):
    """组装应用只读知识依赖，绝不初始化或迁移应用数据库。

    参数：settings明确配置、target必须application-readonly。返回：runtime/store/chunks/index。
    异常：索引/模型/数据库错误；退出屏蔽取消并关闭本次资源。
    """
    if target != "application-readonly" or settings.knowledge_retrieval_mode != "hybrid":
        raise ValueError("生产校准必须明确选择application-readonly及已配置混合检索")
    from app.knowledge.runtime import knowledge_storage
    from app.knowledge.embedding import BgeM3Encoder
    from app.knowledge.rerank import BgeReranker
    from app.knowledge.hybrid_index import MilvusHybridIndex
    from app.knowledge.retrieval import HybridKnowledgeSearch
    from app.knowledge.query_rewrite import QueryExpander
    from app.workflow.model import WorkflowModelAdapter
    from app.observability.runtime import ObservabilityRuntime
    from app.core.llm import get_chat_model
    from app.evaluation.runtime import _verify_vectors
    encoder, reranker, index = BgeM3Encoder(settings), BgeReranker(settings), MilvusHybridIndex(settings)
    observation = ObservabilityRuntime(settings)

    def model_factory(**kwargs):
        """给本次真实模型装一次后台观测回调并请求真实usage。参数：模型选项；返回：模型；异常：配置错误。"""
        kwargs.setdefault("stream_usage", True)
        if observation.enabled:
            kwargs["callbacks"] = [observation.callback_handler]
        return get_chat_model(**kwargs)

    try:
        async with knowledge_storage(settings) as store:
            await index.ensure_ready(create=False)
            chunks = await read_corpus(store)
            await _verify_vectors(index, chunks)
            runtime = ProductionEvaluationRuntime(HybridKnowledgeSearch(store, encoder, index, reranker, settings), settings,
                model=WorkflowModelAdapter(model_factory=model_factory, stream_usage=True),
                query_expander=QueryExpander(model_factory=model_factory, request_budget=settings.request_token_budget),
                observability=observation)
            yield runtime, store, chunks, index
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await reranker.aclose()
            finally:
                try:
                    await encoder.aclose()
                finally:
                    try:
                        await index.aclose()
                    finally:
                        await observation.aclose()


async def run_production(args, settings) -> int:
    """运行固定24校准或48独立验收，逐题保留证据与故障，未支持策略不发布。

    参数：CLI参数与Settings。返回：0通过、1故障、2需要人类取舍。
    异常：目录已存在时FileExistsError；不覆盖任何旧报告或知识资源。
    """
    from app.evaluation.dataset import load_dataset, resolve_evidence
    from app.evaluation.runner import atomic_json
    from app.evaluation.calibration import calibrate_evidence_policy, decision_metrics
    from app.evaluation.runtime import _verify_vectors
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    calibration = args.command == "calibrate-production"
    split = "calibration" if calibration else "evaluation"
    protocols = production_protocols(settings)
    rows = []
    try:
        policy = None if calibration else load_evidence_policy(args.policy, protocols=protocols)
        async with production_runtime(settings, target=args.target) as (runtime, store, chunks, index):
            cases = load_dataset(args.dataset, chunks=chunks)
            cases = [case for case in cases if case.split == split]
            metadata = {"target": args.target, "split": split, "case_count": len(cases),
                "protocols": protocols, "feature_protocol": "evidence-v2", "metrics_version": "selected-sufficiency-v1",
                "dataset_sha256": sha256(Path(args.dataset).read_bytes()).hexdigest(), "corpus_version": corpus_version(chunks),
                "database": settings.mysql_database, "milvus_database": settings.milvus_database,
                "collection": settings.milvus_hybrid_collection, "chat_model": settings.chat_model,
                "intent_model": settings.intent_model, "policy_version": policy.version if policy else None,
                "policy_sha256": evidence_policy_digest(policy) if policy else None,
                "scope": "single-turn-production-knowledge; no business execution or final-answer generation"}
            atomic_json(output / "manifest.json", {"status": "running", **metadata})
            atomic_json(output / "corpus.json", [row.model_dump(mode="json") for row in chunks])
            with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
                for case in cases:
                    start = monotonic()
                    groups = resolve_evidence(case, chunks)
                    row = {"id": case.id, "split": split, "query": case.query, "filters": case.filters,
                           "bucket": case.bucket, "resolved_evidence_groups": [sorted(group) for group in groups], "error": None}
                    try:
                        result = await runtime.execute(case.query, case.filters, retrieval_only=True)
                        ranked, selected = result.pop("ranked"), result.pop("selected")
                        snapshot = result.pop("snapshot")
                        # minimum=0仅用于保存原始Top分/数量；拟合时逐个实际候选门槛重算有效数。
                        features = extract_features(ranked, selected, minimum_score=policy.minimum_score if policy else 0.0)
                        by_id = {item.chunk.id: item for item in ranked}
                        row.update(result, features=features.model_dump(), feature_minimum_score=policy.minimum_score if policy else 0.0,
                            ranked=[item.model_dump(mode="json") for item in ranked],
                            selected=[item.model_dump(mode="json") for item in selected],
                            selected_scores=[by_id[item.chunk_id].score for item in selected],
                            sufficient=selected_evidence_sufficient(groups, selected),
                            snapshot=snapshot.model_dump(mode="json") if snapshot else None)
                        if policy:
                            row["gate"] = assess_evidence(ranked, selected, policy).model_dump(mode="json")
                    except Exception as error:
                        attempt = runtime.last_attempt
                        capture = attempt.get("capture")
                        row.update(error={"stage": attempt.get("stage", "startup"), "type": type(error).__name__,
                                   "kind": "budget" if isinstance(error, (ContextBudgetExceeded, WorkflowBudgetExceeded)) else "fault"},
                            budget=attempt["ledger"].snapshot() if attempt.get("ledger") else None,
                            snapshot=capture.snapshot.model_dump(mode="json") if capture and capture.snapshot else None,
                            sufficient=None)
                    row["elapsed_seconds"] = monotonic() - start
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(json.dumps({"id": row["id"], "error": row["error"], "seconds": round(row["elapsed_seconds"], 2)}, ensure_ascii=False), flush=True)
            after = await read_corpus(store)
            await _verify_vectors(index, after)
            if corpus_version(after) != metadata["corpus_version"]:
                raise ValueError("采样期间生产语料发生变化，本轮不能拟合或验收")
            metadata["corpus_version_after"] = corpus_version(after)
            if calibration:
                summary = calibrate_evidence_policy(rows, metadata)
                if summary.get("candidate_policy"):
                    atomic_json(output / "candidate-policy.json", summary["candidate_policy"])
                if summary["policy"]:
                    destination = Path(settings.evidence_policy_path)
                    if destination.exists():
                        raise FileExistsError("正式证据策略已存在，禁止校准隐式覆盖冻结策略")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(destination, summary["policy"])
                    atomic_json(output / "policy.json", summary["policy"])
            else:
                valid = [row for row in rows if row["error"] is None]
                summary = {"status": "completed" if len(valid) == 48 else "completed_with_errors", "metadata": metadata,
                    "metrics": decision_metrics(valid, [row["gate"]["passed"] for row in valid]),
                    "errors": [{"id": row["id"], "error": row["error"]} for row in rows if row["error"]],
                    "limitation": "48题只运行一次，不用于调参；标签仅验证已标注证据覆盖，新增等价证据需人工复核。"}
            summary["execution_diagnostics"] = execution_diagnostics(rows)
            atomic_json(output / "summary.json", summary)
            (output / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
            atomic_json(output / "manifest.json", {"status": summary["status"], **metadata})
            print(json.dumps({"status": summary["status"], "output_dir": str(output.resolve())}, ensure_ascii=False))
            return 0 if summary["status"] in {"supported", "completed"} else 2
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "completed_rows": len(rows)}
        atomic_json(output / "startup-error.json", failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 1
