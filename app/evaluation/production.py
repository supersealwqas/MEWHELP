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
        self.frozen_policy, self.checkpoint = None, None
        self.faithfulness_judge, self.last_judge_usage = None, None

    async def check_execution(self) -> None:
        """在下一阶段前核验执行锁。参数：无；返回：None；异常：失锁失败关闭。"""
        if self.checkpoint is not None:
            await self.checkpoint()

    def observe_judge_usage(self, record: dict) -> None:
        """保存本题裁判已取得真实用量。参数：record用量记录；返回：None；异常：无。"""
        self.last_judge_usage = record

    async def judge(self, query: str, answer: str, selected: list):
        """独立裁判完整selected，不写长期台账。参数：原问/答案/模型证据全集；返回：裁判模型；异常：模型或失锁错误。"""
        await self.check_execution()
        return await self.faithfulness_judge.judge(query, answer, selected)

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
            await self.check_execution()
            understood = await understand_query(self.model, model_messages(state, self.settings, UNDERSTANDING_PROMPT,
                [understanding_tool_schema()]), ledger)
            state.update(understood.model_dump())
            self.last_attempt["stage"] = "classify"
            await self.check_execution()
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
            await self.check_execution()
            prepared = await pipeline.prepare(understood.resolved_query, actual_filters, messages=[], budget=ledger)
            self.last_attempt.update(stage="retrieve", prepared=prepared)
            await self.check_execution()
            result = await pipeline.retrieve(prepared, capture=capture)
            result.update(route=route, filters=actual_filters, intent=intent.intent, intent_confidence=intent.confidence)
            self.last_attempt["result"] = result
            if not retrieval_only:
                self.last_attempt["stage"] = "generate"
                await self.check_execution()
                if self.frozen_policy is not None:
                    result["gate"] = assess_evidence(result["ranked"], result["selected"], self.frozen_policy).model_dump(mode="json")
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
        kwargs.setdefault("model", settings.chat_model)
        if observation.enabled:
            kwargs["callbacks"] = [observation.callback_handler]
        return get_chat_model(**kwargs)

    try:
        async with knowledge_storage(settings) as store:
            await index.ensure_ready(create=False)
            chunks = await read_corpus(store)
            await _verify_vectors(index, chunks)
            runtime = ProductionEvaluationRuntime(HybridKnowledgeSearch(store, encoder, index, reranker, settings), settings,
                model=WorkflowModelAdapter(model_factory=model_factory, stream_usage=True, intent_model=settings.intent_model),
                query_expander=QueryExpander(model_factory=model_factory, request_budget=settings.request_token_budget),
                observability=observation)
            from app.evaluation.faithfulness import FaithfulnessJudge
            runtime.faithfulness_judge = FaithfulnessJudge(model_factory,
                model_name=settings.judge_model or settings.chat_model, usage_observer=runtime.observe_judge_usage)
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


def _production_signature(settings) -> dict:
    """保存实际生产与测量模型协议，只绑定相关控制规则，不哈希整个应用。

    参数：本轮Settings；返回：可比较口径；异常：配置或序列化错误。
    """
    from app.evaluation.policy import digest_json
    from app.evaluation.faithfulness import _JUDGE_SYSTEM, FaithfulnessDecision
    from app.core.structured import output_tool
    from app.workflow.prompts import KNOWLEDGE_PROMPT
    from app.workflow.model import knowledge_tool_schema
    from app.knowledge.embedding import EMBEDDING_REVISION
    from app.knowledge.rerank import RERANKER_REVISION
    from app.evaluation.runtime import model_digest
    budget = BudgetLedger.from_settings(settings).snapshot()
    budget.pop("records", None)
    return {"protocols": production_protocols(settings),
        "models": {"generation": settings.chat_model, "intent": settings.intent_model,
            "judge": settings.judge_model or settings.chat_model, "embedding": EMBEDDING_REVISION, "reranker": RERANKER_REVISION,
            "endpoint_digest": sha256(settings.chat_base_url.encode()).hexdigest(),
            "embedding_artifacts_digest": model_digest(Path(settings.knowledge_model_path)),
            "reranker_artifacts_digest": model_digest(Path(settings.rerank_model_path))},
        "generation_protocol": digest_json({"protocol": "structured-selfcheck:full-evidence:one-citation-repair:v1",
            "prompt": KNOWLEDGE_PROMPT, "tool": knowledge_tool_schema()}),
        "judge_protocol": digest_json({"prompt": _JUDGE_SYSTEM,
            "tool": output_tool(FaithfulnessDecision, "submit_faithfulness"), "max_input_chars": 65000,
            "max_output_chars": 20000, "max_tokens": 4096, "temperature": 0.0}),
        "budget_config": budget,
        "retrieval_config": {key: getattr(settings, key) for key in ("knowledge_retrieval_mode", "recall_top_k",
            "rerank_top_k", "rerank_min_score", "subquery_split", "context_evidence_max_tokens",
            "knowledge_model_path", "rerank_model_path")}}


def _json_value(value):
    """将真实诊断模型集中转成标准JSON值，禁止隐藏非有限数字。

    参数：模型/字典/列表/基础值；返回：JSON兼容值；异常：写盘时遇非法值显式失败。
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _capture_result(row: dict, result: dict, groups: list[set[int]]) -> None:
    """保留同次已完成检索/生成，原题身份不被理解后的查询或路由筛选覆盖。

    参数：当前记录、共享执行部分结果、真实标注组；返回：None；异常：结果结构错误。
    """
    from app.evaluation.metrics import retrieval_metrics
    result = dict(result)
    if "query" in result:
        result["prepared_query"] = result.pop("query")
    if "filters" in result:
        result["actual_filters"] = result.pop("filters")
    row.update(_json_value(result))
    if "ranked" in result:
        row["ranked_ids"] = [item.chunk.id for item in result["ranked"]]
        row["retrieval"] = retrieval_metrics(row["ranked_ids"], groups)
        row["sufficient"] = selected_evidence_sufficient(groups, result["selected"])
    if row["refused"] is False:
        row["citation_valid"] = True


async def evaluate_production(args, settings, *, checkpoint) -> dict:
    """只读执行48题生产知识子流程，逐题写盘，末次版本核验后才发布完成标记。

    参数：完整CLI参数、冻结设置、各阶段执行lease核验协程。
    返回：完整汇总；异常：目录冲突、启动/版本/失锁/取消；逐题模型故障保留原分母。
    """
    from app.evaluation.dataset import load_dataset, resolve_evidence
    from app.evaluation.runner import atomic_json, production_summary, PRODUCTION_SCOPE
    from app.evaluation.policy import digest_json
    from app.evaluation.runtime import _verify_vectors
    from app.evaluation.faithfulness import FaithfulnessError
    from app.knowledge.errors import LeaseLost
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    identity, rows, metadata = str(uuid4()), [], None
    manifest = {"schema_version": 1, "artifact_id": identity, "scope": PRODUCTION_SCOPE,
        "status": "running", "expected_sample_count": 48, "completed_sample_count": 0}
    atomic_json(output / "manifest.json", manifest)
    invalidated = False
    try:
        await checkpoint()
        signature = await anyio.to_thread.run_sync(_production_signature, settings)
        policy = load_evidence_policy(args.policy, protocols=signature["protocols"])
        dataset_bytes = Path(args.dataset).read_bytes()
        with (output / "dataset.json").open("xb") as frozen_dataset:
            frozen_dataset.write(dataset_bytes)
            frozen_dataset.flush()
            os.fsync(frozen_dataset.fileno())
        await checkpoint()
        async with production_runtime(settings, target=args.target) as (runtime, store, chunks, index):
            runtime.frozen_policy, runtime.checkpoint = policy, checkpoint
            cases = load_dataset(output / "dataset.json", chunks=chunks)
            dataset = [case.model_dump(mode="json") for case in cases]
            cases = [case for case in cases if case.split == "evaluation"]
            corpus = [chunk.model_dump(mode="json") for chunk in chunks]
            before = {"knowledge": digest_json(corpus), "policy": digest_json(policy.model_dump(mode="json")),
                "protocols": signature["protocols"], "configuration": digest_json(signature)}
            metadata = {**signature, "dataset_digest": digest_json(dataset), "dataset_sha256": sha256(dataset_bytes).hexdigest(),
                "triggered_by": args.triggered_by,
                "split": "evaluation", "case_ids": [case.id for case in cases], "corpus_version": before["knowledge"],
                "policy_sha256": before["policy"], "policy_version": policy.version,
                "database": settings.mysql_database, "milvus_database": settings.milvus_database,
                "collection": settings.milvus_hybrid_collection, "versions_before": before, "versions_after": None}
            manifest["metadata"] = metadata
            atomic_json(output / "manifest.json", manifest)
            atomic_json(output / "corpus.json", corpus)
            atomic_json(output / "policy.json", policy.model_dump(mode="json"))
            with (output / "results.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for case in cases:
                    await checkpoint()
                    tick = monotonic()
                    groups = resolve_evidence(case, chunks)
                    row = {"artifact_id": identity, "id": case.id, "split": case.split, "query": case.query,
                        "filters": case.filters, "bucket": case.bucket, "difficulty": case.difficulty, "answerable": case.answerable,
                        "refused": None, "answer": None, "error": None, "faithfulness": None,
                        "supported_count": 0, "total_count": 0, "retrieval": None, "ranked_ids": [], "ranked": [],
                        "selected": [], "citations": [], "citation_valid": None, "budget": None, "judge_usage": None,
                        "resolved_evidence_groups": [sorted(group) for group in groups]}
                    runtime.last_attempt, runtime.last_judge_usage = {}, None
                    stage = "execute"
                    try:
                        result = await runtime.execute(case.query, case.filters, retrieval_only=False)
                        _capture_result(row, result, groups)
                        if row["refused"] is False:
                            stage = "judge"
                            await checkpoint()
                            judged = await runtime.judge(case.query, row["answer"], result["selected"])
                            numbers = {item.n for item in result["selected"]}
                            if not judged.facts or any(not set(fact.evidence_numbers).issubset(numbers)
                                    or (fact.supported and not fact.evidence_numbers) for fact in judged.facts):
                                raise FaithfulnessError("裁判事实为空或引用不合法")
                            row.update(judgment=judged.model_dump(mode="json"), total_count=len(judged.facts),
                                supported_count=sum(fact.supported for fact in judged.facts))
                            row["faithfulness"] = row["supported_count"] / row["total_count"]
                    except LeaseLost:
                        raise
                    except Exception as error:
                        attempt = runtime.last_attempt
                        if stage != "judge":
                            _capture_result(row, attempt.get("result", {}), groups)
                        ledger, capture = attempt.get("ledger"), attempt.get("capture")
                        if ledger is not None:
                            row["budget"] = ledger.snapshot()
                        if capture is not None and capture.snapshot is not None:
                            row["snapshot"] = capture.snapshot.model_dump(mode="json")
                        row["error"] = {"stage": "judge" if stage == "judge" else attempt.get("stage", stage),
                            "type": type(error).__name__, "kind": "budget" if isinstance(error,
                                (WorkflowBudgetExceeded, ContextBudgetExceeded)) else "fault"}
                    row["judge_usage"] = runtime.last_judge_usage
                    row["elapsed_seconds"] = monotonic() - tick
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    print(json.dumps({"id": row["id"], "error": row["error"], "completed": len(rows)}, ensure_ascii=False), flush=True)
            await checkpoint()
            # 末次版本无法确认时也不得发布完整成绩。
            invalidated = True
            after_chunks = await read_corpus(store)
            await _verify_vectors(index, after_chunks)
            after_signature = await anyio.to_thread.run_sync(_production_signature, settings)
            after_policy = load_evidence_policy(args.policy, protocols=after_signature["protocols"])
            metadata["versions_after"] = {"knowledge": digest_json([chunk.model_dump(mode="json") for chunk in after_chunks]),
                "policy": digest_json(after_policy.model_dump(mode="json")), "protocols": after_signature["protocols"],
                "configuration": digest_json(after_signature)}
            if metadata["versions_after"] != before or Path(args.dataset).read_bytes() != dataset_bytes:
                invalidated = True
                raise ValueError("运行期间知识、检索、策略或数据发生变化，本轮成绩失效")
            invalidated = False
            await checkpoint()
            summary = production_summary(rows, metadata, identity)
            atomic_json(output / "summary.json", summary)
            (output / "summary.md").write_text("# 生产知识评估\n\n" + summary["limitation"] +
                f"\n\n状态：{summary['status']}；题量：48；错误：{summary['error_count']}；拒答：{summary['refusal_count']}。\n" +
                "\n完整指标与分母见summary.json，逐题证据与真实用量见results.jsonl。\n", encoding="utf-8")
            atomic_json(output / "manifest.json", {**manifest, "status": summary["status"], "completed_sample_count": 48,
                "summary_digest": digest_json(summary), "results_digest": digest_json(rows)})
            return summary
    except BaseException as error:
        status = "invalidated" if invalidated else "failed" if isinstance(error, Exception) else "interrupted"
        atomic_json(output / "manifest.json", {**manifest, "status": status, "completed_sample_count": len(rows),
            "error_type": type(error).__name__})
        if metadata is not None:
            atomic_json(output / "summary.json", production_summary(rows, metadata, identity, status=status))
        raise


async def run_complete_production(args, settings) -> int:
    """CLI唯一付费执行者；先取得独立执行锁，完成文件后另用短保存锁。

    参数：CLI参数与明确应用设置；返回：0完成、2含逐题错误、3跨进程忙、4仅保存失败、1执行失败。
    异常：取消传播；不覆盖旧目录、不重新校准、不初始化应用知识资源。
    """
    from app.repositories.evaluations import evaluation_repository_runtime
    from app.knowledge.errors import ConcurrentKnowledgeWrite, LeaseLost
    try:
        async with evaluation_repository_runtime(settings) as repository:
            async with repository.execution() as lease:
                async def checkpoint():
                    """每阶段核验原物理执行连接，结束查询隐式事务。参数：无；返回：None；异常：LeaseLost。"""
                    try:
                        await lease.assert_held()
                        await lease.connection.commit()
                    except Exception as error:
                        raise LeaseLost("评估执行锁丢失") from error

                summary = await evaluate_production(args, settings, checkpoint=checkpoint)
                await checkpoint()
                try:
                    saved = await repository.save_completed_run(Path(args.output_dir), args.triggered_by)
                except Exception as error:
                    print(json.dumps({"status": "persist_failed", "error_type": type(error).__name__,
                        "artifact_dir": str(Path(args.output_dir).resolve())}, ensure_ascii=False), flush=True)
                    return 4
                print(json.dumps({"status": summary["status"], "id": saved["id"], "artifact_id": summary["artifact_id"]}, ensure_ascii=False))
                return 0 if summary["status"] == "completed" else 2
    except ConcurrentKnowledgeWrite:
        print(json.dumps({"status": "busy", "message": "应用数据库已有生产评估执行者"}, ensure_ascii=False))
        return 3
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}, ensure_ascii=False))
        return 1
