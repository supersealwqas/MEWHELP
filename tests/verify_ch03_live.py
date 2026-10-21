"""CH03 真实校准、验收与中断恢复工具。

默认必须使用 `--test`，只操作 mewhelp_ch03_test 与 Milvus 测试库；真实
BGE-M3 编码、Milvus 检索、MySQL 权威回读和项目聊天模型均实际调用。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.config import Settings
from app.db.session import create_database
from app.knowledge.chunking import CHUNKING_VERSION, chunk_markdown
from app.knowledge.manifest import load_manifest
from app.knowledge.embedding import EMBEDDING_REVISION
from app.knowledge.runtime import initialize_knowledge, knowledge_runtime
from app.knowledge.schemas import (
    ChunkDraft,
    KnowledgeText,
    Publication,
    RetrievalPolicy,
    VectorHit,
)
from app.knowledge.search import KnowledgeSearch, load_retrieval_policy
from app.knowledge.vector import MilvusVectorIndex
from app.knowledge.text import format_knowledge, content_fingerprint
from app.knowledge.search import KnowledgeSearchUnavailable
from tests.ch03_live_runtime import live_runtime, tcp_server, sample_exchange

DEFAULT_CASES = Path("tests/fixtures/ch03")
DEFAULT_POLICY = Path("data/knowledge/retrieval-policy.json")
DEFAULT_REPORT = Path("dev-notes/ch03-evaluation.md")


@dataclass
class ScoredQuery:
    """带标注正例与真实 Dense 命中的校准查询。"""

    query_id: str  # 固定校准案例标识。
    positive_ids: list[int]  # 无分组时逐个必须覆盖的知识主键。
    expect_empty: bool  # 是否为无知识命中的负例。
    hits: list[VectorHit]  # 经MySQL权威回读后的有序候选。
    support_groups: dict[str, list[int]] = field(default_factory=dict)  # 每个必要事实的可替代支持主键。


def _normalize(value: str) -> str:
    """规范格式但不拼接分隔数字。参数：文本；返回：规范化文本；异常：无。"""
    value = value.replace("*", "").replace("_", "").replace("`", "").lower()
    value = re.sub(r"(?<=\d)\s+(?=\d)", "\x00", value)
    return re.sub(r"\s+", "", value).replace("\x00", " ")


def _contains(answer: str, fact: str) -> bool:
    """按完整数字边界匹配事实。参数：规范文本/事实；返回：是否匹配；异常：无。"""
    fact = _normalize(fact)
    pattern = re.escape(fact)
    if fact and fact[0].isdigit():
        pattern = r"(?<![\d.])" + pattern
    if fact and fact[-1].isdigit():
        pattern += r"(?![\d.])"
    return bool(re.search(pattern, answer))


def choose_threshold(scores: list[ScoredQuery]) -> float:
    """选择能保留全部正例且排除负例的最高阈值。

    参数:
        scores: 真实 Dense 分数与标注。
    返回值:
        float: 校准阈值。
    异常:
        ValueError: 正例未召回或正负例分数无法分开。
    """
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for query in scores:
        top = query.hits[:3]
        if query.expect_empty:
            if top:
                negative_scores.append(max(hit.score for hit in top))
            continue
        groups = query.support_groups or {str(identifier): [identifier] for identifier in query.positive_ids}
        if not groups:
            raise ValueError(f"缺少必要事实标注: {query.query_id}")
        for name, identifiers in groups.items():
            matched = [hit.score for hit in top if hit.id in identifiers]
            if not matched:
                raise ValueError(f"必要事实未进入 Top3: {query.query_id}/{name}")
            positive_scores.append(max(matched))
    if not positive_scores:
        raise ValueError("没有可用于校准的 Dense 正例")
    threshold = min(positive_scores)
    if negative_scores and max(negative_scores) >= threshold:
        raise ValueError("正负例分数重叠，无法生成可信阈值")
    if not math.isfinite(threshold):
        raise ValueError("校准阈值不是有限数值")
    return float(threshold)


def validate_answer(case: dict, observed: dict) -> list[str]:
    """按标注检查答案事实与空命中。参数：案例/观察结果；返回：失败列表；异常：无。"""
    failures: list[str] = []
    answer = _normalize(str(observed.get("answer") or ""))
    if case.get("answer_rule", case.get("id")) == "nest_cleaning":
        # 固定清洗标注接受同义否定，不把“可以”从其他句子任意删掉。
        answer = answer.replace("内部垫芯不可以机洗", "内部垫芯不可机洗")
    if case.get("expect_empty") and observed.get("hits"):
        failures.append("期望空命中，实际返回知识")
    for fact in case.get("must_include", []):
        if not _contains(answer, fact):
            failures.append(f"缺少事实: {fact}")
    for alternatives in case.get("must_include_any", []):
        if not any(_contains(answer, option) for option in alternatives):
            failures.append(f"缺少同义事实: {alternatives}")
    for fact in case.get("must_not_include", []):
        if _contains(answer, fact):
            failures.append(f"包含禁止结论: {fact}")
    rule = case.get("answer_rule", case.get("id"))
    if rule == "shipping":
        clauses = r"[^。；;，,]{0,35}"
        if "实付" not in answer:
            failures.append("缺少实付金额门槛条件")
        if not re.search(r"(?:未满|不足|不满)" + clauses + r"(?<!\d)6元", answer):
            failures.append("缺少未满门槛收6元基础运费的绑定事实")
        if not re.search(r"(?<![未不])满99元" + clauses + r"(?:包邮|免(?:基础)?运费)", answer):
            failures.append("缺少实付满99元包邮的绑定事实")
        if not re.search(r"(?:偏远|附加)" + clauses + r"(?<!\d)12元", answer):
            failures.append("缺少偏远12元附加费的绑定事实")
    if rule == "member_discount":
        for member, discount in (("银卡", "9"), ("金卡", "95")):
            match = re.search(member + r"[^，,。；;\n]{0,24}(?<!\d)" + discount + "折", answer)
            if not match or re.search(r"不享|不打|不是|不能|没有", match.group()):
                failures.append(f"缺少绑定事实: {member}{discount}折")
    if rule == "lite_app":
        # 三项否定必须分别存在；同一并列否定可以支持多个功能，转折不能继承否定。
        lite_answer = "。".join(sentence for sentence in re.split(r"[。；;]", answer)
                               if "pro" not in sentence or "lite" in sentence)
        for feature in ("app联网", "远程监控", "健康记录"):
            if not re.search(r"(?:不含|不支持|没有|无|不能)[^。；;，,但]{0,30}" + feature, lite_answer):
                failures.append(f"缺少Lite否定事实: {feature}")
    if rule == "refund_processing":
        if not re.search(r"退款[^。；;]{0,25}(?:处理|审核)[^。；;]{0,15}(?<!\d)3个工作日", answer):
            failures.append("缺少退款处理时限3个工作日的绑定事实")
    if rule in ("refund_processing", "refund_arrival"):
        if re.search(r"(?:\d+|[一二三四五六七八九十]+)(?:个)?(?:工作日|天|小时)[^。；;，,]{0,8}到账", answer):
            # 允许明确否定具体承诺的澄清，不用裸到账禁词误伤。
            for sentence in re.split(r"[。；;，,]", answer):
                if re.search(r"(?:\d+|[一二三四五六七八九十]+)(?:个)?(?:工作日|天|小时).{0,8}到账", sentence) and not re.search(r"不承诺|不能保证|并非|不代表|不是", sentence):
                    failures.append("把具体时长承诺为退款到账")
    if rule == "negative_weather":
        for clause in re.split(r"[。；;，,]", answer):
            forecast = re.search(r"(?:今天|明天).{0,12}(?:下雨|下雪|晴|阴|多云)|\d+(?:%|℃|度)", clause)
            uncertain = re.search(r"无法|不能|不知道|不清楚|未能|没查到|不具备|会不会|是否|吗", clause)
            if forecast and not uncertain:
                failures.append("无资料负例编造天气结论")
    return failures


def _config(test: bool) -> Settings:
    """加载测试配置；本工具必须使用 --test。参数：是否测试；返回：配置；异常：SystemExit。"""
    if not test:
        raise SystemExit("verify_ch03_live 只允许 --test，避免触碰应用库")
    config = Settings()
    if (
        config.knowledge_test_database != "mewhelp_ch03_test"
        or config.milvus_test_database != "mewhelp_ch03_test"
    ):
        raise SystemExit("CH03 测试库必须是 mewhelp_ch03_test")
    return config.model_copy(
        update={
            "mysql_database": config.knowledge_test_database,
            "milvus_database": config.milvus_test_database,
        }
    )


def _load_json(path: Path) -> dict:
    """读取 JSON。参数：路径；返回：对象；异常：OSError/JSONDecodeError。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: Any) -> str:
    """计算稳定摘要。参数：对象；返回：SHA256；异常：无。"""
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


async def _clean_test_knowledge(config: Settings) -> None:
    """重建测试库知识表并清空测试 Milvus 向量。参数：测试配置；返回：None；异常：资源错误。"""
    await initialize_knowledge(config)
    engine, _ = create_database(config)
    try:
        async with engine.begin() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != "mewhelp_ch03_test":
                raise ValueError("拒绝清理非CH03测试库")
            await connection.execute(text("DELETE FROM qa_extraction_staging"))
            await connection.execute(text("DELETE FROM qa_extraction_batches"))
            await connection.execute(
                text(
                    "UPDATE knowledge_chunks SET prev_chunk_id = NULL, next_chunk_id = NULL"
                )
            )
            await connection.execute(text("DELETE FROM knowledge_chunks"))
    finally:
        await engine.dispose()
    index = MilvusVectorIndex(config)
    try:
        await index.ensure_ready(create=True)
        async for ids in index.iter_ids(1000):
            await index.delete(ids)
    finally:
        await index.aclose()


async def _prepare_kb(config: Settings) -> dict:
    """清空测试库并导入六份权威资料。参数：测试配置；返回：导入报告；异常：导入错误。"""
    await _clean_test_knowledge(config)
    async with knowledge_runtime(config) as jobs:
        report = await jobs.import_documents(Path("data/kb/manifest.json"))
        return report.model_dump(mode="json")


async def _chunks(jobs) -> list:
    """读取当前知识块。参数：jobs；返回：快照；异常：存储错误。"""
    async with jobs.store.writer() as writer:
        frame = await writer.frame()
        return frame.chunks


def _positive_ids(chunks, case: dict) -> list[int]:
    """按逐事实正文映射支持主键。参数：知识块/案例；返回：主键；异常：标注错误。"""
    return sorted({identifier for ids in _support_groups(chunks, case).values() for identifier in ids})


def _support_groups(chunks, case: dict, facts: dict | None = None) -> dict[str, list[int]]:
    """每个必要子事实独立绑定权威支持块。参数：快照/案例/事实表；返回：支持组；异常：标注缺失。"""
    facts = facts or _load_json(DEFAULT_CASES / "facts.json")
    by_id = {fact["id"]: fact for fact in facts["cases"]}
    groups = {}
    for fact_id in case.get("required_facts", []):
        fact = by_id[fact_id]
        fragments = fact.get("support_groups", [[fragment] for fragment in fact["must_include"]])
        for ordinal, group in enumerate(fragments):
            groups[f"{fact_id}:{ordinal}"] = [chunk.id for chunk in chunks
                if chunk.is_active and chunk.source_key == f"doc:{fact['source']}"
                and chunk.section_path == fact["section"]
                and all(_contains(_normalize(chunk.text.answer), fragment) for fragment in group)]
    return groups


async def _retrieve(jobs, query: str, limit: int = 48) -> dict:
    """执行实际keyword的只读分数探针。参数：依赖/查询/候选数；返回：raw与valid；异常：依赖错误。"""
    vectors = await jobs.encoder.encode([query])
    hits = await jobs.index.search(vectors[0], limit)
    if not hits:
        return {"raw": [], "valid": []}
    rows = await jobs.store.get_chunks([hit.id for hit in hits])
    by_id = {row.id: row for row in rows}
    valid: list[VectorHit] = []
    for hit in hits:
        row = by_id.get(hit.id)
        if (
            row is None
            or not row.is_active
            or row.vectorize_status != "done"
            or row.vector_id != str(row.id)
            or row.content_hash != hit.content_hash
            or row.embedding_revision != hit.embedding_revision
        ):
            continue
        valid.append(hit)
    return {"raw": hits, "valid": valid}


async def _observe_cases(config: Settings, cases: dict, facts: dict, policy: RetrievalPolicy) -> list[dict]:
    """逐例走原HTTP工具链并串联分数证据。参数：配置/标注/事实/策略；返回：逐例证据；异常：初始化错误。"""
    results = []
    async with live_runtime(config, policy) as runtime:
        chunks = await _chunks(runtime)
        by_id = {chunk.id: chunk for chunk in chunks}
        async with tcp_server(runtime) as url:
            for case in cases["cases"]:
                observed = await sample_exchange(runtime, url, case)
                observed["exchange_failures"] = list(observed["failures"])
                groups = _support_groups(chunks, case, facts)
                observed.update(id=case["id"], query=case["query"], support_groups=groups)
                keyword = observed["executed_keyword"]
                if keyword is None and case.get("expect_empty") and not observed["exchange_failures"]:
                    keyword = case["query"]
                    observed["probe_kind"] = "无工具负例只读Dense探针"
                elif keyword is not None:
                    observed["probe_kind"] = "实际执行keyword只读Dense探针"
                else:
                    observed["probe_kind"] = "未成功执行，不采Dense分数"
                if keyword is not None:
                    try:
                        scores = await _retrieve(runtime, keyword)
                        valid = scores["valid"]
                        top = [hit for hit in valid if hit.score >= policy.min_score][:3]
                        observed["raw_candidates"] = [hit.model_dump(mode="json") for hit in scores["raw"]]
                        observed["valid_candidates"] = [hit.model_dump(mode="json") for hit in valid]
                        observed["valid_top3"] = [hit.model_dump(mode="json") for hit in valid[:3]]
                        observed["filtered_top3"] = [hit.model_dump(mode="json") for hit in top]
                        expected_hits = [{"id": hit.id, "question": by_id[hit.id].text.questions,
                                          "answer": by_id[hit.id].text.answer, "category": by_id[hit.id].text.category} for hit in top]
                        if observed["effective_keyword"] is not None and observed["hits"] != expected_hits:
                            observed["failures"].append("实际工具hits与同keyword权威回读不一致")
                    except Exception as error:
                        observed["failures"].append(f"Dense证据读取异常: {type(error).__name__}")
                hit_ids = {hit["id"] for hit in observed["hits"] if isinstance(hit, dict) and "id" in hit}
                observed["fact_coverage"] = {name: bool(hit_ids.intersection(ids)) for name, ids in groups.items()}
                if not case.get("expect_empty") and (not groups or not all(observed["fact_coverage"].values())):
                    observed["failures"].append("实际工具Top3未覆盖全部必要事实")
                observed["failures"].extend(validate_answer(case, observed))
                observed["passed"] = not observed["failures"]
                results.append(observed)
    return results


def _report_markdown(title: str, payload: dict) -> str:
    """生成可供人工审阅的 Markdown 报告。参数：标题/数据；返回：正文；异常：无。"""
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


async def calibrate(config: Settings, cases_path: Path, policy_path: Path, report_path: Path) -> dict:
    """真实工具keyword校准后重跑正式SSE。参数：配置/标注/策略/报告；返回：结果；异常：失败写证据后返回。"""
    result = {"passed": False, "mode": "真实TCP SSE校准", "failures": []}
    try:
        result["import"] = await _prepare_kb(config)
        cases = _load_json(cases_path / "calibration_cases.json")
        facts = _load_json(cases_path / "facts.json")
        diagnostic = RetrievalPolicy(min_score=-1, embedding_revision=EMBEDDING_REVISION,
            chunking_version=CHUNKING_VERSION, calibration_digest=_digest(cases))
        observed = await _observe_cases(config, cases, facts, diagnostic)
        result["diagnostic_policy"] = diagnostic.model_dump(mode="json")
        result["diagnostic_cases"] = observed
        scored = []
        for case, item in zip(cases["cases"], observed):
            # 正例必须实际成功执行FAQ；诊断答案/宽阈值负例失败不冒称正式验收。
            exchange_failures = item.get("exchange_failures")
            if exchange_failures is None:
                raise ValueError(f"校准缺少链路证据: {case['id']}")
            if case.get("expect_empty"):
                # 低阈值诊断允许合法负例带命中用于取排除分数，但不允许网络或工具错误。
                exchange_failures = [failure for failure in exchange_failures
                                     if failure != "实际hits与正负例预期不一致"]
            if exchange_failures or (not case.get("expect_empty") and not item.get("executed_keyword")):
                raise ValueError(f"校准不能使用失败或未执行的工具申请: {case['id']}")
            scored.append(ScoredQuery(case["id"], [], bool(case.get("expect_empty")),
                [VectorHit.model_validate(hit) for hit in item.get("valid_candidates", [])], item["support_groups"]))
        threshold = choose_threshold(scored)
        policy = RetrievalPolicy(min_score=threshold, embedding_revision=EMBEDDING_REVISION,
            chunking_version=CHUNKING_VERSION, calibration_digest=_digest({"cases": cases, "facts": facts, "evidence": observed}))
        result["policy"] = policy.model_dump(mode="json")
        result["cases"] = await _observe_cases(config, cases, facts, policy)
        result["passed"] = bool(result["cases"]) and all(item["passed"] for item in result["cases"])
        if result["passed"]:
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.write_text(json.dumps(result["policy"], ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            result["failures"].append("正式策略校准回归未全部通过，保留旧策略")
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}" if isinstance(error, ValueError)
                                  else f"校准依赖异常: {type(error).__name__}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown("CH03 真实校准报告", result), encoding="utf-8")
    return result


async def evaluate(config: Settings, cases_path: Path, policy_path: Path, report_path: Path) -> dict:
    """对未参与调参的固定措辞执行完整SSE验收。参数：配置/标注/策略/报告；返回：结果；异常：失败写证据后返回。"""
    report = {"passed": False, "mode": "独立固定问法真实TCP SSE", "failures": []}
    try:
        cases = _load_json(cases_path / "acceptance_cases.json")
        policy = load_retrieval_policy(policy_path, embedding_revision=EMBEDDING_REVISION, chunking_version=CHUNKING_VERSION)
        report["policy"] = policy.model_dump(mode="json")
        report["cases"] = await _observe_cases(config, cases, _load_json(cases_path / "facts.json"), policy)
        report["passed"] = bool(report["cases"]) and all(item["passed"] for item in report["cases"])
    except Exception as error:
        report["failures"].append(f"验收依赖异常: {type(error).__name__}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown("CH03 独立真实SSE验收报告", report), encoding="utf-8")
    return report


def validate_document_snapshot(drafts: list, chunks: list) -> list[str]:
    """核对已导入语料与源文档。参数：当前切分/库快照；返回：失败项；异常：无。"""
    expected = sorted((draft.source_key, draft.ordinal, draft.section_path or "", content_fingerprint(draft.text)) for draft in drafts)
    active = [row for row in chunks if row.is_active]
    actual = sorted((row.source_key, row.ordinal, row.section_path or "", row.content_hash) for row in active)
    failures = []
    if expected != actual:
        failures.append("已导入知识的来源/顺序/章节/内容hash与当前源文档不一致")
    if any(row.vectorize_status != "done" or row.vector_id != str(row.id)
           or row.embedding_revision != EMBEDDING_REVISION for row in active):
        failures.append("已导入知识未全部达到当前编码版本active/done")
    return failures


async def pipeline(config: Settings, policy_path: Path, report_path: Path) -> dict:
    """只验证当前真实聊天链路，不评检索或回答质量。参数：配置/交付策略/报告；返回：证据；异常：失败记类型。"""
    import httpx

    report = {"passed": False, "scope": "链路验证，检索和生成质量暂缓", "failures": []}
    try:
        policy = load_retrieval_policy(policy_path, embedding_revision=EMBEDDING_REVISION, chunking_version=CHUNKING_VERSION)
        report["policy"] = policy.model_dump(mode="json")
        async with live_runtime(config, policy) as runtime:
            drafts = []
            for spec in load_manifest(ROOT / "data/kb/manifest.json", root=ROOT / "data/kb"):
                drafts.extend(chunk_markdown(spec.path.read_text(encoding="utf-8"), spec, runtime.encoder.count_tokens))
            chunks = await _chunks(runtime)
            snapshot_failures = validate_document_snapshot(drafts, chunks)
            report["snapshot"] = {"active_count": sum(row.is_active for row in chunks),
                "expected_count": len(drafts), "failures": snapshot_failures,
                "rows": [{"id": row.id, "source_key": row.source_key, "content_hash": row.content_hash,
                          "vector_id": row.vector_id, "status": row.vectorize_status} for row in chunks if row.is_active]}
            report["failures"].extend(snapshot_failures)
            if snapshot_failures:
                raise ValueError("语料快照未就绪")
            async with tcp_server(runtime) as url:
                report["cases"] = []
                for case in ({"id": "legal_faq", "query": "邮费", "expect_empty": False},
                             {"id": "no_tool_greeting", "query": "你好", "expect_empty": True}):
                    observed = await sample_exchange(runtime, url, case)
                    observed.update(id=case["id"], query=case["query"], passed=not observed["failures"])
                    report["cases"].append(observed)
                async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                    response = await client.post(url + "/api/chat", json={"session_id": "ch03-invalid-empty", "message": " "})
                    report["request_rejection"] = {"http_status": response.status_code, "body": response.json(),
                                                   "passed": response.status_code == 422}
                report["passed"] = (all(case["passed"] for case in report["cases"])
                                    and report["request_rejection"]["passed"] and not report["failures"])
    except Exception as error:
        report["failures"].append(f"链路验证异常: {type(error).__name__}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown("CH03 当前链路真实验证", report), encoding="utf-8")
    return report

def validate_recovery_snapshots(snapshots: dict) -> list[str]:
    """检查恢复前后不变量。参数：阶段快照；返回：失败项；异常：无。"""
    failures = []
    phases = ("pending", "recovered", "missing", "repaired")
    if any(phase not in snapshots for phase in phases):
        return ["恢复证据缺阶段"]
    original = snapshots["pending"]
    identifier = original["id"]
    for phase in phases:
        current = snapshots[phase]
        for name in ("id", "content_hash", "encoding_hash", "embedding_revision"):
            if current.get(name) != original.get(name):
                failures.append(f"{phase}改变{name}")
        if not current.get("is_active"):
            failures.append(f"{phase}丢失active资格")
        expected_status = "pending" if phase == "pending" else "done"
        if current.get("vectorize_status") != expected_status:
            failures.append(f"{phase}状态错误")
        if phase != "pending" and current.get("vector_id") != str(identifier):
            failures.append(f"{phase}vector_id错误")
        if (identifier in current.get("raw_ids", [])) != (phase != "missing"):
            failures.append(f"{phase}实际向量存在性错误")
        if (identifier in current.get("visible_ids", [])) != (phase in {"recovered", "repaired"}):
            failures.append(f"{phase}在线可见性错误")
    return failures


async def _recovery_snapshot(jobs, identifier: int, query: str) -> dict:
    """记录原文、编码输入及查询状态。参数：依赖/主键/原查询；返回：快照；异常：依赖错误。"""
    row = (await jobs.store.get_chunks([identifier]))[0]
    result = row.model_dump(mode="json")
    result["encoding_hash"] = hashlib.sha256(format_knowledge(row.text).encode("utf-8")).hexdigest()
    vector = (await jobs.encoder.encode([query]))[0]
    result["raw_ids"] = [hit.id for hit in await jobs.index.search(vector, 48)]
    result["query"] = query
    search = KnowledgeSearch(jobs.store, jobs.encoder, jobs.index, RetrievalPolicy(
        min_score=-1, embedding_revision=jobs.encoder.revision, chunking_version=CHUNKING_VERSION,
        calibration_digest="recovery-observation-only"))
    try:
        result["visible_ids"] = [hit["id"] for hit in await search.query_faq(query)]
    except KnowledgeSearchUnavailable:
        result["visible_ids"] = []
        result["query_error"] = "KnowledgeSearchUnavailable"
    return result


async def recovery(config: Settings, report_path: Path) -> dict:
    """验证真实upsert中断与done缺向量补齐。参数：测试配置/报告；返回：逐阶段证据；异常：失败记录类型。"""
    from uuid import uuid4

    source_key = f"doc:recovery-live-{uuid4().hex}.md"
    query = "这是一条用于中断恢复验证的临时知识。"
    report = {"passed": False, "source_key": source_key, "snapshots": {}, "failures": []}
    try:
        async with knowledge_runtime(config) as jobs:
            try:
                draft = ChunkDraft(source_key=source_key, ordinal=0,
                    text=KnowledgeText(category="恢复演示", questions="恢复演示", answer=query),
                    section_path=None, content_type="faq", is_key_clause=False)
                async with jobs.store.writer() as writer:
                    frame = await writer.frame()
                    published = await writer.publish(Publication(expected_token=frame.token,
                        drafts=[draft], embedding_revision=jobs.encoder.revision))
                identifier = published[0].id

                def checkpoint(label: str) -> None:
                    """在真实向量写入后中断。参数：阶段；返回：None；异常：RuntimeError模拟故障。"""
                    if label == "after_upsert":
                        raise RuntimeError("模拟真实向量写入后中断")

                interrupted = False
                try:
                    async with jobs.store.writer() as writer:
                        await jobs.synchronizer.run(writer, checkpoint=checkpoint)
                except RuntimeError as error:
                    if str(error) != "模拟真实向量写入后中断":
                        raise
                    interrupted = True
                if not interrupted:
                    raise ValueError("没有发生预期的after_upsert中断")
                report["snapshots"]["pending"] = await _recovery_snapshot(jobs, identifier, query)
                async with jobs.store.writer() as writer:
                    report["rerun"] = (await jobs.synchronizer.run(writer)).model_dump(mode="json")
                report["snapshots"]["recovered"] = await _recovery_snapshot(jobs, identifier, query)
                await jobs.index.delete([identifier])
                report["snapshots"]["missing"] = await _recovery_snapshot(jobs, identifier, query)
                async with jobs.store.writer() as writer:
                    report["repair"] = (await jobs.synchronizer.run(writer)).model_dump(mode="json")
                report["snapshots"]["repaired"] = await _recovery_snapshot(jobs, identifier, query)
                report["failures"].extend(validate_recovery_snapshots(report["snapshots"]))
            finally:
                # 只撤回本次唯一临时source，其余51块演示知识保持可用。
                async with jobs.store.writer() as writer:
                    frame = await writer.frame()
                    await writer.publish(Publication(expected_token=frame.token, drafts=[],
                        embedding_revision=jobs.encoder.revision, replace_sources=[source_key]))
                    await jobs.synchronizer.run(writer)
                report["cleanup"] = "只清理本次临时source"
            report["passed"] = not report["failures"]
    except Exception as error:
        report["failures"].append(f"恢复验证失败: {type(error).__name__}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown("CH03 真实中断与缺向量恢复报告", report), encoding="utf-8")
    return report

async def _run(args: argparse.Namespace) -> dict:
    """执行 CLI 子命令。参数：参数；返回：结果；异常：任务错误。"""
    config = _config(args.test)
    cases_path = Path(args.cases)
    if args.command == "calibrate":
        return await calibrate(config, cases_path, Path(args.policy), Path(args.report))
    if args.command == "evaluate":
        return await evaluate(config, cases_path, Path(args.policy), Path(args.report))
    if args.command == "pipeline":
        return await pipeline(config, Path(args.policy), Path(args.report))
    return await recovery(config, Path(args.report))


def main() -> None:
    """命令行入口。参数：无；返回：None；异常：任务错误。"""
    parser = argparse.ArgumentParser(description="CH03 真实校准/验收/恢复")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("calibrate", "重建测试知识库并校准检索阈值"),
        ("evaluate", "执行真实检索与聊天模型验收"),
        ("pipeline", "核对已导入语料并验证真实聊天链路，不评质量"),
        ("recovery", "演示真实向量中断恢复"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--test", action="store_true", help="必须只使用 CH03 测试库")
        command.add_argument("--cases", default=str(DEFAULT_CASES))
        command.add_argument("--policy", default=str(DEFAULT_POLICY))
        command.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("passed", True) else 1)


if __name__ == "__main__":
    main()
