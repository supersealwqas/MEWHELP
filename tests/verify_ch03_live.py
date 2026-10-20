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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.config import Settings
from app.core.llm import get_chat_model
from app.db.session import create_database
from app.knowledge.chunking import CHUNKING_VERSION
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

DEFAULT_CASES = Path("tests/fixtures/ch03")
DEFAULT_POLICY = Path("data/knowledge/retrieval-policy.json")
DEFAULT_REPORT = Path("dev-notes/ch03-evaluation.md")


@dataclass
class ScoredQuery:
    """带标注正例与真实 Dense 命中的校准查询。"""

    query_id: str
    positive_ids: list[int]
    expect_empty: bool
    hits: list[VectorHit]


def _normalize(value: str) -> str:
    """移除空白与 Markdown 强调标记，便于事实比较。参数：文本；返回：规范化文本；异常：无。"""
    return re.sub(r"\s+", "", value).replace("*", "").replace("_", "").replace("`", "")


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
        positive_set = set(query.positive_ids)
        matched = [hit.score for hit in top if hit.id in positive_set]
        if not matched:
            raise ValueError(f"正例未进入 Top3: {query.query_id}")
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
    if case.get("expect_empty") and observed.get("hits"):
        failures.append("期望空命中，实际返回知识")
    for fact in case.get("must_include", []):
        if _normalize(fact) not in answer:
            failures.append(f"缺少事实: {fact}")
    for alternatives in case.get("must_include_any", []):
        if not any(_normalize(option) in answer for option in alternatives):
            failures.append(f"缺少同义事实: {alternatives}")
    for fact in case.get("must_not_include", []):
        if _normalize(fact) in answer:
            failures.append(f"包含禁止结论: {fact}")
    return failures


def _config(test: bool) -> Settings:
    """加载测试配置；本工具必须使用 --test。参数：是否测试；返回：配置；异常：SystemExit。"""
    if not test:
        raise SystemExit("verify_ch03_live 只允许 --test，避免触碰应用库")
    config = Settings()
    if config.knowledge_test_database != "mewhelp_ch03_test":
        raise SystemExit("knowledge_test_database 不是 mewhelp_ch03_test")
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
    """按来源/章节映射标注正例主键。参数：知识块/案例；返回：主键；异常：无。"""
    source_key = f"doc:{case['source']}"
    return [
        chunk.id
        for chunk in chunks
        if chunk.is_active
        and chunk.source_key == source_key
        and chunk.section_path == case["section"]
    ]


async def _retrieve(jobs, query: str, limit: int = 48) -> list[VectorHit]:
    """执行真实 Dense 检索并回读校验。参数：jobs/查询/候选数；返回：有效命中；异常：依赖错误。"""
    vectors = await jobs.encoder.encode([query])
    hits = await jobs.index.search(vectors[0], limit)
    if not hits:
        return []
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
    return valid


async def _score_cases(jobs, cases: dict) -> list[ScoredQuery]:
    """构造真实 Dense 校准分数。参数：jobs/标注；返回：ScoredQuery；异常：依赖错误。"""
    chunks = await _chunks(jobs)
    scored: list[ScoredQuery] = []
    for case in cases["cases"]:
        hits = await _retrieve(jobs, case["query"])
        scored.append(
            ScoredQuery(
                query_id=case["id"],
                positive_ids=[] if case.get("expect_empty") else _positive_ids(chunks, case),
                expect_empty=bool(case.get("expect_empty")),
                hits=hits,
            )
        )
    return scored


async def _chat_answer(question: str, hits: list[dict]) -> str:
    """用真实聊天模型按检索命中生成验收回答。参数：问题/命中；返回：文本；异常：模型错误。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    model = get_chat_model(streaming=True, temperature=0.0)
    context = json.dumps(hits, ensure_ascii=False)
    messages = [
        SystemMessage(
            content=(
                "你是商城客服。只依据提供的知识命中回答；命中为空时明确没有查到相关政策，"
                "不得编造金额、包邮门槛、时效或商品功能。回答简洁，不要提内部检索。"
            )
        ),
        HumanMessage(content=f"用户问题：{question}\n知识命中：{context}"),
    ]
    stream = model.astream(messages)
    parts: list[str] = []
    try:
        async for chunk in stream:
            content = getattr(chunk, "content", "")
            if isinstance(content, str):
                parts.append(content)
    finally:
        await stream.aclose()
    return "".join(parts).strip()


def _report_markdown(title: str, payload: dict) -> str:
    """生成可供人工审阅的 Markdown 报告。参数：标题/数据；返回：正文；异常：无。"""
    return f"# {title}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"


async def calibrate(config: Settings, cases_path: Path, policy_path: Path, report_path: Path) -> dict:
    """执行真实校准并写入检索策略。参数：配置/案例/策略/报告；返回：结果；异常：校准错误。"""
    import_report = await _prepare_kb(config)
    async with knowledge_runtime(config) as jobs:
        cases = _load_json(cases_path / "calibration_cases.json")
        scored = await _score_cases(jobs, cases)
        threshold = choose_threshold(scored)
        digest = _digest(
            {
                "cases": cases,
                "scores": [
                    {
                        "id": item.query_id,
                        "positive_ids": item.positive_ids,
                        "expect_empty": item.expect_empty,
                        "hits": [hit.model_dump(mode="json") for hit in item.hits[:3]],
                    }
                    for item in scored
                ],
            }
        )
        policy = RetrievalPolicy(
            min_score=threshold,
            embedding_revision=jobs.encoder.revision,
            chunking_version=CHUNKING_VERSION,
            calibration_digest=digest,
        )
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(
            json.dumps(policy.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {
            "import": import_report,
            "threshold": threshold,
            "policy": policy.model_dump(mode="json"),
            "queries": [
                {
                    "id": item.query_id,
                    "positive_ids": item.positive_ids,
                    "expect_empty": item.expect_empty,
                    "hits": [hit.model_dump(mode="json") for hit in item.hits[:5]],
                }
                for item in scored
            ],
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown("CH03 校准报告", result), encoding="utf-8")
    return result


async def evaluate(config: Settings, cases_path: Path, policy_path: Path, report_path: Path) -> dict:
    """使用真实检索和聊天模型执行验收。参数：配置/案例/策略/报告；返回：结果；异常：依赖错误。"""
    cases = _load_json(cases_path / "acceptance_cases.json")
    async with knowledge_runtime(config) as jobs:
        policy = load_retrieval_policy(
            policy_path,
            embedding_revision=jobs.encoder.revision,
            chunking_version=CHUNKING_VERSION,
        )
        search = KnowledgeSearch(jobs.store, jobs.encoder, jobs.index, policy)
        results = []
        for case in cases["cases"]:
            hits = await search.query_faq(case["query"])
            answer = await _chat_answer(case["query"], hits)
            observed = {"hits": hits, "answer": answer}
            failures = validate_answer(case, observed)
            results.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "passed": not failures,
                    "failures": failures,
                    "hits": hits,
                    "answer": answer,
                }
            )
    report = {
        "passed": all(item["passed"] for item in results),
        "policy": policy.model_dump(mode="json"),
        "cases": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown("CH03 真实验收报告", report), encoding="utf-8")
    return report


async def recovery(config: Settings, report_path: Path) -> dict:
    """在真实 MySQL/Milvus 上演示向量写入后中断与重跑。参数：配置/报告；返回：结果；异常：依赖错误。"""
    source_key = "doc:recovery-live.md"
    async with knowledge_runtime(config) as jobs:
        draft = ChunkDraft(
            source_key=source_key,
            ordinal=0,
            text=KnowledgeText(
                category="恢复演示",
                questions="恢复演示",
                answer="这是一条用于中断恢复验证的临时知识。",
            ),
            section_path=None,
            content_type="faq",
            is_key_clause=False,
        )
        async with jobs.store.writer() as writer:
            frame = await writer.frame()
            published = await writer.publish(
                Publication(
                    expected_token=frame.token,
                    drafts=[draft],
                    embedding_revision=jobs.encoder.revision,
                )
            )
        identifier = published[0].id

        def checkpoint(label: str) -> None:
            """在真实向量写入后中断。参数：阶段；返回：None；异常：RuntimeError。"""
            if label == "after_upsert":
                raise RuntimeError("模拟真实向量写入后中断")

        interrupted = False
        try:
            async with jobs.store.writer() as writer:
                await jobs.synchronizer.run(writer, checkpoint=checkpoint)
        except RuntimeError:
            interrupted = True
        pending = (await jobs.store.get_chunks([identifier]))[0].vectorize_status
        query_vector = (
            await jobs.encoder.encode(["这是一条用于中断恢复验证的临时知识。"])
        )[0]
        hits = await jobs.index.search(query_vector, 10)
        assert interrupted and pending == "pending"
        assert any(hit.id == identifier for hit in hits)
        async with jobs.store.writer() as writer:
            rerun = await jobs.synchronizer.run(writer)
        done = (await jobs.store.get_chunks([identifier]))[0].vectorize_status
        assert done == "done" and rerun.marked_done >= 1
        # 清理本次临时来源，保留测试库可重复运行。
        async with jobs.store.writer() as writer:
            frame = await writer.frame()
            await writer.publish(
                Publication(
                    expected_token=frame.token,
                    drafts=[],
                    embedding_revision=jobs.encoder.revision,
                    replace_sources=[source_key],
                )
            )
            await jobs.synchronizer.run(writer)
        report = {
            "passed": True,
            "chunk_id": identifier,
            "interrupted_status": pending,
            "recovered_status": done,
            "rerun_marked_done": rerun.marked_done,
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report_markdown("CH03 中断恢复报告", report), encoding="utf-8")
    return report


async def _run(args: argparse.Namespace) -> dict:
    """执行 CLI 子命令。参数：参数；返回：结果；异常：任务错误。"""
    config = _config(args.test)
    cases_path = Path(args.cases)
    if args.command == "calibrate":
        return await calibrate(config, cases_path, Path(args.policy), Path(args.report))
    if args.command == "evaluate":
        return await evaluate(config, cases_path, Path(args.policy), Path(args.report))
    return await recovery(config, Path(args.report))


def main() -> None:
    """命令行入口。参数：无；返回：None；异常：任务错误。"""
    parser = argparse.ArgumentParser(description="CH03 真实校准/验收/恢复")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("calibrate", "重建测试知识库并校准检索阈值"),
        ("evaluate", "执行真实检索与聊天模型验收"),
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
