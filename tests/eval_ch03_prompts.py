"""CH03 抽取与关系 Prompt 标注评估工具。

以真实模型/编码器为默认运行路径，用独立标注样例检查来源、事实和关系，
同时记录候选 Dense 分数；只在标注正例可被保留时生成 RelationPolicy。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.core.llm import get_chat_model
from app.knowledge.embedding import BgeM3Encoder
from app.knowledge.llm import KnowledgeModel
from app.knowledge.grounding import GroundingError, turns_digest, validate_grounding
from app.knowledge.prompts import (
    EXTRACTION_PROMPT_VERSION,
    RELATION_PROMPT_VERSION,
)
from app.knowledge.schemas import (
    CompletedTurn,
    EvidenceItem,
    ExtractedQA,
    ExtractionBatch,
    RelationDecision,
    RelationPolicy,
)
from app.knowledge.text import format_knowledge

_POSITIVE_RELATIONS = {"same", "complementary", "conflict"}
DEFAULT_CASES = Path("tests/fixtures/ch03")
DEFAULT_REPORT = Path("dev-notes/ch03-prompts.json")
DEFAULT_POLICY = Path("data/knowledge/relation-policy.json")


def _normalize(value: str) -> str:
    """移除空白，便于比较模型措辞。参数：文本；返回：规范化文本；异常：无。"""
    return re.sub(r"\s+", "", value)


def _digest(payload: Any) -> str:
    """计算稳定 JSON 摘要。参数：可序列化对象；返回：SHA256；异常：无。"""
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cosine(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。参数：向量；返回：分数；异常：无。"""
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _load_json(path: Path) -> dict:
    """读取 JSON 文件。参数：路径；返回：对象；异常：OSError、JSONDecodeError。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _extraction_batch(extraction: dict) -> ExtractionBatch:
    """把抽取标注摊平为固定批次。参数：标注；返回：批次；异常：字段校验失败。"""
    turns = [
        CompletedTurn.model_validate(turn)
        for case in extraction["cases"]
        for turn in case["turns"]
    ]
    turns.sort(key=lambda turn: turn.message_id)
    identifiers = [turn.message_id for turn in turns]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("抽取标注 message_id 重复")
    payload_digest = _digest(extraction)
    return ExtractionBatch(
        batch_no=f"eval-{payload_digest[:12]}",
        source_ids=sorted(set(identifiers)),
        snapshot_upper_id=max(identifiers),
        input_hash=turns_digest(turns),
        prompt_version=EXTRACTION_PROMPT_VERSION,
        turns=turns,
    )


def _evaluate_extraction(extraction: dict, qas: list[ExtractedQA]) -> dict:
    """按标注检查抽取来源、问题原话和事实。参数：标注与 QA；返回：结果；异常：无。"""
    failures: list[str] = []
    case_results: list[dict] = []
    all_case_ids: set[int] = set()
    conversation_by_id: dict[int, int] = {}
    for case in extraction["cases"]:
        for turn in case["turns"]:
            all_case_ids.add(int(turn["message_id"]))
            conversation_by_id[int(turn["message_id"])] = int(turn["conversation_id"])
    for qa in qas:
        try:
            validate_grounding(qa, [CompletedTurn.model_validate(turn)
                                    for case in extraction["cases"] for turn in case["turns"]])
        except GroundingError as error:
            failures.append(f"{qa.question}: 原句证据核验失败: {error}")
        if not set(qa.source_ids).issubset(all_case_ids):
            failures.append(f"{qa.question}: 来源不在标注批次")
        conversations = {
            conversation_by_id[identifier]
            for identifier in qa.source_ids
            if identifier in conversation_by_id
        }
        if len(conversations) > 1:
            failures.append(f"{qa.question}: 抽取结果跨越会话")

    for case in extraction["cases"]:
        case_ids = {int(turn["message_id"]) for turn in case["turns"]}
        case_qas = [qa for qa in qas if set(qa.source_ids) & case_ids]
        case_failures: list[str] = []
        for qa in case_qas:
            if not set(qa.source_ids).issubset(case_ids):
                case_failures.append(f"{qa.question}: 来源跨越案例")
        if case.get("expect_empty"):
            if case_qas:
                case_failures.append(
                    "期望零结果，实际: " + "; ".join(qa.question for qa in case_qas)
                )
        else:
            expected = case.get("expected", [])
            if len(case_qas) != len(expected):
                case_failures.append(
                    f"抽取数量 {len(case_qas)} 与标注 {len(expected)} 不一致"
                )
            for item in expected:
                candidates = [
                    qa
                    for qa in case_qas
                    if qa.question == item["question"]
                    and set(qa.source_ids).issubset(case_ids)
                    and (
                        not item.get("source_ids")
                        or set(item["source_ids"]).issubset(qa.source_ids)
                    )
                ]
                if len(candidates) != 1:
                    case_failures.append(f"缺少唯一问题: {item['question']}")
                    continue
                answer = _normalize(candidates[0].answer)
                for fact in item.get("must_include", []):
                    if _normalize(fact) not in answer:
                        case_failures.append(f"缺少事实: {fact}")
                for fact in item.get("must_not_include", []):
                    if _normalize(fact) in answer:
                        case_failures.append(f"包含禁止结论: {fact}")
                keywords = item.get("category_keywords", [])
                if keywords and not any(word in candidates[0].category for word in keywords):
                    case_failures.append(f"分类不匹配: {candidates[0].category}")
        case_results.append(
            {
                "id": case["id"],
                "passed": not case_failures,
                "failures": case_failures,
            }
        )
        failures.extend(f"{case['id']}: {message}" for message in case_failures)
    return {
        "passed": not failures,
        "qa_count": len(qas),
        "extracted": [qa.model_dump(mode="json") for qa in qas],
        "cases": case_results,
        "failures": failures,
    }


async def _evaluate_relation(model, encoder, relation: dict) -> dict:
    """评估关系判定并计算可保留全部标注正例的最高候选阈值。"""
    items = [EvidenceItem.model_validate(item) for item in relation["items"]]
    refs = {item.ref for item in items}
    pairs = [(case["left"], case["right"]) for case in relation["cases"]]
    decisions = await model.relate(items, pairs)
    decision_map: dict[tuple[str, str], RelationDecision] = {}
    failures: list[str] = []
    for decision in decisions:
        pair = (decision.left_ref, decision.right_ref)
        if pair in decision_map:
            failures.append(f"重复关系 pair: {pair}")
        decision_map[pair] = decision
        if pair not in set(pairs):
            failures.append(f"未请求的关系 pair: {pair}")
        if decision.left_ref not in refs or decision.right_ref not in refs:
            failures.append(f"关系引用未知证据: {pair}")
        if not set(decision.evidence_refs).issubset(refs):
            failures.append(f"关系证据引用未知 ref: {pair}")

    case_results: list[dict] = []
    for case in relation["cases"]:
        pair = (case["left"], case["right"])
        decision = decision_map.get(pair)
        case_failures: list[str] = []
        if decision is None:
            case_failures.append("缺少关系结论")
        elif decision.relation not in case["expected"]:
            case_failures.append(
                f"关系 {decision.relation} 不在允许集合 {case['expected']}"
            )
        elif not decision.reason.strip():
            case_failures.append("关系理由为空")
        case_results.append(
            {
                "pair": pair,
                "expected": case["expected"],
                "actual": decision.relation if decision else None,
                "passed": not case_failures,
                "failures": case_failures,
            }
        )
        failures.extend(f"{pair[0]}|{pair[1]}: {message}" for message in case_failures)
    if set(decision_map) != set(pairs):
        failures.append("关系结果没有覆盖全部请求 pair")

    vectors = await encoder.encode([format_knowledge(item.text) for item in items])
    if len(vectors) != len(items):
        raise RuntimeError("编码器返回向量数量与证据不一致")
    vector_by_ref = {item.ref: vector for item, vector in zip(items, vectors)}
    scores = {
        f"{case['left']}|{case['right']}": _cosine(
            vector_by_ref[case["left"]], vector_by_ref[case["right"]]
        )
        for case in relation["cases"]
    }

    positive_scores = [
        scores[f"{case['left']}|{case['right']}"]
        for case in relation["cases"]
        if _POSITIVE_RELATIONS.intersection(case["expected"])
    ]
    policy = None
    if not positive_scores:
        failures.append("没有可用于校准的关系正例")
    else:
        threshold = min(positive_scores)
        if not math.isfinite(threshold):
            failures.append("校准阈值不是有限数值")
        else:
            calibration_digest = _digest(
                {
                    "cases": relation["cases"],
                    "scores": scores,
                    "decisions": [decision.model_dump(mode="json") for decision in decisions],
                }
            )
            policy = RelationPolicy(
                min_score=threshold,
                embedding_revision=encoder.revision,
                prompt_version=RELATION_PROMPT_VERSION,
                calibration_digest=calibration_digest,
            )
    return {
        "passed": not failures,
        "cases": case_results,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
        "failures": failures,
        "scores": scores,
        "threshold": min(positive_scores) if positive_scores else None,
        "policy": policy.model_dump(mode="json") if policy else None,
    }


async def evaluate_prompts(
    model: KnowledgeModel,
    encoder,
    cases_path: Path,
    *,
    suite: str = "all",
) -> dict:
    """运行抽取和/或关系标注评估。

    参数:
        model: 知识模型（真实或脚本化）。
        encoder: 具备 revision 与 encode 的编码器。
        cases_path: 含 extraction_cases.json/relation_cases.json 的目录。
        suite: all、extraction 或 relation。
    返回值:
        dict: 通过状态、逐例结果、关系分数与可选策略。
    异常:
        OSError、JSONDecodeError、模型/编码异常。
    """
    result: dict[str, Any] = {"cases_path": str(cases_path), "passed": False}
    extraction_result = None
    relation_result = None
    if suite in ("all", "extraction"):
        extraction = _load_json(cases_path / "extraction_cases.json")
        batch = _extraction_batch(extraction)
        qas = await model.extract(batch)
        extraction_result = _evaluate_extraction(extraction, qas)
        result["extraction"] = extraction_result
    if suite in ("all", "relation"):
        relation = _load_json(cases_path / "relation_cases.json")
        relation_result = await _evaluate_relation(model, encoder, relation)
        result["relation"] = relation_result
        result["policy"] = relation_result["policy"]

    parts = [part for part in (extraction_result, relation_result) if part is not None]
    result["passed"] = bool(parts) and all(part["passed"] for part in parts)
    if relation_result is not None and relation_result["policy"] is None:
        result["passed"] = False
    return result


async def _run(args: argparse.Namespace) -> dict:
    """执行评估并写报告/策略。参数：命令行参数；返回：结果；异常：模型/编码/IO 错误。"""
    config = Settings()
    encoder = BgeM3Encoder(config)
    try:
        result = await evaluate_prompts(
            KnowledgeModel(get_chat_model),
            encoder,
            Path(args.cases),
            suite=args.suite,
        )
    finally:
        await encoder.aclose()
    result["evaluation"] = {
        "chat_model": config.chat_model,
        "embedding_revision": encoder.revision,
        "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
        "relation_prompt_version": RELATION_PROMPT_VERSION,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 只发布整套固定标注已通过的策略；失败报告可保留候选，但不得覆盖旧有效文件。
    if result.get("passed") is True and result.get("policy"):
        policy_path = DEFAULT_POLICY
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(
            json.dumps(result["policy"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    """命令行入口。参数：无；返回：None；异常：参数或评估错误。"""
    parser = argparse.ArgumentParser(description="CH03 Prompt 标注评估")
    parser.add_argument("--suite", choices=["all", "extraction", "relation"], default="all")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    extraction = result.get("extraction") or {}
    relation = result.get("relation") or {}
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "extraction": extraction.get("passed"),
                "relation": relation.get("passed"),
                "threshold": relation.get("threshold"),
                "failures": (extraction.get("failures") or [])
                + (relation.get("failures") or []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
