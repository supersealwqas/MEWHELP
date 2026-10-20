"""Prompt 标注评估工具的假判定红—绿测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.knowledge.llm import KnowledgeModelError, load_relation_policy
from app.knowledge.prompts import RELATION_PROMPT_VERSION
from app.knowledge.schemas import ExtractedQA, RelationDecision
from tests.eval_ch03_prompts import evaluate_prompts
from tests.knowledge.fakes import FakeEncoder, ScriptedKnowledgeModel

_TEMP_BASE = Path(".pytest-tmp")

_EXTRACTION = {
    "cases": [
        {
            "id": "shipping",
            "turns": [
                {
                    "message_id": 1,
                    "conversation_id": 1,
                    "turn_id": "t1",
                    "question": "邮费怎么算",
                    "answer": "未满99元收6元。",
                }
            ],
            "expect_empty": False,
            "expected": [
                {
                    "source_ids": [1],
                    "question": "邮费怎么算",
                    "must_include": ["6 元"],
                    "must_not_include": ["10 元"],
                    "category_keywords": ["物流"],
                }
            ],
        }
    ]
}

_RELATION = {
    "items": [
        {
            "ref": "a",
            "origin": "document",
            "text": {"category": "分类", "questions": "问法", "answer": "答案一。"},
            "source_refs": ["doc:a"],
        },
        {
            "ref": "b",
            "origin": "conversation",
            "text": {"category": "分类", "questions": "问法", "answer": "答案二。"},
            "source_refs": ["qa:1"],
        },
    ],
    "cases": [{"left": "a", "right": "b", "expected": ["conflict"]}],
}


def _cases_dir():
    """创建包含假标注的临时目录。参数：无；返回：上下文管理器；异常：无。"""
    _TEMP_BASE.mkdir(parents=True, exist_ok=True)
    manager = tempfile.TemporaryDirectory(prefix="ch03-eval-", dir=_TEMP_BASE)
    root = Path(manager.name)
    (root / "extraction_cases.json").write_text(
        json.dumps(_EXTRACTION, ensure_ascii=False), encoding="utf-8"
    )
    (root / "relation_cases.json").write_text(
        json.dumps(_RELATION, ensure_ascii=False), encoding="utf-8"
    )
    return manager, root


def _qa(answer: str = "未满99元收6元。", source_ids: list[int] | None = None) -> ExtractedQA:
    """构造测试 QA。参数：答案与来源；返回：QA；异常：校验失败。"""
    return ExtractedQA(
        source_ids=source_ids or [1],
        question="邮费怎么算",
        answer=answer,
        category="物流",
    )


def _decision(relation: str = "conflict") -> RelationDecision:
    """构造测试关系决策。参数：关系；返回：决策；异常：校验失败。"""
    return RelationDecision(
        left_ref="a",
        right_ref="b",
        relation=relation,
        evidence_refs=["a", "b"],
        reason="测试理由",
    )


async def test_good_scripted_model_passes_and_generates_policy() -> None:
    """正确脚本模型通过并生成版本匹配的 RelationPolicy。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    manager, root = _cases_dir()
    try:
        model = ScriptedKnowledgeModel(
            extract_qas=[_qa()],
            relation_decisions=[_decision()],
        )
        result = await evaluate_prompts(model, FakeEncoder(), root)
        assert result["passed"] is True
        assert result["policy"]["min_score"] == 1.0
        assert result["policy"]["embedding_revision"] == "fake-encoder-v1"
        assert result["policy"]["prompt_version"] == RELATION_PROMPT_VERSION
    finally:
        manager.cleanup()


@pytest.mark.parametrize(
    "qa",
    [_qa(source_ids=[999]), _qa(answer="未满99元收10元。")],
    ids=["unknown-source", "forbidden-fact"],
)
async def test_missing_source_or_forbidden_fact_fails(qa: ExtractedQA) -> None:
    """未知来源或禁止事实不能被评估器放过。

    参数:
        qa: 错误 QA。
    返回值:
        None
    异常:
        无
    """
    manager, root = _cases_dir()
    try:
        model = ScriptedKnowledgeModel(
            extract_qas=[qa],
            relation_decisions=[_decision()],
        )
        result = await evaluate_prompts(model, FakeEncoder(), root)
        assert result["passed"] is False
        assert result["extraction"]["failures"]
    finally:
        manager.cleanup()


@pytest.mark.parametrize(
    "decisions",
    [[_decision("same")], []],
    ids=["wrong-relation", "missing-decision"],
)
async def test_wrong_or_missing_relation_fails(
    decisions: list[RelationDecision],
) -> None:
    """错误或缺失的关系结论不能通过。

    参数:
        decisions: 错误关系结果。
    返回值:
        None
    异常:
        无
    """
    manager, root = _cases_dir()
    try:
        model = ScriptedKnowledgeModel(
            extract_qas=[_qa()],
            relation_decisions=decisions,
        )
        result = await evaluate_prompts(model, FakeEncoder(), root)
        assert result["passed"] is False
        assert result["relation"]["failures"]
    finally:
        manager.cleanup()


async def test_load_relation_policy_versions() -> None:
    """策略版本不匹配时拒绝使用。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    manager, root = _cases_dir()
    try:
        model = ScriptedKnowledgeModel(
            extract_qas=[_qa()],
            relation_decisions=[_decision()],
        )
        result = await evaluate_prompts(model, FakeEncoder(), root)
        policy_path = root / "relation-policy.json"
        policy_path.write_text(
            json.dumps(result["policy"], ensure_ascii=False), encoding="utf-8"
        )
        policy = load_relation_policy(
            policy_path,
            embedding_revision="fake-encoder-v1",
            prompt_version=RELATION_PROMPT_VERSION,
        )
        assert policy.min_score == 1.0
        with pytest.raises(KnowledgeModelError):
            load_relation_policy(
                policy_path,
                embedding_revision="other-revision",
                prompt_version=RELATION_PROMPT_VERSION,
            )
    finally:
        manager.cleanup()
