"""题集校验回归：用手工语料验证错误标注、事实泄漏及证据或/且关系。"""

import copy
import json
from pathlib import Path

import pytest

from app.knowledge.schemas import ChunkSnapshot, KnowledgeText


def _chunk(identifier: int, answer: str, source: str = "doc:manual.md") -> ChunkSnapshot:
    """构造手工证据块。参数：正整数主键、正文、来源；返回：快照；异常：模型校验失败。"""
    return ChunkSnapshot(id=identifier, source_key=source, ordinal=identifier - 1,
                         text=KnowledgeText(category="手册", questions="保修", answer=answer),
                         section_path="手册 / 保修", content_type="manual", content_hash="a" * 64,
                         embedding_revision="test")


def _case() -> dict:
    """给出独立手工题目。参数：无；返回：题目标注；异常：无。"""
    return {"id": "c-1", "split": "calibration", "fact_group": "warranty",
            "bucket": "A_policy", "difficulty": "medium", "query": "保修多久，怎么寄修？",
            "filters": {}, "answerable": True,
            "evidence_groups": [[{"source_key": "doc:manual.md", "section_path": "手册 / 保修", "quote": "保修十二个月"},
                                  {"source_key": "doc:faq.md", "section_path": "手册 / 保修", "quote": "保修一年"}],
                                 [{"source_key": "doc:manual.md", "section_path": "手册 / 保修", "quote": "联系客服寄修"}]],
            "required_facts": ["保修十二个月", "联系客服寄修"],
            "reference_answer": "保修十二个月，联系客服寄修。", "forbidden_claims": ["终身保修"]}


def _balanced_payload() -> list[dict]:
    """构造满足发布计数的手工题目。参数：无；返回：72题列表；异常：无。"""
    cases = []
    for split, count in (("calibration", 4), ("evaluation", 8)):
        for bucket in ("A_policy", "B_model", "C_colloquial", "D_filter", "E_multi", "F_unanswerable"):
            for index in range(count):
                case = _case()
                case.update(id=f"{split}-{bucket}-{index}", split=split,
                            fact_group=f"{split}-fact", bucket=bucket,
                            query=f"{split} {bucket} 场景 {index}")
                if bucket == "D_filter":
                    case["filters"] = {"content_type": "manual"}
                if bucket == "F_unanswerable":
                    case.update(answerable=False, evidence_groups=[], required_facts=[],
                                reference_answer="现有资料没有说明，无法确认，请联系客服核实。")
                cases.append(case)
    return cases


def _write(path: Path, payload: list[dict]) -> Path:
    """写入测试JSON。参数：目标路径和题目；返回：路径；异常：文件写入错误。"""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _corpus() -> list[ChunkSnapshot]:
    """提供三块手工语料。参数：无；返回：有两份同义证据的快照；异常：模型校验失败。"""
    return [_chunk(11, "保修十二个月。"), _chunk(32, "保修一年。", "doc:faq.md"),
            _chunk(73, "联系客服寄修。")]


def test_evidence_or_groups_and_independent_required_groups() -> None:
    """防止把可替代证据当成全部必需。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.dataset import EvalCase, resolve_evidence
    assert resolve_evidence(EvalCase.model_validate(_case()), _corpus()) == [{11, 32}, {73}]


def test_filter_lists_and_unanswerable_have_explicit_semantics() -> None:
    """防止列表筛选被误当标量或无答案题凭空产生证据。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.dataset import EvalCase, resolve_evidence
    case = _case()
    case["filters"] = {"source_key": ["doc:manual.md", "doc:faq.md"], "is_key_clause": [False]}
    assert resolve_evidence(EvalCase.model_validate(case), _corpus()) == [{11, 32}, {73}]
    case.update(answerable=False, bucket="F_unanswerable", evidence_groups=[], required_facts=[],
                reference_answer="资料未说明，无法确认。")
    assert resolve_evidence(EvalCase.model_validate(case), _corpus()) == []


@pytest.mark.parametrize("mutation", ["missing", "ambiguous", "inactive", "filtered", "duplicate_id"])
def test_resolver_rejects_unusable_evidence(mutation: str) -> None:
    """防止缺失或不唯一证据静默变为检索未命中。参数：错误场景；返回：None；异常：断言失败。"""
    from app.evaluation.dataset import DatasetError, EvalCase, resolve_evidence
    chunks = _corpus()
    case = _case()
    if mutation == "missing":
        chunks.pop()
    elif mutation == "ambiguous":
        chunks.append(_chunk(90, "保修十二个月。"))
    elif mutation == "inactive":
        chunks[0] = chunks[0].model_copy(update={"is_active": False})
    elif mutation == "filtered":
        case["filters"] = {"source_key": "doc:faq.md"}
    else:
        chunks.append(_chunk(11, "其他条款。"))
    with pytest.raises(DatasetError):
        resolve_evidence(EvalCase.model_validate(case), chunks)


def test_loader_accepts_balanced_hand_labeled_corpus(tmp_path: Path) -> None:
    """验证加载端到端而非只读取字段。参数：临时目录；返回：None；异常：断言失败。"""
    from app.evaluation.dataset import load_dataset
    cases = load_dataset(_write(tmp_path / "cases.json", _balanced_payload()), chunks=_corpus())
    assert len(cases) == 72
    assert sum(case.answerable for case in cases) == 60


@pytest.mark.parametrize("mutation", ["id", "count", "split_count", "bucket_count", "leak", "empty_answer",
                                       "answerable", "unanswerable_evidence", "empty_group", "unknown_filter",
                                       "filter_type", "filter_bucket", "unknown_bucket", "missing_evidence",
                                       "ambiguous_evidence", "empty_fact", "extra_field"])
def test_loader_rejects_broken_release(tmp_path: Path, mutation: str) -> None:
    """防止错误题集被用于校准或验收。参数：临时目录和错误类别；返回：None；异常：断言失败。"""
    from app.evaluation.dataset import DatasetError, load_dataset
    payload = copy.deepcopy(_balanced_payload())
    first = payload[0]
    if mutation == "id":
        payload[1]["id"] = first["id"]
    elif mutation == "count":
        payload.pop()
    elif mutation == "split_count":
        first["split"] = "evaluation"
    elif mutation == "bucket_count":
        first["bucket"] = "B_model"
    elif mutation == "leak":
        payload[24]["fact_group"] = first["fact_group"]
    elif mutation == "empty_answer":
        first["reference_answer"] = "  "
    elif mutation == "answerable":
        first["answerable"] = "true"
    elif mutation == "unanswerable_evidence":
        first["answerable"] = False
    elif mutation == "empty_group":
        first["evidence_groups"] = [[]]
    elif mutation == "unknown_filter":
        first["filters"] = {"chunk_id": 11}
    elif mutation == "filter_type":
        first["filters"] = {"is_key_clause": "true"}
    elif mutation == "filter_bucket":
        payload[12]["filters"] = {}
    elif mutation == "unknown_bucket":
        first["bucket"] = "unknown"
    elif mutation == "missing_evidence":
        first["evidence_groups"][0][0]["quote"] = "凭空出现的内容"
    elif mutation == "ambiguous_evidence":
        first["evidence_groups"][0][0]["quote"] = "。"
    elif mutation == "empty_fact":
        first["required_facts"] = [" "]
    else:
        first["unexpected"] = 1
    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path / "broken.json", payload), chunks=_corpus())


def test_published_dataset_resolves_against_real_character_chunks() -> None:
    """逐题以真实字符切分核验发布文件。参数：无；返回：None；异常：数据校验失败或断言失败。"""
    from app.evaluation.dataset import load_dataset
    path = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "ch04.json"
    cases = load_dataset(path)
    assert len(cases) == 72
    assert {case.split for case in cases} == {"calibration", "evaluation"}


def test_loader_accepts_variable_size_corpus_when_expected_size_is_none(tmp_path: Path) -> None:
    """扩展题集题量不固定，expected_size=None 时只要求每桶两种 split 都有题。

    参数：临时目录。
    返回：None。
    异常：AssertionError，变长题集被固定 72 题契约拒绝。
    """
    from app.evaluation.dataset import load_dataset

    payload = _balanced_payload()
    # 每桶再复制一题进校准集，凑成每桶 4 校准 + 8 固定以外的分布。
    extra = []
    for case in payload[:6]:
        clone = copy.deepcopy(case)
        clone["id"] = case["id"] + "-extra"
        clone["fact_group"] = case["fact_group"] + "-extra"
        clone["split"] = "calibration"
        extra.append(clone)
    variable = payload + extra

    cases = load_dataset(_write(tmp_path / "var.json", variable), chunks=_corpus(), expected_size=None)
    assert len(cases) == 78
    # 默认仍按 72 题契约校验，变长题集必须显式放开。
    with pytest.raises(Exception):
        load_dataset(_write(tmp_path / "var2.json", variable), chunks=_corpus())
