"""CH04 标注集加载、发布约束与稳定证据解析。

题集保存来源键、完整章节与逐字原文片段，不保存数据库主键；加载时使用
真实字符切分结果核对证据。可替代证据在同组内取并集，必需事实组分别保留。
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator

from app.knowledge.documents import build_document_chunks
from app.knowledge.manifest import load_manifest
from app.knowledge.schemas import ChunkSnapshot

NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Bucket = Literal["A_policy", "B_model", "C_colloquial", "D_filter", "E_multi", "F_unanswerable"]
FilterValue = str | bool | list[str] | list[bool]
BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_filter", "E_multi", "F_unanswerable")


class DatasetError(ValueError):
    """题集结构、拆分或证据不合法，禁止带着错误标注继续评估。"""


class EvidenceLocator(BaseModel):
    """一条可唯一定位知识块的稳定证据；不携带数据库主键。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_key: NonBlank = Field(description="清单生成的稳定来源键，例如 doc:product-specs.md")
    section_path: NonBlank = Field(description="字符切分器生成的完整章节路径")
    quote: NonBlank = Field(description="知识块正文中逐字一致的非空原文片段；不可仅写改述")


class EvalCase(BaseModel):
    """一条校准或固定验收题，明确回答边界及证据组的或/且关系。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    id: NonBlank = Field(description="题集内唯一且稳定的题号")
    split: Literal["calibration", "evaluation"] = Field(description="校准集或固定验收集")
    fact_group: NonBlank = Field(description="同一知识事实主题的分组键，不允许跨集合")
    bucket: Bucket = Field(description="六类题桶：政策、型号、口语、过滤、多证据、无答案")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="单轮问题难度：易、中、难")
    query: NonBlank = Field(description="无需历史指代即可理解的用户原问")
    filters: dict[str, FilterValue] = Field(default_factory=dict, description="真实知识元数据的等值或列表筛选条件")
    answerable: bool = Field(description="在指定筛选范围内是否有充分证据回答")
    evidence_groups: list[list[EvidenceLocator]] = Field(description="组内任一证据均可，组间每组均为必需")
    required_facts: list[NonBlank] = Field(description="参考答案必须覆盖的事实；无答案题为空")
    reference_answer: NonBlank = Field(description="依据原文核对的参考答案；无答案题明确说明无法确认的边界")
    forbidden_claims: list[NonBlank] = Field(description="容易混淆但原文不支持或明确否定的断言")

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, filters: dict[str, FilterValue]) -> dict[str, FilterValue]:
        """拒绝虚构字段和错误标量类型。参数：筛选映射；返回：原映射；异常：ValueError。"""
        for key, value in filters.items():
            if key not in {"source_key", "category", "content_type", "is_key_clause"}:
                raise ValueError(f"未知筛选字段：{key}")
            values = value if isinstance(value, list) else [value]
            expected = bool if key == "is_key_clause" else str
            if not values or any(type(item) is not expected or (isinstance(item, str) and not item.strip()) for item in values):
                raise ValueError(f"筛选字段取值类型错误或为空：{key}")
        return filters

    @model_validator(mode="after")
    def validate_answer_boundary(self) -> EvalCase:
        """检查答案与证据及题桶一致。参数：当前题目；返回：自身；异常：ValueError。"""
        if self.answerable:
            if not self.evidence_groups or not self.required_facts or any(not group for group in self.evidence_groups):
                raise ValueError("可回答题必须包含非空证据组和关键事实")
            if self.bucket == "F_unanswerable":
                raise ValueError("无答案桶不可标记为可回答")
        elif self.evidence_groups or self.required_facts or self.bucket != "F_unanswerable":
            raise ValueError("无答案题必须属于无答案桶且证据与关键事实为空")
        if self.bucket == "D_filter" and not self.filters:
            raise ValueError("过滤题必须明确提供元数据筛选条件")
        if self.bucket == "E_multi" and len(self.evidence_groups) < 2:
            raise ValueError("多证据题必须包含至少两个必需证据组")
        return self


def _matches_filters(chunk: ChunkSnapshot, filters: dict[str, FilterValue]) -> bool:
    """核验真实元数据筛选范围。参数：快照和合法筛选；返回：是否满足；异常：无。"""
    for key, wanted in filters.items():
        actual = chunk.text.category if key == "category" else getattr(chunk, key)
        if actual not in (wanted if isinstance(wanted, list) else [wanted]):
            return False
    return True


def resolve_evidence(case: EvalCase, chunks: list[ChunkSnapshot]) -> list[set[int]]:
    """把稳定原文标注解析为当前快照主键，严格拒绝每条定位器的缺失或歧义。

    参数：case 为已校验题目；chunks 为当前冻结语料，主键必须唯一。
    返回：必需证据组列表；每组内为可替代证据主键集合，无答案题返回空列表。
    异常：DatasetError，重复主键、来源/章节/片段不匹配、过滤排除或证据歧义。
    """
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise DatasetError("冻结语料包含重复主键")
    eligible = [chunk for chunk in chunks if chunk.is_active and _matches_filters(chunk, case.filters)]
    resolved: list[set[int]] = []
    for group_index, group in enumerate(case.evidence_groups, start=1):
        identifiers: set[int] = set()
        for locator in group:
            matches = [chunk.id for chunk in eligible if chunk.source_key == locator.source_key
                       and chunk.section_path == locator.section_path and locator.quote in chunk.text.answer]
            # 每条 OR 分支也必须有效，不让另一分支掩盖过期或错误标注。
            if len(matches) != 1:
                reason = "缺失" if not matches else "歧义"
                raise DatasetError(f"{case.id} 第{group_index}组证据{reason}：{locator.source_key} / {locator.section_path}")
            identifiers.add(matches[0])
        resolved.append(identifiers)
    return resolved


def _source_chunks() -> list[ChunkSnapshot]:
    """从仓库清单按真实400/60/10字符规则构造只用于校验的临时快照。

    参数：无；返回：以临时正整数编号的六份原文快照；不读写数据库。
    异常：ValueError 或 OSError，清单、原文或切分失败；上层统一转为 DatasetError。
    """
    root = Path(__file__).resolve().parents[2] / "data" / "kb"
    chunks = []
    for document in load_manifest(root / "manifest.json", root=root):
        # len 仅作为离线保守硬限检查；分块本身完全使用生产字符切分器。
        for draft in build_document_chunks(document.path.read_text(encoding="utf-8"), document, len):
            chunks.append(ChunkSnapshot(**draft.model_dump(), id=len(chunks) + 1,
                                        content_hash=sha256(draft.text.model_dump_json().encode()).hexdigest(),
                                        embedding_revision="dataset-validation-only"))
    return chunks


def load_dataset(path: Path, *, chunks: list[ChunkSnapshot] | None = None) -> list[EvalCase]:
    """加载72题发布集并检查计数、事实隔离与全部证据。

    参数：path 为UTF-8 JSON文件，根节点必须是题目数组；chunks 可传冻结语料，
        默认读取仓库六份资料并执行真实字符切分，便于离线验证。
    返回：24条校准题和48条固定题，每桶分别4条和8条，保持文件顺序。
    异常：DatasetError，文件/JSON错误、题目字段错误、计数错误、跨集合事实泄漏或证据错误。
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise DatasetError("题集根节点必须为数组")
        cases = [EvalCase.model_validate(item) for item in payload]
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise DatasetError(f"题集不可读或字段不合法：{path}；{error}") from error
    if len(cases) != 72:
        raise DatasetError("发布题集必须恰好72题")
    if len({case.id for case in cases}) != len(cases):
        raise DatasetError("题号重复")
    counts = Counter((case.split, case.bucket) for case in cases)
    for split, expected in (("calibration", 4), ("evaluation", 8)):
        if any(counts[(split, bucket)] != expected for bucket in BUCKETS):
            raise DatasetError(f"{split} 每桶必须恰好{expected}题")
    fact_splits: dict[str, str] = {}
    for case in cases:
        if fact_splits.setdefault(case.fact_group, case.split) != case.split:
            raise DatasetError(f"事实组跨集合泄漏：{case.fact_group}")
    try:
        corpus = _source_chunks() if chunks is None else chunks
        for case in cases:
            resolve_evidence(case, corpus)
    except (ValueError, OSError) as error:
        if isinstance(error, DatasetError):
            raise
        raise DatasetError(f"题集原文校验失败：{error}") from error
    return cases
