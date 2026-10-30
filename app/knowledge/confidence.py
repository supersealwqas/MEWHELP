"""正式证据特征、校准制品与唯一证据闸；无默认阈值，不将故障当知识不足。"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.memory import ContextBudgetExceeded
from app.knowledge.answering import EvidenceCitation
from app.knowledge.rerank import RankedChunk
from app.knowledge.schemas import KnowledgeUnavailable

FEATURE_PROTOCOL = "evidence-v2"


class EvidencePolicyUnavailable(KnowledgeUnavailable, ValueError):
    """正式制品缺失或不兼容；线上按依赖故障处理，不生成知识缺口。"""


class EvidenceFeatures(BaseModel):
    """全精排去重候选与实际预算内证据的观测，缺第二分保留为空。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    top1_score: float | None = Field(ge=0, le=1, allow_inf_nan=False, description="全部去重精排候选最高分；零候选为空")
    top2_score: float | None = Field(ge=0, le=1, allow_inf_nan=False, description="第二高分；不足两条为空")
    score_gap: float | None = Field(ge=0, le=1, allow_inf_nan=False, description="最高分减第二分；无第二分时为空")
    effective_count: int = Field(ge=0, description="实际预算内达到有效分门槛的去重证据数")
    candidate_count: int = Field(ge=0, description="跨查询最终裁剪前全部精排返回的去重候选数，非召回候选数")

    @model_validator(mode="after")
    def validate_consistency(self):
        """校验缺失分与数量一致。参数：自身；返回：自身；异常：ValueError，不一致。"""
        if ((self.top1_score is None) != (self.candidate_count == 0)
                or (self.top2_score is None) != (self.candidate_count < 2)
                or (self.score_gap is None) != (self.top2_score is None)
                or self.effective_count > self.candidate_count):
            raise ValueError("证据特征数量与缺失分不一致")
        if self.top2_score is not None and (self.top2_score > self.top1_score
                or not math.isclose(self.score_gap, self.top1_score - self.top2_score, abs_tol=1e-12)):
            raise ValueError("证据分差与真实分数不一致")
        return self


class EvidencePolicy(BaseModel):
    """仅由真实校准生成的正式策略，协议兼容与训练来源分别保存。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[2] = Field(default=2, description="正式证据制品结构版本")
    version: str = Field(min_length=1, description="冻结参数及来源确定的策略版本")
    feature_protocol: Literal["evidence-v2"] = Field(description="全候选与实际selected的特征协议")
    minimum_score: float = Field(ge=0, le=1, allow_inf_nan=False, description="预算内证据计为有效的校准门槛")
    count_cap: int = Field(gt=0, description="有效证据数量归一化上限，来自观测正整数")
    weights: dict[str, float] = Field(description="top1/count/gap非负权重，总和必须为1")
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False, description="校准产生的综合分通过阈值")
    protocols: dict[str, str] = Field(description="查询、精排和证据预算协议；知识版本不用于阻止后续评估")
    training_source: dict = Field(description="校准来源、题集及语料摘要，仅用于追溯")

    @model_validator(mode="after")
    def validate_weights(self):
        """拒绝遗漏、负数或非有限权重。参数：自身；返回：自身；异常：ValueError。"""
        if (set(self.weights) != {"top1", "count", "gap"}
                or any(not math.isfinite(value) or value < 0 for value in self.weights.values())
                or not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-12)
                or self.weights["top1"] < 0.25):
            raise ValueError("证据权重必须非负、总和为1且Top1至少0.25")
        return self


def unique_ranked(ranked: list[RankedChunk]) -> list[RankedChunk]:
    """稳定身份去重并取实际最高分，保留第一次出现次序。

    参数：ranked完整精排候选。返回：去重候选；异常：ValueError，同主键/来源序号原文冲突。
    """
    by_identity, ids = {}, {}
    for item in ranked:
        if type(item.score) not in (int, float) or not math.isfinite(item.score):
            raise ValueError("精排分必须是有限真实数值，不能使用布尔或NaN")
        # 重验 model_copy/构造绕过验证的数据，NaN不能进入最高分计算。
        item = RankedChunk.model_validate(item.model_dump())
        identity = (item.chunk.source_key, item.chunk.ordinal)
        text = (item.chunk.text.model_dump(), item.chunk.section_path, item.chunk.content_type,
                item.chunk.content_hash, item.chunk.is_key_clause)
        old_identity = ids.setdefault(item.chunk.id, identity)
        old = by_identity.get(identity)
        if old_identity != identity or (old is not None and (
                old.chunk.id != item.chunk.id or text != (old.chunk.text.model_dump(), old.chunk.section_path,
                old.chunk.content_type, old.chunk.content_hash, old.chunk.is_key_clause))):
            raise ValueError("同一稳定身份对应不同主键或原文")
        if old is None or item.score > old.score:
            by_identity[identity] = item
    return list(by_identity.values())


def extract_features(ranked: list[RankedChunk], selected: list[EvidenceCitation], *, minimum_score: float) -> EvidenceFeatures:
    """从全候选与实际selected提取特征，不能以Top5快照代替。

    参数：ranked全精排、selected预算内原文、minimum_score有限0–1。返回：EvidenceFeatures。
    异常：ValueError，坏参数、候选身份/原文冲突、selected非本次完整证据。
    """
    if type(minimum_score) not in (int, float) or not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
        raise ValueError("有效分门槛必须是0–1有限数值")
    rows = unique_ranked(ranked)
    by_id, selected_ids = {item.chunk.id: item for item in rows}, set()
    for citation in selected:
        row = by_id.get(citation.chunk_id)
        if row is None or (citation.question, citation.answer, citation.section_path, citation.source_key) != (
                row.chunk.text.questions, row.chunk.text.answer, row.chunk.section_path, row.chunk.source_key):
            raise ValueError("selected必须来自本次完整原文")
        selected_ids.add(citation.chunk_id)
    scores = sorted((item.score for item in rows), reverse=True)
    return EvidenceFeatures(top1_score=scores[0] if scores else None,
        top2_score=scores[1] if len(scores) > 1 else None,
        score_gap=scores[0] - scores[1] if len(scores) > 1 else None,
        candidate_count=len(rows), effective_count=sum(by_id[key].score >= minimum_score for key in selected_ids))


def evidence_confidence(features: EvidenceFeatures, policy: EvidencePolicy) -> float:
    """计算有限线性组合，缺分差贡献为0但特征仍保留None。

    参数：已验证特征与校准策略；返回：0–1综合分；异常：无。
    """
    return (policy.weights["top1"] * (features.top1_score or 0.0)
            + policy.weights["count"] * min(features.effective_count / policy.count_cap, 1.0)
            + policy.weights["gap"] * (features.score_gap or 0.0))


def assess_evidence(ranked, selected, policy: EvidencePolicy):
    """产生一次正式判定，预算全拒与坏数据作为故障传播。

    参数：完整ranked、预算selected、正式policy。返回：GateDecision v2。
    异常：ValueError非法证据；ContextBudgetExceeded非空候选却无预算内证据。
    """
    from app.schemas.workflow import GateDecision, RetrievalLowConfidence
    features = extract_features(ranked, selected, minimum_score=policy.minimum_score)
    if features.candidate_count and not selected:
        raise ContextBudgetExceeded("完整知识证据无法放入本轮预算")
    score = evidence_confidence(features, policy)
    passed = features.candidate_count > 0 and score >= policy.threshold
    return GateDecision(schema_version=2, policy_version=policy.version, features=features,
        evidence_confidence=score, max_score=features.top1_score, threshold=policy.threshold, passed=passed,
        low_confidence=None if passed else RetrievalLowConfidence(reason="实际检索证据未达到冻结策略的充分性门槛"))


def load_evidence_policy(path: Path, *, protocols: dict[str, str]) -> EvidencePolicy:
    """读取正式制品并检查行为协议，不以训练语料摘要限制知识更新。

    参数：path策略文件、protocols当前查询/精排/预算协议。返回：EvidencePolicy。
    异常：ValueError，缺失/旧版/不兼容制品；OSError已转为安全错误。
    """
    try:
        policy = EvidencePolicy.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvidencePolicyUnavailable("正式v2证据策略缺失或非法，必须先完成生产校准") from error
    if policy.protocols != protocols:
        raise EvidencePolicyUnavailable("证据策略与当前查询/精排/预算协议不兼容")
    return policy


def evidence_policy_digest(policy: EvidencePolicy | dict) -> str:
    """统一策略内容摘要，与文件排版和字段原始次序无关。

    参数：严格策略或其JSON字段；返回：规范内容SHA256；异常：序列化错误。
    """
    value = policy.model_dump(mode="json") if isinstance(policy, EvidencePolicy) else policy
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
