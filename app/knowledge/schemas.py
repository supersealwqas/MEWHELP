"""CH03 知识库共享数据契约。

定义文档切分、原文快照、向量记录、历史对话抽取、发布事务与检索策略的
Pydantic 模型，以及所有知识库模块共用的安全异常基类。后续任务不得再定义
同义类型，避免 Task 间字段漂移。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeUnavailable(RuntimeError):
    """知识库依赖不可用。

    用于模型、Milvus 或 MySQL 失败时向上层暴露安全错误，避免把故障伪装成
    正常空命中，也避免携带 SQL、凭据或完整聊天内容。
    """


class KnowledgeText(BaseModel):
    """进入向量化的三格知识文本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str = Field(min_length=1, description="知识分类，例如开票与配送")
    questions: str = Field(min_length=1, description="知识问法，可包含换行分隔的真实问法")
    answer: str = Field(min_length=1, description="知识答案原文，保留数字、否定条件和标点")


class ChunkDraft(BaseModel):
    """尚未持久化的知识块草案。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_key: str = Field(min_length=1, description="稳定来源键，文档为 doc:相对路径，挖掘为 qa:暂存ID")
    ordinal: int = Field(ge=0, description="来源内从 0 开始的连续块序号")
    text: KnowledgeText = Field(description="进入向量化的三格文本")
    section_path: str | None = Field(default=None, min_length=1, description="完整章节标题路径，仅作元数据")
    content_type: str = Field(min_length=1, description="内容类型，例如 faq、policy、manual")
    is_key_clause: bool = Field(default=False, description="是否为清单显式标注的关键条款")


class ChunkSnapshot(ChunkDraft):
    """MySQL 中已持久化的知识块快照。"""

    id: int = Field(gt=0, description="MySQL 与 Milvus 共用的正整数主键")
    content_hash: str = Field(min_length=64, max_length=64, description="三格文本与格式规范版本的 SHA256")
    embedding_revision: str = Field(min_length=1, description="编码模型与推理配置标识")
    is_active: bool = Field(default=True, description="是否具备在线发布资格")
    vector_id: str | None = Field(default=None, min_length=1, description="已同步向量主键的字符串形式")
    vectorize_status: Literal["pending", "done"] = Field(default="pending", description="向量同步状态")
    prev_chunk_id: int | None = Field(default=None, gt=0, description="同一来源前一知识块主键")
    next_chunk_id: int | None = Field(default=None, gt=0, description="同一来源后一知识块主键")


class VectorRecord(BaseModel):
    """写入 Milvus 的单条向量记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(gt=0, description="与 MySQL 原文一致的正整数主键")
    vector: list[float] = Field(description="1024 维 Dense 向量")
    text: str | None = Field(default=None, min_length=1, description="与 Dense 编码输入一致的三格原文；CH04 原生 BM25 必填，旧 Dense adapter 忽略")
    content_hash: str = Field(min_length=64, max_length=64, description="用于条件回填的内容指纹")
    embedding_revision: str = Field(min_length=1, description="编码模型与推理配置标识")


class VectorHit(BaseModel):
    """Milvus 返回的向量命中。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(gt=0, description="命中的 MySQL/Milvus 共用主键")
    score: float = Field(allow_inf_nan=False, description="余弦相似度分数")
    content_hash: str = Field(min_length=64, max_length=64, description="向量的内容指纹")
    embedding_revision: str = Field(min_length=1, description="向量的编码规范标识")


class CompletedTurn(BaseModel):
    """从已完成聊天轮次中抽取的通用知识来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: int = Field(gt=0, description="最终客服回答消息 ID，用作来源上界与游标")
    conversation_id: int = Field(gt=0, description="会话 ID，用于保留会话边界")
    turn_id: str = Field(min_length=1, description="轮次稳定标识")
    question: str = Field(min_length=1, description="该轮真实用户问题原文")
    answer: str = Field(min_length=1, description="该轮最终非空客服回答原文")


class ExtractionBatch(BaseModel):
    """一次历史对话抽取的固定输入快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_no: str = Field(min_length=1, description="稳定批次号，用于幂等恢复")
    source_ids: list[Annotated[int, Field(gt=0)]] = Field(min_length=1, description="本批关联的最终回答消息 ID")
    snapshot_upper_id: int = Field(ge=0, description="本批固定快照的最终回答消息上界")
    input_hash: str = Field(min_length=1, description="批次输入的稳定指纹")
    prompt_version: str = Field(min_length=1, description="抽取 Prompt 规范版本")
    turns: list[CompletedTurn] = Field(min_length=1, description="按会话边界组织的完整轮次")


class AnswerExcerpt(BaseModel):
    """模型提供的完整原句引用；是否可信由代码回核，不接受模型自报。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    message_id: int = Field(gt=0, description="原回答消息主键，必须属于候选source_ids")
    quote: str = Field(min_length=1, description="完整原句，句内字符、数字、空白必须逐字一致")


class ExtractedQA(BaseModel):
    """模型抽取出的候选知识。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_ids: list[Annotated[int, Field(gt=0)]] = Field(min_length=1, description="支持该知识的问题来源消息 ID")
    question: str = Field(min_length=1, description="来源用户的真实完整问法")
    answer: str = Field(min_length=1, description="有原文支持的可泛化答案")
    category: str = Field(min_length=1, description="知识分类")
    answer_excerpts: list[AnswerExcerpt] = Field(default_factory=list, description="按来源顺序组合的完整原句；旧数据缺失默认空，不代表已验证")


class StagedQA(ExtractedQA):
    """暂存表中的候选知识及处置状态。"""

    id: int = Field(gt=0, description="暂存记录主键")
    status: Literal["extracted", "kept", "discarded"] = Field(description="候选知识处置状态")
    conflict_group: str | None = Field(default=None, min_length=1, description="持久冲突组身份")
    knowledge_id: int | None = Field(default=None, gt=0, description="关联的已发布知识主键")
    source_evidence: list[dict] = Field(default_factory=list, description="代码回核的消息、原句、offset/hash与校验版本，持久于现有JSON审计列")


class EvidenceItem(BaseModel):
    """关系判定使用的来源证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str = Field(min_length=1, description="证据在本次判定中的稳定引用")
    origin: Literal["document", "conversation"] = Field(description="证据来源类型")
    text: KnowledgeText = Field(description="证据三格文本")
    source_refs: list[str] = Field(min_length=1, description="证据对应的文件章节或对话来源")


class RelationDecision(BaseModel):
    """受约束模型对一对候选知识的关系判定。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_ref: str = Field(min_length=1, description="左侧候选引用")
    right_ref: str = Field(min_length=1, description="右侧候选引用")
    relation: Literal["same", "complementary", "conflict", "unrelated", "uncertain"] = Field(
        description="两者关系"
    )
    evidence_refs: list[str] = Field(min_length=1, description="支持该判定的证据引用")
    reason: str = Field(min_length=1, description="安全可审计的判定理由")


class StageChange(BaseModel):
    """发布事务中暂存记录的原子处置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(gt=0, description="暂存记录主键")
    status: Literal["extracted", "kept", "discarded"] = Field(description="目标处置状态")
    conflict_group: str | None = Field(default=None, min_length=1, description="持久冲突组身份")
    knowledge_source_key: str | None = Field(default=None, min_length=1, description="关联知识的稳定来源键")
    reason: str = Field(min_length=1, description="处置原因")
    evidence_refs: list[str | dict] = Field(default_factory=list, description="处置证据引用或带当时原文的完整文档审计快照")


class StoreFrame(BaseModel):
    """发布前固定下来的数据库状态摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str = Field(min_length=1, description="覆盖发布、内容与隔离状态的稳定令牌")
    chunks: list[ChunkSnapshot] = Field(default_factory=list, description="参与判定的知识快照")
    staged: list[StagedQA] = Field(default_factory=list, description="参与判定的暂存记录")


class Publication(BaseModel):
    """一次原子发布或撤回的完整变更集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_token: str = Field(min_length=1, description="提交前核对的状态令牌")
    drafts: list[ChunkDraft] = Field(default_factory=list, description="本次要写入或更新的知识块")
    embedding_revision: str = Field(min_length=1, description="本次编码快照使用的编码规范标识")
    replace_sources: list[str] = Field(default_factory=list, description="本次完整替换的文档来源键")
    inactive_ids: list[Annotated[int, Field(gt=0)]] = Field(default_factory=list, description="要撤回发布资格的知识主键")
    stage_changes: list[StageChange] = Field(default_factory=list, description="与发布同事务提交的暂存处置")


class SyncReport(BaseModel):
    """一次向量同步的可审计结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scanned: int = Field(ge=0, description="扫描的待同步或待清理记录数")
    upserted: int = Field(ge=0, description="成功写入 Milvus 的记录数")
    marked_done: int = Field(ge=0, description="成功条件回填为 done 的记录数")
    deleted_vectors: int = Field(ge=0, description="成功删除的失效向量数")
    remaining_pending: int = Field(ge=0, description="仍为 active/pending 的记录数")


class JobReport(BaseModel):
    """一次导入或挖掘任务的报告。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_count: int = Field(ge=0, description="处理的抽取批次数")
    extracted: int = Field(ge=0, description="模型抽取出的候选知识数")
    kept: int = Field(ge=0, description="保留的候选知识数")
    discarded: int = Field(ge=0, description="丢弃的候选知识数")
    quarantined: int = Field(ge=0, description="进入持久隔离组的候选知识数")
    sync: SyncReport = Field(description="任务结束后的向量同步结果")


class RetrievalPolicy(BaseModel):
    """在线 Dense 检索的校准策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False, description="经标注校准的最低相似度")
    embedding_revision: str = Field(min_length=1, description="策略适用的编码规范标识")
    chunking_version: str = Field(min_length=1, description="策略适用的切分规范版本")
    calibration_digest: str = Field(min_length=1, description="校准输入与结果的摘要")


class RelationPolicy(BaseModel):
    """离线知识关系判定的校准策略。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False, description="保留候选对的最低相似度")
    embedding_revision: str = Field(min_length=1, description="策略适用的编码规范标识")
    prompt_version: str = Field(min_length=1, description="策略适用的关系判定 Prompt 版本")
    calibration_digest: str = Field(min_length=1, description="校准输入与结果的摘要")
