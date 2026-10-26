"""有来源约束的历史对话抽取与知识关系判定 Adapter。

复用项目聊天模型工厂的流式接口，完整聚合后校验唯一结果工具的原始参数；
拒绝正文 JSON、非有限数、未知来源、改写问题、伪造引用、缺失 pair 与超限输出。模型失败与
合法空结果严格区分，不能把异常吞成空知识。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError
from app.core.structured import (
    StructuredOutputError,
    bind_output_tool,
    collect_tool_output,
    output_tool,
    parse_tool_output,
)
from app.knowledge.grounding import GroundingError, validate_grounding

from app.knowledge.prompts import (
    EXTRACTION_INPUT_BUDGET,
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_TOOL_NAME,
    OUTPUT_TEXT_LIMIT,
    RELATION_PROMPT_VERSION,
    RELATION_TOOL_NAME,
    ExtractionResult,
    RelationResult,
    extraction_messages,
    relation_messages,
    messages_size,
)
from app.knowledge.schemas import (
    EvidenceItem,
    ExtractedQA,
    ExtractionBatch,
    KnowledgeUnavailable,
    RelationDecision,
    RelationPolicy,
)

_Output = TypeVar("_Output", bound=BaseModel)


class KnowledgeModelError(KnowledgeUnavailable):
    """知识模型调用或输出不合法。"""


class KnowledgeModel:
    """带严格来源和引用校验的知识模型 Adapter。"""

    def __init__(self, client_factory: Callable[..., Any]) -> None:
        """保存聊天模型工厂。

        参数:
            client_factory: 接受 streaming/temperature 的模型工厂，例如 get_chat_model。
        返回值:
            None
        异常:
            无
        """
        self._client_factory = client_factory

    async def _stream_output(
        self, messages: list[Any], schema: type[_Output], name: str
    ) -> _Output:
        """强制调用唯一结果工具，聚合原始分片后校验参数。

        参数:
            messages: 系统与用户消息。
            schema: 只暴露模型可填写字段的结果契约。
            name: 本次唯一允许的工具名称。
        返回值:
            _Output: 完成严格参数校验的结果对象。
        异常:
            KnowledgeModelError: 输入超预算，或输出协议、字段与长度不合法。
            网络错误与取消原样向上传播，底层流由公共收集器负责关闭。
        """
        tool = output_tool(schema, name)
        input_size = messages_size(messages, tool=tool)
        if input_size > EXTRACTION_INPUT_BUDGET:
            raise KnowledgeModelError("知识模型输入超过估算预算")
        model = self._client_factory(streaming=True, temperature=0.0)
        bound = bind_output_tool(model, tool)
        try:
            gathered = await collect_tool_output(bound, messages, max_chars=OUTPUT_TEXT_LIMIT)
            return parse_tool_output(gathered, schema, name)
        except StructuredOutputError as error:
            raise KnowledgeModelError(f"知识模型结构化输出不合法：{error}") from error

    async def extract(self, batch: ExtractionBatch) -> list[ExtractedQA]:
        """从固定批次抽取有来源的候选知识。

        参数:
            batch: 已冻结的抽取批次。
        返回值:
            list[ExtractedQA]: 通过来源、原话和字段校验的候选知识。
        异常:
            KnowledgeModelError: 模型失败或输出不符合来源约束。
        """
        parsed = await self._stream_output(
            extraction_messages(batch), ExtractionResult, EXTRACTION_TOOL_NAME
        )
        source_questions = {turn.message_id: turn.question for turn in batch.turns}
        qas: list[ExtractedQA] = []
        for qa in parsed.qas:
            # 工具契约只能保证字段形状，来源归属与原文真实性必须继续由业务层验证。
            if any(identifier not in source_questions for identifier in qa.source_ids):
                raise KnowledgeModelError("抽取结果包含本批之外的来源")
            matching_questions = {
                source_questions[identifier] for identifier in qa.source_ids
            }
            if qa.question not in matching_questions:
                raise KnowledgeModelError("抽取问题不是来源用户原话")
            conversations = {
                turn.conversation_id
                for turn in batch.turns
                if turn.message_id in qa.source_ids
            }
            if len(conversations) != 1:
                raise KnowledgeModelError("抽取结果跨越会话边界")
            try:
                grounded, _ = validate_grounding(qa, batch.turns, require_excerpts=True)
            except (GroundingError, ValidationError) as error:
                raise KnowledgeModelError(str(error)) from error
            qas.append(grounded)
        return qas

    async def relate(
        self, items: list[EvidenceItem], pairs: list[tuple[str, str]]
    ) -> list[RelationDecision]:
        """判定请求中的每一对知识关系。

        参数:
            items: 可用于判定的证据。
            pairs: 必须逐对判定的引用组合。
        返回值:
            list[RelationDecision]: 与请求等长、顺序一致的关系结论。
        异常:
            KnowledgeModelError: 缺少/重复 pair、伪造引用或字段不合法。
        """
        if not pairs:
            return []
        parsed = await self._stream_output(
            relation_messages(items, pairs), RelationResult, RELATION_TOOL_NAME
        )
        refs = {item.ref for item in items}
        requested = list(dict.fromkeys(pairs))
        requested_set = set(requested)
        decisions: dict[tuple[str, str], RelationDecision] = {}
        for decision in parsed.decisions:
            pair = (decision.left_ref, decision.right_ref)
            if decision.left_ref not in refs or decision.right_ref not in refs:
                raise KnowledgeModelError("关系结果引用了未知证据")
            if pair not in requested_set:
                raise KnowledgeModelError("关系结果包含未请求的 pair")
            if pair in decisions:
                raise KnowledgeModelError("关系结果包含重复 pair")
            if not set(decision.evidence_refs).issubset(refs):
                raise KnowledgeModelError("关系证据引用未知 ref")
            decisions[pair] = decision
        if set(decisions) != requested_set:
            raise KnowledgeModelError("关系结果没有覆盖全部请求 pair")
        return [decisions[pair] for pair in requested]


def load_relation_policy(
    path: Path, *, embedding_revision: str, prompt_version: str
) -> RelationPolicy:
    """加载并校验关系判定校准策略。

    参数:
        path: relation-policy.json 路径。
        embedding_revision: 当前编码规范标识。
        prompt_version: 当前关系 Prompt 版本。
    返回值:
        RelationPolicy: 版本匹配的校准策略。
    异常:
        KnowledgeModelError: 文件不可读、字段非法或版本不匹配。
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy = RelationPolicy.model_validate(payload)
    except (OSError, ValueError, ValidationError) as error:
        raise KnowledgeModelError("关系校准策略不可用") from error
    if (
        policy.embedding_revision != embedding_revision
        or policy.prompt_version != prompt_version
    ):
        raise KnowledgeModelError("关系校准策略版本不匹配")
    return policy


__all__ = [
    "KnowledgeModel",
    "KnowledgeModelError",
    "load_relation_policy",
    "EXTRACTION_PROMPT_VERSION",
    "RELATION_PROMPT_VERSION",
]
