"""在线 Dense FAQ 检索。

`keyword` 原样编码，只走 Milvus Dense 单路；命中后按 MySQL 原文权威回读，
要求 active/done、指纹与编码版本一致且 vector_id 等于本行主键。低分是正常
空结果；索引/原文失配或后端异常必须报告安全错误，不能伪装成空命中。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.knowledge.chunking import CHUNKING_VERSION
from app.knowledge.contracts import KnowledgeEncoder, VectorIndex
from app.knowledge.limits import valid_dense_vector
from app.knowledge.schemas import KnowledgeUnavailable, RetrievalPolicy
from app.knowledge.store import KnowledgeStore

MAX_RESULTS = 3
_MAX_KEYWORD_LENGTH = 100
_INITIAL_CANDIDATES = 12
_MAX_CANDIDATES = 48


class KnowledgeSearchUnavailable(KnowledgeUnavailable):
    """在线知识检索依赖或同步状态不可用。"""


def load_retrieval_policy(
    path: Path, *, embedding_revision: str, chunking_version: str
) -> RetrievalPolicy:
    """加载并校验在线检索校准策略。

    参数:
        path: retrieval-policy.json 路径。
        embedding_revision: 当前编码器版本。
        chunking_version: 当前切分规范版本。
    返回值:
        RetrievalPolicy: 版本匹配的校准策略。
    异常:
        KnowledgeSearchUnavailable: 文件不可读、字段非法或版本不匹配。
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy = RetrievalPolicy.model_validate(payload)
    except (OSError, ValueError, ValidationError) as error:
        raise KnowledgeSearchUnavailable("在线检索校准策略不可用") from error
    if (
        policy.embedding_revision != embedding_revision
        or policy.chunking_version != chunking_version
    ):
        raise KnowledgeSearchUnavailable("在线检索校准策略版本不匹配")
    return policy


class KnowledgeSearch:
    """Original-keyword Dense 检索与 MySQL 权威回读。"""

    def __init__(
        self,
        store: KnowledgeStore,
        encoder: KnowledgeEncoder,
        index: VectorIndex,
        policy: RetrievalPolicy,
    ) -> None:
        """保存检索依赖。

        参数:
            store: MySQL 原文/状态存储。
            encoder: 共享 BGE-M3 编码器。
            index: Milvus Dense 索引。
            policy: 已校准的检索策略。
        返回值:
            None
        异常:
            无
        """
        self.store = store
        self.encoder = encoder
        self.index = index
        self.policy = policy

    async def query_faq(self, keyword: str, limit: int = MAX_RESULTS) -> list[dict]:
        """按原始关键词执行 Dense 检索并回读权威正文。

        参数:
            keyword: 用户原话中的连续关键词，1–100 字符。
            limit: 返回条数，1–3。
        返回值:
            list[dict]: 最多 limit 条 id/question/answer/category。
        异常:
            ValueError: 关键词或条数不合法。
            KnowledgeSearchUnavailable: 索引同步/依赖不可用。
        """
        if (
            not keyword.strip()
            or len(keyword) > _MAX_KEYWORD_LENGTH
            or not 1 <= limit <= MAX_RESULTS
        ):
            raise ValueError("FAQ 关键词或条数不合法")
        vectors = await self.encoder.encode([keyword])
        if (
            len(vectors) != 1
            or not valid_dense_vector(vectors[0])
        ):
            raise KnowledgeSearchUnavailable("查询编码结果不合法")
        vector = vectors[0]

        pool = _INITIAL_CANDIDATES
        while True:
            hits = await self.index.search(vector, pool)
            if not hits:
                return []
            candidate_hits = [hit for hit in hits if hit.score >= self.policy.min_score]
            if not candidate_hits:
                return []
            rows = await self.store.get_chunks([hit.id for hit in candidate_hits])
            by_id = {row.id: row for row in rows}
            valid: list[dict] = []
            for hit in candidate_hits:
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
                valid.append(
                    {
                        "id": row.id,
                        "question": row.text.questions,
                        "answer": row.text.answer,
                        "category": row.text.category,
                    }
                )
                if len(valid) >= limit:
                    return valid
            if len(hits) < pool:
                break
            if pool >= _MAX_CANDIDATES:
                break
            pool = _MAX_CANDIDATES

        if valid:
            return valid
        # 有高于阈值的候选但全部无法通过 MySQL 权威校验，属于同步故障，而非零命中。
        raise KnowledgeSearchUnavailable("知识索引与 MySQL 原文状态不一致")
