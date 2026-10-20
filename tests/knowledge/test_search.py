"""在线 Dense 检索与 SupportRepository 注入契约测试。"""

from __future__ import annotations

import pytest

from app.knowledge.schemas import (
    ChunkSnapshot,
    KnowledgeText,
    KnowledgeUnavailable,
    RetrievalPolicy,
    VectorRecord,
)
from app.knowledge.search import KnowledgeSearch, KnowledgeSearchUnavailable
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import VectorIndexUnavailable
from app.repositories.support import SupportRepository
from app.tools.registry import build_registry
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, MemoryVectorIndex

REVISION = "fake-encoder-v1"
QUERY_VECTOR = [1.0] + [0.0] * 1023


def _policy(min_score: float = 0.0) -> RetrievalPolicy:
    """构造检索策略。参数：阈值；返回：策略；异常：校验失败。"""
    return RetrievalPolicy(
        min_score=min_score,
        embedding_revision=REVISION,
        chunking_version="ch03-v2",
        calibration_digest="test-only",
    )


def _snapshot(
    identifier: int,
    answer: str,
    *,
    active: bool = True,
    vector_id: str | None = None,
    revision: str = REVISION,
) -> ChunkSnapshot:
    """构造知识快照。参数：主键/答案/发布与向量状态；返回：快照；异常：校验失败。"""
    text = KnowledgeText(category="物流", questions=f"问法{identifier}", answer=answer)
    return ChunkSnapshot(
        id=identifier,
        source_key=f"doc:{identifier}.md",
        ordinal=0,
        text=text,
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
        content_hash=content_fingerprint(text),
        embedding_revision=revision,
        is_active=active,
        vector_id=str(identifier) if vector_id is None and active else vector_id,
        vectorize_status="done",
    )


def _record(snapshot: ChunkSnapshot, vector: list[float]) -> VectorRecord:
    """由快照与向量构造索引记录。参数：快照/向量；返回：记录；异常：校验失败。"""
    return VectorRecord(
        id=snapshot.id,
        vector=vector,
        content_hash=snapshot.content_hash,
        embedding_revision=snapshot.embedding_revision,
    )


def _search(
    snapshots: list[ChunkSnapshot],
    records: list[VectorRecord],
    *,
    min_score: float = 0.0,
) -> tuple[KnowledgeSearch, FakeEncoder, MemoryVectorIndex]:
    """组装内存检索。参数：快照/记录/阈值；返回：搜索器与依赖；异常：无。"""
    store = MemoryKnowledgeStore(snapshots=snapshots)
    index = MemoryVectorIndex()
    for record in records:
        index.records[record.id] = record
    encoder = FakeEncoder()
    return KnowledgeSearch(store, encoder, index, _policy(min_score)), encoder, index


async def test_keyword_encoded_verbatim() -> None:
    """关键词原样编码，不做同义改写。参数：无；返回：None；异常：无。"""
    snapshot = _snapshot(1, "未满99元收6元。")
    search, encoder, _ = _search([snapshot], [_record(snapshot, QUERY_VECTOR)])
    hits = await search.query_faq("邮费")
    assert encoder.inputs == ["邮费"]
    assert hits[0]["id"] == 1


async def test_results_follow_vector_order_and_mysql_text() -> None:
    """结果按向量顺序并回读 MySQL 正文。参数：无；返回：None；异常：无。"""
    first = _snapshot(1, "第一条权威答案。")
    second = _snapshot(2, "第二条权威答案。")
    third = _snapshot(3, "第三条权威答案。")
    search, _, _ = _search(
        [first, second, third],
        [
            _record(first, QUERY_VECTOR),
            _record(second, [0.8, 0.6] + [0.0] * 1022),
            _record(third, [0.0, 1.0] + [0.0] * 1022),
        ],
    )
    hits = await search.query_faq("邮费")
    assert [row["id"] for row in hits] == [1, 2, 3]
    assert set(hits[0]) == {"id", "question", "answer", "category"}
    assert hits[0]["answer"] == "第一条权威答案。"


@pytest.mark.parametrize("case", ["inactive", "mismatched"])
async def test_inactive_or_mismatched_hits_never_return(case: str) -> None:
    """失效/指纹失配命中不能被当作正常结果。参数：失配类型；返回：None；异常：无。"""
    if case == "inactive":
        snapshot = _snapshot(1, "不应返回。", active=False)
    else:
        snapshot = _snapshot(1, "不应返回。", vector_id="wrong-id")
    broken = _record(snapshot, QUERY_VECTOR)
    search, _, _ = _search([snapshot], [broken])
    with pytest.raises(KnowledgeSearchUnavailable):
        await search.query_faq("邮费")


async def test_backend_error_not_empty_hits() -> None:
    """索引故障不能伪装成空命中。参数：无；返回：None；异常：无。"""
    class _BrokenIndex:
        """搜索即失败的索引替身。"""

        async def search(self, vector, limit):
            """模拟索引不可用。参数：向量/数量；返回：不返回；异常：VectorIndexUnavailable。"""
            raise VectorIndexUnavailable("索引不可用")

    store = MemoryKnowledgeStore(snapshots=[_snapshot(1, "答案。")])
    search = KnowledgeSearch(store, FakeEncoder(), _BrokenIndex(), _policy())
    with pytest.raises(KnowledgeUnavailable):
        await search.query_faq("邮费")


async def test_unrelated_scores_empty_but_no_like_fallback() -> None:
    """低于校准阈值返回空，不调用旧 FAQ LIKE。参数：无；返回：None；异常：无。"""
    snapshot = _snapshot(1, "无关答案。")
    search, _, _ = _search(
        [snapshot],
        [_record(snapshot, [0.0, 1.0] + [0.0] * 1022)],
        min_score=0.9,
    )
    assert await search.query_faq("邮费") == []


async def test_query_faq_rejects_invalid_limit_and_blank_keyword() -> None:
    """仓储层拒绝空白关键词、非法条数，并在无检索时失败关闭。参数：无；返回：None；异常：无。"""
    repository = SupportRepository(None)
    with pytest.raises(ValueError):
        await repository.query_faq("邮费", limit=0)
    with pytest.raises(ValueError):
        await repository.query_faq("   ")
    with pytest.raises(KnowledgeUnavailable):
        await repository.query_faq("邮费")


def test_public_tool_schema_and_envelope_unchanged() -> None:
    """query_faq 仍只暴露 keyword 参数。参数：无；返回：None；异常：无。"""
    schema = build_registry().get("query_faq").tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"keyword"}
