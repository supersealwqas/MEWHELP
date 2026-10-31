"""生命周期批次租约集成测试：真实 writer 和独立 SQLite，仅替换模型与命名锁。"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, async_sessionmaker, create_async_engine

from app.knowledge.errors import LeaseLost
from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.schemas import ChunkDraft, KnowledgeText, RelationDecision, RelationPolicy
from app.knowledge.store import KnowledgeWriter
from tests.knowledge.fakes import FakeEncoder


class _LeaseBoundary:
    """只替代 MySQL 命名锁状态，保留真实持锁连接、会话和发布事务。"""

    def __init__(self, connection: AsyncConnection) -> None:
        """保存连接和同一失锁异常。参数：connection独立连接；返回：None；异常：无。"""
        self.connection = connection
        self.lost = False
        self.error = LeaseLost("测试租约的原始所有者已丢失")

    async def assert_held(self) -> None:
        """核验模拟所有者。参数：无；返回：None；异常：原始LeaseLost。"""
        if self.lost:
            raise self.error
        # 模拟真实核锁查询产生的隐式事务，验证 writer 会在外部请求前结束它。
        await self.connection.scalar(text("SELECT 1"))


class _RelationBoundary:
    """记录真实生命周期发出的模型批次，可在指定批次返回时模拟失锁。"""

    def __init__(self, lease: _LeaseBoundary, *, lose_at: int = 0, malformed: bool = False) -> None:
        """配置模型边界。参数：租约、失锁批次和畸形响应开关；返回：None；异常：无。"""
        self.lease = lease
        self.lose_at = lose_at
        self.malformed = malformed
        self.calls: list[list[tuple[str, str]]] = []

    async def relate(self, items, pairs) -> list[RelationDecision]:
        """返回无关关系。参数：证据和候选对；返回：完整或畸形决定；异常：断言失败。"""
        assert not self.lease.connection.in_transaction(), "模型等待期间不能占用核锁事务"
        self.calls.append(list(pairs))
        if len(self.calls) == self.lose_at:
            self.lease.lost = True
            if self.malformed:
                return []
        return [RelationDecision(left_ref=left, right_ref=right, relation="unrelated",
                                 evidence_refs=[left, right], reason="独立的虚构知识")
                for left, right in pairs]


class _EncoderBoundary(FakeEncoder):
    """仅替代外部编码器，可在计数或异步编码返回时改变锁边界状态。"""

    def __init__(self, lease: _LeaseBoundary, *, loss: str = "", malformed: bool = False) -> None:
        """配置编码故障。参数：租约、失锁阶段和畸形向量开关；返回：None；异常：无。"""
        super().__init__()
        self.lease = lease
        self.loss = loss
        self.malformed = malformed

    def count_tokens(self, value: str) -> int:
        """候选预检查时可失锁。参数：文本；返回：字符数；异常：无。"""
        if self.loss == "before_encode":
            self.lease.lost = True
        return super().count_tokens(value)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """编码返回前可失锁。参数：文本批次；返回：向量；异常：断言失败。"""
        assert not self.lease.connection.in_transaction(), "编码等待期间不能占用核锁事务"
        vectors = await super().encode(texts)
        if self.loss == "after_encode":
            self.lease.lost = True
        return [] if self.malformed else vectors


@pytest.fixture
async def lease_runtime(tmp_path: Path) -> AsyncIterator[tuple]:
    """创建独占 SQLite。参数：pytest临时目录；返回：writer、租约、工厂；异常：数据库错误。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lifecycle-lease.sqlite'}")
    factory = async_sessionmaker(engine)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql("""
                CREATE TABLE knowledge_chunks (
                    id INTEGER PRIMARY KEY, source_key TEXT, ordinal INTEGER,
                    category TEXT, questions TEXT, answer TEXT, section_path TEXT,
                    content_type TEXT, is_key_clause INTEGER, content_hash TEXT,
                    embedding_revision TEXT, is_active INTEGER, prev_chunk_id INTEGER,
                    next_chunk_id INTEGER, vector_id TEXT, vectorize_status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await connection.exec_driver_sql("""
                CREATE TABLE qa_extraction_staging (
                    id INTEGER PRIMARY KEY, batch_no TEXT, source_ref TEXT, source_ids JSON,
                    source_fingerprint TEXT, question TEXT, answer TEXT, category TEXT,
                    status TEXT, conflict_group TEXT, evidence_refs JSON, reason TEXT,
                    knowledge_id INTEGER, created_at DATETIME, updated_at DATETIME
                )
            """)
        async with engine.connect() as connection:
            lease = _LeaseBoundary(connection)
            yield KnowledgeWriter(factory, lease), lease, factory
    finally:
        await engine.dispose()


def _drafts(count: int) -> list[ChunkDraft]:
    """生成互异文档。参数：文档数量；返回：文档草案；异常：字段校验失败。"""
    return [ChunkDraft(source_key=f"doc:new-{index}.md", ordinal=0, content_type="policy",
                       text=KnowledgeText(category="租约示例", questions=f"新问题{index}", answer=f"新答案{index}。"))
            for index in range(count)]


async def _seed(factory, *, chunks: int = 0, staged: int = 0, documents: bool = False) -> None:
    """插入待保护旧记录。参数：会话工厂、旧块数、候选数、文档开关；返回：None；异常：SQL错误。"""
    async with factory.begin() as session:
        for index in range(1, chunks + 1):
            source = f"doc:old-{index}.md" if documents and index > 1 else f"qa:{index}"
            await session.execute(text("""
                INSERT INTO knowledge_chunks
                    (id, source_key, ordinal, category, questions, answer, content_type,
                     is_key_clause, content_hash, embedding_revision, is_active, vectorize_status)
                VALUES (:id, :source, 0, '租约示例', :question, :answer, 'faq', 0,
                        :hash, 'fake-encoder-v1', 1, 'pending')
            """), {"id": index, "source": source, "question": f"旧问题{index}",
                   "answer": f"旧答案{index}。", "hash": "a" * 64})
        for index in range(1, staged + 1):
            await session.execute(text("""
                INSERT INTO qa_extraction_staging
                    (id, batch_no, source_ids, source_fingerprint, question, answer,
                     category, status, evidence_refs)
                VALUES (:id, 'lease-batch', '[1]', :fingerprint, :question, :answer,
                        '租约示例', 'extracted', '[]')
            """), {"id": index, "fingerprint": str(index), "question": f"旧问题{index}",
                   "answer": f"旧答案{index}。"})


async def _state(factory) -> tuple[list, list]:
    """从独立会话读取全部持久状态。参数：工厂；返回：知识行和暂存行；异常：SQL错误。"""
    async with factory() as session:
        chunks = (await session.execute(text("SELECT * FROM knowledge_chunks ORDER BY id"))).all()
        staged = (await session.execute(text("SELECT * FROM qa_extraction_staging ORDER BY id"))).all()
        return chunks, staged


def _lifecycle(encoder, model, **kwargs) -> KnowledgeLifecycle:
    """组合真实生命周期。参数：外部依赖和检查点；返回：生命周期；异常：字段校验失败。"""
    policy = RelationPolicy(min_score=0.0, embedding_revision=encoder.revision,
                            prompt_version="lease-test", calibration_digest="lease-test")
    return KnowledgeLifecycle(encoder, model, policy, **kwargs)


@pytest.mark.parametrize("entry,loss_batch", [("documents", 1), ("document_mined", 1),
                                             ("staging", 1), ("authority", 3)])
async def test_relation_batch_loss_stops_following_requests_and_preserves_rows(lease_runtime, entry, loss_batch):
    """任一关系入口失锁后不得新增模型批次。参数：隔离资源/入口/失锁批次；返回：None；异常：断言失败。"""
    writer, lease, factory = lease_runtime
    if entry == "document_mined":
        await _seed(factory, chunks=9)
    elif entry == "staging":
        await _seed(factory, staged=6)
    elif entry == "authority":
        # 新候选与 qa:1 精确一致，先判九份文档，再补判旧知识与九份文档。
        await _seed(factory, chunks=10, staged=1, documents=True)
    before = await _state(factory)
    encoder = _EncoderBoundary(lease)
    model = _RelationBoundary(lease, lose_at=loss_batch)
    lifecycle = _lifecycle(encoder, model)
    with pytest.raises(LeaseLost) as error:
        if entry in ("staging", "authority"):
            await lifecycle.reconcile_staging(writer)
        else:
            drafts = _drafts(6 if entry == "documents" else 1)
            await lifecycle.publish_documents(writer, drafts, [row.source_key for row in drafts])
    assert error.value is lease.error
    assert len(model.calls) == loss_batch, "失锁后仍向外部模型发送了后续关系批次"
    assert len(model.calls[-1]) == 8
    assert await _state(factory) == before, "失锁不得改变旧知识或暂存审计状态"


@pytest.mark.parametrize("malformed", [False, True])
async def test_encoding_loss_stops_model_and_keeps_original_error(lease_runtime, malformed):
    """编码失锁后不能判关系或用响应校验错误盖住失锁。参数：资源/畸形开关；返回：None；异常：断言失败。"""
    writer, lease, factory = lease_runtime
    await _seed(factory, chunks=1)
    before = await _state(factory)
    encoder = _EncoderBoundary(lease, loss="after_encode", malformed=malformed)
    model = _RelationBoundary(lease)
    with pytest.raises(LeaseLost) as error:
        await _lifecycle(encoder, model).publish_documents(writer, _drafts(1), ["doc:new-0.md"])
    assert error.value is lease.error
    assert encoder.inputs and not model.calls
    assert await _state(factory) == before


async def test_loss_before_encoding_prevents_new_external_work(lease_runtime):
    """候选预检查时失锁不得再开始编码。参数：隔离资源；返回：None；异常：断言失败。"""
    writer, lease, factory = lease_runtime
    await _seed(factory, staged=2)
    before = await _state(factory)
    encoder = _EncoderBoundary(lease, loss="before_encode")
    model = _RelationBoundary(lease)
    with pytest.raises(LeaseLost) as error:
        await _lifecycle(encoder, model).reconcile_staging(writer)
    assert error.value is lease.error
    assert not encoder.inputs and not model.calls
    assert await _state(factory) == before


async def test_relation_response_cannot_mask_original_lease_loss(lease_runtime):
    """失锁与畸形关系响应同时发生时优先传播原失锁。参数：隔离资源；返回：None；异常：断言失败。"""
    writer, lease, factory = lease_runtime
    model = _RelationBoundary(lease, lose_at=1, malformed=True)
    with pytest.raises(LeaseLost) as error:
        await _lifecycle(_EncoderBoundary(lease), model).publish_documents(
            writer, _drafts(6), [row.source_key for row in _drafts(6)])
    assert error.value is lease.error
    assert len(model.calls) == 1
    assert await _state(factory) == ([], [])


@pytest.mark.parametrize("count,old_chunks", [(1, 0), (6, 0), (1, 1)])
async def test_valid_lease_publishes_all_documents_with_or_without_pairs(lease_runtime, count, old_chunks):
    """有效租约下有编码、无编码及无关系对均能发布。参数：资源/新旧数量；返回：None；异常：断言失败。"""
    writer, lease, factory = lease_runtime
    await _seed(factory, chunks=old_chunks)
    model = _RelationBoundary(lease)
    encoder = _EncoderBoundary(lease)
    drafts = _drafts(count)
    published = await _lifecycle(encoder, model).publish_documents(writer, drafts, [row.source_key for row in drafts])
    assert len(published) == count
    assert len((await _state(factory))[0]) == count + old_chunks
    expected_pairs = [1] if old_chunks else ([] if count == 1 else [8, 7])
    assert [len(pairs) for pairs in model.calls] == expected_pairs
    assert bool(encoder.inputs) == bool(old_chunks)


async def test_no_pairs_still_rejects_loss_at_final_publication(lease_runtime):
    """无关系对时仍保留真实writer最终发布核锁。参数：隔离资源；返回：None；异常：断言失败。"""
    writer, lease, factory = lease_runtime
    model = _RelationBoundary(lease)

    def lose_before_publish(label: str) -> None:
        """最终发布前改变锁所有者。参数：检查点标签；返回：None；异常：无。"""
        if label == "before_document_publish":
            lease.lost = True

    with pytest.raises(LeaseLost) as error:
        await _lifecycle(_EncoderBoundary(lease), model, checkpoint=lose_before_publish).publish_documents(
            writer, _drafts(1), ["doc:new-0.md"])
    assert error.value is lease.error
    assert not model.calls
    assert await _state(factory) == ([], [])
