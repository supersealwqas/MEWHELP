"""知识库运行时组装、显式初始化与只读状态。

运行时按应用配置连接 MySQL、独立 Milvus 数据库、本地编码器和项目聊天模型，
加载版本匹配的关系策略；初始化/状态查询不加载模型，也不启动后台长任务。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.core.llm import get_chat_model
from app.db.session import create_database
from app.knowledge.embedding import BgeM3Encoder
from app.knowledge.hybrid_index import MilvusHybridIndex
from app.knowledge.jobs import KnowledgeJobs
from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.llm import KnowledgeModel, load_relation_policy
from app.knowledge.prompts import RELATION_PROMPT_VERSION
from app.knowledge.rerank import BgeReranker
from app.knowledge.retrieval import HybridKnowledgeSearch
from app.knowledge.search import KnowledgeSearch, load_retrieval_policy
from app.knowledge.chunking import CHUNKING_VERSION as LEGACY_CHUNKING_VERSION, chunk_markdown
from app.knowledge.documents import build_document_chunks
from app.knowledge.schemas import ChunkSnapshot, KnowledgeText, SyncReport
from app.knowledge.store import KnowledgeStore
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint
from app.knowledge.vector import MilvusVectorIndex, VectorIndexUnavailable
from app.models.knowledge import (
    KnowledgeBase,
    KnowledgeChunk,
    QAExtractionBatch,
    QAExtractionStaging,
)


@asynccontextmanager
async def knowledge_runtime(config: Settings):
    """组装一次知识导入/挖掘运行时并保证资源关闭。

    参数:
        config: 应用或测试配置。
    返回值:
        AsyncContextManager[KnowledgeJobs]: 可执行任务。
    异常:
        VectorIndexUnavailable、KnowledgeModelError、数据库/模型错误。
    """
    encoder = index = None
    try:
        async with knowledge_storage(config) as store:
            encoder = BgeM3Encoder(config)
            index = (MilvusHybridIndex(config) if config.knowledge_retrieval_mode == "hybrid"
                     else MilvusVectorIndex(config))
            await index.ensure_ready(create=False)
            model = KnowledgeModel(get_chat_model)
            policy = load_relation_policy(
                Path(config.knowledge_relation_policy_path),
                embedding_revision=encoder.revision,
                prompt_version=RELATION_PROMPT_VERSION,
            )
            lifecycle = KnowledgeLifecycle(encoder, model, policy)
            synchronizer = VectorSynchronizer(store, encoder, index)
            chunker = build_document_chunks if config.knowledge_retrieval_mode == "hybrid" else chunk_markdown
            yield KnowledgeJobs(store, encoder, index, model, lifecycle, synchronizer, document_chunker=chunker)
    finally:
        with anyio.CancelScope(shield=True):
            try:
                if encoder is not None:
                    await encoder.aclose()
            finally:
                if index is not None:
                    await index.aclose()


@asynccontextmanager
async def knowledge_storage(config: Settings):
    """建立独立权威原文与写锁连接，业务退出或取消时保证资源关闭。

    参数：config 为已确定 MySQL 目标库的配置。
    返回：AsyncContextManager[KnowledgeStore]，不启动模型或向量连接。
    异常：数据库连接/关闭错误，取消在屏蔽清理完成后传播。
    """
    engine, factory = create_database(config)
    lease_engine = None
    try:
        lease_engine = create_async_engine(engine.url, poolclass=NullPool, connect_args={"connect_timeout": 5})
        yield KnowledgeStore(factory, lease_engine)
    finally:
        with anyio.CancelScope(shield=True):
            try:
                if lease_engine is not None:
                    await lease_engine.dispose()
            finally:
                await engine.dispose()


@asynccontextmanager
async def online_search_runtime(config: Settings, store: KnowledgeStore):
    """组装显式混合或旧 Dense 在线路径并预热，不在故障时改变模式。

    参数：config 决定模式、模型和新旧集合；store 为当前 MySQL 权威读取器。
    返回：AsyncContextManager[HybridKnowledgeSearch | KnowledgeSearch]。
    异常：索引/模型/策略错误阻止启动；取消清理模型/客户端后传播。
    """
    encoder = index = reranker = None
    try:
        encoder = BgeM3Encoder(config)
        if config.knowledge_retrieval_mode == "hybrid":
            index = MilvusHybridIndex(config)
            await index.ensure_ready(create=False)
            reranker = BgeReranker(config)
            await encoder.encode(["知识库启动预热"])
            text = KnowledgeText(category="运行检查", questions="预热", answer="仅检查模型可用，不是知识库事实。")
            warmup = ChunkSnapshot(id=1, source_key="runtime:warmup", ordinal=0, text=text,
                                    content_type="runtime", content_hash=content_fingerprint(text),
                                    embedding_revision=encoder.revision)
            await reranker.rank("模型启动检查", [warmup])
            yield HybridKnowledgeSearch(store, encoder, index, reranker, config)
        else:
            index = MilvusVectorIndex(config)
            await index.ensure_ready(create=False)
            policy = load_retrieval_policy(
                Path(config.knowledge_policy_path), embedding_revision=encoder.revision,
                chunking_version=LEGACY_CHUNKING_VERSION,
            )
            await encoder.encode(["知识库启动预热"])
            yield KnowledgeSearch(store, encoder, index, policy)
    finally:
        # 清理三个独立资源，关闭某一个失败也必须继续释放其余资源。
        with anyio.CancelScope(shield=True):
            try:
                if reranker is not None:
                    await reranker.aclose()
            finally:
                try:
                    if encoder is not None:
                        await encoder.aclose()
                finally:
                    if index is not None:
                        await index.aclose()


async def prepare_hybrid_index(config: Settings, *, confirmed_database: str | None = None) -> SyncReport:
    """在现有知识写锁内创建/补齐独立新混合集合，保留原 Dense 集合。

    参数：config 为应用或显式 CH04 测试配置；confirmed_database 为操作人确认的空库名，不自动推断。
    返回：SyncReport，现有 MySQL 权威记录同步到新集合的结果。
    异常：ValueError/VectorIndexUnavailable，非法确认、缺权威库/模型、外来资源；存储/租约错误。
    """
    if confirmed_database is not None and confirmed_database != config.milvus_database:
        raise VectorIndexUnavailable("Milvus 归属确认必须匹配目标数据库")
    encoder = index = None
    try:
        async with knowledge_storage(config) as store:
            async with store.writer() as writer:
                await writer.assert_held()
                encoder = BgeM3Encoder(config)
                index = MilvusHybridIndex(config)
                await index.ensure_ready(create=True, confirmed_database=confirmed_database)
                await writer.assert_held()
                return await VectorSynchronizer(store, encoder, index).run(writer)
    finally:
        with anyio.CancelScope(shield=True):
            try:
                if encoder is not None:
                    await encoder.aclose()
            finally:
                if index is not None:
                    await index.aclose()


async def initialize_knowledge(
    config: Settings, *, confirmed_database: str | None = None,
) -> None:
    """显式创建缺失的 MySQL 知识表和独立 Milvus 集合。

    参数:
        config: 应用或测试配置。
        confirmed_database: 操作人确认归本项目的既存空 Milvus 库名；None 时不接管未知库。
    返回值:
        None
    异常:
        VectorIndexUnavailable: 确认库名不匹配或 Milvus 归属未知。
        数据库/向量初始化错误。
    """
    if confirmed_database is not None and confirmed_database != config.milvus_database:
        raise VectorIndexUnavailable("Milvus 归属确认必须匹配初始化的目标数据库")
    engine, _ = create_database(config)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(KnowledgeBase.metadata.create_all)
    finally:
        await engine.dispose()
    index = MilvusVectorIndex(config)
    try:
        await index.ensure_ready(create=True, confirmed_database=confirmed_database)
    finally:
        await index.aclose()


async def read_knowledge_status(config: Settings) -> dict:
    """只读汇总知识发布、暂存、批次与隔离状态。

    参数:
        config: 应用或测试配置。
    返回值:
        dict: 计数摘要。
    异常:
        数据库错误。
    """
    engine, factory = create_database(config)
    try:
        async with factory() as session:
            def count(query):
                """执行计数查询。参数：查询；返回：整数；异常：数据库错误。"""
                return session.scalar(query)

            active_pending = await count(
                select(func.count())
                .select_from(KnowledgeChunk)
                .where(
                    KnowledgeChunk.is_active.is_(True),
                    KnowledgeChunk.vectorize_status == "pending",
                )
            )
            active_done = await count(
                select(func.count())
                .select_from(KnowledgeChunk)
                .where(
                    KnowledgeChunk.is_active.is_(True),
                    KnowledgeChunk.vectorize_status == "done",
                )
            )
            inactive = await count(
                select(func.count())
                .select_from(KnowledgeChunk)
                .where(KnowledgeChunk.is_active.is_(False))
            )
            staged_extracted = await count(
                select(func.count())
                .select_from(QAExtractionStaging)
                .where(QAExtractionStaging.status == "extracted")
            )
            staged_kept = await count(
                select(func.count())
                .select_from(QAExtractionStaging)
                .where(QAExtractionStaging.status == "kept")
            )
            staged_discarded = await count(
                select(func.count())
                .select_from(QAExtractionStaging)
                .where(QAExtractionStaging.status == "discarded")
            )
            conflict_groups = await count(
                select(func.count(func.distinct(QAExtractionStaging.conflict_group))).where(
                    QAExtractionStaging.conflict_group.is_not(None)
                )
            )
            batches_pending = await count(
                select(func.count())
                .select_from(QAExtractionBatch)
                .where(QAExtractionBatch.status == "pending")
            )
            batches_done = await count(
                select(func.count())
                .select_from(QAExtractionBatch)
                .where(QAExtractionBatch.status == "done")
            )
        return {
            "active_pending": int(active_pending or 0),
            "active_done": int(active_done or 0),
            "inactive": int(inactive or 0),
            "staged_extracted": int(staged_extracted or 0),
            "staged_kept": int(staged_kept or 0),
            "staged_discarded": int(staged_discarded or 0),
            "conflict_groups": int(conflict_groups or 0),
            "batches_pending": int(batches_pending or 0),
            "batches_done": int(batches_done or 0),
        }
    finally:
        await engine.dispose()
