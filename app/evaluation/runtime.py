"""独立CH04评估资源、冻结语料及真实模型运行时。

所有写入严格限制在mewhelp_ch04_eval与knowledge_ch04_eval；先验证资源身份，
不借用应用表初始化、不清库、不启动服务。生成与裁判仅收到真实模型证据。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings
from app.core.llm import ChatModelAdapter
from app.core.memory import _estimate_message_tokens, _estimate_token_units
from app.db.session import create_database
from app.evaluation.faithfulness import FaithfulnessJudge, _JUDGE_SYSTEM
from app.evaluation.policy import build_corpus, corpus_digest, digest_json
from app.knowledge.answering import (
    EvidenceCitation, KNOWLEDGE_ANSWER_HANDOFF, KNOWLEDGE_ANSWER_RULES, KnowledgeAnswer,
    edge_order, knowledge_answer_handoff,
)
from app.knowledge.documents import CHUNKING_VERSION
from app.knowledge.embedding import BgeM3Encoder, EMBEDDING_REVISION
from app.knowledge.hybrid_index import MilvusHybridIndex
from app.knowledge.rerank import BgeReranker, RERANKER_REVISION
from app.knowledge.query_rewrite import QueryRewriter, QUERY_REWRITE_RULES
from app.knowledge.retrieval import EvaluationSearch, PreparedQuery, prepare_query
from app.knowledge.runtime import knowledge_storage
from app.knowledge.schemas import ChunkSnapshot
from app.knowledge.store import _snapshot_from_row
from app.knowledge.sync import VectorSynchronizer
from app.models.knowledge import KnowledgeChunk

EVAL_DATABASE = "mewhelp_ch04_eval"
EVAL_COLLECTION = "knowledge_ch04_eval"
EVAL_MARKER = "mewhelp:ch04:evaluation:v1"
QUERY_RULES_VERSION = "single-turn:rewrite1:synonyms4:clauses4:policy3:50-50-50:per-clause10:v2"
_IDENTITY_TABLE = "ch04_eval_identity"
_LOCK_NAME = "mewhelp:ch04:evaluation:exclusive"


def eval_config(config: Settings, *, target: str) -> Settings:
    """显式选择唯一评估资源。参数：config 原连接配置、target 必须eval；返回：复制配置；异常：ValueError。"""
    if target != "eval":
        raise ValueError("评估目标必须明确为eval")
    result = config.model_copy(update={"mysql_database": EVAL_DATABASE, "milvus_database": EVAL_DATABASE,
                                      "milvus_hybrid_collection": EVAL_COLLECTION,
                                      "milvus_collection": "knowledge_eval_legacy_unused",
                                      "knowledge_retrieval_mode": "hybrid"})
    validate_resources(result)
    return result


def validate_resources(config: Settings) -> None:
    """拒绝任何半边应用资源。参数：config 已选择配置；返回：None；异常：ValueError，名称不符。"""
    if (config.mysql_database != EVAL_DATABASE or config.milvus_database != EVAL_DATABASE
            or config.milvus_hybrid_collection != EVAL_COLLECTION
            or config.milvus_collection == EVAL_COLLECTION):
        raise ValueError("评估必须同时使用专用MySQL/Milvus库及独占集合")


def check_identity(tables: set[str], record: dict | None) -> None:
    """验证既存库归属，不能仅凭库名接管其他数据。

    参数：tables 真实表名集合；record 唯一身份行。
    返回：None。
    异常：ValueError，缺少身份、外来表或标记不匹配。
    """
    if tables - {_IDENTITY_TABLE, "knowledge_chunks"}:
        raise ValueError("专用评估库含外来表，拒绝接管")
    if not record or record.get("marker") != EVAL_MARKER or record.get("collection_name") != EVAL_COLLECTION:
        raise ValueError("既存评估库身份或集合归属不匹配")


def local_corpus(manifest_path: Path, *, count_tokens=len) -> list[ChunkSnapshot]:
    """从清单构建真实400/60/10切分的稳定语料快照。

    参数：manifest_path 资料清单；count_tokens 真实导入使用编码器计数，离线结构检查默认字符计数。
    返回：快照列表，临时主键只用于解析，摘要不含这些主键。
    异常：清单/切分/文件错误。
    """
    return build_corpus(manifest_path, embedding_revision=EMBEDDING_REVISION, count_tokens=count_tokens)


def generation_input(query: str, selected: list, *, evidence_budget: int, request_budget: int) -> tuple[list, list[EvidenceCitation], list[int]]:
    """组装生成实际输入并只移除末尾整块，保留原始稳定编号。

    参数：query 原问；selected 原序候选；evidence_budget 证据估算单位；request_budget 整请求估算单位。
    返回：模型消息、实际证据全集、被预算丢弃主键。
    异常：ValueError，预算非法或仅原问/系统提示已超限。
    """
    if evidence_budget < 1 or request_budget < 1:
        raise ValueError("生成预算必须为正整数")
    citations = []
    seen = set()
    for item in selected:
        if item.chunk.id in seen:
            continue
        seen.add(item.chunk.id)
        citations.append(EvidenceCitation(n=len(citations) + 1, chunk_id=item.chunk.id,
                                           question=item.chunk.text.questions, answer=item.chunk.text.answer,
                                           section_path=item.chunk.section_path))
    dropped = []
    while True:
        ordered = edge_order(citations)
        # 与线上一致：chunk_id 只留服务端，模型只看 n，避免拿主键当角标。
        payload = [item.model_dump(exclude={"chunk_id"}) for item in ordered]
        messages = [SystemMessage(content=KNOWLEDGE_ANSWER_RULES),
                    HumanMessage(content=json.dumps({**knowledge_answer_handoff(query), "hits": payload}, ensure_ascii=False))]
        if _estimate_token_units(payload) <= evidence_budget and _estimate_message_tokens(messages) <= request_budget:
            return messages, ordered, dropped
        if not citations:
            raise ValueError("原问和系统提示已经超出完整请求预算")
        dropped.append(citations.pop().chunk_id)


def file_digest(path: Path) -> str:
    """流式计算本地文件指纹。参数：path 真实文件；返回：SHA256；异常：OSError。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_digest(path: Path) -> str:
    """记录实际本地模型配置和权重，不以路径名冒充版本。

    参数：path 模型根目录；返回：内容清单SHA256。
    异常：ValueError，目录或权重缺失；OSError，文件不可读。
    """
    files = sorted(file for file in path.rglob("*") if file.is_file() and ".cache" not in file.parts
                   and file.suffix in {".json", ".bin", ".safetensors", ".model", ".txt"})
    if not files or not any(file.suffix in {".bin", ".safetensors"} for file in files):
        raise ValueError("本地模型目录没有可核验权重")
    return digest_json({file.relative_to(path).as_posix(): file_digest(file) for file in files})


def runtime_metadata(config: Settings, chunks: list[ChunkSnapshot], *, evidence_budget: int) -> dict:
    """冻结模型、语料、提示词及源码版本，排除密钥。

    参数：config 运行配置；chunks 固定语料；evidence_budget 本轮证据预算。
    返回：dict，可写报告的版本标识。
    异常：OSError/ValueError，源码或模型不可核验。
    """
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "app").rglob("*.py")) + [root / "uv.lock"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return {"corpus_digest": corpus_digest(chunks), "chunk_count": len(chunks),
            "model": config.chat_model, "judge_model": config.judge_model or config.chat_model,
            "model_endpoint_digest": hashlib.sha256(config.chat_base_url.encode()).hexdigest(),
            "embedding_revision": EMBEDDING_REVISION, "reranker_revision": RERANKER_REVISION,
            "embedding_artifacts_digest": model_digest(Path(config.knowledge_model_path)),
            "reranker_artifacts_digest": model_digest(Path(config.rerank_model_path)),
            "prompt_digest": digest_json({"generation": KNOWLEDGE_ANSWER_RULES, "handoff": KNOWLEDGE_ANSWER_HANDOFF,
                                          "judge": _JUDGE_SYSTEM, "query_rewrite": QUERY_REWRITE_RULES}),
            "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "source_digest": digest_json({path.relative_to(root).as_posix(): file_digest(path) for path in sources}),
            "chunking_version": CHUNKING_VERSION, "query_rules_version": QUERY_RULES_VERSION,
            "request_budget": config.request_token_budget, "evidence_budget": evidence_budget,
            "generation_max_tokens": 4096, "temperature": 0.0,
            "mysql_database": EVAL_DATABASE, "milvus_database": EVAL_DATABASE, "collection": EVAL_COLLECTION}


@asynccontextmanager
async def _resource_lock(config: Settings, *, create: bool):
    """在唯一评估资源上持命名锁，阻止准备和比较交错修改语料。

    参数：config 已限制配置；create 是否显式允许创建数据库。
    返回：AsyncConnection，持锁到上下文退出。
    异常：ValueError，身份或锁冲突；数据库错误；取消在清理后传播。
    """
    validate_resources(config)
    target_engine, _ = create_database(config)
    # URL.set(None)表示保留旧值；首次创建库必须真正移除数据库组件。
    server = create_async_engine(target_engine.url._replace(database=None), connect_args={"connect_timeout": 5})
    try:
        async with server.connect() as connection:
            acquired = await connection.scalar(text("SELECT GET_LOCK(:name, 0)"), {"name": _LOCK_NAME})
            if acquired != 1:
                raise ValueError("另一个评估准备或运行正在占用专用资源")
            try:
                exists = await connection.scalar(text("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME=:name"), {"name": EVAL_DATABASE})
                if not exists:
                    if not create:
                        raise ValueError("评估库尚未prepare")
                    await connection.execute(text("CREATE DATABASE `mewhelp_ch04_eval` CHARACTER SET utf8mb4"))
                await connection.execute(text("USE `mewhelp_ch04_eval`"))
                if await connection.scalar(text("SELECT DATABASE()")) != EVAL_DATABASE:
                    raise ValueError("评估物理连接数据库身份不符")
                await connection.commit()
                yield connection
            finally:
                with anyio.CancelScope(shield=True):
                    if connection.in_transaction():
                        await connection.rollback()
                    await connection.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": _LOCK_NAME})
                    await connection.commit()
    finally:
        with anyio.CancelScope(shield=True):
            await server.dispose()
            await target_engine.dispose()


async def _read_identity(connection, *, create: bool) -> dict:
    """只在空库登记身份，已存在身份必须匹配且不容许外来表。

    参数：connection 专用持锁连接；create 是否允许首次登记。
    返回：dict，唯一资源身份和准备状态。
    异常：ValueError，外来表、重复身份或缺失准备；数据库错误。
    """
    tables = set(await connection.run_sync(lambda sync: inspect(sync).get_table_names(schema=EVAL_DATABASE)))
    if not tables and create:
        await connection.execute(text("""CREATE TABLE ch04_eval_identity (
            id INT PRIMARY KEY, marker VARCHAR(64) NOT NULL, collection_name VARCHAR(128) NOT NULL,
            corpus_digest VARCHAR(64) NULL, state VARCHAR(16) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='MewHelp CH04独立评估资源身份'"""))
        await connection.execute(text("INSERT INTO ch04_eval_identity (id,marker,collection_name,state) VALUES (1,:marker,:collection,'building')"),
                                 {"marker": EVAL_MARKER, "collection": EVAL_COLLECTION})
        await connection.commit()
        tables = {_IDENTITY_TABLE}
    if _IDENTITY_TABLE not in tables:
        raise ValueError("既存评估库没有身份标记，拒绝接管")
    records = (await connection.execute(text("SELECT * FROM ch04_eval_identity"))).mappings().all()
    if len(records) != 1 or records[0].get("id") != 1:
        raise ValueError("评估身份记录必须唯一")
    record = dict(records[0])
    check_identity(tables, record)
    if not create and (record.get("state") != "ready" or "knowledge_chunks" not in tables):
        raise ValueError("评估资源尚未完整准备")
    if connection.in_transaction():
        await connection.commit()
    return record


async def _all_chunks(store) -> list[ChunkSnapshot]:
    """分页读取全部有效权威原文。参数：store 专用存储；返回：快照；异常：数据库错误。"""
    chunks = []
    while True:
        page = await store.active_page(chunks[-1].id if chunks else 0, 500)
        if not page:
            return chunks
        chunks.extend(page)


async def _verify_vectors(index, chunks: list[ChunkSnapshot]) -> None:
    """确认索引没有遗漏或外来ID且与完整权威指纹一致。

    参数：index 专用索引；chunks 冻结原文。
    返回：None。
    异常：ValueError，索引多余/缺失/不同步；索引错误。
    """
    identifiers = set()
    async for page in index.iter_ids(500):
        identifiers.update(page)
    if identifiers != {row.id for row in chunks}:
        raise ValueError("评估索引ID全集与权威语料不一致")
    for offset in range(0, len(chunks), 500):
        page = chunks[offset:offset + 500]
        fingerprints = await index.get_fingerprints([row.id for row in page])
        if any(row.vectorize_status != "done" or row.vector_id != str(row.id)
               or fingerprints.get(row.id) != (row.content_hash, row.embedding_revision) for row in page):
            raise ValueError("评估原文尚未完整同步到索引")


async def _ensure_chunk_table(connection) -> None:
    """幂等创建知识表并恢复DDL中断时遗失的普通索引。

    参数：connection 带评估数据库URL的目标连接。
    返回：None。
    异常：数据库权限/DDL错误；不会删除既有表或索引。
    """
    await connection.run_sync(lambda sync: KnowledgeChunk.__table__.create(sync, checkfirst=True))
    # MySQL建表与建索引不是一个原子事务；表存在不能证明此前所有CREATE INDEX均成功。
    for index in sorted(KnowledgeChunk.__table__.indexes, key=lambda item: item.name):
        await connection.run_sync(lambda sync: index.create(sync, checkfirst=True))


async def prepare_resources(config: Settings, manifest_path: Path) -> dict:
    """显式准备专用语料与独占索引，已冻结语料不覆盖成新版本。

    参数：config eval_config结果；manifest_path 已登记六份原文清单。
    返回：dict，准备状态、同步数量和跨库稳定语料摘要。
    异常：ValueError，外来资源或语料变化；数据库/索引/本地模型错误，保留building状态以便同语料重试。
    """
    validate_resources(config)
    encoder, index = BgeM3Encoder(config), MilvusHybridIndex(config)
    try:
        chunks = local_corpus(manifest_path, count_tokens=encoder.count_tokens)
        expected = corpus_digest(chunks)
        async with _resource_lock(config, create=True) as connection:
            identity = await _read_identity(connection, create=True)
            if identity.get("corpus_digest") not in (None, expected):
                raise ValueError("评估库已冻结为其他语料，禁止覆盖既有评估资源")
            engine, factory = create_database(config)
            try:
                # 身份锁仍保留在引导连接；带库名的新连接负责DDL，checkfirst不会反射None默认schema。
                async with engine.begin() as target_connection:
                    await _ensure_chunk_table(target_connection)
                async with factory.begin() as session:
                    existing = list((await session.scalars(select(KnowledgeChunk))).all())
                    if existing:
                        if corpus_digest([_snapshot_from_row(row) for row in existing]) != expected:
                            raise ValueError("专用库现有语料和本次清单不一致，拒绝重写")
                    else:
                        for chunk in chunks:
                            session.add(KnowledgeChunk(source_key=chunk.source_key, ordinal=chunk.ordinal,
                                category=chunk.text.category, questions=chunk.text.questions, answer=chunk.text.answer,
                                section_path=chunk.section_path, content_type=chunk.content_type, is_key_clause=chunk.is_key_clause,
                                content_hash=chunk.content_hash, embedding_revision=chunk.embedding_revision,
                                is_active=True, vectorize_status="pending"))
                await connection.execute(text("UPDATE ch04_eval_identity SET corpus_digest=:digest,state='building' WHERE id=1"), {"digest": expected})
                await connection.commit()
            finally:
                await engine.dispose()
            await index.ensure_ready(create=True, confirmed_database=EVAL_DATABASE)
            async with knowledge_storage(config) as store:
                async with store.writer() as writer:
                    sync = await VectorSynchronizer(store, encoder, index).run(writer)
                actual = await _all_chunks(store)
                if corpus_digest(actual) != expected:
                    raise ValueError("同步后的语料摘要与准备输入不同")
                await _verify_vectors(index, actual)
            await connection.execute(text("UPDATE ch04_eval_identity SET state='ready' WHERE id=1"))
            await connection.commit()
            return {"status": "ready", "database": EVAL_DATABASE, "collection": EVAL_COLLECTION,
                    "corpus_digest": expected, "chunk_count": len(chunks), "sync": sync.model_dump()}
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await encoder.aclose()
            finally:
                await index.aclose()


class EvaluationRuntime:
    """只读知识、生成与独立裁判的评估实例，没有应用写仓储。"""

    def __init__(self, search, chunks, config, *, evidence_budget: int, metadata: dict) -> None:
        """保存冻结依赖。参数：search/chunks/config/evidence_budget/metadata；返回：None；异常：无。"""
        self.search, self.chunks, self.config = search, chunks, config
        self.evidence_budget, self.metadata = evidence_budget, metadata
        self.answer_model = ChatModelAdapter()
        self.query_rewriter = QueryRewriter(request_budget=config.request_token_budget)
        # 裁判可显式换用其它模型，避免生成与评判同一模型带来的自评偏差。
        self.judge_model = FaithfulnessJudge(model_name=config.judge_model or config.chat_model)

    async def prepare(self, query: str) -> PreparedQuery:
        """只按原问准备一次，不接收题集答案。参数：query 当前原问；返回：四策略共用结果；异常：改写/预算错误。"""
        rewrite = await self.query_rewriter.rewrite(query)
        return prepare_query(query, policy=any(term in query for term in ("退款", "退货", "退换")),
                             subquery_split=self.config.subquery_split, rewrite=rewrite)

    async def retrieve(self, prepared, strategy, filters):
        """读取门禁前排名。参数：本题缓存/策略/过滤；返回：检索结果；异常：检索或模型错误。"""
        return await self.search.retrieve(prepared, strategy=strategy, corpus=self.chunks, filters=filters)

    async def generate(self, query, selected):
        """以统一Prompt和预算生成知识自评。参数：query 原问、selected 实际候选；返回：decision/citations；异常：模型错误。"""
        messages, citations, _ = generation_input(query, selected, evidence_budget=self.evidence_budget,
                                                   request_budget=self.config.request_token_budget)
        if not citations:
            return KnowledgeAnswer(useful=False, reason="完整证据无法放入生成预算", answer="", citation_numbers=[]), []
        return await self.answer_model.answer_knowledge(messages), citations

    async def judge(self, query, answer, citations):
        """独立调用既有FaithfulnessJudge。参数：问答与实际全集；返回：逐事实判定；异常：裁判/网络错误。"""
        return await self.judge_model.judge(query, answer, citations)


@asynccontextmanager
async def evaluation_runtime(config: Settings, *, manifest_path: Path, evidence_budget: int = 8000):
    """核验已准备资源和模型，持独占锁提供只读评估。

    参数：config 专用配置；manifest_path 当前原文清单；evidence_budget 统一证据估算预算。
    返回：EvaluationRuntime，上下文退出释放全部模型与连接。
    异常：ValueError，未prepare/原文改变/索引不一致；外部资源错误。
    """
    validate_resources(config)
    encoder, reranker, index = BgeM3Encoder(config), BgeReranker(config), MilvusHybridIndex(config)
    try:
        async with _resource_lock(config, create=False) as connection:
            identity = await _read_identity(connection, create=False)
            async with knowledge_storage(config) as store:
                chunks = await _all_chunks(store)
                expected = corpus_digest(local_corpus(manifest_path, count_tokens=encoder.count_tokens))
                if identity["corpus_digest"] != expected or corpus_digest(chunks) != expected:
                    raise ValueError("评估库、当前原文和资源身份的语料摘要不一致")
                await index.ensure_ready(create=False)
                await _verify_vectors(index, chunks)
                metadata = runtime_metadata(config, chunks, evidence_budget=evidence_budget)
                yield EvaluationRuntime(EvaluationSearch(store, encoder, index, reranker, config), chunks, config,
                                        evidence_budget=evidence_budget, metadata=metadata)
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await reranker.aclose()
            finally:
                try:
                    await encoder.aclose()
                finally:
                    await index.aclose()
