"""Milvus 2.4.x Dense 向量索引 Adapter。

集合只保存 id、1024 维向量、内容指纹与编码规范，不保存权威正文；初始化
只创建缺失资源并严格校验 description、schema、COSINE 索引与一致性，遇到
来源不明或不兼容集合立即失败，绝不以 drop/recreate 修复。
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable

import anyio

from app.config import Settings
from app.knowledge.schemas import KnowledgeUnavailable, VectorHit, VectorRecord

OWNERSHIP_MARKER = "mewhelp-ch03:knowledge:v1"
VECTOR_DIM = 1024
MAX_SIGNED_INT64 = 2**63 - 1
_RPC_TIMEOUT_SECONDS = 10.0
_INDEX_TYPE = "AUTOINDEX"
_METRIC_TYPE = "COSINE"


class VectorIndexUnavailable(KnowledgeUnavailable):
    """Milvus 索引不可用或所有权校验失败。"""


def _create_bootstrap_client(config: Settings) -> Any:
    """创建连接默认数据库的引导客户端。

    参数:
        config: Milvus 连接配置。
    返回值:
        MilvusClient: 用于检查/创建独立数据库的客户端。
    异常:
        Exception: 连接配置不可用时由 SDK 抛出。
    """
    from pymilvus import MilvusClient

    return MilvusClient(
        uri=config.milvus_uri,
        token=config.milvus_token.get_secret_value(),
        timeout=_RPC_TIMEOUT_SECONDS,
    )


def _create_client(config: Settings) -> Any:
    """创建连接目标数据库的客户端。

    参数:
        config: Milvus 连接配置。
    返回值:
        MilvusClient: 目标数据库客户端。
    异常:
        Exception: 连接配置不可用时由 SDK 抛出。
    """
    from pymilvus import MilvusClient

    return MilvusClient(
        uri=config.milvus_uri,
        db_name=config.milvus_database,
        token=config.milvus_token.get_secret_value(),
        timeout=_RPC_TIMEOUT_SECONDS,
    )


class MilvusVectorIndex:
    """有所有权校验的 Milvus Dense 索引。"""

    def __init__(
        self,
        config: Settings,
        *,
        client_factory: Callable[[Settings], Any] | None = None,
        bootstrap_factory: Callable[[Settings], Any] | None = None,
    ) -> None:
        """保存配置与可选客户端工厂。

        参数:
            config: Milvus 连接与集合配置。
            client_factory: 测试注入的目标数据库客户端工厂。
            bootstrap_factory: 测试注入的默认数据库客户端工厂。
        返回值:
            None
        异常:
            无
        """
        self._config = config
        self._client_factory = client_factory or _create_client
        self._bootstrap_factory = bootstrap_factory or _create_bootstrap_client
        self._client: Any | None = None
        self._ready = False
        self._lock = anyio.Lock()

    async def ensure_ready(self, create: bool = False) -> None:
        """检查或显式创建索引资源。

        参数:
            create: 是否允许创建缺失的独立数据库/集合；运行时必须为 False。
        返回值:
            None
        异常:
            VectorIndexUnavailable: 数据库/集合缺失、归属不明或 schema 不兼容。
        """
        async with self._lock:
            if self._ready and self._client is not None:
                return
            try:
                client = await anyio.to_thread.run_sync(self._prepare, create)
            except VectorIndexUnavailable:
                raise
            except Exception as error:
                raise VectorIndexUnavailable("Milvus 索引初始化失败") from error
            self._client = client
            self._ready = True

    def _prepare(self, create: bool) -> Any:
        """同步检查数据库、集合与索引并返回客户端。

        参数:
            create: 是否允许创建缺失资源。
        返回值:
            MilvusClient: 已就绪客户端。
        异常:
            VectorIndexUnavailable: 资源缺失或归属/结构不匹配。
        """
        target_database = self._config.milvus_database
        bootstrap = self._bootstrap_factory(self._config)
        try:
            databases = bootstrap.list_databases()
            if target_database not in databases:
                if not create:
                    raise VectorIndexUnavailable("Milvus 目标数据库不存在")
                bootstrap.create_database(target_database)
        finally:
            bootstrap.close()

        client = self._client_factory(self._config)
        try:
            collections = client.list_collections()
            if self._config.milvus_collection not in collections:
                if not create:
                    raise VectorIndexUnavailable("Milvus 目标集合不存在")
                if collections:
                    raise VectorIndexUnavailable("Milvus 隔离数据库已有其他集合，拒绝自动初始化")
                self._create_collection(client)
            self._validate(client)
            client.load_collection(
                self._config.milvus_collection, timeout=_RPC_TIMEOUT_SECONDS
            )
            return client
        except Exception:
            client.close()
            raise

    def _create_collection(self, client: Any) -> None:
        """按固定 schema 创建集合并建立 COSINE 索引。

        参数:
            client: 目标数据库客户端。
        返回值:
            None
        异常:
            VectorIndexUnavailable: SDK 创建失败。
        """
        from pymilvus import DataType
        from pymilvus.milvus_client.index import IndexParams
        from pymilvus.orm.schema import CollectionSchema, FieldSchema

        fields = [
            FieldSchema(
                "id", DataType.INT64, is_primary=True, auto_id=False, description="知识主键"
            ),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM, description="Dense 向量"),
            FieldSchema("content_hash", DataType.VARCHAR, max_length=64, description="内容指纹"),
            FieldSchema(
                "embedding_revision",
                DataType.VARCHAR,
                max_length=128,
                description="编码规范",
            ),
        ]
        schema = CollectionSchema(fields=fields, description=OWNERSHIP_MARKER)
        index_params = IndexParams()
        index_params.add_index(
            "vector", index_type=_INDEX_TYPE, metric_type=_METRIC_TYPE
        )
        try:
            client.create_collection(
                self._config.milvus_collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
                timeout=_RPC_TIMEOUT_SECONDS,
            )
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 集合创建失败") from error

    def _validate(self, client: Any) -> None:
        """验证集合归属、字段、索引与一致性。

        参数:
            client: 目标数据库客户端。
        返回值:
            None
        异常:
            VectorIndexUnavailable: 任一结构或归属不匹配。
        """
        from pymilvus import DataType
        from pymilvus.client.types import ConsistencyLevel

        name = self._config.milvus_collection
        try:
            description = client.describe_collection(name)
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 集合结构读取失败") from error

        if description.get("description") != OWNERSHIP_MARKER:
            raise VectorIndexUnavailable("Milvus 集合归属标识不匹配")
        if description.get("auto_id"):
            raise VectorIndexUnavailable("Milvus 集合不应使用 auto_id")
        if description.get("enable_dynamic_field"):
            raise VectorIndexUnavailable("Milvus 集合不应启用动态字段")

        fields = {field.get("name"): field for field in description.get("fields", [])}
        if set(fields) != {"id", "vector", "content_hash", "embedding_revision"}:
            raise VectorIndexUnavailable("Milvus 集合字段集合不匹配")
        primary = fields["id"]
        vector = fields["vector"]
        if int(primary.get("type")) != int(DataType.INT64) or not primary.get(
            "is_primary"
        ):
            raise VectorIndexUnavailable("Milvus 主键字段不匹配")
        if primary.get("auto_id"):
            raise VectorIndexUnavailable("Milvus 主键不应自动生成")
        if int(vector.get("type")) != int(DataType.FLOAT_VECTOR):
            raise VectorIndexUnavailable("Milvus 向量类型不匹配")
        if int(vector.get("params", {}).get("dim", 0)) != VECTOR_DIM:
            raise VectorIndexUnavailable("Milvus 向量维度不是 1024")
        for field_name, minimum in (("content_hash", 64), ("embedding_revision", 128)):
            field = fields[field_name]
            if int(field.get("type")) != int(DataType.VARCHAR):
                raise VectorIndexUnavailable(f"Milvus {field_name} 类型不匹配")
            if int(field.get("params", {}).get("max_length", 0)) < minimum:
                raise VectorIndexUnavailable(f"Milvus {field_name} 长度不足")

        consistency = description.get("consistency_level")
        if consistency is not None and consistency not in (
            ConsistencyLevel.Strong,
            "Strong",
            "STRONG",
        ):
            raise VectorIndexUnavailable("Milvus 集合一致性级别不是 Strong")

        try:
            indexes = client.list_indexes(name, field_name="vector")
            if not indexes:
                raise VectorIndexUnavailable("Milvus 集合缺少向量索引")
            index_info = client.describe_index(name, indexes[0])
        except VectorIndexUnavailable:
            raise
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 向量索引读取失败") from error
        metric = index_info.get("metric_type") or index_info.get("params", {}).get(
            "metric_type"
        )
        if str(metric).upper() != _METRIC_TYPE:
            raise VectorIndexUnavailable("Milvus 向量索引不是 COSINE")

    def _require_client(self) -> Any:
        """返回已就绪客户端。参数：无；返回：客户端；异常：VectorIndexUnavailable。"""
        if self._client is None or not self._ready:
            raise VectorIndexUnavailable("Milvus 索引尚未初始化")
        return self._client

    @staticmethod
    def _validate_ids(ids: list[int]) -> None:
        """验证主键处于正有符号 INT64 范围。参数：主键；返回：None；异常：VectorIndexUnavailable。"""
        if any(identifier <= 0 or identifier > MAX_SIGNED_INT64 for identifier in ids):
            raise VectorIndexUnavailable("Milvus 主键超出正有符号 INT64 范围")

    def _validate_records(self, records: list[VectorRecord]) -> None:
        """验证待写入记录。参数：记录；返回：None；异常：VectorIndexUnavailable。"""
        self._validate_ids([record.id for record in records])
        for record in records:
            if len(record.vector) != VECTOR_DIM:
                raise VectorIndexUnavailable("向量维度不是 1024")
            if not all(math.isfinite(value) for value in record.vector):
                raise VectorIndexUnavailable("向量包含非有限数值")

    async def upsert(self, records: list[VectorRecord]) -> None:
        """按主键全量写入向量。参数：记录；返回：None；异常：VectorIndexUnavailable。"""
        if not records:
            return
        self._validate_records(records)
        client = self._require_client()
        data = [
            {
                "id": record.id,
                "vector": record.vector,
                "content_hash": record.content_hash,
                "embedding_revision": record.embedding_revision,
            }
            for record in records
        ]
        try:
            await anyio.to_thread.run_sync(
                partial(
                    client.upsert,
                    self._config.milvus_collection,
                    data,
                    timeout=_RPC_TIMEOUT_SECONDS,
                )
            )
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 向量写入失败") from error

    async def search(self, vector: list[float], limit: int) -> list[VectorHit]:
        """按 COSINE 搜索向量。参数：查询向量与数量；返回：命中；异常：VectorIndexUnavailable。"""
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        if len(vector) != VECTOR_DIM or not all(math.isfinite(value) for value in vector):
            raise VectorIndexUnavailable("查询向量维度或数值不合法")
        client = self._require_client()
        try:
            results = await anyio.to_thread.run_sync(
                partial(
                    client.search,
                    self._config.milvus_collection,
                    data=[vector],
                    limit=limit,
                    output_fields=["content_hash", "embedding_revision"],
                    search_params={"metric_type": _METRIC_TYPE},
                    consistency_level="Strong",
                    timeout=_RPC_TIMEOUT_SECONDS,
                )
            )
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 向量检索失败") from error
        hits: list[VectorHit] = []
        for hit in results[0] if results else []:
            score = hit.get("distance", hit.get("score"))
            entity = hit.get("entity") or {}
            content_hash = str(entity.get("content_hash") or hit.get("content_hash") or "")
            revision = str(
                entity.get("embedding_revision") or hit.get("embedding_revision") or ""
            )
            if score is None or len(content_hash) != 64 or not revision:
                raise VectorIndexUnavailable("Milvus 命中缺少同步校验字段")
            hits.append(
                VectorHit(
                    id=int(hit["id"]),
                    score=float(score),
                    content_hash=content_hash,
                    embedding_revision=revision,
                )
            )
        return hits

    async def iter_ids(self, batch_size: int = 1000):
        """分页迭代集合主键。参数：页大小；返回：异步主键批次；异常：VectorIndexUnavailable。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        client = self._require_client()
        try:
            iterator = await anyio.to_thread.run_sync(
                partial(
                    client.query_iterator,
                    self._config.milvus_collection,
                    batch_size=batch_size,
                    limit=-1,
                    filter="",
                    output_fields=["id"],
                )
            )
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 主键分页初始化失败") from error
        try:
            while True:
                try:
                    batch = await anyio.to_thread.run_sync(iterator.next)
                except Exception as error:
                    raise VectorIndexUnavailable("Milvus 主键分页失败") from error
                if not batch:
                    break
                ids = [int(row["id"]) for row in batch]
                if ids:
                    yield ids
        finally:
            try:
                await anyio.to_thread.run_sync(iterator.close)
            except Exception:
                pass

    async def delete(self, ids: list[int]) -> None:
        """按主键幂等删除向量。参数：主键；返回：None；异常：VectorIndexUnavailable。"""
        if not ids:
            return
        self._validate_ids(ids)
        client = self._require_client()
        try:
            await anyio.to_thread.run_sync(
                partial(
                    client.delete,
                    self._config.milvus_collection,
                    ids=list(ids),
                    timeout=_RPC_TIMEOUT_SECONDS,
                )
            )
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 向量删除失败") from error

    async def aclose(self) -> None:
        """关闭客户端。参数：无；返回：None；异常：无。"""
        client, self._client = self._client, None
        self._ready = False
        if client is not None:
            try:
                await anyio.to_thread.run_sync(client.close)
            except Exception:
                pass
