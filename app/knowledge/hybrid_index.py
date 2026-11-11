"""Milvus 原生 Chinese BM25 与 Dense/RRF 的新集合 Adapter。

继承旧 Dense adapter 的有界 RPC、初始化及取消清理机制，新集合保存同一三格
原文供 BM25 函数生成 sparse_vector；默认两路各 50，RRF 融合输出 50。
原 knowledge 集合不被删除或改写，已存在新集合必须通过所有权和结构核验。
"""

from __future__ import annotations

import json
import math
from functools import partial
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.knowledge.limits import VECTOR_DIM, valid_dense_vector
from app.knowledge.schemas import VectorHit, VectorRecord
from app.knowledge.vector import (
    MAX_SIGNED_INT64, MilvusVectorIndex, VectorIndexUnavailable, _RPC_TIMEOUT_SECONDS, _run_rpc,
)

OWNERSHIP_MARKER = "mewhelp-ch04:hybrid:char-reference:v1"
_TEXT_MAX_BYTES = 8192
_SYNC_FIELDS = ["content_hash", "embedding_revision"]
INSPECTION_FIELDS = ["id", "text", "content_hash", "embedding_revision"]


def _limit(value: int) -> None:
    """验证 SDK 召回数量。参数：value 为 1–16384 正整数；返回：None；异常：ValueError。"""
    if type(value) is not int or not 1 <= value <= 16384:
        raise ValueError("检索 limit 必须为 1–16384 的整数")


def _id_filter(ids: list[int] | None) -> str:
    """构造只含整数的原生筛选。参数：ids 权威主键或None；返回：表达式；异常：ValueError，非法主键。"""
    if ids is None:
        return ""
    if not ids or any(type(identifier) is not int or identifier <= 0 for identifier in ids):
        raise ValueError("过滤主键必须是非空正整数列表")
    return "id in " + json.dumps(sorted(set(ids)), separators=(",", ":"))


def _parse_hits(results: list) -> list[VectorHit]:
    """将 SDK 命中转为带同步指纹的有限分数对象。

    参数：results 为单查询的 MilvusClient 搜索结果。
    返回：list[VectorHit]，原顺序候选。
    异常：VectorIndexUnavailable，缺同步字段、重复主键或非法分数。
    """
    hits: list[VectorHit] = []
    try:
        for hit in results[0] if results else []:
            entity = hit.get("entity") or {}
            hits.append(VectorHit(id=hit["id"], score=hit.get("distance", hit.get("score")),
                                  content_hash=entity.get("content_hash", hit.get("content_hash")),
                                  embedding_revision=entity.get("embedding_revision", hit.get("embedding_revision"))))
        if len({hit.id for hit in hits}) != len(hits):
            raise ValueError("重复主键")
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise VectorIndexUnavailable("Milvus 命中缺少合法同步字段或包含重复主键") from error
    return hits


class MilvusHybridIndex(MilvusVectorIndex):
    """独立集合上的原生 BM25/RRF 索引，不把旧集合当混合索引接管。"""

    def __init__(self, config: Settings, **factories: Any) -> None:
        """保存新集合和旧集合的不同配置。

        参数：config 包含连接和 milvus_hybrid_collection；factories 为可选 SDK 客户端工厂。
        返回：None。
        异常：ValueError，新旧集合名称相同。
        """
        if config.milvus_hybrid_collection == config.milvus_collection:
            raise ValueError("混合集合必须与旧 Dense 集合不同")
        self._legacy_config = config
        super().__init__(config.model_copy(update={"milvus_collection": config.milvus_hybrid_collection}), **factories)

    def _prepare(self, create: bool, confirmed_database: str | None = None) -> Any:
        """验证目标库归属，允许在已验证 CH03 集合旁创建本章新集合。

        参数：create 是否允许创建；confirmed_database 为明确确认的空库名。
        返回：Any，已验证并加载的 SDK 客户端。
        异常：VectorIndexUnavailable，缺少资源、外来库或结构不匹配；SDK 异常。
        """
        bootstrap = self._bootstrap_factory(self._config)
        try:
            databases = bootstrap.list_databases(timeout=_RPC_TIMEOUT_SECONDS)
            if self._config.milvus_database not in databases:
                # 新空库创建及具名确认沿用旧 adapter 的严格保护，不自行宽松接管。
                missing_database = True
            else:
                missing_database = False
        finally:
            bootstrap.close()
        if missing_database:
            return super()._prepare(create, confirmed_database)
        client = self._client_factory(self._config)
        try:
            collections = client.list_collections(timeout=_RPC_TIMEOUT_SECONDS)
            if self._config.milvus_collection not in collections:
                if not create:
                    raise VectorIndexUnavailable("Milvus 混合集合不存在，请先显式准备新索引")
                if collections:
                    if self._legacy_config.milvus_collection not in collections:
                        raise VectorIndexUnavailable("目标库没有已确认的 CH03 集合，拒绝接管其他集合")
                    # 旧集合需要完整 schema/所有权验证，名称相同不等于归本项目。
                    MilvusVectorIndex(self._legacy_config)._validate(client)
                elif confirmed_database != self._config.milvus_database:
                    raise VectorIndexUnavailable("已有空数据库归属不明，请具名确认目标库")
                self._create_collection(client)
            self._validate(client)
            client.load_collection(self._config.milvus_collection, timeout=_RPC_TIMEOUT_SECONDS)
            return client
        except Exception:
            client.close()
            raise

    def _create_collection(self, client: Any) -> None:
        """按 Chinese analyzer/BM25 function 创建新集合和两个索引。

        参数：client 为目标库 SDK 客户端。
        返回：None。
        异常：VectorIndexUnavailable，SDK 创建失败。
        """
        from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType
        from pymilvus.milvus_client.index import IndexParams

        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False, description="知识主键"),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM, description="BGE-M3 Dense 向量"),
            FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR, description="原生 BM25 生成向量"),
            FieldSchema("text", DataType.VARCHAR, max_length=_TEXT_MAX_BYTES, enable_analyzer=True,
                        analyzer_params={"type": "chinese"}, description="MySQL 三格权威原文"),
            FieldSchema("content_hash", DataType.VARCHAR, max_length=64, description="正文指纹"),
            FieldSchema("embedding_revision", DataType.VARCHAR, max_length=128, description="编码规范版本"),
        ]
        schema = CollectionSchema(fields, description=OWNERSHIP_MARKER, enable_dynamic_field=False)
        schema.add_function(Function("text_bm25", FunctionType.BM25, ["text"], ["sparse_vector"]))
        indexes = IndexParams()
        indexes.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        indexes.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        try:
            client.create_collection(self._config.milvus_collection, schema=schema, index_params=indexes,
                                     consistency_level="Strong", timeout=_RPC_TIMEOUT_SECONDS)
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 混合集合创建失败") from error

    def _validate(self, client: Any) -> None:
        """核验所有权、六字段、Chinese analyzer、BM25 函数及两个索引。

        参数：client 为目标库 SDK 客户端。
        返回：None。
        异常：VectorIndexUnavailable，任一结构或归属不匹配。
        """
        from pymilvus import DataType, FunctionType
        from pymilvus.client.types import ConsistencyLevel

        name = self._config.milvus_collection
        try:
            schema = client.describe_collection(name, timeout=_RPC_TIMEOUT_SECONDS)
            if schema.get("description") != OWNERSHIP_MARKER:
                raise ValueError("混合集合归属标识不匹配")
            if schema.get("auto_id") or schema.get("enable_dynamic_field"):
                raise ValueError("混合集合不能启用自动主键或动态字段")
            if schema.get("consistency_level") not in (ConsistencyLevel.Strong, "Strong", "STRONG"):
                raise ValueError("混合集合必须使用 Strong 一致性")
            fields = {field["name"]: field for field in schema.get("fields", [])}
            expected = {"id", "vector", "sparse_vector", "text", *_SYNC_FIELDS}
            if set(fields) != expected:
                raise ValueError("混合集合字段集合不匹配")
            if (int(fields["id"]["type"]) != int(DataType.INT64) or not fields["id"].get("is_primary")
                    or fields["id"].get("auto_id")):
                raise ValueError("混合集合主键不匹配")
            if (int(fields["vector"]["type"]) != int(DataType.FLOAT_VECTOR)
                    or int(fields["vector"].get("params", {}).get("dim", 0)) != VECTOR_DIM
                    or int(fields["sparse_vector"]["type"]) != int(DataType.SPARSE_FLOAT_VECTOR)):
                raise ValueError("混合集合向量类型或维度不匹配")
            for field, size in (("text", _TEXT_MAX_BYTES), ("content_hash", 64), ("embedding_revision", 128)):
                if (int(fields[field]["type"]) != int(DataType.VARCHAR)
                        or int(fields[field].get("params", {}).get("max_length", 0)) < size):
                    raise ValueError(f"混合集合 {field} 类型或长度不匹配")
            params = fields["text"].get("params", {})
            analyzer = params.get("analyzer_params")
            if isinstance(analyzer, str):
                analyzer = json.loads(analyzer)
            if str(params.get("enable_analyzer")).lower() != "true" or analyzer != {"type": "chinese"}:
                raise ValueError("混合集合必须启用 Chinese analyzer")
            functions = schema.get("functions", [])
            if len(functions) != 1:
                raise ValueError("混合集合 BM25 函数缺失或多余")
            function = functions[0]
            if (function.get("name") != "text_bm25" or int(function.get("type", 0)) != int(FunctionType.BM25)
                    or function.get("input_field_names") != ["text"]
                    or function.get("output_field_names") != ["sparse_vector"]):
                raise ValueError("混合集合 BM25 函数绑定不匹配")
            for field, metric in (("vector", "COSINE"), ("sparse_vector", "BM25")):
                index_names = client.list_indexes(name, field_name=field, timeout=_RPC_TIMEOUT_SECONDS)
                if not index_names:
                    raise ValueError(f"混合集合缺少 {field} 索引")
                info = client.describe_index(name, index_names[0], timeout=_RPC_TIMEOUT_SECONDS)
                actual = info.get("metric_type") or info.get("params", {}).get("metric_type")
                if str(actual).upper() != metric:
                    raise ValueError(f"混合集合 {field} 度量不匹配")
                if field == "sparse_vector" and info.get("index_type") != "SPARSE_INVERTED_INDEX":
                    raise ValueError("BM25 字段必须使用稀疏倒排索引")
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 混合集合归属或结构校验失败") from error

    async def upsert(self, records: list[VectorRecord]) -> None:
        """写入 Dense 向量与同一三格正文，不由客户端生成 sparse。

        参数：records 为带非空 text 的合法同步记录。
        返回：None。
        异常：VectorIndexUnavailable，正文缺失、超字节限、主键/向量非法或 RPC 失败。
        """
        if not records:
            return
        self._validate_records(records)
        for record in records:
            if not record.text or not record.text.strip() or len(record.text.encode("utf-8")) > _TEXT_MAX_BYTES:
                raise VectorIndexUnavailable("BM25 同步正文缺失或超出字节上限")
        data = [record.model_dump() for record in records]
        try:
            await _run_rpc(partial(self._require_client().upsert, self._config.milvus_collection, data,
                                   timeout=_RPC_TIMEOUT_SECONDS))
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 混合索引写入失败") from error

    async def _search_field(self, data: list, field: str, metric: str, limit: int,
                            filter_ids: list[int] | None = None) -> list[VectorHit]:
        """明确字段和前置过滤执行单路召回。参数：data 查询、field 字段、metric 度量、limit 数量、filter_ids 可选权威主键；返回：命中；异常：ValueError/索引错误。"""
        _limit(limit)
        expression = _id_filter(filter_ids)
        try:
            results = await _run_rpc(partial(self._require_client().search, self._config.milvus_collection,
                data=data, anns_field=field, limit=limit, output_fields=_SYNC_FIELDS, filter=expression,
                search_params={"metric_type": metric}, consistency_level="Strong", timeout=_RPC_TIMEOUT_SECONDS))
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 单路检索失败") from error
        return _parse_hits(results)

    async def search(self, vector: list[float], limit: int, *, filter_ids: list[int] | None = None) -> list[VectorHit]:
        """在混合集合执行纯 Dense 召回。参数：vector 合法向量、limit 正数、filter_ids 可选前置过滤主键；返回：命中；异常：ValueError/索引错误。"""
        if not valid_dense_vector(vector):
            raise VectorIndexUnavailable("查询向量不合法")
        return await self._search_field([vector], "vector", "COSINE", limit, filter_ids)

    async def search_bm25(self, text: str, limit: int = 50, *, filter_ids: list[int] | None = None) -> list[VectorHit]:
        """执行 Milvus 原生 BM25 召回。参数：text 非空查询、limit 默认 50、filter_ids 可选前置过滤主键；返回：命中；异常：ValueError/索引错误。"""
        if not text.strip():
            raise ValueError("BM25 查询不能为空")
        return await self._search_field([text], "sparse_vector", "BM25", limit, filter_ids)

    async def hybrid_search(
        self, vector: list[float], text: str, *, dense_limit: int = 50,
        sparse_limit: int = 50, limit: int = 50, filter_ids: list[int] | None = None,
    ) -> list[VectorHit]:
        """两路分别召回并由 Milvus RRF 融合，默认单次输出最多 50。

        参数：vector 为合法 Dense 查询向量；text 非空 BM25 查询；dense_limit/sparse_limit 两路各 50；limit 融合输出 50；filter_ids 可选权威主键，同一筛选用于两路。
        返回：list[VectorHit]，融合排名及同步指纹；不把两路并集 100 全交给重排。
        异常：ValueError，限额或查询非法；VectorIndexUnavailable，向量、SDK 或命中非法。
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        for value in (dense_limit, sparse_limit, limit):
            _limit(value)
        if not text.strip():
            raise ValueError("混合查询不能为空")
        if not valid_dense_vector(vector):
            raise VectorIndexUnavailable("查询向量不合法")
        expression = _id_filter(filter_ids)
        requests = [
            AnnSearchRequest([vector], "vector", {"metric_type": "COSINE"}, dense_limit, expr=expression),
            AnnSearchRequest([text], "sparse_vector", {"metric_type": "BM25"}, sparse_limit, expr=expression),
        ]
        try:
            results = await _run_rpc(partial(self._require_client().hybrid_search, self._config.milvus_collection,
                reqs=requests, ranker=RRFRanker(60), limit=limit, output_fields=_SYNC_FIELDS,
                consistency_level="Strong", timeout=_RPC_TIMEOUT_SECONDS))
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 混合检索失败") from error
        return _parse_hits(results)

    async def weighted_search(
        self, vector: list[float], text: str, *, dense_weight: float = .5,
        sparse_weight: float = .5, dense_limit: int = 50, sparse_limit: int = 50,
        limit: int = 50,
    ) -> list[VectorHit]:
        """使用SDK归一化加权融合，仅发送正权重分支。

        参数：vector合法Dense向量；text原问；权重有限0–1且合计正数；三项limit为正整数。
        返回：list[VectorHit]，Milvus原生加权分数与指纹。
        异常：ValueError，权重/文本/限额非法；VectorIndexUnavailable，向量或RPC失败。
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        for value in (dense_limit, sparse_limit, limit):
            _limit(value)
        weights = (dense_weight, sparse_weight)
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
               for value in weights) or sum(weights) <= 0:
            raise ValueError("加权检索权重必须有限、非负且合计大于零")
        if not text.strip():
            raise ValueError("加权查询不能为空")
        if not valid_dense_vector(vector):
            raise VectorIndexUnavailable("查询向量不合法")
        requests, positive = [], []
        for data, field, metric, size, weight in (
            ([vector], "vector", "COSINE", dense_limit, dense_weight),
            ([text], "sparse_vector", "BM25", sparse_limit, sparse_weight),
        ):
            # 零权重路不能进入融合，否则会产生无法解释的贡献。
            if weight > 0:
                requests.append(AnnSearchRequest(data, field, {"metric_type": metric}, size))
                positive.append(weight / sum(weights))
        try:
            result = await _run_rpc(partial(self._require_client().hybrid_search,
                self._config.milvus_collection, reqs=requests, ranker=WeightedRanker(*positive, norm_score=True),
                limit=limit, output_fields=_SYNC_FIELDS, consistency_level="Strong", timeout=_RPC_TIMEOUT_SECONDS))
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 加权检索失败") from error
        return _parse_hits(result)

    async def inspection_collection(self) -> dict:
        """读取公开集合结构、索引及Strong实时数量。

        参数：无，只使用启动已核验的共享client；返回：dict，公开结构与实际count。
        异常：VectorIndexUnavailable，SDK故障或描述/数量不合法。
        """
        def read() -> dict:
            """参数：无；返回：公开结构；异常：SDK或结构错误。"""
            from pymilvus import DataType, FunctionType

            client, name = self._require_client(), self._config.milvus_collection
            schema = client.describe_collection(name, timeout=_RPC_TIMEOUT_SECONDS)
            count = client.query(name, filter="", output_fields=["count(*)"],
                                 consistency_level="Strong", timeout=_RPC_TIMEOUT_SECONDS)
            value = count[0]["count(*)"]
            if type(value) is not int or value < 0:
                raise ValueError("集合数量非法")
            fields = []
            for field in schema["fields"]:
                readable = field["name"] != "sparse_vector"
                fields.append({"name": field["name"], "type": DataType(int(field["type"])).name,
                    "description": field.get("description", ""), "params": field.get("params", {}),
                    "is_primary": bool(field.get("is_primary", False)), "readable": readable,
                    "note": "" if readable else "由Milvus BM25函数自动生成，SDK不能读取原始稀疏向量"})
            functions = [{"name": function["name"], "type": FunctionType(int(function["type"])).name,
                "input_fields": function.get("input_field_names", []),
                "output_fields": function.get("output_field_names", [])} for function in schema.get("functions", [])]
            indexes = []
            for field in ("vector", "sparse_vector"):
                for index_name in client.list_indexes(name, field_name=field, timeout=_RPC_TIMEOUT_SECONDS):
                    info = client.describe_index(name, index_name, timeout=_RPC_TIMEOUT_SECONDS)
                    indexes.append({"name": index_name, "field_name": field,
                        "index_type": info.get("index_type"),
                        "metric_type": info.get("metric_type") or info.get("params", {}).get("metric_type")})
            dimension = next(int(field["params"]["dim"]) for field in fields if field["name"] == "vector")
            return {"database": self._config.milvus_database, "collection": name, "row_count": value,
                    "dense_dimension": dimension, "fields": fields, "functions": functions, "indexes": indexes}

        try:
            return await _run_rpc(read)
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 集合读取暂时不可用") from error

    async def inspection_page(self, after_id: int, limit: int) -> list[dict]:
        """在同一有界同步调用内创建、读取并关闭短寿命游标。

        参数：after_id为0–INT64最大值，limit为1–100；返回：排序后的limit+1条标量行。
        异常：ValueError，游标/页大小非法；VectorIndexUnavailable，RPC或结果非法。
        """
        if type(after_id) is not int or not 0 <= after_id <= MAX_SIGNED_INT64:
            raise ValueError("游标必须为非负signed INT64")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("页大小必须为1–100整数")

        def read() -> list[dict]:
            """参数：无；返回：单批原始标量行；异常：SDK错误且finally关闭。"""
            iterator = self._require_client().query_iterator(self._config.milvus_collection,
                filter=f"id > {after_id}", output_fields=INSPECTION_FIELDS,
                batch_size=limit + 1, limit=limit + 1, consistency_level="Strong", timeout=_RPC_TIMEOUT_SECONDS)
            try:
                return iterator.next()
            finally:
                iterator.close()

        try:
            rows = await _run_rpc(read)
            self._validate_inspection_rows(rows)
            if len(rows) > limit + 1 or any(row["id"] <= after_id for row in rows):
                raise ValueError("分页返回超额或游标外行")
            return sorted(rows, key=lambda row: row["id"])
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 记录读取暂时不可用") from error

    async def inspection_records(self, ids: list[int], *, vector: bool = False) -> list[dict]:
        """批量读取明确的可读字段，仅详情请求Dense向量。

        参数：ids为非重复正signed INT64列表；vector是否读取Dense向量。
        返回：原始标量行，可选vector；空主键返回空列表。
        异常：ValueError，主键非法；VectorIndexUnavailable，RPC或重复主键非法。
        """
        if not ids:
            return []
        if any(type(identifier) is not int or not 0 < identifier <= MAX_SIGNED_INT64 for identifier in ids):
            raise ValueError("主键必须为正signed INT64")
        try:
            fields = [*INSPECTION_FIELDS, "vector"] if vector else INSPECTION_FIELDS
            rows = await _run_rpc(partial(self._require_client().get, self._config.milvus_collection,
                ids=sorted(set(ids)), output_fields=fields, consistency_level="Strong", timeout=_RPC_TIMEOUT_SECONDS))
            self._validate_inspection_rows(rows)
            if any(row["id"] not in ids for row in rows):
                raise ValueError("回读返回范围外主键")
            return rows
        except Exception as error:
            raise VectorIndexUnavailable("Milvus 记录回读暂时不可用") from error

    @staticmethod
    def _validate_inspection_rows(rows: list[dict]) -> None:
        """核验可读行身份，阻止重复主键错配正文。

        参数：rows SDK原始行；返回：None；异常：ValueError，身份/标量字段非法。
        """
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or any(type(value) is not int or not 0 < value <= MAX_SIGNED_INT64 for value in ids):
            raise ValueError("回读主键重复或非法")
        if any(any(not isinstance(row.get(field), str) for field in INSPECTION_FIELDS[1:]) for row in rows):
            raise ValueError("回读标量字段缺失")
