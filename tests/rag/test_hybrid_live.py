"""仅写独占 CH04 测试集合的真实 Milvus 原生 BM25/RRF 验证。

默认跳过；设置 MEWHELP_RUN_RAG_INTEGRATION=1 才访问本机 Milvus。只创建/删除
本例随机命名集合，保留旧库、旧集合及测试数据库本身，不运行 MySQL 清库夹具。
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.knowledge.hybrid_index import MilvusHybridIndex
from app.knowledge.schemas import VectorRecord
from tests.knowledge.fakes import make_settings


@pytest.mark.skipif(os.getenv("MEWHELP_RUN_RAG_INTEGRATION") != "1", reason="需要显式 MEWHELP_RUN_RAG_INTEGRATION=1")
async def test_native_chinese_bm25_and_rrf_use_only_unique_test_collection() -> None:
    """真实 SDK/服务端须支持中文 BM25，融合输出最多 50，重启可复用。参数：无；返回：None；异常：SDK/断言错误。"""
    from pymilvus import MilvusClient

    database = "mewhelp_ch04_test"
    collection = "reference_" + uuid4().hex
    config = make_settings(milvus_database=database, milvus_hybrid_collection=collection)
    bootstrap = MilvusClient(uri=config.milvus_uri, timeout=10)
    client = None
    index = None
    created = False
    try:
        if database not in bootstrap.list_databases(timeout=10):
            bootstrap.create_database(database, timeout=10)
        client = MilvusClient(uri=config.milvus_uri, db_name=database, timeout=10)
        # 随机名是本例明确授权的独占资源；不以空库确认接管其他测试集合。
        index = MilvusHybridIndex(config)
        index._create_collection(client)
        created = True
        await index.ensure_ready()
        texts = ["智能猫砂盆 Lite 型号 MH-LP50 不支持 App 联网功能", "自动喂食器 MH-FD30 断电不支持视频回看"]
        texts.extend(f"退款处理时限与退货政策条款 {number} 必须保留包装" for number in range(60))
        records = [VectorRecord(id=number, vector=[1.0] + [0.0] * 1023,
                                text=text, content_hash=f"{number:064x}", embedding_revision="integration-only")
                   for number, text in enumerate(texts, 1)]
        await index.upsert(records)
        bm25 = await index.search_bm25("MH-LP50 Lite App 联网", 50)
        assert bm25 and bm25[0].id == 1
        hybrid = await index.hybrid_search([1.0] + [0.0] * 1023, "退款处理时限")
        assert len(hybrid) == 50 and len({hit.id for hit in hybrid}) == 50
        assert await index.get_fingerprints([1]) == {1: (f"{1:064x}", "integration-only")}
        await index.aclose()
        index = MilvusHybridIndex(config)
        await index.ensure_ready(create=False)
        assert (await index.search_bm25("MH-FD30 视频回看", 50))[0].id == 2
    finally:
        if index is not None:
            await index.aclose()
        if client is not None:
            # 只删除本例成功创建且名称完全相同的集合，绝不清库。
            if created:
                client.drop_collection(collection, timeout=10)
            client.close()
        bootstrap.close()
