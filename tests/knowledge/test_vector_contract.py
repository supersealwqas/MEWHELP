"""Milvus Adapter 的主键范围与所有权失败关闭契约。"""

from __future__ import annotations

from typing import Any
import asyncio
import threading

import pytest

from app.knowledge.schemas import VectorRecord
from app.knowledge.vector import MilvusVectorIndex, VectorIndexUnavailable
from tests.knowledge.fakes import make_settings


class _FakeBootstrap:
    """模拟默认数据库客户端。"""

    def __init__(self, databases: list[str]) -> None:
        """保存数据库列表。参数：列表；返回：None；异常：无。"""
        self._databases = databases
        self.closed = False

    def list_databases(self) -> list[str]:
        """返回数据库列表。参数：无；返回：列表；异常：无。"""
        return list(self._databases)

    def close(self) -> None:
        """标记关闭。参数：无；返回：None；异常：无。"""
        self.closed = True


class _ForeignClient:
    """描述错误归属集合的模拟客户端。"""

    def __init__(self) -> None:
        """初始化计数。参数：无；返回：None；异常：无。"""
        self.drop_calls = 0
        self.closed = False

    def list_collections(self) -> list[str]:
        """返回目标集合名。参数：无；返回：列表；异常：无。"""
        return ["knowledge"]

    def describe_collection(self, name: str) -> dict[str, Any]:
        """返回错误归属描述。参数：集合名；返回：描述；异常：无。"""
        del name
        return {
            "description": "foreign-owner",
            "fields": [],
            "auto_id": False,
            "enable_dynamic_field": False,
        }

    def drop_collection(self, *args: Any, **kwargs: Any) -> None:
        """记录被误调用。参数：任意；返回：None；异常：无。"""
        del args, kwargs
        self.drop_calls += 1

    def close(self) -> None:
        """标记关闭。参数：无；返回：None；异常：无。"""
        self.closed = True


async def test_primary_key_signed_range_is_rejected() -> None:
    """主键超过正有符号 INT64 时在连接前拒绝。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    index = MilvusVectorIndex(make_settings())
    record = VectorRecord(
        id=2**63,
        vector=[0.0] * 1024,
        content_hash="a" * 64,
        embedding_revision="rev",
    )
    with pytest.raises(VectorIndexUnavailable, match="INT64"):
        await index.upsert([record])


async def test_schema_or_owner_mismatch_never_drops() -> None:
    """归属不匹配时失败关闭且绝不 drop 集合。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    settings = make_settings()
    client = _ForeignClient()
    bootstrap = _FakeBootstrap([settings.milvus_database])
    index = MilvusVectorIndex(
        settings,
        client_factory=lambda config: client,
        bootstrap_factory=lambda config: bootstrap,
    )
    with pytest.raises(VectorIndexUnavailable, match="归属"):
        await index.ensure_ready(create=True)
    assert client.drop_calls == 0
    assert client.closed is True
    assert bootstrap.closed is True


async def test_get_fingerprints_reads_only_sync_fields_with_strong_consistency() -> None:
    """对照只读主键及两种指纹并显式强一致。参数：无；返回：None；异常：断言失败。"""
    class Client:
        """验证实际 SDK get 边界参数。"""

        def get(self, name, ids, output_fields, timeout, **kwargs):
            """返回一个存在的主键。参数：SDK 参数；返回：结果行；异常：断言失败。"""
            assert name == "knowledge" and ids == [1, 2]
            assert output_fields == ["id", "content_hash", "embedding_revision"]
            assert timeout > 0 and kwargs["consistency_level"] == "Strong"
            return [{"id": 1, "content_hash": "a" * 64, "embedding_revision": "rev"}]

    index = MilvusVectorIndex(make_settings())
    index._ready, index._client = True, Client()
    assert callable(getattr(index, "get_fingerprints", None)), "缺少主键指纹回读接口"
    assert await index.get_fingerprints([1, 2]) == {1: ("a" * 64, "rev")}


@pytest.mark.parametrize("operation", ["upsert", "delete", "get", "iterator", "next"])
async def test_sdk_operations_have_bounded_wait(monkeypatch, operation) -> None:
    """SDK 不遵守 timeout 时调用方仍有界退出。参数：补丁、操作；返回：None；异常：断言失败。"""
    from app.knowledge import vector as vector_module

    entered, finish = threading.Event(), threading.Event()

    class Client:
        """仅选中操作阻塞的 SDK 替身。"""

        def _call(self, name):
            """阻塞指定操作。参数：操作名；返回：None；异常：无。"""
            if operation == name:
                entered.set()
                finish.wait(1)

        def upsert(self, *args, **kwargs):
            """模拟写入。参数：SDK 参数；返回：None；异常：无。"""
            self._call("upsert")

        def delete(self, *args, **kwargs):
            """模拟删除。参数：SDK 参数；返回：None；异常：无。"""
            self._call("delete")

        def get(self, *args, **kwargs):
            """模拟回读。参数：SDK 参数；返回：行；异常：无。"""
            self._call("get")
            return []

        def query_iterator(self, *args, **kwargs):
            """模拟分页初始化。参数：SDK 参数；返回：迭代器；异常：无。"""
            assert kwargs.get("timeout", 0) > 0
            assert kwargs.get("consistency_level") == "Strong"
            self._call("iterator")
            return self

        def next(self):
            """模拟翻页。参数：无；返回：空页；异常：无。"""
            self._call("next")
            return []

        def close(self):
            """结束迭代。参数：无；返回：None；异常：无。"""

    index = MilvusVectorIndex(make_settings())
    index._ready, index._client = True, Client()
    monkeypatch.setattr(vector_module, "_RPC_TIMEOUT_SECONDS", 0.02)
    if operation == "get":
        assert callable(getattr(index, "get_fingerprints", None)), "缺少主键指纹回读接口"
        call = index.get_fingerprints([1])
    elif operation == "upsert":
        call = index.upsert([VectorRecord(id=1, vector=[1.0] * 1024,
                                        content_hash="a" * 64, embedding_revision="rev")])
    elif operation == "delete":
        call = index.delete([1])
    else:
        call = anext(index.iter_ids(1))
    try:
        with pytest.raises(VectorIndexUnavailable):
            await asyncio.wait_for(call, timeout=0.2)
        assert entered.is_set(), "失败必须发生在真实边界调用后"
    finally:
        finish.set()
