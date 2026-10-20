"""Milvus Adapter 的主键范围与所有权失败关闭契约。"""

from __future__ import annotations

from typing import Any

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
