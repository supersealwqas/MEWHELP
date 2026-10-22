"""Milvus Adapter 的主键范围与所有权失败关闭契约。"""

from __future__ import annotations

from typing import Any
import asyncio
import threading

import anyio
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

    def list_databases(self, **kwargs) -> list[str]:
        """返回数据库列表。参数：kwargs 为 SDK 超时；返回：列表；异常：无。"""
        return list(self._databases)

    def create_database(self, name: str, **kwargs) -> None:
        """记录成功新建数据库。参数：库名、SDK 超时；返回：None；异常：无。"""
        self._databases.append(name)

    def close(self) -> None:
        """标记关闭。参数：无；返回：None；异常：无。"""
        self.closed = True


class _ForeignClient:
    """描述错误归属集合的模拟客户端。"""

    def __init__(self) -> None:
        """初始化计数。参数：无；返回：None；异常：无。"""
        self.drop_calls = 0
        self.closed = False

    def list_collections(self, **kwargs) -> list[str]:
        """返回目标集合名。参数：kwargs 为 SDK 超时；返回：列表；异常：无。"""
        return ["knowledge"]

    def describe_collection(self, name: str, **kwargs) -> dict[str, Any]:
        """返回错误归属描述。参数：集合名、SDK 超时；返回：描述；异常：无。"""
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


class _InitializationClient:
    """仅替换 Milvus SDK，保留 Adapter 的创建、归属和结构校验流程。"""

    def __init__(self, collections=None, *, fail_create=False) -> None:
        """配置集合和创建故障。参数：集合列表、故障开关；返回：None；异常：无。"""
        self.collections = list(collections or [])
        self.fail_create = fail_create
        self.created = 0
        self.loaded = []
        self.closed = False
        self.close_count = 0
        self.description = "mewhelp-ch03:knowledge:v1"

    def list_collections(self, **kwargs):
        """读取现有集合。参数：kwargs 为 SDK 超时；返回：集合名列表；异常：无。"""
        return list(self.collections)

    def create_collection(self, name, **kwargs):
        """记录实际 SDK 创建调用。参数：集合名、SDK参数；返回：None；异常：RuntimeError。"""
        self.created += 1
        if self.fail_create:
            raise RuntimeError("创建中断")
        self.collections.append(name)

    def describe_collection(self, name, **kwargs):
        """返回手工核对的合法集合结构。参数：集合名、SDK 超时；返回：SDK描述；异常：无。"""
        return {"description": self.description, "auto_id": False,
            "enable_dynamic_field": False, "consistency_level": "Strong", "fields": [
                {"name": "id", "type": 5, "is_primary": True, "auto_id": False},
                {"name": "vector", "type": 101, "params": {"dim": 1024}},
                {"name": "content_hash", "type": 21, "params": {"max_length": 64}},
                {"name": "embedding_revision", "type": 21, "params": {"max_length": 128}},
            ]}

    def list_indexes(self, name, field_name, **kwargs):
        """返回向量索引名。参数：集合名、字段、SDK 超时；返回：索引列表；异常：无。"""
        return ["vector"]

    def describe_index(self, name, index_name, **kwargs):
        """返回合法 COSINE 索引。参数：集合名、索引名、SDK 超时；返回：索引描述；异常：无。"""
        return {"metric_type": "COSINE"}

    def load_collection(self, name, **kwargs):
        """记录已验证集合的加载。参数：集合名、SDK参数；返回：None；异常：无。"""
        self.loaded.append(name)

    def close(self):
        """记录客户端释放。参数：无；返回：None；异常：无。"""
        self.close_count += 1
        self.closed = True


_INITIALIZATION_RPCS = (
    "list_databases", "create_database", "list_collections", "create_collection",
    "describe_collection", "list_indexes", "describe_index", "load_collection",
)


class _RpcGate:
    """在真实初始化经过的 SDK 边界阻塞线程，并记录释放顺序。

    参数：通过构造方法传入原 SDK 替身方法及故障开关。
    返回：可调用的线程事件门；异常：超时或主动模拟的 SDK 故障。
    """

    def __init__(self, call, *, fail=False):
        """保存边界调用与事件。参数：call 原方法、fail 故障开关；返回：None；异常：无。"""
        self.call = call
        self.fail = fail
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self, *args, **kwargs):
        """等待测试明确放行后调用 SDK 替身。

        参数：args、kwargs 为真实 Adapter 发出的 SDK 参数。
        返回：原方法结果；异常：AssertionError 表示未清理线程，RuntimeError 模拟 SDK 故障。
        """
        self.entered.set()
        try:
            assert self.release.wait(5), "测试必须显式释放初始化线程"
            if self.fail:
                raise RuntimeError("初始化 RPC 故障")
            return self.call(*args, **kwargs)
        finally:
            self.finished.set()


@pytest.mark.parametrize("operation", _INITIALIZATION_RPCS)
async def test_initialization_passes_timeout_to_every_sdk_rpc(monkeypatch, operation):
    """初始化每个网络请求都必须显式传超时，不能依赖客户端构造参数。

    参数：monkeypatch 为 SDK 边界替换器，operation 为逐项检查的请求名。
    返回：None；异常：AssertionError 表示某个请求缺少有效期限。
    """
    config = make_settings()
    bootstrap, client = _FakeBootstrap([]), _InitializationClient()
    target = bootstrap if operation in _INITIALIZATION_RPCS[:2] else client
    original = getattr(target, operation)
    seen = []

    def record_timeout(*args, **kwargs):
        """记录实际请求期限。参数：SDK 请求参数；返回：原方法结果；异常：原方法异常。"""
        seen.append(kwargs.get("timeout"))
        return original(*args, **kwargs)

    monkeypatch.setattr(target, operation, record_timeout)
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    try:
        await index.ensure_ready(create=True)
        assert len(seen) == 1 and seen[0] is not None and seen[0] > 0, operation
    finally:
        await index.aclose()


@pytest.mark.parametrize("operation", _INITIALIZATION_RPCS)
async def test_initialization_timeout_keeps_worker_owned_until_close(monkeypatch, operation):
    """SDK 忽略期限时调用方有界失败，关闭仍等待后台线程并释放客户端。

    参数：monkeypatch 为 SDK 替换器，operation 为忽略期限的请求名。
    返回：None；异常：AssertionError 表示等待无界、提前关闭或客户端泄漏。
    """
    from app.knowledge import vector as vector_module

    config = make_settings()
    bootstrap, client = _FakeBootstrap([]), _InitializationClient()
    target = bootstrap if operation in _INITIALIZATION_RPCS[:2] else client
    gate = _RpcGate(getattr(target, operation))
    monkeypatch.setattr(target, operation, gate)
    monkeypatch.setattr(vector_module, "_RPC_TIMEOUT_SECONDS", 0.02)
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    preparing = asyncio.create_task(index.ensure_ready(create=True))
    closing = None
    try:
        assert await asyncio.to_thread(gate.entered.wait, 1)
        done, _ = await asyncio.wait({preparing}, timeout=0.2)
        assert preparing in done, "初始化超过期限后仍等待同步 SDK"
        assert isinstance(preparing.exception(), VectorIndexUnavailable)
        closing = asyncio.create_task(index.aclose())
        await asyncio.wait({closing}, timeout=0.02)
        assert not closing.done() and not client.closed, "后台工作完成前不能结束关闭"
    finally:
        gate.release.set()
        await asyncio.gather(*[task for task in (preparing, closing) if task],
            return_exceptions=True)
        await index.aclose()
    assert gate.finished.is_set() and bootstrap.closed
    assert client.closed and client.close_count == 1 and index._client is None


async def test_timeout_retry_reuses_pending_initialization(monkeypatch):
    """超时重试必须等待原任务，不能并行创建第二个客户端。

    参数：monkeypatch 为 SDK 替换器；返回：None；异常：AssertionError 表示重复初始化或结果丢失。
    """
    from app.knowledge import vector as vector_module

    config = make_settings()
    client = _InitializationClient(["knowledge"])
    gate = _RpcGate(client.load_collection)
    monkeypatch.setattr(client, "load_collection", gate)
    monkeypatch.setattr(vector_module, "_RPC_TIMEOUT_SECONDS", 0.02)
    clients_created = []

    def create_client(settings):
        """记录 SDK 客户端创建。参数：settings 连接配置；返回：客户端；异常：无。"""
        clients_created.append(client)
        return client

    index = MilvusVectorIndex(config, client_factory=create_client,
        bootstrap_factory=lambda _: _FakeBootstrap([config.milvus_database]))
    preparing = asyncio.create_task(index.ensure_ready())
    retry = None
    try:
        assert await asyncio.to_thread(gate.entered.wait, 1)
        with pytest.raises(VectorIndexUnavailable, match="超时"):
            await preparing
        monkeypatch.setattr(vector_module, "_RPC_TIMEOUT_SECONDS", 1)
        retry = asyncio.create_task(index.ensure_ready())
        await asyncio.wait({retry}, timeout=0.02)
        assert not retry.done() and clients_created == [client]
        gate.release.set()
        await retry
        assert index._ready and client.loaded == ["knowledge"] and clients_created == [client]
    finally:
        gate.release.set()
        await asyncio.gather(*[task for task in (preparing, retry) if task], return_exceptions=True)
        await index.aclose()
    assert client.close_count == 1


@pytest.mark.parametrize("cancel_count,worker_error", [(0, False), (1, False), (3, False), (3, True)])
async def test_initialization_cancel_and_close_never_lose_client(monkeypatch, cancel_count, worker_error):
    """关闭与原生、重复取消竞争时，等待初始化并且只释放一次客户端。

    参数：monkeypatch 为 SDK 替换器，cancel_count 为取消次数，worker_error 为线程故障开关。
    返回：None；异常：AssertionError 表示提前关闭、泄漏或线程故障覆盖取消。
    """
    config = make_settings()
    bootstrap, client = _FakeBootstrap([config.milvus_database]), _InitializationClient(["knowledge"])
    gate = _RpcGate(client.load_collection, fail=worker_error)
    monkeypatch.setattr(client, "load_collection", gate)
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    preparing = asyncio.create_task(index.ensure_ready())
    closing = None
    try:
        assert await asyncio.to_thread(gate.entered.wait, 1)
        for _ in range(cancel_count):
            preparing.cancel()
            await asyncio.sleep(0)
        closing = asyncio.create_task(index.aclose())
        await asyncio.wait({closing}, timeout=0.02)
        assert not closing.done() and not client.closed, "初始化完成前关闭不能丢失客户端"
    finally:
        gate.release.set()
        results = await asyncio.gather(*[task for task in (preparing, closing) if task],
            return_exceptions=True)
        await index.aclose()
    assert isinstance(results[0], asyncio.CancelledError) if cancel_count else results[0] is None
    assert results[1] is None and gate.finished.is_set()
    assert bootstrap.closed and client.close_count == 1 and index._client is None


async def test_anyio_cancelled_initialization_remains_owned_by_close(monkeypatch):
    """AnyIO 域取消也不能使正在初始化的客户端脱离关闭流程。

    参数：monkeypatch 为同步 SDK 替换器；返回：None；异常：AssertionError 表示取消后的资源泄漏。
    """
    config = make_settings()
    client = _InitializationClient(["knowledge"])
    gate = _RpcGate(client.list_collections)
    monkeypatch.setattr(client, "list_collections", gate)
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: _FakeBootstrap([config.milvus_database]))
    scopes = []

    async def prepare_in_scope():
        """在可取消域初始化。参数：无；返回：None；异常：域取消由 CancelScope 抑制。"""
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await index.ensure_ready()

    preparing = asyncio.create_task(prepare_in_scope())
    closing = None
    try:
        assert await asyncio.to_thread(gate.entered.wait, 1)
        scopes[0].cancel()
        closing = asyncio.create_task(index.aclose())
        await asyncio.wait({closing}, timeout=0.02)
        assert not closing.done() and not client.closed
    finally:
        gate.release.set()
        results = await asyncio.gather(*[task for task in (preparing, closing) if task],
            return_exceptions=True)
        await index.aclose()
    assert results == [None, None] and client.close_count == 1 and index._client is None


@pytest.mark.parametrize("operation", _INITIALIZATION_RPCS)
async def test_initialization_failure_cleans_clients_and_allows_retry(monkeypatch, operation):
    """任何初始化 RPC 失败后都释放已创建客户端，后续可重新检查资源。

    参数：monkeypatch 为 SDK 替换器，operation 为故障位置。
    返回：None；异常：AssertionError 表示失败清理遗漏或不能重试。
    """
    config = make_settings()
    bootstrap, client = _FakeBootstrap([]), _InitializationClient()
    target = bootstrap if operation in _INITIALIZATION_RPCS[:2] else client
    original = getattr(target, operation)

    def fail_rpc(*args, **kwargs):
        """模拟服务不可用。参数：SDK 请求参数；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("初始化故障")

    monkeypatch.setattr(target, operation, fail_rpc)
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    try:
        with pytest.raises(VectorIndexUnavailable):
            await index.ensure_ready(create=True)
        assert bootstrap.closed and index._client is None and not index._ready
        assert client.close_count == (0 if target is bootstrap else 1)
        monkeypatch.setattr(target, operation, original)
        await index.ensure_ready(create=True, confirmed_database=config.milvus_database)
        assert index._ready and client.loaded == ["knowledge"]
    finally:
        await index.aclose()


@pytest.mark.parametrize("phase", ["initialize", "close"])
async def test_cancelled_close_retains_cleanup_and_rejects_reinitialization(monkeypatch, phase):
    """关闭请求反复被取消时，仍保留清理任务且禁止重新初始化。

    参数：monkeypatch 为 SDK 替换器，phase 为关闭正在等待的初始化或释放阶段。
    返回：None；异常：AssertionError 表示提前复用、重复释放或遗失清理任务。
    """
    config = make_settings()
    client = _InitializationClient(["knowledge"])
    operation = "load_collection" if phase == "initialize" else "close"
    gate = _RpcGate(getattr(client, operation))
    monkeypatch.setattr(client, operation, gate)
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: _FakeBootstrap([config.milvus_database]))
    preparing = asyncio.create_task(index.ensure_ready())
    closing = retry_close = None
    try:
        if phase == "initialize":
            assert await asyncio.to_thread(gate.entered.wait, 1)
        else:
            await preparing
        closing = asyncio.create_task(index.aclose())
        if phase == "close":
            assert await asyncio.to_thread(gate.entered.wait, 1)
        else:
            await asyncio.wait({closing}, timeout=0.02)
        for _ in range(3):
            closing.cancel()
            await asyncio.sleep(0)
        retry_close = asyncio.create_task(index.aclose())
        await asyncio.wait({retry_close}, timeout=0.02)
        assert not retry_close.done(), "重试关闭必须继续等待同一清理任务"
        with pytest.raises(VectorIndexUnavailable, match="关闭"):
            await index.ensure_ready()
    finally:
        gate.release.set()
        results = await asyncio.gather(
            *[task for task in (preparing, closing, retry_close) if task], return_exceptions=True,
        )
        await index.aclose()
    assert isinstance(results[1], asyncio.CancelledError) and results[2] is None
    assert client.close_count == 1 and index._client is None and not index._ready


async def test_existing_empty_database_is_rejected_before_writing() -> None:
    """未知既存空库必须在创建集合前拒绝，并释放客户端。参数：无；返回：None；异常：断言失败。"""
    config = make_settings()
    bootstrap = _FakeBootstrap([config.milvus_database])
    client = _InitializationClient()
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    try:
        with pytest.raises(VectorIndexUnavailable, match="归属.*确认"):
            await index.ensure_ready(create=True)
        assert client.created == 0 and client.loaded == []
        assert client.closed and bootstrap.closed
    finally:
        await index.aclose()


async def test_new_database_can_initialize_without_confirmation() -> None:
    """本次明确新建的空库可以创建并加载合法集合。参数：无；返回：None；异常：断言失败。"""
    config = make_settings()
    bootstrap = _FakeBootstrap([])
    client = _InitializationClient()
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    try:
        await index.ensure_ready(create=True)
        assert bootstrap.list_databases() == [config.milvus_database]
        assert client.created == 1 and client.loaded == ["knowledge"]
    finally:
        await index.aclose()


@pytest.mark.parametrize("confirmed_database,allowed", [
    ("mewhelp_ch03", True), ("different-project", False),
])
async def test_existing_empty_database_confirmation_matches_exact_name(confirmed_database, allowed):
    """具名确认仅作用于同一个库，不能拿另一个库的确认绕过保护。

    参数：confirmed_database 为操作人确认的库名，allowed 表示该名是否与目标相同。
    返回：None，核对实际创建和加载结果；异常：AssertionError 表示范围保护失效。
    """
    config = make_settings()
    bootstrap = _FakeBootstrap([config.milvus_database])
    client = _InitializationClient()
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    try:
        if allowed:
            await index.ensure_ready(create=True, confirmed_database=confirmed_database)
            assert client.created == 1 and client.loaded == ["knowledge"]
        else:
            with pytest.raises(VectorIndexUnavailable, match="确认"):
                await index.ensure_ready(create=True, confirmed_database=confirmed_database)
            assert client.created == 0 and client.loaded == []
    finally:
        await index.aclose()


async def test_interrupted_new_database_requires_confirmation_on_retry():
    """建库后建集合中断的空库，重跑时先拒绝，具名确认后可恢复。

    参数：无。返回：None，断言中断后无自动接管，确认后建立集合。
    异常：AssertionError 表示未确认重跑写入或确认后不能恢复。
    """
    config = make_settings()
    bootstrap = _FakeBootstrap([])
    failed_client = _InitializationClient(fail_create=True)
    first = MilvusVectorIndex(config, client_factory=lambda _: failed_client,
        bootstrap_factory=lambda _: bootstrap)
    with pytest.raises(VectorIndexUnavailable, match="创建失败"):
        await first.ensure_ready(create=True)
    assert failed_client.closed and bootstrap.list_databases() == [config.milvus_database]
    client = _InitializationClient()
    retry = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: bootstrap)
    try:
        with pytest.raises(VectorIndexUnavailable, match="归属.*确认"):
            await retry.ensure_ready(create=True)
        assert client.created == 0
        await retry.ensure_ready(create=True, confirmed_database=config.milvus_database)
        assert client.created == 1 and client.loaded == ["knowledge"]
    finally:
        await first.aclose()
        await retry.aclose()


@pytest.mark.parametrize("create", [False, True])
async def test_existing_owned_collection_is_reused_without_confirmation(create):
    """已有合法集合继续通过归属和结构校验，不要求空库确认。

    参数：create 为初始化模式或普通启动模式；返回：None；异常：AssertionError 表示重建或不能加载。
    """
    config = make_settings()
    client = _InitializationClient(["knowledge"])
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: _FakeBootstrap([config.milvus_database]))
    try:
        await index.ensure_ready(create=create)
        assert client.created == 0 and client.loaded == ["knowledge"]
    finally:
        await index.aclose()


@pytest.mark.parametrize("collections,description,error", [
    (["foreign"], "mewhelp-ch03:knowledge:v1", "其他集合"),
    (["knowledge"], "foreign-owner", "归属"),
])
async def test_confirmation_does_not_adopt_nonempty_foreign_resources(collections, description, error):
    """空库确认不能绕过非空外来库及集合归属校验。

    参数：collections 为已有集合，description 为归属标记，error 为预期原因。
    返回：None；异常：AssertionError 表示确认扩大了操作范围或加载了外来集合。
    """
    config = make_settings()
    client = _InitializationClient(collections)
    client.description = description
    index = MilvusVectorIndex(config, client_factory=lambda _: client,
        bootstrap_factory=lambda _: _FakeBootstrap([config.milvus_database]))
    try:
        with pytest.raises(VectorIndexUnavailable, match=error):
            await index.ensure_ready(create=True, confirmed_database=config.milvus_database)
        assert client.created == 0 and client.loaded == [] and client.closed
    finally:
        await index.aclose()


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
