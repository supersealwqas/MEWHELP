"""MySQL 写锁获取、持有校验与释放的离线契约测试。"""

from __future__ import annotations

from typing import Any

import pytest

from app.knowledge.lease import (
    ConcurrentKnowledgeWrite,
    KnowledgeStoreUnavailable,
    LeaseLost,
    MySqlWriteLease,
)


class _FakeConnection:
    """模拟 MySQL 锁相关查询的连接。"""

    def __init__(self, lock_result: int | None = 1, connection_id: int = 7) -> None:
        """保存锁结果与连接身份。

        参数:
            lock_result: GET_LOCK 返回值。
            connection_id: CONNECTION_ID 返回值。
        返回值:
            None
        异常:
            无
        """
        self.lock_result = lock_result
        self.connection_id = connection_id
        self.owner = connection_id
        self.closed = False
        self.commits = 0

    async def scalar(self, statement: Any, parameters: dict | None = None) -> Any:
        """按 SQL 文本返回模拟值。

        参数:
            statement: SQLAlchemy 语句。
            parameters: 绑定参数。
        返回值:
            Any: 锁或连接身份模拟值。
        异常:
            AssertionError: 出现未预期 SQL 时抛出。
        """
        del parameters
        sql = str(statement)
        if "GET_LOCK" in sql:
            return self.lock_result
        if "CONNECTION_ID" in sql:
            return self.connection_id
        if "IS_USED_LOCK" in sql:
            return self.owner
        if "RELEASE_LOCK" in sql:
            return 1
        raise AssertionError(f"未预期 SQL: {sql}")

    async def commit(self) -> None:
        """记录事务提交。参数：无；返回：None；异常：无。"""
        self.commits += 1

    async def close(self) -> None:
        """记录连接关闭。参数：无；返回：None；异常：无。"""
        self.closed = True


class _FakeEngine:
    """返回固定模拟连接的引擎。"""

    def __init__(self, connection: _FakeConnection) -> None:
        """保存模拟连接。参数：连接；返回：None；异常：无。"""
        self.connection = connection

    async def connect(self) -> _FakeConnection:
        """返回模拟连接。参数：无；返回：连接；异常：无。"""
        return self.connection


@pytest.mark.parametrize("lock_result", [0, None])
async def test_null_or_zero_get_lock_fails_closed(lock_result: int | None) -> None:
    """GET_LOCK 超时或错误时关闭连接并拒绝写入。

    参数:
        lock_result: 模拟的锁返回值。
    返回值:
        None
    异常:
        无
    """
    connection = _FakeConnection(lock_result=lock_result)
    with pytest.raises((ConcurrentKnowledgeWrite, KnowledgeStoreUnavailable)):
        await MySqlWriteLease.acquire(_FakeEngine(connection))
    assert connection.closed is True


async def test_connection_replacement_loses_lease() -> None:
    """锁持有连接身份变化时立即失败关闭。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    connection = _FakeConnection()
    lease = await MySqlWriteLease.acquire(_FakeEngine(connection))
    connection.owner = connection.connection_id + 1
    with pytest.raises(LeaseLost):
        await lease.assert_held()
    await lease.release()


async def test_release_closes_physical_connection() -> None:
    """释放锁后关闭专用物理连接。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    connection = _FakeConnection()
    lease = await MySqlWriteLease.acquire(_FakeEngine(connection))
    await lease.release()
    assert connection.closed is True
    assert connection.commits >= 1
