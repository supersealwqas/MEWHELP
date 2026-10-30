"""MySQL 跨进程知识库写锁。

用独立 AsyncConnection 持有会话级 GET_LOCK，记录 CONNECTION_ID；每次写入
前核对锁持有者仍是当前连接。连接重建、释放失败或所有权变化都必须失败关闭，
不能把新连接默认视为仍持锁。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.knowledge.errors import (
    ConcurrentKnowledgeWrite,
    KnowledgeStoreUnavailable,
    LeaseLost,
)

KNOWLEDGE_WRITE_LOCK = "mewhelp_ch03_knowledge_write"
logger = logging.getLogger(__name__)


async def finish_cleanup(operation, *, timeout_seconds: float = 5.0) -> None:
    """屏蔽重复取消并等待有界清理真正结束，随后传播取消。

    参数：operation为协作式异步清理；timeout_seconds为正数截止秒数。
    返回：None；异常：原取消、清理异常或TimeoutError，不把后台遗留工作视为成功。
    """
    async def bounded():
        """参数：无；返回：None；异常：清理错误或超时。"""
        async with asyncio.timeout(timeout_seconds):
            await operation
    task = asyncio.create_task(bounded())
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            # shield只隔离清理任务；仍持续等待其完成，不把它留在后台。
            cancelled = error
    task.result()
    if cancelled is not None:
        raise cancelled


class MySqlWriteLease:
    """一条持有 MySQL 命名锁的专用物理连接。"""

    def __init__(self, connection: AsyncConnection, *, lock_name: str) -> None:
        """保存连接与锁名。

        参数:
            connection: 已获得命名锁的异步连接。
            lock_name: MySQL GET_LOCK 名称。
        返回值:
            None
        异常:
            无
        """
        self._connection = connection
        self._lock_name = lock_name

    @property
    def connection(self) -> AsyncConnection:
        """返回持锁连接。参数：无；返回：连接；异常：无。"""
        return self._connection

    @classmethod
    async def acquire(
        cls,
        engine: AsyncEngine,
        *,
        timeout_seconds: float = 5.0,
        lock_name: str = KNOWLEDGE_WRITE_LOCK,
    ) -> "MySqlWriteLease":
        """在新物理连接上获取写锁。

        参数:
            engine: 使用 NullPool 的锁专用引擎。
            timeout_seconds: 等待锁的秒数。
            lock_name: MySQL 命名锁名称。
        返回值:
            MySqlWriteLease: 已持锁的租约。
        异常:
            ConcurrentKnowledgeWrite: 锁被其他连接持有且等待超时。
            KnowledgeStoreUnavailable: 锁查询返回错误或连接身份不可用。
        """
        connection = await engine.connect()
        try:
            result = await connection.scalar(
                text("SELECT GET_LOCK(:name, :timeout)"),
                {"name": lock_name, "timeout": max(0, int(timeout_seconds))},
            )
            if result == 1:
                connection_id = await connection.scalar(text("SELECT CONNECTION_ID()"))
                if connection_id is None:
                    raise KnowledgeStoreUnavailable("无法确认知识库写锁连接身份")
                # GET_LOCK 不随事务提交释放；结束隐式事务，后续短事务继续复用同一连接。
                await connection.commit()
                return cls(connection, lock_name=lock_name)
            if result == 0:
                raise ConcurrentKnowledgeWrite("另一个知识库写入任务正在运行")
            raise KnowledgeStoreUnavailable("知识库写锁查询失败")
        except BaseException:
            # GET_LOCK等待取消也必须关闭；否则命名锁可能在调用方消失后才取得。
            await finish_cleanup(connection.close())
            raise

    async def assert_held(self) -> None:
        """核对锁仍由当前物理连接持有。

        参数:
            无
        返回值:
            None
        异常:
            LeaseLost: 锁持有者已变化或锁已丢失。
            KnowledgeStoreUnavailable: 查询连接身份失败。
        """
        owner = await self._connection.scalar(
            text("SELECT IS_USED_LOCK(:name)"), {"name": self._lock_name}
        )
        current = await self._connection.scalar(text("SELECT CONNECTION_ID()"))
        if owner is None or current is None or int(owner) != int(current):
            raise LeaseLost("知识库写锁已丢失，停止写入")

    async def release(self) -> None:
        """释放命名锁并关闭专用连接。

        参数:
            无
        返回值:
            None
        异常:
            CancelledError、物理关闭异常或超时；释放SQL失败记录类型后仍关闭连接。
        """
        try:
            async with asyncio.timeout(5.0):
                await self._connection.scalar(
                    text("SELECT RELEASE_LOCK(:name)"), {"name": self._lock_name}
                )
                await self._connection.commit()
        except Exception as error:
            # 释放失败不能让连接回到池中；关闭仍由 finally 保证。
            logger.warning("lease release query failed lock=%s error_type=%s", self._lock_name, type(error).__name__)
        finally:
            await finish_cleanup(self._connection.close())


__all__ = [
    "KNOWLEDGE_WRITE_LOCK",
    "MySqlWriteLease",
    "ConcurrentKnowledgeWrite",
    "KnowledgeStoreUnavailable",
    "LeaseLost",
]
