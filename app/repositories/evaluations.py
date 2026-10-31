"""固定eval_runs表的完整产物保存、跨进程幂等与应用评估独立执行锁。"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.knowledge.lease import MySqlWriteLease, finish_cleanup
from app.models.flywheel import EvalRun

EVALUATION_EXECUTION_LOCK = "mewhelp_ch09_evaluation_execute"
EVALUATION_SAVE_LOCK = "mewhelp_ch09_evaluation_save"


def _payload(row) -> dict:
    """投影固定字段，不把未知指标变成零。参数：EvalRun实体；返回：API字典；异常：无。"""
    return {"id": str(row.id), "triggered_by": row.triggered_by, "dataset_size": row.dataset_size,
        "metrics": row.metrics, "created_at": row.created_at}


class EvaluationRepository:
    """独立Session读取；保存查重和提交始终绑定同一持锁物理连接。"""

    def __init__(self, factory, lease_engine):
        """注入数据库资源。参数：独立会话工厂、NullPool租约引擎；返回：None；异常：无。"""
        self.factory, self.lease_engine = factory, lease_engine

    @asynccontextmanager
    async def execution(self):
        """在任何生产资源/模型调用之前争用共同锁。参数：无；返回：执行lease；异常：忙、失锁或取消。"""
        lease = await MySqlWriteLease.acquire(self.lease_engine, timeout_seconds=0, lock_name=EVALUATION_EXECUTION_LOCK)
        try:
            yield lease
        finally:
            await finish_cleanup(lease.release())

    async def execution_busy(self) -> bool:
        """读取数据库执行占用事实，不把本实例idle宣称全库空闲。参数：无；返回：bool；异常：数据库错误。"""
        async with self.factory() as session:
            return await session.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": EVALUATION_EXECUTION_LOCK}) is not None

    async def save_completed_run(self, artifact_dir: Path, triggered_by: Literal["定时", "手动"]) -> dict:
        """本地联合核验完成产物后，在独立短锁内查重并提交。

        参数：artifact_dir已完成目录；triggered_by中文触发来源。
        返回：固定轮次字段及saved（重复为False）；异常：无效产物/同ID不同内容、锁或数据库错误。
        """
        from app.evaluation.runner import validate_completed_artifact
        if triggered_by not in {"定时", "手动"}:
            raise ValueError("评估触发来源非法")
        metrics = validate_completed_artifact(Path(artifact_dir))
        if metrics["metadata"]["triggered_by"] != triggered_by:
            raise ValueError("触发来源与原始评估产物不同")
        lease = await MySqlWriteLease.acquire(self.lease_engine, lock_name=EVALUATION_SAVE_LOCK)
        try:
            await lease.assert_held()
            # 核锁SQL的autobegin必须先结束，否则Session提交不会提交外部事务。
            await lease.connection.commit()
            async with AsyncSession(bind=lease.connection, expire_on_commit=False) as session:
                async with session.begin():
                    existing = (await session.scalars(select(EvalRun).where(
                        EvalRun.metrics["artifact_id"].as_string() == metrics["artifact_id"]))).all()
                    if existing:
                        if len(existing) != 1 or existing[0].metrics != metrics:
                            raise ValueError("相同artifact_id对应不同内容或重复历史记录")
                        return {**_payload(existing[0]), "saved": False}
                    row = EvalRun(triggered_by=triggered_by, dataset_size=metrics["dataset_size"], metrics=metrics)
                    session.add(row)
                    await session.flush()
                    await session.refresh(row)
                    # 同一连接上最后核验，断线绝不改用工厂连接提交。
                    await lease.assert_held()
                return {**_payload(row), "saved": True}
        finally:
            await finish_cleanup(lease.release())

    async def list_runs(self, after_id: int = 0, limit: int = 50) -> dict:
        """按稳定主键分页完整轮次。参数：非负after_id、1–200的limit；返回：items/next_after_id；异常：参数或数据库错误。"""
        if after_id < 0 or not 1 <= limit <= 200:
            raise ValueError("评估分页参数非法")
        async with self.factory() as session:
            rows = (await session.scalars(select(EvalRun).where(EvalRun.id > after_id).order_by(EvalRun.id).limit(limit + 1))).all()
            return {"items": [_payload(row) for row in rows[:limit]],
                "next_after_id": str(rows[limit - 1].id) if len(rows) > limit else None}


@asynccontextmanager
async def evaluation_repository_runtime(config):
    """CLI自有应用库资源；不建表、不加载知识或模型。参数：明确应用Settings；返回：仓储；异常：配置/数据库错误。"""
    from app.db.session import create_database
    engine, factory = create_database(config)
    # 保存发生在独立持锁连接上，CURRENT_TIMESTAMP必须沿用应用库的UTC约定。
    lease_engine = create_async_engine(
        engine.url,
        poolclass=NullPool,
        connect_args={"connect_timeout": 5, "init_command": "SET time_zone = '+00:00'"},
    )
    try:
        yield EvaluationRepository(factory, lease_engine)
    finally:
        try:
            await finish_cleanup(lease_engine.dispose())
        finally:
            await finish_cleanup(engine.dispose())
