"""每日评估只排未来时刻，并与人工触发共用进程所有者。"""

import asyncio
import importlib
import importlib.util
from datetime import datetime, timezone

import pytest


def scheduler_module():
    """加载待实现调度器。参数：无；返回：模块；异常：缺失时断言失败。"""
    assert importlib.util.find_spec("app.evaluation.scheduler"), "缺少每日生产评估调度器"
    return importlib.import_module("app.evaluation.scheduler")


@pytest.mark.parametrize("now,want", [
    ("2026-10-31T02:59:59+08:00", "2026-10-31T03:00:00+08:00"),
    ("2026-10-31T03:00:00+08:00", "2026-11-01T03:00:00+08:00"),
    ("2026-10-31T04:00:00+08:00", "2026-11-01T03:00:00+08:00"),
    ("2026-10-30T18:59:59+00:00", "2026-10-31T03:00:00+08:00"),
])
def test_restart_does_not_replay_missed_days(now, want):
    """到点重启不得追补，UTC和上海同一时刻一致。参数：时刻/预期；返回：None；异常：断言失败。"""
    assert scheduler_module().next_run(datetime.fromisoformat(now)).isoformat() == want


def test_naive_time_is_rejected():
    """拒绝依赖本机默认时区。参数：无；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        scheduler_module().next_run(datetime(2026, 10, 31))


@pytest.mark.parametrize("manual_first", [True, False])
async def test_manual_and_daily_trigger_share_one_job(monkeypatch, tmp_path, manual_first):
    """手动与到点交错不能启动第二个进程。参数：替身/目录/顺序；返回：None；异常：断言失败。"""
    from tests.rag.test_evaluation_jobs import ProcessDouble, _manager
    module = scheduler_module()
    manager, calls = _manager(monkeypatch, tmp_path, [ProcessDouble()])
    clock = datetime(2026, 10, 30, 18, 59, 59, tzinfo=timezone.utc)
    entered, due = asyncio.Event(), asyncio.Event()

    async def sleep(seconds):
        """控制第一次到点，其后阻塞至关闭。参数：秒数；返回：None；异常：取消。"""
        nonlocal clock
        entered.set()
        await due.wait()
        due.clear()
        clock = datetime(2026, 10, 30, 19, 0, tzinfo=timezone.utc)

    scheduler = module.EvaluationScheduler(manager, tmp_path / "data.json", tmp_path / "policy.json",
        now=lambda: clock, sleep=sleep)
    try:
        scheduler.start()
        await entered.wait()
        assert calls == []
        if manual_first:
            manager.start(tmp_path / "data.json", [], profile="production", policy_path=tmp_path / "policy.json")
        due.set()
        await asyncio.sleep(0.03)
        if not manual_first:
            with pytest.raises(RuntimeError):
                manager.start(tmp_path / "data.json", [], profile="production", policy_path=tmp_path / "policy.json")
        assert len(calls) == 1 and calls[0][0][3] == "evaluate-production"
        assert scheduler.next_at.isoformat() == "2026-11-01T03:00:00+08:00"
    finally:
        await scheduler.close()
        await manager.close()
    assert len(calls) == 1


async def test_close_sleep_never_triggers(tmp_path):
    """关闭睡眠不得误触发评估或遗留任务。参数：目录；返回：None；异常：断言失败。"""
    from app.evaluation.jobs import EvaluationJobManager
    manager = EvaluationJobManager(tmp_path, tmp_path / "reports")
    scheduler = scheduler_module().EvaluationScheduler(manager, tmp_path / "data", tmp_path / "policy")
    scheduler.start()
    await asyncio.sleep(0)
    await scheduler.close()
    await scheduler.close()
    assert manager.status()["status"] == "idle"
    await manager.close()
