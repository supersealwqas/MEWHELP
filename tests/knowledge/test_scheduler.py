"""每日北京时间调度与停止响应测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.knowledge.scheduler import next_run_at, run_schedule

BEIJING = ZoneInfo("Asia/Shanghai")


def test_next_run_at_rolls_to_next_day() -> None:
    """未到 02:00 用当天，已到则顺延。参数：无；返回：None；异常：无。"""
    early = datetime(2026, 10, 20, 1, 0, tzinfo=BEIJING)
    target = next_run_at(early)
    assert target == datetime(2026, 10, 20, 2, 0, tzinfo=BEIJING)
    late = datetime(2026, 10, 20, 3, 0, tzinfo=BEIJING)
    assert next_run_at(late) == datetime(2026, 10, 21, 2, 0, tzinfo=BEIJING)


async def test_scheduler_recovers_then_runs_at_beijing_two() -> None:
    """启动先恢复，02:00 执行一次任务后响应停止。参数：无；返回：None；异常：无。"""
    now = [datetime(2026, 10, 20, 1, 0, tzinfo=BEIJING)]
    calls: list[tuple[str, datetime]] = []
    stop = asyncio.Event()

    def clock() -> datetime:
        """返回虚拟时间。参数：无；返回：时间；异常：无。"""
        return now[0]

    async def sleep(seconds: float) -> None:
        """推进虚拟时间。参数：秒数；返回：None；异常：无。"""
        now[0] += timedelta(seconds=seconds)

    async def recover() -> None:
        """记录恢复调用。参数：无；返回：None；异常：无。"""
        calls.append(("recover", now[0]))

    async def job() -> None:
        """记录任务调用并停止。参数：无；返回：None；异常：无。"""
        calls.append(("job", now[0]))
        stop.set()

    await run_schedule(
        job,
        recover,
        clock=clock,
        sleep=sleep,
        stop=stop,
    )
    assert calls[0][0] == "recover"
    assert calls[-1][0] == "job"
    assert calls[-1][1].hour == 2


async def test_startup_recovery_retries_before_daily_job() -> None:
    """启动恢复暂失败后限频重试，成功前不运行任务。参数：无；返回：None；异常：无。"""
    now = [datetime(2026, 10, 21, 1, 59, tzinfo=BEIJING)]
    calls: list[str] = []
    waits: list[float] = []
    stop = asyncio.Event()

    async def recover() -> None:
        """首轮模拟故障。参数：无；返回：None；异常：RuntimeError 首次调用故障。"""
        calls.append("recover")
        if calls.count("recover") == 1:
            raise RuntimeError("temporary")

    async def sleep(seconds: float) -> None:
        """推进虚拟时钟。参数：seconds 非负等待秒数；返回：None；异常：无。"""
        waits.append(seconds)
        now[0] += timedelta(seconds=seconds)

    async def job() -> None:
        """记录首次任务并停止。参数：无；返回：None；异常：无。"""
        calls.append("job")
        stop.set()

    await run_schedule(job, recover, clock=lambda: now[0], sleep=sleep, stop=stop)
    assert calls == ["recover", "recover", "job"]
    assert waits[0] == 60.0


async def test_recovery_retry_wait_is_interrupted_by_stop() -> None:
    """恢复重试的等待可被停止事件立即打断。参数：无；返回：None；异常：无。"""
    stop = asyncio.Event()
    waiting = asyncio.Event()
    cancelled = asyncio.Event()
    calls = []

    async def recover() -> None:
        """模拟持续故障。参数：无；返回：None；异常：RuntimeError。"""
        calls.append("recover")
        raise RuntimeError("temporary")

    async def sleep(seconds: float) -> None:
        """模拟阻塞等待。参数：seconds 等待秒数；返回：None；异常：CancelledError 停止时取消。"""
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def job() -> None:
        """不应运行的任务。参数：无；返回：None；异常：AssertionError。"""
        pytest.fail("恢复未成功不能运行任务")

    task = asyncio.create_task(run_schedule(
        job, recover, clock=lambda: datetime(2026, 10, 21, tzinfo=BEIJING),
        sleep=sleep, stop=stop,
    ))
    waiter = asyncio.create_task(waiting.wait())
    try:
        # 旧实现直接把恢复异常向外抛，不能进入等待。
        await asyncio.wait([task, waiter], timeout=0.2, return_when=asyncio.FIRST_COMPLETED)
        assert not task.done(), "启动故障应进入重试等待"
        assert waiting.is_set()
        stop.set()
        await asyncio.wait_for(task, timeout=0.2)
        assert cancelled.is_set()
        assert calls == ["recover"]
    finally:
        stop.set()
        if not task.done():
            task.cancel()
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(task, waiter, return_exceptions=True)


async def test_clock_rollback_does_not_repeat_same_day_job() -> None:
    """首次触发后回拨时钟，不在同一天重复触发。参数：无；返回：None；异常：无。"""
    now = [datetime(2026, 10, 21, 1, 59, tzinfo=BEIJING)]
    dates = []
    stop = asyncio.Event()

    async def sleep(seconds: float) -> None:
        """推进虚拟时钟。参数：seconds 非负等待秒数；返回：None；异常：无。"""
        now[0] += timedelta(seconds=seconds)

    async def recover() -> None:
        """成功恢复。参数：无；返回：None；异常：无。"""

    async def job() -> None:
        """首次回拨一小时，第二次停止。参数：无；返回：None；异常：无。"""
        dates.append(now[0].date())
        if len(dates) == 1:
            now[0] -= timedelta(hours=1)
        else:
            stop.set()

    await run_schedule(job, recover, clock=lambda: now[0], sleep=sleep, stop=stop)
    assert dates == [datetime(2026, 10, 21).date(), datetime(2026, 10, 22).date()]


async def test_job_failure_recovers_until_success_without_repeating_job() -> None:
    """任务故障后持续恢复，恢复失败也限频且能停止。参数：无；返回：None；异常：无。"""
    now = [datetime(2026, 10, 21, 1, 59, tzinfo=BEIJING)]
    stop = asyncio.Event()
    calls = []
    waits = []

    async def sleep(seconds: float) -> None:
        """推进虚拟时钟。参数：seconds 等待秒数；返回：None；异常：无。"""
        waits.append(seconds)
        now[0] += timedelta(seconds=seconds)

    async def recover() -> None:
        """启动成功，故障后连续两次失败再停止。参数：无；返回：None；异常：RuntimeError。"""
        calls.append("recover")
        count = calls.count("recover")
        if count in (2, 3):
            raise RuntimeError("temporary")
        if count == 4:
            stop.set()

    async def job() -> None:
        """模拟每日任务失败。参数：无；返回：None；异常：RuntimeError。"""
        calls.append("job")
        raise RuntimeError("temporary")

    await run_schedule(job, recover, clock=lambda: now[0], sleep=sleep, stop=stop)
    assert calls == ["recover", "job", "recover", "recover", "recover"]
    assert waits[:4] == [60.0, 60.0, 60.0, 60.0]


async def test_clock_jump_then_rollback_does_not_repeat_actual_run_day() -> None:
    """跨日跳钟后再回拨，不重复实际已触发日期。参数：无；返回：None；异常：无。"""
    now = [datetime(2026, 10, 21, 1, 59, tzinfo=BEIJING)]
    dates = []
    stop = asyncio.Event()
    jumped = False

    async def sleep(seconds: float) -> None:
        """首轮跨日跳钟，其后正常推进。参数：seconds 等待秒数；返回：None；异常：无。"""
        nonlocal jumped
        if not jumped:
            now[0] = datetime(2026, 10, 22, 2, 0, tzinfo=BEIJING)
            jumped = True
        else:
            now[0] += timedelta(seconds=seconds)

    async def recover() -> None:
        """成功恢复。参数：无；返回：None；异常：无。"""

    async def job() -> None:
        """首次回拨，第二次停止。参数：无；返回：None；异常：无。"""
        dates.append(now[0].date())
        if len(dates) == 1:
            now[0] -= timedelta(hours=1)
        else:
            stop.set()

    await run_schedule(job, recover, clock=lambda: now[0], sleep=sleep, stop=stop)
    assert dates == [datetime(2026, 10, 22).date(), datetime(2026, 10, 23).date()]
