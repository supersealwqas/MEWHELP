"""每日北京时间调度与停止响应测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
