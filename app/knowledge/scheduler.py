"""知识挖掘的独立定时调度。

默认每天北京时间 02:00 触发；启动先恢复 pending，不领取新来源。等待按
最多 60 秒分段以响应停止；失败后最多每 60 秒做一次恢复，不忙等、不静默
推进来源。时钟、等待和停止事件均可注入，便于确定性测试。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

_BEIJING = ZoneInfo("Asia/Shanghai")
_LOGGER = logging.getLogger(__name__)
_SLEEP_CHUNK_SECONDS = 60.0


def next_run_at(
    now: datetime, *, hour: int = 2, zone: str = "Asia/Shanghai"
) -> datetime:
    """计算下一次每日运行时间。

    参数:
        now: 当前时间；无时区时按目标时区解释。
        hour: 每日小时，0–23。
        zone: IANA 时区名。
    返回值:
        datetime: 目标时区下一次运行时间。
    异常:
        ValueError: hour 越界或时区无效。
    """
    if not 0 <= hour <= 23:
        raise ValueError("hour 必须在 0 到 23 之间")
    zone_info = ZoneInfo(zone)
    local_now = now.replace(tzinfo=zone_info) if now.tzinfo is None else now.astimezone(zone_info)
    target = local_now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= local_now:
        target += timedelta(days=1)
    return target


async def run_schedule(
    job: Callable[[], Awaitable[object]],
    recover: Callable[[], Awaitable[object]],
    *,
    clock: Callable[[], datetime],
    sleep: Callable[[float], Awaitable[None]],
    stop: asyncio.Event,
    hour: int = 2,
    zone: str = "Asia/Shanghai",
) -> None:
    """运行启动恢复与每日定时任务。

    参数:
        job: 完整挖掘任务。
        recover: 仅恢复 pending 的任务。
        clock: 当前时间函数。
        sleep: 可注入等待函数。
        stop: 停止事件。
        hour: 每日触发小时。
        zone: 时区。
    返回值:
        None
    异常:
        无；任务异常被记录为恢复尝试，不向调用方传播。
    """
    if not await _recover_until_ready(recover, sleep=sleep, stop=stop):
        return
    last_run_date = None
    while not stop.is_set():
        target = next_run_at(clock(), hour=hour, zone=zone)
        # 用已经触发的日期作为高水位，时钟回拨不能再次领取同一天。
        if last_run_date is not None and target.date() <= last_run_date:
            target = target.replace(
                year=last_run_date.year, month=last_run_date.month, day=last_run_date.day,
            ) + timedelta(days=1)
        while not stop.is_set() and clock() < target:
            remaining = (target - clock()).total_seconds()
            if not await _wait_or_stop(
                min(_SLEEP_CHUNK_SECONDS, max(0.0, remaining)), sleep=sleep, stop=stop,
            ):
                return
        if stop.is_set():
            break
        # 失败任务也算已经触发；随后只恢复它，不再次领取同日新任务。
        current = clock()
        local_current = (
            current.replace(tzinfo=ZoneInfo(zone)) if current.tzinfo is None
            else current.astimezone(ZoneInfo(zone))
        )
        # 跨日向前跳钟时，以实际触发日记账，随后回拨也不能再触发该日。
        last_run_date = max(target.date(), local_current.date())
        try:
            await job()
        except Exception as error:
            _LOGGER.warning("knowledge scheduled job failed: %s", type(error).__name__)
            if not await _wait_or_stop(_SLEEP_CHUNK_SECONDS, sleep=sleep, stop=stop):
                return
            if not await _recover_until_ready(recover, sleep=sleep, stop=stop):
                return


async def _wait_or_stop(
    seconds: float, *, sleep: Callable[[float], Awaitable[None]], stop: asyncio.Event,
) -> bool:
    """等待期限或停止事件，退出时清理未完成的等待。

    参数：seconds 非负等待秒数；sleep 可注入异步等待；stop 停止事件。
    返回：bool，正常等完为 True，停止为 False。
    异常：CancelledError 外部取消；等待函数异常原样传播。
    """
    if stop.is_set():
        return False
    sleeper = asyncio.create_task(sleep(seconds))
    stopper = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait([sleeper, stopper], return_when=asyncio.FIRST_COMPLETED)
        if stop.is_set():
            return False
        await sleeper
        return True
    finally:
        for task in (sleeper, stopper):
            if not task.done():
                task.cancel()
        await asyncio.gather(sleeper, stopper, return_exceptions=True)


async def _recover_until_ready(
    recover: Callable[[], Awaitable[object]], *,
    sleep: Callable[[float], Awaitable[None]], stop: asyncio.Event,
) -> bool:
    """启动和任务故障后共用限频恢复，成功前不领取每日任务。

    参数：recover 恢复函数；sleep 异步等待函数；stop 停止事件。
    返回：bool，恢复成功为 True，停止为 False。
    异常：CancelledError 外部取消；恢复业务异常仅记录类型并重试。
    """
    while not stop.is_set():
        try:
            await recover()
            return not stop.is_set()
        except Exception as error:
            _LOGGER.warning("knowledge scheduled recovery failed: %s", type(error).__name__)
            if not await _wait_or_stop(_SLEEP_CHUNK_SECONDS, sleep=sleep, stop=stop):
                return False
    return False
