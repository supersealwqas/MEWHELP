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
    await recover()
    while not stop.is_set():
        target = next_run_at(clock(), hour=hour, zone=zone)
        while not stop.is_set() and clock() < target:
            remaining = (target - clock()).total_seconds()
            await sleep(min(_SLEEP_CHUNK_SECONDS, max(0.0, remaining)))
        if stop.is_set():
            break
        try:
            await job()
        except Exception as error:
            _LOGGER.warning("knowledge scheduled job failed: %s", type(error).__name__)
            await sleep(_SLEEP_CHUNK_SECONDS)
            try:
                await recover()
            except Exception as recovery_error:
                _LOGGER.warning(
                    "knowledge scheduled recovery failed: %s",
                    type(recovery_error).__name__,
                )
