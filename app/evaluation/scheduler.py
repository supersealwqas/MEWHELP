"""应用内每日生产评估调度；只安排未来一次，不追补停机或冲突期间的历史任务。"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone as utc_timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def next_run(now: datetime, *, hour: int = 3, minute: int = 0, timezone: str = "Asia/Shanghai") -> datetime:
    """计算严格晚于当前时刻的下一次本地执行时间。

    参数：now必须带时区；hour为0–23，minute为0–59，timezone为IANA时区。
    返回：目标时区datetime；异常：ValueError或ZoneInfoNotFoundError，参数非法。
    """
    if now.tzinfo is None or now.utcoffset() is None or not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("每日评估需要有效带时区时间及钟点")
    local = now.astimezone(ZoneInfo(timezone))
    candidate = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return candidate if candidate > local else candidate + timedelta(days=1)


class EvaluationScheduler:
    """仅拥有定时睡眠任务，执行进程始终由注入的共同manager持有。"""

    def __init__(self, manager, dataset: Path, policy_path: Path, *, hour: int = 3, minute: int = 0,
                 timezone: str = "Asia/Shanghai", now=None, sleep=asyncio.sleep):
        """保存调度参数。参数：manager共享所有者、题集/策略、钟点、时区、可注入时钟；返回：None；异常：时间配置错误。"""
        self.manager, self.dataset, self.policy_path = manager, dataset, policy_path
        self.hour, self.minute, self.timezone = hour, minute, timezone
        self.now, self.sleep = now or (lambda: datetime.now(utc_timezone.utc)), sleep
        self.next_at = next_run(self.now(), hour=hour, minute=minute, timezone=timezone)
        self._task, self._closed = None, False

    def start(self) -> None:
        """安排下一次未来任务，重复start幂等。参数：无；返回：None；异常：RuntimeError已关闭或无事件循环。"""
        if self._closed:
            raise RuntimeError("每日评估调度已关闭")
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        """睡眠到点后尝试一次，冲突/失败不排队补跑。参数：无；返回：None；异常：取消传播。"""
        while not self._closed:
            self.next_at = next_run(self.now(), hour=self.hour, minute=self.minute, timezone=self.timezone)
            await self.sleep(max(0, (self.next_at - self.now()).total_seconds()))
            if self._closed:
                return
            try:
                self.manager.start(self.dataset, [], profile="production", policy_path=self.policy_path, triggered_by="定时")
            except Exception as error:
                logger.info("daily evaluation skipped error_type=%s", type(error).__name__)

    async def close(self) -> None:
        """停止新触发并收回睡眠任务，随后应由调用方关闭manager。参数：无；返回：None；异常：外部取消。"""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
