"""评估作业的进程所有者：串行启动校准/比较、维护状态并在超时或关闭时回收自有子树。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import psutil


class EvaluationJobManager:
    """单应用实例的评估作业所有者，仅管理本实例亲自启动并记录身份的进程。"""

    def __init__(self, repo_root: Path, reports_root: Path, *, timeout_seconds: float = 3600,
                 cleanup_seconds: float = 3, state_path: Path | None = None, log_path: Path | None = None):
        """配置实例。

        参数：repo_root项目根目录；reports_root产物根目录；timeout_seconds整体时限（正秒数）；
            cleanup_seconds每轮清理等待（正秒数）；state_path/log_path可选状态文件与日志路径。
        返回：None。
        异常：ValueError，期限非正数。
        """
        if timeout_seconds <= 0 or cleanup_seconds <= 0:
            raise ValueError("作业与清理期限必须大于零")
        self.repo_root, self.reports_root = repo_root, reports_root
        self.state_path = state_path or reports_root / "rageval-job.json"
        self.log_path = log_path or reports_root / "rageval-job.log"
        self.timeout_seconds, self.cleanup_seconds = timeout_seconds, cleanup_seconds
        self._state: dict | None = None
        self._task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._owned: dict[tuple[int, float | None], psutil.Process] = {}
        self._closed = False

    def start(self, dataset: Path, strategies: list[str], *, profile: str = "legacy", policy_path: Path | None = None,
              triggered_by: str = "手动") -> dict:
        """同步接受本实例作业，生产数据库准入由子进程报告。

        参数：题集/策略列表、legacy或production、生产冻结策略路径、中文触发来源。
        返回：started/本实例状态，production初始admission=pending；异常：配置非法、关闭/忙/清理冲突。
        """
        if profile not in {"legacy", "production"} or triggered_by not in {"手动", "定时"}:
            raise ValueError("评估profile或触发来源非法")
        if profile == "production" and (policy_path is None or strategies):
            raise ValueError("生产评估只接受一个明确冻结策略路径，不接受旧策略列表")
        if self._closed:
            raise RuntimeError("评估管理器已关闭")
        if self._task is not None and not self._task.done():
            raise RuntimeError("已有评估任务在运行，请等待结束")
        if self._owned or (self._state or {}).get("cleanup_error"):
            raise RuntimeError("上次评估进程清理未完成，不能启动新任务")
        self.reports_root.mkdir(parents=True, exist_ok=True)
        # 随机后缀避免同秒重跑与已有CLI产物冲突，不创建CLI要求尚不存在的运行目录。
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        token = uuid4().hex[:8]
        calibration = self.reports_root / f"{stamp}-calibrate-{token}"
        comparison = self.reports_root / f"{stamp}-compare-{token}"
        if profile == "production":
            comparison = self.reports_root / f"{stamp}-production-{token}"
        self._state = {"status": "running", "pid": None, "process_created_at": None,
            "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
            "returncode": None, "dataset": dataset.name, "strategies": list(strategies),
            "calibration_dir": calibration.name, "comparison_dir": comparison.name,
            "message": "正在校准并比较策略，完成后刷新页面查看"}
        self._state.update(profile=profile, triggered_by=triggered_by)
        if profile == "production":
            self._state.update(admission="pending", calibration_dir=None, artifact_dir=comparison.name,
                message="本实例已接受；子进程取得数据库执行锁后才开始冻结策略评估")
        self._write_state()
        self._task = asyncio.create_task(self._run(dataset, list(strategies), calibration, comparison,
            profile=profile, policy_path=policy_path, triggered_by=triggered_by))
        return {"started": True, **self._state}

    def status(self) -> dict:
        """只读内存或历史状态，不按落盘PID探活。参数：无；返回：状态副本；异常：无。"""
        if self._state is not None:
            return dict(self._state)
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("状态必须为对象")
        except (OSError, ValueError, UnicodeError):
            return {"status": "idle", "message": "尚未运行"}
        if state.get("status") == "running":
            # 文件不是进程所有权凭证；服务异常退出后的旧PID绝不能被本实例接管或终止。
            state = {**state, "status": "interrupted",
                     "message": "先前作业的管理器已退出，无法确认其进程是否回收"}
        return state

    async def close(self) -> None:
        """关闭入口并等候自有作业完成清理。参数：无；返回：None；异常：取消在自有清理后传播。"""
        self._closed = True
        if self._task is not None and not self._task.done():
            if not self._task.cancelling():
                self._task.cancel()
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError:
                # 任务尚未执行到首行也可能被取消；此时没有进程，仍需落下终态。
                if not self._task.done():
                    await asyncio.shield(self._task)
        if self._state is not None and self._state["status"] == "running":
            self._finish("interrupted", "应用关闭，评估任务已取消")
        if self._owned:
            error = await self._clean_owned(ignore_cancellation=True)
            if error:
                self._finish("failed", "评估进程清理失败", cleanup_error=error)

    async def _run(self, dataset: Path, strategies: list[str], calibration: Path, comparison: Path, *,
                   profile: str, policy_path: Path | None, triggered_by: str) -> None:
        """共用超时/进程回收；生产只运行一次evaluate。参数：题集/策略/目录/profile/冻结路径/来源；返回：None；异常：记录终态。"""
        status, message, returncode = "done", "评估完成，刷新页面查看最新结果", 0
        cleanup_error = None
        try:
            with self.log_path.open("w", encoding="utf-8", buffering=1) as log:
                async with asyncio.timeout(self.timeout_seconds):
                    stages = (("evaluate-production", "evaluation", comparison),) if profile == "production" else (
                        ("calibrate", "calibration", calibration), ("compare", "evaluation", comparison))
                    for index, (command, split, output) in enumerate(stages, 1):
                        argv = [sys.executable, "-m", "app.evaluation.cli", command, "--split", split,
                                "--dataset", str(dataset), "--strategies", *strategies, "--output-dir", str(output)]
                        if command == "compare":
                            argv += ["--policy", str(calibration / "policy.json")]
                        if profile == "production":
                            argv = [sys.executable, "-m", "app.evaluation.cli", command, "--dataset", str(dataset),
                                "--policy", str(policy_path), "--triggered-by", triggered_by,
                                "--target", "application-readonly", "--output-dir", str(output)]
                        log.write("\n$ " + " ".join(argv) + "\n")
                        # Popen同步取得句柄：创建与登记之间没有取消点，不丢失已启动进程。
                        process = psutil.Popen(argv, cwd=str(self.repo_root), stdout=log, stderr=subprocess.STDOUT,
                                               env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                                               **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
                        # 先保留带psutil身份保护的句柄，元信息读取失败也不能丢所有权。
                        self._owned[(process.pid, None)] = process
                        self._state["pid"] = process.pid
                        identity = (process.pid, process.create_time())
                        self._owned[identity] = process
                        del self._owned[(process.pid, None)]
                        self._state.update(pid=process.pid, process_created_at=identity[1], stage=index)
                        self._write_state()
                        while True:
                            self._remember_children()
                            returncode = process.poll()
                            if returncode is not None:
                                break
                            await asyncio.sleep(0.02)
                        cleanup_error = await self._clean_owned()
                        if cleanup_error:
                            status, message = "failed", "评估进程清理失败"
                            break
                        if profile == "production":
                            status, message = {0: ("completed", "冻结策略评估已完成并保存"),
                                2: ("completed_with_errors", "评估已保存，部分题发生真实错误，分母保持完整"),
                                3: ("busy", "数据库已有评估执行者，本次未调用模型"),
                                4: ("persist_failed", "评估文件已完成，仅落表失败，可使用persist-run补存")}.get(
                                    returncode, ("failed", f"生产评估失败，退出码 {returncode}，见日志"))
                            self._state["admission"] = "busy" if returncode == 3 else "accepted" if returncode in (0, 2, 4) else "unknown"
                            break
                        if returncode != 0:
                            status, message = "failed", f"第 {index} 阶段失败，退出码 {returncode}，见日志"
                            break
        except TimeoutError:
            status, message, returncode = "failed", f"评估整体超时（{self.timeout_seconds:g}秒），已停止任务", None
        except asyncio.CancelledError:
            status, message, returncode = "interrupted", "评估任务已取消", None
        except Exception as error:
            status, message, returncode = "failed", f"评估启动或运行失败：{type(error).__name__}", None
        finally:
            # 终态必须在回收后写入；即使期限已耗尽，也给terminate/kill各一次有界等待。
            if self._owned and cleanup_error is None:
                cleanup_error = await self._clean_owned(ignore_cancellation=True)
            if cleanup_error:
                status, message = "failed", "评估进程清理失败，仍有资源未确认回收"
            self._finish(status, message, returncode=returncode, cleanup_error=cleanup_error)

    async def _clean_owned(self, *, ignore_cancellation: bool = False) -> str | None:
        """持有清理线程直到有界等待结束。参数：ignore_cancellation终态清理时延迟取消；返回：错误或None；异常：延迟的CancelledError。"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(asyncio.to_thread(self._cleanup))
        task, cancelled = self._cleanup_task, False
        # to_thread取消不会终止系统调用；持有同一个任务，避免重复清理与提前关闭。
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        self._cleanup_task = None
        if cancelled and not ignore_cancellation:
            raise asyncio.CancelledError
        return result

    def _remember_children(self) -> None:
        """冻结自有子树身份，使父退出后仍能回收已发现的子进程。参数：无；返回：None；异常：psutil访问错误。"""
        for process in list(self._owned.values()):
            try:
                if process.is_running():
                    for child in process.children(recursive=True):
                        self._owned[(child.pid, None)] = child
                        self._owned[(child.pid, child.create_time())] = child
                        del self._owned[(child.pid, None)]
            except psutil.NoSuchProcess:
                pass

    def _cleanup(self) -> str | None:
        """仅终止冻结的自有进程并分两轮等待。参数：无；返回：错误说明或None；异常：系统错误被转换为说明。"""
        discovery_error = None
        try:
            # 必须先冻结全部子进程，再终止父；不能用终止后的父PID重新枚举。
            self._remember_children()
        except psutil.Error as error:
            discovery_error = type(error).__name__
        processes = list(reversed(self._owned.values()))
        try:
            for method in ("terminate", "kill"):
                alive = []
                for process in processes:
                    try:
                        if process.is_running():
                            alive.append(process)
                            getattr(process, method)()
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied:
                        # 暂时无法温和结束仍继续强制结束；最终以是否存活决定成功。
                        pass
                _, processes = psutil.wait_procs(alive, timeout=self.cleanup_seconds)
            self._owned = {identity: process for identity, process in self._owned.items() if process.is_running()}
        except (psutil.Error, OSError) as error:
            return f"无法确认自有进程回收：{type(error).__name__}"
        if self._owned or discovery_error:
            return f"未回收进程数：{len(self._owned)}；子树检查：{discovery_error or '完成'}"
        return None

    def _finish(self, status: str, message: str, **fields) -> None:
        """写入包含完成时间的终态。参数：status状态/message说明/fields结果；返回：None；异常：无。"""
        self._state.update(status=status, message=message, finished_at=datetime.now(timezone.utc).isoformat(), **fields)
        self._write_state()

    def _write_state(self) -> None:
        """原子替换状态文件，写失败仍保留内存事实。参数：无；返回：None；异常：无。"""
        try:
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._state, ensure_ascii=False, indent=1), encoding="utf-8")
            temporary.replace(self.state_path)
        except OSError as error:
            self._state["state_write_error"] = type(error).__name__
