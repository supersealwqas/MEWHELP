"""评估作业所有权、截止时间、进程身份与资源清理的离线行为测试。"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
from types import SimpleNamespace
from contextlib import asynccontextmanager

import psutil
import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("startup_failure", [False, True])
def test_scheduler_then_manager_close_before_runtime(monkeypatch, startup_failure):
    """定时器和自有进程必须在数据库退出前收回，启动失败也不漏清理。参数：替身/失败开关；返回：None；异常：断言失败。"""
    import app.main as main
    events = []

    class Manager:
        """记录真实lifespan调用的外部作业所有者。"""
        async def close(self):
            """记录清理。参数：无；返回：None；异常：无。"""
            events.append("manager")

    class Scheduler:
        """以最小边界记录应用启动/清理顺序。"""
        def __init__(self, *args, **kwargs):
            """接收调度配置。参数：任意；返回：None；异常：无。"""

        def start(self):
            """模拟启动失败。参数：无；返回：None；异常：RuntimeError。"""
            if startup_failure:
                raise RuntimeError("调度启动失败")

        async def close(self):
            """记录先停止新触发。参数：无；返回：None；异常：无。"""
            events.append("scheduler")

    @asynccontextmanager
    async def runtime():
        """标记数据库资源退出时点。参数：无；返回：服务；异常：透传。"""
        try:
            yield SimpleNamespace(evaluations=object())
        finally:
            events.append("runtime")

    monkeypatch.setattr(main, "create_job_manager", lambda: Manager())
    monkeypatch.setattr("app.evaluation.scheduler.EvaluationScheduler", Scheduler)
    application = main.create_app(runtime_factory=runtime)
    if startup_failure:
        with pytest.raises(RuntimeError):
            with TestClient(application):
                pass
    else:
        with TestClient(application):
            pass
    assert events[:3] == ["scheduler", "manager", "runtime"]


def test_production_api_shares_manager_and_reports_global_busy(monkeypatch, tmp_path):
    """生产/旧重跑共用忙状态，分页保留None且API明确数据库占用。参数：替身/目录；返回：None；异常：断言失败。"""
    import app.main as main
    from app.api import rageval
    events = []
    parent = ProcessDouble()
    manager, calls = _manager(monkeypatch, tmp_path, [parent])

    class Repository:
        """只替换外部数据库读取。"""
        async def list_runs(self, after_id, limit):
            """记录分页边界并保留缺失指标。参数：游标/页长；返回：页；异常：无。"""
            events.append((after_id, limit))
            return {"items": [{"id": "7", "metrics": {"faithfulness": None}}], "next_after_id": "7"}

        async def execution_busy(self):
            """模拟其他进程已持有数据库执行锁。参数：无；返回：bool；异常：无。"""
            return True

    @asynccontextmanager
    async def runtime():
        """只保留API必需资源。参数：无；返回：服务；异常：无。"""
        yield SimpleNamespace(evaluations=Repository())

    monkeypatch.setattr(main, "create_job_manager", lambda: manager)
    with TestClient(main.create_app(runtime_factory=runtime)) as client:
        assert client.get("/api/rag-eval/production/job").json()["database_busy"] is True
        page = client.get("/api/rag-eval/production/runs?after_id=5&limit=2")
        assert page.status_code == 200 and page.json()["items"][0]["metrics"]["faithfulness"] is None
        assert events == [(5, 2)]
        response = client.post("/api/rag-eval/production/rerun")
        assert response.status_code == 200 and response.json()["admission"] == "pending"
        assert client.post("/api/rag-eval/rerun").status_code == 409
        assert client.post("/api/rag-eval/production/rerun").status_code == 409
    assert len(calls) <= 1 and not parent.is_running() if calls else True


def test_production_api_without_lifespan_is_unavailable():
    """无数据库生命周期时所有生产入口返回503。参数：无；返回：None；异常：断言失败。"""
    from app.main import create_app
    client = TestClient(create_app())
    for path in ("job", "runs"):
        assert client.get("/api/rag-eval/production/" + path).status_code == 503
    assert client.post("/api/rag-eval/production/rerun").status_code == 503


class ProcessDouble:
    """模拟进程系统边界；管理、状态落盘和阶段串行均使用真实代码。"""

    def __init__(self, pid=101, code=None, *, children=(), stuck=False):
        """保存脚本状态。参数：pid身份/code退出码/children子进程/stuck拒绝结束；返回：None；异常：无。"""
        self.pid, self.returncode = pid, code
        self.descendants, self.stuck = list(children), stuck
        self.signals, self.reused = [], False

    def create_time(self):
        """返回冻结创建时间。参数：无；返回：float；异常：无。"""
        return 123.0

    def is_running(self):
        """检查原进程身份。参数：无；返回：bool；异常：无。"""
        return self.returncode is None and not self.reused

    def children(self, recursive=False):
        """只提供本进程子树。参数：recursive递归开关；返回：list；异常：无。"""
        assert recursive
        return self.descendants if self.is_running() else []

    def poll(self):
        """读取退出码。参数：无；返回：int或None；异常：无。"""
        return self.returncode

    def terminate(self):
        """记录温和结束，保持子进程列表模拟父进程先退出。参数：无；返回：None；异常：AccessDenied。"""
        self.signals.append("terminate")
        if self.stuck:
            raise psutil.AccessDenied(self.pid)
        self.returncode = -15

    def kill(self):
        """记录强制结束。参数：无；返回：None；异常：AccessDenied。"""
        self.signals.append("kill")
        if self.stuck:
            raise psutil.AccessDenied(self.pid)
        self.returncode = -9

    def wait(self, timeout=None):
        """记录回收结果。参数：timeout期限；返回：退出码；异常：TimeoutExpired。"""
        if self.returncode is None:
            raise psutil.TimeoutExpired(timeout or 0, self.pid)
        return self.returncode


def _manager(monkeypatch, tmp_path, processes, *, timeout=1):
    """替换外部进程调用并建立真实管理器。参数：替换器/目录/脚本进程/期限；返回：管理器和启动记录；异常：断言失败。"""
    assert importlib.util.find_spec("app.evaluation.jobs") is not None, "评估作业缺少可关闭的生命周期所有者"
    from app.evaluation import jobs

    launched = []

    def spawn(argv, **kwargs):
        """提供逐阶段进程。参数：argv/kwargs系统调用参数；返回：进程；异常：脚本异常。"""
        launched.append((argv, kwargs))
        outcome = processes[len(launched) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def wait_procs(procs, timeout):
        """立即返回脚本中的存活情况。参数：procs/timeout；返回：gone,alive；异常：无。"""
        return [p for p in procs if not p.is_running()], [p for p in procs if p.is_running()]

    monkeypatch.setattr(jobs.psutil, "Popen", spawn)
    monkeypatch.setattr(jobs.psutil, "wait_procs", wait_procs)
    manager = jobs.EvaluationJobManager(tmp_path, tmp_path / "reports", timeout_seconds=timeout, cleanup_seconds=0.1)
    return manager, launched


async def _finished(manager):
    """有界等待终态。参数：manager真实管理器；返回：dict状态；异常：TimeoutError。"""
    async with asyncio.timeout(3):
        while manager.status()["status"] == "running":
            await asyncio.sleep(0.01)
    return manager.status()


@pytest.mark.parametrize("code,status", [(0, "completed"), (2, "completed_with_errors"), (3, "busy"), (4, "persist_failed"), (1, "failed")])
async def test_production_runs_once_and_reports_database_admission(monkeypatch, tmp_path, code, status):
    """生产作业不得校准且退出码不能掩盖争锁/补存失败。参数：外部进程替身及状态；返回：None；异常：断言失败。"""
    manager, calls = _manager(monkeypatch, tmp_path, [ProcessDouble(code=code)])
    result = manager.start(tmp_path / "dataset.json", [], profile="production",
        policy_path=tmp_path / "frozen.json", triggered_by="定时")
    try:
        assert result["admission"] == "pending" and result["profile"] == "production"
        assert (await _finished(manager))["status"] == status
        assert len(calls) == 1
        argv = calls[0][0]
        assert argv[3] == "evaluate-production" and "--split" not in argv and "--strategies" not in argv
        assert argv[argv.index("--policy") + 1] == str(tmp_path / "frozen.json")
        assert argv[argv.index("--triggered-by") + 1] == "定时"
    finally:
        await manager.close()


async def test_success_runs_both_stages_and_keeps_unique_artifacts(monkeypatch, tmp_path):
    """串行policy路径、解释器或独占产物命名错误时失败。参数：隔离边界；返回：None；异常：断言失败。"""
    manager, calls = _manager(monkeypatch, tmp_path, [ProcessDouble(code=0) for _ in range(4)])
    first = manager.start(tmp_path / "dataset.json", ["dense"])
    try:
        assert (await _finished(manager))["status"] == "done"
        second = manager.start(tmp_path / "dataset.json", ["dense"])
        assert (await _finished(manager))["status"] == "done"
        assert first["comparison_dir"] != second["comparison_dir"]
        assert calls[0][0][:5] == [sys.executable, "-m", "app.evaluation.cli", "calibrate", "--split"]
        assert str(tmp_path / "reports" / first["calibration_dir"] / "policy.json") in calls[1][0]
    finally:
        await manager.close()


@pytest.mark.parametrize("stages,expected_calls,code", [([ProcessDouble(code=2)], 1, 2),
    ([ProcessDouble(code=0), ProcessDouble(code=3)], 2, 3), ([OSError("spawn failed")], 1, None)])
async def test_failed_stage_records_terminal_state(monkeypatch, tmp_path, stages, expected_calls, code):
    """启动错误或阶段失败继续执行、留下running时失败。参数：脚本阶段；返回：None；异常：断言失败。"""
    manager, calls = _manager(monkeypatch, tmp_path, stages)
    manager.start(tmp_path / "dataset.json", ["dense"])
    try:
        state = await _finished(manager)
        assert state["status"] == "failed" and state["returncode"] == code
        assert state["finished_at"] and len(calls) == expected_calls
        assert json.loads((tmp_path / "reports/rageval-job.json").read_text(encoding="utf-8"))["status"] == "failed"
    finally:
        await manager.close()


async def test_deadline_cleans_parent_and_children_before_terminal_state(monkeypatch, tmp_path):
    """超时仅改状态不结束子树、或先杀父导致丢子时失败。参数：隔离进程；返回：None；异常：断言失败。"""
    child = ProcessDouble(pid=102)
    parent = ProcessDouble(children=[child])
    manager, calls = _manager(monkeypatch, tmp_path, [parent], timeout=0.03)
    manager.start(tmp_path / "dataset.json", ["dense"])
    try:
        state = await _finished(manager)
        assert state["status"] == "failed" and "超时" in state["message"]
        assert not parent.is_running() and not child.is_running()
        assert len(calls) == 1 and state["finished_at"]
    finally:
        await manager.close()


async def test_deadline_is_shared_by_calibration_and_comparison(monkeypatch, tmp_path):
    """每阶段重置超时预算导致总运行翻倍时失败。参数：隔离进程；返回：None；异常：断言失败。"""
    first, second = ProcessDouble(), ProcessDouble(pid=102)
    manager, calls = _manager(monkeypatch, tmp_path, [first, second], timeout=0.2)
    loop = asyncio.get_running_loop()
    loop.call_later(0.12, setattr, first, "returncode", 0)
    loop.call_later(0.29, setattr, second, "returncode", 0)
    manager.start(tmp_path / "dataset.json", ["dense"])
    try:
        state = await _finished(manager)
        assert len(calls) == 2 and state["status"] == "failed" and "超时" in state["message"]
        assert second.signals
    finally:
        await manager.close()


async def test_close_waits_for_owned_processes_and_blocks_concurrent_start(monkeypatch, tmp_path):
    """并发启动或关闭返回时还遗留进程时失败。参数：隔离进程；返回：None；异常：断言失败。"""
    child = ProcessDouble(pid=102)
    parent = ProcessDouble(children=[child])
    manager, calls = _manager(monkeypatch, tmp_path, [parent])
    manager.start(tmp_path / "dataset.json", ["dense"])
    with pytest.raises(RuntimeError, match="运行"):
        manager.start(tmp_path / "dataset.json", ["dense"])
    await asyncio.sleep(0.01)
    assert manager.status()["pid"] == 101 and not parent.signals
    await manager.close()
    assert len(calls) == 1 and not parent.is_running() and not child.is_running()
    assert manager.status()["status"] == "interrupted"


async def test_cancel_during_spawn_cannot_lose_process_handle(monkeypatch, tmp_path):
    """启动返回句柄时到达取消导致子进程泄漏时失败。参数：隔离进程；返回：None；异常：断言失败。"""
    parent = ProcessDouble()
    manager, _ = _manager(monkeypatch, tmp_path, [parent])
    from app.evaluation import jobs

    def cancelled_spawn(*args, **kwargs):
        """模拟句柄返回同一时刻发生取消。参数：启动参数；返回：进程；异常：无。"""
        asyncio.current_task().cancel()
        return parent

    monkeypatch.setattr(jobs.psutil, "Popen", cancelled_spawn)
    manager.start(tmp_path / "dataset.json", ["dense"])
    try:
        assert (await _finished(manager))["status"] == "interrupted"
        assert not parent.is_running()
    finally:
        await manager.close()


async def test_cleanup_failure_is_reported_and_prevents_new_job(monkeypatch, tmp_path):
    """清理失败仍宣称资源回收成功或允许重跑时失败。参数：拒绝结束进程；返回：None；异常：断言失败。"""
    parent = ProcessDouble(stuck=True)
    manager, _ = _manager(monkeypatch, tmp_path, [parent], timeout=0.02)
    manager.start(tmp_path / "dataset.json", ["dense"])
    try:
        state = await _finished(manager)
        assert state["status"] == "failed" and state["cleanup_error"]
        assert parent.signals == ["terminate", "kill"]
        with pytest.raises(RuntimeError, match="清理"):
            manager.start(tmp_path / "dataset.json", ["dense"])
    finally:
        parent.stuck = False
        await manager.close()


async def test_recycled_identity_is_never_signalled(monkeypatch, tmp_path):
    """原PID被重用时错误终止新进程。参数：隔离进程；返回：None；异常：断言失败。"""
    parent = ProcessDouble()
    manager, _ = _manager(monkeypatch, tmp_path, [parent])
    manager.start(tmp_path / "dataset.json", ["dense"])
    await asyncio.sleep(0.01)
    parent.reused = True
    await manager.close()
    assert not parent.signals


async def test_unowned_persisted_pid_is_neither_queried_nor_signalled(monkeypatch, tmp_path):
    """状态文件里的任意PID被当作拥有的进程时失败。参数：隔离目录；返回：None；异常：断言失败。"""
    manager, _ = _manager(monkeypatch, tmp_path, [])
    from app.evaluation import jobs
    root = tmp_path / "reports"
    root.mkdir()
    (root / "rageval-job.json").write_text(json.dumps({"status": "running", "pid": 99999,
        "process_created_at": 123.0}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        """禁止按落盘PID访问进程系统。参数：任意；返回：无；异常：AssertionError。"""
        raise AssertionError("无所有权不得读取或终止任意进程")

    monkeypatch.setattr(jobs.psutil, "Process", forbidden)
    assert manager.status()["status"] == "interrupted"
    await manager.close()


async def test_cancelled_job_reclaims_process_and_records_interruption(monkeypatch, tmp_path):
    """取消旧后台协程仍遗留进程的回归。参数：隔离外部进程；返回：None；异常：断言失败。"""
    parent = ProcessDouble()
    manager, _ = _manager(monkeypatch, tmp_path, [parent])
    manager.start(tmp_path / "dataset.json", ["dense"])
    await asyncio.sleep(0.01)
    await manager.close()
    assert manager.status()["status"] == "interrupted"
    assert not parent.is_running()


async def test_close_during_cleanup_waits_for_the_existing_cleanup(monkeypatch, tmp_path):
    """清理线程被取消后丢失、重复清理或提前返回时失败。参数：线程握手边界；返回：None；异常：断言失败。"""
    parent = ProcessDouble(code=0)
    manager, _ = _manager(monkeypatch, tmp_path, [parent, ProcessDouble(code=0)])
    from app.evaluation import jobs
    entered, release = threading.Event(), threading.Event()
    active, peak = 0, 0

    def slow_wait(procs, timeout):
        """模拟尚在回收的系统调用。参数：procs/timeout；返回：gone/alive；异常：无。"""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        release.wait(1)
        active -= 1
        return list(procs), []

    monkeypatch.setattr(jobs.psutil, "wait_procs", slow_wait)
    manager.start(tmp_path / "dataset.json", ["dense"])
    async with asyncio.timeout(2):
        while not entered.is_set():
            await asyncio.sleep(0.005)
        closer = asyncio.create_task(manager.close())
        await asyncio.sleep(0.03)
        try:
            assert not closer.done()
            assert peak == 1
        finally:
            release.set()
            await closer
    assert active == 0 and manager.status()["status"] == "interrupted"


async def test_identity_read_failure_does_not_lose_new_process(monkeypatch, tmp_path):
    """取得句柄后查询创建时间失败导致丢所有权时失败。参数：隔离进程；返回：None；异常：断言失败。"""
    parent = ProcessDouble()
    manager, _ = _manager(monkeypatch, tmp_path, [parent])

    def denied():
        """模拟身份元信息读取被拒。参数：无；返回：无；异常：AccessDenied。"""
        raise psutil.AccessDenied(parent.pid)

    parent.create_time = denied
    manager.start(tmp_path / "dataset.json", ["dense"])
    try:
        assert (await _finished(manager))["status"] == "failed"
        assert not parent.is_running()
    finally:
        await manager.close()


def test_http_owner_reports_running_conflict_and_cleans_on_shutdown(monkeypatch, tmp_path):
    """HTTP重跑绕过唯一owner、状态查询发信号或关闭遗留进程时失败。参数：隔离应用；返回：None；异常：断言失败。"""
    from app.api import rageval
    from app.main import create_app

    parent = ProcessDouble()
    _, launched = _manager(monkeypatch, tmp_path, [parent])
    monkeypatch.setattr(rageval, "REPORTS_ROOT", tmp_path / "reports")
    monkeypatch.setattr(rageval, "JOB_STATE", tmp_path / "reports/job.json")
    monkeypatch.setattr(rageval, "JOB_LOG", tmp_path / "reports/job.log")
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")

    @asynccontextmanager
    async def runtime():
        """替代聊天数据库/模型资源。参数：无；返回：服务占位；异常：无。"""
        yield object()

    application = create_app(runtime_factory=runtime)
    with TestClient(application) as client:
        response = client.post("/api/rag-eval/rerun", params={"dataset": str(dataset), "strategies": "dense"})
        assert response.status_code == 200 and response.json()["started"] is True
        assert client.post("/api/rag-eval/rerun", params={"dataset": str(dataset)}).status_code == 409
        assert client.get("/api/rag-eval/job").json()["status"] == "running"
        assert client.get("/api/rag-eval/overview").json()["job"]["status"] == "running"
        assert not parent.signals
    assert len(launched) == 1 and not parent.is_running()
    assert json.loads((tmp_path / "reports/job.json").read_text(encoding="utf-8"))["status"] == "interrupted"


def test_http_without_lifespan_does_not_start_unowned_job(monkeypatch, tmp_path):
    """没有应用生命周期时仍创建无主后台任务会失败。参数：隔离目录；返回：None；异常：断言失败。"""
    from app.main import create_app
    _, launched = _manager(monkeypatch, tmp_path, [])
    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    client = TestClient(create_app())
    assert client.post("/api/rag-eval/rerun", params={"dataset": str(dataset)}).status_code == 503
    assert not launched


@pytest.mark.parametrize("action", ["close", "timeout", "complete"])
async def test_real_owned_parent_and_child_are_reaped_without_query_side_effects(monkeypatch, tmp_path, action):
    """真实Windows父子进程在关闭/超时/完成后残留或查询改变存活时失败。参数：独占目录/action；返回：None；异常：断言失败。"""
    from app.evaluation import jobs

    original_popen = psutil.Popen
    created = []
    child_lifetime = "0.3" if action == "complete" else "20"
    ready = tmp_path / "ready.json"
    script = ("import subprocess,sys,time,json; from pathlib import Path; "
              f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep({child_lifetime})']); "
              f"Path({str(ready)!r}).write_text(json.dumps({{'pid':p.pid}})); p.wait()")

    def safe_spawn(argv, **kwargs):
        """只启动本测试的短命Python父子，不执行模型CLI。参数：管理器启动参数；返回：真实进程；异常：OSError。"""
        process = original_popen([sys.executable, "-c", script], **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(jobs.psutil, "Popen", safe_spawn)
    manager = jobs.EvaluationJobManager(tmp_path, tmp_path / "reports",
        timeout_seconds=1 if action == "timeout" else 5, cleanup_seconds=0.3)
    manager.start(tmp_path / "unused.json", ["dense"])
    owned = []
    try:
        async with asyncio.timeout(3):
            while not ready.exists():
                await asyncio.sleep(0.01)
        parent = created[0]
        children = parent.children(recursive=True)
        owned = [parent, *children]
        assert children and json.loads(ready.read_text())["pid"] in [p.pid for p in children]
        before = [(p.pid, p.create_time()) for p in owned]
        for _ in range(3):
            assert manager.status()["status"] == "running"
        assert all(p.is_running() for p in owned)
        if action == "close":
            await manager.close()
            assert manager.status()["status"] == "interrupted"
        else:
            state = await _finished(manager)
            assert state["status"] == ("failed" if action == "timeout" else "done")
        assert not any(p.is_running() for p in owned)
        assert all(p.poll() is not None for p in created)
        print(json.dumps({"action": action, "owned_identities": before,
                          "reaped_count": len(owned), "status": manager.status()["status"]}))
    finally:
        await manager.close()
        # 双重保险只回收已记录的自建对象；不用状态文件或任意PID重新取得身份。
        for process in [*owned, *created]:
            try:
                if process.is_running():
                    process.kill()
                process.wait(timeout=2)
            except psutil.NoSuchProcess:
                pass
