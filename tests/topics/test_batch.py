"""旁路批次边界：分页、HTTP核验、取消、恢复和版本证据。"""

import asyncio
from contextlib import asynccontextmanager
import importlib
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from app.knowledge.errors import ConcurrentKnowledgeWrite, LeaseLost
from app.topics.taxonomy import load_taxonomy, taxonomy_digest


class MemoryRepository:
    """替代外部MySQL边界的持久输入，SQL与真实锁由专库另验。"""

    def __init__(self, count=201):
        """参数：count池大小；返回：None；异常：无。"""
        self.rows = [dict(question_id=str(i), text=f"问题{i}", created_at="2020-01-01T00:00:00")
                     for i in range(1, count + 1)]
        self.saved, self.busy, self.lost = {}, False, False

    @asynccontextmanager
    async def execution(self):
        """参数：无；返回：租约；异常：占用错误。"""
        if self.busy:
            raise ConcurrentKnowledgeWrite("忙")
        self.busy = True
        try:
            yield self
        finally:
            self.busy = False

    async def assert_held(self):
        """参数：无；返回：None；异常：失锁错误。"""
        if self.lost:
            raise LeaseLost("失锁")

    async def high_watermark(self):
        """参数：无；返回：最大ID；异常：无。"""
        return max((int(r["question_id"]) for r in self.rows), default=0)

    async def pending(self, limit, reclassify=False, after_id=0):
        """参数：分页及重跑开关；返回：候选；异常：无。"""
        return [r for r in self.rows if int(r["question_id"]) > after_id
                and (reclassify or r["question_id"] not in self.saved)][:limit]

    async def save(self, rows, *, lease):
        """参数：结果及租约；返回：保存计数；异常：失锁错误。"""
        await lease.assert_held()
        for row in rows:
            self.saved[row["question_id"]] = row.copy()
        return {"saved": len(rows)}

    async def classified_rows(self):
        """参数：无；返回：当前分类；异常：无。"""
        return list(self.saved.values())

    async def execution_busy(self):
        """参数：无；返回：锁占用；异常：无。"""
        return self.busy


def make_service(tmp_path, repository, handler):
    """构建真实服务+受控HTTP。参数：目录/仓储/HTTP处理器；返回：服务；异常：缺实现断言。"""
    assert importlib.util.find_spec("app.services.topics"), "缺少主题批次服务"
    config = NS(topic_enabled=True, topic_service_url="http://127.0.0.1:18010", topic_batch_size=200,
                topic_reports_dir=str(tmp_path), topic_poll_seconds=60, topic_batch_min_size=20,
                topic_max_wait_seconds=86400, mysql_database="mewhelp_ch10_test")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return importlib.import_module("app.services.topics").TopicBatchService(repository, config, client)


def response(request, version="v1", uncertain=False):
    """构造独立合同响应。参数：HTTP请求/版本/未确定；返回：HTTP响应；异常：JSON错误。"""
    digest = taxonomy_digest(load_taxonomy())
    if request.url.path == "/health":
        return httpx.Response(200, json=dict(status="ready", model_version=version, taxonomy_digest=digest))
    texts = json.loads(request.content)["texts"]
    return httpx.Response(200, json=dict(model_version=version, taxonomy_digest=digest, results=[
        dict(labels=[] if uncertain else ["物流"], probabilities=[0.5] * 17, uncertain=uncertain, truncated=False)
        for _ in texts]))


async def test_uncertain_first_200_does_not_starve_201(tmp_path):
    """重读第一页会使第201条永远不归类。参数：隔离目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository()
    def handler(request):
        """参数：请求；返回：响应；异常：无。"""
        texts = json.loads(request.content).get("texts", []) if request.content else []
        return response(request, uncertain=bool(texts and texts[0] != "问题201"))
    service = make_service(tmp_path, repo, handler)
    report = await service.process()
    assert report["status"] == "complete" and report["counts"] == {"checked": 201, "saved": 1, "uncertain": 200}
    assert list(repo.saved) == ["201"] and report["cursor"] == "201"
    assert len(report["batches"]) == 2 and len(report["batches"][0]["uncertain_ids"]) == 200
    await service.close()


@pytest.mark.parametrize("kind", ["mixed", "nan", "labels", "duplicate", "count", "disconnect", "uncertain", "probability"])
async def test_invalid_second_http_block_never_commits_first(tmp_path, kind):
    """先块成功不能掩盖后块失败而落半批。参数：目录/故障类型；返回：None；异常：断言失败。"""
    repo, calls = MemoryRepository(100), []
    def handler(request):
        """参数：请求；返回：故障响应；异常：模拟断线。"""
        value = response(request)
        if request.url.path == "/classify":
            calls.append(1)
            if len(calls) == 2:
                if kind == "disconnect":
                    raise httpx.ConnectError("断线", request=request)
                body = value.json()
                if kind == "mixed": body["model_version"] = "v2"
                elif kind == "nan": body["results"][0]["probabilities"][0] = float("nan")
                elif kind == "labels": body["results"][0]["labels"] = ["其他", "物流"]
                elif kind == "duplicate": body["results"][0]["labels"] = ["物流", "物流"]
                elif kind == "count": body["results"].pop()
                elif kind == "uncertain": body["results"][0]["uncertain"] = True
                elif kind == "probability": body["results"][0]["probabilities"] = [1.1] * 17
                return httpx.Response(200, content=json.dumps(body))
        return value
    service = make_service(tmp_path, repo, handler)
    result = await service.process()
    assert result["status"] == "failed" and result["cursor"] == "0" and repo.saved == {}
    assert result["error"]["question_ids"] == [str(i) for i in range(1, 101)]
    await service.close()


async def test_restart_resumes_completed_cursor_and_reclassify_last_page(tmp_path):
    """失败后首批重复和重跑漏后页都会失败。参数：隔离目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(201)
    def failing(request):
        """参数：请求；返回：响应；异常：第201条断线。"""
        if request.content and "问题201" in request.content.decode():
            raise httpx.ConnectError("断线", request=request)
        return response(request)
    first = make_service(tmp_path, repo, failing)
    assert (await first.process())["cursor"] == "200"
    await first.close()
    second = make_service(tmp_path, repo, response)
    recovered = await second.process()
    assert recovered["status"] == "complete" and len(repo.saved) == 201
    assert len(recovered["batches"]) == 2
    rerun = await second.process(reclassify=True)
    assert rerun["counts"]["saved"] == 201 and len(rerun["batches"]) == 2
    await second.close()


async def test_close_cancels_http_without_commit(tmp_path):
    """停机遗留HTTP任务或半批提交会失败。参数：隔离目录；返回：None；异常：断言失败。"""
    repo, entered = MemoryRepository(3), asyncio.Event()
    async def handler(request):
        """参数：请求；返回：响应；异常：等待取消。"""
        if request.url.path == "/classify":
            entered.set()
            await asyncio.Future()
        return response(request)
    service = make_service(tmp_path, repo, handler)
    task = asyncio.create_task(service.process())
    await asyncio.wait_for(entered.wait(), 3)
    await service.close()
    assert task.cancelled() and repo.saved == {} and not repo.busy


async def test_lost_lease_does_not_advance_or_save(tmp_path):
    """网络期间失锁不能续存。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(3)
    def handler(request):
        """参数：请求；返回：响应；异常：无。"""
        if request.url.path == "/classify": repo.lost = True
        return response(request)
    service = make_service(tmp_path, repo, handler)
    report = await service.process()
    assert report["status"] == "failed" and report["cursor"] == "0" and not repo.saved
    await service.close()


async def test_model_version_requires_current_rows_matching_manifest(tmp_path):
    """最新批次不能冒充全表版本，原件丢失/覆盖必须未知。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(1)
    service = make_service(tmp_path, repo, response)
    await service.process()
    assert (await service.status())["model_version"] == "v1"
    repo.saved["1"]["labels"] = ["发票"]
    assert (await service.status())["model_version"] == "unknown"
    await service.close()


async def test_close_immediately_after_trigger_releases_unstarted_lease(tmp_path):
    """任务还未开始就取消时也必须释放已取得锁。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(1)
    service = make_service(tmp_path, repo, response)
    await service.trigger()
    await service.close()
    assert not repo.busy and not service._active


async def test_close_cancels_task_still_acquiring_execution_lease(tmp_path):
    """获取锁期间也属本实例任务，停机不能遗漏。参数：目录；返回：None；异常：断言失败。"""
    entered = asyncio.Event()
    class Acquiring(MemoryRepository):
        """等待外部连接，尚未进入分类步骤的仓储边界。"""
        @asynccontextmanager
        async def execution(self):
            """参数：无；返回：租约；异常：等待取消。"""
            entered.set()
            await asyncio.Future()
            yield self
    service = make_service(tmp_path, Acquiring(1), response)
    task = asyncio.create_task(service.process())
    await entered.wait()
    try:
        await service.close()
        assert task.cancelled()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_new_ids_after_frozen_high_watermark_wait_next_round(tmp_path):
    """扫描时追加新问题不能扩展当前轮上界。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(1)
    def handler(request):
        """参数：请求；返回：响应；异常：无。"""
        if request.url.path == "/classify" and len(repo.rows) == 1:
            repo.rows.append(dict(question_id="2", text="新问题", created_at=None))
        return response(request)
    service = make_service(tmp_path, repo, handler)
    report = await service.process()
    assert report["high_watermark"] == "1" and len(repo.saved) == 1
    assert (await service.process())["counts"]["saved"] == 1 and len(repo.saved) == 2
    await service.close()


async def test_restart_reconciles_commit_before_manifest_complete(tmp_path, monkeypatch):
    """提交后complete写盘失败不能重复首批或丢游标。参数：目录/替换器；返回：None；异常：断言失败。"""
    import app.services.topics as topics
    repo = MemoryRepository(201)
    first = make_service(tmp_path, repo, response)
    original, armed = topics.atomic_json, True
    def write(path, value):
        """参数：路径/原件；返回：None；异常：一次模拟写盘中断。"""
        nonlocal armed
        if armed and path.name.startswith("batch-") and value["status"] == "complete":
            armed = False
            raise OSError("确认写盘中断")
        original(path, value)
    monkeypatch.setattr(topics, "atomic_json", write)
    assert (await first.process())["status"] == "failed" and len(repo.saved) == 200
    await first.close()
    second = make_service(tmp_path, repo, response)
    result = await second.process()
    assert result["status"] == "complete" and result["counts"]["saved"] == 201
    assert len(result["batches"]) == 2
    await second.close()


async def test_partial_reclassification_reports_mixed_full_table_version(tmp_path):
    """未确定保留旧行，最新模型不能覆盖旧行版本声明。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(2)
    first = make_service(tmp_path, repo, response)
    await first.process()
    await first.close()
    def changed(request):
        """参数：请求；返回：新版本部分未确定；异常：无。"""
        result = response(request, version="v2")
        if request.url.path == "/classify":
            body = result.json()
            body["results"][0].update(labels=[], uncertain=True)
            return httpx.Response(200, json=body)
        return result
    second = make_service(tmp_path, repo, changed)
    old = repo.saved["1"].copy()
    report = await second.process(reclassify=True)
    assert report["counts"]["uncertain"] == 1 and repo.saved["1"] == old
    assert (await second.status())["model_version"] == "mixed"
    await second.close()


async def test_dependency_failure_is_persisted_and_not_empty_data(tmp_path):
    """服务离线不能返回空池成功，重启应能读到失败状态。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(0)
    def unavailable(request):
        """参数：请求；返回：503；异常：无。"""
        return httpx.Response(503, json={"detail": "未配置"})
    service = make_service(tmp_path, repo, unavailable)
    report = await service.process()
    assert report["status"] == "failed"
    assert (await service.status())["retry_after_seconds"] == 60
    await service.close()
    restarted = make_service(tmp_path, repo, unavailable)
    assert (await restarted.status())["status"] == "failed"
    await restarted.close()


async def test_pre_cancelled_trigger_still_releases_lease(tmp_path):
    """在第一次运行前外部取消任务仍须由close收回锁。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(1)
    service = make_service(tmp_path, repo, response)
    await service.trigger()
    for task in list(service._active): task.cancel()
    await service.close()
    assert not repo.busy


async def test_trigger_status_write_failure_releases_acquired_lock(tmp_path, monkeypatch):
    """后台启动前写状态失败也不能泄漏已取得锁。参数：目录/替换器；返回：None；异常：断言失败。"""
    import app.services.topics as topics
    repo = MemoryRepository(1)
    service = make_service(tmp_path, repo, response)
    def fail(path, value):
        """参数：路径/状态；返回：无；异常：模拟写盘失败。"""
        raise OSError("状态写盘失败")
    monkeypatch.setattr(topics, "atomic_json", fail)
    with pytest.raises(OSError):
        await service.trigger()
    assert not repo.busy
    await service.close()
