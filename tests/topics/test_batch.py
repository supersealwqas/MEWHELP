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


class HandoffRepository(MemoryRepository):
    """用独立连接令牌模拟旧租约失效与新执行者接手，产品服务和文件操作仍用真实实现。"""

    def __init__(self, count=1):
        """初始化共享池与当前所有者。参数：count为非负池行数；返回：None；异常：无。"""
        super().__init__(count)
        self.owner = None

    @asynccontextmanager
    async def execution(self):
        """取得本轮独立租约。参数：无；返回：带核锁方法的令牌租约；异常：共同锁忙或失锁。"""
        if self.owner is not None:
            raise ConcurrentKnowledgeWrite("共同锁忙")
        token = object()
        self.owner = token

        async def assert_held():
            """核对本轮连接身份。参数：无；返回：None；异常：LeaseLost表示连接失效或所有者改变。"""
            if self.owner is not token:
                raise LeaseLost("本轮连接已失锁")

        try:
            yield NS(assert_held=assert_held)
        finally:
            # 旧连接收尾不能释放接手者的新令牌。
            if self.owner is token:
                self.owner = None

    async def execution_busy(self):
        """读取共同锁占用。参数：无；返回：bool，是否有当前所有者；异常：无。"""
        return self.owner is not None


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


@pytest.mark.parametrize("stage", ["watermark_failure", "precheck_failure", "watermark_cancel", "precheck_cancel"])
async def test_new_scan_preparation_preserves_completed_archive_and_allows_upgrade(tmp_path, stage):
    """新轮准备故障不得污染旧完成事实。参数：tmp_path隔离目录、stage高水位或自动预检查的失败/取消；返回：None；异常：断言失败。"""
    class PreparationRepository(MemoryRepository):
        """在新轮准备边界注入一次外部依赖故障。"""
        armed = False

        async def fault(self, boundary):
            """仅在指定边界失败一次。参数：boundary为watermark/precheck；返回：None；异常：RuntimeError或CancelledError。"""
            if self.armed and stage.startswith(boundary):
                self.armed = False
                if stage.endswith("cancel"):
                    raise asyncio.CancelledError()
                raise RuntimeError("准备阶段暂时不可用")

        async def high_watermark(self):
            """读上界并注入准备故障。参数：无；返回：int最大池ID；异常：一次故障或取消。"""
            await self.fault("watermark")
            return await super().high_watermark()

        async def pending(self, limit, reclassify=False, after_id=0):
            """读候选并注入预检查故障。参数：limit为1至200、reclassify重跑开关、after_id非负游标；返回：list[dict]；异常：一次故障或取消。"""
            await self.fault("precheck")
            return await super().pending(limit, reclassify, after_id)

    repo = PreparationRepository(1)
    first = make_service(tmp_path, repo, response)
    complete = await first.process()
    await first.close()
    paths = [tmp_path / "scan.json", tmp_path / f"scan-{complete['batch_id']}.json"]
    originals = [path.read_bytes() for path in paths]
    repo.rows.append(dict(question_id="2", text="新问题", created_at="2020-01-01T00:00:00"))
    repo.armed = True
    upgraded = make_service(tmp_path, repo, lambda request: response(request, version="v2"))
    try:
        if stage.startswith("precheck"):
            async with repo.execution() as lease:
                if stage.endswith("cancel"):
                    with pytest.raises(asyncio.CancelledError):
                        await upgraded._process_locked(lease, automatic=True)
                else:
                    assert (await upgraded._process_locked(lease, automatic=True))["status"] == "failed"
        elif stage.endswith("cancel"):
            with pytest.raises(asyncio.CancelledError):
                await upgraded.process()
        else:
            assert (await upgraded.process())["status"] == "failed"
        assert [path.read_bytes() for path in paths] == originals
        retried = await upgraded.process()
        assert retried["status"] == "complete" and retried["model_version"] == "v2"
        assert retried["counts"]["saved"] == 1 and "2" in repo.saved
        assert paths[1].read_bytes() == originals[1]
    finally:
        await upgraded.close()


@pytest.mark.parametrize("length,prefix", [(2600, ""), (4000, ""), (4000, "联系人张三，"), (4000, "姓名:王，")])
async def test_legal_chat_length_at_front_does_not_block_later_page(tmp_path, length, prefix):
    """合法长原问脱敏后也须走完分页。参数：tmp_path隔离目录、length为2600/4000字符、prefix为虚构个人字段或空串；返回：None；异常：断言失败。"""
    from app.schemas.chat import ChatRequest
    from app.topics.pool import derive_pool
    raw = prefix + "问" * (length - len(prefix))
    assert ChatRequest(session_id="test-long", message=raw).message == raw
    record = NS(id=1, conversation_id=1, source="self_check", raw_question=raw, matched_review_id=None, created_at=None)
    repo = MemoryRepository(201)
    repo.rows[0].update(derive_pool([record], [], [], [])[0])

    def handler(request):
        """返回已截断的合法合同结果。参数：request受控HTTP请求；返回：httpx.Response；异常：无。"""
        value = response(request)
        if request.url.path == "/classify":
            body = value.json()
            texts = json.loads(request.content)["texts"]
            for result, text in zip(body["results"], texts):
                result["truncated"] = len(text) > 128
            return httpx.Response(200, json=body)
        return value

    service = make_service(tmp_path, repo, handler)
    try:
        report = await service.process()
        assert report["status"] == "complete" and report["counts"]["saved"] == 201
        assert report["cursor"] == "201" and len(report["batches"]) == 2
        first_batch = json.loads((tmp_path / f"batch-{report['batches'][0]['batch_id']}.json").read_text(encoding="utf-8"))
        assert first_batch["results"][0]["truncated"] is True
    finally:
        await service.close()


async def test_over_chat_limit_refuses_whole_batch_before_http(tmp_path):
    """超出业务长度仍不能静默截断或半批保存。参数：tmp_path隔离目录；返回：None；异常：断言失败。"""
    repo, calls = MemoryRepository(2), []
    repo.rows[0]["text"] = "问" * 4001

    def handler(request):
        """记录分类外发数量。参数：request受控请求；返回：httpx.Response；异常：无。"""
        if request.url.path == "/classify":
            calls.append(request)
        return response(request)

    service = make_service(tmp_path, repo, handler)
    try:
        assert (await service.process())["status"] == "failed"
        assert not repo.saved and not calls
    finally:
        await service.close()


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


@pytest.mark.parametrize("stage", ["verify", "publish_failure", "verify_ignores_cancel"])
async def test_close_waits_for_trigger_verification_and_failed_handoff_cleanup(tmp_path, monkeypatch, stage):
    """关闭须收回核锁或失败移交中的请求，不能在关闭后启动后台任务。参数：隔离目录/替换器/等待阶段；返回：None；异常：断言失败。"""
    verifying, verification_release = asyncio.Event(), asyncio.Event()
    cleaning, cleanup_release = asyncio.Event(), asyncio.Event()

    class PausedTriggerRepository(HandoffRepository):
        """只暂停外部核锁与释放，使用真实trigger和close验证完整移交生命周期。"""

        @asynccontextmanager
        async def execution(self):
            """取得独立令牌并在释放时等待确认。参数：无；返回：受控核锁租约；异常：失锁或取消。"""
            async with super().execution() as lease:
                async def assert_held():
                    """暂停首次核锁。参数：无；返回：None；异常：取消，指定场景模拟依赖吞掉取消。"""
                    if stage != "publish_failure":
                        verifying.set()
                        try:
                            await verification_release.wait()
                        except asyncio.CancelledError:
                            if stage != "verify_ignores_cancel":
                                raise
                    await lease.assert_held()

                try:
                    yield NS(assert_held=assert_held)
                finally:
                    cleaning.set()
                    await cleanup_release.wait()

    repo = PausedTriggerRepository()
    service = make_service(tmp_path, repo, response)
    if stage == "publish_failure":
        def fail_publish(**values):
            """在真实移交边界模拟写盘失败。参数：状态更新字段；返回：无；异常：OSError。"""
            raise OSError("后台移交前写盘失败")
        monkeypatch.setattr(service, "_set_state", fail_publish)
    trigger_task = asyncio.create_task(service.trigger())
    close_task, cleanup_wait = None, None
    try:
        await asyncio.wait_for((cleaning if stage == "publish_failure" else verifying).wait(), 3)
        close_task = asyncio.create_task(service.close())
        cleanup_wait = asyncio.create_task(cleaning.wait())
        await asyncio.wait({close_task, cleanup_wait}, timeout=3, return_when=asyncio.FIRST_COMPLETED)
        assert cleaning.is_set() and not close_task.done(), "关闭在请求租约清理前返回"
        # 释放还被明确阻塞时，关闭必须持续等待，HTTP客户端也不能提前关闭。
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(close_task), 0.1)
        assert await repo.execution_busy() and not service.client.is_closed
        cleanup_release.set()
        await asyncio.wait_for(close_task, 3)
        if stage == "verify_ignores_cancel":
            assert isinstance(trigger_task.exception(), RuntimeError)
        else:
            assert trigger_task.cancelled()
        assert not await repo.execution_busy()
        assert not service._active and not service._leases and service.client.is_closed
        assert not list(tmp_path.glob("*.json")), "关闭请求不得发布运行状态或生成后台产物"
    finally:
        verification_release.set()
        cleanup_release.set()
        if not trigger_task.done():
            trigger_task.cancel()
        await asyncio.gather(trigger_task, return_exceptions=True)
        if cleanup_wait is not None:
            cleanup_wait.cancel()
            await asyncio.gather(cleanup_wait, return_exceptions=True)
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await service.close()


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
    assert (await service.status())["status"] == "failed"
    await service.close()


async def test_shared_latest_failed_status_overrides_local_completed_state(tmp_path):
    """A旧完成不能掩盖同目录B最新失败。参数：目录；返回：None；异常：断言失败。"""
    repo = MemoryRepository(1)
    first = make_service(tmp_path, repo, response)
    assert (await first.process())["status"] == "complete"
    def fail_classify(request):
        """参数：请求；返回：健康成功/分类503；异常：无。"""
        return response(request) if request.url.path == "/health" else httpx.Response(503)
    second = make_service(tmp_path, repo, fail_classify)
    try:
        assert (await second.process(reclassify=True))["status"] == "failed"
        state = await first.status()
        assert state["status"] == state["last_batch"]["status"] == "failed"
        assert state["error"]["type"] == "HTTPStatusError"
    finally:
        await second.close()
        await first.close()


async def test_unheld_poller_does_not_overwrite_shared_owner_status(tmp_path):
    """竞争失败的poller不能把owner已写状态覆盖成running。参数：目录；返回：None；异常：断言失败。"""
    attempted = asyncio.Event()
    class Contended(MemoryRepository):
        """在锁获取边界标记一次poll。"""
        @asynccontextmanager
        async def execution(self):
            """参数：无；返回：租约；异常：共同锁忙。"""
            attempted.set()
            async with super().execution() as lease:
                yield lease
    repo = Contended(1)
    owner = make_service(tmp_path, repo, response)
    owner._set_state(status="failed", error={"type": "OwnerFailure"})
    repo.busy = True
    poller = make_service(tmp_path, repo, response)
    poller.start()
    try:
        await asyncio.wait_for(attempted.wait(), 3)
        assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["status"] == "failed"
        assert (await poller.status())["status"] == "running"
        repo.busy = False
        assert (await poller.status())["status"] == "failed"
    finally:
        await poller.close()
        await owner.close()


@pytest.mark.parametrize("zero_watermark", [False, True])
async def test_lease_lost_during_empty_read_never_completes_or_advances(tmp_path, zero_watermark):
    """空候选页及零上界末路都必须核锁。参数：目录/零上界开关；返回：None；异常：断言失败。"""
    class LostOnRead(MemoryRepository):
        """在末次只读操作中丢失租约的外部边界。"""
        async def high_watermark(self):
            """参数：无；返回：冻结上界；异常：无。"""
            if zero_watermark:
                self.lost = True
                return 0
            return 1
        async def pending(self, limit, reclassify=False, after_id=0):
            """参数：分页参数；返回：空页；异常：无。"""
            self.lost = True
            return []
    repo = LostOnRead(0)
    service = make_service(tmp_path, repo, response)
    try:
        result = await service.process()
        assert result["status"] == "failed" and result["cursor"] == "0"
        assert result["error"]["type"] == "LeaseLost"
        if zero_watermark:
            # 上界读取时已失锁，首轮共享扫描尚未获准创建。
            assert not (tmp_path / "scan.json").exists()
        else:
            shared = json.loads((tmp_path / "scan.json").read_text(encoding="utf-8"))
            assert shared["status"] == "running" and shared["cursor"] == "0" and shared["error"] is None
        evidence = list(tmp_path.glob("interrupted-*.json"))
        assert len(evidence) == 1
        local = json.loads(evidence[0].read_text(encoding="utf-8"))
        assert local["status"] == "failed"
        if zero_watermark:
            assert local["last_batch"] is None
        else:
            assert local["last_batch"]["cursor"] == "0"
        assert local["error"]["type"] == "LeaseLost"
    finally:
        await service.close()


async def test_new_local_prelease_failure_overrides_old_success_until_new_shared_attempt(tmp_path, monkeypatch):
    """新取锁失败可见，随后他实例更新又须胜过旧本地失败。参数：目录/替换器；返回：None；异常：断言失败。"""
    repo, attempted = MemoryRepository(1), asyncio.Event()
    first = make_service(tmp_path, repo, response)
    assert (await first.process())["status"] == "complete"
    original_execution = repo.execution
    @asynccontextmanager
    async def connection_failed():
        """参数：无；返回：无；异常：取租约前ConnectionError。"""
        attempted.set()
        raise ConnectionError("租约新连接不可用")
        yield
    monkeypatch.setattr(repo, "execution", connection_failed)
    first.start()
    second = None
    try:
        await asyncio.wait_for(attempted.wait(), 3)
        state = await first.status()
        assert state["status"] == "failed" and state["error"]["type"] == "ConnectionError"
        assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["status"] == "complete"
        monkeypatch.setattr(repo, "execution", original_execution)
        def fail_classify(request):
            """参数：请求；返回：健康成功/分类503；异常：无。"""
            return response(request) if request.url.path == "/health" else httpx.Response(503)
        second = make_service(tmp_path, repo, fail_classify)
        assert (await second.process(reclassify=True))["status"] == "failed"
        latest = await first.status()
        assert latest["status"] == "failed" and latest["error"]["type"] == "HTTPStatusError"
    finally:
        if second is not None: await second.close()
        await first.close()


@pytest.mark.parametrize("entry", ["process", "trigger"])
async def test_manual_prelease_failure_is_visible_without_shared_write(tmp_path, monkeypatch, entry):
    """手动入口新连接失败也不能留下旧完成状态。参数：目录/替换器/入口；返回：None；异常：断言失败。"""
    repo = MemoryRepository(1)
    service = make_service(tmp_path, repo, response)
    await service.process()
    @asynccontextmanager
    async def unavailable():
        """参数：无；返回：无；异常：取锁前连接失败。"""
        raise ConnectionError("租约连接失败")
        yield
    monkeypatch.setattr(repo, "execution", unavailable)
    try:
        with pytest.raises(ConnectionError):
            await getattr(service, entry)()
        state = await service.status()
        assert state["status"] == "failed" and state["error"]["type"] == "ConnectionError"
        assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["status"] == "complete"
    finally:
        await service.close()


@pytest.mark.parametrize("outcome", ["response", "disconnect", "cancelled"])
async def test_lost_owner_cannot_overwrite_successor_scan_or_status(tmp_path, outcome):
    """旧HTTP返回、失败或取消不得覆盖接手者已完成的新轮。参数：隔离目录/结束方式；返回：None；异常：断言失败。"""
    repo, entered, release = HandoffRepository(), asyncio.Event(), asyncio.Event()

    async def delayed(request):
        """暂停旧分类请求。参数：HTTP请求；返回：合同响应；异常：模拟断线或外部取消。"""
        if request.url.path == "/classify":
            entered.set()
            await release.wait()
            if outcome == "disconnect":
                raise httpx.ConnectError("旧请求断线", request=request)
        return response(request)

    old, successor = make_service(tmp_path, repo, delayed), make_service(tmp_path, repo, response)
    task = asyncio.create_task(old.process())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        old_scan = json.loads((tmp_path / "scan.json").read_text(encoding="utf-8"))
        repo.owner = None  # 外部MySQL连接失效会释放该会话的命名锁。
        resumed = await successor.process()
        current = await successor.process(reclassify=True)
        assert resumed["batch_id"] == old_scan["batch_id"] and resumed["status"] == "complete"
        assert current["batch_id"] != old_scan["batch_id"]
        assert current["status"] == "complete" and current["cursor"] == "1"
        before = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
        if outcome == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            expected_status, expected_error = "cancelled", "CancelledError"
        else:
            release.set()
            failure = await asyncio.wait_for(task, 3)
            expected_status = "failed"
            expected_error = "ConnectError" if outcome == "disconnect" else "LeaseLost"
            assert failure["status"] == expected_status and failure["error"]["type"] == expected_error
        # 两个扫描原件、当前指针、完成批次和共享状态均保持接手者的字节原件。
        for path, contents in before.items():
            assert path.read_bytes() == contents, path.name
        evidence = list(tmp_path.glob("interrupted-*.json"))
        assert len(evidence) == 1
        local = json.loads(evidence[0].read_text(encoding="utf-8"))
        assert local["status"] == expected_status and local["error"]["type"] == expected_error
        assert local["last_batch"]["batch_id"] == old_scan["batch_id"]
        assert local["last_batch"]["cursor"] == "0"
        assert set(tmp_path.glob("*.json")) == set(before) | set(evidence)
        latest = await old.status()
        assert latest["status"] == "complete" and latest["last_batch"]["batch_id"] == current["batch_id"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await old.close()
        await successor.close()


@pytest.mark.parametrize("boundary", ["health", "watermark", "save"])
async def test_lease_handoff_during_await_preserves_successor_artifacts(tmp_path, boundary):
    """外部等待返回后须重新核锁，旧执行者不能续写共享原件。参数：隔离目录/等待边界；返回：None；异常：断言失败。"""
    entered, release, paused = asyncio.Event(), asyncio.Event(), False

    async def wait_once(point):
        """仅暂停旧执行者一次。参数：point为健康、上界或提交边界；返回：None；异常：外部取消。"""
        nonlocal paused
        if point == boundary and not paused:
            paused = True
            entered.set()
            await release.wait()

    class PausedRepository(HandoffRepository):
        """在真实服务依赖边界暂停旧连接，让接手者完成恢复和新重跑。"""

        async def high_watermark(self):
            """读取上界时允许失锁。参数：无；返回：最大问题ID；异常：取消。"""
            await wait_once("watermark")
            return await super().high_watermark()

        async def save(self, rows, *, lease):
            """模拟提交完成但调用尚未返回。参数：结果行/本轮租约；返回：保存计数；异常：失锁或取消。"""
            result = await super().save(rows, lease=lease)
            await wait_once("save")
            return result

    async def handler(request):
        """暂停健康依赖响应。参数：HTTP请求；返回：合同响应；异常：外部取消。"""
        if request.url.path == "/health":
            await wait_once("health")
        return response(request)

    repo = PausedRepository()
    old, successor = make_service(tmp_path, repo, handler), make_service(tmp_path, repo, response)
    task = asyncio.create_task(old.process())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        repo.owner = None
        assert (await successor.process())["status"] == "complete"
        current = await successor.process(reclassify=True)
        before = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
        release.set()
        failure = await asyncio.wait_for(task, 3)
        assert failure["status"] == "failed" and failure["error"]["type"] == "LeaseLost"
        for path, contents in before.items():
            assert path.read_bytes() == contents, path.name
        evidence = list(tmp_path.glob("interrupted-*.json"))
        assert len(evidence) == 1 and set(tmp_path.glob("*.json")) == set(before) | set(evidence)
        latest = await old.status()
        assert latest["status"] == "complete" and latest["last_batch"]["batch_id"] == current["batch_id"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await old.close()
        await successor.close()


async def test_trigger_lost_before_worker_start_preserves_successor_state(tmp_path):
    """后台任务首次调度前失锁不能覆盖新执行者状态。参数：隔离目录；返回：None；异常：断言失败。"""
    repo = HandoffRepository()
    old, successor = make_service(tmp_path, repo, response), make_service(tmp_path, repo, response)
    try:
        await old.trigger()
        repo.owner = None
        assert (await successor.process())["status"] == "complete"
        current = await successor.process(reclassify=True)
        before = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
        await asyncio.wait_for(asyncio.gather(*old._active), 3)
        for path, contents in before.items():
            assert path.read_bytes() == contents, path.name
        evidence = list(tmp_path.glob("interrupted-*.json"))
        assert len(evidence) == 1
        local = json.loads(evidence[0].read_text(encoding="utf-8"))
        assert local["last_batch"] is None and local["error"]["type"] == "LeaseLost"
        latest = await old.status()
        assert latest["status"] == "complete" and latest["last_batch"]["batch_id"] == current["batch_id"]
    finally:
        await old.close()
        await successor.close()
