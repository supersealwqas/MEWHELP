"""主题分类旁路：共同租约、持久扫描轮次、整批HTTP核验与版本事实对账。"""

import asyncio
from contextvars import Context
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4

import httpx

from app.evaluation.runner import atomic_json
from app.knowledge.errors import ConcurrentKnowledgeWrite
from app.knowledge.lease import finish_cleanup
from app.topics.taxonomy import load_taxonomy, taxonomy_digest, validate_labels


class TopicBatchService:
    """拥有本实例扫描/HTTP任务，MySQL命名锁负责跨实例仲裁；不接实时图或模型。"""

    def __init__(self, repository, config, client=None):
        """参数：主题仓储/Settings/可选HTTP客户端；返回：None；异常：目录创建错误。"""
        self.repository, self.config = repository, config
        self.root = Path(config.topic_reports_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.taxonomy = load_taxonomy()
        self.digest = taxonomy_digest(self.taxonomy)
        # 服务推理默认等待30秒；HTTP多留5秒以接收服务明确的504。禁止环境代理及外网重定向。
        self.client = client or httpx.AsyncClient(base_url=config.topic_service_url, timeout=35,
            follow_redirects=False, trust_env=False)
        self._active, self._background, self._closing = set(), None, False
        self._leases = {}
        self._local_attempt_error = None
        self._state = dict(status="idle", last_batch=None, model_version="unknown", error=None,
                           retry_after_seconds=config.topic_poll_seconds)

    def _read(self, path):
        """读取原件并核对目标库。参数：JSON路径；返回：字典或None；异常：JSON/目标不符/OSError。"""
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("database") != self.config.mysql_database:
            raise ValueError("主题原件数据库身份不符")
        return value

    def _write_scan(self, scan):
        """持久记录当前扫描状态。参数：扫描原件；返回：None；异常：写盘错误。"""
        atomic_json(self.root / "scan.json", scan)
        atomic_json(self.root / f"scan-{scan['batch_id']}.json", scan)

    def _set_state(self, **values):
        """持久保存依赖失败等状态，重启不得遗忘错误。参数：状态更新字段；返回：None；异常：写盘错误。"""
        self._state.update(values)
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        self._local_attempt_error = None
        atomic_json(self.root / "status.json", {**self._state, "database": self.config.mysql_database})

    def _record_local_attempt_error(self, error):
        """记录本实例新启动或清理失败，不覆盖其他持锁实例的共享记录。参数：依赖异常；返回：None；异常：无。"""
        self._local_attempt_error = dict(type=type(error).__name__, message="主题执行依赖不可用，稍后重试",
            occurred_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"))
        self._state.update(status="failed", error=self._local_attempt_error)

    def _new_scan(self, high_watermark, reclassify):
        """创建独立扫描轮次。参数：冻结上界/重跑开关；返回：扫描字典；异常：无。"""
        return dict(database=self.config.mysql_database, batch_id=uuid4().hex, status="running",
            reclassify=reclassify, high_watermark=str(high_watermark), cursor="0", batches=[],
            counts=dict(checked=0, saved=0, uncertain=0), model_version=None, taxonomy_digest=self.digest,
            pending_batch=None, error=None)

    async def _identity(self, response):
        """核对响应模型和权威主题表。参数：HTTP响应；返回：JSON字典；异常：HTTP/JSON/身份错误。"""
        response.raise_for_status()
        body = response.json()
        if (not isinstance(body, dict) or not isinstance(body.get("model_version"), str)
                or not body["model_version"].strip() or body.get("taxonomy_digest") != self.digest):
            raise ValueError("主题服务模型身份或主题摘要非法")
        return body

    async def _classify(self, rows, model_version):
        """分块HTTP后核验所有结果，尚不写数据库。参数：最多200行/固定版本；返回：逐ID核验结果；异常：HTTP/合同错误/取消。"""
        if any(not isinstance(row.get("text"), str) or not row["text"].strip()
               or len(row["text"]) > 2000 for row in rows):
            raise ValueError("池文本为空或超过2000字符，需复核原件")
        output = []
        for offset in range(0, len(rows), 64):
            block = rows[offset:offset + 64]
            body = await self._identity(await self.client.post(self.config.topic_service_url + "/classify",
                json={"texts": [row["text"] for row in block]}, timeout=35))
            if body["model_version"] != model_version:
                raise ValueError("同一扫描模型版本改变，整批拒绝")
            results = body.get("results")
            if not isinstance(results, list) or len(results) != len(block):
                raise ValueError("主题服务结果数量不符")
            for row, result in zip(block, results, strict=True):
                if not isinstance(result, dict):
                    raise ValueError("主题结果结构非法")
                probabilities = result.get("probabilities")
                if (not isinstance(probabilities, list) or len(probabilities) != 17 or any(
                    type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)):
                    raise ValueError("主题概率必须为权威顺序17维有限数值")
                uncertain, truncated, labels = result.get("uncertain"), result.get("truncated"), result.get("labels")
                if type(uncertain) is not bool or type(truncated) is not bool or not isinstance(labels, list):
                    raise ValueError("主题结果标记或labels非法")
                if uncertain != (labels == []):
                    raise ValueError("未确定标记与空标签不一致")
                if labels:
                    labels = validate_labels(labels, self.taxonomy)
                output.append(dict(question_id=row["question_id"], labels=labels, probabilities=probabilities,
                                   uncertain=uncertain, truncated=truncated))
        return output

    def _complete_batch(self, scan, batch):
        """仅在数据库已确认后推进游标。参数：扫描/已提交批次；返回：None；异常：写盘错误。"""
        batch["status"] = "complete"
        atomic_json(self.root / f"batch-{batch['batch_id']}.json", batch)
        # 相同批次恢复只能累计一次。
        if not any(row["batch_id"] == batch["batch_id"] for row in scan["batches"]):
            scan["batches"].append(dict(batch_id=batch["batch_id"], status="complete", counts=batch["counts"],
                model_version=batch["model_version"], uncertain_ids=batch["uncertain_ids"], cursor=batch["cursor"]))
            for key in scan["counts"]:
                scan["counts"][key] += batch["counts"][key]
        scan["cursor"], scan["pending_batch"] = batch["cursor"], None
        self._write_scan(scan)

    async def _recover(self, scan, lease):
        """恢复提交与文件确认间断点，逐行对账。参数：扫描/持锁lease；返回：None；异常：部分冲突或原件缺失。"""
        if not scan.get("pending_batch"):
            return
        batch = self._read(self.root / f"batch-{scan['pending_batch']}.json")
        if not batch or batch["scan_id"] != scan["batch_id"] or batch["taxonomy_digest"] != self.digest:
            raise ValueError("待确认批次原件缺失或身份不符")
        await lease.assert_held()
        actual = {r["question_id"]: r for r in await self.repository.classified_rows()}
        await lease.assert_held()
        matches = [actual.get(row["question_id"]) == row for row in batch["saved_rows"]]
        if all(matches):
            self._complete_batch(scan, batch)
        elif any(matches):
            raise ValueError("提交批次只部分匹配，不能推进游标，需复核")
        else:
            # 原事务未提交：保留失败原件，同游标重新调用；不以文件pending假装完成。
            batch["status"] = "unconfirmed"
            atomic_json(self.root / f"batch-{batch['batch_id']}.json", batch)
            scan["pending_batch"] = None
            self._write_scan(scan)

    async def _process_locked(self, lease, *, reclassify=False, automatic=False):
        """持共同锁执行完整扫描，高水位后新行留下一轮。参数：lease/重跑/自动最小批限制；返回：扫描报告；异常：取消传播，其余写真实失败状态。"""
        scan = None
        current_ids = []
        self._set_state(status="running", error=None)
        try:
            if self._closing:
                raise RuntimeError("主题服务已关闭")
            # 即使零候选也先确认依赖，不能把服务不可用解释成空数据。
            health = await self._identity(await self.client.get(self.config.topic_service_url + "/health", timeout=35))
            if health.get("status") != "ready":
                raise ValueError("主题服务未就绪")
            scan = self._read(self.root / "scan.json")
            if scan and scan["status"] != "complete":
                if scan["reclassify"] != reclassify or scan["taxonomy_digest"] != self.digest:
                    raise ValueError("尚有未完成扫描，须沿用原重跑方式和主题表恢复")
                await self._recover(scan, lease)
            else:
                high = await self.repository.high_watermark()
                if automatic:
                    pending = await self.repository.pending(self.config.topic_batch_min_size)
                    await lease.assert_held()
                    known_times = [datetime.fromisoformat(r["created_at"]) for r in pending if r.get("created_at")]
                    now = datetime.now(timezone.utc)
                    old = any((now - (time.replace(tzinfo=timezone.utc) if time.tzinfo is None else time)).total_seconds()
                              >= self.config.topic_max_wait_seconds for time in known_times)
                    if len(pending) < self.config.topic_batch_min_size and not old:
                        self._set_state(status="waiting", error=None)
                        return dict(status="waiting", batch_id=None, counts=dict(checked=0, saved=0, uncertain=0),
                                    model_version="unknown", error=None)
                scan = self._new_scan(high, reclassify)
                self._write_scan(scan)
            if scan["model_version"] not in {None, health["model_version"]}:
                raise ValueError("恢复扫描时模型版本改变，需复核原轮")
            scan.update(model_version=health["model_version"], status="running", error=None)
            self._write_scan(scan)
            while int(scan["cursor"]) < int(scan["high_watermark"]):
                await lease.assert_held()
                rows = await self.repository.pending(self.config.topic_batch_size, reclassify=reclassify,
                                                     after_id=int(scan["cursor"]))
                # 空页也可能在数据库读取期间失锁，不能跳过save就绕过所有权保护。
                await lease.assert_held()
                rows = [row for row in rows if int(row["question_id"]) <= int(scan["high_watermark"])]
                if not rows:
                    break
                current_ids = [row["question_id"] for row in rows]
                results = await self._classify(rows, scan["model_version"])
                await lease.assert_held()
                stamp = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat()
                saved = [dict(question_id=row["question_id"], labels=row["labels"], classified_at=stamp)
                         for row in results if not row["uncertain"]]
                batch = dict(database=self.config.mysql_database, batch_id=uuid4().hex, scan_id=scan["batch_id"],
                    status="pending", high_watermark=scan["high_watermark"], after_id=scan["cursor"],
                    cursor=rows[-1]["question_id"], input_ids=[r["question_id"] for r in rows], results=results,
                    saved_rows=saved, uncertain_ids=[r["question_id"] for r in results if r["uncertain"]],
                    counts=dict(checked=len(rows), saved=len(saved), uncertain=len(rows) - len(saved)),
                    model_version=scan["model_version"], taxonomy_digest=self.digest)
                # 原件pending先落盘；数据库提交后complete。网络期间不持行事务。
                atomic_json(self.root / f"batch-{batch['batch_id']}.json", batch)
                scan["pending_batch"] = batch["batch_id"]
                self._write_scan(scan)
                await self.repository.save(saved, lease=lease)
                self._complete_batch(scan, batch)
                current_ids = []
            # 零高水位不进入循环；最终完成和高水位游标更新仍必须受原租约保护。
            await lease.assert_held()
            scan.update(status="complete", cursor=scan["high_watermark"], error=None)
            self._write_scan(scan)
            self._set_state(status="complete", last_batch=scan, error=None)
            return scan
        except asyncio.CancelledError:
            if scan:
                scan.update(status="cancelled", error=dict(type="CancelledError", message="主题批次已取消，未确认游标不推进"))
                self._write_scan(scan)
            self._set_state(status="cancelled", last_batch=scan, error=dict(type="CancelledError"))
            raise
        except Exception as error:
            # 错误只公开类型和稳定摘要，不暴露问题文本、凭据或驱动内部地址。
            summary = dict(type=type(error).__name__, message="主题依赖或批次核验失败，保留原件与游标供重试",
                           question_ids=current_ids)
            if scan:
                scan.update(status="failed", error=summary)
                self._write_scan(scan)
            self._set_state(status="failed", last_batch=scan, error=summary)
            return scan or dict(batch_id=None, status="failed", counts=dict(checked=0, saved=0, uncertain=0),
                                model_version="unknown", error=summary)

    async def process(self, *, reclassify=False):
        """手动或测试执行整轮，同数据库锁先于任何池读取。参数：重跑全部分页；返回：扫描报告；异常：占用/租约获取/取消。"""
        task = asyncio.current_task()
        self._active.add(task)
        try:
            async with self.repository.execution() as lease:
                return await self._process_locked(lease, reclassify=reclassify)
        except ConcurrentKnowledgeWrite:
            self._local_attempt_error = None
            raise
        except Exception as error:
            self._record_local_attempt_error(error)
            raise
        finally:
            # 获取连接和释放租约期间也属于本服务；停机须一直等待到收尾结束。
            self._active.discard(task)

    async def trigger(self, *, reclassify=False):
        """先真实取得共同锁再后台登记，消除先查busy后启动竞态。参数：重跑开关；返回：202接受状态；异常：占用/依赖/关闭。"""
        if self._closing:
            raise RuntimeError("主题服务已关闭")
        manager = self.repository.execution()
        request_task = asyncio.current_task()
        self._active.add(request_task)
        try:
            lease = await manager.__aenter__()
        except ConcurrentKnowledgeWrite:
            self._local_attempt_error = None
            raise
        except Exception as error:
            self._record_local_attempt_error(error)
            raise
        finally:
            self._active.discard(request_task)
        async def run():
            """参数：闭包持锁资源；返回：None；异常：取消传播，租约清理保证完成。"""
            try:
                await self._process_locked(lease, reclassify=reclassify)
            finally:
                try:
                    await finish_cleanup(manager.__aexit__(None, None, None))
                finally:
                    self._leases.pop(asyncio.current_task(), None)
                    self._active.discard(asyncio.current_task())
        try:
            self._set_state(status="running", error=None)
            task = asyncio.create_task(run(), context=Context())
            self._active.add(task)
            self._leases[task] = manager
        except BaseException as error:
            if isinstance(error, Exception):
                self._record_local_attempt_error(error)
            await finish_cleanup(manager.__aexit__(None, None, None))
            raise
        return dict(status="running", message="主题批次已在后台启动")

    async def _model_version(self):
        """从当前行与完整批次原件证明全表版本。参数：无；返回：版本/unknown/mixed；异常：数据库或原件错误。"""
        actual = await self.repository.classified_rows()
        if not actual:
            return "unknown"
        evidence = {}
        for path in self.root.glob("batch-*.json"):
            batch = self._read(path)
            if batch["status"] != "complete" or batch.get("taxonomy_digest") != self.digest:
                continue
            for row in batch["saved_rows"]:
                key = (row["question_id"], tuple(row["labels"]), row["classified_at"])
                evidence.setdefault(key, set()).add(batch["model_version"])
        versions = set()
        for row in actual:
            key = (row["question_id"], tuple(row["labels"]), row["classified_at"])
            matches = evidence.get(key, set())
            if not matches:
                return "unknown"
            versions.update(matches)
        return next(iter(versions)) if len(versions) == 1 else "mixed"

    async def status(self):
        """刷新跨实例占用、持久报告和当前分类版本事实。参数：无；返回：状态字典；异常：无，依赖错误明确返回failed。"""
        try:
            state = self._state.copy()
            stored_state = self._read(self.root / "status.json")
            # 共享持久状态是最后持锁执行者的事实，不受本实例曾完成/失败的旧缓存限制。
            if stored_state:
                state.update({key: value for key, value in stored_state.items() if key != "database"})
            persisted = self._read(self.root / "scan.json")
            busy = await self.repository.execution_busy()
            state["model_version"] = await self._model_version()
            if persisted:
                state["last_batch"] = persisted
                if not stored_state:
                    state["status"] = persisted["status"]
            # 本地新取锁失败必须可见，但更晚的共享执行结果仍可覆盖它。旧原件无时间视为历史。
            local_error = self._local_attempt_error
            if (not busy and local_error and local_error["occurred_at"] >=
                    (stored_state or {}).get("updated_at", "")):
                state.update(status="failed", error=local_error, updated_at=local_error["occurred_at"])
            if busy:
                state["status"] = "running"
            elif state["status"] == "running":
                # 本地活动任务也可能正在取连接或已经失锁，只有实际共同锁可证明运行中。
                state["status"] = "interrupted"
            return state
        except Exception as error:
            return dict(status="failed", last_batch=None, model_version="unknown",
                error=dict(type=type(error).__name__, message="主题数据库或原件暂不可用"),
                retry_after_seconds=self.config.topic_poll_seconds)

    async def distribution(self, scope="pending"):
        """在全池统计上补充实际状态。参数：pending/all；返回：完整API分布；异常：仓储读取错误。"""
        return {**await self.repository.distribution(scope), **await self.status()}

    def start(self):
        """登记独立扫描，禁止继承请求上下文。参数：无；返回：None；异常：无事件循环。"""
        if not self._closing and (self._background is None or self._background.done()):
            self._background = asyncio.create_task(self._run(), context=Context())

    async def _run(self):
        """按间隔扫描/重试，依赖失败不冒充空池。参数：无；返回：None；异常：取消传播。"""
        while not self._closing:
            try:
                async with self.repository.execution() as lease:
                    scan = self._read(self.root / "scan.json")
                    reclassify = scan["reclassify"] if scan and scan["status"] != "complete" else False
                    await self._process_locked(lease, reclassify=reclassify, automatic=True)
            except ConcurrentKnowledgeWrite:
                # 未持锁的poller不能覆盖owner刚写的完成/失败状态；API另查实际锁。
                self._state.update(status="running", error=None)
                # 能取得明确busy响应说明本地连接已恢复，旧启动错误不再参与时间比较。
                self._local_attempt_error = None
            except Exception as error:
                self._record_local_attempt_error(error)
            await asyncio.sleep(self.config.topic_poll_seconds)

    async def close(self):
        """取消并收回本实例全部任务与HTTP客户端。参数：无；返回：None；异常：清理超时或取消传播。"""
        self._closing = True
        tasks = self._active | ({self._background} if self._background is not None else set())
        tasks.discard(asyncio.current_task())
        async def settle():
            """参数：闭包任务/HTTP资源；返回：None；异常：HTTP关闭错误。"""
            try:
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                # 若任务在第一次调度前已被取消，其finally没有机会执行，仍须释放取得的锁。
                for task in tasks:
                    manager = self._leases.pop(task, None)
                    if manager is not None:
                        await manager.__aexit__(None, None, None)
                    self._active.discard(task)
            finally:
                await self.client.aclose()
        await finish_cleanup(settle(), timeout_seconds=20)
