"""BgeM3Encoder 的离线 TDD：超限拒绝、GPU 串行、取消排空与向量形状。"""

from __future__ import annotations

import asyncio
import math
import threading
from typing import Any

import anyio
import pytest
from langchain_core.messages import AIMessage

from app.knowledge import embedding
from app.knowledge.embedding import BgeM3Encoder, EmbeddingUnavailable
from app.schemas.chat import ChatRequest
from app.schemas.tools import ToolContext
from tests.chat_fakes import make_chat_service
from tests.knowledge.fakes import make_settings


class _FakeTokenizer:
    """返回固定 token 数量的测试 tokenizer。"""

    def __init__(self, token_count: int) -> None:
        """保存测试 token 数。

        参数:
            token_count: 每次调用返回的 input_ids 长度。
        返回值:
            None
        异常:
            无
        """
        self._token_count = token_count
        self.closed = False

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        """返回与输入无关的固定 token 序列。

        参数:
            text: 待计数文本；本替身不读取内容。
            add_special_tokens: 是否包含特殊 token；由编码器固定传 True。
        返回值:
            dict[str, list[int]]: 含指定长度 input_ids 的映射。
        异常:
            无
        """
        del text, add_special_tokens
        return {"input_ids": list(range(self._token_count))}

    def close(self) -> None:
        """记录 tokenizer 资源释放。参数：无；返回：None；异常：无。"""
        self.closed = True


class _FixedBackend:
    """记录调用参数的固定向量测试后端。"""

    def __init__(self) -> None:
        """初始化记录容器。

        参数:
            无
        返回值:
            None
        异常:
            无
        """
        self.calls: list[list[str]] = []
        self.kwargs: dict[str, Any] = {}
        self.closed = False

    def encode(self, sentences: list[str], **kwargs: Any) -> dict[str, list[list[float]]]:
        """返回固定 1024 维单位向量。

        参数:
            sentences: 待编码文本。
            **kwargs: 编码参数，记录以供断言。
        返回值:
            dict[str, list[list[float]]]: 含 dense_vecs 的结果。
        异常:
            无
        """
        self.calls.append(list(sentences))
        self.kwargs = dict(kwargs)
        vector = [1.0] + [0.0] * 1023
        return {"dense_vecs": [list(vector) for _ in sentences]}

    def close(self) -> None:
        """标记测试后端已关闭。

        参数:
            无
        返回值:
            None
        异常:
            无
        """
        self.closed = True


class _BlockingBackend(_FixedBackend):
    """每次 encode 等待显式放行，用于验证串行与取消排空。"""

    def __init__(self) -> None:
        """初始化阻塞事件与并发计数。

        参数:
            无
        返回值:
            None
        异常:
            无
        """
        super().__init__()
        self.started = threading.Event()
        self.second_started = threading.Event()
        self.release = threading.Event()
        self.guard = threading.Lock()
        self.attempts = 0
        self.inflight = 0
        self.max_inflight = 0
        self.completed: list[list[str]] = []
        self.error: Exception | None = None
        self.closed_while_running = False

    def encode(self, sentences: list[str], **kwargs: Any) -> dict[str, list[list[float]]]:
        """阻塞到测试放行，并记录并发峰值。

        参数:
            sentences: 待编码文本。
            **kwargs: 透传给父类记录。
        返回值:
            dict[str, list[list[float]]]: 固定 dense_vecs。
        异常:
            RuntimeError: 测试指定首轮推理故障时抛出。
            AssertionError: 测试没有按时释放工作线程时抛出。
        """
        with self.guard:
            self.attempts += 1
            ordinal = self.attempts
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        (self.started if ordinal == 1 else self.second_started).set()
        try:
            assert self.release.wait(timeout=5), "测试没有按时释放工作线程"
            if ordinal == 1 and self.error is not None:
                raise self.error
            return super().encode(sentences, **kwargs)
        finally:
            with self.guard:
                self.inflight -= 1
                self.completed.append(list(sentences))

    def close(self) -> None:
        """记录关闭是否抢在推理结束前发生。参数：无；返回：None；异常：无。"""
        self.closed_while_running = self.inflight > 0
        super().close()


def _make_encoder(monkeypatch: pytest.MonkeyPatch, backend: _FixedBackend) -> BgeM3Encoder:
    """仅替换本地权重和 tokenizer 的外部边界。

    参数：monkeypatch为pytest替换器；backend为同步推理替身。
    返回：真实BgeM3Encoder；异常：配置不合法时抛出ValidationError。
    """
    tokenizer = _FakeTokenizer(2)

    def load_tokenizer(path: str) -> _FakeTokenizer:
        """返回测试 tokenizer。参数：本地路径；返回：tokenizer；异常：无。"""
        del path
        return tokenizer

    def load_model(path: str) -> _FixedBackend:
        """返回内存推理边界。参数：本地路径；返回：后端；异常：无。"""
        del path
        return backend

    monkeypatch.setattr(embedding, "_load_tokenizer", load_tokenizer)
    monkeypatch.setattr(embedding, "_load_model", load_model)
    return BgeM3Encoder(make_settings())


async def test_encoder_rejects_oversize_before_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超过 2048 token 的输入在调用模型前失败。

    参数:
        monkeypatch: pytest 替换器，用于注入 tokenizer 并禁止模型加载。
    返回值:
        None
    异常:
        无
    """
    monkeypatch.setattr(embedding, "_load_tokenizer", lambda path: _FakeTokenizer(2049))

    def fail_load(path: str) -> None:
        """禁止超限路径加载模型。

        参数:
            path: 模型路径；本函数不使用。
        返回值:
            None
        异常:
            AssertionError: 只要被调用即说明超限检查顺序错误。
        """
        del path
        raise AssertionError("超限输入不应加载模型")

    monkeypatch.setattr(embedding, "_load_model", fail_load)
    encoder = BgeM3Encoder(make_settings())
    with pytest.raises(ValueError, match="2048|硬限"):
        await encoder.encode(["超长文本"])
    await encoder.aclose()


@pytest.mark.parametrize("cancel_count,worker_error", [(1, False), (3, False), (3, True)])
async def test_encoder_native_cancel_keeps_slot_until_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
    cancel_count: int,
    worker_error: bool,
) -> None:
    """原生及重复取消不能提前结束编码或释放仍在运行的推理槽。

    参数:
        monkeypatch: pytest替换器；cancel_count: 原生取消次数。
        worker_error: 首轮线程是否在放行后失败，失败不能覆盖原取消。
    返回值:
        None
    异常:
        无
    """
    backend = _BlockingBackend()
    if worker_error:
        backend.error = RuntimeError("首轮推理失败")
    encoder = _make_encoder(monkeypatch, backend)
    first = asyncio.create_task(encoder.encode(["第一句"]))
    second = None
    try:
        assert await asyncio.to_thread(backend.started.wait, 1)
        for _ in range(cancel_count):
            first.cancel()
            await asyncio.wait({first}, timeout=0.02)
            assert not first.done(), "底层线程未结束，取消不能先释放编码锁"
        second = asyncio.create_task(encoder.encode(["第二句"]))
        assert not await asyncio.to_thread(backend.second_started.wait, 0.1)
        assert backend.inflight == 1 and backend.max_inflight == 1
    finally:
        backend.release.set()
        results = await asyncio.gather(*[task for task in (first, second) if task], return_exceptions=True)
        await encoder.aclose()
    assert isinstance(results[0], asyncio.CancelledError)
    assert isinstance(results[1], list)
    assert backend.max_inflight == 1
    assert backend.completed == [["第一句"], ["第二句"]]
    assert backend.closed and not backend.closed_while_running


async def test_encoder_returns_1024_finite_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编码器返回与输入等长的 1024 维有限向量，并使用约定参数。

    参数:
        monkeypatch: pytest 替换器，用于注入固定后端。
    返回值:
        None
    异常:
        无
    """
    backend = _FixedBackend()
    monkeypatch.setattr(embedding, "_load_tokenizer", lambda path: _FakeTokenizer(3))
    monkeypatch.setattr(embedding, "_load_model", lambda path: backend)
    encoder = BgeM3Encoder(make_settings())
    vectors = await encoder.encode(["甲", "乙"])

    assert len(vectors) == 2
    assert all(len(vector) == 1024 for vector in vectors)
    assert all(math.isfinite(value) for vector in vectors for value in vector)
    assert backend.kwargs["batch_size"] == 1
    assert backend.kwargs["max_length"] == 2048
    assert backend.kwargs["return_dense"] is True
    assert backend.kwargs["return_sparse"] is False
    assert backend.kwargs["return_colbert_vecs"] is False
    await encoder.aclose()
    assert backend.closed is True


async def test_encoder_cancelled_load_retains_model_for_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """加载取消后必须保留已完成的模型，下一次编码不能重复加载。

    参数：monkeypatch为本地权重与tokenizer替换器。
    返回：None；异常：AssertionError表示加载丢失、取消过早结束或重复加载。
    """
    backend = _FixedBackend()
    encoder = _make_encoder(monkeypatch, backend)
    started, release = threading.Event(), threading.Event()
    loads = 0

    def load_model(path: str) -> _FixedBackend:
        """阻塞模型加载边界。参数：本地路径；返回：模型；异常：超时AssertionError。"""
        nonlocal loads
        del path
        loads += 1
        started.set()
        assert release.wait(5), "测试没有按时释放加载线程"
        return backend

    monkeypatch.setattr(embedding, "_load_model", load_model)
    first = asyncio.create_task(encoder.encode(["首次请求"]))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        await asyncio.wait({first}, timeout=0.05)
        assert not first.done(), "加载线程未完成时取消不能释放模型所有权"
        second = asyncio.create_task(encoder.encode(["后续请求"]))
        await asyncio.wait({second}, timeout=0.05)
        assert loads == 1 and not second.done()
    finally:
        release.set()
        results = await asyncio.gather(*[task for task in (first, second) if task], return_exceptions=True)
        await encoder.aclose()
    assert isinstance(results[0], asyncio.CancelledError)
    assert isinstance(results[1], list)
    assert loads == 1 and backend.calls == [["后续请求"]] and backend.closed


async def test_encoder_anyio_cancel_drains_and_keeps_next_request_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnyIO取消域也须排空线程，不能留后台推理并释放编码槽。

    参数：monkeypatch为同步外部边界替换器。
    返回：None；异常：AssertionError表示取消提前结束或两线程并发。
    """
    backend = _BlockingBackend()
    encoder = _make_encoder(monkeypatch, backend)
    scope_ready = asyncio.Event()
    scope_holder: list[anyio.CancelScope] = []

    async def cancelled_request() -> None:
        """在取消域内编码。参数：无；返回：None；异常：域内取消由CancelScope抑制。"""
        with anyio.CancelScope() as scope:
            scope_holder.append(scope)
            scope_ready.set()
            await encoder.encode(["域取消请求"])

    first = asyncio.create_task(cancelled_request())
    second = None
    try:
        await scope_ready.wait()
        assert await asyncio.to_thread(backend.started.wait, 1)
        scope_holder[0].cancel()
        await asyncio.wait({first}, timeout=0.05)
        assert not first.done()
        second = asyncio.create_task(encoder.encode(["后续请求"]))
        assert not await asyncio.to_thread(backend.second_started.wait, 0.1)
        assert backend.max_inflight == 1
    finally:
        backend.release.set()
        results = await asyncio.gather(*[task for task in (first, second) if task], return_exceptions=True)
        await encoder.aclose()
    assert results[0] is None and isinstance(results[1], list)
    assert backend.completed == [["域取消请求"], ["后续请求"]]


@pytest.mark.parametrize("cancel_encoding", [False, True])
async def test_encoder_close_waits_for_active_thread_and_rejects_reuse(
    monkeypatch: pytest.MonkeyPatch, cancel_encoding: bool,
) -> None:
    """关闭必须等待活动推理结束，关闭后的编码与token计数明确拒绝。

    参数：monkeypatch为外部边界替换器；cancel_encoding决定活动编码是否被取消。
    返回：None；异常：AssertionError表示线程在使用已关闭资源或关闭后仍可使用。
    """
    backend = _BlockingBackend()
    encoder = _make_encoder(monkeypatch, backend)
    first = asyncio.create_task(encoder.encode(["正在推理"]))
    closing = None
    try:
        assert await asyncio.to_thread(backend.started.wait, 1)
        tokenizer = encoder._tokenizer
        if cancel_encoding:
            first.cancel()
        closing = asyncio.create_task(encoder.aclose())
        await asyncio.wait({closing}, timeout=0.05)
        assert not closing.done() and not backend.closed
        assert tokenizer is not None and not tokenizer.closed
    finally:
        backend.release.set()
        results = await asyncio.gather(*[task for task in (first, closing) if task], return_exceptions=True)
        await encoder.aclose()
    assert isinstance(results[0], asyncio.CancelledError if cancel_encoding else list)
    assert results[1] is None and backend.closed and not backend.closed_while_running
    assert tokenizer.closed
    with pytest.raises(EmbeddingUnavailable):
        await encoder.encode(["关闭后使用"])
    with pytest.raises(EmbeddingUnavailable):
        await encoder.encode([])
    with pytest.raises(EmbeddingUnavailable):
        encoder.count_tokens("关闭后计数")


async def test_encoder_inference_failure_releases_slot_and_close_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """推理异常后后续编码仍能执行，资源close异常也不能阻止释放其他资源。

    参数：monkeypatch为外部模型和tokenizer替换器。
    返回：None；异常：AssertionError表示失败后锁或资源没有释放。
    """
    backend = _BlockingBackend()
    backend.error = RuntimeError("推理错误")
    backend.release.set()
    encoder = _make_encoder(monkeypatch, backend)
    with pytest.raises(EmbeddingUnavailable, match="编码失败"):
        await encoder.encode(["失败请求"])
    assert len(await encoder.encode(["恢复请求"])) == 1
    tokenizer = encoder._tokenizer

    def broken_close() -> None:
        """模拟模型关闭失败。参数：无；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("关闭错误")

    monkeypatch.setattr(backend, "close", broken_close)
    await encoder.aclose()
    assert backend.inflight == 0 and tokenizer.closed
    with pytest.raises(EmbeddingUnavailable):
        await encoder.encode(["资源已关闭"])


async def test_encoder_close_waits_for_cancelled_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """加载被取消时关闭仍须等待加载结束，再释放该模型及tokenizer。

    参数：monkeypatch为本地加载边界替换器。
    返回：None；异常：AssertionError表示关闭丢失模型、过早完成或取消后仍推理。
    """
    backend = _FixedBackend()
    encoder = _make_encoder(monkeypatch, backend)
    started, release = threading.Event(), threading.Event()

    def load_model(path: str) -> _FixedBackend:
        """阻塞加载至测试放行。参数：模型路径；返回：模型；异常：超时AssertionError。"""
        del path
        started.set()
        assert release.wait(5), "测试没有按时释放加载线程"
        return backend

    monkeypatch.setattr(embedding, "_load_model", load_model)
    first = asyncio.create_task(encoder.encode(["加载中取消"]))
    closing = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        tokenizer = encoder._tokenizer
        first.cancel()
        closing = asyncio.create_task(encoder.aclose())
        await asyncio.wait({closing}, timeout=0.05)
        assert not closing.done() and not backend.closed and not tokenizer.closed
    finally:
        release.set()
        results = await asyncio.gather(*[task for task in (first, closing) if task], return_exceptions=True)
        await encoder.aclose()
    assert isinstance(results[0], asyncio.CancelledError) and results[1] is None
    assert backend.closed and tokenizer.closed and backend.calls == []


async def test_encoder_cancelled_close_can_be_retried_after_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """等待编码的关闭被取消时不能释放活动资源，之后重试关闭仍能完成。

    参数：monkeypatch为同步外部边界替换器。
    返回：None；异常：AssertionError表示取消关闭丢失资源或阻塞后续清理。
    """
    backend = _BlockingBackend()
    encoder = _make_encoder(monkeypatch, backend)
    first = asyncio.create_task(encoder.encode(["关闭取消竞态"]))
    closing = None
    try:
        assert await asyncio.to_thread(backend.started.wait, 1)
        tokenizer = encoder._tokenizer
        closing = asyncio.create_task(encoder.aclose())
        await asyncio.wait({closing}, timeout=0.02)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not backend.closed and not tokenizer.closed and backend.inflight == 1
    finally:
        backend.release.set()
        await asyncio.gather(*[task for task in (first, closing) if task], return_exceptions=True)
        await encoder.aclose()
    assert backend.closed and tokenizer.closed and not backend.closed_while_running


async def test_tool_executor_timeout_retries_only_after_encoder_thread_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实工具超时重试只能在上一线程结束后开始，不能提前写入超时结果。

    参数：monkeypatch替换模型与数据库外部边界，执行器及业务FAQ工具保持真实。
    返回：None；异常：AssertionError表示并发推理、提前重试或结果持久化错误。
    """
    backend = _BlockingBackend()
    encoder = _make_encoder(monkeypatch, backend)
    service, repository, support, _ = make_chat_service()
    entered_retry = asyncio.Event()

    async def query_faq(keyword: str) -> list[dict]:
        """以真实编码代替查询外部边界。参数：查询词；返回：空命中；异常：编码错误或取消。"""
        await encoder.encode([keyword])
        return []

    async def status(frame: dict) -> None:
        """记录执行器重试阶段。参数：工具状态帧；返回：None；异常：无。"""
        if frame["status"] == "retrying":
            entered_retry.set()

    support.query_faq = query_faq
    context = ToolContext(session_id="timeout", user_id="user", turn_id="turn",
                          tool_call_id="call", user_message="邮费是多少")
    context = await repository.start_turn(context, context.user_message)
    call = {"id": "call", "name": "query_faq", "args": {"keyword": "邮费"}}
    execution = asyncio.create_task(service.executor.execute(call, context, status))
    try:
        assert await asyncio.to_thread(backend.started.wait, 1)
        await asyncio.wait({execution}, timeout=0.15)
        assert not entered_retry.is_set() and not execution.done() and not repository.results
        assert backend.attempts == 1 and backend.max_inflight == 1
        backend.release.set()
        result = await execution
    finally:
        backend.release.set()
        await asyncio.gather(execution, return_exceptions=True)
        await encoder.aclose()
    assert entered_retry.is_set() and backend.attempts == 2 and backend.max_inflight == 1
    assert result.ok and result.data["hits"] == []
    assert next(iter(repository.results.values())) == result


async def test_chat_stream_close_drains_encoder_before_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实聊天关闭工具流时等待编码排空，下一轮只在旧线程结束后进入。

    参数：monkeypatch为模型与数据库替换器，聊天服务和FAQ工具保持真实。
    返回：None；异常：AssertionError表示关闭提前完成、旧线程遗留或下一轮并发。
    """
    backend = _BlockingBackend()
    encoder = _make_encoder(monkeypatch, backend)
    decision = AIMessage(content="", tool_calls=[
        {"id": "faq", "name": "query_faq", "args": {"keyword": "邮费"}}])
    service, repository, support, _ = make_chat_service(decision)
    service.executor.timeout = 2

    async def query_faq(keyword: str) -> list[dict]:
        """执行真实编码边界。参数：查询词；返回：空命中；异常：编码错误或取消。"""
        await encoder.encode([keyword])
        return []

    support.query_faq = query_faq
    request = ChatRequest(session_id="chat-close", message="邮费是多少")
    stream = service.stream_turn(request)
    closing = None
    try:
        assert (await anext(stream)).event == "tool_status"
        assert await asyncio.to_thread(backend.started.wait, 1)
        closing = asyncio.create_task(stream.aclose())
        await asyncio.wait({closing}, timeout=0.1)
        assert not closing.done(), "关闭流必须等待仍在运行的编码线程"
        assert backend.inflight == 1 and not repository.results
        backend.release.set()
        await closing
        assert next(iter(repository.turns.values())).turn_status == "cancelled"
        events = [event async for event in service.stream_turn(request)]
        assert events[-1].event == "done"
    finally:
        backend.release.set()
        await asyncio.gather(*[task for task in (closing,) if task], return_exceptions=True)
        await stream.aclose()
        await encoder.aclose()
    assert backend.attempts == 2 and backend.max_inflight == 1 and backend.inflight == 0
    assert [turn.turn_status for turn in repository.turns.values()] == ["cancelled", "completed"]
