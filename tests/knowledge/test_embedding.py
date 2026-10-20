"""BgeM3Encoder 的离线 TDD：超限拒绝、GPU 串行、取消排空与向量形状。"""

from __future__ import annotations

import asyncio
import math
import threading
from typing import Any

import pytest

from app.knowledge import embedding
from app.knowledge.embedding import BgeM3Encoder
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
    """第一次 encode 会阻塞，用于验证串行与取消排空。"""

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
        self.release = threading.Event()
        self.inflight = 0
        self.max_inflight = 0
        self.completed: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> dict[str, list[list[float]]]:
        """阻塞到测试放行，并记录并发峰值。

        参数:
            sentences: 待编码文本。
            **kwargs: 透传给父类记录。
        返回值:
            dict[str, list[list[float]]]: 固定 dense_vecs。
        异常:
            无
        """
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.started.set()
        self.release.wait(timeout=5)
        self.inflight -= 1
        self.completed.append(list(sentences))
        return super().encode(sentences, **kwargs)


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


async def test_encoder_serializes_gpu_calls_and_drains_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU 编码串行执行，取消时等待已开始的线程结束。

    参数:
        monkeypatch: pytest 替换器，用于注入 tokenizer 与阻塞后端。
    返回值:
        None
    异常:
        无
    """
    backend = _BlockingBackend()
    monkeypatch.setattr(embedding, "_load_tokenizer", lambda path: _FakeTokenizer(2))
    monkeypatch.setattr(embedding, "_load_model", lambda path: backend)
    encoder = BgeM3Encoder(make_settings())
    assert backend.calls == []

    first = asyncio.create_task(encoder.encode(["第一句"]))
    assert await asyncio.to_thread(backend.started.wait, 2.0)
    assert backend.max_inflight == 1
    second = asyncio.create_task(encoder.encode(["第二句"]))
    await asyncio.sleep(0.05)
    assert backend.max_inflight == 1

    first.cancel()
    backend.release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert backend.max_inflight == 1
    assert backend.completed == [["第一句"], ["第二句"]]
    await encoder.aclose()


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
