"""本地 BGE-M3 Dense 编码器。

模型只在首次显式编码时加载，使用本地权重、FP16/CUDA、CLS 池化与归一化；
同进程 GPU 编码通过异步锁串行，线程取消时等待已开始的推理结束，避免半途
释放模型资源。token 计数使用与编码一致的 tokenizer，并在推理前拒绝超限输入。
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from typing import Any

import anyio

from app.config import Settings
from app.knowledge.limits import MAX_INPUT_TOKENS, VECTOR_DIM, valid_dense_vector
from app.knowledge.schemas import KnowledgeUnavailable

EMBEDDING_DIM = VECTOR_DIM
EMBEDDING_REVISION = "bge-m3:flagembedding-1.4.2:fp16:cls:normalized:max2048:v1"


class EmbeddingUnavailable(KnowledgeUnavailable):
    """本地 BGE-M3 模型或编码过程不可用。"""


def _load_tokenizer(model_path: str) -> Any:
    """从本地目录加载与编码器一致的 tokenizer。

    参数:
        model_path: 本地 BGE-M3 模型目录。
    返回值:
        transformers tokenizer 实例。
    异常:
        OSError、ValueError: 本地文件缺失或 tokenizer 配置不合法时抛出。
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=True,
    )


def _load_model(model_path: str) -> Any:
    """按固定 Dense-only 配置加载本地 BGE-M3。

    参数:
        model_path: 本地 BGE-M3 模型目录。
    返回值:
        FlagEmbedding BGEM3FlagModel 实例。
    异常:
        OSError、ValueError: 本地权重缺失或推理配置不合法时抛出。
    """
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(
        model_path,
        normalize_embeddings=True,
        use_fp16=True,
        query_instruction_for_retrieval=None,
        devices=["cuda:0"],
        pooling_method="cls",
        trust_remote_code=False,
    )


class BgeM3Encoder:
    """受控的本地 BGE-M3 Dense 编码器。"""

    def __init__(self, config: Settings) -> None:
        """保存配置与并发控制资源，不加载模型。

        参数:
            config: 应用配置，读取 knowledge_model_path。
        返回值:
            None
        异常:
            无
        """
        self._config = config
        self.revision = EMBEDDING_REVISION
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._tokenizer_lock = threading.Lock()
        self._encode_lock = anyio.Lock()
        self._closed = False

    def count_tokens(self, text: str) -> int:
        """按编码器 tokenizer 统计含特殊 token 的长度。

        参数:
            text: 待统计文本。
        返回值:
            int: input_ids 长度。
        异常:
            EmbeddingUnavailable: 编码器已关闭或 tokenizer 无法加载时抛出。
        """
        tokenizer = self._ensure_tokenizer()
        encoded = tokenizer(text, add_special_tokens=True)
        return len(encoded["input_ids"])

    def _ensure_tokenizer(self) -> Any:
        """惰性加载 tokenizer，并保证并发下只加载一次。

        参数:
            无
        返回值:
            tokenizer 实例。
        异常:
            EmbeddingUnavailable: 编码器已关闭或本地 tokenizer 加载失败时抛出。
        """
        if self._closed:
            raise EmbeddingUnavailable("本地 BGE-M3 编码器已关闭")
        if self._tokenizer is None:
            with self._tokenizer_lock:
                # 关闭也持有 tokenizer 锁，禁止正在等待加载的计数请求重新创建资源。
                if self._closed:
                    raise EmbeddingUnavailable("本地 BGE-M3 编码器已关闭")
                if self._tokenizer is None:
                    try:
                        self._tokenizer = _load_tokenizer(self._config.knowledge_model_path)
                    except Exception as error:
                        raise EmbeddingUnavailable("本地 BGE-M3 tokenizer 加载失败") from error
        return self._tokenizer

    async def _run_sync(self, function: Callable[..., Any], *args: Any) -> Any:
        """保留同步工作任务，取消时排空线程后再传播原取消。

        参数:
            function: 在线程执行的同步加载或编码函数。
            *args: 传给同步函数的业务参数。
        返回值:
            Any: 同步函数的返回值。
        异常:
            asyncio.CancelledError: 调用取消后等待线程结束，再重抛首次取消。
            Exception: 未取消时原样传播同步函数异常。
        """
        # 保留强引用，并屏蔽原生 task.cancel；取消协程不代表底层线程已经退出。
        worker = asyncio.create_task(anyio.to_thread.run_sync(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # AnyIO 的重复域取消由屏蔽域拦截；原生重复 cancel 仍须逐次接住并继续等待。
            with anyio.CancelScope(shield=True):
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
            # 排空期间的线程失败不能覆盖原取消，但必须取走异常，避免后台未读取告警。
            if not worker.cancelled():
                worker.exception()
            raise

    def _load_model_sync(self) -> Any:
        """加载完成后立即保存模型，防止外层取消导致已加载的模型丢失。

        参数:
            无；模型路径来自当前配置，调用方须持有编码锁。
        返回值:
            Any: 已加载并由编码器持有的本地模型。
        异常:
            OSError、ValueError: 本地权重或配置无法加载时抛出。
        """
        self._model = _load_model(self._config.knowledge_model_path)
        return self._model

    async def _ensure_model(self) -> Any:
        """在工作线程中惰性加载模型，避免阻塞事件循环。

        参数:
            无
        返回值:
            已加载的 BGE-M3 模型。
        异常:
            EmbeddingUnavailable: 本地模型加载失败时抛出。
            asyncio.CancelledError: 加载线程结束后传播调用方取消。
        """
        if self._model is None:
            try:
                await self._run_sync(self._load_model_sync)
            except Exception as error:
                raise EmbeddingUnavailable("本地 BGE-M3 模型加载失败") from error
        return self._model

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """串行编码一组文本，返回 1024 维 Dense 向量。

        参数:
            texts: 待编码文本列表；每项必须不超过 2048 token。
        返回值:
            list[list[float]]: 与输入等长的有限浮点向量。
        异常:
            ValueError: 任一输入超过 2048 token。
            EmbeddingUnavailable: 编码器已关闭、模型加载、推理或返回形状异常。
            asyncio.CancelledError: 已开始的加载或推理线程结束后传播取消。
        """
        if self._closed:
            raise EmbeddingUnavailable("本地 BGE-M3 编码器已关闭")
        if not texts:
            return []
        # 先按与 encode 相同的 tokenizer 校验，超限时不加载模型、不调用推理。
        for index, text in enumerate(texts):
            token_count = self.count_tokens(text)
            if token_count > MAX_INPUT_TOKENS:
                raise ValueError(
                    f"知识文本超过硬限: index={index}, tokens={token_count}, max={MAX_INPUT_TOKENS}"
                )
        async with self._encode_lock:
            # 编码请求可能在关闭前开始排队，拿到锁后必须重新检查资源生命周期。
            if self._closed:
                raise EmbeddingUnavailable("本地 BGE-M3 编码器已关闭")
            model = await self._ensure_model()
            try:
                vectors = await self._run_sync(self._encode_sync, model, list(texts))
            except EmbeddingUnavailable:
                raise
            except Exception as error:
                raise EmbeddingUnavailable("本地 BGE-M3 编码失败") from error
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable("本地 BGE-M3 返回向量数量不一致")
        for vector in vectors:
            if not valid_dense_vector(vector):
                raise EmbeddingUnavailable("本地 BGE-M3 返回向量维度或数值不合法")
        return vectors

    def _encode_sync(self, model: Any, texts: list[str]) -> list[list[float]]:
        """在工作线程中执行同步推理。

        参数:
            model: 已加载的 BGE-M3 模型。
            texts: 已通过 token 上限校验的文本。
        返回值:
            list[list[float]]: Python 浮点向量列表。
        异常:
            KeyError、TypeError: 模型返回结构不符合预期时抛出。
        """
        result = model.encode(
            texts,
            batch_size=1,
            max_length=MAX_INPUT_TOKENS,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [[float(value) for value in row] for row in result["dense_vecs"]]

    async def aclose(self) -> None:
        """等待活动加载或推理结束，再释放资源并拒绝后续使用。

        参数:
            无
        返回值:
            None
        异常:
            asyncio.CancelledError: 等待编码锁时取消会向外传播，资源仍保留供重试关闭。
            资源自身的关闭失败会被抑制，不覆盖调用方原始错误。
        """
        # 同一锁只在线程实际结束后释放，因此关闭不能抢先清理线程仍在使用的模型。
        async with self._encode_lock:
            with self._tokenizer_lock:
                if self._closed:
                    return
                self._closed = True
                model, tokenizer = self._model, self._tokenizer
                self._model = None
                self._tokenizer = None
            for resource in (model, tokenizer):
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            # 仅在 torch 已被编码器加载时清理缓存，避免导入模块阶段引入 torch。
            torch = sys.modules.get("torch")
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
