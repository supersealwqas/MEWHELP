"""本地 BGE-M3 Dense 编码器。

模型只在首次显式编码时加载，使用本地权重、FP16/CUDA、CLS 池化与归一化；
同进程 GPU 编码通过异步锁串行，线程取消时等待已开始的推理结束，避免半途
释放模型资源。token 计数使用与编码一致的 tokenizer，并在推理前拒绝超限输入。
"""

from __future__ import annotations

import math
import sys
import threading
from typing import Any

import anyio

from app.config import Settings
from app.knowledge.schemas import KnowledgeUnavailable

EMBEDDING_DIM = 1024
MAX_INPUT_TOKENS = 2048
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

    def count_tokens(self, text: str) -> int:
        """按编码器 tokenizer 统计含特殊 token 的长度。

        参数:
            text: 待统计文本。
        返回值:
            int: input_ids 长度。
        异常:
            EmbeddingUnavailable: tokenizer 无法加载时抛出。
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
            EmbeddingUnavailable: 本地 tokenizer 加载失败时抛出。
        """
        if self._tokenizer is None:
            with self._tokenizer_lock:
                if self._tokenizer is None:
                    try:
                        self._tokenizer = _load_tokenizer(self._config.knowledge_model_path)
                    except Exception as error:
                        raise EmbeddingUnavailable("本地 BGE-M3 tokenizer 加载失败") from error
        return self._tokenizer

    async def _ensure_model(self) -> Any:
        """在工作线程中惰性加载模型，避免阻塞事件循环。

        参数:
            无
        返回值:
            已加载的 BGE-M3 模型。
        异常:
            EmbeddingUnavailable: 本地模型加载失败时抛出。
        """
        if self._model is None:
            try:
                self._model = await anyio.to_thread.run_sync(
                    _load_model, self._config.knowledge_model_path
                )
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
            EmbeddingUnavailable: 模型加载、推理或返回形状异常。
        """
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
            model = await self._ensure_model()
            try:
                vectors = await anyio.to_thread.run_sync(self._encode_sync, model, list(texts))
            except EmbeddingUnavailable:
                raise
            except Exception as error:
                raise EmbeddingUnavailable("本地 BGE-M3 编码失败") from error
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable("本地 BGE-M3 返回向量数量不一致")
        for vector in vectors:
            if len(vector) != EMBEDDING_DIM:
                raise EmbeddingUnavailable("本地 BGE-M3 返回向量维度不是 1024")
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingUnavailable("本地 BGE-M3 返回非有限向量")
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
        """释放模型与 tokenizer 引用，并清理 CUDA 缓存。

        参数:
            无
        返回值:
            None
        异常:
            无；关闭失败不覆盖调用方原始错误。
        """
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
