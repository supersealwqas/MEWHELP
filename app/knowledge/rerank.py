"""本地 bge-reranker-large 相关度评分与完整长块窗口重排。

以 tokenizer 校验 query 和 pair 长度，连续窗口覆盖完整三格原文，窗口最高分
映回同一 chunk；默认取 10 个原始块。本模块只管理本地推理，不虚构 HTTP 429。
推理/加载在线程执行，取消时排空后释放锁，关闭不能抢先销毁正在使用的资源。
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, Callable

import anyio
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.knowledge.schemas import ChunkSnapshot, KnowledgeUnavailable
from app.knowledge.text import format_knowledge

RERANKER_REVISION = "bge-reranker-large:flagembedding-1.4.2:sigmoid:pair512:window64:v1"
_MAX_PAIR_TOKENS = 512
_MAX_QUERY_TOKENS = 128
_WINDOW_OVERLAP_TOKENS = 64
_MAX_WINDOWS_PER_CHUNK = 32


class RankedChunk(BaseModel):
    """同一权威 chunk 的重排相关度，不将推理窗口发布成新知识。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk: ChunkSnapshot = Field(description="MySQL 校验通过的完整原文与来源快照")
    score: float = Field(ge=0, le=1, allow_inf_nan=False, description="模型 logit 经 sigmoid 的相关度分数，不是正确概率")


class RerankUnavailable(KnowledgeUnavailable):
    """重排本地资源、模型输出或推理不可用，不能伪装成低分未命中。"""


def _load_tokenizer(path: str) -> Any:
    """只从本地加载模型 tokenizer。参数：path 为本地模型目录；返回：fast tokenizer；异常：OSError/ValueError。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def _load_model(config: Settings) -> Any:
    """加载固定模型的本地权重，禁止把缺失目录当远端模型 ID。

    参数：config 提供 rerank_model_path、设备与批次。
    返回：Any，FlagReranker 的本地模型载体。
    异常：OSError/ValueError，目录或权重缺失；模型依赖错误。
    """
    path = Path(config.rerank_model_path).resolve()
    if (not path.is_dir() or not (path / "config.json").is_file()
            or not any((path / file).is_file() for file in ("model.safetensors", "pytorch_model.bin"))):
        raise OSError("本地重排模型文件缺失")
    from FlagEmbedding import FlagReranker

    return FlagReranker(str(path), use_fp16=config.rerank_device.startswith("cuda"),
                        devices=[config.rerank_device], trust_remote_code=False,
                        batch_size=config.rerank_batch_size, max_length=_MAX_PAIR_TOKENS,
                        query_max_length=_MAX_QUERY_TOKENS, normalize=True)


def _compute_scores(model: Any, tokenizer: Any, pairs: list[tuple[str, str]], config: Settings) -> list[float]:
    """对已校验完整 pair 进行一次有界批次推理，不隐式截断或循环降低批次。

    参数：model 为 FlagReranker 权重载体；tokenizer 为同目录 tokenizer；pairs 为完整 query/窗口；config 为设备与批次。
    返回：list[float]，每个 pair 的 sigmoid 相关度。
    异常：RuntimeError/模型依赖错误（含 OOM）直接上抛，由上层安全包装，不自动换模型。
    """
    import torch

    backend = model.model
    if config.rerank_device.startswith("cuda"):
        backend.half()
    backend.to(config.rerank_device)
    backend.eval()
    scores: list[float] = []
    # FlagEmbedding 1.4.2 compute_score 内部会吞 RuntimeError 不停缩小 batch；
    # 使用同一已加载权重的明确 forward，保留 sigmoid，失败立即报告而非无界重试。
    with torch.inference_mode():
        for offset in range(0, len(pairs), config.rerank_batch_size):
            batch = pairs[offset:offset + config.rerank_batch_size]
            inputs = tokenizer([query for query, _ in batch], [body for _, body in batch],
                               padding=True, truncation=False, return_tensors="pt")
            logits = backend(**inputs.to(config.rerank_device), return_dict=True).logits.reshape(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


def _windows(query: str, text: str, tokenizer: Any) -> list[str]:
    """将三格原文按 tokenizer 连续偏移分窗，实际 pair 再计数避免隐形截尾。

    参数：query 已过 128 token 校验；text 原始完整三格；tokenizer 支持 offset_mapping。
    返回：list[str]，重叠约 64 token，首尾与中间均覆盖的原文连续窗口。
    异常：ValueError，窗口资源超限；RerankUnavailable，offset 或 pair 无法保持完整。
    """
    query_size = len(tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"])
    budget = _MAX_PAIR_TOKENS - query_size - tokenizer.num_special_tokens_to_add(pair=True)
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(encoded["input_ids"]) or not offsets:
        raise RerankUnavailable("重排 tokenizer 无法提供完整窗口偏移")
    windows: list[str] = []
    start = 0
    while start < len(offsets):
        if len(windows) >= _MAX_WINDOWS_PER_CHUNK:
            raise ValueError("知识块超过重排窗口上限，请缩小原文")
        end = min(start + budget, len(offsets))
        left = 0 if start == 0 else offsets[start][0]
        while end > start:
            # 末窗保留最后 token 之后的原始空白，非末窗纳入 token 间分隔字符。
            right = len(text) if end == len(offsets) else offsets[end][0]
            body = text[left:right]
            pair_size = len(tokenizer(query, body, add_special_tokens=True, truncation=False)["input_ids"])
            if pair_size <= _MAX_PAIR_TOKENS:
                break
            end -= 1
        if end <= start or not body:
            raise RerankUnavailable("重排无法构造不截断的完整 pair")
        windows.append(body)
        if end == len(offsets):
            break
        # 始终推进 token 起点；保留重叠覆盖边界，不能跳过正文中间部分。
        start = max(start + 1, end - _WINDOW_OVERLAP_TOKENS)
    return windows


class BgeReranker:
    """串行本地长块重排与资源生命周期，输出仍对应完整原 chunk。"""

    def __init__(self, config: Settings) -> None:
        """保存配置并建立推理锁。参数：config 为模型/设备配置；返回：None；异常：无。"""
        self.config = config
        self.revision = RERANKER_REVISION
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = anyio.Lock()
        self._closed = False

    async def _run_sync(self, function: Callable, *args: Any) -> Any:
        """在线程执行加载或推理，重复取消仍等待真实工作结束。

        参数：function 同步函数；args 函数参数。
        返回：Any，同步结果。
        异常：asyncio.CancelledError，排空后传播原取消；同步错误未取消时原样传播。
        """
        worker = asyncio.create_task(anyio.to_thread.run_sync(function, *args))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
            if not worker.cancelled():
                worker.exception()
            raise

    def _rank_sync(self, query: str, chunks: list[ChunkSnapshot], top_n: int) -> list[RankedChunk]:
        """持有已加载资源并在完整窗口上评分后映回原主键。

        参数：query 原意查询；chunks 去重权威快照；top_n 返回上限。
        返回：list[RankedChunk]，按分数降序、主键升序稳定排列。
        异常：ValueError，query/窗口超限；RerankUnavailable，模型输出不合法；加载/推理错误。
        """
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.config.rerank_model_path)
        query_size = len(self._tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"])
        if query_size > _MAX_QUERY_TOKENS:
            raise ValueError("查询超过重排 128 token 上限，请拆分问题，不能自动截断")
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for chunk in chunks:
            for window in _windows(query, format_knowledge(chunk.text), self._tokenizer):
                pairs.append((query, window))
                owners.append(chunk.id)
        if self._model is None:
            # 先存到资源实例，再返回线程结果，取消不会遗失加载成功的模型。
            self._model = _load_model(self.config)
        values = _compute_scores(self._model, self._tokenizer, pairs, self.config)
        if len(values) != len(pairs) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise RerankUnavailable("重排模型返回分数数量或数值不合法")
        scores: dict[int, float] = {}
        for identifier, score in zip(owners, values):
            scores[identifier] = max(scores.get(identifier, 0), score)
        ranked = [RankedChunk(chunk=chunk, score=scores[chunk.id]) for chunk in chunks]
        return sorted(ranked, key=lambda item: (-item.score, item.chunk.id))[:top_n]

    async def rank(self, query: str, chunks: list[ChunkSnapshot], *, top_n: int = 10) -> list[RankedChunk]:
        """按原 query 对完整 chunk 重排，默认最多 10。

        参数：query 非空完整查询；chunks 权威快照；top_n 为 1–50 正整数。
        返回：list[RankedChunk]，同主键只出现一次，空候选不加载模型。
        异常：ValueError，输入/限额非法；RerankUnavailable，本地依赖/模型失败；取消排空后传播。
        """
        if type(top_n) is not int or not 1 <= top_n <= 50 or not query.strip():
            raise ValueError("重排 query 或 top_n 不合法")
        async with self._lock:
            if self._closed:
                raise RerankUnavailable("本地重排器已关闭")
            if not chunks:
                return []
            by_id: dict[int, ChunkSnapshot] = {}
            for chunk in chunks:
                if chunk.id in by_id and by_id[chunk.id] != chunk:
                    raise RerankUnavailable("同主键候选正文不一致")
                by_id[chunk.id] = chunk
            try:
                return await self._run_sync(self._rank_sync, query, list(by_id.values()), top_n)
            except (ValueError, RerankUnavailable):
                raise
            except Exception as error:
                raise RerankUnavailable("本地重排模型加载或推理失败") from error

    async def aclose(self) -> None:
        """排空活动加载/推理后释放资源。参数：无；返回：None；异常：等待锁时取消可重试，资源关闭异常抑制。"""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            resources = (self._model, self._tokenizer)
            self._model = self._tokenizer = None
            for resource in resources:
                close = getattr(resource, "stop_self_pool", None) or getattr(resource, "close", None)
                if callable(close):
                    try:
                        await self._run_sync(close)
                    except Exception:
                        pass
