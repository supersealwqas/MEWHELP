"""本地重排的行为测试：长块全文、稳定主键、非法输出与取消排空。"""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.knowledge.schemas import ChunkSnapshot, KnowledgeText
from app.knowledge.text import content_fingerprint
from tests.knowledge.fakes import make_settings


class CharacterTokenizer:
    """逐字计数的轻量 tokenizer，仅替换真实分词依赖。"""

    def __call__(self, text: str, pair: str | None = None, *, add_special_tokens: bool = True,
                 return_offsets_mapping: bool = False, **kwargs) -> dict:
        """生成明确偏移。参数：原文/配对/计数选项；返回：编码；异常：无。"""
        value = text + (pair or "")
        data = {"input_ids": list(range(len(value) + (4 if pair is not None else 2) * add_special_tokens))}
        if return_offsets_mapping:
            data["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return data

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """返回配对特殊 token 数。参数：pair 是否配对；返回：4 或 2；异常：无。"""
        return 4 if pair else 2


def _chunk(identifier: int, answer: str) -> ChunkSnapshot:
    """构造真实知识快照。参数：主键/正文；返回：快照；异常：模型校验错误。"""
    text = KnowledgeText(category="售后", questions="处理规定", answer=answer)
    return ChunkSnapshot(id=identifier, source_key=f"doc:{identifier}.md", ordinal=0, text=text,
                         content_type="policy", content_hash=content_fingerprint(text),
                         embedding_revision="fake-encoder-v1", vectorize_status="done", vector_id=str(identifier))


def _reranker(monkeypatch, score):
    """仅替换本地模型与同步推理边界。参数：pytest 替换器/评分函数；返回：真实重排服务；异常：无。"""
    from app.knowledge import rerank

    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(rerank, "_load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(rerank, "_load_model", lambda config: object())
    monkeypatch.setattr(rerank, "_compute_scores", score)
    return rerank.BgeReranker(make_settings())


async def test_tail_only_evidence_scores_original_chunk_once(monkeypatch) -> None:
    """正文尾部不能因模型 512 输入默认截断而丢失，窗口不得变成新 chunk。参数：替换器；返回：None；异常：断言失败。"""
    seen = []

    def score(model, tokenizer, pairs, config):
        """模拟外部推理，尾部证据更相关。参数：推理依赖/完整 pair；返回：分数；异常：断言失败。"""
        seen.extend(pairs)
        assert all(len(query) + len(body) + 4 <= 512 for query, body in pairs)
        return [0.9 if "尾部退款时限" in body else 0.1 for _, body in pairs]

    ranking = _reranker(monkeypatch, score)
    # 使用唯一字符序列，避免重复「背景」让原文位置匹配产生歧义。
    target = _chunk(1, "".join(chr(0x4E00 + index) for index in range(1200)) + "尾部退款时限：3个工作日。")
    try:
        result = await ranking.rank("退款时限", [target])
        assert len(result) == 1 and result[0].chunk == target and result[0].score == 0.9
        windows = [body for _, body in seen]
        assert len(windows) > 1 and windows[-1].endswith("尾部退款时限：3个工作日。")
        # 每个后窗与前窗重叠，保证完整原文覆盖而不是只看首尾两窗。
        full = "分类：售后\n问法：处理规定\n答案：" + target.text.answer
        cursor = 0
        for window in windows:
            start = full.find(window, max(0, cursor - len(window)))
            assert 0 <= start <= cursor
            cursor = max(cursor, start + len(window))
        assert cursor == len(full)
    finally:
        await ranking.aclose()


async def test_equal_scores_have_stable_id_order_and_duplicate_input_is_removed(monkeypatch) -> None:
    """并列分按主键稳定排序，同一个 chunk 不能占两个输出位置。参数：替换器；返回：None；异常：断言失败。"""
    def score(model, tokenizer, pairs, config):
        """外部模型返回并列相关度。参数：模型/分词/pair/配置；返回：分数；异常：无。"""
        return [0.5] * len(pairs)

    ranking = _reranker(monkeypatch, score)
    first, second = _chunk(2, "乙。"), _chunk(1, "甲。")
    try:
        result = await ranking.rank("退款条件", [first, second, first])
        assert [item.chunk.id for item in result] == [1, 2]
    finally:
        await ranking.aclose()


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [0.5, 0.6], [], [-0.1], [1.1]])
async def test_invalid_or_incomplete_scores_fail_closed(monkeypatch, scores: list) -> None:
    """非法归一分数或缺项不能伪装成低分未命中。参数：替换器/错误分数；返回：None；异常：断言失败。"""
    from app.knowledge.rerank import RerankUnavailable

    ranking = _reranker(monkeypatch, lambda *args: scores)
    try:
        with pytest.raises(RerankUnavailable):
            await ranking.rank("退款条件", [_chunk(1, "甲。")])
    finally:
        await ranking.aclose()


async def test_query_over_token_limit_is_not_truncated_or_sent_to_model(monkeypatch) -> None:
    """超长 query 需明确请用户拆分，不能截掉否定或型号。参数：替换器；返回：None；异常：断言失败。"""
    def forbid(*args):
        """禁止超长 query 推理。参数：任意；返回：不返回；异常：AssertionError。"""
        raise AssertionError("不能执行推理")

    ranking = _reranker(monkeypatch, forbid)
    try:
        with pytest.raises(ValueError, match="拆分|128"):
            await ranking.rank("否" * 129, [_chunk(1, "甲。")])
    finally:
        await ranking.aclose()


@pytest.mark.parametrize("limit", [0, -1, True])
async def test_invalid_top_n_is_rejected_before_model(monkeypatch, limit: int) -> None:
    """显式零值不能被默认 10 吞掉。参数：替换器/限额；返回：None；异常：断言失败。"""
    ranking = _reranker(monkeypatch, lambda *args: [0.5])
    try:
        with pytest.raises(ValueError):
            await ranking.rank("退款条件", [_chunk(1, "甲。")], top_n=limit)
    finally:
        await ranking.aclose()


async def test_cancelled_inference_drains_before_close(monkeypatch) -> None:
    """反复取消不得提前释放仍在使用的模型资源。参数：替换器；返回：None；异常：断言失败。"""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def score(model, tokenizer, pairs, config):
        """阻塞真实服务的外部推理线程。参数：依赖/输入；返回：分数；异常：断言失败。"""
        entered.set()
        try:
            assert release.wait(5)
            return [0.5] * len(pairs)
        finally:
            finished.set()

    ranking = _reranker(monkeypatch, score)
    job = asyncio.create_task(ranking.rank("退款条件", [_chunk(1, "甲。")]))
    close = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        for _ in range(3):
            job.cancel()
            await asyncio.wait({job}, timeout=0.01)
            assert not job.done()
        close = asyncio.create_task(ranking.aclose())
        await asyncio.wait({close}, timeout=0.02)
        assert not close.done() and not finished.is_set()
    finally:
        release.set()
        results = await asyncio.gather(*[task for task in (job, close) if task], return_exceptions=True)
        await ranking.aclose()
    assert isinstance(results[0], asyncio.CancelledError) and finished.is_set()
    from app.knowledge.rerank import RerankUnavailable
    with pytest.raises(RerankUnavailable):
        await ranking.rank("退款条件", [_chunk(1, "甲。")])
