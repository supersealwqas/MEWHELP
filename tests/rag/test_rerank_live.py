"""显式开启的本地 bge-reranker-large 真实推理检查，不联网加载模型。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.knowledge.rerank import BgeReranker, _windows
from app.knowledge.schemas import ChunkSnapshot, KnowledgeText
from app.knowledge.text import content_fingerprint, format_knowledge
from tests.knowledge.fakes import make_settings


@pytest.mark.skipif(os.getenv("MEWHELP_RUN_RERANK_LIVE") != "1", reason="需要显式 MEWHELP_RUN_RERANK_LIVE=1")
async def test_real_local_reranker_scores_tail_evidence_without_truncation() -> None:
    """真实权重应识别长正文尾部答案，所有实际 pair 不超 512。参数：无；返回：None；异常：模型/断言错误。"""
    model_path = os.getenv("MEWHELP_LOCAL_RERANK_MODEL", "models/bge-reranker-large")
    assert Path(model_path).is_dir(), "实测必须提供本地模型目录，不下载"
    reranker = BgeReranker(make_settings(rerank_model_path=model_path))
    chunks = []
    for identifier, answer in enumerate([
        "背景说明。" * 180 + "退款处理时限为3个工作日，不承诺到账时间。",
        "猫爬架高度1.6米，需自行组装。",
    ], 1):
        text = KnowledgeText(category="售后", questions="退款处理时限" if identifier == 1 else "猫爬架", answer=answer)
        chunks.append(ChunkSnapshot(id=identifier, source_key=f"doc:{identifier}.md", ordinal=0, text=text,
                                    content_type="policy", content_hash=content_fingerprint(text), embedding_revision="live-only"))
    try:
        query = "退款处理要几个工作日"
        ranked = await reranker.rank(query, chunks)
        assert ranked[0].chunk.id == 1 and ranked[0].score > 0.3
        assert len(ranked[0].chunk.text.answer) == 921
        tokenizer = reranker._tokenizer
        windows = _windows(query, format_knowledge(chunks[0].text), tokenizer)
        assert len(windows) > 1
        assert windows[-1].endswith("退款处理时限为3个工作日，不承诺到账时间。")
        assert all(len(tokenizer(query, body, truncation=False)["input_ids"]) <= 512 for body in windows)
    finally:
        await reranker.aclose()
