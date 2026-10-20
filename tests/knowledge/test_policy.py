"""在线检索校准策略版本与非法值测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.knowledge.schemas import RetrievalPolicy
from app.knowledge.search import KnowledgeSearchUnavailable, load_retrieval_policy

_TEMP_BASE = Path(".pytest-tmp")


def _write_policy(root: Path, payload: dict) -> Path:
    """写入临时策略。参数：目录与内容；返回：路径；异常：无。"""
    path = root / "retrieval-policy.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _payload(**overrides) -> dict:
    """构造策略 JSON。参数：覆盖字段；返回：字典；异常：无。"""
    values = {
        "min_score": 0.8,
        "embedding_revision": "rev-1",
        "chunking_version": "ch03-v2",
        "calibration_digest": "digest",
    }
    values.update(overrides)
    return values


def test_policy_version_mismatch_is_rejected() -> None:
    """编码/切分版本不一致时拒绝启动检索。参数：无；返回：None；异常：无。"""
    _TEMP_BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ch03-policy-", dir=_TEMP_BASE) as temp:
        path = _write_policy(Path(temp), _payload())
        policy = load_retrieval_policy(
            path, embedding_revision="rev-1", chunking_version="ch03-v2"
        )
        assert policy.min_score == 0.8
        with pytest.raises(KnowledgeSearchUnavailable):
            load_retrieval_policy(
                path, embedding_revision="rev-2", chunking_version="ch03-v2"
            )
        with pytest.raises(KnowledgeSearchUnavailable):
            load_retrieval_policy(
                path, embedding_revision="rev-1", chunking_version="ch03-v1"
            )


def test_missing_or_non_finite_policy_is_rejected() -> None:
    """缺失文件或非有限阈值不能使用经验默认值。参数：无；返回：None；异常：无。"""
    _TEMP_BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ch03-policy-", dir=_TEMP_BASE) as temp:
        root = Path(temp)
        with pytest.raises(KnowledgeSearchUnavailable):
            load_retrieval_policy(
                root / "missing.json",
                embedding_revision="rev-1",
                chunking_version="ch03-v2",
            )
        bad = _payload(min_score=float("nan"))
        path = _write_policy(root, bad)
        with pytest.raises(KnowledgeSearchUnavailable):
            load_retrieval_policy(
                path, embedding_revision="rev-1", chunking_version="ch03-v2"
            )


def test_retrieval_policy_rejects_out_of_range() -> None:
    """Pydantic 拒绝超出余弦范围的阈值。参数：无；返回：None；异常：无。"""
    with pytest.raises(Exception):
        RetrievalPolicy(
            min_score=2.0,
            embedding_revision="rev",
            chunking_version="ch03-v2",
            calibration_digest="digest",
        )
