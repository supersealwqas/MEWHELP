"""知识模块实际使用的编码、token 计数、索引与模型小接口。"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from app.knowledge.schemas import (
    EvidenceItem, ExtractedQA, ExtractionBatch, RelationDecision, VectorHit, VectorRecord,
)


class TokenCounter(Protocol):
    """与实际编码 tokenizer 一致的文本长度函数。"""

    def __call__(self, text: str) -> int:
        """统计含特殊 token 的长度。参数：文本；返回：token 数；异常：编码依赖异常。"""
        ...


class KnowledgeEncoder(Protocol):
    """Dense 编码与资源生命周期接口。"""

    revision: str  # 编码规范指纹。

    def count_tokens(self, text: str) -> int:
        """统计长度。参数：文本；返回：token 数；异常：编码依赖异常。"""
        ...

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码。参数：合规文本；返回：1024 维向量；异常：编码不可用。"""
        ...

    async def aclose(self) -> None:
        """释放编码资源。参数：无；返回：None；异常：依赖异常。"""
        ...


class VectorIndex(Protocol):
    """同步与检索所需的 Dense 索引接口。"""

    async def ensure_ready(self, create: bool = False) -> None:
        """验证资源。参数：是否允许创建；返回：None；异常：索引不可用。"""
        ...

    async def upsert(self, records: list[VectorRecord]) -> None:
        """同主键全量写入。参数：记录；返回：None；异常：索引不可用。"""
        ...

    async def search(self, vector: list[float], limit: int) -> list[VectorHit]:
        """检索 Dense 向量。参数：向量、条数；返回：命中；异常：索引不可用。"""
        ...

    async def get_fingerprints(self, ids: list[int]) -> dict[int, tuple[str, str]]:
        """强一致批量读取同步指纹。参数：主键；返回：内容/编码指纹；异常：索引不可用。"""
        ...

    def iter_ids(self, batch_size: int = 1000) -> AsyncIterator[list[int]]:
        """分页遍历主键。参数：正页大小；返回：异步主键批次；异常：索引不可用。"""
        ...

    async def delete(self, ids: list[int]) -> None:
        """幂等删除。参数：主键；返回：None；异常：索引不可用。"""
        ...

    async def aclose(self) -> None:
        """关闭资源。参数：无；返回：None；异常：依赖异常。"""
        ...


class KnowledgeModel(Protocol):
    """从完整批次抽取 QA，并对指定证据 pair 判定关系。"""

    async def extract(self, batch: ExtractionBatch) -> list[ExtractedQA]:
        """抽取批次知识。参数：批次；返回：QA；异常：模型/验证失败。"""
        ...

    async def relate(
        self, items: list[EvidenceItem], pairs: list[tuple[str, str]]
    ) -> list[RelationDecision]:
        """核对指定关系。参数：证据、pair；返回：决策；异常：模型/验证失败。"""
        ...
