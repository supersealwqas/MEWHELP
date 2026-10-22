"""跨评估轮次的编造个案台账仓储。

按 eval_id 唯一键原子累计被判编造次数，覆盖最近答案/裁判/证据全集；复发时
退回未解决并清说明，但保持首次和上次处置时间。人工处置必须说明解决办法，
仅持短事务，不把数据库事务延伸到裁判模型等待。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert

from app.knowledge.answering import EvidenceCitation
from app.models.rag_quality import FaithCase


class FaithCaseDraft(BaseModel):
    """裁判确认编造后写入台账的本轮完整快照，不接受人工处置字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    eval_id: str = Field(min_length=1, max_length=16, description="评估题号，一题一行")
    bucket: str = Field(min_length=1, max_length=24, description="评估题目类型桶")
    query: str = Field(min_length=1, max_length=512, description="当前评估原问题，保留原话")
    strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = Field(description="产生本轮答案的真实检索策略")
    answer: str = Field(min_length=1, description="被裁判判编造的完整本轮答案")
    reason: str = Field(min_length=1, description="裁判给的具体无证据事实与理由")
    citations: list[EvidenceCitation] = Field(description="本轮实际送入模型的全部Top-K证据，不是答案引用子集")
    judge_model: str | None = Field(default=None, max_length=64, description="裁判模型标识")

    @field_validator("answer", "reason")
    @classmethod
    def check_text_bytes(cls, value: str) -> str:
        """校验文本适合 MySQL TEXT。参数：value 原文本；返回：原值；异常：ValueError，空白或超过65535字节。"""
        if not value.strip() or len(value.encode("utf-8")) > 65535:
            raise ValueError("台账答案/理由为空或超过TEXT上限")
        return value

    @field_validator("citations")
    @classmethod
    def check_complete_numbering(cls, value: list[EvidenceCitation]) -> list[EvidenceCitation]:
        """校验全集编号连续且主键不重复。参数：value 全部证据；返回：原列表；异常：ValueError，重复或缺号。"""
        if ([item.n for item in value] != list(range(1, len(value) + 1))
                or len({item.chunk_id for item in value}) != len(value)):
            raise ValueError("台账证据编号不连续或主键重复")
        return value


class FaithCaseRepository:
    """以数据库唯一键处理并发累计和短事务人工处置，不覆盖复发时间。"""

    def __init__(self, session_factory) -> None:
        """保存异步会话工厂。参数：session_factory 短事务工厂；返回：None；异常：无。"""
        self.factory = session_factory

    async def record(self, draft: FaithCaseDraft) -> None:
        """按题号原子更新最近编造事实或创建首条记录。

        参数：draft 已校验裁判快照，citations 允许空（无证据却回答也可能编造）。
        返回：None，一题一行，重复出现累计seen_count。
        异常：数据库连接/事务错误，不吞掉台账写入失败。
        """
        statement = insert(FaithCase).values(**draft.model_dump(mode="json"))
        # FIRST/resolved 时间不放进 UPDATE；复发时原处置时间仍可回溯。
        statement = statement.on_duplicate_key_update(
            bucket=statement.inserted.bucket, query=statement.inserted.query, strategy=statement.inserted.strategy,
            answer=statement.inserted.answer, reason=statement.inserted.reason, citations=statement.inserted.citations,
            judge_model=statement.inserted.judge_model, status="未解决", resolution=None,
            seen_count=FaithCase.seen_count + 1, last_seen_at=func.current_timestamp(),
        )
        async with self.factory.begin() as session:
            await session.execute(statement)

    async def resolve(self, eval_id: str, status: str, resolution: str | None) -> None:
        """显式人工处置，重新未解决清说明但保留上次处置时间。

        参数：eval_id 非空且最多16字符；status 未解决/已解决/无需解决；后两者resolution必须非空且最多300字符。
        返回：None，更新已存在行；没有对应题号不创建新个案。
        异常：ValueError，输入非法；LookupError，题号不存在；数据库错误。
        """
        note = resolution.strip() if isinstance(resolution, str) else ""
        if (not eval_id.strip() or len(eval_id) > 16 or status not in {"未解决", "已解决", "无需解决"}
                or len(note) > 300 or (status != "未解决" and not note)):
            raise ValueError("台账处置状态或说明不合法")
        async with self.factory.begin() as session:
            row = await session.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id).with_for_update())
            if row is None:
                raise LookupError("台账题号不存在")
            row.status = status
            row.resolution = note if status != "未解决" else None
            if status != "未解决":
                row.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
