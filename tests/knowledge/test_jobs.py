"""知识导入、挖掘、恢复与同步任务编排测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.knowledge.chunking import ChunkingError
from app.knowledge.jobs import KnowledgeJobs, _batch_from_turns
from app.knowledge.llm import KnowledgeModelError
from app.knowledge.lifecycle import KnowledgeLifecycle
from app.knowledge.prompts import RELATION_PROMPT_VERSION
from app.knowledge.schemas import (
    ChunkSnapshot,
    CompletedTurn,
    ExtractedQA,
    KnowledgeText,
    RelationDecision,
    RelationPolicy,
)
from app.knowledge.sync import VectorSynchronizer
from app.knowledge.text import content_fingerprint
from tests.knowledge.fakes import (
    FakeEncoder,
    MemoryKnowledgeStore,
    MemoryVectorIndex,
    ScriptedKnowledgeModel,
)

POLICY = RelationPolicy(
    min_score=0.0,
    embedding_revision="fake-encoder-v1",
    prompt_version=RELATION_PROMPT_VERSION,
    calibration_digest="test-only",
)
_TEMP_BASE = Path(".pytest-tmp")


def _turn(identifier: int, conversation_id: int, question: str, answer: str) -> CompletedTurn:
    """构造完成轮次。参数：ID/会话/问答；返回：轮次；异常：校验失败。"""
    return CompletedTurn(
        message_id=identifier,
        conversation_id=conversation_id,
        turn_id=f"turn-{identifier}",
        question=question,
        answer=answer,
    )


def _qa(source_ids: list[int], question: str, answer: str) -> ExtractedQA:
    """构造候选知识。参数：来源/问答；返回：QA；异常：校验失败。"""
    return ExtractedQA(
        source_ids=source_ids,
        question=question,
        answer=answer,
        category="物流",
    )


def _decision(left: str, right: str) -> RelationDecision:
    """构造无关关系。参数：引用；返回：决策；异常：校验失败。"""
    return RelationDecision(
        left_ref=left,
        right_ref=right,
        relation="unrelated",
        evidence_refs=[left, right],
        reason="不同主题",
    )


def _jobs(
    turns: list[CompletedTurn],
    qas: list[ExtractedQA],
    *,
    decisions: list[RelationDecision] | None = None,
    snapshots: list[ChunkSnapshot] | None = None,
):
    """组装内存任务编排。参数：轮次/QA/关系/初始快照；返回：jobs 等；异常：无。"""
    store = MemoryKnowledgeStore(turns=turns, snapshots=snapshots or [])
    encoder = FakeEncoder()
    index = MemoryVectorIndex()
    model = ScriptedKnowledgeModel(
        extract_qas=qas, relation_decisions=decisions or []
    )
    lifecycle = KnowledgeLifecycle(encoder, model, POLICY)
    synchronizer = VectorSynchronizer(store, encoder, index)
    jobs = KnowledgeJobs(store, encoder, index, model, lifecycle, synchronizer)
    return jobs, store, index, model


async def test_mine_once_publishes_and_syncs() -> None:
    """单轮挖掘产出知识、暂存和向量。参数：无；返回：None；异常：无。"""
    jobs, store, index, _ = _jobs(
        [_turn(1, 1, "邮费怎么算", "未满99元收6元。")],
        [_qa([1], "邮费怎么算", "未满99元收6元。")],
    )
    report = await jobs.mine_once()
    assert report.batch_count == 1
    assert report.extracted == 1
    assert report.kept == 1
    assert report.sync.marked_done == 1
    assert index.ids
    async with store.writer() as writer:
        frame = await writer.frame()
    assert any(row.source_key == "qa:1" and row.is_active for row in frame.chunks)
    assert next(row for row in frame.staged if row.id == 1).status == "kept"


async def test_batch_preserves_conversation_boundaries() -> None:
    """不同会话不混入同一抽取批次。参数：无；返回：None；异常：无。"""
    jobs, store, _, model = _jobs(
        [
            _turn(1, 1, "邮费怎么算", "未满99元收6元。"),
            _turn(2, 2, "退货怎么申请", "在订单页申请退款。"),
        ],
        [
            _qa([1], "邮费怎么算", "未满99元收6元。"),
            _qa([2], "退货怎么申请", "在订单页申请退款。"),
        ],
        decisions=[_decision("staged:1", "staged:2")],
    )
    await jobs.mine_once()
    batches = list(store._batches.values())
    assert len(batches) == 1
    batch = batches[0]
    assert {turn.conversation_id for turn in batch.turns} == {1, 2}
    assert [turn.message_id for turn in batch.turns] == [1, 2]


async def test_empty_batch_done_not_recalled() -> None:
    """零 QA 批次完成，下一轮不重复领取。参数：无；返回：None；异常：无。"""
    jobs, store, _, model = _jobs(
        [_turn(1, 1, "你好", "您好。")],
        [],
    )
    first = await jobs.mine_once()
    assert first.batch_count == 1 and first.extracted == 0
    second = await jobs.mine_once()
    assert second.batch_count == 0
    assert list(store._batch_status.values()) == ["done"]
    assert len(model.batches) == 1


async def test_pending_batch_reuses_source_ids() -> None:
    """恢复 pending 批次不重复领取来源。参数：无；返回：None；异常：无。"""
    turn = _turn(1, 1, "邮费怎么算", "未满99元收6元。")
    jobs, store, _, model = _jobs(
        [turn],
        [_qa([1], "邮费怎么算", "未满99元收6元。")],
    )
    batch = _batch_from_turns([turn], turn.message_id)
    async with store.writer() as writer:
        await writer.reserve_batch(batch)
    report = await jobs.recover_pending()
    assert report.batch_count == 1
    async with store.writer() as writer:
        assert await writer.pending_batches() == []
    assert len(model.batches) == 1


async def test_import_documents_publishes_and_syncs() -> None:
    """清单导入切分、发布并同步。参数：无；返回：None；异常：无。"""
    _TEMP_BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ch03-jobs-", dir=_TEMP_BASE) as temp:
        root = Path(temp)
        (root / "doc.md").write_text(
            "# 甲\n\n## 乙\n\n未满99元收6元。\n", encoding="utf-8"
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "documents": [
                        {"path": "doc.md", "content_type": "faq", "key_sections": []}
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        jobs, store, index, _ = _jobs([], [])
        report = await jobs.import_documents(root / "manifest.json")
        async with store.writer() as writer:
            frame = await writer.frame()
    assert report.sync.marked_done == 1
    assert any(row.source_key == "doc:doc.md" and row.is_active for row in frame.chunks)


async def test_late_committed_small_id_is_picked_up() -> None:
    """已分配高 ID 后迟提交的小 ID 仍从 0 补扫发现。参数：无；返回：None；异常：无。"""
    jobs, store, _, model = _jobs(
        [_turn(2, 2, "高 ID 问题", "高 ID 回答。")],
        [_qa([2], "高 ID 问题", "高 ID 回答。")],
    )
    await jobs.mine_once()
    store._turns.append(_turn(1, 1, "迟到的低 ID 问题", "迟到回答。"))
    model.extract_qas = [_qa([1], "迟到的低 ID 问题", "迟到回答。")]
    model.relation_decisions = [_decision("staged:2", "chunk:1")]
    report = await jobs.mine_once()
    assert report.batch_count == 1
    assert report.extracted == 1
    assert any(batch.source_ids == [1] for batch in store._batches.values())


async def test_model_failure_keeps_batch_recoverable() -> None:
    """抽取失败不推进来源，恢复后可重跑完成批次。参数：无；返回：None；异常：无。"""
    turn = _turn(1, 1, "邮费怎么算", "未满99元收6元。")
    jobs, store, _, model = _jobs([turn], [])
    model.extract_error = KnowledgeModelError("模拟抽取失败")
    try:
        await jobs.mine_once()
    except KnowledgeModelError:
        pass
    else:
        raise AssertionError("抽取失败应抛出")
    async with store.writer() as writer:
        assert [batch.batch_no for batch in await writer.pending_batches()]
    model.extract_error = None
    model.extract_qas = [_qa([1], "邮费怎么算", "未满99元收6元。")]
    report = await jobs.recover_pending()
    assert report.batch_count == 1
    assert report.kept == 1


async def test_import_documents_is_atomic_on_chunk_failure() -> None:
    """后续文档切分失败时不发布前面的文档。参数：无；返回：None；异常：无。"""
    _TEMP_BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ch03-atomic-", dir=_TEMP_BASE) as temp:
        root = Path(temp)
        (root / "good.md").write_text("# 甲\n\n## 乙\n\n正常知识。\n", encoding="utf-8")
        (root / "bad.md").write_text("# 甲\n\n## 乙\n\n" + "长" * 1601 + "\n", encoding="utf-8")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "documents": [
                        {"path": "good.md", "content_type": "faq", "key_sections": []},
                        {"path": "bad.md", "content_type": "faq", "key_sections": []},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        jobs, store, _, _ = _jobs([], [])
        with pytest.raises(ChunkingError):
            await jobs.import_documents(root / "manifest.json")
        async with store.writer() as writer:
            assert (await writer.frame()).chunks == []


async def test_over_turn_input_budget_is_rejected_before_batch() -> None:
    """单轮超过输入预算时在预留批次前失败。参数：无；返回：None；异常：无。"""
    turn = _turn(1, 1, "长" * 9000, "答" * 9000)
    jobs, store, _, model = _jobs([turn], [])
    with pytest.raises(ValueError, match="单轮"):
        await jobs.mine_once()
    assert store._batches == {}
    assert model.batches == []


async def test_sync_pending_marks_done() -> None:
    """独立同步命令补齐 pending。参数：无；返回：None；异常：无。"""
    text = KnowledgeText(category="分类", questions="问法", answer="答案。")
    snapshot = ChunkSnapshot(
        id=1,
        source_key="doc:a.md",
        ordinal=0,
        text=text,
        section_path="甲",
        content_type="faq",
        is_key_clause=False,
        content_hash=content_fingerprint(text),
        embedding_revision="fake-encoder-v1",
        is_active=True,
        vectorize_status="pending",
    )
    jobs, store, index, _ = _jobs([], [], snapshots=[snapshot])
    report = await jobs.sync_pending()
    assert report.marked_done == 1
    assert (await store.get_chunks([1]))[0].vectorize_status == "done"
    assert index.ids == {1}


async def test_complete_snapshot_reserves_all_201_turns_before_extract() -> None:
    """201轮必须先全预留再调用模型。参数：无；返回：None；异常：无。"""
    turns = [_turn(i, i, "你好", "您好。") for i in range(1, 202)]
    jobs, store, _, model = _jobs(turns, [])
    original = model.extract
    async def checked(batch):
        """验证预留数量。参数：批次；返回：QA；异常：断言失败。"""
        assert sum(len(b.source_ids) for b in store._batches.values()) == 201
        return await original(batch)
    model.extract = checked
    await jobs.mine_once()
    assert sum(len(b.source_ids) for b in store._batches.values()) == 201
    assert set(store._batch_status.values()) == {"done"}


async def test_recovery_completes_old_done_half_snapshot_without_new_upper() -> None:
    """恢复旧done半快照时补完upper内，不能领取upper后的来源。参数：无；返回：None；异常：无。"""
    turns = [_turn(i, i, "你好", "您好。") for i in range(1, 204)]
    jobs, store, _, model = _jobs(turns, [])
    batch = _batch_from_turns(turns[:20], 201)
    async with store.writer() as writer:
        await writer.reserve_batch(batch)
        await writer.complete_batch(batch.batch_no, [])
    await jobs.recover_pending()
    assigned = {i for batch in store._batches.values() for i in batch.source_ids}
    assert assigned == set(range(1, 202))
    assert all(batch.batch_no != b.batch_no for b in model.batches)


async def test_reserved_source_change_is_rejected() -> None:
    """预留后修改原文时hash核验必须拒绝。参数：无；返回：None；异常：无。"""
    from app.knowledge.errors import SnapshotChanged
    turn = _turn(1, 1, "邮费怎么算", "未满99元收6元。")
    jobs, store, _, model = _jobs([turn], [])
    async with store.writer() as writer:
        await writer.reserve_batch(_batch_from_turns([turn], 1))
    store._turns[0] = turn.model_copy(update={"answer": "未满99元收600元。"})
    with pytest.raises(SnapshotChanged, match="指纹|来源"):
        await jobs.recover_pending()
    assert model.batches == []


async def test_complete_batch_validates_all_qas_before_done() -> None:
    """第二条候选伪造时不能部分写入或done。参数：无；返回：None；异常：无。"""
    from app.knowledge.grounding import GroundingError
    turn = _turn(1, 1, "邮费怎么算", "未满99元收6元。")
    _, store, _, _ = _jobs([turn], [])
    batch = _batch_from_turns([turn], 1)
    async with store.writer() as writer:
        await writer.reserve_batch(batch)
        with pytest.raises(GroundingError):
            await writer.complete_batch(batch.batch_no, [_qa([1], turn.question, turn.answer), _qa([1], turn.question, "未满99元收600元。")])
        assert (await writer.frame()).staged == []
        assert len(await writer.pending_batches()) == 1


async def test_evidence_persisted_and_disposition_preserves_it() -> None:
    """暂存原句审计经过发布处置仍可重读。参数：无；返回：None；异常：无。"""
    jobs, store, _, _ = _jobs([_turn(1, 1, "邮费怎么算", "未满99元收6元。")], [_qa([1], "邮费怎么算", "未满99元收6元。")])
    await jobs.mine_once()
    async with store.writer() as writer:
        row = (await writer.frame()).staged[0]
    assert row.source_evidence[0]["message_id"] == 1
    assert row.source_evidence[0]["quote"] == "未满99元收6元。"
    assert row.source_evidence[0]["verification_version"]
    assert row.source_evidence[0]["end"] > row.source_evidence[0]["start"]


async def test_pending_old_prompt_keeps_identity_uses_new_prompt() -> None:
    """旧pending恢复稳定批次身份同时执行当前Prompt。参数：无；返回：None；异常：无。"""
    from app.knowledge.prompts import EXTRACTION_PROMPT_VERSION
    turn = _turn(1, 1, "邮费怎么算", "未满99元收6元。")
    jobs, store, _, model = _jobs([turn], [])
    batch = _batch_from_turns([turn], 1).model_copy(update={"prompt_version": "ch03-extraction-v1"})
    async with store.writer() as writer:
        await writer.reserve_batch(batch)
    await jobs.recover_pending()
    assert model.batches[0].batch_no == batch.batch_no
    assert model.batches[0].prompt_version == EXTRACTION_PROMPT_VERSION


async def test_failure_resumes_complete_upper_then_reconciles_once() -> None:
    """第二批失败后恢复全upper，只在全部抽取完成才统一去重。参数：无；返回：None；异常：无。"""
    jobs, store, _, model = _jobs([_turn(i, i, "你好", "您好。") for i in range(1, 202)], [])
    calls = []
    original = model.extract
    async def fail_second(batch):
        """模拟第二批失败。参数：batch；返回：QA；异常：KnowledgeModelError。"""
        if len(model.batches) == 1:
            raise KnowledgeModelError("第二批暂失败")
        return await original(batch)
    async def reconcile(writer):
        """校验所有批次完成才去重。参数：writer；返回：None；异常：断言失败。"""
        assert set(store._batch_status.values()) == {"done"}
        calls.append("reconcile")
    model.extract = fail_second
    jobs.lifecycle.reconcile_staging = reconcile
    with pytest.raises(KnowledgeModelError):
        await jobs.mine_once()
    assert sum(len(batch.source_ids) for batch in store._batches.values()) == 201
    assert calls == []
    store._turns.append(_turn(202, 202, "新轮", "新答。"))
    model.extract = original
    await jobs.recover_pending()
    assert calls == ["reconcile"]
    assert 202 not in {i for batch in store._batches.values() for i in batch.source_ids}


async def test_legacy_staging_missing_source_blocks_publish() -> None:
    """旧无源kept阻发布并保留历史记录。参数：无；返回：None；异常：无。"""
    from app.knowledge.schemas import StagedQA, Publication
    from app.knowledge.grounding import GroundingError
    row = StagedQA(id=1, source_ids=[99], question="旧问题", answer="旧回答。", category="分类", status="kept")
    store = MemoryKnowledgeStore(staged=[row])
    async with store.writer() as writer:
        frame = await writer.frame()
        with pytest.raises(GroundingError):
            await writer.publish(Publication(expected_token=frame.token, embedding_revision="fake-encoder-v1"))
        assert (await writer.frame()).staged == [row]


async def test_mine_once_resumes_pending_without_new_upper() -> None:
    """完整挖掘入口恢复中断时也不能领旧upper之后来源。参数：无；返回：None；异常：无。"""
    turns = [_turn(i, i, "你好", "您好。") for i in range(1, 4)]
    jobs, store, _, _ = _jobs(turns, [])
    async with store.writer() as writer:
        await writer.reserve_batch(_batch_from_turns(turns[:1], 2))
    await jobs.mine_once()
    assert {i for batch in store._batches.values() for i in batch.source_ids} == {1, 2}


async def test_done_batch_idempotent_read_rechecks_source_hash() -> None:
    """done重读也核来源，变更拒绝且保留原暂存审计。参数：无；返回：None；异常：无。"""
    from app.knowledge.errors import SnapshotChanged
    turn = _turn(1, 1, "邮费怎么算", "运费6元。")
    _, store, _, model = _jobs([turn], [])
    batch = _batch_from_turns([turn], 1)
    async with store.writer() as writer:
        await writer.reserve_batch(batch)
        before = await writer.complete_batch(batch.batch_no, [_qa([1], turn.question, turn.answer)])
        assert await writer.complete_batch(batch.batch_no, []) == before
        store._turns[0] = turn.model_copy(update={"answer": "运费600元。"})
        with pytest.raises(SnapshotChanged, match="指纹"):
            await writer.complete_batch(batch.batch_no, [])
        assert (await writer.frame()).staged == before
    assert store._batch_status[batch.batch_no] == "done"
    assert model.batches == []
