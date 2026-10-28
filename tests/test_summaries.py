"""CH07后台摘要契约：独立真实仓储、受控模型延迟及来源校验，语义另用冻结标注评估。"""

import asyncio
import importlib
import json
import logging

import pytest
from sqlalchemy import text

from app.repositories.context import ContextRepository
from tests.test_context_repository import context_db, seed_context, seed_interleaved_context
from tests.workflow.graph_fakes import dependencies
from tests.workflow.service_fakes import collect, service_runtime
from tests.workflow.test_model import sdk_runtime, call_response, text_response


def summary_modules():
    """延迟加载被测模块便于观察缺功能红测。参数：无；返回：生成器/管理器模块；异常：缺模块。"""
    return importlib.import_module("app.workflow.summarization"), importlib.import_module("app.services.summaries")


class ControlledGenerator:
    """只控制模型完成时刻，范围、触发、保存仍由真实管理器实现。"""

    def __init__(self, *, error=None, blocked=False, max_messages=100, no_facts=False):
        """参数：错误/阻塞/模拟输入上限/无事实输出；返回：None；异常：无。"""
        self.error, self.max_messages, self.no_facts = error, max_messages, no_facts
        self.calls, self.started, self.release = [], asyncio.Event(), asyncio.Event()
        if not blocked:
            self.release.set()

    def fits(self, messages, background):
        """模拟模型输入容量，非摘要触发规则。参数：原始消息/背景；返回：能否放下；异常：无。"""
        return len(messages) <= self.max_messages

    async def generate(self, messages, background):
        """记录原始范围并等待受控释放。参数：原始消息/旧摘要背景；返回：合法草稿；异常：测试注入错误。"""
        draft, _ = summary_modules()
        self.calls.append(([row.model_copy(deep=True) for row in messages], background))
        self.started.set()
        await self.release.wait()
        if self.error:
            raise self.error
        return draft.SummaryDraft(content="" if self.no_facts else "用户希望继续处理之前的问题。",
                                  source_msg_ids=[messages[0].id], no_facts=self.no_facts)


def manager(factory, generator, **options):
    """使用演示预算创建管理器。参数：真实工厂、模型替身及控制选项；返回：管理器；异常：模块或配置错误。"""
    _, module = summary_modules()
    config = dependencies(model_context_window=18000, max_output_tokens=2000, max_agent_steps=3).settings
    return module.SummaryManager(ContextRepository(factory), generator, config, **options)


async def wait_idle(worker):
    """等待管理器完成有限任务，超时表示挂死。参数：worker；返回：None；异常：TimeoutError。"""
    async with asyncio.timeout(3):
        while worker.active_count:
            await asyncio.sleep(0.01)


async def seed_pending(factory):
    """创建超过层2预算的冻结区间。参数：factory；返回：上下文/原始轮；异常：数据库错误。"""
    ctx, turns = await seed_context(factory, count=5)
    await ContextRepository(factory).advance_layer1(ctx.conversation_id, expected_summary_upto=0,
                                                   upto_msg_id=turns[3].messages[-1].id)
    return ctx, turns


async def test_duplicate_triggers_freeze_range_and_append_only(context_db):
    """慢模型期间的边界增长不能混进已经送出的源区间。参数：真实离线库；返回：None；异常：断言失败。"""
    ctx, turns = await seed_pending(context_db)
    generator = ControlledGenerator(blocked=True)
    worker = manager(context_db, generator)
    try:
        assert await worker.maybe_schedule(ctx.conversation_id)
        await asyncio.wait_for(generator.started.wait(), 2)
        assert not await worker.maybe_schedule(ctx.conversation_id)
        assert len(generator.calls) == 1
        frozen = generator.calls[0][0][-1].id
        assert frozen == turns[3].messages[-1].id
        await ContextRepository(context_db).advance_layer1(ctx.conversation_id, expected_summary_upto=0,
                                                         upto_msg_id=turns[4].messages[-1].id)
        generator.release.set()
        await wait_idle(worker)
        loaded = await ContextRepository(context_db).load_context(ctx.conversation_id)
        assert loaded["summaries"][0]["upto_msg_id"] == frozen
        assert "完整客服原文" * 100 in generator.calls[0][0][1].content
    finally:
        await worker.close()


async def test_slow_summary_does_not_block_real_chat_or_write_active_checkpoint(tmp_path):
    """摘要一直pending时真实ChatService仍须完成，checkpoint不能被后台改写。参数：tmp_path；返回：None；异常：断言或超时。"""
    generator = ControlledGenerator(blocked=True)
    async with service_runtime(tmp_path, intents=["闲聊"] * 6, model_context_window=18000,
                               max_output_tokens=2000, max_agent_steps=3) as (service, deps, factory):
        worker = manager(factory, generator)
        service.summaries = worker
        try:
            for index in range(6):
                events = await asyncio.wait_for(collect(service, f"第{index}轮" + "请记录这次咨询内容。" * 12), 2)
                assert events[-1].event == "done"
            await asyncio.wait_for(generator.started.wait(), 2)
            assert not generator.release.is_set() and worker.active_count == 1
            key = {"configurable": {"thread_id":"mewhelp:ch05:1"}}
            before = (await service.graph.aget_state(key)).values
            generator.release.set()
            await wait_idle(worker)
            after = (await service.graph.aget_state(key)).values
            assert before == after and len(after["messages"]) == 12
        finally:
            await worker.close()


@pytest.mark.parametrize("failure", [TimeoutError("network-secret"), ValueError("bad-draft")])
async def test_failure_keeps_anchor_and_retries_only_after_backoff(context_db, failure):
    """模型失败不能提交覆盖，立即重试不能忙循环。参数：context_db/failure；返回：None；异常：断言失败。"""
    ctx, _ = await seed_pending(context_db)
    generator = ControlledGenerator(error=failure)
    worker = manager(context_db, generator, retry_seconds=60)
    try:
        assert await worker.maybe_schedule(ctx.conversation_id)
        await wait_idle(worker)
        assert not await worker.maybe_schedule(ctx.conversation_id)
        loaded = await ContextRepository(context_db).load_context(ctx.conversation_id)
        assert loaded["summary_upto_msg_id"] == 0 and loaded["summaries"] == []
        assert len(generator.calls) == 1
    finally:
        await worker.close()


async def test_close_cancels_pending_summary_and_restart_recovers(context_db):
    """关闭中断不前移，重启从数据库锚点恢复而非内存队列。参数：context_db；返回：None；异常：断言失败。"""
    ctx, _ = await seed_pending(context_db)
    blocked = ControlledGenerator(blocked=True)
    worker = manager(context_db, blocked, close_seconds=0.01)
    await worker.maybe_schedule(ctx.conversation_id)
    await asyncio.wait_for(blocked.started.wait(), 2)
    await worker.close()
    assert not await worker.maybe_schedule(ctx.conversation_id)
    assert (await ContextRepository(context_db).load_context(ctx.conversation_id))["summary_upto_msg_id"] == 0
    resumed = manager(context_db, ControlledGenerator())
    try:
        await resumed.recover()
        await wait_idle(resumed)
        assert (await ContextRepository(context_db).load_context(ctx.conversation_id))["summary_upto_msg_id"] > 0
    finally:
        await resumed.close()


async def test_source_batches_split_on_whole_turns_and_never_recompress_old_segments(context_db):
    """大区间应拆完整轮，旧摘要只能作为背景，不得混入source。参数：context_db；返回：None；异常：断言失败。"""
    ctx, _ = await seed_pending(context_db)
    generator = ControlledGenerator(max_messages=2)
    worker = manager(context_db, generator)
    try:
        await worker.maybe_schedule(ctx.conversation_id)
        await wait_idle(worker)
        loaded = await ContextRepository(context_db).load_context(ctx.conversation_id)
        assert len(loaded["summaries"]) == 4
        assert [len(rows) for rows, _ in generator.calls] == [2] * 4
        assert [row.id for rows, _ in generator.calls for row in rows] == list(range(1, 9))
        assert generator.calls[0][1] == ""
        assert all("用户希望继续处理之前的问题。" in background for _, background in generator.calls[1:])
        assert all("完整客服原文" * 100 == rows[1].content for rows, _ in generator.calls)
    finally:
        await worker.close()


@pytest.mark.parametrize("capacity", [4, 6])
async def test_interleaved_summary_batch_keeps_all_originals_in_one_range(context_db, capacity):
    """交错轮次不能拆批漏掉中间新问，放不下整组时不前移。参数：隔离库、模型消息容量；返回：None；异常：断言失败。"""
    ctx, _ = await seed_interleaved_context(context_db)
    await ContextRepository(context_db).advance_layer1(ctx.conversation_id, expected_summary_upto=0, upto_msg_id=6)
    generator = ControlledGenerator(max_messages=capacity)
    worker = manager(context_db, generator)
    try:
        await worker.maybe_schedule(ctx.conversation_id)
        await wait_idle(worker)
        loaded = await ContextRepository(context_db).load_context(ctx.conversation_id)
        if capacity == 6:
            assert loaded["summary_upto_msg_id"] == 6
            assert [(row["from_msg_id"], row["upto_msg_id"]) for row in loaded["summaries"]] == [(1, 6)]
            assert [[row.id for row in rows] for rows, _ in generator.calls] == [[1, 2, 3, 4, 5, 6]]
            assert generator.calls[0][0][0].content == "请保留完整的订单查询原文。" * 50
        else:
            assert loaded["summary_upto_msg_id"] == 0 and loaded["summaries"] == []
            assert not generator.calls
    finally:
        await worker.close()


async def test_oversized_single_turn_skips_without_moving_anchor_or_busy_loop(context_db):
    """单轮放不下不能偷偷切字或越过它总结后面的轮。参数：context_db；返回：None；异常：断言失败。"""
    ctx, _ = await seed_pending(context_db)
    generator = ControlledGenerator(max_messages=1)
    worker = manager(context_db, generator, retry_seconds=60)
    try:
        await worker.maybe_schedule(ctx.conversation_id)
        await wait_idle(worker)
        assert not generator.calls and not await worker.maybe_schedule(ctx.conversation_id)
        loaded = await ContextRepository(context_db).load_context(ctx.conversation_id)
        assert loaded["summaries"] == [] and loaded["summary_upto_msg_id"] == 0
    finally:
        await worker.close()


async def test_explicit_no_facts_advances_without_injecting_meaningless_text(context_db):
    """显式无事实与模型空响应必须分开。参数：context_db；返回：None；异常：断言失败。"""
    ctx, _ = await seed_pending(context_db)
    worker = manager(context_db, ControlledGenerator(no_facts=True))
    try:
        await worker.maybe_schedule(ctx.conversation_id)
        await wait_idle(worker)
        loaded = await ContextRepository(context_db).load_context(ctx.conversation_id)
        assert loaded["summary_upto_msg_id"] == 8 and loaded["summary"] == ""
        assert loaded["summaries"][0]["content"] == ""
    finally:
        await worker.close()


@pytest.mark.parametrize("content,ids,no_facts", [("   ",[1],False),("",[],False),("订单9999",[1],False),
    ("订单1001",[2],False),("订单1001",[88],False),("仍有内容",[1],True),("长"*201,[1],False)],
    ids=["blank", "empty", "invented-number", "assistant-only", "unknown-source", "no-facts-text", "too-long"])
def test_invalid_summary_draft_is_rejected(content, ids, no_facts):
    """只校验可机械证明的来源/数字/长度，不能用此代替语义评估。参数：坏草稿；返回：None；异常：断言失败。"""
    from app.schemas.history import StoredMessage
    module, _ = summary_modules()
    rows = [StoredMessage(id=1, role="user", content="订单1001待处理"),
            StoredMessage(id=2, role="assistant", content="订单1001已批准")]
    with pytest.raises(ValueError):
        draft = module.SummaryDraft(content=content, source_msg_ids=ids, no_facts=no_facts)
        module.validate_summary(draft, rows)


async def test_summary_sdk_input_budget_and_usage_are_independent(sdk_runtime, caplog):
    """实际SDK输入保留原文，后台账本独立且记录真实usage。参数：网络边界/日志；返回：None；异常：断言失败。"""
    from app.schemas.history import StoredMessage
    from app.core.model_budget import BudgetLedger, WorkflowBudgetExceeded
    module, _ = summary_modules()
    config = dependencies(model_context_window=18000, max_output_tokens=2000, max_agent_steps=3).settings
    rows = [StoredMessage(id=1, role="user", content="订单A-1001没到，请查物流。"),
            StoredMessage(id=2, role="assistant", content="完整正文\n  " * 100)]
    generator = module.SummaryGenerator(config, model_factory=sdk_runtime.factory)
    before = BudgetLedger.from_settings(config).snapshot()
    sdk_runtime.responses.append(call_response("submit_summary", json.dumps({
        "content":"用户自报订单A-1001未收到，要求查询物流。", "source_msg_ids":[1], "no_facts":False}), usage=True))
    with caplog.at_level(logging.INFO):
        draft = await generator.generate(rows, "旧背景来自另一个批次")
    request = sdk_runtime.requests[0]
    data = json.loads(request["messages"][1]["content"])
    assert data["source"] == [row.model_dump() for row in rows]
    assert data["background"] == "旧背景来自另一个批次"
    assert request["max_completion_tokens"] == 2000
    assert [tool["function"]["name"] for tool in request["tools"]] == ["submit_summary"]
    assert draft.content.startswith("用户自报")
    usage = next(json.loads(r.message.removeprefix("summary usage ")) for r in caplog.records if r.message.startswith("summary usage "))
    assert usage["actual_tokens"] == 112 and usage["max_model_calls"] == 2 and usage["model_calls"] == 1
    assert before["model_calls"] == 0
    too_large = [rows[0].model_copy(update={"content":"超长原文"*10000})]
    assert not generator.fits(too_large, "")
    with pytest.raises(WorkflowBudgetExceeded):
        await generator.generate(too_large, "")
    assert len(sdk_runtime.requests) == 1


async def test_summary_rejects_text_json_and_background_numbers(sdk_runtime):
    """正文JSON及只有背景包含的编号都不算合法摘要。参数：真实SDK边界；返回：None；异常：断言失败。"""
    from app.schemas.history import StoredMessage
    module, _ = summary_modules()
    generator = module.SummaryGenerator(dependencies().settings, model_factory=sdk_runtime.factory)
    rows = [StoredMessage(id=1, role="user", content="想问粮食怎么保存")]
    output = json.dumps({"content":"订单9999需要退货", "source_msg_ids":[1], "no_facts":False})
    sdk_runtime.responses.extend([text_response(output), text_response(output),
        call_response("submit_summary", output), call_response("submit_summary", output)])
    for _ in range(2):
        with pytest.raises(ValueError):
            await generator.generate(rows, "用户之前询问订单9999")
    assert len(sdk_runtime.requests) == 4


async def test_timeout_is_bounded_and_logs_do_not_leak_failure_text(context_db, caplog):
    """挂死的模型须超时收回；失败仅记录类型而非凭证文本。参数：真实库/日志；返回：None；异常：断言失败。"""
    ctx, _ = await seed_pending(context_db)
    worker = manager(context_db, ControlledGenerator(blocked=True), timeout_seconds=0.03)
    try:
        with caplog.at_level(logging.INFO):
            await worker.maybe_schedule(ctx.conversation_id)
            await wait_idle(worker)
        assert any("summary trigger" in row.message for row in caplog.records)
        assert any("summary start" in row.message for row in caplog.records)
        assert any("type=TimeoutError" in row.message and "elapsed_ms=" in row.message for row in caplog.records)
        assert (await ContextRepository(context_db).load_context(ctx.conversation_id))["summary_upto_msg_id"] == 0
    finally:
        await worker.close()


async def test_recovery_drains_more_conversations_than_queue_capacity(context_db):
    """启动扫描须分页服从容量，不能漏掉首批以后的会话。参数：真实库；返回：None；异常：断言失败。"""
    ids = []
    for index in range(4):
        ctx, turns = await seed_context(context_db, session_id=f"queue-{index}", count=5)
        ids.append(ctx.conversation_id)
        await ContextRepository(context_db).advance_layer1(ctx.conversation_id, expected_summary_upto=0,
                                                         upto_msg_id=turns[3].messages[-1].id)
    generator = ControlledGenerator(blocked=True)
    worker = manager(context_db, generator, concurrency=1, max_pending=2)
    try:
        await worker.recover()
        await asyncio.wait_for(generator.started.wait(), 2)
        assert len(generator.calls) == 1
        assert not await worker.maybe_schedule(ids[-1])
        generator.release.set()
        await wait_idle(worker)
        assert len(generator.calls) == 4
        for identifier in ids:
            assert (await ContextRepository(context_db).load_context(identifier))["summary_upto_msg_id"] > 0
    finally:
        await worker.close()


async def test_within_budget_never_calls_summary_even_with_old_messages(context_db):
    """有层2不等于超量；默认配置不能按条数触发模型。参数：真实库；返回：None；异常：断言失败。"""
    _, module = summary_modules()
    ctx, _ = await seed_pending(context_db)
    generator = ControlledGenerator()
    worker = module.SummaryManager(ContextRepository(context_db), generator, dependencies().settings)
    try:
        await worker.recover()
        await wait_idle(worker)
        assert not generator.calls
        assert (await ContextRepository(context_db).load_context(ctx.conversation_id))["summaries"] == []
    finally:
        await worker.close()


def test_existing_checkpoint_budget_before_summary_stage_can_resume():
    """新增后台阶段不能使旧会话账本失效。参数：无；返回：None；异常：断言失败。"""
    from app.core.model_budget import BudgetLedger
    old = BudgetLedger().snapshot()
    old["output_limits"].pop("summary")
    restored = BudgetLedger.from_snapshot(old)
    assert restored.output_limit("answer") == old["output_limits"]["answer"]
    assert restored.snapshot()["records"] == old["records"]


def test_failed_tool_is_failure_source_but_never_numeric_fact_source():
    """可记住查询失败，不能借失败data中的编号金额当已核实事实。参数：无；返回：None；异常：断言失败。"""
    from app.schemas.history import StoredMessage
    module, _ = summary_modules()
    rows = [StoredMessage(id=1, role="user", content="订单1001请再查一下"),
            StoredMessage(id=2, role="tool", content='{"ok":false,"data":{"amount":999},"error":"timeout"}')]
    module.validate_summary(module.SummaryDraft(content="用户查询订单1001，查询失败待处理。", source_msg_ids=[1,2], no_facts=False), rows)
    with pytest.raises(ValueError):
        module.validate_summary(module.SummaryDraft(content="查询失败，金额999。", source_msg_ids=[1,2], no_facts=False), rows)


async def test_invalid_summary_source_gets_one_bounded_repair(sdk_runtime):
    """来源校验失败后只补救一次，完整本段与旧背景保持隔离。参数：SDK网络边界；返回None；异常：断言失败。"""
    from app.schemas.history import StoredMessage
    module, _ = summary_modules()
    rows = [StoredMessage(id=31, role="user", content="这笔订单先不催，只查询配送情况"),
            StoredMessage(id=32, role="assistant", content="订单9876已发货")]
    bad = {"content":"订单9876已发货", "source_msg_ids":[31,32], "no_facts":False}
    good = {"content":"用户暂不催单，仅要求查询配送情况。", "source_msg_ids":[31], "no_facts":False}
    sdk_runtime.responses.extend([call_response("submit_summary", json.dumps(item)) for item in [bad,good]])
    generator = module.SummaryGenerator(dependencies().settings, model_factory=sdk_runtime.factory)
    draft = await generator.generate(rows, "此前提过订单9876")
    assert draft.content == good["content"]
    assert len(sdk_runtime.requests) == 2
    first, second = sdk_runtime.requests
    assert second["messages"][:2] == first["messages"]
    assert "9876" not in second["messages"][-1]["content"]
    assert "31" in second["messages"][-1]["content"]
    assert "32" not in second["messages"][-1]["content"]


def test_summary_repair_supplies_only_numbers_from_current_trusted_sources():
    """补救必须明确本批可用编号，避免从旧背景或助手猜号补全。参数无；返回None；异常断言失败。"""
    from app.schemas.history import StoredMessage
    module, _ = summary_modules()
    generator = module.SummaryGenerator(dependencies().settings)
    rows = [StoredMessage(id=1, role="user", content="这笔订单什么状态？"),
            StoredMessage(id=2, role="assistant", content="订单9876金额123")]
    repair = generator._repair(rows).content
    constraint = json.loads(repair.split("\n")[-1])
    assert constraint == {"source_msg_ids":[1], "numeric_facts":[]}
    rows[0] = rows[0].model_copy(update={"content":"另查AB-3456，电话13900139000"})
    constraint = json.loads(generator._repair(rows).content.split("\n")[-1])
    assert constraint["numeric_facts"] == ["13900139000","AB-3456"]


async def test_repeated_bad_summary_does_not_retry_without_limit(sdk_runtime):
    """两次来源都不合法即失败，不让生成器无限刷答案。参数：SDK边界；返回None；异常：断言失败。"""
    from app.schemas.history import StoredMessage
    module, _ = summary_modules()
    bad = {"content":"订单9999需要退款", "source_msg_ids":[1], "no_facts":False}
    sdk_runtime.responses.extend([call_response("submit_summary", json.dumps(bad)) for _ in range(2)])
    generator = module.SummaryGenerator(dependencies().settings, model_factory=sdk_runtime.factory)
    with pytest.raises(ValueError):
        await generator.generate([StoredMessage(id=1, role="user", content="想问保存方法")], "订单9999")
    assert len(sdk_runtime.requests) == 2
