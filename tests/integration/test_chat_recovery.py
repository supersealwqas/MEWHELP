"""在真实 MySQL 和应用 lifespan 中验证重启恢复，不以函数替身冒充恢复流程。"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from langchain_core.messages import ToolMessage
from app.models.customer_service import Conversation, Message, Ticket
from app.schemas.tools import ToolContext
from app.schemas.chat import ChatRequest
from tests.chat_fakes import ScriptedModel

pytestmark = pytest.mark.mysql


@pytest.mark.parametrize("with_result", [True, False])
async def test_new_application_recovers_committed_facts_without_reexecution(database, repositories, monkeypatch, with_result):
    """新 engine 恢复真实已提交事实，未配对申请不得回灌。参数：测试库、仓库、替换器、结果开关；返回：None；异常：断言失败。"""
    from app import main
    chat, support = repositories
    ctx = await chat.start_turn(ToolContext(session_id="restart", user_id="demo-user", turn_id="old-turn",
        tool_call_id="original-call", user_message="请建立工单"), "请建立工单")
    await chat.save_tool_calls(ctx, [{"name": "create_ticket", "id": "original-call",
        "args": {"description": "物流延迟", "ticket_type": "咨询"}}])
    result = await support.create_ticket(ctx, "物流延迟", "咨询") if with_result else None
    url = database[0].url
    await database[0].dispose()
    new_engine = create_async_engine(url)
    new_factory = async_sessionmaker(new_engine, expire_on_commit=False)
    monkeypatch.setattr(main, "create_database", lambda: (new_engine, new_factory))
    application = main.create_app(runtime_factory=main.database_runtime)
    model = ScriptedModel(chunks=["已保存的处理状态"])
    async with application.router.lifespan_context(application):
        service = application.state.chat_service
        service.model = model
        assert not model.inputs
        async with new_factory() as session:
            rows = (await session.scalars(select(Message).where(Message.turn_id == "old-turn"))).all()
            assert all(row.turn_status == "cancelled" for row in rows)
            if with_result:
                assert (await session.get(Conversation, ctx.conversation_id)).status == "已转人工"
                assert await session.get(Ticket, result.data["ticket_no"]) is not None
        events = [e async for e in service.stream_turn(ChatRequest(session_id="restart", message="刚才提交了吗"))]
        assert events[-1].event == "done"
        facts = [m for m in model.inputs[0][1] if isinstance(m, ToolMessage)]
        assert len(facts) == (1 if with_result else 0)
        if with_result:
            assert result.data["ticket_no"] in facts[0].content
        assert any("未完成" in m.content for m in model.inputs[0][1])
        assert await service.chat.recover_interrupted_turns() == 0


async def test_history_read_preserves_other_running_turn(repositories):
    """正常历史查询不能取消另一会话。参数：仓库；返回：None；异常：断言失败。"""
    chat, _ = repositories
    a = await chat.start_turn(ToolContext(session_id="a", user_id="u", turn_id="a-turn", user_message="a"), "a")
    b = await chat.start_turn(ToolContext(session_id="b", user_id="u", turn_id="b-turn", user_message="b"), "b")
    await chat.load_turns(a.conversation_id)
    turns = await chat.load_turns(b.conversation_id)
    assert len(turns) == 1 and turns[0].turn_status == "running" and len(turns[0].messages) == 1
