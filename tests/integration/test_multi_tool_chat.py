"""在已有 MySQL 测试库核验多工具持久化、工单防重和取消后的事实恢复。

仅模型决策使用脚本替身，聊天服务、工具执行器及两个仓库均运行真实代码。
每例只创建并清理自己的唯一会话，不建表、不清库，也不恢复其他会话。
"""

import json
from pathlib import Path
from uuid import uuid4

import pytest
import httpx
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.memory import assemble_turns
from app.db.session import create_database
from app.models.customer_service import Conversation, Message, Ticket
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.schemas.chat import ChatRequest
from app.schemas.tools import ToolResult
from app.services.chat import ChatService

pytestmark = pytest.mark.mysql


async def _row_totals(session: AsyncSession) -> dict[str, int]:
    """读取三张测试涉及表的总行数。参数：session 只读会话；返回：表名与总数；异常：数据库读取错误。"""
    return {
        model.__tablename__: int(await session.scalar(select(func.count()).select_from(model)))
        for model in (Conversation, Message, Ticket)
    }


@pytest.fixture
async def isolated_multi_tool_mysql(request: pytest.FixtureRequest):
    """提供唯一会话及真实服务工厂，并精确清理本例数据。

    参数：request 当前测试节点，用于标记前后行数记录。
    返回：异步产出会话工厂、唯一 session_id 和聊天服务构造函数。
    异常：ValueError 表示数据库目标错误或会话标识碰撞；数据库错误、行数校验失败向外传播。
    """
    config = Settings(
        _env_file=Path(__file__).resolve().parents[2] / ".env",
        token_budget=12000,
        request_token_budget=16000,
        tool_max_calls_per_turn=4,
        tool_timeout_seconds=5,
        tool_retry_delay_seconds=0,
    )
    if config.mysql_test_database != "mewhelp_test":
        raise ValueError("多工具集成测试只允许 mewhelp_test")
    engine, factory = create_database(config, database="mewhelp_test")
    session_id = f"multi-tool-mysql-{uuid4().hex}"
    before = None
    try:
        async with factory() as session:
            if await session.scalar(text("SELECT DATABASE()")) != "mewhelp_test":
                raise ValueError("实际连接不是 mewhelp_test，拒绝写入")
            if await session.scalar(select(Conversation.id).where(Conversation.session_id == session_id)) is not None:
                raise ValueError("测试会话标识已存在，拒绝操作原会话")
            # 只有确认本例标识不存在后才允许 finally 清理；不触发父目录的清库夹具。
            before = await _row_totals(session)

        def build_service(decisions):
            """构造真实新图服务和仓库。参数：decisions有限决策脚本；返回：服务/仓储/模型；异常：构造错误。"""
            from langgraph.checkpoint.memory import InMemorySaver
            from app.schemas.workflow import AgentDecision
            from app.tools.executor import ToolExecutor
            from app.tools.registry import build_agent_registry
            from app.workflow.graph import WorkflowDependencies, build_workflow
            from tests.workflow.fakes import ScriptedWorkflowModel
            from tests.workflow.graph_fakes import StrictSearch, StrictExpander
            chat = ChatRepository(factory)
            model = ScriptedWorkflowModel(intents=["物流"], decisions=[*decisions, AgentDecision(kind="answer")],
                                          answers=[["已按工具结果处理。"]])
            executor = ToolExecutor(build_agent_registry(), chat, SupportRepository(factory))
            deps = WorkflowDependencies(model, StrictSearch(), executor, chat, config, StrictExpander())
            # 本例只验证业务MySQL配对；官方文件SQLite重开另由workflow专项与Task8覆盖。
            graph = build_workflow(deps, checkpointer=InMemorySaver())
            return ChatService(chat, graph, executor, config), chat, model

        # 不把含连接及模型凭证的 Settings 直接交给 pytest，避免失败时展示配置详情。
        yield factory, session_id, build_service
    finally:
        try:
            if before is not None:
                async with factory.begin() as session:
                    if await session.scalar(text("SELECT DATABASE()")) != "mewhelp_test":
                        raise ValueError("清理时目标库不匹配，拒绝删除")
                    with_test = await _row_totals(session)
                    conversation_id = await session.scalar(
                        select(Conversation.id).where(Conversation.session_id == session_id)
                    )
                    if conversation_id is not None:
                        # 按本例唯一会话逐表删除，先删引用记录，绝不执行无条件删除。
                        await session.execute(delete(Ticket).where(Ticket.conversation_id == conversation_id))
                        await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
                        await session.execute(delete(Conversation).where(
                            Conversation.id == conversation_id, Conversation.session_id == session_id
                        ))
                async with factory() as session:
                    after = await _row_totals(session)
                print(f"{request.node.name}: rows_before={before}; rows_with_test={with_test}; rows_after={after}")
                assert after == before, "本例清理后总行数与测试前不同，需核查并发写入或残留数据"
        finally:
            await engine.dispose()


async def _read_records(factory, session_id: str) -> tuple[Conversation, list[Message], list[Ticket]]:
    """读取本例真实落库记录。参数：factory 会话工厂、session_id 唯一标识；返回：会话、消息、工单；异常：数据库错误或无会话。"""
    async with factory() as session:
        conversation = await session.scalar(select(Conversation).where(Conversation.session_id == session_id))
        assert conversation is not None
        messages = list((await session.scalars(select(Message).where(
            Message.conversation_id == conversation.id
        ).order_by(Message.id))).all())
        tickets = list((await session.scalars(select(Ticket).where(
            Ticket.conversation_id == conversation.id
        ))).all())
    return conversation, messages, tickets


async def test_mysql_order_logistics_persist_and_restore_in_order(isolated_multi_tool_mysql):
    """多次决策必须落库两组完整工具事实。参数：isolated_multi_tool_mysql 隔离资源；返回：None；异常：断言失败。"""
    factory, session_id, build_service = isolated_multi_tool_mysql
    from app.schemas.workflow import AgentDecision
    service, chat, model = build_service([
        AgentDecision(kind="tools", tool_calls=[{"id": "order", "name": "query_order", "args": {"order_id": "1001"}}]),
        # 本例模拟取用订单工具返回的当前单号；历史 MOCK 前缀兼容在专门用例中验证。
        AgentDecision(kind="tools", tool_calls=[{"id": "shipping", "name": "query_logistics", "args": {"tracking_no": "SF213502378238"}}]),
    ])
    events = [event async for event in service.stream_turn(
        ChatRequest(session_id=session_id, message="订单1001的物流到哪了")
    )]
    assert events[-1].event == "done" and sum(event.event == "done" for event in events) == 1
    conversation, rows, tickets = await _read_records(factory, session_id)
    assert not tickets
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert all(row.turn_status == "completed" for row in rows)
    assert [rows[1].tool_calls[0]["name"], rows[3].tool_calls[0]["name"]] == ["query_order", "query_logistics"]
    assert [rows[2].tool_call_id, rows[4].tool_call_id] == ["order", "shipping"]
    order = ToolResult.model_validate_json(rows[2].content)
    shipping = ToolResult.model_validate_json(rows[4].content)
    assert order.ok and shipping.ok
    assert order.data["tracking_no"] == rows[3].tool_calls[0]["args"]["tracking_no"]
    assert shipping.data["latest_node"] == "包裹已到达杭州转运中心"
    assert rows[-1].content == "".join(event.data.get("delta", "") for event in events)
    # 审计载荷保留来源标记；模型输入仅去掉标记，全部订单业务事实必须完整保留。
    order_messages = [message for message in model.inputs[2]["messages"]
                      if message["role"] == "tool" and message["tool_call_id"] == "order"]
    assert len(order_messages) == 1
    model_order = json.loads(order_messages[0]["content"])
    assert set(model_order) == {"ok", "data", "error"}
    assert model_order["ok"] is True and model_order["error"] is None
    assert model_order["data"] == order.data and order.is_mock
    turns = await chat.load_turns(conversation.id)
    assert len(turns) == 1 and turns[0].turn_status == "completed"
    history = assemble_turns(turns)[0]
    assert [message.type for message in history] == ["human", "ai", "tool", "ai", "tool", "ai"]
    assert [history[2].content, history[4].content] == [rows[2].content, rows[4].content]


async def test_mysql_chat_suggestion_requires_independent_confirmation(isolated_multi_tool_mysql):
    """普通聊天只保存建议，确认API重复提交仍只有一票及独立历史。参数：隔离资源；返回：None；异常：断言失败。"""
    from fastapi import FastAPI
    from app.api.actions import get_ticket_action_service, router
    from app.repositories.actions import ActionRepository
    from app.schemas.workflow import AgentDecision
    from app.services.actions import TicketActionService
    from app.tools.executor import ToolExecutor
    from app.tools.registry import build_ticket_registry
    factory, session_id, build_service = isolated_multi_tool_mysql
    service, chat, _ = build_service([AgentDecision(kind="answer", suggested_actions=["create_ticket"])])
    question = "  订单1001未送达，请创建工单  "
    events = [event async for event in service.stream_turn(ChatRequest(session_id=session_id, message=question))]
    assert events[-1].event == "done"
    conversation, source_rows, tickets = await _read_records(factory, session_id)
    assert tickets == [] and all(row.turn_status == "completed" for row in source_rows)
    suggestion = events[-2].data["suggested_actions"][0]
    actions = TicketActionService(ActionRepository(factory),
        ToolExecutor(build_ticket_registry(), chat, SupportRepository(factory)), service.guard)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_ticket_action_service] = lambda: actions
    body = {"session_id": session_id, "source_turn_id": suggestion["source_turn_id"], "action_id": suggestion["action_id"], "confirmed": True}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/api/actions/tickets", json=body)
        retry = await client.post("/api/actions/tickets", json=body)
    assert first.status_code == retry.status_code == 200 and first.json() == retry.json()
    conversation, rows, tickets = await _read_records(factory, session_id)
    assert len(tickets) == 1 and tickets[0].description == question
    assert conversation.status == "进行中" and all(row.turn_status == "completed" for row in rows)
    turns = await chat.load_turns(conversation.id)
    assert len(turns) == 2 and turns[0].turn_id == suggestion["source_turn_id"]
    assert [message.type for message in assemble_turns(turns)[1]] == ["human", "ai", "tool", "ai"]
    assert tickets[0].ticket_no in turns[1].messages[-1].content


# 原“聊天回执处取消”的建单场景已迁至test_ch05_actions中的真实硬退出/恢复验收，
# 使用独立确认动作及非破坏性父资源锁，避免继续把写工具暴露给普通聊天模型。
