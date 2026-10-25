"""在已有 MySQL 测试库核验多工具持久化、工单防重和取消后的事实恢复。

仅模型决策使用脚本替身，聊天服务、工具执行器及两个仓库均运行真实代码。
每例只创建并清理自己的唯一会话，不建表、不清库，也不恢复其他会话。
"""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
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
from app.services.chat import ChatService, LegacyChatService
from app.tools.registry import build_registry
from tests.chat_fakes import ScriptedModel

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

        def build_service(decisions, *, workflow=False):
            """构造真实服务和仓库。参数：decisions脚本，workflow显式新图入口；返回：服务/仓储/模型；异常：构造错误。"""
            if workflow:
                from langgraph.checkpoint.memory import InMemorySaver
                from app.schemas.workflow import AgentDecision
                from app.tools.executor import ToolExecutor
                from app.tools.registry import build_agent_registry
                from app.workflow.graph import WorkflowDependencies, build_workflow
                from tests.workflow.fakes import ScriptedWorkflowModel
                from tests.workflow.graph_fakes import StrictSearch
                chat = ChatRepository(factory)
                model = ScriptedWorkflowModel(intents=["物流"], decisions=[*decisions, AgentDecision(kind="answer")],
                                              answers=[["已按工具结果处理。"]])
                executor = ToolExecutor(build_agent_registry(), chat, SupportRepository(factory))
                deps = WorkflowDependencies(model, StrictSearch(), executor, chat, config)
                # 本例只验证业务MySQL配对；官方文件SQLite重开另由workflow专项与Task8覆盖。
                graph = build_workflow(deps, checkpointer=InMemorySaver())
                return ChatService(chat, graph, executor, config), chat, model
            model = ScriptedModel(decisions=decisions, chunks=["已按工具结果处理。"])
            chat = ChatRepository(factory)
            service = LegacyChatService(chat, SupportRepository(factory), model, build_registry(), config)
            return service, chat, model

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


def _decision(*calls: dict) -> AIMessage:
    """构造模型工具申请。参数：calls 一个决策中的完整工具调用；返回：AIMessage；异常：消息校验错误。"""
    return AIMessage(content="", tool_calls=list(calls))


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
    ], workflow=True)
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


@pytest.mark.parametrize("same_batch", [True, False], ids=["same-batch", "successive-decisions"])
async def test_mysql_duplicate_ticket_new_id_creates_only_one_ticket(isolated_multi_tool_mysql, same_batch):
    """新ID的重复建单只能留下一个真实工单。参数：isolated_multi_tool_mysql 隔离资源、same_batch 是否同批；返回：None；异常：断言失败。"""
    factory, session_id, build_service = isolated_multi_tool_mysql
    first = {"id": "ticket-1", "name": "create_ticket", "args": {"description": "订单1001未送达", "ticket_type": "售后"}}
    second = {"id": "ticket-2", "name": "create_ticket", "args": {"description": "补充请人工确认订单状态", "ticket_type": "售后"}}
    decisions = [_decision(first, second)] if same_batch else [_decision(first), _decision(second)]
    service, chat, _ = build_service(decisions)
    events = [event async for event in service.stream_turn(
        ChatRequest(session_id=session_id, message="订单1001未送达，请创建工单并请人工确认")
    )]
    assert events[-1].event == "done"
    conversation, rows, tickets = await _read_records(factory, session_id)
    assert len(tickets) == 1 and tickets[0].description == "订单1001未送达"
    assert tickets[0].status == "待处理" and conversation.status == "已转人工"
    calls = [call for row in rows for call in (row.tool_calls or [])]
    results = [row for row in rows if row.role == "tool"]
    assert [call["id"] for call in calls] == ["ticket-1", "ticket-2"]
    assert [row.tool_call_id for row in results] == ["ticket-1", "ticket-2"]
    saved = ToolResult.model_validate_json(results[0].content)
    rejected = ToolResult.model_validate_json(results[1].content)
    assert saved.ok and saved.data["ticket_no"] == tickets[0].ticket_no
    assert not rejected.ok and rejected.error.code == "WRITE_TOOL_LIMIT"
    assert all(row.turn_status == "completed" for row in rows)
    assert tickets[0].ticket_no in rows[-1].content
    assert rows[-1].content == "".join(event.data.get("delta", "") for event in events)
    history = assemble_turns(await chat.load_turns(conversation.id))[0]
    assert [message.tool_call_id for message in history if isinstance(message, ToolMessage)] == ["ticket-1", "ticket-2"]


async def test_mysql_cancel_at_ticket_receipt_retains_partial_batch_fact(isolated_multi_tool_mysql):
    """回执处取消须保留已提交工单并隐藏未执行申请。参数：isolated_multi_tool_mysql 隔离资源；返回：None；异常：断言失败。"""
    factory, session_id, build_service = isolated_multi_tool_mysql
    service, chat, model = build_service([_decision(
        {"id": "ticket", "name": "create_ticket", "args": {"description": "订单1001未送达", "ticket_type": "售后"}},
        {"id": "order", "name": "query_order", "args": {"order_id": "1001"}},
    )])
    stream = service.stream_turn(ChatRequest(session_id=session_id, message="创建工单并查询订单1001"))
    receipt = ""
    try:
        async for event in stream:
            if event.event == "delta":
                receipt = event.data["delta"]
                break
        assert "工单已创建" in receipt
    finally:
        # 此时第二个工具尚未开始，关闭真实生成器并等待取消终态写入。
        await stream.aclose()
    conversation, rows, tickets = await _read_records(factory, session_id)
    assert len(tickets) == 1 and tickets[0].ticket_no in receipt
    assert [row.role for row in rows] == ["user", "assistant", "tool"]
    assert all(row.turn_status == "cancelled" for row in rows)
    assert [call["id"] for call in rows[1].tool_calls] == ["ticket", "order"]
    assert rows[2].tool_call_id == "ticket"
    assert [phase for phase, _ in model.inputs] == ["decide"]
    turns = await chat.load_turns(conversation.id)
    assert len(turns) == 1 and turns[0].turn_status == "cancelled"
    history = assemble_turns(turns)[0]
    assert [message.type for message in history] == ["human", "ai", "tool", "ai"]
    assert [call["id"] for call in history[1].tool_calls] == ["ticket"]
    assert history[2].tool_call_id == "ticket" and tickets[0].ticket_no in history[2].content
    assert "未完成" in history[-1].content
