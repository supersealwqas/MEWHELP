"""验证 MySQL 数据约束、原子工单、幂等种子和进程中断恢复。"""

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSessionTransaction

from app.db.seed import seed_database
from app.knowledge.schemas import KnowledgeUnavailable
from app.models.customer_service import Conversation, FAQ, Message, Ticket
from app.repositories.support import SupportRepository
from app.schemas.tools import ToolContext, ToolResult

pytestmark = pytest.mark.mysql


def context(turn="turn-1", call="call-1"):
    """生成隔离上下文。参数：turn/call 为轮次及调用 ID；返回：ToolContext；异常：字段校验失败。"""
    return ToolContext(session_id="session-1", user_id="demo-user", turn_id=turn,
                       tool_call_id=call, user_message="请创建人工工单")


async def prepare(chat, ctx):
    """保存一轮申请。参数：chat 存储、ctx 上下文；返回：None；异常：数据库写入失败。"""
    ctx = await chat.start_turn(ctx, ctx.user_message)
    await chat.save_tool_calls(ctx, [{"id": ctx.tool_call_id, "name": "create_ticket",
                                     "args": {"description": "未收到订单", "ticket_type": "投诉"}}])
    return ctx


async def test_tool_result_unique(database, repositories):
    """同一调用只保留原结果。参数：测试库及存储夹具；返回：None；异常：断言失败。"""
    chat, _ = repositories
    ctx = context()
    ctx = await prepare(chat, ctx)
    original = ToolResult(ok=True, data={"value": 1})
    await chat.save_tool_result(ctx, original)
    repeated = await chat.save_tool_result(ctx, ToolResult(ok=True, data={"value": 2}))
    assert repeated == original
    async with database[1]() as session:
        assert await session.scalar(select(func.count()).select_from(Message).where(Message.role == "tool")) == 1


async def test_ticket_commit_ack_lost(database, repositories, monkeypatch):
    """模拟提交后调用方异常并重试。参数：测试库和存储；返回：None；异常：断言失败。"""
    chat, support = repositories
    ctx = context()
    ctx = await prepare(chat, ctx)
    original_exit = AsyncSessionTransaction.__aexit__

    async def commit_then_disconnect(transaction, *args):
        """提交成功后模拟确认丢失。参数：事务及退出参数；返回：不返回；异常：OperationalError。"""
        await original_exit(transaction, *args)
        raise OperationalError(None, None, RuntimeError("injected lost commit acknowledgement"))

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSessionTransaction, "__aexit__", commit_then_disconnect)
        with pytest.raises(OperationalError):
            await support.create_ticket(ctx, "未收到订单", "投诉")
    # 通过新会话核实真实提交，不能把抛异常等同于回滚。
    first = await chat.get_tool_result(ctx)
    assert first is not None and first.ok
    from app.repositories.support import SupportRepository
    repeated = await SupportRepository(database[1]).create_ticket(ctx, "未收到订单", "投诉")
    assert first == repeated
    async with database[1]() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 1
        assert await session.scalar(select(func.count()).select_from(Message).where(Message.role == "tool")) == 1
        assert (await session.get(Conversation, ctx.conversation_id)).status == "进行中"


async def test_rollback_has_no_partial_ticket(database, repositories, monkeypatch):
    """写结果失败时回滚整个工单事务。参数：测试夹具及替换器；返回：None；异常：断言失败。"""
    chat, support = repositories
    ctx = context()
    ctx = await prepare(chat, ctx)
    from app.repositories import support as module
    original = module.Message

    def broken_message(**kwargs):
        """模拟结果写入前失败。参数：模型字段；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("injected result failure")

    monkeypatch.setattr(module, "Message", broken_message)
    with pytest.raises(RuntimeError):
        await support.create_ticket(ctx, "未收到订单", "投诉")
    monkeypatch.setattr(module, "Message", original)
    async with database[1]() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 0
        assert (await session.get(Conversation, ctx.conversation_id)).status == "进行中"
    result = await support.create_ticket(ctx, "未收到订单", "投诉")
    assert result.ok


async def test_recover_running_preserves_ticket(database, repositories):
    """恢复保留工单事实且重复执行无变化。参数：测试夹具；返回：None；异常：断言失败。"""
    chat, support = repositories
    ctx = context()
    ctx = await prepare(chat, ctx)
    result = await support.create_ticket(ctx, "未收到订单", "投诉")
    assert await chat.recover_interrupted_turns() == 1
    assert await chat.recover_interrupted_turns() == 0
    turns = await chat.load_turns(ctx.conversation_id)
    assert turns[0].turn_status == "cancelled"
    assert turns[0].messages[-1].role == "tool"
    assert (await chat.get_tool_result(ctx)) == result
    async with database[1]() as session:
        assert (await session.get(Conversation, ctx.conversation_id)).status == "进行中"


async def test_result_role_and_id_constraints(database, repositories):
    """数据库拒绝非法角色关联及重复结果。参数：测试夹具；返回：None；异常：断言失败。"""
    chat, _ = repositories
    ctx = context()
    ctx = await prepare(chat, ctx)
    for role, call_id in [("tool", None), ("user", "wrong"), ("system", None)]:
        async with database[1]() as session:
            session.add(Message(conversation_id=ctx.conversation_id, turn_id=ctx.turn_id,
                                role=role, content="bad", tool_call_id=call_id))
            # asyncmy 将 MySQL CHECK/ENUM 拒绝映射为 OperationalError；检查实际错误码。
            with pytest.raises(DBAPIError) as failure:
                await session.commit()
            assert failure.value.orig.args[0] in {3819, 1265}
    await chat.save_tool_result(ctx, ToolResult(ok=True, data={}))
    async with database[1]() as session:
        session.add(Message(conversation_id=ctx.conversation_id, turn_id=ctx.turn_id,
                            role="tool", content="{}", tool_call_id=ctx.tool_call_id))
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_faq_requires_injected_knowledge_search(database, repositories):
    """FAQ 不再回退 LIKE：无检索失败关闭，注入检索后按契约返回。参数：测试夹具；返回：None；异常：断言失败。"""
    _, support = repositories
    with pytest.raises(KnowledgeUnavailable):
        await support.query_faq("退货")

    class _Search:
        """最小知识检索替身。"""

        async def query_faq(self, keyword, limit=3):
            """按固定关键词返回命中。参数：关键词/条数；返回：问答；异常：无。"""
            if keyword == "退货":
                return [{"id": 1, "question": "退货政策", "answer": "演示答案", "category": "售后"}]
            return []

    injected = SupportRepository(database[1], _Search())
    assert (await injected.query_faq("退货"))[0]["id"] == 1
    assert await injected.query_faq("不存在") == []


async def test_seed_idempotent(database):
    """种子覆盖四表且不会覆盖已有数据。参数：测试库；返回：None；异常：断言失败。"""
    factory = database[1]
    await seed_database(factory)
    async with factory.begin() as session:
        faq = await session.get(FAQ, 1)
        faq.answer = "用户修改后的答案"
    await seed_database(factory)
    async with factory() as session:
        for model in [FAQ, Conversation, Message, Ticket]:
            assert await session.scalar(select(func.count()).select_from(model)) > 0
        assert (await session.get(FAQ, 1)).answer == "用户修改后的答案"


async def test_user_ownership_conflict(repositories):
    """会话归属不可被后续请求篡改。参数：存储夹具；返回：None；异常：断言失败。"""
    chat, _ = repositories
    ctx = context()
    await chat.start_turn(ctx, "hello")
    with pytest.raises(ValueError):
        await chat.start_turn(ctx.model_copy(update={"user_id": "other", "turn_id": "turn-2"}), "hello")


async def test_seed_repairs_interrupted_prefix(database, repositories):
    """初始化中断后补齐缺失种子。参数：真实测试库和存储；返回：None；异常：断言失败。"""
    chat, _ = repositories
    ctx = ToolContext(session_id="demo-seed-session", user_id="demo-seed-user",
                      turn_id="demo-seed-turn", user_message="已保存的用户原文")
    ctx = await chat.start_turn(ctx, ctx.user_message)
    await seed_database(database[1])
    await seed_database(database[1])
    turns = await chat.load_turns(ctx.conversation_id)
    assert [message.role for message in turns[0].messages] == ["user", "assistant", "tool", "assistant"]
    assert turns[0].messages[0].content == "已保存的用户原文"
    assert turns[0].turn_status == "completed"
    async with database[1]() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 1


async def test_reset_requires_explicit_test_database():
    """拒绝误清应用库。参数：无；返回：None；异常：断言失败。"""
    from app.db.init_db import initialize_database
    with pytest.raises(ValueError, match="--test"):
        await initialize_database(reset=True)


async def test_reset_refreshes_only_verified_test_database(database, monkeypatch):
    """受控刷新后读取新种子，错误目标不会删表。参数：测试库和替换器；返回：None；异常：断言失败。"""
    from app.db import init_db
    from app.db.seed import FAQ_SEEDS
    await seed_database(database[1])
    async with database[1].begin() as session:
        (await session.get(FAQ, 1)).answer = "旧政策"
    def test_database(**kwargs):
        """提供已隔离的真实测试引擎。参数：连接配置；返回：测试库资源；异常：无。"""
        return database

    monkeypatch.setattr(init_db, "create_database", test_database)
    await init_db.initialize_database(test=True, reset=True)
    async with database[1]() as session:
        assert (await session.get(FAQ, 1)).answer == FAQ_SEEDS[0][2]


@pytest.mark.parametrize("actual_name", ["mewhelp", "MEWHELP_TEST", None])
async def test_reset_rejects_unverified_database(database, monkeypatch, actual_name):
    """实际库名不符时保留原记录。参数：测试库、替换器和不合法身份；返回：None；异常：断言失败。"""
    from app.db import init_db
    await seed_database(database[1])
    async with database[1].begin() as session:
        (await session.get(FAQ, 1)).answer = "必须保留"

    def test_database(**kwargs):
        """复用隔离库资源。参数：配置参数；返回：测试引擎和工厂；异常：无。"""
        return database

    async def wrong_identity(connection, statement, *args, **kwargs):
        """模拟连接实际目标不符。参数：连接和查询参数；返回：错误库名；异常：无。"""
        return actual_name

    with monkeypatch.context() as patch:
        patch.setattr(init_db, "create_database", test_database)
        patch.setattr(AsyncConnection, "scalar", wrong_identity)
        with pytest.raises(ValueError, match="实际连接"):
            await init_db.initialize_database(test=True, reset=True)
    async with database[1]() as session:
        assert (await session.get(FAQ, 1)).answer == "必须保留"
