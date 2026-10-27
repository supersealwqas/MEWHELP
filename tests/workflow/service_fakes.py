"""在线服务测试资源：真实仓储/图搭配独占SQLite表和有限模型脚本，不连接业务数据库。"""

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.repositories.chat import ChatRepository
from app.services.chat import ChatService
from app.workflow.runtime import workflow_runtime
from tests.workflow.graph_fakes import dependencies


@asynccontextmanager
async def service_runtime(path, **script):
    """创建真实服务及独立文件资源。参数：path独占目录、script脚本；返回：服务/依赖/工厂；异常：数据库及图错误。"""
    engine = create_async_engine("sqlite+aiosqlite:///" + (path / "business.sqlite3").as_posix())
    # 仅在测试专用SQLite文件建立等价列；MySQL行锁/并发提交另由Task8验收。
    async with engine.begin() as connection:
        for ddl in (
            "CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, session_id TEXT UNIQUE, user_id TEXT, status TEXT DEFAULT '进行中', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, role TEXT, content TEXT, tool_calls JSON, rag_metadata JSON, tool_call_id TEXT, turn_id TEXT, turn_status TEXT DEFAULT 'running', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(conversation_id, turn_id, tool_call_id))",
            "CREATE TABLE IF NOT EXISTS low_confidence_questions (id INTEGER PRIMARY KEY, conversation_id INTEGER, raw_question TEXT, source TEXT, reason TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS refund_applications (refund_no TEXT PRIMARY KEY, action_id TEXT NOT NULL UNIQUE, conversation_id INTEGER NOT NULL, source_turn_id TEXT NOT NULL, user_id TEXT NOT NULL, order_id TEXT NOT NULL, reason_code TEXT NOT NULL, original_question TEXT NOT NULL, status TEXT DEFAULT '待处理' NOT NULL CHECK(status='待处理'), created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
        ):
            await connection.exec_driver_sql(ddl)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    deps = dependencies(**script)
    deps.chat = ChatRepository(factory)
    deps.executor.chat = deps.chat
    try:
        async with workflow_runtime(deps, path / "workflow.sqlite3") as graph:
            service = ChatService(deps.chat, graph, deps.executor, deps.settings)
            yield service, deps, factory
    finally:
        await engine.dispose()


async def collect(service, message="你好", session="s"):
    """完整消费一轮。参数：service服务、message原问、session会话；返回：事件列表；异常：服务取消。"""
    from app.schemas.chat import ChatRequest
    return [event async for event in service.stream_turn(ChatRequest(session_id=session, message=message))]
