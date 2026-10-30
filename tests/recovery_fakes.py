"""恢复验收共用边界：真实业务仓储/主lifespan/SQLite图，仅替换外部资源。

离线SQLite表用于验证恢复契约，不证明MySQL行锁、DDL或网络行为。
"""

from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.evaluation.jobs import EvaluationJobManager
from app.models.customer_service import Conversation, Message, Ticket
from app.schemas.chat import ChatRequest
from app.schemas.tools import ToolContext
from app.schemas.workflow import AgentDecision
from tests.workflow.fakes import ScriptedWorkflowModel


@asynccontextmanager
async def offline_recovery_database(path: Path):
    """在独占SQLite文件创建恢复所需业务表，始终释放引擎。

    参数：path pytest独占目录，不接受现有业务库路径。
    返回：异步上下文提供真实SQLAlchemy引擎及会话工厂。
    异常：SQLite/SQLAlchemy错误原样传播。
    """
    engine = create_async_engine("sqlite+aiosqlite:///" + (path / "business.sqlite3").as_posix())
    try:
        async with engine.begin() as connection:
            # 仅适配SQLite的主键/默认时间写法；真实仓储负责提交、恢复与读取。
            for ddl in (
                "CREATE TABLE conversations (id INTEGER PRIMARY KEY, session_id TEXT UNIQUE NOT NULL, user_id TEXT NOT NULL, status TEXT DEFAULT '进行中' NOT NULL, summary TEXT, summary_upto_msg_id INTEGER, layer1_from_msg_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)",
                "CREATE TABLE conversation_summaries (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, seq INTEGER NOT NULL, from_msg_id INTEGER NOT NULL, upto_msg_id INTEGER NOT NULL, content TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(conversation_id, seq))",
                "CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id), role TEXT NOT NULL CHECK(role IN ('user','assistant','tool')), content TEXT, tool_calls JSON, rag_metadata JSON, tool_call_id TEXT, turn_id TEXT NOT NULL, turn_status TEXT DEFAULT 'running' NOT NULL CHECK(turn_status IN ('running','awaiting_input','completed','failed','cancelled')), created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, UNIQUE(conversation_id,turn_id,tool_call_id), CHECK((role='tool' AND tool_call_id IS NOT NULL AND tool_call_id <> '') OR (role <> 'tool' AND tool_call_id IS NULL)))",
                "CREATE TABLE tickets (ticket_no TEXT PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id), description TEXT NOT NULL, ticket_type TEXT NOT NULL CHECK(ticket_type IN ('售后','投诉','咨询')), status TEXT DEFAULT '待处理' NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)",
            ):
                await connection.exec_driver_sql(ddl)
        yield engine, async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


class ForbiddenRecoverySearch:
    """恢复业务场景不允许意外读取知识模型、编码器或向量库。"""

    async def retrieve(self, *args, **kwargs):
        """阻断知识读取。参数：原检索契约；返回：不返回；异常：AssertionError，走错路由。"""
        raise AssertionError("恢复业务验收不得访问知识资源")

    async def retrieve_policy(self, *args, **kwargs):
        """阻断政策读取。参数：原检索契约；返回：不返回；异常：AssertionError，走错路由。"""
        raise AssertionError("恢复业务验收不得访问知识资源")


class RecoveryRuntimeBoundary:
    """为主应用工厂隔离测试库、lease引擎、知识资源、checkpoint、模型与评估路径。"""

    def __init__(self, monkeypatch, path: Path, url, model: ScriptedWorkflowModel):
        """安装实例独占的外部边界，保留database_runtime全部业务流程。

        参数：monkeypatch 自动还原器、path独占目录、url已验证测试库地址、model有限脚本模型。
        返回：None，通过属性保留资源创建/释放证据。
        异常：AssertionError，库地址越出测试边界。
        """
        from app import main
        from app.workflow import model as workflow_model

        if url.get_backend_name() == "sqlite":
            assert Path(url.database).resolve().parent == path.resolve()
        else:
            assert url.get_backend_name() == "mysql" and url.database == "mewhelp_test"
        self.root, self.path = path, path / "workflow.sqlite3"
        self.url, self.model = url, model
        self.engines, self.factories, self.lease_calls, self.disposed = [], [], [], []
        self.search_opened = self.search_closed = 0
        values = main.settings.model_dump()
        values.update(knowledge_retrieval_mode="hybrid", workflow_checkpoint_path=self.path.as_posix())
        self.settings = Settings(_env_file=None, **values)
        monkeypatch.setattr(main, "settings", self.settings)
        monkeypatch.setattr(main, "create_database", self.create_database)
        monkeypatch.setattr(main, "check_ch09_tables", self.check_ch09_tables)
        monkeypatch.setattr(main, "create_async_engine", self.create_lease_engine)
        monkeypatch.setattr(main, "online_search_runtime", self.search_runtime)
        monkeypatch.setattr(main, "create_job_manager", self.create_job_manager)
        # 模型必须在根图闭包创建前注入；当前ChatService不读取service.model属性。
        monkeypatch.setattr(main, "WorkflowModelAdapter", self.create_model)
        monkeypatch.setattr(workflow_model, "get_chat_model", self.forbid_external_model)

    async def check_ch09_tables(self, config):
        """恢复测试隔离新增外部 MySQL 核验。参数：配置；返回：None；异常：无。"""

    def forbid_external_model(self, *args, **kwargs):
        """阻断遗漏注入后的外部模型。参数：模型工厂原参数；返回：不返回；异常：AssertionError。"""
        raise AssertionError("恢复验收不得调用真实模型工厂")

    def create_model(self):
        """把有限脚本交给main实际调用的图模型工厂。参数：无；返回：脚本模型；异常：无。"""
        return self.model

    def create_job_manager(self):
        """为主生命周期创建真实评估管理器，路径限定到本例临时目录。

        参数：无，使用构造时已指定的独占目录。
        返回：EvaluationJobManager，保留真实关闭行为但不启动评估。
        异常：ValueError，管理器配置非法时透传。
        """
        return EvaluationJobManager(self.root, self.root / "evaluation-reports")

    def _track_engine(self, engine):
        """记录真实引擎释放事件。参数：engine本测试拥有的资源；返回：原引擎；异常：事件注册错误。"""
        def record_disposal(sync_engine):
            """记录SQLAlchemy释放事件。参数：sync_engine同步代理；返回：None；异常：无。"""
            self.disposed.append(engine)

        event.listen(engine.sync_engine, "engine_disposed", record_disposal)
        self.engines.append(engine)
        return engine

    def create_database(self):
        """按已验证测试地址重建业务连接。参数：无；返回：新引擎/工厂；异常：SQLAlchemy错误。"""
        engine = self._track_engine(create_async_engine(self.url))
        factory = async_sessionmaker(engine, expire_on_commit=False)
        self.factories.append(factory)
        return engine, factory

    def create_lease_engine(self, url, **kwargs):
        """隔离主lifespan的独立lease连接，SQLite仅适配MySQL驱动参数。

        参数：url 必须同测试库、kwargs 主应用提供的连接参数。
        返回：真实NullPool异步引擎，与业务引擎不同实例。
        异常：AssertionError，连接目标或pool改变；SQLAlchemy错误透传。
        """
        assert url == self.url and kwargs["poolclass"] is NullPool
        self.lease_calls.append((url, dict(kwargs)))
        # SQLite没有connect_timeout参数；MySQL验收保留生产连接参数。
        if url.get_backend_name() == "sqlite":
            kwargs = {**kwargs, "connect_args": {}}
        return self._track_engine(create_async_engine(url, **kwargs))

    @asynccontextmanager
    async def search_runtime(self, settings, store):
        """仅替换知识外部生命周期，记录主应用是否关闭它。

        参数：settings 主应用实例配置、store 主应用构造的真实仓储对象。
        返回：有限的禁止检索替身；业务仓储与图不变。
        异常：AssertionError，配置引用错误。
        """
        assert settings is self.settings
        self.search_opened += 1
        try:
            yield ForbiddenRecoverySearch()
        finally:
            self.search_closed += 1


async def assert_application_recovery_contract(database, repositories, monkeypatch, path, with_result):
    """两次主应用lifespan共享同一验收：回灌确定事实、不重放、后续正常回复。

    参数：database 已隔离数据库、repositories真实业务仓储、monkeypatch替换器、path独占checkpoint目录、with_result是否提交业务结果。
    返回：None，验证数据库事实、三个模型阶段输入、持久化checkpoint和资源释放。
    异常：AssertionError，恢复或注入契约破坏；真实数据库错误透传。
    """
    from app import main

    chat, support = repositories
    context = await chat.start_turn(ToolContext(session_id="restart", user_id="demo-user", turn_id="old-turn",
        tool_call_id="original-call", user_message="请建立工单"), "请建立工单")
    await chat.save_tool_calls(context, [{"name": "create_ticket", "id": "original-call",
        "args": {"description": "物流延迟", "ticket_type": "咨询"}}])
    result = await support.create_ticket(context, "物流延迟", "咨询") if with_result else None
    url = database[0].url
    await database[0].dispose()
    model = ScriptedWorkflowModel(intents=["订单", "订单"],
        decisions=[AgentDecision(kind="answer"), AgentDecision(kind="answer")],
        answers=[["已保存的处理状态"], ["后续正常答复"]])
    boundary = RecoveryRuntimeBoundary(monkeypatch, path, url, model)
    assert not boundary.path.exists()

    for index, (question, answer) in enumerate((("刚才提交了吗", "已保存的处理状态"), ("请继续说明", "后续正常答复"))):
        application = main.create_app(runtime_factory=main.database_runtime)
        async with application.router.lifespan_context(application):
            manager = application.state.evaluation_jobs
            # 主应用新增评估资源时，恢复验收也必须限制在本例独占目录。
            assert manager.repo_root == path
            assert manager.reports_root == path / "evaluation-reports"
            assert manager.state_path.parent == manager.reports_root
            assert manager.log_path.parent == manager.reports_root
            service = application.state.chat_service
            assert len(model.inputs) == index * 4
            factory = boundary.factories[-1]
            async with factory() as session:
                old_rows = (await session.scalars(select(Message).where(Message.turn_id == "old-turn"))).all()
                assert old_rows and all(row.turn_status == "cancelled" for row in old_rows)
                assert (await session.get(Conversation, context.conversation_id)).status == "进行中"
                tickets = (await session.scalars(select(Ticket))).all()
                assert len(tickets) == (1 if with_result else 0)
                if with_result:
                    assert tickets[0].ticket_no == result.data["ticket_no"]
            events = [event async for event in service.stream_turn(ChatRequest(session_id="restart", message=question))]
            assert len(model.inputs) == (index + 1) * 4, "脚本模型未控制当前图的understand/classify/decide/answer"
            assert events[-1].event == "done"
            assert "".join(event.data["delta"] for event in events if event.event == "delta") == answer
            assert not any(event.event in ("error", "tool_status") for event in events)
            current_inputs = model.inputs[index * 4:]
            assert [item["stage"] for item in current_inputs] == ["understand", "classify", "decide", "answer"]
            for item in current_inputs:
                messages = item["messages"]
                facts = [message for message in messages if message["role"] == "tool"]
                assert len(facts) == (1 if with_result else 0)
                assert any("未完成" in message["content"] for message in messages)
                if with_result:
                    assert facts[0]["tool_call_id"] == "original-call"
                    assert result.data["ticket_no"] in facts[0]["content"]
                else:
                    assert not any(message.get("tool_calls") for message in messages)
            assert await service.chat.recover_interrupted_turns() == 0
            config = {"configurable": {"thread_id": f"mewhelp:ch05:{context.conversation_id}"}}
            checkpoint = (await service.graph.aget_state(config)).values
            assert checkpoint["answer"] == answer and checkpoint["completion_metadata"]["status"] == "completed"
            async with factory() as session:
                assert len((await session.scalars(select(Ticket))).all()) == (1 if with_result else 0)
                old_results = (await session.scalars(select(Message).where(Message.turn_id == "old-turn", Message.role == "tool"))).all()
                assert len(old_results) == (1 if with_result else 0)
                latest = await service.chat.load_turns(context.conversation_id)
                assert latest[-1].turn_status == "completed" and latest[-1].messages[-1].content == answer
        assert not hasattr(application.state, "chat_service")
        assert not hasattr(application.state, "evaluation_jobs")
        assert boundary.search_opened == boundary.search_closed == index + 1
        assert len(boundary.engines) == len(boundary.disposed) == (index + 1) * 2
        assert boundary.engines[-1] is not boundary.engines[-2]
        assert boundary.path.is_file()
    assert len(boundary.lease_calls) == 2
    assert not model.intents and not model.decisions and not model.answers
