"""用真实SQL事务核验引用持久化、归属隔离及完成重试。"""

from contextlib import asynccontextmanager

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.knowledge.answering import EvidenceCitation
from app.repositories.chat import ChatRepository, PersistenceUncertain
from app.schemas.tools import ToolContext


class AsyncSessionBridge:
    """仅替代异步驱动，用真实SQLite事务运行仓储SQL。"""

    def __init__(self, session):
        """保存同步会话。参数：session SQLAlchemy会话；返回：None；异常：无。"""
        self.session = session

    async def scalar(self, statement):
        """执行单值查询。参数：statement SQL；返回：查询值；异常：SQL错误。"""
        return self.session.scalar(statement)

    async def scalars(self, statement):
        """执行多值查询。参数：statement SQL；返回：结果；异常：SQL错误。"""
        return self.session.scalars(statement)

    async def get(self, model, key):
        """按主键查询。参数：model模型/key主键；返回：对象；异常：SQL错误。"""
        return self.session.get(model, key)

    async def execute(self, statement):
        """执行语句。参数：statement SQL；返回：结果；异常：SQL错误。"""
        return self.session.execute(statement)

    async def flush(self):
        """刷新事务。参数：无；返回：None；异常：SQL错误。"""
        self.session.flush()

    def add(self, value):
        """加入写入对象。参数：value ORM实例；返回：None；异常：SQL错误。"""
        self.session.add(value)


class Factory:
    """提供短事务边界，DDL按SQLite声明以支持自增。"""

    def __init__(self, engine):
        """保存引擎。参数：engine SQL引擎；返回：None；异常：无。"""
        self.engine = engine

    @asynccontextmanager
    async def begin(self):
        """提供真实提交或回滚。参数：无；返回：会话桥；异常：SQL错误。"""
        with Session(self.engine) as session, session.begin():
            yield AsyncSessionBridge(session)

    def __call__(self):
        """提供读取边界。参数：无；返回：上下文；异常：无。"""
        return self.begin()


@pytest.fixture
def repository():
    """建立测试内存库，仅测试SQL行为，不替代MySQL集成。参数：无；返回：仓储；异常：SQL错误。"""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE conversations (id INTEGER PRIMARY KEY, session_id TEXT, user_id TEXT, status TEXT DEFAULT '进行中', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
        connection.execute(text("CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, role TEXT, content TEXT, tool_calls JSON, rag_metadata JSON, tool_call_id TEXT, turn_id TEXT, turn_status TEXT DEFAULT 'running', created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
        connection.execute(text("CREATE TABLE knowledge_chunks (id INTEGER PRIMARY KEY, source_key TEXT, ordinal INTEGER, questions TEXT, answer TEXT, category TEXT, section_path TEXT, is_active INTEGER)"))
        connection.execute(text("INSERT INTO knowledge_chunks VALUES (20, 'qa:10', 0, '来源问法', '权威正文', '分类', NULL, 1)"))
    yield ChatRepository(Factory(engine))
    engine.dispose()


async def test_snapshot_saved_and_only_readable_for_completed_owner(repository):
    """完整快照保存后仅同会话用户可读，重试不得替换证据。参数：仓储；返回：None；异常：断言错误。"""
    context = await repository.start_turn(ToolContext(session_id="s", user_id="owner", turn_id="t", user_message="政策"), "政策")
    citation = EvidenceCitation(n=1, chunk_id=5, question="条件", answer="旧的完整正文", source_key="qa:8")
    assert await repository.get_citation("s", "owner", "t", 1) is None
    metadata = await repository.finish_turn(context, "答复[1]", citations=[citation])
    assert metadata["citations"][0]["answer"] == "旧的完整正文"
    snapshot = await repository.get_citation("s", "owner", "t", 1)
    assert snapshot == metadata["citations"][0] and snapshot["source_type"] == "published_qa"
    for scope in [("s", "other", "t", 1), ("other", "owner", "t", 1), ("s", "owner", "other", 1), ("s", "owner", "t", 9)]:
        assert await repository.get_citation(*scope) is None
    assert await repository.finish_turn(context, "答复[1]", citations=[citation]) == metadata
    with pytest.raises(PersistenceUncertain):
        await repository.finish_turn(context, "答复[1]", citations=[citation.model_copy(update={"answer": "新原文"})])


@pytest.mark.parametrize("answer,expected_source", [("权威正文", "qa:10"), ("被改写的正文", None)])
async def test_source_supplement_requires_exact_authoritative_body(repository, answer, expected_source):
    """来源补齐按主键核对完整正文，变化后不补成新来源。参数：仓储/正文/期望；返回：None；异常：断言错误。"""
    context = await repository.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="问"), "问")
    citation = EvidenceCitation(n=1, chunk_id=20, question="来源问法", answer=answer)
    metadata = await repository.finish_turn(context, "答复[1]", citations=[citation])
    assert metadata["citations"][0]["source_key"] == expected_source
    assert await repository.finish_turn(context, "答复[1]", citations=[citation]) == metadata
