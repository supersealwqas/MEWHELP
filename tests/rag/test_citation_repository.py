"""用真实SQL事务核验引用持久化、归属隔离及完成重试。"""

from contextlib import asynccontextmanager
from functools import partial
import hashlib
import json

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.knowledge.answering import EvidenceCitation
from app.knowledge.documents import build_document_chunks
from app.knowledge.manifest import load_manifest
from app.knowledge.rerank import RankedChunk
from app.knowledge.schemas import ChunkSnapshot
from app.knowledge.sources import read_source, verified_source_revision
from app.knowledge.text import content_fingerprint
from app.repositories.chat import ChatRepository, PersistenceUncertain
from app.schemas.tools import ToolContext
from app.workflow.routing import citations_from_ranked


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
        connection.execute(text("CREATE TABLE conversations (id INTEGER PRIMARY KEY, session_id TEXT, user_id TEXT, status TEXT DEFAULT '进行中', summary TEXT, summary_upto_msg_id INTEGER, layer1_from_msg_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
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


@pytest.mark.parametrize("legacy_snapshot", [False, True])
@pytest.mark.parametrize("answer,expected_source", [("权威正文", "qa:10"), ("被改写的正文", None)])
async def test_source_supplement_requires_exact_authoritative_body(repository, answer, expected_source, legacy_snapshot):
    """来源补齐不破坏原始引用幂等，旧快照也保留兼容。参数：仓储/正文/期望/旧快照标记；返回：None；异常：断言错误。"""
    context = await repository.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="问"), "问")
    citation = EvidenceCitation(n=1, chunk_id=20, question="来源问法", answer=answer)
    metadata = await repository.finish_turn(context, "答复[1]", citations=[citation])
    assert metadata["citations"][0]["source_key"] == expected_source
    if legacy_snapshot:
        from sqlalchemy import select
        from app.models.customer_service import Message
        async with repository.factory.begin() as session:
            row = await session.scalar(select(Message).where(Message.role == "assistant"))
            # 以新dict赋值触发SQLAlchemy JSON字段持久化，模拟修复前没有原始快照的记录。
            metadata = {key: value for key, value in row.rag_metadata.items() if key != "citation_inputs"}
            row.rag_metadata = metadata
    assert await repository.finish_turn(context, "答复[1]", citations=[citation]) == metadata


async def test_legacy_snapshot_still_rejects_erasing_saved_section(repository):
    """无原始快照的旧记录也不能把真实章节悄悄改为空。参数：repository真实SQL仓储；返回：None；异常：断言失败。"""
    from sqlalchemy import select
    from app.models.customer_service import Message
    context = await repository.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="政策"), "政策")
    citation = EvidenceCitation(n=1, chunk_id=5, question="条件", answer="完整原文", section_path="退货/七天", source_key="qa:8")
    await repository.finish_turn(context, "答复[1]", citations=[citation])
    async with repository.factory.begin() as session:
        row = await session.scalar(select(Message).where(Message.role == "assistant"))
        row.rag_metadata = {key: value for key, value in row.rag_metadata.items() if key != "citation_inputs"}
    with pytest.raises(PersistenceUncertain):
        await repository.finish_turn(context, "答复[1]", citations=[citation.model_copy(update={"section_path": None})])


@pytest.fixture
async def document_snapshot(repository, tmp_path, monkeypatch):
    """准备真实文件、完整权威块及生产引用。参数：仓储/独占目录/替换器；返回：文件、清单、正文、引用；异常：IO/SQL错误。"""
    from app.repositories import chat
    body = "# 示例政策\n\n## 条件\n\n示例商品需保留包装。\n"
    document = tmp_path / "policy.md"
    document.write_bytes(body.encode("utf-8"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [{"path": "policy.md", "content_type": "policy"}]}), encoding="utf-8")
    draft = build_document_chunks(body, load_manifest(manifest, root=tmp_path)[0], len)[0]
    async with repository.factory.begin() as session:
        await session.execute(text("DELETE FROM knowledge_chunks"))
        await session.execute(text("INSERT INTO knowledge_chunks VALUES (20, 'doc:policy.md', 0, '条件', '示例商品需保留包装。', '示例政策', '示例政策 / 条件', 1)"))
    # 仅替换服务器白名单位置，文件版本和整份切块核验仍执行真实算法。
    monkeypatch.setattr(chat, "verified_source_revision", partial(verified_source_revision, manifest_path=manifest))
    chunk = ChunkSnapshot(**draft.model_dump(), id=20, content_hash=content_fingerprint(draft.text),
                          embedding_revision="test-v1", vectorize_status="done", vector_id="20")
    return document, manifest, body, citations_from_ranked([RankedChunk(chunk=chunk, score=0.9)])[0]


@pytest.mark.parametrize("existing_source", [False, True])
async def test_new_document_citation_detects_later_source_change(repository, document_snapshot, existing_source):
    """已带来源的新引用也保存文件版本，原件变化后可识别且重试不改旧证据。参数：仓储/文档/来源标记；返回：None；异常：断言失败。"""
    document, manifest, body, citation = document_snapshot
    if not existing_source:
        citation = citation.model_copy(update={"source_key": None})
    context = await repository.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="退货条件"), "退货条件")
    saved = await repository.finish_turn(context, "需保留包装。[1]", citations=[citation])
    original = saved["citations"][0]
    assert original["source_revision"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert read_source(original, manifest_path=manifest)["source_changed"] is False
    document.write_bytes(body.replace("需保留包装", "无需保留包装").encode("utf-8"))
    assert read_source(original, manifest_path=manifest)["source_changed"] is True
    assert original["answer"] == "示例商品需保留包装。"
    assert await repository.finish_turn(context, "需保留包装。[1]", citations=[citation]) == saved


@pytest.mark.parametrize("mismatch", ["file", "body", "source", "missing"])
async def test_document_version_is_not_invented_for_changed_evidence(repository, document_snapshot, mismatch):
    """来源或完整正文不一致时不能补造原文版本。参数：仓储/文档/变化类型；返回：None；异常：断言失败。"""
    document, _, body, citation = document_snapshot
    if mismatch == "file":
        document.write_bytes(body.replace("需保留包装", "无需保留包装").encode("utf-8"))
    else:
        statements = {"body": "UPDATE knowledge_chunks SET answer='新正文' WHERE id=20",
                      "source": "UPDATE knowledge_chunks SET source_key='doc:other.md' WHERE id=20",
                      "missing": "DELETE FROM knowledge_chunks WHERE id=20"}
        async with repository.factory.begin() as session:
            await session.execute(text(statements[mismatch]))
    context = await repository.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="退货条件"), "退货条件")
    saved = await repository.finish_turn(context, "需保留包装。[1]", citations=[citation])
    original = saved["citations"][0]
    assert original["source_key"] == "doc:policy.md"
    assert original["source_revision"] is None
    assert original["answer"] == "示例商品需保留包装。"


async def test_document_citation_keeps_its_existing_revision(repository, document_snapshot):
    """已有旧版本不能被完成保存时的当前文件版本覆盖。参数：仓储/文档；返回：None；异常：断言失败。"""
    _, manifest, _, citation = document_snapshot
    citation = citation.model_copy(update={"source_revision": "a" * 64})
    context = await repository.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="退货条件"), "退货条件")
    saved = await repository.finish_turn(context, "需保留包装。[1]", citations=[citation])
    assert saved["citations"][0]["source_revision"] == "a" * 64
    assert read_source(saved["citations"][0], manifest_path=manifest)["source_changed"] is True


@pytest.mark.parametrize("existing_source", [False, True])
async def test_withdrawn_document_chunk_cannot_receive_current_file_revision(repository, document_snapshot, existing_source):
    """文档缩短后保留的旧尾块不能被赋予当前文件版本。参数：仓储/当前文档/旧来源是否已知；返回：None；异常：断言失败。"""
    _, _, _, citation = document_snapshot
    # 当前文件及active块仍完全一致，但生成时看到的另一块已被发布事务撤回。
    async with repository.factory.begin() as session:
        await session.execute(text("INSERT INTO knowledge_chunks VALUES (21, 'doc:policy.md', 1, '旧规则', '需要保留旧票据。', '示例政策', '示例政策 / 旧规则', 0)"))
    citation = citation.model_copy(update={"chunk_id": 21, "question": "旧规则", "answer": "需要保留旧票据。",
                                          "section_path": "示例政策 / 旧规则", "source_key": "doc:policy.md" if existing_source else None})
    context = await repository.start_turn(ToolContext(session_id="s", user_id="u", turn_id="t", user_message="旧规则"), "旧规则")
    saved = await repository.finish_turn(context, "需要保留旧票据。[1]", citations=[citation])
    original = saved["citations"][0]
    assert original["source_revision"] is None
    assert original["source_key"] == ("doc:policy.md" if existing_source else None)
    assert original["answer"] == "需要保留旧票据。"
