"""引用JSON独立增量迁移的可重复性和拒绝不兼容结构测试。"""

import pytest
from sqlalchemy import create_engine, inspect, text


def test_migration_only_adds_nullable_json_and_is_repeatable():
    """旧messages只增加可空JSON，不改质量表或旧行。参数：无；返回：None；异常：断言错误。"""
    from app.db.migrate_citations import migrate_connection
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)"))
        connection.execute(text("INSERT INTO messages(content) VALUES ('旧消息')"))
        assert migrate_connection(connection) == ["messages.rag_metadata"]
        assert migrate_connection(connection) == []
        assert inspect(connection).get_table_names() == ["messages"]
        assert connection.execute(text("SELECT content, rag_metadata FROM messages")).one() == ("旧消息", None)
    engine.dispose()


def test_migration_rejects_existing_wrong_type_without_altering_it():
    """已有同名非JSON列不能强制覆盖。参数：无；返回：None；异常：断言错误。"""
    from app.db.migrate_citations import CitationSchemaError, migrate_connection
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE messages (id INTEGER PRIMARY KEY, rag_metadata TEXT)"))
        with pytest.raises(CitationSchemaError):
            migrate_connection(connection)
        assert str(inspect(connection).get_columns("messages")[1]["type"]) == "TEXT"
    engine.dispose()
