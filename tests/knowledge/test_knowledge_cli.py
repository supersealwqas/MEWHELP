"""知识库 CLI 测试库切换契约。"""

from __future__ import annotations

from app.knowledge.cli import _config


def test_test_flag_switches_both_stores() -> None:
    """--test 同时切换 MySQL 与 Milvus，不能半边测试半边应用。参数：无；返回：None；异常：无。"""
    base = _config(False)
    test = _config(True)
    assert test.mysql_database == base.knowledge_test_database
    assert test.milvus_database == base.milvus_test_database
    assert test.mysql_test_database == base.mysql_test_database
