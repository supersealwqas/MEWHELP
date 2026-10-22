"""知识库 CLI 测试库切换契约。"""

from __future__ import annotations

import sys
import pytest
from app.knowledge import cli
from app.knowledge.cli import _config
from app.knowledge import runtime
from app.knowledge.vector import VectorIndexUnavailable
from tests.knowledge.fakes import make_settings


def test_test_flag_switches_both_stores() -> None:
    """--test 同时切换 MySQL 与 Milvus，不能半边测试半边应用。参数：无；返回：None；异常：无。"""
    base = _config(False)
    test = _config(True)
    assert test.mysql_database == base.knowledge_test_database
    assert test.milvus_database == base.milvus_test_database
    assert test.mysql_test_database == base.mysql_test_database


@pytest.mark.parametrize("confirmed_database", [None, "mewhelp_ch03"])
def test_init_cli_forwards_explicit_database_confirmation(monkeypatch, confirmed_database):
    """初始化入口仅传递操作人明确给出的库名，不自动确认归属。

    参数：monkeypatch 为替换器，confirmed_database 为省略或明确指定的目标名。
    返回：None，断言实际 CLI 传递匹配的配置与确认；异常：AssertionError 表示误确认或传递丢失。
    """
    async def initialize(config, *, confirmed_database=None):
        """替换外部初始化。参数：配置、确认库名；返回：None；异常：无。"""
        observed.append((config.milvus_database, confirmed_database))

    observed = []
    arguments = ["knowledge", "init"]
    if confirmed_database:
        arguments += ["--confirm-milvus-database", confirmed_database]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(cli, "initialize_knowledge", initialize)
    cli.main()
    assert observed == [("mewhelp_ch03", confirmed_database)]


async def test_initialization_rejects_wrong_confirmation_before_database_access(monkeypatch):
    """错误库名确认必须在连接或创建 MySQL 表之前拒绝。

    参数：monkeypatch 为替换器；返回：None；异常：AssertionError 表示错误确认产生外部副作用。
    """
    def fail_database(config):
        """禁止错误确认访问数据库。参数：配置；返回：无；异常：AssertionError。"""
        raise AssertionError("确认不匹配时不能连接数据库")

    monkeypatch.setattr(runtime, "create_database", fail_database)
    with pytest.raises(VectorIndexUnavailable, match="确认"):
        await runtime.initialize_knowledge(make_settings(), confirmed_database="foreign-project")
