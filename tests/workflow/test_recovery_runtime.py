"""离线恢复回归：真实SQLAlchemy业务仓储、官方SQLite保存器和主应用生命周期。"""

import pytest

from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from tests.recovery_fakes import assert_application_recovery_contract, offline_recovery_database


@pytest.mark.parametrize("with_result", [True, False])
async def test_main_lifespan_recovers_committed_facts_without_reexecution(tmp_path, monkeypatch, with_result):
    """旧模型注入失效、未提交事实回灌或重启重放时失败。

    参数：tmp_path独占目录、monkeypatch恢复边界、with_result提交结果开关。
    返回：None，离线运行与MySQL验收相同恢复契约。
    异常：AssertionError或本地数据库错误。
    """
    async with offline_recovery_database(tmp_path) as database:
        repositories = ChatRepository(database[1]), SupportRepository(database[1])
        await assert_application_recovery_contract(database, repositories, monkeypatch, tmp_path, with_result)
