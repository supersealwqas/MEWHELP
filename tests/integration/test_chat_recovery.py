"""在真实 MySQL 和应用 lifespan 中验证重启恢复，不以函数替身冒充恢复流程。"""

import pytest

from app.schemas.tools import ToolContext
from tests.recovery_fakes import assert_application_recovery_contract

pytestmark = pytest.mark.mysql


@pytest.mark.parametrize("with_result", [True, False])
async def test_new_application_recovers_committed_facts_without_reexecution(
    database, repositories, monkeypatch, tmp_path, with_result
):
    """按当前图的模型边界验证真实MySQL恢复，不读取默认知识与checkpoint资源。

    参数：database隔离mewhelp_test、repositories真实仓储、monkeypatch替换器、tmp_path独占checkpoint、with_result提交结果开关。
    返回：None，复用离线同契约的事实回灌、后续答复与不重放断言。
    异常：AssertionError或MySQL/SQLite连接错误。
    """
    await assert_application_recovery_contract(database, repositories, monkeypatch, tmp_path, with_result)


async def test_history_read_preserves_other_running_turn(repositories):
    """正常历史查询不能取消另一会话。参数：仓库；返回：None；异常：断言失败。"""
    chat, _ = repositories
    a = await chat.start_turn(ToolContext(session_id="a", user_id="u", turn_id="a-turn", user_message="a"), "a")
    b = await chat.start_turn(ToolContext(session_id="b", user_id="u", turn_id="b-turn", user_message="b"), "b")
    await chat.load_turns(a.conversation_id)
    turns = await chat.load_turns(b.conversation_id)
    assert len(turns) == 1 and turns[0].turn_status == "running" and len(turns[0].messages) == 1
