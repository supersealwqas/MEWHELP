"""验证工具执行器的限时、有限重试、持久化顺序及取消。"""

import asyncio
import pytest
from langchain_core.tools import tool
from app.repositories.chat import PersistenceUncertain
from app.schemas.tools import ToolContext, ToolResult
from app.tools.registry import ToolRegistry, build_registry
from app.tools.executor import ToolExecutor


class ResultStore:
    """执行器单测的存储边界替身，不用于应用运行。"""
    def __init__(self, result=None):
        """初始化已提交结果。参数：可选原结果；返回：None；异常：无。"""
        self.result, self.writes = result, 0

    async def get_tool_result(self, context):
        """读取已提交事实。参数：调用身份；返回：结果或空；异常：无。"""
        return self.result

    async def save_tool_result(self, context, result):
        """模拟唯一结果持久化。参数：身份和结果；返回：原结果；异常：无。"""
        if self.result is None:
            self.result, self.writes = result, self.writes + 1
        return self.result


def context():
    """生成服务端身份。参数：无；返回：工具上下文；异常：无。"""
    return ToolContext(session_id="s", conversation_id=1, user_id="u", turn_id="t",
                       tool_call_id="call-1", user_message="邮费是多少")


def call(name="slow", args=None):
    """生成匹配调用。参数：名称和参数；返回：申请字典；异常：无。"""
    return {"id": "call-1", "name": name, "args": args or {}}


async def ignore_status(status):
    """测试无需展示状态。参数：状态字典；返回：None；异常：无。"""


async def test_timeout_retry_once():
    """超时只尝试两次，最终仅存一个失败。参数：无；返回：None；异常：断言失败。"""
    attempts, states = [], []
    @tool
    async def slow() -> dict:
        """模拟超时边界。参数：无；返回：不会返回；异常：取消。"""
        attempts.append(1)
        await asyncio.Event().wait()
    registry, store = ToolRegistry(), ResultStore()
    registry.register(slow)
    async def status(frame):
        """记录执行顺序。参数：状态帧；返回：None；异常：断言失败。"""
        if frame["status"] == "failed":
            assert store.result is not None
        states.append(frame["status"])
    result = await ToolExecutor(registry, store, None, timeout_seconds=0.01, retry_delay_seconds=0).execute(call(), context(), status)
    assert len(attempts) == 2 and store.writes == 1
    assert not result.ok and result.error.code == "TOOL_TIMEOUT"
    assert states == ["running", "retrying", "failed"]


async def test_validation_and_not_found_not_retried():
    """参数错误和零命中不重试。参数：无；返回：None；异常：断言失败。"""
    class Support:
        """受控 FAQ 数据库边界。"""
        count = 0
        async def query_faq(self, keyword):
            """返回空查询。参数：原文关键词；返回：列表；异常：无。"""
            self.count += 1
            return []
    support = Support()
    for args, expected in [({"keyword": "邮费"}, True), ({"keyword": "运费"}, False), ({"keyword": " "}, False)]:
        store = ResultStore()
        result = await ToolExecutor(build_registry(), store, support).execute(call("query_faq", args), context(), ignore_status)
        assert result.ok is expected and store.writes == 1
    assert support.count == 1


async def test_cancel_during_backoff():
    """取消退避后不再执行或保存失败。参数：无；返回：None；异常：断言失败。"""
    attempts, retrying = [], asyncio.Event()
    @tool
    async def slow() -> dict:
        """模拟可重试超时。参数：无；返回：不返回；异常：TimeoutError。"""
        attempts.append(1)
        raise TimeoutError()
    registry, store = ToolRegistry(), ResultStore()
    registry.register(slow)
    async def status(frame):
        """通知已进入退避。参数：状态帧；返回：None；异常：无。"""
        if frame["status"] == "retrying":
            retrying.set()
    task = asyncio.create_task(ToolExecutor(registry, store, None, retry_delay_seconds=1).execute(call(), context(), status))
    await asyncio.wait_for(retrying.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(attempts) == 1 and store.writes == 0


async def test_error_redacted():
    """原始异常不能回灌或写入结果。参数：无；返回：None；异常：断言失败。"""
    @tool
    async def slow() -> dict:
        """模拟不可重试异常。参数：无；返回：不返回；异常：RuntimeError。"""
        raise RuntimeError("SECRET mysql://password@host SELECT * FROM tickets")
    registry, store = ToolRegistry(), ResultStore()
    registry.register(slow)
    result = await ToolExecutor(registry, store, None).execute(call(), context(), ignore_status)
    assert not result.ok
    assert "SECRET" not in result.model_dump_json() and "SELECT" not in result.model_dump_json()


async def test_reuse_persisted_result():
    """已提交结果不再调用业务。参数：无；返回：None；异常：断言失败。"""
    original = ToolResult(ok=True, data={"ticket_no": "T-original"})
    store = ResultStore(original)
    result = await ToolExecutor(build_registry(), store, None).execute(
        call("create_ticket", {"description": "投诉", "ticket_type": "投诉"}), context(), ignore_status)
    assert result == original and store.writes == 0


async def test_unavailable_verification_never_executes():
    """无法核实提交状态就安全结束。参数：无；返回：None；异常：断言失败。"""
    class UnavailableStore(ResultStore):
        """数据库故障边界。"""
        async def get_tool_result(self, context):
            """模拟读取失败。参数：上下文；返回：不返回；异常：ConnectionError。"""
            raise ConnectionError("secret")
    store = UnavailableStore()
    with pytest.raises(PersistenceUncertain):
        await ToolExecutor(build_registry(), store, None).execute(call("create_ticket"), context(), ignore_status)
    assert store.writes == 0
