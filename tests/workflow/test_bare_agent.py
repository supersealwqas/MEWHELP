"""验证裸 SDK 循环的真实消息、只读执行边界、预算和命令行资源选择。"""

import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from app.schemas.tools import ToolContext
from app.tools.business import create_ticket, query_logistics, query_order, query_product
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from tests.chat_fakes import MemoryRepository


def tool_call(name, args, call_id="call-1"):
    """构造 SDK 原始申请。参数：name 工具名、args 参数字典或原字符串、call_id 唯一 ID；返回：dict；异常：序列化错误。"""
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def completion(content=None, calls=None, *, finish_reason=None, usage=True):
    """构造完整非流式响应。参数：content 正文、calls 申请、finish_reason 停止原因、usage 是否有用量；返回：dict；异常：无。"""
    return {"id": "chatcmpl-offline", "created": 0, "model": "test-model", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                         "tool_calls": calls, "refusal": None}, "logprobs": None,
                         "finish_reason": finish_reason or ("tool_calls" if calls else "stop")}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120} if usage else None}


@pytest.fixture
async def bare_runtime():
    """仅替代模型网络与数据库。参数：无；返回：真实 SDK、执行器与内存审计；异常：夹具或 SDK 错误。"""
    requests, responses = [], []

    async def handle(request):
        """记录真实 SDK 发包并响应。参数：request HTTP 请求；返回：httpx.Response；异常：无脚本响应时断言失败。"""
        requests.append(json.loads(request.content))
        assert responses, "不应追加预算外的模型请求"
        response = responses.pop(0)
        if callable(response):
            response = response(requests[-1])
        return httpx.Response(200, json=response)

    registry = ToolRegistry()
    for tool in (query_order, query_logistics, query_product):
        registry.register(tool)
    repository = MemoryRepository()
    context = await repository.start_turn(ToolContext(session_id="bare-test", user_id="demo-user",
        turn_id="bare-turn", user_message="订单 1001 的物流到哪了"), "订单 1001 的物流到哪了")
    async with AsyncOpenAI(api_key="offline-key", base_url="http://offline.invalid/v1",
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
        yield SimpleNamespace(client=client, executor=ToolExecutor(registry, repository, None),
                              registry=registry, context=context, repository=repository,
                              requests=requests, responses=responses)


async def run(runtime, question="订单 1001 的物流到哪了", **limits):
    """调用待测真实循环。参数：runtime 运行夹具、question 用户问题、limits 限制；返回：BareAgentResult；异常：透传实现异常。"""
    from examples.ch05_bare_agent import run_bare_agent

    return await run_bare_agent(question, client=runtime.client, executor=runtime.executor,
                               registry=runtime.registry, context=runtime.context, **limits)


async def test_bare_loop_feeds_order_tracking_into_next_call(bare_runtime):
    """漏传订单事实或工具 ID 时失败。参数：bare_runtime 隔离运行夹具；返回：None；异常：断言失败。"""
    rt = bare_runtime
    first = tool_call("query_order", {"order_id": "1001"})

    def request_logistics(request):
        """按实际收到的订单结果生成后续申请。参数：request 请求；返回：dict 响应；异常：缺少订单结果时失败。"""
        result_message = request["messages"][-1]
        assert result_message["role"] == "tool" and result_message["tool_call_id"] == first["id"]
        tracking = json.loads(result_message["content"])["data"]["tracking_no"]
        return completion(calls=[tool_call("query_logistics", {"tracking_no": tracking}, "call-2")])

    rt.responses.extend([completion(calls=[first]), request_logistics, completion("包裹运输中")])
    result = await run(rt)
    messages = rt.repository.turns[rt.context.turn_id].messages
    calls = [call for message in messages for call in (message.tool_calls or [])]
    first_tool_result = rt.repository.results[(rt.context.turn_id, "call-1")]
    assert result.tool_names == ["query_order", "query_logistics"]
    assert calls[1]["args"]["tracking_no"] == first_tool_result.data["tracking_no"]
    assert first_tool_result.ok and first_tool_result.is_mock
    assert result.answer == "包裹运输中" and result.steps == 3 and result.stop_reason == "answer"
    assert "is_mock" not in json.loads(rt.requests[1]["messages"][-1]["content"])
    assert rt.requests[0]["stream"] is False and rt.requests[0]["max_completion_tokens"] == 2048


async def test_bare_loop_stops_without_tool_calls(bare_runtime):
    """终止后仍追加模型或工具请求时失败。参数：bare_runtime 隔离运行夹具；返回：None；异常：断言失败。"""
    bare_runtime.responses.append(completion("你好，有什么可以帮您？"))
    result = await run(bare_runtime, "你好")
    assert result.answer == "你好，有什么可以帮您？" and result.steps == 1
    assert result.stop_reason == "answer" and result.tool_names == []
    assert len(bare_runtime.requests) == 1 and bare_runtime.repository.results == {}
    assert result.usage[0]["total_tokens"] == 120


async def test_bare_loop_returns_clarification_with_zero_tools_and_missing_usage(bare_runtime):
    """追问被误当失败或缺失 usage 被伪造为零时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    bare_runtime.responses.append(completion("请提供订单号或物流单号。", usage=False))
    result = await run(bare_runtime, "我要查物流")
    assert result.answer == "请提供订单号或物流单号。" and result.stop_reason == "answer"
    assert result.tool_names == [] and result.usage == [None]


async def test_batch_results_keep_each_call_id_and_order(bare_runtime):
    """批量申请漏结果、错 ID 或提前再次决策时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    rt = bare_runtime
    rt.responses.extend([completion(calls=[tool_call("query_order", {"order_id": "1001"}),
        tool_call("query_product", {"keyword": "猫粮"}, "call-2")]), completion("查询完成")])
    result = await run(rt)
    submitted = rt.requests[1]["messages"]
    assert [message["tool_call_id"] for message in submitted[-2:]] == ["call-1", "call-2"]
    assert all(message["role"] == "tool" for message in submitted[-2:])
    assert result.tool_names == ["query_order", "query_product"]
    assert len(rt.repository.results) == 2


async def test_model_request_limit_stops_without_an_extra_request(bare_runtime):
    """超过模型次数上限时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    rt = bare_runtime
    rt.responses.append(completion(calls=[tool_call("query_order", {"order_id": "1001"})]))
    result = await run(rt, max_steps=1)
    assert result.stop_reason == "max_steps" and result.steps == 1
    assert result.tool_names == ["query_order"] and len(rt.repository.results) == 1


async def test_tool_limit_rejects_whole_batch_with_paired_results(bare_runtime):
    """超限批次部分执行或丢弃拒绝结果时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    rt = bare_runtime
    rt.responses.append(completion(calls=[tool_call("query_order", {"order_id": "1001"}),
        tool_call("query_product", {"keyword": "猫粮"}, "call-2")]))
    result = await run(rt, max_tool_calls=1)
    assert result.stop_reason == "max_tool_calls" and result.tool_names == []
    assert len(rt.repository.results) == 2
    assert all(not saved.ok for saved in rt.repository.results.values())


@pytest.mark.parametrize("raw", ['{"order_id":"1001"', "[]", '{"order_id":NaN}',
    '{"order_id":1e999}', '{"order_id":"1001","order_id":"2003"}'])
async def test_invalid_json_is_rejected_before_any_tool_execution(bare_runtime, raw):
    """宽松补全、非有限数值或重复键被接受时失败。参数：bare_runtime 夹具、raw 非法原文；返回：None；异常：断言失败。"""
    rt = bare_runtime
    rt.responses.append(completion(calls=[tool_call("query_order", raw)]))
    result = await run(rt)
    assert result.stop_reason == "protocol_error" and result.tool_names == []
    saved = rt.repository.results[(rt.context.turn_id, "call-1")]
    assert not saved.ok and saved.error.code == "INVALID_TOOL_ARGUMENTS"


@pytest.mark.parametrize("name", ["unknown_tool", "create_ticket"])
async def test_unknown_or_write_tool_rejects_entire_batch(bare_runtime, name):
    """模型的未知/写申请被执行或连带放行同批其他工具时失败。参数：bare_runtime 夹具、name 工具名；返回：None；异常：断言失败。"""
    rt = bare_runtime
    rt.responses.append(completion(calls=[tool_call("query_order", {"order_id": "1001"}),
        tool_call(name, {"description": "投诉", "ticket_type": "投诉"}, "call-2")]))
    result = await run(rt)
    assert result.stop_reason == "protocol_error" and result.tool_names == []
    assert len(rt.repository.results) == 2 and all(not item.ok for item in rt.repository.results.values())


async def test_identity_is_hidden_and_model_cannot_override_owner(bare_runtime):
    """公开 schema 暴露身份或接受模型伪造身份时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    rt = bare_runtime
    rt.responses.append(completion(calls=[tool_call("query_order", {"order_id": "2003", "user_id": "demo-other-user"})]))
    result = await run(rt)
    visible = json.dumps(rt.requests[0]["tools"])
    assert "user_id" not in visible and "conversation_id" not in visible
    assert {tool["function"]["name"] for tool in rt.requests[0]["tools"]} == {
        "query_order", "query_logistics", "query_product"}
    assert result.stop_reason == "protocol_error" and result.tool_names == []
    assert all(not item.ok for item in rt.repository.results.values())


async def test_executor_injects_actual_owner_for_unowned_order(bare_runtime):
    """省略可信身份注入或查到他人订单时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    rt = bare_runtime
    rt.responses.extend([completion(calls=[tool_call("query_order", {"order_id": "2003"})]),
                         completion("没有找到您的这笔订单")])
    result = await run(rt)
    assert result.tool_names == ["query_order"]
    saved = rt.repository.results[(rt.context.turn_id, "call-1")]
    assert not saved.ok and saved.error.code == "order_not_owned"


async def test_unsafe_registry_is_rejected_before_model_request(bare_runtime):
    """误用含建单工具的全量注册表时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    bare_runtime.registry.register(create_ticket)
    with pytest.raises(ValueError, match="只读"):
        await run(bare_runtime)
    assert bare_runtime.requests == [] and bare_runtime.repository.results == {}


async def test_input_budget_stops_before_sending_or_truncating_question(bare_runtime):
    """超长当前事实被截断或仍发给模型时失败。参数：bare_runtime 夹具；返回：None；异常：断言失败。"""
    result = await run(bare_runtime, "很长的问题" * 3000)
    assert result.stop_reason == "input_budget" and result.steps == 0
    assert bare_runtime.requests == [] and result.usage == []


@pytest.mark.parametrize("response,reason", [
    (completion("未完成的正文", finish_reason="length"), "output_limit"),
    (completion(""), "protocol_error"),
    (completion(calls=[tool_call("query_order", {"order_id": "1001"})], finish_reason="stop"), "protocol_error"),
    (completion(calls=[tool_call("query_order", {"order_id": "1001"}),
                       tool_call("query_product", {"keyword": "猫粮"})]), "protocol_error"),
])
async def test_incomplete_or_ambiguous_protocol_never_executes(bare_runtime, response, reason):
    """截断响应、空答、终态冲突或重复 ID 被接受时失败。参数：bare_runtime 夹具、response 响应、reason 原因；返回：None；异常：断言失败。"""
    bare_runtime.responses.append(response)
    result = await run(bare_runtime)
    assert result.stop_reason == reason and result.tool_names == []
    assert len(bare_runtime.requests) == 1


@pytest.mark.parametrize("missing_field", ["function", "name", "arguments"])
async def test_missing_tool_structure_returns_protocol_error_with_diagnostic(bare_runtime, missing_field):
    """SDK 宽松接受缺字段后被误当资源故障时失败。

    参数：bare_runtime 真实 SDK 与执行器夹具；missing_field 本次删除的申请字段。
    返回：None。
    异常：断言失败，或未被产品正确分类的异常。
    """
    rt = bare_runtime
    malformed = tool_call("query_order", {"order_id": "1001"})
    if missing_field == "function":
        del malformed[missing_field]
    else:
        del malformed["function"][missing_field]
    rt.responses.append(completion(calls=[malformed]))
    result = await run(rt)
    assert result.stop_reason == "protocol_error" and result.tool_names == []
    assert result.steps == 1 and rt.repository.results == {}
    diagnostic = rt.repository.turns[rt.context.turn_id].messages[-1].tool_calls
    assert diagnostic == [{"type": "invalid_tool_decision", "raw_tool_calls": [malformed]}]


async def test_cli_missing_function_is_protocol_exit_two(bare_runtime, monkeypatch, capsys):
    """缺字段响应误返回资源失败退出码 3 时失败，循环和 SDK 保持真实。

    参数：bare_runtime 外部边界夹具；monkeypatch 替换数据库/客户端构造；capsys 捕获 CLI 输出。
    返回：None。
    异常：断言失败。
    """
    import openai
    from app import config
    from app.db import session
    from app.repositories import chat
    from examples.ch05_bare_agent import build_parser, run_cli

    async def scalar(statement):
        """替代测试库身份查询。参数：statement SQL；返回：str 测试库名；异常：无。"""
        return "mewhelp_test"

    @asynccontextmanager
    async def connect():
        """替代数据库连接边界。参数：无；返回：异步连接上下文；异常：无。"""
        yield SimpleNamespace(scalar=scalar)

    async def dispose():
        """释放无外部资源的连接替身。参数：无；返回：None；异常：无。"""

    def database(settings, *, database):
        """提供隔离数据库边界。参数：settings 配置、database 目标库；返回：引擎与工厂；异常：无。"""
        return SimpleNamespace(connect=connect, dispose=dispose), None

    def repository(factory):
        """复用既有内存仓储。参数：factory 会话工厂；返回：MemoryRepository；异常：无。"""
        return bare_runtime.repository

    def client(**options):
        """复用真实 SDK。参数：options CLI 构造选项；返回：AsyncOpenAI；异常：无。"""
        return bare_runtime.client

    def settings(**options):
        """提供已有离线配置。参数：options 环境选项；返回：Settings；异常：无。"""
        return config.settings

    monkeypatch.setattr(config, "Settings", settings)
    monkeypatch.setattr(session, "create_database", database)
    monkeypatch.setattr(chat, "ChatRepository", repository)
    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    bare_runtime.responses.append(completion(calls=[{"id": "missing-function", "type": "function"}]))
    code = await run_cli(build_parser().parse_args(["--test", "--question", "查订单 1001"]))
    output = json.loads(capsys.readouterr().out)
    assert code == 2 and output["stop_reason"] == "protocol_error" and output["tool_names"] == []
    turn = bare_runtime.repository.turns[bare_runtime.repository.contexts[-1].turn_id]
    assert turn.turn_status == "failed" and turn.messages[-1].tool_calls[0]["type"] == "invalid_tool_decision"


def test_cli_requires_explicit_resource_mode_without_loading_credentials():
    """缺少资源模式时默认触达应用库或 --help 依赖配置时失败。参数：无；返回：None；异常：断言失败。"""
    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CHAT_", "MYSQL_"))}
    help_result = subprocess.run([sys.executable, "-m", "examples.ch05_bare_agent", "--help"],
                                 cwd=root, env=env, capture_output=True, text=True)
    assert help_result.returncode == 0 and "--test" in help_result.stdout and "--app" in help_result.stdout
    missing = subprocess.run([sys.executable, "-m", "examples.ch05_bare_agent", "--question", "你好"],
                             cwd=root, env=env, capture_output=True, text=True)
    assert missing.returncode == 2 and "--test" in missing.stderr and "Traceback" not in missing.stderr
