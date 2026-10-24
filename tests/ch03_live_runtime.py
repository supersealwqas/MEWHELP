"""CH03真实聊天运行时、TCP SSE采样及当前轮持久化证据校验。"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import httpx
import uvicorn
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.llm import ChatModelAdapter
from app.db.session import create_database
from app.knowledge.embedding import BgeM3Encoder
from app.knowledge.search import KnowledgeSearch
from app.knowledge.store import KnowledgeStore
from app.knowledge.vector import MilvusVectorIndex
from app.main import create_app
from app.models.customer_service import Base as CustomerBase, Conversation, Message
from app.models.knowledge import KnowledgeBase
from app.repositories.chat import ChatRepository
from app.repositories.support import SupportRepository
from app.schemas.tools import KeywordArgs, ToolResult
from app.services.chat import ChatService
from app.tools.registry import build_registry


def inspect_exchange(case: dict, observed: dict) -> dict:
    """检查聊天证据。参数：固定案例/采样；返回：含失败项的证据；异常：无。"""
    failures = list(observed.get("failures", []))
    events = observed.get("events", [])
    messages = observed.get("messages", [])
    answer = "".join(event["data"]["delta"] for event in events
                     if event.get("event") == "delta" and isinstance(event.get("data"), dict)
                     and isinstance(event["data"].get("delta"), str))
    result = {**observed, "answer": answer, "hits": [], "failures": failures,
              "raw_keyword": None, "effective_keyword": None, "tool_pairs": [],
              "executed_keyword": None,
              "delta_count": sum(event.get("event") == "delta" for event in events)}
    if observed.get("http_status") != 200 or not observed.get("content_type", "").startswith("text/event-stream"):
        failures.append("HTTP状态或SSE类型错误")
    done = [index for index, event in enumerate(events) if event.get("event") == "done"]
    if done != [len(events) - 1]:
        failures.append("[DONE]必须唯一且是末帧")
    if not answer.strip():
        failures.append("没有非空文本增量")
    if any(event.get("event") not in {"delta", "done", "tool_status", "answer_metadata"} for event in events):
        failures.append("SSE出现error或非法事件")
    metadata = [(index, event.get("data", {})) for index, event in enumerate(events) if event.get("event") == "answer_metadata"]
    if metadata and (len(metadata) != 1 or metadata[0][0] != len(events) - 2
                     or metadata[0][1].get("status") != "completed"
                     or not isinstance(metadata[0][1].get("citations"), list)
                     or not messages or metadata[0][1].get("turn_id") != messages[0].get("turn_id")
                     or (observed.get("session_id") and metadata[0][1].get("session_id") != observed["session_id"])):
        failures.append("回答完成元数据身份、顺序或结构不一致")
    users = [message for message in messages if message["role"] == "user"]
    finals = [message for message in messages if message["role"] == "assistant" and not message.get("tool_calls")]
    if len(users) != 1 or users[0].get("content") != case["query"]:
        failures.append("当前轮原始用户消息缺失或不一致")
    if not messages or len({message["turn_id"] for message in messages}) != 1 or any(message["turn_status"] != "completed" for message in messages):
        failures.append("当前轮不存在、混轮或未completed")
    if len(finals) != 1 or finals[0].get("content") != answer:
        failures.append("最终落库回答与SSE增量不一致")
    calls = [call for message in messages for call in (message.get("tool_calls") or [])]
    call_ids = {call.get("id") for call in calls}
    for event in events:
        if event.get("event") == "tool_status" and (event["data"].get("tool_call_id") not in call_ids
                                                      or event["data"].get("name") != "query_faq"):
            failures.append("出现未知或未配对的工具状态")
    tool_rows = [message for message in messages if message["role"] == "tool"]
    expect_empty = bool(case.get("expect_empty"))
    if len(calls) != (0 if expect_empty and not calls else 1):
        failures.append("缺少唯一合法query_faq申请")
    if len(tool_rows) != len(calls):
        failures.append("工具申请和结果数量不一致")
    for call in calls:
        if call.get("name") != "query_faq" or not call.get("id"):
            failures.append("出现非query_faq或非法工具申请")
            continue
        try:
            keyword = KeywordArgs.model_validate(call.get("args")).keyword
            result["raw_keyword"] = call["args"]["keyword"]
            result["effective_keyword"] = keyword
            if keyword not in case["query"]:
                failures.append("执行keyword不是用户原文连续片段")
        except (ValueError, TypeError, KeyError):
            failures.append("工具keyword参数非法")
        paired = [message for message in tool_rows if message.get("tool_call_id") == call["id"]]
        if len(paired) != 1:
            failures.append("工具申请未唯一配对结果")
            continue
        try:
            tool_result = ToolResult.model_validate_json(paired[0]["content"])
            if not tool_result.ok or tool_result.error is not None or tool_result.is_mock:
                failures.append("query_faq失败或返回演示数据")
            hits = (tool_result.data or {}).get("hits")
            if not isinstance(hits, list):
                failures.append("工具结果缺少显式hits列表")
            else:
                result["hits"] = hits
                if (tool_result.ok and tool_result.error is None and not tool_result.is_mock
                        and result["effective_keyword"] and result["effective_keyword"] in case["query"]):
                    # 参数合法且真实工具成功后，才有可证明实际执行的编码输入。
                    result["executed_keyword"] = result["effective_keyword"]
                if bool(hits) == expect_empty:
                    failures.append("实际hits与正负例预期不一致")
            result["tool_pairs"].append({"call": call, "result_message_id": paired[0]["id"],
                                        "result": tool_result.model_dump(mode="json")})
        except (ValueError, TypeError):
            failures.append("工具结果不是有效ToolResult JSON")
        succeeded = [index for index, event in enumerate(events)
                     if event.get("event") == "tool_status" and event.get("data", {}).get("tool_call_id") == call["id"]
                     and event["data"].get("name") == "query_faq" and event["data"].get("status") == "succeeded"]
        delta_positions = [index for index, event in enumerate(events) if event.get("event") == "delta"]
        if len(succeeded) != 1 or not delta_positions or succeeded[0] >= min(delta_positions):
            failures.append("成功工具状态未在回答增量之前出现")
    return result


@asynccontextmanager
async def live_runtime(config, policy):
    """组装测试库专用真实服务并释放资源。参数：测试配置/策略；返回：依赖集合；异常：初始化错误。"""
    if config.mysql_database != "mewhelp_ch03_test" or config.milvus_database != "mewhelp_ch03_test":
        raise ValueError("真实验收只允许两个mewhelp_ch03_test库")
    engine = lease_engine = encoder = index = None
    try:
        engine, factory = create_database(config)
        async with engine.begin() as connection:
            if await connection.scalar(text("SELECT DATABASE()")) != "mewhelp_ch03_test":
                raise ValueError("实际连接不是CH03测试库")
            await connection.run_sync(CustomerBase.metadata.create_all)
            await connection.run_sync(KnowledgeBase.metadata.create_all)
        repository = ChatRepository(factory)
        await repository.recover_interrupted_turns()
        lease_engine = create_async_engine(engine.url, poolclass=NullPool, connect_args={"connect_timeout": 5})
        store = KnowledgeStore(factory, lease_engine)
        encoder = BgeM3Encoder(config)
        index = MilvusVectorIndex(config)
        await index.ensure_ready(create=False)
        await encoder.encode(["知识库启动预热"])
        search = KnowledgeSearch(store, encoder, index, policy)
        service = ChatService(repository, SupportRepository(factory, search), ChatModelAdapter(), build_registry(), config)
        yield SimpleNamespace(service=service, factory=factory, store=store, encoder=encoder, index=index)
    finally:
        if encoder is not None:
            await encoder.aclose()
        if index is not None:
            await index.aclose()
        if lease_engine is not None:
            await lease_engine.dispose()
        if engine is not None:
            await engine.dispose()


@asynccontextmanager
async def tcp_server(runtime):
    """启动本任务独占的本机TCP服务器。参数：已组装依赖；返回：本机URL；异常：启动或关闭失败。"""
    @asynccontextmanager
    async def factory():
        """让原create_app使用真实服务。参数：无；返回：服务；异常：透传请求异常。"""
        yield runtime.service

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    address = sock.getsockname()
    server = uvicorn.Server(uvicorn.Config(create_app(factory), lifespan="on", log_level="warning", access_log=False,
                                         timeout_graceful_shutdown=15))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(30):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("验收服务器启动失败")
                await asyncio.sleep(.05)
        yield f"http://{address[0]}:{address[1]}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=20)
        finally:
            sock.close()


def parse_frame(lines: list[str], elapsed: float) -> dict:
    """解析单个SSE帧并保留耗时。参数：帧行/秒数；返回：事件；异常：ValueError协议损坏。"""
    event = "message"
    data = []
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif not line.startswith(":"):
            raise ValueError("未知SSE字段")
    payload = "\n".join(data)
    if payload == "[DONE]" and event == "message":
        return {"event": "done", "data": payload, "elapsed_seconds": elapsed}
    parsed = json.loads(payload)
    if event == "message" and isinstance(parsed, dict) and isinstance(parsed.get("delta"), str):
        event = "delta"
    if event not in {"delta", "tool_status", "answer_metadata", "error"} or not isinstance(parsed, dict):
        raise ValueError("未知SSE事件")
    return {"event": event, "data": parsed, "elapsed_seconds": elapsed}


async def sample_exchange(runtime, url: str, case: dict) -> dict:
    """采集真实HTTP SSE和唯一会话落库证据。参数：依赖/地址/案例；返回：证据；异常：已转为失败类型记录。"""
    session_id = f"ch03-{uuid4().hex}"
    observed = {"session_id": session_id, "transport": "localhost TCP /api/chat", "endpoint": url + "/api/chat",
                "events": [], "messages": [], "failures": []}
    started = time.monotonic()
    lines = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10), trust_env=False) as client:
            async with client.stream("POST", url + "/api/chat", json={"session_id": session_id, "user_id": "ch03-evaluation", "message": case["query"]}) as response:
                observed.update(http_status=response.status_code, content_type=response.headers.get("content-type", ""))
                async for line in response.aiter_lines():
                    if line:
                        lines.append(line)
                    elif lines:
                        observed["events"].append(parse_frame(lines, time.monotonic() - started))
                        lines = []
                if lines:
                    observed["failures"].append("SSE末帧没有空行终止")
    except Exception as error:
        # 不保存网关异常原文，避免认证信息进入报告；已收到的帧仍保留。
        observed["failures"].append(f"HTTP/SSE采样异常: {type(error).__name__}")
    try:
        async with runtime.factory() as session:
            conversation = await session.scalar(select(Conversation).where(Conversation.session_id == session_id))
            if conversation is not None:
                observed["conversation_id"] = conversation.id
                rows = (await session.scalars(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id))).all()
                observed["messages"] = [{name: getattr(row, name) for name in
                    ("id", "turn_id", "role", "content", "tool_calls", "tool_call_id", "turn_status")} for row in rows]
    except Exception as error:
        observed["failures"].append(f"持久化读取异常: {type(error).__name__}")
    return inspect_exchange(case, observed)
