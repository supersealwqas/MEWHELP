"""CH05两个真实验收入口共用的隔离运行时、证据采集和精确清理。

不初始化、不同步、不清共享知识；模型与RAG保持真实，只有A10在工具外部边界注入超时。
"""

import asyncio
from contextlib import aclosing, asynccontextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid4, uuid5

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "tests/fixtures/ch05/agent_cases.json"
LOCK_NAME = "mewhelp:ch04:live:exclusive"
ROOT_NODES = {"resolve_reference", "classify_intent", "retrieve_knowledge", "confidence_check", "fetch_order", "retrieve_policy",
              "main_agent", "complaint_reply", "chitchat_reply", "fallback_reply", "log_turn"}


def write_json(path: Path, value) -> None:
    """原子保存安全证据。参数：path文件、value JSON内容；返回：None；异常：文件/JSON错误。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_agent_cases(path: Path) -> tuple[list[dict], str]:
    """读取冻结10例11轮。参数：path样例文件；返回：样例与摘要；异常：ValueError，规模或形状错误。"""
    raw = path.read_bytes()
    cases = json.loads(raw)
    if (len(cases) != 10 or len({row["id"] for row in cases}) != 10
            or sum(len(row["turns"]) for row in cases) != 11
            or any(row.get("mandatory") is not True for row in cases)):
        raise ValueError("Agent标注必须为10例11轮，且全部强制")
    for case in cases:
        for turn in case["turns"]:
            if not {"query", "intent", "nodes", "tools", "actions", "required", "forbidden"} <= turn.keys():
                raise ValueError("Agent轮次标注不完整")
    return cases, hashlib.sha256(raw).hexdigest()


def inspect_turn(expected: dict, row: dict) -> dict:
    """按冻结标注评判原始证据，保留语义规则失败供人工复核。

    参数：expected冻结单轮、row真实观察；返回：附去重失败及正文的观察；异常：结构错误。
    """
    failures = list(row.get("failures", []))
    state, events = row.get("state", {}), row.get("events", [])
    answer = "".join(event.get("data", {}).get("delta", "") for event in events if event["event"] == "delta")
    checks = {"intent": state.get("intent") == expected["intent"],
        "raw_query": state.get("raw_query") == expected["query"],
        "node_order": row.get("nodes") == ["resolve_reference", "classify_intent", *expected["nodes"], "log_turn"],
        "tool_sequence": row.get("tools") == expected["tools"],
        "completed": [e["event"] for e in events][-2:] == ["answer_metadata", "done"]
            and state.get("completion_metadata", {}).get("status") == "completed",
        "unconfirmed_ticket": not row.get("tickets"),
        "unexpected_pool": len(row.get("pool_rows", [])) == expected.get("pool_rows", 0)}
    actions = state.get("suggested_actions", [])
    kinds = {action["kind"] for action in actions}
    required, allowed = set(expected["actions"]), set(expected["actions"] + expected.get("optional_actions", []))
    checks["suggested_actions"] = required <= kinds <= allowed and len(kinds) == len(actions)
    checks["action_snapshot"] = all(action.get("question") == expected["query"] and
        (action["kind"] != "create_ticket" or action.get("ticket_type") == expected.get("ticket_type")) for action in actions)
    for key in ("stop_reason", "gate"):
        if key in expected:
            actual = (state.get("gate") or {}).get("passed") if key == "gate" else state.get(key)
            checks[key] = actual == expected[key]
    if "fixed_answer" in expected:
        checks["fixed_answer"] = answer == expected["fixed_answer"]
    if "model_calls" in expected:
        checks["model_calls"] = len(row.get("models", [])) == expected["model_calls"]
    if expected.get("retrieval"):
        checks["retrieval_method"] = row.get("retrievals") == [expected["retrieval"]]
    if expected.get("tool_error"):
        checks["tool_error"] = [result.get("error", {}).get("code") for result in row.get("tool_results", [])
                                 if result.get("error")] == [expected["tool_error"]]
    if expected.get("decision_kind"):
        checks["decision_kind"] = any(item.get("kind") == expected["decision_kind"] for item in row.get("decisions", []))
    if expected.get("dependent_tools"):
        batches = [item["tool_calls"] for item in row.get("decisions", []) if item.get("kind") == "tools"]
        checks["dependent_decisions"] = len(batches) == 2 and all(len(batch) == 1 for batch in batches)
    if expected.get("citations"):
        numbers = {str(item["n"]) for item in state.get("evidence", [])}
        cited = re.findall(r"\[(\d+)\]", answer)
        checks["citations"] = bool(cited) and set(cited) <= numbers
    failures.extend(name for name, passed in checks.items() if not passed)
    failures.extend("missing_fact:" + pattern for pattern in expected["required"] if not re.search(pattern, answer))
    failures.extend("forbidden_claim:" + pattern for pattern in expected["forbidden"] if re.search(pattern, answer))
    return {**row, "answer": answer, "checks": checks, "failures": list(dict.fromkeys(failures))}


def validate_cleanup(entry: dict, snapshot: dict) -> None:
    """清理前拒绝任何外来身份或子记录。参数：登记清单和数据库快照；返回：None；异常：ValueError，归属冲突。"""
    identity = snapshot.get("conversation") or {}
    if any(identity.get(key) != entry[key] for key in ("id", "session_id", "user_id")):
        raise ValueError("清理会话三身份不匹配")
    if not {item["turn_id"] for item in snapshot["messages"]} <= set(entry["turn_ids"]):
        raise ValueError("存在未登记轮次，拒绝清理")
    if any(row["conversation_id"] != entry["id"] or row["id"] not in entry["pool_ids"] for row in snapshot["pool_rows"]):
        raise ValueError("存在未登记问题池，拒绝清理")
    if any(row["conversation_id"] != entry["id"] or row["ticket_no"] not in entry["ticket_numbers"] for row in snapshot["tickets"]):
        raise ValueError("存在未登记工单，拒绝清理")
    if any(row["conversation_id"] != entry["id"] or row["refund_no"] not in entry.get("refund_numbers", [])
           or row["source_turn_id"] not in entry["turn_ids"] or row["user_id"] != entry["user_id"]
           for row in snapshot.get("refunds", [])):
        raise ValueError("存在未登记或身份冲突的退款申请，拒绝清理")


def collect_tool_audit(messages: list) -> dict:
    """区分执行申请和非法决策审计。参数：本轮实际消息；返回：工具/非法块/失败；异常：无。"""
    tools, invalid = [], []
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call.get("type") == "invalid_tool_decision" or not {"name", "args"} <= call.keys():
                invalid.append(call)
            else:
                tools.append({"name": call["name"], "args": call["args"]})
    return {"tools": tools, "invalid_tool_decisions": invalid, "failures": ["invalid_tool_decision"] if invalid else []}


def build_config(args, *, resume: bool = False):
    """只显式加载用户env并将同一配置传给全部依赖。参数：CLI参数；返回：Settings；异常：非法资源/路径。"""
    from tests.verify_ch04_live import build_config as prior_config, parser
    if not args.test or args.env_file is None:
        raise ValueError("真实运行要求--test及显式--env-file")
    options = ["--test", "--env-file", str(args.env_file), "--report-dir", str(args.report_dir),
        "--knowledge-model-path", str((args.env_file.resolve().parent / "models/bge-m3").resolve()),
        "--rerank-model-path", str((args.env_file.resolve().parent / "models/bge-reranker-large").resolve())]
    for name in ("token_budget", "request_token_budget"):
        value = getattr(args, name, None)
        if value is not None:
            options.extend(["--" + name.replace("_", "-"), str(value)])
    config = prior_config(parser().parse_args(options))
    checkpoint = args.report_dir.resolve() / "checkpoints.sqlite3"
    if checkpoint.exists() != resume:
        raise ValueError("本次SQLite必须为新文件")
    for name in ("knowledge_model_path", "rerank_model_path"):
        path = Path(getattr(config, name))
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("两个权重路径必须为已存在的绝对目录")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise ValueError("请使用当前工作树独立uv环境")
    return config.model_copy(update={"workflow_checkpoint_path": str(checkpoint)})


async def guard_database(factory, owner: int, entries: list[dict]) -> None:
    """核实实际库、父锁和外来running。参数：会话工厂/锁拥有者/本次清单；返回：None；异常：ValueError。"""
    from sqlalchemy import select, text
    from app.models.customer_service import Message
    async with factory() as session:
        actual = await session.scalar(text("SELECT DATABASE()"))
        lock = await session.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": LOCK_NAME})
        running = set((await session.scalars(select(Message.conversation_id).where(Message.turn_status == "running"))).all())
        if actual != "mewhelp_ch04_test" or lock != owner or not running <= {row["id"] for row in entries}:
            raise ValueError("数据库/父锁/外来running检查失败")


async def snapshot(factory, entry: dict | None = None) -> dict:
    """读取五表计数或精确会话数据。参数：会话工厂/可选登记；返回：安全JSON；异常：数据库错误。"""
    from sqlalchemy import func, select
    from app.models.customer_service import Conversation, Message, Ticket, RefundApplication
    from app.models.rag_quality import LowConfidenceQuestion
    async with factory() as session:
        if entry is None:
            return {model.__tablename__: await session.scalar(select(func.count()).select_from(model))
                    for model in (Conversation, Message, Ticket, RefundApplication, LowConfidenceQuestion)}
        conversation = await session.get(Conversation, entry["id"])
        result = {"conversation": {key: getattr(conversation, key) for key in ("id", "session_id", "user_id", "status")} if conversation else None}
        for key, model, fields in [("messages", Message, ("id", "turn_id", "role", "content", "tool_calls", "tool_call_id", "turn_status", "rag_metadata")),
            ("pool_rows", LowConfidenceQuestion, ("id", "conversation_id", "raw_question", "source", "reason")),
            ("tickets", Ticket, ("ticket_no", "conversation_id", "description", "ticket_type", "status")),
            ("refunds", RefundApplication, ("refund_no", "action_id", "conversation_id", "source_turn_id", "user_id", "order_id", "reason_code", "original_question", "status"))]:
            rows = (await session.scalars(select(model).where(model.conversation_id == entry["id"]))).all()
            result[key] = [{field: getattr(row, field) for field in fields} for row in rows]
        result["messages"].sort(key=lambda row: row["id"])
        return result


async def verify_corpus_identity(runtime) -> dict:
    """只读复用既有来源/向量校验，不导入或同步。参数：已预热真实运行时；返回：语料摘要/数量；异常：来源冲突。"""
    from sqlalchemy import select
    from app.evaluation.policy import corpus_digest
    from app.evaluation.runtime import _verify_vectors, local_corpus
    from app.knowledge.store import _snapshot_from_row
    from app.models.knowledge import KnowledgeChunk
    from tests.verify_ch04_live import MANIFEST, verify_corpus
    async with runtime.factory() as session:
        actual = [_snapshot_from_row(row) for row in (await session.scalars(select(KnowledgeChunk))).all()]
    expected = local_corpus(MANIFEST, count_tokens=runtime.search.encoder.count_tokens)
    if not actual:
        raise ValueError("测试知识未准备，拒绝自动导入")
    verify_corpus(actual, expected)
    await _verify_vectors(runtime.search.index, actual)
    return {"chunk_count": len(actual), "corpus_digest": corpus_digest(actual), "vectors_match": True,
            "operation": "read_only_reuse", "initialized": False, "synchronized": False}


async def register_session(factory, entries: list, report_dir: Path, *, prefix="ch05-live-") -> dict:
    """创建前核对随机会话不存在并先登记。参数：工厂/清单/报告；返回：本次条目；异常：冲突/数据库错误。"""
    from sqlalchemy import select
    from app.models.customer_service import Conversation
    if prefix not in {"ch05-live-", "web-"}:
        raise ValueError("验收会话前缀非法")
    entry = {"session_id": prefix + uuid4().hex, "user_id": "demo-user", "turn_ids": [], "pool_ids": [], "ticket_numbers": [], "refund_numbers": []}
    async with factory.begin() as session:
        if await session.scalar(select(Conversation.id).where(Conversation.session_id == entry["session_id"])) is not None:
            raise ValueError("随机会话已存在")
        conversation = Conversation(session_id=entry["session_id"], user_id=entry["user_id"])
        session.add(conversation)
        await session.flush()
        entry["id"] = conversation.id
        entries.append(entry)
        write_json(report_dir / "manifest.json", entries)
    return entry


async def register_children(factory, entry: dict, entries: list, report_dir: Path) -> dict:
    """在唯一会话且轮次已登记时补记实际问题池/票据主键。参数：资源/登记/报告；返回：快照；异常：未知轮次拒绝。"""
    data = await snapshot(factory, entry)
    if not {item["turn_id"] for item in data["messages"]} <= set(entry["turn_ids"]):
        raise ValueError("本次会话出现未登记轮次")
    entry["pool_ids"] = [row["id"] for row in data["pool_rows"]]
    entry["ticket_numbers"] = [row["ticket_no"] for row in data["tickets"]]
    entry["refund_numbers"] = [row["refund_no"] for row in data["refunds"]]
    validate_cleanup(entry, data)
    write_json(report_dir / "manifest.json", entries)
    return data


async def cleanup(factory, owner: int, entries: list, report_dir: Path, before: dict) -> dict:
    """先验证清单，再在事务内按五表精确条件清理。参数：本次资源/清单/报告/基线；返回：前后证据；异常：冲突停止。"""
    from sqlalchemy import delete, select
    from app.models.customer_service import Conversation, Message, Ticket, RefundApplication
    from app.models.rag_quality import LowConfidenceQuestion
    await guard_database(factory, owner, entries)
    last = await snapshot(factory)
    snapshots = []
    for entry in entries:
        snapshots.append(await register_children(factory, entry, entries, report_dir))
    write_json(report_dir / "before-cleanup.json", {"counts": last, "snapshots": snapshots})
    async with factory.begin() as session:
        for entry, observed in zip(entries, snapshots):
            identity = await session.scalar(select(Conversation).where(Conversation.id == entry["id"],
                Conversation.session_id == entry["session_id"], Conversation.user_id == entry["user_id"]).with_for_update())
            if identity is None:
                raise ValueError("清理事务会话身份变化")
            validate_cleanup(entry, observed)
            await session.execute(delete(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == entry["id"], LowConfidenceQuestion.id.in_(entry["pool_ids"])))
            await session.execute(delete(Ticket).where(Ticket.conversation_id == entry["id"], Ticket.ticket_no.in_(entry["ticket_numbers"])))
            await session.execute(delete(RefundApplication).where(RefundApplication.conversation_id == entry["id"], RefundApplication.refund_no.in_(entry["refund_numbers"])))
            await session.execute(delete(Message).where(Message.conversation_id == entry["id"], Message.turn_id.in_(entry["turn_ids"])))
            # 未登记的子记录不会被扩大删除；外键会阻止误删会话。
            await session.execute(delete(Conversation).where(Conversation.id == entry["id"], Conversation.session_id == entry["session_id"], Conversation.user_id == entry["user_id"]))
    after = await snapshot(factory)
    result = {"before": before, "before_cleanup": last, "after": after, "restored": before == after}
    write_json(report_dir / "cleanup.json", result)
    if before != after:
        raise ValueError("五表总数未恢复，保留证据")
    return result


class RecordingChat:
    """原样转发真实模型，只记录输入、分片、计量和关闭时间。"""
    def __init__(self, model, records: list, stage: str, tool_schemas=(), *, turn_id=None):
        """参数：真实模型/记录列表/阶段；返回：None；异常：无。"""
        self.model, self.records, self.stage = model, records, stage
        self.tool_schemas = list(tool_schemas)
        self.turn_id = turn_id

    def bind_tools(self, tools, **kwargs):
        """原样绑定只读定义。参数：工具定义及选项；返回：透明包装；异常：模型配置错误。"""
        names = {tool.get("function", tool).get("name") for tool in tools if isinstance(tool, dict)}
        stages = {"submit_understanding":"understand", "submit_intent":"classify", "submit_expanded_queries":"expand", "submit_decision":"decide"}
        stage = next((stage for name, stage in stages.items() if name in names),
                     "decide" if names & {"query_order", "query_logistics", "query_product"} else self.stage)
        return RecordingChat(self.model.bind_tools(tools, **kwargs), self.records, stage, tools, turn_id=self.turn_id)

    async def astream(self, messages):
        """采集真实流，取消时关闭同一上游。参数：消息；返回：原分片；异常：网络/取消透传。"""
        row = {"stage": self.stage, "turn_id": self.turn_id, "started": time.monotonic(), "closed": False,
               "inputs": [message.model_dump(mode="json") for message in messages], "tool_schemas": self.tool_schemas,
               "chunks": [], "output": None}
        self.records.append(row)
        gathered = None
        try:
            async with aclosing(self.model.astream(messages)) as stream:
                async for chunk in stream:
                    gathered = chunk if gathered is None else gathered + chunk
                    row["chunks"].append({"clock": time.monotonic(), **chunk.model_dump(mode="json")})
                    yield chunk
        except BaseException as error:
            row["error_type"] = type(error).__name__
            raise
        finally:
            row.update(ended=time.monotonic(), closed=True,
                output=gathered.model_dump(mode="json") if gathered is not None else None)


def make_trace(entries: list, report_dir: Path):
    """创建公开回调，记录实际节点和入口轮次。参数：登记/报告；返回：回调；异常：未知入口身份拒绝。"""
    from langchain_core.callbacks import BaseCallbackHandler

    class Trace(BaseCallbackHandler):
        """只记录语义节点，不改变图输入或输出。"""
        run_inline = True
        raise_error = True

        def __init__(self):
            """参数：无；返回：None；异常：无。"""
            self.events, self.active = [], {}
            self.current_turn = None

        def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
            """记录真实进入并登记服务生成的轮ID。参数：框架回调；返回：None；异常：未知会话。"""
            name = kwargs.get("name")
            if name not in ROOT_NODES | {"decision", "tools", "answer"} or not isinstance(inputs, dict):
                return
            context = inputs.get("context", {})
            if context.get("turn_id"):
                self.current_turn = context.get("turn_id")
                entry = next((row for row in entries if row["id"] == context.get("conversation_id") and row["session_id"] == context.get("session_id")), None)
                if entry is None:
                    raise ValueError("图入口不是本次登记会话")
                if context["turn_id"] not in entry["turn_ids"]:
                    entry["turn_ids"].append(context["turn_id"])
                    write_json(report_dir / "manifest.json", entries)
            identity = {"node": name, "turn_id": context.get("turn_id"), "run_id": str(run_id)}
            self.active[run_id] = identity
            self.events.append({**identity, "phase": "enter", "clock": time.monotonic()})

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            """记录退出时间。参数：框架输出；返回：None；异常：无。"""
            identity = self.active.pop(run_id, None)
            if identity:
                self.events.append({**identity, "phase": "exit", "clock": time.monotonic()})

        def on_chain_error(self, error, *, run_id, **kwargs):
            """仅保存错误类型。参数：框架错误；返回：None；异常：无。"""
            identity = self.active.pop(run_id, None)
            if identity:
                self.events.append({**identity, "phase": "error", "clock": time.monotonic(), "error_type": type(error).__name__})
    return Trace()


@asynccontextmanager
async def live_runtime(config, factory, entries: list, report_dir: Path):
    """装配真实CH05所有依赖并统一关闭。参数：显式配置/工厂/清单/报告；返回：运行时；异常：启动错误。"""
    import httpx
    from app.core import llm
    from app.knowledge.query_rewrite import QueryRewriter, QueryExpander
    from app.knowledge.runtime import knowledge_storage, online_search_runtime
    from app.repositories.chat import ChatRepository
    from app.repositories.support import SupportRepository
    from app.services.chat import ChatService
    from app.tools.executor import ToolExecutor
    from app.tools.registry import build_agent_registry
    from app.workflow.graph import WorkflowDependencies
    from app.workflow.model import WorkflowModelAdapter
    from app.workflow.runtime import workflow_runtime
    records, retrievals, trace = [], [], make_trace(entries, report_dir)
    original = llm.settings
    llm.settings = config
    sync_http, async_http = httpx.Client(), httpx.AsyncClient()

    def model_factory(*, evaluation_stage=None, **options):
        """显式注入同一模型配置和已管理HTTP客户端。参数：阶段/模型选项；返回：透明真实模型；异常：配置错误。"""
        stage = evaluation_stage or "answer"
        return RecordingChat(llm.get_chat_model(http_client=sync_http, http_async_client=async_http, **options), records, stage,
                             turn_id=trace.current_turn)

    def rewrite_factory(**options):
        """为真实改写记录独立阶段。参数：模型选项；返回：真实模型；异常：配置错误。"""
        return model_factory(evaluation_stage="rewrite", **options)

    try:
        async with knowledge_storage(config) as store, online_search_runtime(config, store) as search:
            search.query_rewriter = QueryRewriter(model_factory=rewrite_factory, request_budget=config.request_token_budget)
            # 仅观察调用入口，普通检索与退款检索原实现、返回值均保留。
            ordinary, policy = search.retrieve, search.retrieve_policy

            async def retrieve(*args, **kwargs):
                """记录普通知识入口。参数：原参数；返回：真实排名；异常：检索错误。"""
                retrievals.append("retrieve")
                return await ordinary(*args, **kwargs)

            async def retrieve_policy(*args, **kwargs):
                """记录退款入口。参数：原参数；返回：真实排名；异常：检索错误。"""
                retrievals.append("retrieve_policy")
                return await policy(*args, **kwargs)

            search.retrieve, search.retrieve_policy = retrieve, retrieve_policy
            repository = ChatRepository(factory)
            executor = ToolExecutor(build_agent_registry(), repository, SupportRepository(factory, search),
                config.tool_timeout_seconds, config.tool_max_retries, config.tool_retry_delay_seconds,
                knowledge_timeout_seconds=config.knowledge_timeout_seconds)
            model = WorkflowModelAdapter(model_factory=model_factory, stream_usage=config.workflow_stream_usage)
            deps = WorkflowDependencies(model, search, executor, repository, config,
                QueryExpander(model_factory=model_factory, request_budget=config.request_token_budget))
            async with workflow_runtime(deps, Path(config.workflow_checkpoint_path)) as graph:
                graph = graph.with_config(callbacks=[trace])
                yield SimpleNamespace(service=ChatService(repository, graph, executor, config), factory=factory,
                    trace=trace, models=records, retrievals=retrievals, search=search)
    finally:
        llm.settings = original
        await async_http.aclose()
        sync_http.close()


@asynccontextmanager
async def fault_boundary(runtime, enabled: bool):
    """A10仅在外部工具coroutine注入超时，保留执行器两次尝试。参数：运行时/开关；返回：计数；异常：原流程错误。"""
    tool = runtime.service.executor.registry.get("query_order")
    original, attempts = tool.coroutine, []

    async def timeout(*args, **kwargs):
        """模拟已约定外部超时。参数：真实工具参数；返回：不返回；异常：TimeoutError。"""
        attempts.append(time.monotonic())
        raise TimeoutError("本次验收工具边界超时")

    if enabled:
        tool.coroutine = timeout
    try:
        yield attempts
    finally:
        tool.coroutine = original


async def sample_turn(runtime, url: str, expected: dict, entry: dict, entries: list, report_dir: Path, *, inject=False, stop=False,
                      endpoint="/api/chat", payload=None, inspect=True) -> dict:
    """真实TCP收流并保存对应数据库/SQLite/模型证据。参数：资源/标注/会话/报告及故障或停止开关；返回：观察；异常：持久化核验错误。"""
    import httpx
    # 保留旧轮对象及其后续关闭证据；绝不通过下一轮clear隐藏仍存活的旧模型。
    preexisting = [row for row in runtime.models if not row["closed"]]
    runtime.retrievals.clear()
    row = {"query": expected["query"], "session_id": entry["session_id"], "events": [], "failures": [], "transport": "localhost TCP"}
    if preexisting or runtime.trace.active:
        row["failures"].append("prior_turn_still_active")
    started, lines = time.monotonic(), []
    async with fault_boundary(runtime, inject) as attempts:
        try:
            async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
                async with client.stream("POST", url + endpoint, json=payload or {"session_id": entry["session_id"], "user_id": entry["user_id"], "message": expected["query"]}) as response:
                    row["http_status"] = response.status_code
                    async for line in response.aiter_lines():
                        if line:
                            lines.append(line)
                        elif lines:
                            event = parse_workflow_frame(lines, time.monotonic() - started)
                            event["received_at"] = time.monotonic()
                            row["events"].append(event)
                            lines = []
                            if stop and event["event"] == "delta":
                                break
        except Exception as error:
            row["failures"].append("http_error:" + type(error).__name__)
        row["fault_attempts"] = len(attempts)
    # 给真实断开清理一个有界窗口，不能通过直接修改状态来制造通过。
    async with asyncio.timeout(20):
        while entry["session_id"] in runtime.service.guard._active:
            await asyncio.sleep(.05)
    data = await register_children(runtime.factory, entry, entries, report_dir)
    from app.workflow.state import checkpoint_thread_id
    state = (await runtime.service.graph.aget_state({"configurable": {"thread_id": checkpoint_thread_id(entry["id"])}})).values
    turn_id = entry["turn_ids"][-1] if entry["turn_ids"] else None
    messages = [item for item in data["messages"] if item["turn_id"] == turn_id]
    models = json.loads(json.dumps([model for model in runtime.models if model["turn_id"] == turn_id], ensure_ascii=False))
    trace = [event for event in runtime.trace.events if event["turn_id"] == turn_id]
    # 先保存原始证据，后续判定器即使出错也不会丢掉模型输出和真实审计。
    write_json(report_dir / f"raw-turn-{turn_id}-{uuid4().hex[:8]}.json", {**row, "state": state, "models": models,
        "trace": trace, "database": data})
    audit = collect_tool_audit(messages)
    row["failures"].extend(audit["failures"])
    decisions = []
    for model in models:
        output = model.get("output") or {}
        if model["stage"] == "decide":
            calls = output.get("tool_calls") or []
            if calls:
                decisions.append({"kind": "tools", "tool_calls": calls})
            else:
                try:
                    decisions.append(json.loads(output.get("content", "")))
                except (ValueError, TypeError):
                    decisions.append({"kind": "invalid"})
    row.update(state=state, trace=trace,
        nodes=[event["node"] for event in trace if event["phase"] == "enter" and event["node"] in ROOT_NODES],
        models=models, decisions=decisions, retrievals=list(runtime.retrievals), messages=messages,
        tools=audit["tools"], invalid_tool_decisions=audit["invalid_tool_decisions"],
        tool_results=[json.loads(message["content"]) for message in messages if message["role"] == "tool"],
        pool_rows=data["pool_rows"], tickets=data["tickets"], refunds=data["refunds"], elapsed_seconds=time.monotonic() - started,
        active_model_streams=sum(not model["closed"] for model in models), active_graph_nodes=len(runtime.trace.active))
    if inject and row["fault_attempts"] != runtime.service.settings.tool_max_retries + 1:
        row["failures"].append("fault_retry_count")
    if stop:
        row["checks"] = {"no_done": not any(event["event"] == "done" for event in row["events"]),
            "cancelled": bool(messages) and all(item["turn_status"] == "cancelled" for item in messages),
            "streams_closed": row["active_model_streams"] == 0, "graph_closed": row["active_graph_nodes"] == 0,
            "guard_released": entry["session_id"] not in runtime.service.guard._active}
        row["failures"].extend(name for name, passed in row["checks"].items() if not passed)
        return row
    return inspect_turn(expected, row) if inspect else row


def parse_workflow_frame(lines: list[str], elapsed: float) -> dict:
    """补充CH06安全阶段和动作帧，旧事件仍复用解析器。参数：帧行/耗时；返回：事件；异常：非法JSON。"""
    from tests.ch03_live_runtime import parse_frame
    name = next((line[6:].strip() for line in lines if line.startswith("event:")), "message")
    if name in {"workflow_stage", "select_order", "action_status"}:
        return {"event":name, "data":json.loads("\n".join(line[5:].removeprefix(" ") for line in lines if line.startswith("data:"))), "elapsed_seconds":elapsed}
    return parse_frame(lines, elapsed)


def append_result(report_dir: Path, summary: dict, row: dict) -> None:
    """逐例立即落盘，失败不覆盖。参数：报告/摘要/观察；返回：None；异常：文件错误。"""
    with (report_dir / "cases.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    summary["results"].append({"id": row["id"], "session_id": row.get("session_id"), "failures": row["failures"], "passed": not row["failures"]})
    write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary["results"][-1], ensure_ascii=False), flush=True)


async def run_suite(args, config, summary: dict, cases: list) -> int:
    """持资源锁运行并在任何失败后精确清理。参数：参数/配置/摘要/冻结集；返回：0通过或1未通过；异常：资源错误。"""
    from sqlalchemy import text
    from app.db.session import create_database
    from tests.ch03_live_runtime import tcp_server
    from tests.verify_ch04_live import resource_lock
    engine, factory = create_database(config)
    entries = []
    try:
        async with resource_lock(config, prepare=False) as locked:
            owner = await locked.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name": LOCK_NAME})
            await guard_database(factory, owner, entries)
            before = await snapshot(factory)
            summary.update(stage="runtime_start", lock_owner=owner, counts_before=before)
            write_json(args.report_dir / "summary.json", summary)
            try:
                async with live_runtime(config, factory, entries, args.report_dir) as runtime, tcp_server(runtime) as url:
                    summary.update(stage="cases", url=url, resources=await verify_corpus_identity(runtime))
                    write_json(args.report_dir / "summary.json", summary)
                    if summary["suite"] == "agent":
                        for case in cases:
                            await guard_database(factory, owner, entries)
                            entry = await register_session(factory, entries, args.report_dir)
                            for number, expected in enumerate(case["turns"], 1):
                                row = await sample_turn(runtime, url, expected, entry, entries, args.report_dir,
                                    inject=bool(case.get("external_fault_injection")))
                                row.update(id=f"{case['id']}.{number}", expected=expected)
                                append_result(args.report_dir, summary, row)
                    else:
                        from tests.verify_ch05_live import acceptance
                        await acceptance(runtime, url, cases, entries, args, summary)
                summary["runtime_closed"] = True
                if summary["suite"] == "verify" and summary.get("restart_request"):
                    from tests.verify_ch05_live import restart_process
                    await restart_process(args, config, factory, owner, entries, summary)
            finally:
                summary["cleanup"] = await cleanup(factory, owner, entries, args.report_dir, before)
        summary["lock_released"] = True
    finally:
        await engine.dispose()
        summary["database_engine_closed"] = True
    summary.update(stage="completed", status="completed", ended_at=datetime.now(timezone.utc).isoformat())
    summary["passed"] = all(row["passed"] for row in summary["results"])
    write_json(args.report_dir / "summary.json", summary)
    return 0 if summary["passed"] else 1


def run_cli(args, *, suite: str) -> int:
    """两个入口共用的先保护后装配流程。参数：参数/套件名；返回：0通过/1断言失败/2运行失败；异常：参数或报告目录已存在。"""
    if not args.test:
        raise ValueError("必须显式--test")
    cases, fingerprint = load_agent_cases(args.cases)
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    summary = {"suite": suite, "status": "running", "stage": "configuration", "results": [], "case_sha256": fingerprint,
        "started_at": datetime.now(timezone.utc).isoformat(), "real_model": True, "real_rag": True,
        "business_data_origin": "订单、物流、商品为项目演示查询；MySQL/SQLite/Milvus和模型实际运行",
        "unknown_usage": "上游不返回usage时记录null，保守预算单位不是实际账单tokens",
        "versions": {name: version(name) for name in ("langgraph", "langgraph-checkpoint-sqlite", "langchain-openai", "SQLAlchemy")},
        "python": sys.version, "python_executable": sys.executable}
    source_paths = [ROOT / "app/workflow/prompts.py", ROOT / "app/knowledge/query_rewrite.py", ROOT / "uv.lock"]
    summary["source_sha256"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    summary["implementation_sha256"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*sorted((ROOT / "app").rglob("*.py")), Path(__file__), ROOT / "tests/verify_ch05_live.py", ROOT / "tests/eval_ch05_prompts.py"]}
    for path in source_paths[:2]:
        (args.report_dir / path.name).write_bytes(path.read_bytes())
    write_json(args.report_dir / "frozen-cases.json", cases)
    write_json(args.report_dir / "summary.json", summary)
    try:
        config = build_config(args)
        summary["config"] = {name: getattr(config, name) for name in ("mysql_database", "milvus_database", "milvus_hybrid_collection",
            "knowledge_model_path", "rerank_model_path", "rerank_device", "chat_model", "knowledge_retrieval_mode", "workflow_checkpoint_path",
            "rerank_min_score", "token_budget", "request_token_budget", "workflow_max_model_calls", "workflow_total_token_budget", "workflow_max_decisions", "workflow_stream_usage")}
        code = asyncio.run(run_suite(args, config, summary, cases))
    except BaseException as error:
        summary.update(status="error", error_type=type(error).__name__, ended_at=datetime.now(timezone.utc).isoformat())
        # 外部异常原文可能含凭据；只保存调用位置与类型。
        trace = error.__traceback__
        locations = []
        while trace:
            locations.append({"file": Path(trace.tb_frame.f_code.co_filename).name, "line": trace.tb_lineno})
            trace = trace.tb_next
        summary["error_locations"] = locations
        code = 2
    summary["exit_code"] = code
    write_json(args.report_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "exit_code": code, "report_dir": str(args.report_dir)}, ensure_ascii=False), flush=True)
    return code
