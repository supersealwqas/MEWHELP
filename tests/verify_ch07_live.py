"""CH07真实长会话验收：固定题集、默认/演示窗口、浏览器或HTTP收流、六表精确清理。"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import time

import httpx

from tests.ch05_live_runtime import (ROOT, LOCK_NAME, RecordingChat, build_config, cleanup, guard_database,
    live_runtime, register_children, register_session, snapshot, validate_cleanup, verify_corpus_identity)


def write_json(path, value):
    """保存包含LangChain完整消息的JSON证据。参数：路径/结构；返回：None；异常：序列化或IO错误。"""
    def encode(item):
        """保留模型或日期原字段。参数：对象；返回：JSON值；异常：未知类型。"""
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, datetime):
            return item.isoformat()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")


def load_cases() -> dict:
    """读取同一冻结题集，不为配置筛题。参数：无；返回：标注；异常：文件/格式错误。"""
    data = json.loads((ROOT / "tests/fixtures/ch07/dialogue_cases.json").read_text(encoding="utf-8"))
    if len(data["queries"]) < 21 or not all(isinstance(query, str) and query.strip() for query in data["queries"]):
        raise ValueError("CH07必须使用不少于21轮的固定题集")
    return data


def inspect_record(row: dict) -> dict:
    """对照真实流与数据库，不以页面提示代替完成。参数：单轮证据；返回：带失败项记录；异常：结构错误。"""
    failures = []
    answer = "".join(event.get("data", {}).get("delta", "") for event in row["events"] if event["event"] == "delta")
    kinds = [event["event"] for event in row["events"]]
    if row["http_status"] != 200 or "error" in kinds or kinds[-2:] != ["answer_metadata", "done"]:
        failures.append("HTTP_or_SSE_error")
    if not answer.strip():
        failures.append("empty_answer")
    users = [message for message in row["messages"] if message["role"] == "user"]
    finals = [message for message in row["messages"] if message["role"] == "assistant" and not message.get("tool_calls")]
    if len(users) != 1 or users[0]["content"] != row["query"] or len(finals) != 1 or finals[0]["content"] != answer:
        failures.append("original_or_saved_answer")
    if row["state"].get("completion_metadata", {}).get("status") != "completed":
        failures.append("checkpoint_not_completed")
    return {**row, "answer":answer, "failures":failures}


def parse_events(raw: str) -> list[dict]:
    """解析浏览器收到的完整SSE原文。参数：raw原响应；返回：有序事件；异常：非法协议。"""
    from tests.ch05_live_runtime import parse_workflow_frame
    return [parse_workflow_frame(part.splitlines(), 0) for part in raw.replace("\r\n", "\n").split("\n\n") if part.strip()]


async def capture(runtime, body, entries, report_dir, report):
    """在原请求释放后保存浏览器流、DB、checkpoint和摘要快照。参数：资源/原始流/清单/报告；返回：序号；异常：身份/状态错误。"""
    from app.repositories.context import ContextRepository
    from app.workflow.state import checkpoint_thread_id
    entry = next(row for row in entries if row["session_id"] == body["session_id"])
    async with asyncio.timeout(20):
        while entry["session_id"] in runtime.service.guard._active:
            await asyncio.sleep(.05)
    data = await register_children(runtime.factory, entry, entries, report_dir)
    state = (await runtime.service.graph.aget_state({"configurable":{"thread_id":checkpoint_thread_id(entry["id"])}})).values
    turn_id = state["context"]["turn_id"]
    messages = [row for row in data["messages"] if row["turn_id"] == turn_id]
    context = await ContextRepository(runtime.factory).load_context(entry["id"], include_turns=False)
    row = inspect_record({"query":body["message"], "session_id":entry["session_id"], "turn_id":turn_id,
        "http_status":body["http_status"], "events":parse_events(body["response"]), "messages":messages,
        "state":state, "context":context, "received_at":time.monotonic()})
    write_json(report_dir / f"turn-{turn_id}.json", row)
    report["turns"].append({key:row[key] for key in ("query","session_id","turn_id","answer","failures","received_at")})
    write_json(report_dir / "summary.json", report)
    print(json.dumps({"turn":len(report["turns"]), "failures":row["failures"],
        "summary_upto":context["summary_upto_msg_id"], "layer1_from":context["layer1_from_msg_id"]}), flush=True)
    return {"captured":len(report["turns"])}


def configure_browser(application, runtime, entries, report_dir, report):
    """仅为独占本机验收应用加探针路由，生产应用无这些入口。参数：资源/报告；返回：None；异常：文件错误。"""
    from fastapi import Request
    from fastapi.responses import HTMLResponse, FileResponse
    from app.main import STATIC_DIR

    async def page():
        """返回原HTML并先加载透明网络取证。参数：无；返回：HTML；异常：IO错误。"""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        version = (ROOT / "tests/fixtures/ch07/browser-probe.js").stat().st_mtime_ns
        return HTMLResponse(html.replace('<script type="module"', f'<script src="/__live__/probe.js?v={version}"></script><script type="module"', 1))

    async def bootstrap():
        """以登记会话进入原页面。参数：无；返回：测试HTML；异常：无。"""
        return HTMLResponse('<script>sessionStorage.setItem("mewhelp.session",' + json.dumps(entries[0]["session_id"]) + ');location.replace("/");</script>')

    async def probe():
        """提供原样转发fetch的取证脚本。参数：无；返回：JS文件；异常：IO错误。"""
        return FileResponse(ROOT / "tests/fixtures/ch07/browser-probe.js", media_type="text/javascript")

    async def record(request: Request):
        """接收实际页面收到的完整流并核验DB。参数：测试请求；返回：已记录数；异常：核验错误。"""
        return await capture(runtime, await request.json(), entries, report_dir, report)

    async def state():
        """只读后台状态，尾问前可明确等待摘要结束。参数：无；返回：任务数及已收流数；异常：无。"""
        return {"active_summaries":runtime.service.summaries.active_count, "captured":len(report["turns"])}

    async def finish(request: Request):
        """登记外部浏览器核对结果，不直接宣称通过。参数：测试检查结果；返回：确认；异常：文件错误。"""
        write_json(report_dir / "browser-finish.json", await request.json())
        return {"saved":True}

    application.router.routes[:] = [route for route in application.router.routes if getattr(route,"path",None) != "/"]
    for path, handler, methods in [("/",page,["GET"]),("/__live__/browser",bootstrap,["GET"]),
        ("/__live__/probe.js",probe,["GET"]),("/__live__/record",record,["POST"]),
        ("/__live__/state",state,["GET"]),("/__live__/finish",finish,["POST"])]:
        application.add_api_route(path, handler, methods=methods)


@asynccontextmanager
async def context_runtime(config, factory, entries, args, report):
    """复用真实模型和知识运行时，接上本章真实后台任务。参数：资源；返回：运行时；异常：初始化/模型错误。"""
    from app.core.llm import get_chat_model
    from app.repositories.context import ContextRepository
    from app.services.summaries import SummaryManager
    from app.workflow.summarization import SummaryGenerator
    async with live_runtime(config, factory, entries, args.report_dir) as runtime:
        # 浏览器新会话仍走生产start_turn；观察器在第一节点登记其真实DB身份，不预造模型历史。
        original = runtime.trace.on_chain_start

        def observe(serialized, inputs, **kwargs):
            """登记本机测试浏览器实际创建的新会话，然后使用既有审计。参数：回调；返回：None；异常：未知身份拒绝。"""
            context = inputs.get("context", {}) if isinstance(inputs, dict) else {}
            if args.browser and kwargs.get("name") == "resolve_reference" and not any(row["id"] == context.get("conversation_id") for row in entries):
                if context.get("user_id") != "demo-user" or not re.fullmatch(r"web-[0-9a-f]{32}", context.get("session_id", "")):
                    raise ValueError("测试新会话身份不合法")
                entries.append({"id":context["conversation_id"],"session_id":context["session_id"],"user_id":"demo-user",
                    "turn_ids":[],"pool_ids":[],"ticket_numbers":[],"refund_numbers":[]})
                write_json(args.report_dir / "manifest.json", entries)
            return original(serialized, inputs, **kwargs)

        runtime.trace.on_chain_start = observe
        summary_models = []
        with httpx.Client() as sync_client:
            async with httpx.AsyncClient() as async_client:
                def model_factory(**options):
                    """透明记录后台模型，不混入前台轮次账本。参数：模型配置；返回：真实模型包装；异常：SDK错误。"""
                    model = get_chat_model(http_client=sync_client, http_async_client=async_client, **options)
                    return RecordingChat(model, summary_models, "summary")
                worker = SummaryManager(ContextRepository(factory), SummaryGenerator(config, model_factory=model_factory), config)
                runtime.service.summaries = worker
                if args.browser:
                    runtime.configure_test_app = lambda app: configure_browser(app, runtime, entries, args.report_dir, report)
                try:
                    yield runtime
                finally:
                    await worker.close()
                    write_json(args.report_dir / "models.json", runtime.models)
                    write_json(args.report_dir / "summary-models.json", summary_models)
                    write_json(args.report_dir / "trace.json", runtime.trace.events)


async def cleanup_summaries(factory, owner, entries, report_dir):
    """先核对注册身份及完整消息范围，再只删这些会话的摘要。参数：登记资源；返回：被清理快照；异常：归属冲突。"""
    from sqlalchemy import select, delete
    from app.models.customer_service import ConversationSummary
    await guard_database(factory, owner, entries)
    observed = []
    for entry in entries:
        raw = await register_children(factory, entry, entries, report_dir)
        validate_cleanup(entry, raw)
        ids = {row["id"] for row in raw["messages"]}
        async with factory.begin() as session:
            segments = (await session.scalars(select(ConversationSummary).where(ConversationSummary.conversation_id == entry["id"]))).all()
            if any(row.from_msg_id not in ids or row.upto_msg_id not in ids for row in segments):
                raise ValueError("摘要范围不是本次登记原文")
            saved = [{"id":row.id,"conversation_id":row.conversation_id,"seq":row.seq,"from_msg_id":row.from_msg_id,
                      "upto_msg_id":row.upto_msg_id,"content":row.content} for row in segments]
            observed.extend(saved)
            await session.execute(delete(ConversationSummary).where(ConversationSummary.conversation_id == entry["id"],
                ConversationSummary.id.in_([row["id"] for row in saved])))
    write_json(report_dir / "summaries-before-cleanup.json", observed)
    return observed


async def run(args, config, report):
    """持隔离库锁运行全套同题测试并精确清理。参数：CLI/显式配置/报告；返回：0或2；异常：依赖错误。"""
    from sqlalchemy import text
    from app.db.session import create_database
    from app.repositories.context import ContextRepository
    from app.workflow.context import shared_budget
    from tests.ch03_live_runtime import tcp_server
    from tests.verify_ch04_live import resource_lock
    cases, entries = load_cases(), []
    engine, factory = create_database(config)
    report["budget"] = asdict(shared_budget(config))
    report["model"] = config.chat_model
    try:
        async with resource_lock(config, prepare=False) as locked:
            owner = await locked.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name":LOCK_NAME})
            await guard_database(factory, owner, entries)
            before = await snapshot(factory)
            async with factory() as session:
                summary_before = await session.scalar(text("SELECT COUNT(*) FROM conversation_summaries"))
            try:
                await register_session(factory, entries, args.report_dir, prefix="web-")
                async with context_runtime(config, factory, entries, args, report) as runtime, tcp_server(runtime) as url:
                    report["resources"] = await verify_corpus_identity(runtime)
                    report["url"] = url
                    write_json(args.report_dir / "summary.json", report)
                    if args.browser:
                        write_json(args.report_dir / "browser-ready.json", {"url":url,"bootstrap_url":url+"/__live__/browser",
                            "session_id":entries[0]["session_id"],"queries":cases["queries"]})
                        print("browser ready " + url, flush=True)
                        async with asyncio.timeout(3600):
                            while not (args.report_dir / "browser-finish.json").exists():
                                await asyncio.sleep(.5)
                        report["browser"] = json.loads((args.report_dir / "browser-finish.json").read_text(encoding="utf-8"))
                    else:
                        async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
                            for query in cases["queries"]:
                                if query == cases["queries"][-1]:
                                    async with asyncio.timeout(120):
                                        while runtime.service.summaries.active_count:
                                            await asyncio.sleep(.1)
                                response = await client.post(url+"/api/chat", json={"session_id":entries[0]["session_id"],"user_id":"demo-user","message":query})
                                await capture(runtime, {"session_id":entries[0]["session_id"],"message":query,
                                    "http_status":response.status_code,"response":response.text}, entries, args.report_dir, report)
                    async with asyncio.timeout(120):
                        while runtime.service.summaries.active_count:
                            await asyncio.sleep(.1)
                    report["context_final"] = await ContextRepository(factory).load_context(entries[0]["id"], include_turns=False)
            finally:
                report["saved_segments"] = await cleanup_summaries(factory, owner, entries, args.report_dir)
                report["cleanup"] = await cleanup(factory, owner, entries, args.report_dir, before)
                async with factory() as session:
                    summary_after = await session.scalar(text("SELECT COUNT(*) FROM conversation_summaries"))
                report["summary_cleanup_restored"] = summary_before == summary_after
    finally:
        await engine.dispose()
    states = [json.loads((args.report_dir/f"turn-{row['turn_id']}.json").read_text(encoding="utf-8"))["state"] for row in report["turns"]]
    report["checks"] = final_checks(report, cases, entries[0]["session_id"], states, args.profile, args.browser)
    report["passed"] = all(report["checks"].values())
    return 0 if report["passed"] else 2


def final_checks(report, cases, session_id, states, profile, browser):
    """核对主长会话及所有额外轮。参数：报告/题集/主会话/全部状态/配置/浏览器模式；返回：检查字典；异常：记录缺字段。"""
    stress = [row for row in report["turns"] if row["session_id"] == session_id][:len(cases["queries"])]
    checks = {"same_queries": [row["query"] for row in stress] == cases["queries"],
        "all_answered": len(stress) == len(cases["queries"]) and all(not row["failures"] for row in stress),
        "all_recorded_answered": bool(report["turns"]) and all(not row["failures"] and row["answer"].strip() for row in report["turns"]),
        "cleanup":report["cleanup"]["restored"] and report["summary_cleanup_restored"]}
    if profile == "default":
        checks["no_compression"] = all(not state.get("history_context",{}).get("layer1_from_msg_id") for state in states) and not report["saved_segments"]
    else:
        last_state = states[len(cases["queries"])-1] if len(states) >= len(cases["queries"]) else {}
        checks["cascade"] = bool(report["context_final"]["summaries"]) and any(state.get("history_context",{}).get("summary_required") for state in states)
        checks["early_order_in_injected_summary"] = cases["early_order_id"] in last_state.get("history_context",{}).get("summary", "")
        checks["early_order_answer"] = bool(stress) and cases["early_order_id"] in stress[-1]["answer"] and bool(re.search("物流|配送|运输|送达|收货|收到", stress[-1]["answer"]))
    if browser:
        required = ("reply_blocks_switch", "twenty_one_frontend_turns", "new_clears_only_page",
            "three_distinct_sessions", "full_original_history", "refresh_restores_history",
            "continues_old_session", "sidebar_failure_keeps_chat", "chat_after_sidebar_failure")
        evidence = report.get("browser", {})
        observed = evidence.get("checks", {})
        # 缺失检查不能被all(已存在结果)当成通过；驱动错误和字符串等非布尔结果同样拒绝。
        checks["browser"] = (evidence.get("error") is None and isinstance(observed, dict)
            and all(observed.get(name) is True for name in required)
            and all(value is True for value in observed.values()))
    return checks


def main(argv=None):
    """显式运行真实验收并保存有限错误诊断。参数：argv；返回：0/2/3；异常：参数错误由CLI显示。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=["default","demo"])
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args(argv)
    if not args.test:
        parser.error("必须显式--test")
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    path = ROOT / "tests/fixtures/ch07/dialogue_cases.json"
    report = {"profile":args.profile,"real_model":True,"browser_mode":args.browser,"turns":[],
              "case_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"started_at":datetime.now(timezone.utc).isoformat()}
    write_json(args.report_dir / "frozen-cases.json", load_cases())
    handler = logging.FileHandler(args.report_dir / "app.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    # 生产app日志刻意不向root传播，验收必须在同一命名空间收取原日志。
    app_logger = logging.getLogger("app")
    app_logger.addHandler(handler)
    try:
        config = build_config(args)
        if args.profile == "demo":
            config = config.model_copy(update={"model_context_window":18000,"max_output_tokens":2000,
                "max_user_input_tokens":2000,"workflow_max_decisions":3,"tool_result_max_tokens":1200,"rerank_top_k":5})
        report["config"] = {name:getattr(config,name) for name in ("model_context_window","max_output_tokens","max_user_input_tokens",
            "workflow_max_decisions","tool_result_max_tokens","rerank_top_k","context_non_ascii_tokens","context_steady_turn_tokens")}
        code = asyncio.run(run(args, config, report))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error_locations=[])
        trace = error.__traceback__
        while trace:
            report["error_locations"].append({"file":Path(trace.tb_frame.f_code.co_filename).name,"line":trace.tb_lineno})
            trace = trace.tb_next
        code = 3
    finally:
        app_logger.removeHandler(handler)
        handler.close()
    report.update(exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
    write_json(args.report_dir / "summary.json", report)
    print(json.dumps({"exit_code":code,"checks":report.get("checks"),"error_type":report.get("error_type")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
