"""复用CH05真实运行时验收CH06四场景、等待重开、退款回执和精确清理。"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx

from tests.ch05_live_runtime import (ROOT, LOCK_NAME, append_result, build_config, cleanup, guard_database,
    live_runtime, register_children, register_session, sample_turn, snapshot, verify_corpus_identity, write_json)
from tests.eval_ch06_prompts import inspect_understanding, raw_result


def configure_browser_probe(application, runtime, report_dir):
    """仅给本机验收应用添加浏览器初始化和网络故障探针。参数：测试应用/资源/报告；返回：None；异常：文件错误。"""
    from fastapi import Request
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from app.main import STATIC_DIR

    async def bootstrap():
        """预置已登记随机会话再进入原页面。参数：无；返回：测试HTML；异常：未登记。"""
        session_id = runtime.browser_entry["session_id"]
        return HTMLResponse('<script>sessionStorage.setItem("mewhelp.session",' + json.dumps(session_id) + ');location.replace("/");</script>')

    async def page():
        """原HTML仅插入测试网络记录器，生产文件不变。参数：无；返回：HTML；异常：文件错误。"""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("</head>", '<script src="/__live__/probe.js" defer></script></head>'))

    async def script():
        """提供测试探针。参数：无；返回：JavaScript文件；异常：文件错误。"""
        return FileResponse(ROOT / "tests/fixtures/ch06/browser-probe.js", media_type="text/javascript")

    async def evidence(request: Request):
        """只保存本次浏览器网络证据。参数：测试请求；返回：确认；异常：身份/文件错误。"""
        data = await request.json()
        if data.get("session_id") != runtime.browser_entry["session_id"]:
            return JSONResponse({"error":"test_identity"}, status_code=400)
        write_json(report_dir / "browser-network.json", data)
        return {"saved":True}

    # 测试钩子只挂到当前随机端口应用，不更改app.main或线上路由。
    application.router.routes[:] = [route for route in application.router.routes if getattr(route, "path", None) != "/"]
    application.add_api_route("/", page, methods=["GET"])
    application.add_api_route("/__live__/browser", bootstrap, methods=["GET"])
    application.add_api_route("/__live__/probe.js", script, methods=["GET"])
    application.add_api_route("/__live__/evidence", evidence, methods=["POST"])


def judge(row: dict, expected: dict, *, waiting=False) -> dict:
    """核对真实HTTP、节点、原问和业务边界。参数：观察/标注/等待终态；返回：附失败；异常：结构错误。"""
    state, events = row["state"], row["events"]
    failures = row["failures"]
    understood = raw_result(row["models"], "understand") or {}
    failures.extend(inspect_understanding(expected["query"], expected["expected"], {**understood, **state}))
    if state.get("raw_query") != expected["query"]:
        failures.append("raw_query")
    if state.get("intent") != expected["expected"]["intent"]:
        failures.append("intent")
    kinds = [event["event"] for event in events]
    if row["http_status"] != 200 or "error" in kinds:
        failures.append("HTTP_or_SSE_error")
    if waiting:
        if not kinds or kinds[-1] != "select_order" or "done" in kinds:
            failures.append("waiting_terminal")
        if row["tools"] or "retrieve_policy" in row["nodes"]:
            failures.append("side_effect_before_selection")
    else:
        if kinds[-2:] != ["answer_metadata", "done"] or state.get("completion_metadata", {}).get("status") != "completed":
            failures.append("completion")
        if state.get("route") == "order_policy":
            nodes = row["nodes"]
            if not all(node in nodes for node in ("fetch_order","retrieve_policy","main_agent")) or nodes.index("fetch_order") > nodes.index("retrieve_policy"):
                failures.append("order_before_policy")
            if any(model["stage"] == "decide" for model in row["models"]):
                failures.append("policy_entered_decide")
        if state.get("intent") == "商品咨询":
            if state.get("expanded_queries") != [state.get("standard_query")] or any(model["stage"] == "expand" for model in row["models"]):
                failures.append("ordinary_single_query")
    if any(record.get("failure") for record in state.get("budget", {}).get("records", [])):
        failures.append("model_protocol_or_reference_fallback")
    if row["active_model_streams"] or row["active_graph_nodes"]:
        failures.append("unclosed_stream")
    row["answer"] = "".join(event["data"].get("delta", "") for event in events if event["event"] == "delta")
    row["failures"] = list(dict.fromkeys(failures))
    return row


async def current(url: str, entry: dict) -> dict:
    """通过真实HTTP读取当前保存状态。参数：地址/登记会话；返回：状态；异常：HTTP错误。"""
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.get(url + "/api/actions/current", params={key:entry[key] for key in ("session_id","user_id")})
        response.raise_for_status()
        return response.json()


async def run_turn(runtime, url, item, entry, entries, args, summary, *, waiting=False, payload=None):
    """复用真实取证，逐条落盘。参数：运行资源和期望；返回：当前证据；异常：资源错误。"""
    row = await sample_turn(runtime, url, item, entry, entries, args.report_dir, inspect=False,
        endpoint="/api/chat/resume" if payload else "/api/chat", payload=payload)
    row = judge(row, item, waiting=waiting)
    row.update(id=item["id"], expected=item["expected"])
    append_result(args.report_dir, summary, row)
    return row


async def refund_receipt(url, entry, action, runtime, entries, args, summary):
    """实测重复同键回执和换原因冲突，绝不执行真实退款。参数：地址/来源/资源；返回：None；异常：HTTP错误。"""
    payload = {key:entry[key] for key in ("session_id","user_id")}
    payload.update(source_turn_id=action["source_turn_id"], action_id=action["action_id"], confirmed=True, reason_code="wrong_purchase")
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        first = await client.post(url + "/api/actions/refunds", json=payload)
        again = await client.post(url + "/api/actions/refunds", json=payload)
        changed = await client.post(url + "/api/actions/refunds", json={**payload,"reason_code":"other"})
    data = await register_children(runtime.factory, entry, entries, args.report_dir)
    saved = await current(url, entry)
    checks = {"same_receipt":first.status_code == again.status_code == 200 and first.json() == again.json(),
        "changed_reason_conflict":changed.status_code == 409, "one_pending_application":len(data["refunds"]) == 1 and data["refunds"][0]["status"] == "待处理",
        "current_receipt":saved.get("refund_receipt") == first.json(), "no_ticket":not data["tickets"]}
    append_result(args.report_dir, summary, {"id":"refund-idempotency","checks":checks,"receipt":first.json(),"failures":[key for key, ok in checks.items() if not ok]})


async def run(args, config, summary) -> int:
    """持测试资源锁验收并在finally中精确清理。参数：CLI/限定配置/摘要；返回：0或2；异常：依赖错误。"""
    from sqlalchemy import text
    from app.db.session import create_database
    from tests.ch03_live_runtime import tcp_server
    from tests.verify_ch04_live import resource_lock
    engine, factory = create_database(config)
    entries = []
    cases = json.loads((ROOT / "tests/fixtures/ch06/dialogue_cases.json").read_text(encoding="utf-8"))
    write_json(args.report_dir / "frozen-cases.json", cases)
    waiting_item = {"id":"S4-wait", "query":"我买的东西想退货，帮我看一下可以吗？", "expected":{"intent":"退款退货","order_id":None,"preserve_raw":True}}
    try:
        async with resource_lock(config, prepare=False) as locked:
            owner = await locked.scalar(text("SELECT IS_USED_LOCK(:name)"), {"name":LOCK_NAME})
            await guard_database(factory, owner, entries)
            before = await snapshot(factory)
            summary["counts_before"] = before
            try:
                async with live_runtime(config, factory, entries, args.report_dir) as runtime, tcp_server(runtime) as url:
                    summary["resources"] = await verify_corpus_identity(runtime)
                    for case in cases[:3]:
                        entry = await register_session(factory, entries, args.report_dir)
                        turns = case["turns"][:(2 if case["id"] == "D03" else 3)]
                        for item in turns:
                            await run_turn(runtime, url, item, entry, entries, args, summary)
                    wait_entry = await register_session(factory, entries, args.report_dir)
                    paused = await run_turn(runtime, url, waiting_item, wait_entry, entries, args, summary, waiting=True)
                    wait_current = await current(url, wait_entry)
                    write_json(args.report_dir / "before-restart.json", {"current":wait_current,"budget":paused["state"].get("budget")})
                # 完整关闭保存器/模型/HTTP后重新打开同一SQLite，不能用内存图假装恢复。
                async with live_runtime(config, factory, entries, args.report_dir) as runtime:
                    if args.browser:
                        runtime.configure_test_app = lambda application: configure_browser_probe(application,runtime,args.report_dir)
                    async with tcp_server(runtime) as url:
                        await after_restart(runtime,url,wait_entry,waiting_item,paused,wait_current,entries,args,summary)
            finally:
                summary["cleanup"] = await cleanup(factory,owner,entries,args.report_dir,before)
    finally:
        await engine.dispose()
    summary.update(passed=all(row["passed"] for row in summary["results"]),ended_at=datetime.now(timezone.utc).isoformat())
    write_json(args.report_dir / "summary.json",summary)
    return 0 if summary["passed"] else 2


async def after_restart(runtime,url,wait_entry,waiting_item,paused,wait_current,entries,args,summary):
    """在重开保存器之后续跑并可选开放浏览器。参数：已登记资源/前后证据；返回：None；异常：验收错误。"""
    await runtime.service.reconcile_waiting_turns()
    restored = await current(url, wait_entry)
    checks = {"same_wait":restored == wait_current, "awaiting_input":restored["turn_status"] == "awaiting_input"}
    append_result(args.report_dir, summary, {"id":"waiting-restart","checks":checks,"failures":[key for key, ok in checks.items() if not ok]})
    action = restored["order_selection"]
    payload = {key:wait_entry[key] for key in ("session_id","user_id")}
    payload.update(action_id=action["action_id"],source_turn_id=action["source_turn_id"],order_id="1002")
    item = {**waiting_item,"id":"S4-resume","expected":{**waiting_item["expected"],"order_id":"1002"}}
    # 原问在续跑后仍保持原文，order_id由用户选单确定。
    resumed = await run_turn(runtime, url, item, wait_entry, entries, args, summary, payload=payload)
    old_records = paused["state"]["budget"]["records"]
    checks = {"same_turn":resumed["state"]["context"]["turn_id"] == action["source_turn_id"] and len(wait_entry["turn_ids"]) == 1,
        "budget_preserved":resumed["state"]["budget"]["records"][:len(old_records)] == old_records,
        "no_reclassification":not any(row["stage"] in {"understand","classify"} for row in resumed["models"]),
        "single_user_message":sum(row["role"] == "user" for row in resumed["messages"]) == 1}
    replay = await sample_turn(runtime,url,item,wait_entry,entries,args.report_dir,inspect=False,endpoint="/api/chat/resume",payload=payload)
    checks["replay_no_model"] = replay["state"]["budget"] == resumed["state"]["budget"]
    append_result(args.report_dir,summary,{"id":"same-turn-resume","checks":checks,"failures":[key for key,ok in checks.items() if not ok]})
    saved = await current(url, wait_entry)
    await refund_receipt(url,wait_entry,saved["refund_action"],runtime,entries,args,summary)
    foreign = await register_session(runtime.factory, entries, args.report_dir)
    await run_turn(runtime,url,{"id":"foreign-order","query":"我想退订单2003的东西。","expected":{"intent":"退款退货","preserve_raw":True}},foreign,entries,args,summary,waiting=True)
    if args.browser:
        browser_entry = await register_session(runtime.factory,entries,args.report_dir,prefix="web-")
        runtime.browser_entry = browser_entry
        write_json(args.report_dir / "browser-ready.json", {"url":url,"bootstrap_url":url+"/__live__/browser","entry":browser_entry})
        print("浏览器验收服务已就绪，完成后创建browser-finish.json。", flush=True)
        async with asyncio.timeout(3600):
            while not (args.report_dir / "browser-finish.json").exists():
                await asyncio.sleep(.25)
        browser = json.loads((args.report_dir / "browser-finish.json").read_text(encoding="utf-8"))
        append_result(args.report_dir,summary,{"id":"browser-acceptance",**browser})
        for entry in entries:
            await register_children(runtime.factory, entry, entries, args.report_dir)
        write_json(args.report_dir / "browser-models.json", runtime.models)
        write_json(args.report_dir / "browser-trace.json", runtime.trace.events)


def main(argv=None) -> int:
    """只在显式test和新报告目录运行。参数：CLI；返回：0/2/3；异常：参数由argparse报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--env-file",required=True,type=Path)
    parser.add_argument("--report-dir",required=True,type=Path)
    parser.add_argument("--browser",action="store_true",help="保持本机服务供人工浏览器取证")
    args = parser.parse_args(argv)
    if not args.test:
        parser.error("真实验收必须显式--test")
    args.report_dir = args.report_dir.resolve()
    summary = {"suite":"ch06-live","results":[],"real_model":True,"real_rag":True,
        "started_at":datetime.now(timezone.utc).isoformat(),"model":"gemini-3.5-flash-lite",
        "business_data_origin":"订单物流为项目演示数据；MySQL、SQLite、Milvus、模型和HTTP真实运行",
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "app").rglob("*.py"))}}
    try:
        args.report_dir.mkdir(parents=True,exist_ok=False)
        config = build_config(args)
        if config.chat_model != "gemini-3.5-flash-lite":
            raise ValueError("真实验收固定Lite模型")
        summary["config"] = {key:getattr(config,key) for key in ("chat_model","mysql_database","milvus_database","milvus_hybrid_collection","workflow_checkpoint_path")}
        write_json(args.report_dir / "summary.json",summary)
        return asyncio.run(run(args,config,summary))
    except Exception as error:
        locations, tb = [], error.__traceback__
        while tb:
            locations.append({"file":Path(tb.tb_frame.f_code.co_filename).name,"line":tb.tb_lineno})
            tb = tb.tb_next
        summary.update(passed=False,dependency_error=type(error).__name__,error_locations=locations)
        if args.report_dir.exists() and "config" in summary:
            write_json(args.report_dir / "summary.json",summary)
        print(json.dumps({"error_type":type(error).__name__,"locations":locations}),flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
