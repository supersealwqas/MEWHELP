"""CH05独立真实验收：显式测试资源、确认增票、真流停止和新进程SQLite恢复。

不重复Task7纯UI点击；本入口证明真实模型分类到建议再到确认接口的完整服务器链。
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

from tests.ch05_live_runtime import (CASE_PATH, ROOT, append_result, build_config, guard_database,
    live_runtime, register_children, register_session, run_cli, sample_turn, snapshot, write_json)


async def confirm_complaint(runtime, url: str, row: dict, entry: dict, entries: list, args) -> dict:
    """真实建议必须经true确认才增票，重复确认同票。参数：资源/来源/登记/参数；返回：证据；异常：HTTP/持久化错误。"""
    import httpx
    actions = row["state"].get("suggested_actions", [])
    action = next((item for item in actions if item["kind"] == "create_ticket"), None)
    result = {"id": "confirmed_ticket", "session_id": entry["session_id"], "failures": [], "before": await snapshot(runtime.factory, entry)}
    if action is None:
        result["failures"].append("missing_real_complaint_suggestion")
        return result
    body = {"session_id": entry["session_id"], "user_id": entry["user_id"], "source_turn_id": action["source_turn_id"],
            "action_id": action["action_id"], "confirmed": True}
    entry["turn_ids"].append(str(uuid5(NAMESPACE_URL, f"mewhelp:ch05:confirm:{action['action_id']}")))
    write_json(args.report_dir / "manifest.json", entries)
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        rejected = await client.post(url + "/api/actions/tickets", json={**body, "confirmed": False})
        result["unconfirmed_status"] = rejected.status_code
        result["after_rejected"] = await snapshot(runtime.factory, entry)
        confirmed = await client.post(url + "/api/actions/tickets", json=body)
        result.update(confirmed_status=confirmed.status_code, receipt=confirmed.json())
        repeated = await client.post(url + "/api/actions/tickets", json=body)
        result.update(retry_status=repeated.status_code, retry_receipt=repeated.json())
    result["after"] = await register_children(runtime.factory, entry, entries, args.report_dir)
    tickets = result["after"]["tickets"]
    checks = {"real_two_suggestions": {item["kind"] for item in actions} == {"handoff", "create_ticket"},
        "no_write_before_confirmation": not result["before"]["tickets"] and not result["after_rejected"]["tickets"] and rejected.status_code == 422,
        "one_ticket": len(tickets) == 1 and confirmed.status_code == 200,
        "same_receipt": repeated.status_code == 200 and result["receipt"] == result["retry_receipt"],
        "original_question": len(tickets) == 1 and tickets[0]["description"] == row["query"],
        "conversation_continues": result["after"]["conversation"]["status"] == "进行中"}
    result.update(checks=checks, request=body)
    result["failures"] = [key for key, passed in checks.items() if not passed]
    return result


async def acceptance(runtime, url: str, cases: list, entries: list, args, summary: dict) -> None:
    """逐项执行真实五项验收和保护。参数：资源/冻结集/报告；返回：None；异常：资源错误。"""
    from app.knowledge.answering import EvidenceCitation
    from app.schemas.tools import ToolContext
    from app.schemas.workflow import GateDecision
    by_id = {case["id"]: case for case in cases}
    complaint_result = None
    for identifier in ("A02", "A03", "A05", "A06", "A07"):
        entry = await register_session(runtime.factory, entries, args.report_dir)
        expected = by_id[identifier]["turns"][0]
        row = await sample_turn(runtime, url, expected, entry, entries, args.report_dir)
        row.update(id="acceptance_" + identifier, expected=expected)
        append_result(args.report_dir, summary, row)
        if identifier == "A02":
            first = next((event["received_at"] for event in row["events"] if event["event"] == "delta"), None)
            ends = [model.get("ended") for model in row["models"] if model["stage"] == "answer"]
            passed = first is not None and bool(ends) and first < ends[-1]
            append_result(args.report_dir, summary, {"id": "real_stream_timing", "first_delta_at": first,
                "answer_model_ended": ends, "failures": [] if passed else ["first_delta_not_before_answer_end"]})
        elif identifier == "A05":
            complaint_result = await confirm_complaint(runtime, url, row, entry, entries, args)
            append_result(args.report_dir, summary, complaint_result)
        elif identifier == "A07":
            state = row["state"]
            gate = state.get("gate") or {}
            if gate.get("passed") is False and state.get("completion_metadata", {}).get("status") == "completed":
                # 真实完成事务重试；不得改变自然检索结果来制造弱证据。
                await runtime.service.chat.finish_turn(ToolContext.model_validate(state["context"]), state["answer"],
                    low_confidence=GateDecision.model_validate(gate).low_confidence,
                    citations=[EvidenceCitation.model_validate(item) for item in state["evidence"]],
                    suggested_actions=[], workflow_metadata={key: state[key] for key in ("intent", "route", "stop_reason", "budget", "tool_count", "decision_count")})
                after = await register_children(runtime.factory, entry, entries, args.report_dir)
                failures = [] if len(row["pool_rows"]) == len(after["pool_rows"]) == 1 else ["pool_not_idempotent"]
            else:
                after, failures = {}, ["natural_low_evidence_not_observed"]
            append_result(args.report_dir, summary, {"id": "low_evidence_once", "before": row["pool_rows"], "after": after, "failures": failures})
    clarify = await register_session(runtime.factory, entries, args.report_dir)
    row = await sample_turn(runtime, url, by_id["A04"]["turns"][0], clarify, entries, args.report_dir)
    row["id"] = "before_process_restart"
    append_result(args.report_dir, summary, row)
    summary["restart_request"] = {"entry_id": clarify["id"], "expected": by_id["A04"]["turns"][1],
        "prior_turn_id": row["state"].get("context", {}).get("turn_id"),
        "ticket_request": complaint_result.get("request") if complaint_result else None,
        "receipt": complaint_result.get("receipt") if complaint_result else None}
    entry = await register_session(runtime.factory, entries, args.report_dir)
    stopped = await sample_turn(runtime, url, by_id["A02"]["turns"][0], entry, entries, args.report_dir, stop=True)
    stopped["id"] = "stop_real_answer_stream"
    append_result(args.report_dir, summary, stopped)
    # 同一会话下一轮应正常开始且不带旧物流意图/工具。
    continued = await sample_turn(runtime, url, by_id["A06"]["turns"][0], entry, entries, args.report_dir)
    continued["id"] = "chat_after_stop"
    append_result(args.report_dir, summary, continued)


async def restart_process(args, config, factory, owner: int, entries: list, summary: dict) -> None:
    """旧HTTP/SQLite关闭后启动新解释器，父进程继续持锁。参数：本次资源；返回：None；异常：子进程超时或失败。"""
    payload = {**summary["restart_request"], "owner": owner}
    request_path = args.report_dir / "restart-request.json"
    write_json(request_path, payload)
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "tests.verify_ch05_live", "--test",
        "--env-file", str(args.env_file), "--report-dir", str(args.report_dir), "--restart-request", str(request_path),
        "--token-budget", str(config.token_budget), "--request-token-budget", str(config.request_token_budget),
        cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 600)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    summary["restart_process"] = {"exit_code": process.returncode, "stdout_bytes": len(stdout), "stderr_bytes": len(stderr)}
    # 只载入同一manifest，子进程新增的轮次已经由公开入口回调登记。
    entries[:] = json.loads((args.report_dir / "manifest.json").read_text(encoding="utf-8"))
    await guard_database(factory, owner, entries)
    result_path = args.report_dir / "restart-result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {"id": "process_restart", "failures": ["child_produced_no_evidence"]}
    if process.returncode != 0:
        result["failures"].append("restart_child_exit:" + str(process.returncode))
    append_result(args.report_dir, summary, result)


async def child_restart(args) -> int:
    """新进程读取旧SQLite并真实续问，再重试原票。参数：受父锁保护的请求路径；返回：退出码；异常：资源错误。"""
    import httpx
    from app.db.session import create_database
    from app.workflow.state import checkpoint_thread_id
    from tests.ch03_live_runtime import tcp_server
    config = build_config(args, resume=True)
    payload = json.loads(args.restart_request.read_text(encoding="utf-8"))
    entries = json.loads((args.report_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = next(row for row in entries if row["id"] == payload["entry_id"])
    engine, factory = create_database(config)
    result = {"id": "process_restart", "failures": []}
    try:
        await guard_database(factory, payload["owner"], entries)
        async with live_runtime(config, factory, entries, args.report_dir) as runtime, tcp_server(runtime) as url:
            previous = (await runtime.service.graph.aget_state({"configurable": {"thread_id": checkpoint_thread_id(entry["id"])}})).values
            result["persisted_prior_turn"] = previous.get("context", {}).get("turn_id")
            if result["persisted_prior_turn"] != payload["prior_turn_id"]:
                result["failures"].append("sqlite_previous_turn_missing")
            row = await sample_turn(runtime, url, payload["expected"], entry, entries, args.report_dir)
            result["continued_turn"] = row
            result["failures"].extend(row["failures"])
            if payload["ticket_request"]:
                async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
                    response = await client.post(url + "/api/actions/tickets", json=payload["ticket_request"])
                result.update(ticket_status=response.status_code, receipt=response.json())
                if response.status_code != 200 or response.json() != payload["receipt"]:
                    result["failures"].append("restart_ticket_receipt_changed")
            else:
                result["failures"].append("restart_ticket_request_missing")
    finally:
        await engine.dispose()
    write_json(args.report_dir / "restart-result.json", result)
    return 1 if result["failures"] else 0


def main(argv=None) -> int:
    """先验证test再导入应用配置。参数：可选命令参数；返回：退出码；异常：参数错误由argparse退出2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=CASE_PATH)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--request-token-budget", type=int)
    parser.add_argument("--restart-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.restart_request:
        try:
            return asyncio.run(child_restart(args))
        except Exception as error:
            write_json(args.report_dir / "restart-result.json", {"id": "process_restart", "failures": ["child_error:" + type(error).__name__]})
            return 2
    return run_cli(args, suite="verify")


if __name__ == "__main__":
    raise SystemExit(main())
