"""对现有 SSE 入口执行真实模型标注样例，核对独立 MySQL 测试库并生成可复核报告。"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
from sqlalchemy import select, text
from app.config import Settings
from app.db.seed import FAQ_SEEDS
from app.db.session import create_database
from app.models.customer_service import Conversation, FAQ, Message, Ticket


async def sample(client, base_url, case, session_id):
    """收集真实网络增量。参数：HTTP 客户端、地址、标注和会话；返回：事件记录；异常：网络或协议错误。"""
    started = monotonic()
    events, name, data = [], "message", []
    async with client.stream("POST", f"{base_url}/api/chat", json={"session_id": session_id, "message": case["message"]}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                raw = "\n".join(data)
                payload = raw if raw == "[DONE]" else json.loads(raw)
                events.append({"event": name, "data": payload, "seconds": round(monotonic() - started, 3)})
                name, data = "message", []
    return events


async def run(args):
    """执行标注集并输出证据。参数：命令行选项；返回：是否全部确定性检查通过；异常：配置或数据库错误。"""
    config = Settings(_env_file=ROOT / ".env")
    if config.mysql_test_database != "mewhelp_test":
        raise ValueError("评估只能使用 mewhelp_test")
    engine, factory = create_database(config, database="mewhelp_test")
    run_id = uuid4().hex[:12]
    report = ["# Ch02 真实模型与 MySQL 评估", "",
        f"时间：{datetime.now(timezone(timedelta(hours=8))).isoformat()}；运行：{run_id}",
        "", "以下自动检查只验证路由、参数、数据库事实与流协议；回答忠实度仍需逐例复核。", ""]
    passed = True
    try:
        async with factory() as session:
            if await session.scalar(text("SELECT DATABASE()")) != "mewhelp_test":
                raise ValueError("目标不是测试库，拒绝评估")
            rows = (await session.scalars(select(FAQ).order_by(FAQ.id))).all()
            snapshot = [(r.id, r.question, r.answer, r.category) for r in rows]
        if snapshot != FAQ_SEEDS:
            raise ValueError("实际 FAQ 与当前种子不一致，请停评估服务后只重建测试库")
        report.extend(["## 已核对 FAQ 快照", "", "```json", json.dumps(snapshot, ensure_ascii=False, indent=2), "```", ""])
        cases = json.loads((ROOT / "tests/fixtures/ch02_cases.json").read_text(encoding="utf-8"))
        async with httpx.AsyncClient(timeout=120) as client:
            for case in cases:
                if args.only and case["id"] not in args.only.split(","):
                    continue
                session_id = f"eval-{run_id}-{case.get('session', case['id'])}"
                events = await sample(client, args.base_url.rstrip("/"), case, session_id)
                async with factory() as session:
                    conversation = await session.scalar(select(Conversation).where(Conversation.session_id == session_id))
                    if conversation is None:
                        raise ValueError("未在测试库发现本次会话，请确认服务 MYSQL_DATABASE=mewhelp_test")
                    messages = (await session.scalars(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id))).all()
                    turn_id = messages[-1].turn_id
                    messages = [m for m in messages if m.turn_id == turn_id]
                    calls = [c for m in messages for c in (m.tool_calls or [])]
                    results = [json.loads(m.content) for m in messages if m.role == "tool"]
                    ticket = None
                    if case["tool"] == "create_ticket" and results and results[0].get("ok"):
                        ticket = await session.get(Ticket, results[0]["data"]["ticket_no"])
                deltas = [e for e in events if isinstance(e["data"], dict) and "delta" in e["data"]]
                answer = "".join(e["data"]["delta"] for e in deltas)
                checks = {
                    "route": [c["name"] for c in calls] == ([case["tool"]] if case["tool"] else []),
                    "arguments": not case.get("args") or (len(calls) == 1 and calls[0]["args"] == case["args"]),
                    "done": bool(events) and events[-1]["data"] == "[DONE]",
                    "completed": all(m.turn_status == "completed" for m in messages),
                    "nonempty_answer": bool(answer.strip()),
                    "result_pairing": len(results) == len(calls),
                    "result_success": all(r["ok"] for r in results),
                }
                if case["tool"]:
                    checks["status_before_text"] = bool(deltas) and events[0]["event"] == "tool_status" and events[0]["seconds"] <= deltas[0]["seconds"]
                if "matches" in case:
                    checks["faq_matches"] = bool(results) and bool(results[0]["data"]) is case["matches"]
                if case["tool"] == "create_ticket":
                    checks["ticket_committed"] = ticket is not None and ticket.conversation_id == conversation.id
                ok = all(checks.values())
                passed = passed and ok
                print(f"{case['id']}: {'PASS' if ok else 'FAIL'} | chunks={len(deltas)} | {round(events[-1]['seconds'], 2) if events else 0}s", flush=True)
                evidence = {"session_id": session_id, "input": case["message"], "checks": checks,
                    "calls": calls, "results": results, "answer": answer, "delta_count": len(deltas),
                    "events": [{"event": e["event"], "data": e["data"], "seconds": e["seconds"]} for e in events if e not in deltas],
                    "first_delta_seconds": deltas[0]["seconds"] if deltas else None,
                    "last_delta_seconds": deltas[-1]["seconds"] if deltas else None}
                report.extend([f"## {case['id']} — {'PASS' if ok else 'FAIL'}", "", f"回答复核要求：{case['review']}", "",
                    "```json", json.dumps(evidence, ensure_ascii=False, indent=2), "```", ""])
                # 每例立即落盘；中断不会丢失已取得的证据。
                Path(args.report).write_text("\n".join(report), encoding="utf-8")
    finally:
        await engine.dispose()
    return passed


def main():
    """解析评估命令。参数：命令行；返回：退出码；异常：配置与连接异常。"""
    parser = argparse.ArgumentParser(description="真实模型单工具聊天评估（只读 mewhelp_test 记录）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--report", default="dev-notes/ch02-evaluation.md")
    parser.add_argument("--only", default="", help="逗号分隔样例 ID；指代样例必须同时运行前后两条")
    return 0 if asyncio.run(run(parser.parse_args())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
