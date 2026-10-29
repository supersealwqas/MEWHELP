"""CH08真实验收入口及严格报告判定，缺少强制证据时保持失败状态。"""

from pathlib import Path
import asyncio
from dataclasses import asdict
import json
import hashlib
import re
import shutil
import struct
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4


REQUIRED_CASES = frozenset({
    "builtin_hot_registration", "mcp_three_tools", "mcp_hot_discovery",
    "ticket_confirm", "ticket_cancel", "read_retry_write_once",
    "recovery_and_faults", "browser_actions", "annotated_dialogues",
})


def validate_database(database: str) -> None:
    """拒绝应用库或其他章节数据库作为本章写入目标。

    参数：database为显式目标名称，仅允许mewhelp_test；返回：None；
    异常：ValueError，数据库不在许可范围。
    """
    if database != "mewhelp_test":
        raise ValueError("CH08真实验收仅允许mewhelp_test")


def report_failures(report: dict, report_dir: Path | None = None) -> list[str]:
    """逐项检查真实依赖、必验项目及保留的失败，不以进程退出或自报成功替代证据。

    参数：report为本次实际观察汇总，report_dir为可选证据目录；
    返回：稳定的失败原因列表，空列表仅表示报告证据完整；
    异常：无，报告结构非法也返回失败项。
    """
    failures = []
    if report.get("database") != "mewhelp_test":
        failures.append("database")
    dependencies = report.get("dependencies", {})
    for name in ("model", "mysql", "mcp"):
        if not isinstance(dependencies, dict) or dependencies.get(name) != "real":
            failures.append(f"dependency:{name}")
    rows = report.get("cases", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return [*failures, "case_structure"]
    seen = set()
    for row in rows:
        name = row.get("id")
        if not isinstance(name, str) or name in seen:
            failures.append("duplicate_or_invalid_case")
            continue
        seen.add(name)
        evidence = row.get("evidence")
        if (row.get("status") != "passed" or row.get("failures") != []
                or not isinstance(evidence, list) or not evidence
                or any(not isinstance(path, str) or not path for path in evidence)):
            failures.append(f"case:{name}")
            continue
        if report_dir is not None:
            root = report_dir.resolve()
            # 证据必须真实存在且属于本次目录，不能拿另一轮报告冒充。
            for source in evidence:
                path = (root / source).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    failures.append(f"evidence:{name}")
                    break
    failures.extend(f"missing:{name}" for name in sorted(REQUIRED_CASES - seen))
    return failures


def require(condition: bool, message: str) -> None:
    """在普通和优化模式均执行证据门禁。参数：条件/失败说明；返回：None；异常：ValueError，证据不足。"""
    if not condition:
        raise ValueError(message)


def validate_model_evidence(source: Path, model: str) -> dict:
    """重新核验冻结题集、当前源码、逐轮原始模型与署名审阅，不相信摘要passed。

    参数：source模型评估目录、model本次真实模型名称；返回：重新计算的通过证据。
    异常：ValueError/文件或JSON错误，缺失、旧版、控制模型及错绑证据；审阅者可明确署名为AI。
    """
    from langchain_core.messages import AIMessageChunk
    from app.workflow.model import _text
    from app.tools.contracts import json_hash
    from tests import eval_ch08_tools as evaluation
    metadata = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    cases, fingerprint = evaluation.load_cases(source / "frozen-cases.json")
    require(metadata.get("case_sha256") == fingerprint and metadata.get("model") == model, "模型或冻结题集不匹配")
    require(metadata.get("suite") == "ch08_tools_model_evaluation" and
            metadata.get("model_origin") == "configured_chat_openai", "缺少真实配置模型来源")
    require(metadata.get("feedback_origin") == "simulation" and metadata.get("business_execution") == "none"
            and metadata.get("actual_interrupt") is False, "模型评估与真实业务执行边界不明确")
    evaluation.validate_repetitions(metadata.get("repetitions"))
    current = evaluation._source_hashes()
    require(metadata.get("source_sha256") == current == metadata.get("source_sha256_after"), "模型报告源码已过期或采样期间发生改变")
    require(metadata.get("suite_errors") == [], "模型采样仍有运行错误")
    catalog_rows = metadata.get("catalog", [])
    catalog = {definition["name"]: definition for definition in catalog_rows}
    require(len(catalog) == len(catalog_rows) and all(name in catalog and
            (catalog[name]["identity"]["source"], catalog[name]["identity"]["server"]) == origin
            for name, origin in evaluation.REQUIRED_TOOLS.items()), "模型报告缺少完整可信来源工具清单")
    require(all(definition["schema_hash"] == json_hash(definition["parameters"]) and
            definition["access"] == ("write" if name == "create_ticket" else "read")
            for name, definition in catalog.items()), "模型清单schema或读写类型不符")
    expected = {f"{case['id']}.{repeat}.{number}": (case, number, turn)
                for repeat in range(1, metadata["repetitions"] + 1) for case in cases
                for number, turn in enumerate(case["turns"], 1)}
    rows = json.loads((source / "results.json").read_text(encoding="utf-8"))
    require(isinstance(rows, list) and len(rows) == len(expected) and
            {row.get("id") for row in rows} == set(expected), "模型逐轮记录不完整或重复")
    calls = 0
    for row in rows:
        case, number, turn = expected[row["id"]]
        require(row.get("case_id") == case["id"] and row.get("turn") == number and
                row.get("query") == turn["query"] and row.get("expected") == turn["expected"], "模型题目或标注被替换")
        require(row.get("actual_preview_executed") is False and row.get("database_used") is False and
                row.get("business_tools_executed") == 0 and
                all(item.get("origin") == "simulation" for item in row.get("feedback", [])), "模型报告混入业务执行")
        require(evaluation.inspect_turn(turn["expected"], row)["structural_status"] == "passed", "重算模型结构核验失败")
        records = row.get("models")
        require(isinstance(records, list) and bool(records), "逐轮缺少真实模型请求和原始分片")
        observed_calls, answers = [], []
        for record in records:
            require(record.get("closed") is True and not record.get("error_type") and
                    record.get("turn_id") == row["id"] and record.get("stage") in {"decide", "answer"}, "模型调用未结束或身份错配")
            require(isinstance(record.get("inputs"), list) and any(message.get("type") == "human" and
                    message.get("content") == row["query"] for message in record["inputs"]), "模型请求没有当前用户原问")
            chunks = record.get("chunks")
            require(isinstance(chunks, list) and bool(chunks) and isinstance(record.get("output"), dict), "缺少原始模型分片或输出")
            gathered = None
            for chunk in chunks:
                parsed = AIMessageChunk.model_validate({key: value for key, value in chunk.items() if key != "clock"})
                gathered = parsed if gathered is None else gathered + parsed
            output = gathered.model_dump(mode="json")
            require(output == record["output"], "原始分片与保存模型输出不一致")
            require(output.get("response_metadata", {}).get("finish_reason") in {"stop", "tool_calls"}, "模型未正常结束")
            if record["stage"] == "decide":
                require(bool(record.get("tool_schemas")), "模型决定缺少实际绑定schema")
                bound = {schema["function"]["name"]: schema["function"] for schema in record["tool_schemas"]}
                require(len(bound) == len(record["tool_schemas"]) and set(bound) == {*catalog, "submit_decision"} and
                        all(bound[name].get("parameters") == definition["parameters"] and
                            bound[name].get("description") == definition["description"] for name, definition in catalog.items()),
                        "模型实际绑定工具与发现清单不一致")
                observed_calls.extend({key: call[key] for key in ("id", "name", "args")}
                                      for call in output.get("tool_calls", []) if call["name"] != "submit_decision")
            elif not gathered.tool_calls and _text(gathered).strip():
                answers.append(_text(gathered))
            calls += 1
        require(any(record["stage"] == "decide" for record in records) and observed_calls ==
                [{key: call[key] for key in ("id", "name", "args")} for call in row["calls"]], "工具申请与原始模型输出不一致")
        require((not answers and row.get("answer") == "") if row["preview_candidate"] else
                bool(answers) and row.get("answer") == "".join(answers), "模型正文缺失或与原始回答流不一致")
    reviews = json.loads((source / "semantic-review-applied.json").read_text(encoding="utf-8"))
    # 清掉行内自报结论后再应用完整署名审阅，缺一条不能复用旧semantic_status。
    unsigned = [{key: value for key, value in row.items() if key != "semantic_review"} for row in rows]
    for row in unsigned:
        row["semantic_status"] = "pending"
    reviewed = evaluation.apply_semantic_reviews(unsigned, reviews)
    calculated = evaluation.summarize(reviewed, expected_rows=len(expected))
    require(calculated["passed"], "逐项语义审阅不完整或存在失败")
    return {**calculated, "verified_rows": len(rows), "verified_model_calls": calls, "model": model,
            "case_sha256": fingerprint, "source_sha256": current, "feedback_origin": "simulation"}


def validate_write_timeout(value: dict, attempts: int, timeout: float) -> None:
    """只接受一次真正慢写超时。参数：执行结果/尝试数/秒数；返回：None；异常：ValueError，普通错误或计量不符。"""
    result, database = value["result"], value["database"]
    require(attempts == 1 and result.get("ok") is False and
            (result.get("error") or {}).get("code") == "TOOL_TIMEOUT", "慢写没有发生一次明确超时")
    audits = database["audits"]
    # Windows事件循环可能按monotonic时钟精度提前唤醒；只扣一个实际时钟刻度，仍拒绝立即失败。
    minimum_ms = max(1, round((timeout - time.get_clock_info("monotonic").resolution) * 1000) - 1)
    require(not database["tickets"] and len(audits) == 1 and audits[0]["status"] == "超时" and
            audits[0]["retry_count"] == 0 and audits[0]["duration_ms"] >= minimum_ms,
            "慢写超时审计、耗时或禁止重试证据不符")


async def browser_database(runtime, session_id: str, user_id: str) -> dict:
    """用本次真实factory只读采集本人源消息、工单和审计。

    参数：runtime当前实例、session_id/user_id明确身份；返回：JSON可保存快照；异常：数据库错误或归属不符。
    """
    from sqlalchemy import select, text
    from app.models.customer_service import Conversation, Message, Ticket
    from app.models.tool_audit import ToolAuditLog
    async with runtime.factory() as session:
        validate_database(await session.scalar(text("SELECT DATABASE()")))
        conversation = await session.scalar(select(Conversation).where(Conversation.session_id == session_id))
        require(conversation is not None and conversation.user_id == user_id, "浏览器快照会话不存在或用户不符")
        result = {"session_id": session_id, "user_id": user_id, "conversation_id": conversation.id}
        for key, model in (("messages", Message), ("tickets", Ticket), ("audits", ToolAuditLog)):
            rows = (await session.scalars(select(model).where(model.conversation_id == conversation.id))).all()
            result[key] = [{column.name: getattr(row, column.name) for column in model.__table__.columns} for row in rows]
        return json.loads(json.dumps(result, default=str))


async def capture_browser_requests(runtime, report: Path, context: dict) -> None:
    """处理本轮目录中的只读快照请求，不增加HTTP接口。

    参数：runtime当前实例、report本轮目录、context运行身份及内存快照索引；返回：None。
    异常：路径、身份、文件或数据库错误；请求及输出不得越出本次目录或覆盖已有快照。
    """
    from tests.ch08_live_runtime import write_json
    requests = report / "browser-snapshot-requests"
    requests.mkdir(exist_ok=True)
    for request in sorted(requests.glob("*.json")):
        require(request.resolve().is_relative_to(report.resolve()) and
                re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", request.stem) is not None, "非法浏览器快照请求路径")
        relative = f"browser-snapshots/{request.stem}.json"
        if relative in context["snapshots"]:
            continue
        value = json.loads(request.read_text(encoding="utf-8"))
        require(set(value) == {"session_id", "user_id"} and all(isinstance(item, str) and
                0 < len(item) <= 64 and item.strip() == item for item in value.values()), "非法浏览器快照身份")
        database = await browser_database(runtime, **value)
        target = report / relative
        target.parent.mkdir(exist_ok=True)
        require(target.resolve().is_relative_to(report.resolve()) and not target.exists(), "快照目标已存在或越界")
        write_json(target, {"run_id": context["run_id"], "runtime": context["runtime"], "captured_at": time.time(),
                            "database": database})
        context["snapshots"][relative] = hashlib.sha256(target.read_bytes()).hexdigest()


def browser_artifact(report: Path, reference: dict) -> bytes:
    """读取本轮非空、摘要匹配的证据。参数：目录/相对引用；返回：原始字节；异常：ValueError或文件错误。"""
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"} and
            isinstance(reference["path"], str) and isinstance(reference["sha256"], str), "浏览器证据引用不完整")
    path = (report / reference["path"]).resolve()
    require(path.is_relative_to(report.resolve()) and path.is_file(), "浏览器证据缺失或越出本轮目录")
    raw = path.read_bytes()
    require(bool(raw) and hashlib.sha256(raw).hexdigest() == reference["sha256"], "浏览器证据摘要不符")
    return raw


def browser_pending(database: dict, source_turn_id: str) -> dict | None:
    """取唯一原用户行上的冻结预览。参数：数据库快照/原轮；返回：pending_ticket或None；异常：ValueError，来源不唯一。"""
    rows = [row for row in database["messages"] if row["role"] == "user" and row["turn_id"] == source_turn_id]
    require(len(rows) == 1, "浏览器动作缺少唯一原用户消息")
    return (rows[0].get("rag_metadata") or {}).get("pending_ticket")


def browser_user_messages(database: dict) -> list:
    """提取不可被恢复新增或修改的用户原文。参数：快照；返回：有序身份原文；异常：坏快照。"""
    return sorted((row["id"], row["turn_id"], row["content"]) for row in database["messages"] if row["role"] == "user")


async def validate_browser_evidence(runtime, value: dict, report: Path, context: dict) -> dict:
    """验证八个实际场景的原始证据，再以当前factory回读工单/拒绝审计/源消息。

    参数：当前实例、浏览器报告、本轮目录、内存运行及快照索引；返回：逐项实际库核对结果。
    异常：ValueError/文件/数据库错误，缺场景、旧运行、改快照、缺网络或实际库结果不符。
    """
    required = {"missing_description", "confirm", "cancel", "double_click", "refresh_preview",
                "disconnect_resume", "session_switch", "narrow_screen"}
    require(value.get("schema_version") == 1 and value.get("run_id") == context["run_id"] and
            value.get("runtime") == context["runtime"], "浏览器报告不属于当前运行实例")
    checks = value.get("checks")
    require(isinstance(checks, list) and len(checks) == len(required) and
            {row.get("id") for row in checks if isinstance(row, dict)} == required, "八个浏览器场景必须逐项提供完整记录")
    verified = []
    for row in checks:
        name, session_id, user_id = row["id"], row["session_id"], row["user_id"]
        require(isinstance(row.get("steps"), list) and len(row["steps"]) >= 2 and
                all(isinstance(step, str) and step.strip() for step in row["steps"]), "缺少实际操作步骤")
        snapshots = []
        for phase in ("before", "after"):
            reference = row[phase]
            require(context["snapshots"].get(reference.get("path")) == reference.get("sha256"), "快照不是本运行时采集的原件")
            snapshot = json.loads(browser_artifact(report, reference))
            require(snapshot["run_id"] == context["run_id"] and snapshot["runtime"] == context["runtime"] and
                    snapshot["database"]["session_id"] == session_id and snapshot["database"]["user_id"] == user_id,
                    "浏览器快照运行或会话身份错配")
            snapshots.append(snapshot)
        require(context["started_at"] <= snapshots[0]["captured_at"] < snapshots[1]["captured_at"], "前后快照时间不成立")
        before, after = [snapshot["database"] for snapshot in snapshots]
        require(isinstance(row.get("screenshots"), list) and len(row["screenshots"]) >= 2, "缺少前后页面截图")
        dimensions = []
        for reference in row["screenshots"]:
            raw = browser_artifact(report, reference)
            require(len(raw) >= 33 and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR", "截图不是PNG原件")
            dimensions.append(struct.unpack(">II", raw[16:24]))
        network = json.loads(browser_artifact(report, row["network"]))
        require(network.get("run_id") == context["run_id"] and network.get("session_id") == session_id and
                network.get("user_id") == user_id and bool(network.get("entries")), "缺少当前会话网络证据")
        entries, resumes = network["entries"], []
        origin = urlsplit(context["url"])
        for entry in entries:
            url = urlsplit(entry["url"])
            require((url.scheme, url.netloc) == (origin.scheme, origin.netloc) and
                    context["started_at"] <= entry["started_at"] <= entry["ended_at"] <= time.time(), "网络证据来源或时间错误")
            require(entry.get("status") in {200, 409} or entry.get("disconnected") is True, "网络操作未得到有效响应")
            require(isinstance(entry.get("response_body"), str) and
                    (bool(entry["response_body"]) or entry.get("disconnected") is True), "缺少网络原始响应")
            if url.path == "/api/chat/resume" and entry["method"] == "POST":
                body = entry["request_body"]
                require(body == {"type": "confirm_ticket", "session_id": session_id, "user_id": user_id,
                    "source_turn_id": row["source_turn_id"], "action_id": row["action_id"],
                    "decision": "cancel" if name == "cancel" else "confirm"}, "恢复请求不是原会话原动作")
                resumes.append(entry)
        pending = browser_pending(after, row["source_turn_id"])
        require(isinstance(pending, dict) and pending.get("action", {}).get("action_id") == row["action_id"] and
                pending["action"].get("tool_call_id") == row["tool_call_id"], "浏览器动作与数据库冻结预览不符")
        current = await browser_database(runtime, session_id, user_id)
        saved = browser_pending(current, row["source_turn_id"])
        require(saved is not None and saved["action"] == pending["action"] and
                browser_user_messages(after) == browser_user_messages(current), "当前库不再支持原动作或源消息证据")
        if name == "missing_description":
            from app.schemas.chat import ChatRequest
            require(not before["tickets"] and not after["tickets"] and
                    all(not (message.get("rag_metadata") or {}).get("pending_ticket") for message in before["messages"])
                    and len(browser_user_messages(after)) == len(browser_user_messages(before)) + 1,
                    "缺描述追问或跨轮补充证据不成立")
            chats = [entry for entry in entries if entry["method"] == "POST" and urlsplit(entry["url"]).path == "/api/chat"]
            require(len(chats) >= 2 and all(entry["request_body"].get("session_id") == session_id and
                    # 普通聊天允许省略演示用户字段，按真实请求模型解析，不改网络原文。
                    ChatRequest.model_validate(entry["request_body"]).user_id == user_id for entry in chats),
                    "缺少实际追问与补充请求")
        else:
            previous = browser_pending(before, row["source_turn_id"])
            require(previous is not None and previous["action"] == pending["action"] and
                    previous.get("decision") is None and not before["tickets"] and
                    browser_user_messages(before) == browser_user_messages(after), "预览前置条件或恢复原文身份不符")
        if name in {"confirm", "cancel", "double_click", "disconnect_resume", "session_switch"}:
            require(bool(resumes), "缺少实际确认或取消网络请求")
            decision = "cancel" if name == "cancel" else "confirm"
            require(pending.get("decision") == saved.get("decision") == decision, "首次决定没有保存在原消息")
            expected_count = 0 if name == "cancel" else 1
            require(len(after["tickets"]) == len(current["tickets"]) == expected_count and
                    after["tickets"] == current["tickets"], "确认、取消或重发产生了错误工单数量")
            audits = [audit for audit in current["audits"] if audit["tool_call_id"] == row["tool_call_id"]]
            require(len(audits) == 1 and audits[0]["tool_name"] == "create_ticket" and
                    audits[0]["tool_source"] == "builtin" and audits[0]["status"] == ("权限拒绝" if name == "cancel" else "成功"),
                    "真实库缺少唯一建单/拒绝审计")
            results = [message for message in current["messages"] if message["role"] == "tool" and
                       message["turn_id"] == row["source_turn_id"] and message["tool_call_id"] == row["tool_call_id"]]
            require(len(results) == 1, "缺少唯一原工具结果")
            result = json.loads(results[0]["content"])
            if decision == "confirm":
                ticket = current["tickets"][0]
                require(result.get("ok") is True and result.get("data", {}).get("ticket_no") == ticket["ticket_no"] and
                        ticket["description"] == pending["action"]["description"] and
                        ticket["ticket_type"] == pending["action"]["ticket_type"], "工单回执或冻结参数与真实库不符")
            else:
                require(result.get("ok") is False and (result.get("error") or {}).get("code") == "USER_CANCELLED", "取消没有保存明确拒绝结果")
        if name == "double_click":
            require(row.get("ui", {}).get("click_count", 0) >= 2, "缺少实际双击操作记录")
        if name == "disconnect_resume":
            require(len(resumes) >= 2 and any(entry.get("disconnected") is True for entry in resumes[:-1]) and
                    resumes[-1]["status"] == 200, "缺少断流及同动作重发证据")
        if name in {"refresh_preview", "narrow_screen"}:
            require(not after["tickets"] and pending.get("decision") is None and not resumes and
                    any(entry["method"] == "GET" and urlsplit(entry["url"]).path == "/api/actions/current" and
                        parse_qs(urlsplit(entry["url"]).query).get("session_id") == [session_id] for entry in entries),
                    "预览回读缺失或刷新触发了写入")
        if name == "session_switch":
            ui = row["ui"]
            require(row["other_session_id"] != session_id and ui["visible_session_id"] == row["other_session_id"] and
                    row["action_id"] not in ui["visible_action_ids"] and
                    current["tickets"][0]["ticket_no"] not in ui["visible_ticket_numbers"] and
                    any(entry["started_at"] <= ui["switched_at"] < entry["ended_at"] for entry in resumes),
                    "缺少切换后迟到回执隔离证据")
            other = await browser_database(runtime, row["other_session_id"], user_id)
            require(other["conversation_id"] != current["conversation_id"] and
                    not {ticket["ticket_no"] for ticket in current["tickets"]} & {ticket["ticket_no"] for ticket in other["tickets"]},
                    "两个会话的真实工单归属串用")
        if name == "narrow_screen":
            ui = row["ui"]
            require(0 < ui["viewport_width"] <= 480 and ui["description"] == pending["action"]["description"] and
                    ui["buttons"] == ["确认提交", "取消"] and all(0 < width <= 480 and height > 0 for width, height in dimensions),
                    "窄屏截图、完整描述或按钮证据不符（截图请使用deviceScaleFactor=1）")
        verified.append({"id": name, "session_id": session_id, "action_id": row["action_id"], "database": current})
    return {"run_id": context["run_id"], "runtime": context["runtime"], "checks": verified, "passed": True}


def successful_calls(evidence: dict) -> set[str]:
    """从真实保存结果找出成功工具，不用模型正文猜测执行。

    参数：单轮数据库证据；返回：成功工具名集合；异常：损坏结果JSON。
    """
    messages = evidence["database"]["messages"]
    names = {call["id"]: call["name"] for row in messages for call in row.get("tool_calls") or []
             if "id" in call and "name" in call}
    return {names[row["tool_call_id"]] for row in messages if row["role"] == "tool"
            and row["tool_call_id"] in names and json.loads(row["content"]).get("ok") is True}


async def sample_resume(runtime, action, decision: str, session_id: str) -> dict:
    """用生产恢复入口续跑同一动作并保存真实结果。参数：运行时、预览、决定、会话；返回：证据；异常：恢复或数据库错误。"""
    from app.schemas.workflow import TicketResume
    from tests.ch08_live_runtime import database_snapshot, write_json
    request = TicketResume(type="confirm_ticket", session_id=session_id, user_id="demo-user",
        source_turn_id=action.source_turn_id, action_id=action.action_id, decision=decision)
    await runtime.service.validate_resume(request)
    events = [event.model_dump(mode="json") async for event in runtime.service.stream_resume(request)]
    result = {"request": request.model_dump(mode="json"), "events": events,
              "database": await database_snapshot(runtime, session_id)}
    write_json(runtime.report / ("resume-" + uuid4().hex + ".json"), result)
    return result


async def verify_hot_builtin(runtime) -> dict:
    """在同一已启动实例注册新用途并由真实模型调用。参数：运行时；返回：证据；异常：未满足验收断言。"""
    from app.schemas.tools import ToolResult
    from app.tools.contracts import ToolDefinition, ToolIdentity, ResultPolicy
    from tests.ch08_live_runtime import runtime_identity, sample_turn
    before = runtime_identity(runtime)

    async def hours(arguments, context, scope):
        """返回仅测试注册的营业时间。参数：公开参数/身份/资源；返回：ToolResult；异常：无。"""
        return ToolResult.success({"store_hours": "周一至周六10:00—20:00", "timezone": "Asia/Shanghai"})

    runtime.executor.registry.register_definition(ToolDefinition(ToolIdentity("builtin", None, "query_store_hours"),
        "查询商店门店营业时间和时区；只读用途，不是售后服务时间。", {"type": "object", "properties": {}, "additionalProperties": False},
        hours, "read", ResultPolicy(is_mock=True)))
    after = runtime_identity(runtime)
    evidence = await sample_turn(runtime, "请查询商店门店几点营业、几点关门。")
    assert "query_store_hours" in successful_calls(evidence), "真实模型没有调用新注册工具"
    assert all(before[key] == after[key] for key in ("pid", "registry_id", "graph_id"))
    assert any(row["tool_name"] == "query_store_hours" and row["tool_source"] == "builtin"
               and row["status"] == "成功" for row in evidence["database"]["audits"])
    return {"before": before, "after": after, "turn": evidence}


async def verify_mcp_tools(runtime, servers) -> dict:
    """由真实模型依次调用两个独立MCP来源的三种工具。参数：运行时/进程；返回：证据；异常：验收断言。"""
    from app.tools.mock_data import order_snapshot
    from tests.ch08_live_runtime import sample_turn
    shipped = next(identifier for identifier in ("1001", "1002", "1003", "1004") if order_snapshot(identifier)["tracking_no"])
    rows = []
    for name, query, server in (("query_logistics", f"请查订单{shipped}的物流轨迹。", "logistics"),
        ("query_warranty", "请查订单1001的商品现在是否在保。", "after_sales"),
        ("query_return_progress", "请查订单1001现有退货单的当前处理进度。", "after_sales")):
        evidence = await sample_turn(runtime, query)
        assert name in successful_calls(evidence), f"真实模型未成功调用{name}"
        assert any(row["tool_name"] == name and row["tool_source"] == "mcp" and row["mcp_server"] == server
                   and row["status"] == "成功" for row in evidence["database"]["audits"])
        rows.append(evidence)
    identities = [server.identity() for server in servers]
    assert len({row["pid"] for row in identities}) == 2
    return {"servers": identities, "turns": rows, "data_origin": "真实MCP传输，业务数据为明确标记的演示数据"}


async def verify_mcp_hot(runtime, servers) -> dict:
    """只重启售后进程并热授权新工具，记录单来源离线隔离。参数：运行时/两进程；返回：证据；异常：验收失败。"""
    from tests.ch08_live_runtime import runtime_identity, sample_turn, write_json
    before = runtime_identity(runtime)
    old_server = servers[1].identity()
    await servers[1].stop()
    await runtime.mcp.refresh()
    offline = runtime_identity(runtime)
    names = {row["name"] for row in offline["tools"]}
    assert "query_warranty" not in names and {"query_logistics", "query_store_hours"} <= names
    still_available = await sample_turn(runtime, "请查询商店门店营业时间。")
    assert "query_store_hours" in successful_calls(still_available)
    other_server = await sample_turn(runtime, "请查订单1001的物流轨迹。")
    assert "query_logistics" in successful_calls(other_server), "售后来源离线不能阻断物流来源实际调用"
    await servers[1].start(hot=True)
    permission_path = Path(runtime.config.mcp_permissions_path)
    permissions = json.loads(permission_path.read_text(encoding="utf-8"))
    permissions["tools"].append({"source": "mcp", "server": "after_sales", "name": "query_service_hours",
        "enabled": True, "access": "read", "ownership": "none", "result_policy": {"is_mock": True}})
    write_json(permission_path, permissions)
    evidence = await sample_turn(runtime, "请查询售后人工服务每天几点开始、几点结束。")
    after = runtime_identity(runtime)
    assert "query_service_hours" in successful_calls(evidence)
    assert all(before[key] == after[key] for key in ("pid", "registry_id", "graph_id"))
    assert old_server["pid"] != servers[1].identity()["pid"]
    assert any(row["tool_name"] == "query_service_hours" and row["mcp_server"] == "after_sales"
               and row["status"] == "成功" for row in evidence["database"]["audits"])
    return {"before": before, "offline": offline, "offline_builtin_turn": still_available, "offline_other_mcp_turn": other_server,
            "after": after, "old_server": old_server, "new_server": servers[1].identity(), "turn": evidence}


async def verify_ticket(runtime, *, cancel: bool) -> dict:
    """验证缺描述追问、预览、确认/取消和重发不重复。参数：运行时、取消标记；返回：证据；异常：验收失败。"""
    from tests.ch08_live_runtime import sample_turn
    missing = await sample_turn(runtime, "帮我建个工单")
    session_id = missing["database"]["session_id"]
    assert not missing["database"]["tickets"]
    actions = await runtime.service.get_current_actions(session_id, "demo-user")
    assert actions.ticket_confirmation is None, "缺描述不应出预览"
    preview = await sample_turn(runtime, "订单1001收到时包装破损，猫粮撒漏了。", session_id=session_id)
    actions = await runtime.service.get_current_actions(session_id, "demo-user")
    action = actions.ticket_confirmation
    assert action is not None, "补全描述没有生成预览"
    assert not preview["database"]["tickets"]
    decision = "cancel" if cancel else "confirm"
    resumed = await sample_resume(runtime, action, decision, session_id)
    repeated = await sample_resume(runtime, action, decision, session_id)
    tickets = repeated["database"]["tickets"]
    audits = [row for row in repeated["database"]["audits"] if row["tool_call_id"] == action.tool_call_id]
    assert len(tickets) == (0 if cancel else 1)
    assert len(audits) == 1 and audits[0]["status"] == ("权限拒绝" if cancel else "成功")
    assert resumed["database"]["tickets"] == tickets
    return {"missing": missing, "preview": preview, "action": action.model_dump(mode="json"),
            "resumed": resumed, "repeated": repeated}


async def verify_faults(runtime) -> dict:
    """受控注入读超时、写超时及审计失败，真实回读业务库。参数：运行时；返回：计数和证据；异常：验收失败。"""
    from app.schemas.tools import ToolContext, ToolResult
    from app.tools.contracts import ToolDefinition, ToolIdentity, ToolExecutionScope, WriteAuthorization, json_hash
    from tests.ch08_live_runtime import database_snapshot
    executor = runtime.executor
    counts = {"read": 0, "write": 0, "audit": 0}
    original = (executor.timeout, executor.delay, executor.retries, executor.support, executor.audit)
    rows = []

    async def slow_read(arguments, context, scope):
        """首试超时、次试成功。参数：执行资源；返回：ToolResult；异常：取消。"""
        counts["read"] += 1
        if counts["read"] == 1:
            await asyncio.sleep(0.2)
        return ToolResult.success({"attempts": counts["read"], "status": "已核实"})

    async def status(frame):
        """保存执行状态。参数：状态帧；返回：None；异常：无。"""
        rows.append(frame)

    async def authorize(binding, arguments, context):
        """为明确标记的故障注入提供测试授权。参数：原绑定/参数/上下文；返回：冻结凭据；异常：无。"""
        return WriteAuthorization(context.conversation_id, context.turn_id, "fault-injection", context.tool_call_id,
                                  json_hash(arguments), "confirm", "action")

    async def execute(name, arguments):
        """保存故障测试的原申请后经同一执行器调用。参数：名称/参数；返回：结果和数据库证据；异常：执行失败。"""
        context = ToolContext(session_id="ch08-fault-" + uuid4().hex, user_id="demo-user", turn_id=str(uuid4()),
            tool_call_id="fault-" + uuid4().hex, user_message="CH08受控故障注入")
        context = await executor.chat.start_turn(context, context.user_message)
        call = {"id": context.tool_call_id, "name": name, "args": arguments}
        await executor.chat.save_tool_calls(context, [call])
        result = await executor.execute(call, context, status, scope=ToolExecutionScope(authorize_write=authorize))
        if result.ok:
            await executor.chat.finish_turn(context, "受控故障注入已结束，结果以原工具回执为准。")
        else:
            await executor.chat.mark_turn(context, "failed")
        return {"result": result.model_dump(mode="json"), "database": await database_snapshot(runtime, context.session_id)}

    class SlowSupport:
        """只在本验收写边界睡眠，模拟未提交的外部超时。"""
        async def create_ticket(self, context, description, ticket_type):
            """记录写尝试并阻塞。参数：可信建单参数；返回：不会正常返回；异常：超时取消。"""
            counts["write"] += 1
            await asyncio.sleep(0.2)
            raise RuntimeError("慢写未被期限取消，不能将普通错误算作超时通过")

    class BrokenAudit:
        """故意失败的独立审计边界，用于证明真实回执优先。"""
        async def record(self, entry):
            """记录一次注入并报错。参数：审计内容；返回：不会返回；异常：OSError。"""
            counts["audit"] += 1
            raise OSError("CH08 deliberate audit fault")

    executor.registry.register_definition(ToolDefinition(ToolIdentity("builtin", None, "query_retry_probe"),
        "验收专用读取超时探针，普通用户不要调用。", {"type": "object", "properties": {}, "additionalProperties": False}, slow_read, "read"))
    try:
        executor.timeout, executor.delay, executor.retries = 0.05, 0.01, 1
        read = await execute("query_retry_probe", {})
        executor.support = SlowSupport()
        write = await execute("create_ticket", {"description": "受控超时测试", "ticket_type": "咨询"})
        validate_write_timeout(write, counts["write"], executor.timeout)
        assert counts["read"] == 2 and counts["write"] == 1
        assert read["result"]["ok"] and not write["result"]["ok"] and not write["database"]["tickets"]
        assert read["database"]["audits"][0]["retry_count"] == 1
        assert write["database"]["audits"][0]["retry_count"] == 0
        assert read["database"]["audits"][0]["duration_ms"] >= 50
        executor.timeout, executor.support, executor.audit = original[0], original[3], BrokenAudit()
        receipt = await execute("create_ticket", {"description": "审计故障不丢失工单回执的验收记录", "ticket_type": "咨询"})
        assert receipt["result"]["ok"] and len(receipt["database"]["tickets"]) == 1 and counts["audit"] == 1
        assert receipt["database"]["tickets"][0]["ticket_no"] == receipt["result"]["data"]["ticket_no"]
        return {"injection": "仅测试驱动直接提供冻结授权；真实用户确认另由ticket_confirm与浏览器证明",
                "counts": counts, "statuses": rows, "read": read, "write": write, "audit_fault_receipt": receipt}
    finally:
        executor.timeout, executor.delay, executor.retries, executor.support, executor.audit = original


async def run_suite(args) -> int:
    """启动本次独立服务、执行验收并严格汇总缺项。参数：CLI参数；返回：退出码；异常：取消透传。"""
    import uvicorn
    from app.main import create_app
    from contextlib import asynccontextmanager
    from tests.ch08_live_runtime import OwnedServer, configure, free_port, live_runtime, runtime_identity, write_json
    validate_database(args.database)
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    summary = {"database": args.database, "dependencies": {}, "cases": [], "failures": []}
    servers = [OwnedServer("mcp_servers.logistics", free_port(), report, "logistics"),
               OwnedServer("tests.fixtures.ch08.hot_tool", free_port(), report, "after-sales")]
    config, knowledge_config = configure(args.env_file, args.database, report, tuple(server.port for server in servers))

    async def record(name, operation):
        """保留每项成功证据或原始异常，失败不静默丢弃。参数：名称/异步操作；返回：证据或None；异常：控制信号。"""
        try:
            evidence = await operation()
            row = {"id": name, "status": "passed", "failures": [], "evidence": [name + ".json"]}
        except Exception as error:
            evidence = {"error_type": type(error).__name__, "message": str(error)}
            row = {"id": name, "status": "failed", "failures": [str(error)], "evidence": [name + ".json"]}
        write_json(report / (name + ".json"), evidence)
        summary["cases"].append(row)
        write_json(report / "summary.json", summary)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return evidence if row["status"] == "passed" else None

    try:
        for server in servers:
            await server.start()
        async with live_runtime(config, knowledge_config, report) as runtime:
            summary["dependencies"] = {"model": "real", "mysql": "real", "mcp": "real"}
            summary["runtime"] = runtime_identity(runtime)
            summary["model"] = config.chat_model
            await record("builtin_hot_registration", lambda: verify_hot_builtin(runtime))
            await record("mcp_three_tools", lambda: verify_mcp_tools(runtime, servers))
            await record("mcp_hot_discovery", lambda: verify_mcp_hot(runtime, servers))
            await record("ticket_confirm", lambda: verify_ticket(runtime, cancel=False))
            await record("ticket_cancel", lambda: verify_ticket(runtime, cancel=True))
            faults = await record("read_retry_write_once", lambda: verify_faults(runtime))

            async def recovery():
                """检查真实审计故障回执及落库后重复回读证据。参数：无；返回：引用内容；异常：断言。"""
                assert faults is not None, "故障注入失败"
                confirmed = next(row for row in summary["cases"] if row["id"] == "ticket_confirm")
                assert confirmed["status"] == "passed", "确认重发恢复未通过"
                return {"audit_failure": faults["audit_fault_receipt"], "durable_replay": "ticket_confirm.json",
                        "scope": "实际库回执及同动作重发；刷新、断流和重启由浏览器与集成测试另列"}
            await record("recovery_and_faults", recovery)

            @asynccontextmanager
            async def runtime_factory():
                """让浏览器HTTP复用本次已验收实例。参数：无；返回：共享服务；异常：无。"""
                yield runtime.service

            from tests.fixtures.ch08.browser_observer import BrowserObserver
            observed_app = BrowserObserver(create_app(runtime_factory), runtime, report)
            server = uvicorn.Server(uvicorn.Config(observed_app, host="127.0.0.1",
                                                  port=args.serve_port or free_port(), log_level="warning"))
            web_task = asyncio.create_task(server.serve())
            try:
                while not server.started and not web_task.done():
                    await asyncio.sleep(0.1)
                if web_task.done():
                    await web_task
                    raise RuntimeError("验收网页服务未启动")
                browser_context = {"run_id": uuid4().hex, "url": f"http://127.0.0.1:{server.config.port}",
                    "runtime": runtime_identity(runtime), "database": args.database, "started_at": time.time(), "snapshots": {}}
                write_json(report / "browser-ready.json", {key: value for key, value in browser_context.items() if key != "snapshots"})
                deadline = asyncio.get_running_loop().time() + args.browser_timeout
                browser = report / "browser-evidence.json"
                while not browser.is_file() and asyncio.get_running_loop().time() < deadline:
                    await capture_browser_requests(runtime, report, browser_context)
                    await asyncio.sleep(1)
                await capture_browser_requests(runtime, report, browser_context)

                async def browser_evidence():
                    """核对实际浏览器记录和逐项证据文件。参数：无；返回：浏览器记录；异常：缺证据。"""
                    value = json.loads(browser.read_text(encoding="utf-8"))
                    return await validate_browser_evidence(runtime, value, report, browser_context)
                await record("browser_actions", browser_evidence)
            finally:
                server.should_exit = True
                await web_task

            async def model_evidence():
                """导入冻结标注的实际人工语义复核结果。参数：无；返回：汇总；异常：缺项或未通过。"""
                if args.model_report is None:
                    raise ValueError("缺少46轮模型及逐条语义复核报告")
                source = args.model_report.resolve()
                model_summary = validate_model_evidence(source, config.chat_model)
                shutil.copytree(source, report / "model-evaluation")
                return model_summary
            await record("annotated_dialogues", model_evidence)
    except Exception as error:
        summary["failures"].append({"error_type": type(error).__name__, "message": str(error)})
    finally:
        for server in reversed(servers):
            await server.stop()
        summary["validation_failures"] = report_failures(summary, report)
        summary["passed"] = not summary["failures"] and not summary["validation_failures"]
        write_json(report / "summary.json", summary)
    return 0 if summary["passed"] else 1


def main() -> int:
    """解析显式真实运行参数。参数：命令行；返回：退出码；异常：目标或路径非法。"""
    import argparse
    parser = argparse.ArgumentParser(description="CH08真实模型、MySQL、MCP、确认及浏览器验收")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--serve-port", type=int)
    parser.add_argument("--browser-timeout", type=int, default=0)
    return asyncio.run(run_suite(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
