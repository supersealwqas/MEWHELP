"""CH09受控真实知识模型样例：默认图、原策略、独立SQLite及逐条原始流报告。"""

import argparse
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path: Path, value) -> None:
    """创建报告原件，不覆盖先前结果。参数：路径/可序列化内容；返回：None；异常：文件冲突或写入错误。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


async def evaluate(case: dict, directory: Path, config) -> dict:
    """默认图处理受控证据并真实调用知识模型，标注仅由外部判定读取。

    参数：case标注、directory独占目录、config正式配置；返回：原始事件/结构结果/入池及判断。
    异常：网络或流程错误保留为逐条失败，不修改生产数据库。
    """
    from sqlalchemy import select
    from langgraph.types import Command
    from app.core.llm import get_chat_model
    from app.core.structured import parse_tool_output
    from app.knowledge.answering import KnowledgeAnswer, FIXED_KNOWLEDGE_REFUSAL
    from app.models.rag_quality import LowConfidenceQuestion
    from app.schemas.chat import ChatRequest
    from app.schemas.tools import ToolContext, ToolResult
    from app.schemas.workflow import AgentDecision, UnderstandingResult
    from app.workflow.model import WorkflowModelAdapter
    from app.workflow.state import checkpoint_thread_id, new_turn_state
    from tests.ch05_live_runtime import RecordingChat
    from tests.workflow.graph_fakes import ranked
    from tests.workflow.service_fakes import service_runtime
    from tests.workflow.ch08_ticket_fakes import ticket_runtime, saved_decision
    from tests.workflow.test_ch08_ticket_interrupt import ticket_call

    records, events, pool, final = [], [], [], None
    directory.mkdir()
    fixture = ranked(0.99, answer=case["evidence"])
    decisions = [AgentDecision(kind="answer")]
    if case.get("ticket"):
        decisions = [AgentDecision(kind="tools", tool_calls=[ticket_call(description=case["query"].split("，", 1)[1])]), AgentDecision(kind="answer")]
    scripts = {"intents": [case.get("intent", "商品咨询")], "decisions": decisions, "ranked": [fixture]}
    if case.get("order"):
        scripts["results"] = [ToolResult(ok=True, data=case["order"])]
        scripts["understandings"] = [UnderstandingResult(resolved_query=case["query"], standard_query=case["query"],
            needs_resolution=False, order_id="1001")]

    def factory(**options):
        """透明记录真实模型输入输出，不改写请求。参数：正式适配器选项；返回：模型包装；异常：配置错误。"""
        return RecordingChat(get_chat_model(**options), records, "answer")

    runtime = ticket_runtime if case.get("ticket") else service_runtime
    error, detail = None, None
    try:
        async with runtime(directory, **scripts) as resources:
            service, deps, database = resources[:3]
            # 复用正式冻结v2制品，不用依赖辅助器生成的测试门槛；实际证据是明确标注的受控夹具。
            deps.settings = config
            deps.model.answer_knowledge = WorkflowModelAdapter(model_factory=factory, stream_usage=True).answer_knowledge
            if case.get("ticket"):
                context = await deps.chat.start_turn(ToolContext(session_id=case["id"], user_id="demo-user",
                    turn_id=case["id"], user_message=case["query"]), case["query"])
                graph_config = {"configurable": {"thread_id": checkpoint_thread_id(context.conversation_id)}}
                async for item in service.graph.astream(new_turn_state(context, [], 0), graph_config,
                        stream_mode="custom", durability="sync", subgraphs=True):
                    events.append(item[1])
                snapshot = await service.graph.aget_state(graph_config)
                if not snapshot.interrupts:
                    raise ValueError("混合工单样例没有进入真实确认暂停")
                async for item in service.graph.astream(Command(resume=await saved_decision(database, context)),
                        graph_config, stream_mode="custom", durability="sync", subgraphs=True):
                    events.append(item[1])
                final = (await service.graph.aget_state(graph_config)).values
            else:
                if case.get("history"):
                    prior = await deps.chat.start_turn(ToolContext(session_id=case["id"], user_id="demo-user",
                        turn_id="history", user_message="退款批准后多久到银行卡？"), "退款批准后多久到银行卡？")
                    await deps.chat.finish_turn(prior, case["history"])
                events = [item.model_dump(mode="json") async for item in service.stream_turn(
                    ChatRequest(session_id=case["id"], message=case["query"]))]
            async with database() as session:
                rows = (await session.scalars(select(LowConfidenceQuestion))).all()
                pool = [{"source": row.source, "reason": row.reason, "raw_question": row.raw_question,
                         "retrieved_chunks": row.retrieved_chunks} for row in rows]
    except Exception as failure:
        error = type(failure).__name__
        detail = str(failure) if isinstance(failure, ValueError) else None
    from langchain_core.messages import AIMessageChunk
    parsed = []
    for record in records:
        try:
            parsed.append(parse_tool_output(AIMessageChunk.model_validate(record["output"]), KnowledgeAnswer,
                "submit_knowledge_answer").model_dump(mode="json"))
        except (ValueError, TypeError):
            parsed.append(None)
    answer = "".join(item["data"]["delta"] for item in events if item.get("event") == "delta")
    latest = parsed[-1] if parsed else None
    checks = {"no_error": error is None, "raw_useful": latest is not None and latest["useful"] == case["expected_useful"],
        "completion": (final or {}).get("completion_metadata", {}).get("status") == "completed" if case.get("ticket")
            else bool(events) and events[-1]["event"] == "done",
        "required": all(text in answer for text in case.get("required", [])),
        "fixed_refusal": case["expected_useful"] or answer.endswith(FIXED_KNOWLEDGE_REFUSAL),
        "pool": (not pool if case["expected_useful"] else len(pool) == 1 and pool[0]["source"] == "self_check")}
    result = {"id": case["id"], "case": case, "answer": answer, "raw_structured": parsed, "records": records,
        "events": events, "pool": pool, "checks": checks, "passed": all(checks.values()), "error_type": error,
        "validation_detail": detail}
    save(directory / "result.json", result)
    return result


async def run(args) -> int:
    """执行冻结样例并保存模型/策略/样例指纹和每条原件。参数：CLI；返回：退出码；异常：路径/配置错误。"""
    from app.config import settings
    from app.knowledge.confidence import load_evidence_policy, evidence_policy_digest
    from app.knowledge.pipeline import production_protocols
    from app.workflow.prompts import KNOWLEDGE_PROMPT
    path = ROOT / "tests/fixtures/ch09/self_check_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            raise ValueError("样例ID不存在")
    output = args.output_dir.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    policy = load_evidence_policy(Path(settings.evidence_policy_path), protocols=production_protocols(settings))
    metadata = {"model": settings.chat_model, "policy_version": policy.version, "policy_digest": evidence_policy_digest(policy),
        "dataset_sha256": sha256(path.read_bytes()).hexdigest(), "prompt_sha256": sha256(KNOWLEDGE_PROMPT.encode()).hexdigest(),
        "limitation": "受控证据与路由脚本；真实生产知识模型、默认图、冻结正式策略、独立SQLite完成事务。非真实检索质量或生产全链路验收。"}
    save(output / "manifest.json", metadata)
    results = []
    for case in cases:
        result = await evaluate(case, output / case["id"], settings)
        results.append(result)
        print(json.dumps({"id": case["id"], "passed": result["passed"], "checks": result["checks"],
                          "error_type": result["error_type"]}, ensure_ascii=False), flush=True)
    summary = {**metadata, "passed": sum(row["passed"] for row in results), "total": len(results), "output": str(output)}
    save(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["passed"] == len(results) else 1


def main() -> int:
    """解析唯一自评套件参数。参数：命令行；返回：退出码；异常：参数或执行错误。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["self-check"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", help="仅重试指定失败样例ID，保留此前全部原件")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
