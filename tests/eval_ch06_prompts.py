"""CH06冻结标注真实评估，分别记录原始协议、服务端兜底、语义边界和confidence。"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import time

import httpx

from tests.ch05_live_runtime import RecordingChat, ROOT, write_json


def confidence_report(rows: list[dict]) -> tuple[dict, str]:
    """统计原始有效分类，回退零分单列。参数：逐例结果；返回：统计及中文报告；异常：结构错误。"""
    classified = [row for row in rows if "raw_classification_valid" in row]
    valid = [row for row in classified if row["raw_classification_valid"]]
    wrong = [row for row in valid if not row.get("intent_correct", row["correct"])]
    scores = [row["confidence"] for row in wrong]
    high = [{key: row.get(key) for key in ("id", "expected_intent", "predicted_intent", "confidence")}
            for row in wrong if row["confidence"] >= .9]
    data = {"valid_correct":len(valid) - len(wrong), "valid_wrong":len(wrong),
        "fallbacks":sum(not row["raw_classification_valid"] and row.get("error_type") is None for row in classified),
        "dependency_errors":sum(row.get("error_type") is not None for row in classified),
        "wrong_min":min(scores) if scores else None, "wrong_median":median(scores) if scores else None,
        "wrong_max":max(scores) if scores else None,
        "wrong_buckets":[sum(low <= score < high_bound or (high_bound == 1 and score == 1) for score in scores)
                         for low, high_bound in ((0,.5),(.5,.8),(.8,.9),(.9,1))],
        "high_confidence_errors":high}
    report = "## 错判样本的 confidence 分布\n\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n\n"
    report += ("本次无错判样本，不能据此认定confidence已校准或设置升级阈值。" if not wrong else
               "应追加这些易混类别的独立评估；本次结果不足以直接启用升级或confidence路由。")
    report += "\n分桶为[0,0.5)、[0.5,0.8)、[0.8,0.9)、[0.9,1]；解析回退零分不进入原始分布。\n"
    return data, report


def load_cases(suite: str) -> tuple[list[dict], str]:
    """读取并核对冻结规模。参数：三类suite；返回：数据与指纹；异常：ValueError，标注不完整。"""
    raw = (ROOT / f"tests/fixtures/ch06/{suite}_cases.json").read_bytes()
    cases = json.loads(raw)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("标注ID重复")
    if suite == "intent" and (len(cases) != 32 or set(Counter(row["expected_intent"] for row in cases).values()) != {4}):
        raise ValueError("单轮要求八类各四条")
    if suite == "dialogue" and (len(cases) != 6 or any(not 4 <= len(row["turns"]) <= 6 for row in cases)):
        raise ValueError("对话要求六组各四至六轮")
    if suite == "understanding" and len(cases) != 12:
        raise ValueError("理解扩写专例要求十二条")
    return cases, hashlib.sha256(raw).hexdigest()


def history_messages(history: list[dict]) -> list[dict]:
    """将标注中的历史事实转换为标准消息，剔除所有判定字段。参数：历史输入；返回：消息；异常：无。"""
    messages = []
    for index, row in enumerate(history):
        content = row["content"]
        if row["role"] == "tool":
            data = dict(content)
            if "product" in data:
                data["product_name"] = data.pop("product")
            call_id = f"fixture-{index}"
            messages.append({"role":"assistant", "content":"", "tool_calls":[{"id":call_id,"name":row["name"],"args":{"order_id":data.get("order_id")}}]})
            messages.append({"role":"tool", "tool_call_id":call_id, "name":row["name"],
                "content":json.dumps({"ok":row.get("trusted") is True,"data":data}, ensure_ascii=False)})
        else:
            messages.append({"role":row["role"], "content":content})
    return messages


def evaluation_messages(history: list[dict], query: str, config, prompt: str, tools: list) -> list[dict]:
    """复用线上整轮组装和预算裁剪，评估不另造分类消息路径。

    参数：history已剔除标注的标准历史消息；query当前原问或完整问；config本次配置；prompt/tools阶段协议。
    返回：与根图同规则的SDK消息；异常：历史校验错误或当前输入超预算。
    """
    from app.workflow.agent import model_messages
    turns = []
    for message in history:
        if message["role"] == "user" or not turns:
            turns.append({"turn_id": f"evaluation-{len(turns)}", "turn_status": "completed", "messages": []})
        turns[-1]["messages"].append(message)
    return model_messages({"history_turns": turns, "messages": [{"role": "user", "content": query}]},
                          config, prompt, tools)


def inspect_understanding(query: str, expected: dict, result: dict) -> list[str]:
    """核对显式语义边界；不将词表检查冒充全部语义审查。参数：原问/标注/预测；返回：失败项；异常：无。"""
    failures = []
    for key in ("order_id", "needs_resolution"):
        if key in expected and result.get(key) != expected[key]:
            failures.append(key)
    if expected.get("preserve_raw") and result.get("resolved_query") != query:
        failures.append("preserve_raw")
    resolved = result.get("resolved_query", "")
    queries = resolved + "\n" + result.get("standard_query", "")
    if any(value not in resolved for value in expected.get("resolved_contains", [])):
        failures.append("resolved_contains")
    alternatives = list(expected.get("resolved_any", []))
    if set(alternatives) & {"物流", "运输", "配送"}:
        alternatives.extend(["快递", "包裹", "送到", "送达"])
    if alternatives and not any(value in resolved for value in alternatives):
        failures.append("resolved_business")
    if any(value in queries for value in expected.get("forbidden_in_queries", []) + expected.get("forbidden_order_ids", [])):
        failures.append("forbidden_in_queries")
    if any(value not in queries for value in expected.get("required_in_queries", [])):
        failures.append("required_in_queries")
    return failures


def raw_result(records: list[dict], stage: str):
    """独立解析模型聚合原片，绝不把服务端回退当原始合法。参数：记录/阶段；返回：原始字典或None；异常：无。"""
    from langchain_core.messages import AIMessageChunk
    from app.core.structured import parse_tool_output
    from app.schemas.workflow import IntentResult, UnderstandingResult
    from app.knowledge.query_rewrite import ExpandedQueries
    schemas = {"understand":(UnderstandingResult,"submit_understanding"), "classify":(IntentResult,"submit_intent"),
               "expand":(ExpandedQueries,"submit_expanded_queries")}
    selected = [row for row in records if row["stage"] == stage]
    try:
        schema, name = schemas[stage]
        return parse_tool_output(AIMessageChunk.model_validate(selected[-1]["output"]), schema, name).model_dump(mode="json")
    except (ValueError, TypeError, IndexError):
        return None


async def evaluate_case(case: dict, history: list[dict], factory) -> dict:
    """使用真实适配器运行一个标注；expected始终留在判定侧。参数：标注/输入历史/模型工厂；返回：证据；异常：文件错误。"""
    from app.core.model_budget import BudgetLedger
    from app.config import settings
    from app.workflow.model import WorkflowModelAdapter, understanding_tool_schema, intent_tool_schema
    from app.workflow.prompts import UNDERSTANDING_PROMPT, INTENT_PROMPT
    from app.knowledge.query_rewrite import QueryExpander
    ledger = BudgetLedger(max_model_calls=settings.workflow_max_model_calls,
        total_token_budget=settings.workflow_total_token_budget, max_decisions=settings.workflow_max_decisions,
        request_token_budget=settings.request_token_budget)
    records = []

    def record_factory(**options):
        """为本例绑定透明记录。参数：模型选项；返回：包装；异常：模型配置错误。"""
        return RecordingChat(factory(**options), records, "answer", turn_id=case["id"])

    expected = case.get("expected", {"intent":case.get("expected_intent")})
    row = {"id":case["id"], "query":case["query"], "expected":expected, "mandatory":case.get("mandatory", True),
        "expected_intent":expected.get("intent"), "failures":[], "error_type":None}
    started = time.monotonic()
    try:
        if case.get("kind") == "expand":
            expanded = await QueryExpander(model_factory=record_factory).expand(case["query"], simple_query=case["simple_query"], budget=ledger)
            queries = expanded.queries
            row.update(queries=queries, raw_expansion=raw_result(records, "expand"))
            if len(queries) != expected["query_count"] or len(set(queries)) != len(queries):
                row["failures"].append("query_count_or_duplicates")
            if expected.get("single_query_equals_simple_query") and queries != [case["simple_query"]]:
                row["failures"].append("single_query")
            if set(re.findall(r"\d+", " ".join(queries))) - set(expected.get("allowed_numeric_tokens", [])):
                row["failures"].append("invented_numbers")
        else:
            adapter = WorkflowModelAdapter(model_factory=record_factory)
            # 与根图一致：理解读取原问，分类读取完整问，两阶段均独立组装和裁剪历史。
            understood = None
            if "expected" in case:
                understood = await adapter.understand(evaluation_messages(history, case["query"], settings,
                    UNDERSTANDING_PROMPT, [understanding_tool_schema()]), ledger)
                row["understanding"] = understood.model_dump(mode="json")
                row["raw_understanding"] = raw_result(records, "understand")
                row["failures"].extend(inspect_understanding(case["query"], expected, row["understanding"]))
            intent = await adapter.classify(evaluation_messages(history,
                understood.resolved_query if understood else case["query"], settings, INTENT_PROMPT, [intent_tool_schema()]), ledger)
            original = raw_result(records, "classify")
            row.update(predicted_intent=intent.intent, confidence=intent.confidence,
                raw_classification=original, raw_classification_valid=original is not None, intent_correct=intent.intent == expected["intent"])
            if not row["intent_correct"]:
                row["failures"].append("intent")
            if original is None:
                row["failures"].append("raw_classification_invalid")
        # 所有mandatory优化例要求实际正确；来源/结构回退保留，不能伪造模型通过。
        if any(record.get("failure") for record in ledger.snapshot()["records"]):
            row["failures"].append("server_fallback")
    except Exception as error:
        row.update(error_type=type(error).__name__)
        row["failures"].append("dependency_or_runtime_error")
    row.update(budget=ledger.snapshot(), models=records, elapsed_seconds=round(time.monotonic()-started, 3), correct=not row["failures"])
    return row


async def evaluate(args) -> int:
    """显式加载环境、运行真实Lite并写证据。参数：CLI；返回：0通过、2失败、3依赖错误；异常：配置/文件错误。"""
    from app.evaluation.cli import _load_environment
    _load_environment(args.env_file)
    from app.config import settings
    from app.core.llm import get_chat_model
    if settings.chat_model != "gemini-3.5-flash-lite":
        raise ValueError("CH06评估固定gemini-3.5-flash-lite")
    cases, fingerprint = load_cases(args.suite)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT / name for name in ("app/workflow/prompts.py", "app/workflow/routing.py", "app/knowledge/query_rewrite.py",
        "app/workflow/agent.py", "app/workflow/graph.py", "app/workflow/model.py", "app/core/memory.py",
        "app/schemas/history.py", "tests/eval_ch06_prompts.py")]
    summary = {"suite":args.suite,"real_model":True,"model":settings.chat_model,"case_sha256":fingerprint,
        "started_at":datetime.now(timezone.utc).isoformat(), "source_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    write_json(args.report_dir / "frozen-cases.json", cases)
    write_json(args.report_dir / "summary.json", summary)
    for path in source_files:
        (args.report_dir / path.name).write_bytes(path.read_bytes())
    results = []
    with httpx.Client() as sync_http:
        async with httpx.AsyncClient() as async_http:
            def factory(**options):
                """共用受管理HTTP客户端。参数：模型选项；返回：真实客户端；异常：配置错误。"""
                return get_chat_model(http_client=sync_http, http_async_client=async_http, **options)

            for case in cases:
                history = history_messages(case.get("history", []))
                for item in case.get("turns", [case]):
                    row = await evaluate_case(item, history, factory)
                    results.append(row)
                    with (args.report_dir / "cases.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    # 后续历史仅取实际输入/预测；绝不将expected或标准答案回灌给模型。
                    history.append({"role":"user", "content":item["query"]})
                    if row.get("understanding"):
                        history.append({"role":"assistant", "content":"当前询问：" + row["understanding"]["resolved_query"]})
                    if item.get("selection"):
                        # selection是模拟用户的明确选单输入，区别于expected；用公开用户事实恢复后续指代。
                        history.append({"role":"user", "content":"我选择订单" + item["selection"]["order_id"]})
                        history.append({"role":"assistant", "content":"已记录您选择的订单，等待下一个问题。"})
                    print(json.dumps({key:row.get(key) for key in ("id","predicted_intent","correct","failures","error_type")}, ensure_ascii=False), flush=True)
    confidence, report = confidence_report(results)
    regular = [row for row in results if row["expected_intent"] != "其他"]
    correct = sum(row["correct"] for row in results)
    passed = not any(not row["correct"] for row in results if row["mandatory"] or row["expected_intent"] == "其他")
    passed = passed and (not regular or sum(row["correct"] for row in regular) / len(regular) >= .95)
    summary.update(total=len(results),correct=correct,passed=passed,confidence=confidence,
        dependency_errors=[row["id"] for row in results if row["error_type"]],
        failures=[{"id":row["id"],"failures":row["failures"]} for row in results if not row["correct"]],
        ended_at=datetime.now(timezone.utc).isoformat(),
        semantic_review="自动检查只覆盖显式约束；开放语义需结合cases.jsonl中的完整原始输出人工复核")
    write_json(args.report_dir / "summary.json", summary)
    (args.report_dir / "report.md").write_text(f"# CH06 {args.suite}真实评估\n\n{correct}/{len(results)}，通过：{passed}。模型：{settings.chat_model}。\n\n" + report, encoding="utf-8")
    return 3 if summary["dependency_errors"] else (0 if passed else 2)


def main(argv=None) -> int:
    """解析明确的评估入口。参数：可选命令参数；返回：退出码；异常：参数错误由CLI显示。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=["intent","dialogue","understanding"])
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return asyncio.run(evaluate(args))
    except Exception as error:
        print(json.dumps({"error_type":type(error).__name__}), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
