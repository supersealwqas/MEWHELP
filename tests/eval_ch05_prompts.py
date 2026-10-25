"""CH05 冻结标注的真实模型评估入口；只调用分类模型，不连接数据库或执行工具。"""

import argparse
import asyncio
from collections import Counter
from contextlib import aclosing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx


class RecordingModel:
    """透明记录真实分类流的文本，标注与断言不进入模型上下文。"""

    def __init__(self, model, outputs: list[str]) -> None:
        """参数：model 真实客户端、outputs 本例原始输出容器；返回：None；异常：无。"""
        self.model, self.outputs = model, outputs

    async def astream(self, messages):
        """转发真实流。参数：messages 无标注输入；返回：模型原分片流；异常：模型错误及取消原样传播。"""
        async with aclosing(self.model.astream(messages)) as stream:
            async for chunk in stream:
                if isinstance(chunk.content, str):
                    self.outputs.append(chunk.content)
                yield chunk


def load_cases(path: Path) -> tuple[list[dict], str]:
    """验证冻结集形状。参数：path 28例 JSON；返回：例集与 SHA256；异常：ValueError，规模/字段不合法。"""
    raw = path.read_bytes()
    cases = json.loads(raw)
    counts = Counter(case["expected_intent"] for case in cases)
    if len(cases) != 28 or len(counts) != 7 or set(counts.values()) != {4}:
        raise ValueError("评估集必须包含七类各四例")
    if len({case["id"] for case in cases}) != 28:
        raise ValueError("评估 ID 不能重复")
    for case in cases:
        if set(case) != {"id", "history", "query", "expected_intent", "mandatory"}:
            raise ValueError("评估字段不完整")
        if not isinstance(case["history"], list) or not isinstance(case["query"], str) or type(case["mandatory"]) is not bool:
            raise ValueError("评估字段类型错误")
    return cases, hashlib.sha256(raw).hexdigest()


async def evaluate(args: argparse.Namespace) -> int:
    """逐例实际请求并写完整证据。参数：args 命令参数；返回：0通过/2未达标；异常：环境/文件错误。"""
    if args.env_file:
        from app.evaluation.cli import _load_environment
        _load_environment(args.env_file)
    # 必须先加载显式环境，再导入会读取 settings 的模型模块。
    from app.config import settings
    from app.core.llm import get_chat_model
    from app.core.model_budget import BudgetLedger
    from app.workflow.model import WorkflowModelAdapter
    from app.workflow.prompts import INTENT_PROMPT

    cases, fingerprint = load_cases(args.cases)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    results = []
    sync_http, async_http = httpx.Client(), httpx.AsyncClient()
    try:
        for case in cases:
            outputs = []
            def factory(**options):
                """创建本例真实客户端。参数：options 适配器配置；返回：透明记录模型；异常：模型配置错误。"""
                model = get_chat_model(http_client=sync_http, http_async_client=async_http, **options)
                return RecordingModel(model, outputs)

            ledger = BudgetLedger()
            adapter = WorkflowModelAdapter(model_factory=factory)
            started, predicted, error_type = time.monotonic(), None, None
            try:
                result = await adapter.classify([*case["history"], {"role":"user", "content":case["query"]}], ledger)
                predicted = result.intent
            except Exception as error:
                # 只记录异常类型，避免上游异常文本泄露端点或凭据。
                error_type = type(error).__name__
            row = {**case, "predicted_intent":predicted, "json_valid":predicted is not None,
                   "correct":predicted == case["expected_intent"], "error_type":error_type,
                   "raw_output":"".join(outputs), "elapsed_seconds":round(time.monotonic() - started, 3),
                   "budget":ledger.snapshot()}
            results.append(row)
            with (args.report_dir / "cases.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"id":row["id"], "expected":row["expected_intent"], "predicted":predicted,
                              "correct":row["correct"], "error_type":error_type}, ensure_ascii=False), flush=True)
    finally:
        # 使用官方公开的 HTTP 客户端注入边界统一管理资源，不读取模型内部客户端字段。
        await async_http.aclose()
        sync_http.close()
    per_class = {intent:sum(row["correct"] for row in results if row["expected_intent"] == intent)
                 for intent in sorted({case["expected_intent"] for case in cases})}
    correct = sum(row["correct"] for row in results)
    legal = sum(row["json_valid"] for row in results)
    mandatory_failures = [row["id"] for row in results if row["mandatory"] and not row["correct"]]
    confusion = Counter((row["expected_intent"], row["predicted_intent"] or "INVALID") for row in results)
    summary = {"created_at":datetime.now(timezone.utc).isoformat(), "suite":"intent", "real_model":True,
               "model":settings.chat_model, "case_sha256":fingerprint,
               "prompt_sha256":hashlib.sha256(INTENT_PROMPT.encode()).hexdigest(),
               "json_valid":legal, "correct":correct, "total":len(results), "per_class":per_class,
               "mandatory_failures":mandatory_failures,
               "confusion":[{"expected":key[0],"predicted":key[1],"count":value} for key,value in sorted(confusion.items())],
               "failures":[row for row in results if not row["correct"]],
               "passed":legal == 28 and correct >= 26 and min(per_class.values()) >= 3 and not mandatory_failures}
    (args.report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.report_dir / "prompt.txt").write_text(INTENT_PROMPT, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["passed"] else 2


def main(argv: list[str] | None = None) -> int:
    """解析并执行真实评估。参数：argv 可选命令参数；返回：退出码；异常：参数错误由 argparse 报告。"""
    parser = argparse.ArgumentParser(description="CH05 真实意图评估，不执行工具或数据库写入")
    parser.add_argument("--suite", choices=["intent"], required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, help="只读加载指定环境文件，不修改或复制凭据")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(evaluate(args))
    except Exception as error:
        print(json.dumps({"error_type":type(error).__name__}, ensure_ascii=False), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
