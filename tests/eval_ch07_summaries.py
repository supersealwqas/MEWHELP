"""CH07摘要冻结样例真实评估：只读源样例、不写业务库，结构检查后必须逐条复核开放语义。"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import monotonic

import httpx

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    """写UTF-8评估证据。参数：目标/value可序列化内容；返回：None；异常：文件或JSON错误。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def evaluate(args) -> int:
    """调用真实摘要生成器，仅向模型发送source/background。参数：CLI；返回：0结构通过/2失败；异常：配置和文件错误。"""
    from app.evaluation.cli import _load_environment
    _load_environment(args.env_file)
    from app.config import settings
    from app.core.llm import get_chat_model
    from app.core.logging import setup_logging
    from app.schemas.history import StoredMessage
    from app.workflow.summarization import SummaryGenerator
    setup_logging()
    path = ROOT / "tests/fixtures/ch07/summary_cases.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    args.report_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.report_dir / "frozen-cases.json", frozen)
    sources = [ROOT / "app/workflow/summarization.py", Path(__file__)]
    report = {"started_at":datetime.now(timezone.utc).isoformat(), "real_model":True, "model":settings.chat_model,
        "case_sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "source_sha256":{
            item.relative_to(ROOT).as_posix():hashlib.sha256(item.read_bytes()).hexdigest() for item in sources},
        "semantic_review":"pending：必须对照标注逐条人工复核，结构通过不是语义通过", "results":[]}
    with httpx.Client() as sync_client:
        async with httpx.AsyncClient() as async_client:
            def factory(**options):
                """复用有关闭边界的真实HTTP客户端。参数：模型选项；返回：模型；异常：SDK配置错误。"""
                return get_chat_model(http_client=sync_client, http_async_client=async_client, **options)

            generator = SummaryGenerator(settings, model_factory=factory)
            for case in frozen["cases"]:
                started, result = monotonic(), {"id":case["id"], "structural_failures":[]}
                try:
                    async with asyncio.timeout(90):
                        draft = await generator.generate([StoredMessage.model_validate(row) for row in case["source"]], case["background"])
                    result["draft"] = draft.model_dump()
                    if draft.no_facts != case["no_facts"]:
                        result["structural_failures"].append("no_facts")
                    if any(identifier not in draft.content for identifier in case["identifiers"]):
                        result["structural_failures"].append("missing_identifier")
                except Exception as error:
                    result["error_type"] = type(error).__name__
                    result["structural_failures"].append("generation_failed")
                result["elapsed_seconds"] = round(monotonic()-started, 3)
                report["results"].append(result)
                write_json(args.report_dir / "summary.json", report)
                print(json.dumps({key:result[key] for key in ("id", "structural_failures", "elapsed_seconds")}), flush=True)
    report.update(ended_at=datetime.now(timezone.utc).isoformat(),
                  structural_passed=all(not row["structural_failures"] for row in report["results"]))
    write_json(args.report_dir / "summary.json", report)
    return 0 if report["structural_passed"] else 2


def main(argv=None) -> int:
    """解析显式环境与独占报告目录。参数：argv可选参数；返回：退出码；异常：参数错误由CLI显示。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    try:
        return asyncio.run(evaluate(parser.parse_args(argv)))
    except Exception as error:
        print(json.dumps({"error_type":type(error).__name__}), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
