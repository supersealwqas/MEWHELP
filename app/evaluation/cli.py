"""CH04题集验证、专用资源准备、校准与固定集比较命令行入口。题量由数据集决定。

CLI先解析显式环境文件再加载应用依赖，--help无需模型或数据库。
每次准备/运行使用新目录，失败留下状态文件并返回非零码。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def build_parser() -> argparse.ArgumentParser:
    """建立可独立查看的CLI参数。参数：无；返回：ArgumentParser；异常：无。"""
    parser = argparse.ArgumentParser(description="CH04 四策略独立评估（仅mewhelp_ch04_eval）")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "calibrate", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--env-file", type=Path, help="本进程显式读取的环境文件；不修改文件或用户服务")
        command.add_argument("--dataset", type=Path, default=Path("data/evaluation/ch04.json"))
        command.add_argument("--manifest", type=Path, default=Path("data/kb/manifest.json"))
        if name != "validate":
            command.add_argument("--output-dir", type=Path, help="必须尚不存在的本次运行目录；默认reports/ch04/<时间>-<随机ID>")
            command.add_argument("--embedding-model-path", type=Path, help="显式本地BGE-M3目录")
            command.add_argument("--reranker-model-path", type=Path, help="显式本地reranker目录")
            command.add_argument("--mysql-port", type=int, help="显式独占MySQL端口，覆盖环境文件")
            command.add_argument("--milvus-uri", help="显式独占Milvus地址，覆盖环境文件")
        if name == "prepare":
            command.add_argument("--target", required=True, choices=["eval"], help="具名确认仅准备独立评估库")
        if name in {"calibrate", "compare"}:
            split = "calibration" if name == "calibrate" else "evaluation"
            command.add_argument("--split", choices=[split], default=split)
            command.add_argument("--strategies", nargs="+", choices=["dense", "bm25", "hybrid", "hybrid_rerank"],
                                 default=["dense", "bm25", "hybrid", "hybrid_rerank"])
            command.add_argument("--evidence-budget", type=int, default=8000)
        if name == "compare":
            command.add_argument("--policy", type=Path, required=True, help="真实校准运行产生的policy.json（校准题数须与当前数据集一致）")
    return parser


def _load_environment(path: Path | None) -> None:
    """在导入应用配置前读取显式环境。参数：path 可选.env；返回：None；异常：OSError。"""
    if path is not None:
        from dotenv import dotenv_values
        if not path.is_file():
            raise FileNotFoundError("显式环境文件不存在")
        for key, value in dotenv_values(path).items():
            if value is not None:
                os.environ[key] = value


def _progress(row: dict) -> None:
    """输出无正文或凭据的逐题状态。参数：row 记录；返回：None；异常：终端写入失败。"""
    print(json.dumps({"id": row["id"], "strategy": row["strategy"], "error": row["error"],
                      "seconds": round(row["elapsed_seconds"], 3)}, ensure_ascii=False), flush=True)


async def run(args: argparse.Namespace) -> int:
    """执行一条已解析命令，并将准备/运行失败落到独占目录。

    参数：args build_parser的结果。
    返回：int，成功0；逐题或启动失败1。
    异常：取消直接传播，保留已写记录；文件系统报告错误可能传播。
    """
    _load_environment(args.env_file)
    from app.evaluation.dataset import load_dataset, resolve_evidence
    from app.evaluation.policy import POLICY_VERSION, build_corpus, calibrate_threshold, digest_json, validate_policy

    if args.command == "validate":
        chunks = build_corpus(args.manifest, embedding_revision="offline-validation-only")
        # 题量由数据集决定：validate 不限定题数，只校验每桶两种 split 都有题。
        cases = load_dataset(args.dataset, chunks=chunks, expected_size=None)
        for case in cases:
            resolve_evidence(case, chunks)
        print(json.dumps({"valid": True, "case_count": len(cases), "chunk_count": len(chunks),
                          "splits": dict(Counter(case.split for case in cases)),
                          "buckets": dict(Counter(case.bucket for case in cases)),
                          "dataset_digest": digest_json([case.model_dump(mode="json") for case in cases]),
                          "validation": "原文字符切分与证据定位；真实token硬限在prepare检查"}, ensure_ascii=False, indent=2))
        return 0
    # 离线validate只需要题集和原文；下面的真实运行依赖才允许加载聊天/数据库配置。
    from app.evaluation.runner import atomic_json, evaluate_dataset
    from app.evaluation.runtime import eval_config, evaluation_runtime, prepare_resources
    from app.config import Settings

    output = args.output_dir or Path("reports/ch04") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.command + "-" + uuid4().hex[:8])
    if output.exists():
        raise FileExistsError("本次运行目录已存在，禁止覆盖已有评估")
    config = eval_config(Settings(), target="eval")
    overrides = {}
    if args.embedding_model_path:
        overrides["knowledge_model_path"] = str(args.embedding_model_path.resolve())
    if args.reranker_model_path:
        overrides["rerank_model_path"] = str(args.reranker_model_path.resolve())
    if args.mysql_port is not None:
        if not 1 <= args.mysql_port <= 65535:
            raise ValueError("MySQL端口必须在1–65535")
        overrides["mysql_port"] = args.mysql_port
    if args.milvus_uri:
        overrides["milvus_uri"] = args.milvus_uri
    config = config.model_copy(update=overrides)
    owns_output = False
    try:
        if args.command == "prepare":
            output.mkdir(parents=True, exist_ok=False)
            owns_output = True
            atomic_json(output / "manifest.json", {"status": "preparing", "target": "eval"})
            result = await prepare_resources(config, args.manifest)
            atomic_json(output / "manifest.json", result)
            print(json.dumps({**result, "output_dir": str(output.resolve())}, ensure_ascii=False, indent=2))
            return 0
        async with evaluation_runtime(config, manifest_path=args.manifest, evidence_budget=args.evidence_budget) as runtime:
            cases = load_dataset(args.dataset, chunks=runtime.chunks, expected_size=None)
            runtime.metadata["dataset_digest"] = digest_json([case.model_dump(mode="json") for case in cases])
            selected = [case for case in cases if case.split == args.split]
            # 题量由数据集决定，不写死 24/48；只要求本次要跑的 split 有题。
            if not selected:
                raise ValueError(f"数据集没有 split={args.split} 的题目")
            thresholds = None
            if args.command == "compare":
                policy = json.loads(args.policy.read_text(encoding="utf-8"))
                # 校准题数取自数据集本身，保证 policy 与当前题集是同一次校准的产物。
                calibration_count = sum(1 for case in cases if case.split == "calibration")
                thresholds = validate_policy(policy, runtime.metadata, args.strategies,
                                             calibration_case_count=calibration_count)
                runtime.metadata["policy_digest"] = digest_json(policy)
            result = await evaluate_dataset(selected, runtime, output, strategies=args.strategies,
                                              thresholds=thresholds, retrieval_only=args.command == "calibrate", progress=_progress)
            owns_output = True
            if args.command == "calibrate":
                rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
                policy = {"version": POLICY_VERSION, "calibration_case_count": len(selected), "metadata": runtime.metadata,
                          "thresholds": {strategy: calibrate_threshold([row for row in rows if row["strategy"] == strategy], strategy=strategy)
                                         for strategy in args.strategies}}
                atomic_json(output / "policy.json", policy)
            print(json.dumps({"status": result["status"], "output_dir": str(output.resolve())}, ensure_ascii=False))
            return int(result["status"] != "completed")
    except Exception as error:
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            owns_output = True
        # 只写本调用确实创建的目录；运行器内部失败已保留自己的interrupted报告。
        if owns_output:
            failure = {"status": "failed", "stage": args.command, "error_type": type(error).__name__}
            atomic_json(output / "startup-error.json", failure)
            if args.command == "prepare":
                atomic_json(output / "manifest.json", failure)
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "output_dir": str(output.resolve())}, ensure_ascii=False))
        return 1


def main() -> None:
    """可通过python -m调用。参数：无；返回：None；异常：SystemExit携带真实退出码。"""
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
