"""知识库独立命令行入口。

子命令固定为 init、import-docs、sync、mine-once、mine-schedule、status；
--test 同时切换 MySQL 与 Milvus 专用测试库，不接受半边测试半边应用库。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.knowledge.runtime import (
    initialize_knowledge,
    knowledge_runtime,
    read_knowledge_status,
)
from app.knowledge.scheduler import run_schedule


def _config(test: bool) -> Settings:
    """按 --test 切换两类测试库。参数：是否测试；返回：配置；异常：无。"""
    config = Settings()
    if test:
        if (
            config.knowledge_test_database != "mewhelp_ch03_test"
            or config.milvus_test_database != "mewhelp_ch03_test"
        ):
            raise SystemExit("CH03 测试库必须是 mewhelp_ch03_test")
        config = config.model_copy(
            update={
                "mysql_database": config.knowledge_test_database,
                "milvus_database": config.milvus_test_database,
            }
        )
    return config


def _add_command(subparsers, name: str, help_text: str):
    """添加带 --test 的子命令。参数：解析器/名称/帮助；返回：子解析器；异常：无。"""
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--test", action="store_true", help="只使用 CH03 专用测试库")
    return parser


async def _run(args: argparse.Namespace) -> None:
    """执行一次 CLI 命令。参数：解析结果；返回：None；异常：任务/配置错误。"""
    config = _config(args.test)
    if args.command == "init":
        await initialize_knowledge(config)
        print("knowledge init completed")
        return
    if args.command == "status":
        print(json.dumps(await read_knowledge_status(config), ensure_ascii=False, indent=2))
        return
    async with knowledge_runtime(config) as jobs:
        if args.command == "import-docs":
            report = await jobs.import_documents(Path(args.manifest))
            print(report.model_dump_json(indent=2))
        elif args.command == "sync":
            print((await jobs.sync_pending()).model_dump_json(indent=2))
        elif args.command == "mine-once":
            print((await jobs.mine_once()).model_dump_json(indent=2))
        elif args.command == "mine-schedule":
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for signal_name in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(signal_name, stop.set)
                except NotImplementedError:
                    pass
            await run_schedule(
                jobs.mine_once,
                jobs.recover_pending,
                clock=lambda: datetime.now().astimezone(),
                sleep=asyncio.sleep,
                stop=stop,
            )
            print("knowledge schedule stopped")


def main() -> None:
    """命令行入口。参数：无；返回：None；异常：参数/任务错误。"""
    parser = argparse.ArgumentParser(description="CH03 知识库运维命令")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_command(subparsers, "init", "创建缺失的知识表与 Milvus 集合")
    import_docs = _add_command(subparsers, "import-docs", "导入 data/kb 清单")
    import_docs.add_argument("--manifest", default="data/kb/manifest.json")
    _add_command(subparsers, "sync", "补齐 pending 向量并清理失效向量")
    _add_command(subparsers, "mine-once", "恢复 pending 并执行一次挖掘")
    _add_command(subparsers, "mine-schedule", "启动北京时间 02:00 定时挖掘")
    _add_command(subparsers, "status", "只读输出知识库状态")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
