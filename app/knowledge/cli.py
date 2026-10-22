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
    prepare_hybrid_index,
    read_knowledge_status,
)
from app.knowledge.scheduler import run_schedule


def _config(test: bool) -> Settings:
    """加载命令行配置，并按 --test 同时切换两类测试库。

    参数:
        test: 为 True 时使用 CH03 专用 MySQL 与 Milvus 测试库，否则保留应用库配置。
    返回值:
        Settings: 已校验的配置；测试模式下两个数据库均为 mewhelp_ch03_test。
    异常:
        pydantic.ValidationError: 必需配置缺失或字段不合法。
        SystemExit: 测试模式下任一测试库名称不是 mewhelp_ch03_test。
    """
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


def _hybrid_config(test: bool, *, config: Settings | None = None) -> Settings:
    """同时选择 CH04 两类专用测试资源，不改旧 --test 的默认语义。

    参数：test 是否测试模式；config 可选已加载配置，None 时从环境加载。
    返回：Settings，显式 hybrid 模式，测试时 MySQL/Milvus 都为 mewhelp_ch04_test。
    异常：SystemExit，任一测试库不匹配；配置校验错误。
    """
    config = config if config is not None else Settings()
    updates = {"knowledge_retrieval_mode": "hybrid"}
    if test:
        if (config.knowledge_rag_test_database != "mewhelp_ch04_test"
                or config.milvus_rag_test_database != "mewhelp_ch04_test"):
            raise SystemExit("CH04 两类测试库必须同时为 mewhelp_ch04_test")
        updates.update(mysql_database=config.knowledge_rag_test_database, milvus_database=config.milvus_rag_test_database)
    return config.model_copy(update=updates)


def _add_command(subparsers, name: str, help_text: str):
    """添加带 --test 的子命令。参数：解析器/名称/帮助；返回：子解析器；异常：无。"""
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--test", action="store_true", help="只使用 CH03 专用测试库；prepare-hybrid 使用 CH04 专用库")
    parser.add_argument("--mode", choices=["dense", "hybrid"], help="显式选择任务索引模式，未指定时使用配置；init 仍只创建旧 CH03 集合")
    return parser


async def _run(args: argparse.Namespace) -> None:
    """执行一次 CLI 命令。参数：解析结果；返回：None；异常：任务/配置错误。"""
    if args.command == "prepare-hybrid":
        config = _hybrid_config(args.test)
        report = await prepare_hybrid_index(config, confirmed_database=args.confirm_milvus_database)
        print(report.model_dump_json(indent=2))
        return
    if args.mode == "hybrid":
        config = _hybrid_config(args.test)
    else:
        config = _config(args.test)
        # 旧 --test 默认保持 CH03 路径；新测试需显式 --mode hybrid，避免误写旧测试库。
        if args.test or args.mode == "dense":
            config = config.model_copy(update={"knowledge_retrieval_mode": "dense"})
    if args.command == "init":
        await initialize_knowledge(config, confirmed_database=args.confirm_milvus_database)
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
    initialize = _add_command(subparsers, "init", "创建缺失的知识表与 Milvus 集合")
    initialize.add_argument(
        "--confirm-milvus-database", metavar="数据库名",
        help="明确确认该既存空 Milvus 库归本项目所有；必须与目标库名完全一致",
    )
    prepare = _add_command(subparsers, "prepare-hybrid", "持写锁准备独立混合集合并同步现有权威原文，保留旧集合")
    prepare.add_argument("--confirm-milvus-database", metavar="数据库名", help="仅对明确归本项目的既存空库具名确认")
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
