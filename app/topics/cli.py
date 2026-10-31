"""CH10 只读导出和可恢复语料准备CLI，输出模型预标及待人工复核材料。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.topics.data import augment_training, clean_question, split_examples, write_dataset
from app.topics.labelling import TopicLabeller
from app.topics.pool import read_pool
from app.topics.taxonomy import label_names, load_taxonomy, taxonomy_digest


async def export_pool(output_dir: Path) -> dict:
    """只读导出全池派生视图。参数：output_dir不存在目录；返回：数量摘要；异常：数据库/IO错误。"""
    from app.db.session import create_database
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("导出目录已存在")
    engine, factory = create_database()
    try:
        rows = await read_pool(factory)
    finally:
        await engine.dispose()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "pool.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(total=len(rows), history_uncertain=sum(row["history_uncertain"] for row in rows),
                   provenance_limit="应用演示/未核实历史来源，不等于真实客户语料；未覆盖业务原话")
    (output_dir / "export.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def prepare_dataset(source: Path, output_dir: Path, *, per_label: int = 60, multi: int = 204,
                          seed: int = 42, max_calls: int = 60, labeller=None) -> dict:
    """恢复预标/生成批次并冻结制品。参数：源JSON、输出目录及数量/种子/请求上限；返回：清单；异常：模型/IO/校验错误。"""
    source, output_dir = Path(source), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("制品目录已经存在，请生成新目录")
    if not 0 <= per_label <= 500 or not 0 <= multi <= 500 or not 1 <= max_calls <= 60:
        raise ValueError("数量超出实验上限或请求上限无效")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("source须为export-pool生成的pool.json数组")
    taxonomy = load_taxonomy()
    plan = []
    for offset in range(0, len(data), 32):
        plan.append(dict(kind="label", offset=offset, count=min(32, len(data) - offset)))
    for name in label_names(taxonomy):
        for offset in range(0, per_label, 60):
            plan.append(dict(kind="generate", labels=[name], count=min(60, per_label - offset), tag=f"single:{name}:{offset}"))
    # 明确四组近邻与多诉求训练场景，不将原池留出原话作为改写种子。
    combos = [["退换货", "尺码"], ["退换货", "质量问题"], ["物流", "运费"], ["发票", "支付"]]
    for offset in range(0, multi, 60):
        plan.append(dict(kind="generate", labels=combos[(offset // 60) % len(combos)], count=min(60, multi - offset), tag=f"multi:{offset}"))
    if len(plan) > max_calls:
        raise ValueError(f"计划需要{len(plan)}个逻辑请求，超过max-calls")
    config = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), taxonomy_sha256=taxonomy_digest(taxonomy),
                  seed=seed, per_label=per_label, multi=multi, max_calls=max_calls, plan=plan)
    # 缓存放在制品旁，失败后目录尚未创建；同参数重跑复用成功批次。
    cache = output_dir.with_name(output_dir.name + ".resume")
    cache.mkdir(parents=True, exist_ok=True)
    frozen = cache / "config.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != config:
        raise ValueError("恢复目录参数或输入摘要不同，请指定新制品目录")
    frozen.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    model = labeller
    rows, pending = [], []
    for batch, step in enumerate(plan):
        checkpoint = cache / f"batch-{batch:03d}.json"
        if checkpoint.exists():
            outputs = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            if model is None:
                model = TopicLabeller(report_dir=cache / "raw")
            if step["kind"] == "label":
                inputs = data[step["offset"]:step["offset"] + step["count"]]
                outputs = await model.label([clean_question(item["text"]) for item in inputs])
            else:
                outputs = await model.generate(step["labels"], step["count"], f"seed:{seed}:{step['tag']}")
            # 成功输出原件独占写入；重试只写未完成批次，不覆盖已成功付费原件。
            with checkpoint.open("x", encoding="utf-8") as stream:
                json.dump(outputs, stream, ensure_ascii=False, indent=2)
        for output in outputs:
            row = {**output, "id": f"batch:{batch}:item:{output['index']}", "text": clean_question(output["text"]),
                   "pool_ids": [], "parent_id": None, "annotation_status": "model_prelabel"}
            if step["kind"] == "label":
                original = data[step["offset"] + output["index"]]
                row.update(source=original["source"], group_id=original["group_id"], pool_ids=[original["question_id"]],
                           provenance=original.get("provenance", {}), history_uncertain=original["history_uncertain"],
                           original_text=clean_question(original["text"]), pool_review_status=original.get("review_status"))
            else:
                row["source"] = "synthetic"
            if not row["labels"]:
                pending.append(row)
            else:
                rows.append(row)
    parts = split_examples(rows, taxonomy, seed)
    parts["train"] = augment_training(parts["train"], taxonomy)
    manifest = write_dataset(output_dir, parts, taxonomy, seed=seed, metadata={**config, "logical_requests": len(plan),
                "raw_records": str(cache / "raw"), "pending_count": len(pending), "pending_review_records": str(cache / "pending.json"),
                "customer_generalization": "尚无可证实客户语料支持", "token_usage": "见逐请求原件；未返回用量保持null"}, pending_rows=pending)
    (cache / "pending.json").write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """执行只读数据CLI。参数：命令行；返回：无；异常：输出错误类型并非零退出。"""
    parser = argparse.ArgumentParser(description="CH10 主题派生语料（模型预标，人工复核待完成）")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-pool")
    export.add_argument("--output-dir", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--per-label", type=int, default=60)
    prepare.add_argument("--multi", type=int, default=204)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-calls", type=int, default=60)
    args = parser.parse_args()
    try:
        if args.command == "export-pool":
            result = asyncio.run(export_pool(args.output_dir))
        else:
            result = asyncio.run(prepare_dataset(args.source, args.output_dir, per_label=args.per_label, multi=args.multi, seed=args.seed, max_calls=args.max_calls))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        # CLI不打印可能包含凭据的网络异常，只提示具体异常类别；原件保留恢复目录。
        parser.exit(1, f"主题数据任务失败：{type(error).__name__}；请查看本地恢复目录原件。\n")


if __name__ == "__main__":
    main()
