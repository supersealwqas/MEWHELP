"""CH10数据准备、离线全参数训练和独立主题推理CLI。"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.topics.data import augment_training, clean_question, split_examples, write_dataset
from app.topics.labelling import MAX_LABEL_TEXT_LENGTH, TopicLabeller
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


def _pool_metadata(original: dict) -> dict:
    """保留原池来源与分组，供预标和隔离记录共用。

    参数：original为export-pool派生行，须含来源、分组、池ID、原文和历史不确定标记。
    返回：dict，包含来源证据、完整脱敏原文及原池审核状态。
    异常：必需字段缺失抛KeyError；原文清洗失败抛ValueError。
    """
    return dict(source=original["source"], group_id=original["group_id"], pool_ids=[original["question_id"]],
                provenance=original.get("provenance", {}), history_uncertain=original["history_uncertain"],
                original_text=clean_question(original["text"]), pool_review_status=original.get("review_status"))


async def prepare_dataset(source: Path, output_dir: Path, *, per_label: int = 60, multi: int = 204,
                          seed: int = 42, max_calls: int = 60, labeller=None) -> dict:
    """复用成功预标/生成批次并冻结可核对的数据制品。

    参数：source为export-pool输出的pool.json数组文件；output_dir为尚不存在的制品目录；
    per_label为每类模拟问题数量、multi为多诉求模拟数量，均为0至500；seed为确定性分组整数种子；
    max_calls为1至60的逻辑请求上限；labeller为可选label/generate接口实例，None时按需构建TopicLabeller。
    返回：dict制品清单，包含切分/标签/来源计数、摘要、预标状态、超长隔离数与待复核记录路径；
    超过外发预算的脱敏原问完整留待复核，不进入模型请求或三份数据；全部超长时仍写出抽审包。
    异常：目标已存在抛FileExistsError；输入、上限或恢复身份不符抛ValueError；模型网络、输出校验和IO错误传播。
    """
    source, output_dir = Path(source), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("制品目录已经存在，请生成新目录")
    if not 0 <= per_label <= 500 or not 0 <= multi <= 500 or not 1 <= max_calls <= 60:
        raise ValueError("数量超出实验上限或请求上限无效")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("source须为export-pool生成的pool.json数组")
    eligible, pending = [], []
    for source_index, original in enumerate(data):
        text = clean_question(original["text"])
        if len(text) <= MAX_LABEL_TEXT_LENGTH:
            eligible.append(original)
            continue
        # 外发预算不等于源数据准入限制：保留完整脱敏原话及来源，不截断或冒称模型已预标。
        pending.append(dict(id=f"source:{source_index}:overlong", text=text, labels=[], parent_id=None,
                            annotation_status="pending_review", needs_review=True,
                            review_reason=f"脱敏原问超过{MAX_LABEL_TEXT_LENGTH}字符外发上限，未发送模型，请复核完整原问",
                            **_pool_metadata(original)))
    overlong_count = len(pending)
    # 请求计划、批内连续序号及缓存回配统一使用同一筛选列表；改变批次的旧缓存会被config拒绝。
    data = eligible
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
    pending_path = cache / "pending.json"
    # 先保留本地隔离结果，即使之后模型中断也能查看长题；重跑按冻结输入重建，不累加重复行。
    pending_path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    model = labeller
    rows = []
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
                row.update(_pool_metadata(original))
            else:
                row["source"] = "synthetic"
            if not row["labels"]:
                pending.append(row)
            else:
                rows.append(row)
    parts = split_examples(rows, taxonomy, seed)
    reserved = {text for split in ("validation", "test") for row in parts[split]
                for text in (row["text"], row.get("original_text", row["text"]))}
    parts["train"] = augment_training(parts["train"], taxonomy, reserved_texts=reserved)
    pending_path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = write_dataset(output_dir, parts, taxonomy, seed=seed, metadata={**config, "logical_requests": len(plan),
                "raw_records": str(cache / "raw"), "pending_count": len(pending), "overlong_count": overlong_count,
                "pending_review_records": str(pending_path),
                "customer_generalization": "尚无可证实客户语料支持", "token_usage": "见逐请求原件；未返回用量保持null"}, pending_rows=pending)
    return manifest


def main() -> None:
    """执行数据或显式本地模型CLI。参数：命令行；返回：无；异常：输出错误类型并非零退出。"""
    parser = argparse.ArgumentParser(description="CH10 主题数据、离线全参数训练和推理")
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
    train = commands.add_parser("train")
    train.add_argument("--dataset-dir", type=Path, required=True)
    train.add_argument("--base-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--micro-batch", type=int, default=2)
    train.add_argument("--accumulate", type=int, default=8)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict")
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--text", action="append", required=True)
    args = parser.parse_args()
    try:
        if args.command == "export-pool":
            result = asyncio.run(export_pool(args.output_dir))
        elif args.command == "prepare":
            result = asyncio.run(prepare_dataset(args.source, args.output_dir, per_label=args.per_label, multi=args.multi, seed=args.seed, max_calls=args.max_calls))
        elif args.command == "train":
            from app.topics.training import train_model
            result = train_model(args.dataset_dir, args.base_dir, args.output_dir, epochs=args.epochs,
                                 micro_batch=args.micro_batch, accumulate=args.accumulate, max_length=args.max_length, seed=args.seed)
        else:
            from app.topics.inference import TopicPredictor
            predictor = TopicPredictor(args.model_dir)
            metadata = predictor.metadata
            result = dict(model_version=metadata["version"], taxonomy_digest=metadata["taxonomy_digest"], results=predictor.predict(args.text))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        # CLI不打印可能包含凭据的网络异常，只提示具体异常类别；原件保留恢复目录。
        parser.exit(1, f"主题任务失败：{type(error).__name__}；请查看本地制品或恢复目录。\n")


if __name__ == "__main__":
    main()
