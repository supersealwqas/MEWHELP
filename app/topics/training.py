"""全参数多标签BERT训练、验证早停、一次测试评估和离线模型包发布。"""

import json
import math
import os
import random
from collections import Counter
from pathlib import Path

from app.topics.data import validate_dataset
from app.topics.inference import file_sha256, validate_model_package
from app.topics.metrics import compute_metrics, decode_labels, evaluation_plan, select_threshold
from app.topics.taxonomy import label_names, load_taxonomy, taxonomy_digest

OFFICIAL_REPO = "hfl/chinese-roberta-wwm-ext"
OFFICIAL_REVISION = "5c58d0b8ec1d9014354d691c538661bf00bfdb44"
OFFICIAL_WEIGHT_SHA = "1ded5a5a1c7841dee6e47942f7b5bf2bcf6f73ff19197580f852f7f638f86b35"
OFFICIAL_CONFIG_SHA = "61609babfbada201546297d62f7f3a642d8fd2f1cefec5571c6cbd1e2e2132d9"


def _write_json(path: Path, payload: dict) -> None:
    """及时落盘训练状态。参数：path目标、payload报告；返回：无；异常：IO错误。"""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_base(base_dir: Path) -> dict:
    """校验本地固定来源，禁止把任意目录冒称官方模型。参数：base_dir；返回：来源清单；异常：IO/ValueError。"""
    source = json.loads((base_dir / "source-manifest.json").read_text(encoding="utf-8"))
    if not source.get("revision") or not source.get("repo") or not isinstance(source.get("files"), dict):
        raise ValueError("基础模型来源清单缺失")
    files = source["files"]
    if not {"config.json", "tokenizer_config.json", "vocab.txt"} <= files.keys():
        raise ValueError("基础模型清单不完整")
    if not any(name.endswith((".bin", ".safetensors")) for name in files):
        raise ValueError("基础模型权重缺失")
    actual = {path.relative_to(base_dir).as_posix() for path in base_dir.rglob("*") if path.is_file()
              and path.name != "source-manifest.json"}
    if actual != set(files):
        raise ValueError("基础模型包含未登记或缺失文件")
    for name, details in files.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or (base_dir / relative).is_symlink()
                or not (base_dir / relative).resolve().is_relative_to(base_dir.resolve())):
            raise ValueError("基础模型文件路径非法")
        path = base_dir / relative
        if path.stat().st_size != details["bytes"] or file_sha256(path) != details["sha256"]:
            raise ValueError("基础模型文件摘要不匹配")
    if source["repo"] == OFFICIAL_REPO:
        if (source["revision"] != OFFICIAL_REVISION
                or files.get("pytorch_model.bin", {}).get("sha256") != OFFICIAL_WEIGHT_SHA
                or files["config.json"]["sha256"] != OFFICIAL_CONFIG_SHA):
            raise ValueError("官方基础模型必须使用已核验固定revision和权重")
    elif not source["repo"].startswith("local/"):
        raise ValueError("非官方本地实验必须明确使用local来源标识")
    return source


def _inputs(tokenizer, rows: list[dict], taxonomy: dict, device, max_length: int):
    """构造真实分词和浮点多热标签。参数：分词器、样本、主题表、设备、长度；返回：输入及float32标签；异常：分词/张量错误。"""
    import torch
    tokens = tokenizer([row["text"] for row in rows], padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    names = label_names(taxonomy)
    labels = torch.tensor([[float(name in row["labels"]) for name in names] for row in rows],
                          dtype=torch.float32, device=device)
    return tokens, labels


def _train_epoch(model, tokenizer, rows: list[dict], taxonomy: dict, optimizer, scaler, *,
                 device, micro_batch: int, accumulate: int, max_length: int) -> dict:
    """执行一轮全参数训练并按尾组实际样本数归一化梯度。

    参数：model为全部参数参与训练的分类模型；tokenizer为配套分词器；rows为非空规范样本列表；
    taxonomy为权威标签表；optimizer为覆盖全部可训练参数的优化器；scaler为设备对应的梯度缩放器；
    device为torch设备；micro_batch和accumulate为正整数；max_length为至少2的token上限。
    返回：dict，loss为样本加权平均损失，optimizer_steps为本轮更新次数，train_samples为样本数。
    异常：缺失或非有限梯度/损失抛ValueError；模型、分词或设备执行错误原样传播。
    """
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, steps = 0.0, 0
    group_size = micro_batch * accumulate
    for start in range(0, len(rows), group_size):
        group = rows[start:start + group_size]
        for offset in range(0, len(group), micro_batch):
            batch = group[offset:offset + micro_batch]
            tokens, labels = _inputs(tokenizer, batch, taxonomy, device, max_length)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**tokens, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限")
            total_loss += float(loss.detach()) * len(batch)
            # BCE默认按样本均值；用该尾组实际样本数加权，完整和不等大小微批都更新正确。
            scaler.scale(loss * (len(batch) / len(group))).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("存在缺失或非有限训练梯度")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    return dict(loss=total_loss / len(rows), optimizer_steps=steps, train_samples=len(rows))


def _evaluate(model, tokenizer, rows: list[dict], taxonomy: dict, *, device, micro_batch: int, max_length: int) -> dict:
    """前向评估一个显式数据集合。参数：模型/分词/样本/主题表/设备/批长；返回：损失、概率和参考标签；异常：非有限输出ValueError。"""
    import torch
    model.eval()
    total_loss, probabilities, targets = 0.0, [], []
    with torch.inference_mode():
        for start in range(0, len(rows), micro_batch):
            batch = rows[start:start + micro_batch]
            tokens, labels = _inputs(tokenizer, batch, taxonomy, device, max_length)
            output = model(**tokens, labels=labels)
            if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all() or output.logits.shape[-1] != 17:
                raise ValueError("评估损失或模型输出非法")
            total_loss += float(output.loss) * len(batch)
            probabilities.extend(torch.sigmoid(output.logits.float()).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return dict(loss=total_loss / len(rows), probabilities=probabilities, targets=targets)


def _test_reports(output_dir: Path, rows: list[dict], predictions: list[dict], taxonomy: dict) -> None:
    """按整体/来源/审核状态分别统计一次测试预测。参数：输出目录、样本、预测、主题表；返回：无；异常：IO/指标错误。"""
    expected = [row["labels"] for row in rows]
    predicted = [row["labels"] for row in predictions]
    reports = dict(overall=compute_metrics(expected, predicted, taxonomy), by_source={}, by_annotation_status={})
    for field, target in (("source", "by_source"), ("annotation_status", "by_annotation_status")):
        for value in sorted({row[field] for row in rows}):
            indices = [index for index, row in enumerate(rows) if row[field] == value]
            reports[target][value] = compute_metrics([expected[i] for i in indices], [predicted[i] for i in indices], taxonomy)
    _write_json(output_dir / "test_metrics.json", reports)
    errors = ["# 测试错误列表", "", "以下按模型预标或真实审核状态评估；不能当作真实客户泛化证明。", "",
              "|ID|来源|审核状态|文本|参考标签|预测|", "|---|---|---|---|---|---|"]
    with (output_dir / "test_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row, guess in zip(rows, predictions):
            record = {**row, "expected_labels": row["labels"], **guess}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if set(row["labels"]) != set(guess["labels"]):
                fields = [row["id"], row["source"], row["annotation_status"], row["text"], row["labels"], guess["labels"]]
                errors.append("|" + "|".join(str(value).replace("|", "\\|").replace("\n", " ").replace("<", "&lt;") for value in fields) + "|")
        stream.flush()
        os.fsync(stream.fileno())
    (output_dir / "test_errors.md").write_text("\n".join(errors), encoding="utf-8")


def train_model(dataset_dir: Path, base_dir: Path, output_dir: Path, *, epochs=8, micro_batch=2,
                accumulate=8, max_length=128, seed=42) -> dict:
    """训练并发布带可核对摘要的完整全参数模型包。

    参数：dataset_dir为已冻结且可校验的train/validation/test制品目录；base_dir为本地官方底座目录；
    output_dir为尚不存在的新目录；epochs为正整数最大轮数，micro_batch为正整数微批量，
    accumulate为正整数累计次数，max_length为至少2的token上限，seed为整数随机种子。
    返回：dict训练报告，含配置、逐轮指标、最佳轮次/阈值、更新次数、来源及数据/模型摘要。
    异常：输出目录已存在抛FileExistsError；配置/数据/模型非法抛ValueError；IO和训练错误传播，
    训练阶段失败或取消时保留已生成状态与原件，不发布残缺模型包。
    """
    dataset_dir, base_dir, output_dir = Path(dataset_dir), Path(base_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("训练输出目录已存在")
    for name, value in (("epochs", epochs), ("micro_batch", micro_batch), ("accumulate", accumulate), ("max_length", max_length)):
        if type(value) is not int or value < (2 if name == "max_length" else 1):
            raise ValueError("训练参数必须为正整数，长度至少为2")
    if type(seed) is not int:
        raise ValueError("训练种子必须为整数")
    taxonomy = load_taxonomy()
    dataset_manifest = validate_dataset(dataset_dir)
    source = _validate_base(base_dir)
    parts = {split: [json.loads(line) for line in (dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
             for split in ("train", "validation", "test")}
    if not all(parts.values()):
        raise ValueError("训练、验证和测试集合均不能为空")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", epochs=[], optimizer_steps=0, seed=seed,
                  configuration=dict(epochs=epochs, micro_batch=micro_batch, accumulate=accumulate, max_length=max_length,
                                     learning_rate=2e-5, weight_decay=.01, early_stopping_patience=2, optimizer="AdamW",
                                     problem_type="multi_label_classification", gradient_clip_norm=1.0,
                                     gradient_checkpointing=dict(enabled=True, use_reentrant=False),
                                     amp=dict(enabled_on="cuda", dtype="float16", init_scale=1024),
                                     shuffle="seed_plus_epoch"), evaluation_plan=evaluation_plan(),
                  taxonomy_digest=taxonomy_digest(taxonomy), data_sha256=file_sha256(dataset_dir / "manifest.json"),
                  base=source, annotation_status=dataset_manifest["human_review_status"],
                  annotation_counts=dict(Counter(row["annotation_status"] for rows in parts.values() for row in rows)))
    state_path, staging = output_dir / "training.json", output_dir / ".model-building"
    _write_json(state_path, report)
    try:
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        names = label_names(taxonomy)
        tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            base_dir, local_files_only=True, trust_remote_code=False,
            use_safetensors="pytorch_model.bin" not in source["files"],
            num_labels=17, problem_type="multi_label_classification", id2label=dict(enumerate(names)),
            label2id={name: i for i, name in enumerate(names)}, output_loading_info=True)
        if max_length > model.config.max_position_embeddings:
            raise ValueError("训练长度超出基础模型位置上限")
        # 全参数优化器覆盖编码器和分类头；不冻结、不插入低秩适配器。
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
        report.update(device=device.type, amp=device.type == "cuda", gradient_checkpointing=True,
                      all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
                      total_parameters=sum(p.numel() for p in model.parameters()), label_dtype="float32",
                      loading=json.loads(json.dumps(loading, default=list)))
        _write_json(state_path, report)
        best_macro_f1, stale, best_epoch = -math.inf, 0, None
        for epoch in range(1, epochs + 1):
            shuffled = list(parts["train"])
            random.Random(seed + epoch).shuffle(shuffled)
            training = _train_epoch(model, tokenizer, shuffled, taxonomy, optimizer, scaler, device=device,
                                    micro_batch=micro_batch, accumulate=accumulate, max_length=max_length)
            validation = _evaluate(model, tokenizer, parts["validation"], taxonomy, device=device,
                                   micro_batch=micro_batch, max_length=max_length)
            validation_threshold = select_threshold(validation["probabilities"], validation["targets"], taxonomy)
            validation_metrics = compute_metrics([row["labels"] for row in parts["validation"]],
                                                 [decode_labels(row, validation_threshold, taxonomy)
                                                  for row in validation["probabilities"]], taxonomy)
            macro_f1 = validation_metrics["macro_f1"]
            if macro_f1 is None:
                raise ValueError("验证集没有有支持类别，无法按宏F1选择模型")
            report["optimizer_steps"] += training["optimizer_steps"]
            # 训练前已归档每轮验证阈值和宏F1口径；损失仅记录，不参与选模或早停。
            if macro_f1 > best_macro_f1:
                best_macro_f1, stale, best_epoch = macro_f1, 0, epoch
                model.save_pretrained(staging)
            else:
                stale += 1
            report["epochs"].append(dict(epoch=epoch, **training, validation_loss=validation["loss"],
                                          validation_macro_f1=macro_f1, validation_threshold=validation_threshold,
                                          validation_exact_match=validation_metrics["exact_match"],
                                          validation_unsupported=validation_metrics["unsupported"],
                                          best_epoch=best_epoch, stale_epochs=stale))
            _write_json(state_path, report)
            print(json.dumps(report["epochs"][-1], ensure_ascii=False), flush=True)
            if stale >= report["evaluation_plan"]["early_stopping_patience"]:
                break
        # 只从最佳验证宏F1模型确定导出阈值；测试集合不参与早停或阈值选择。
        del optimizer, scaler, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = AutoModelForSequenceClassification.from_pretrained(staging, local_files_only=True, use_safetensors=True,
                                                                   trust_remote_code=False).to(device).eval()
        validation = _evaluate(model, tokenizer, parts["validation"], taxonomy, device=device,
                               micro_batch=micro_batch, max_length=max_length)
        threshold = select_threshold(validation["probabilities"], validation["targets"], taxonomy)
        validation_metrics = compute_metrics([row["labels"] for row in parts["validation"]],
                                             [decode_labels(row, threshold, taxonomy) for row in validation["probabilities"]], taxonomy)
        tokenizer.save_pretrained(staging)
        test = _evaluate(model, tokenizer, parts["test"], taxonomy, device=device,
                         micro_batch=micro_batch, max_length=max_length)
        full = tokenizer([row["text"] for row in parts["test"]], truncation=False, add_special_tokens=True)
        predictions = []
        for values, ids in zip(test["probabilities"], full["input_ids"]):
            labels = decode_labels(values, threshold, taxonomy)
            predictions.append(dict(labels=labels, probabilities=values, uncertain=not labels, truncated=len(ids) > max_length))
        _test_reports(output_dir, parts["test"], predictions, taxonomy)
        report.update(status="completed", best_epoch=best_epoch, threshold=threshold,
                      validation_loss=validation["loss"], validation_macro_f1=validation_metrics["macro_f1"],
                      test_loss=test["loss"], test_evaluations=1)
        _write_json(state_path, report)
        training_sha = file_sha256(state_path)
        manifest = dict(version="ch10-topic-" + training_sha[:16], taxonomy=taxonomy, taxonomy_digest=taxonomy_digest(taxonomy),
                        threshold=threshold, base=source, training_sha256=training_sha, data_sha256=report["data_sha256"],
                        annotation_status=report["annotation_status"], max_length=max_length,
                        training=dict(seed=seed, configuration=report["configuration"], evaluation_plan=report["evaluation_plan"]),
                        model_files={path.relative_to(staging).as_posix(): file_sha256(path) for path in staging.rglob("*") if path.is_file()})
        _write_json(staging / "topic_manifest.json", manifest)
        validate_model_package(staging)
        staging.rename(output_dir / "model")
        return report
    except BaseException as error:
        # KeyboardInterrupt等取消也落盘；只记录异常类型，不留下可加载的伪完成模型。
        report.update(status="cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                      error_type=type(error).__name__)
        for manifest_path in (output_dir / "model/topic_manifest.json", staging / "topic_manifest.json"):
            if manifest_path.exists():
                manifest_path.unlink()
        _write_json(state_path, report)
        raise
