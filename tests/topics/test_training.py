"""用离线微型真实BERT验证训练程序，不代替本章真实GPU训练。"""

import hashlib
import json
import socket
from pathlib import Path

import pytest

from app.topics.data import write_dataset
from app.topics.taxonomy import label_names, load_taxonomy


@pytest.fixture
def cpu_offline(monkeypatch):
    """禁用触网并强制CPU小模型实验。参数：替换器；返回：无；异常：任何网络尝试AssertionError。"""
    import torch
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    original_connect = socket.socket.connect

    def reject_network(*args, **kwargs):
        """拒绝网络连接。参数：套接字任意参数；返回：无；异常：AssertionError。"""
        raise AssertionError("Task2离线测试禁止网络")

    def local_event_loop_only(connection, address):
        """允许Windows事件循环内部本机通信，拒绝外部连接。参数：socket、地址；返回：原结果；异常：AssertionError。"""
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(connection, address)
        return reject_network(connection, address)

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", local_event_loop_only)
    yield
    torch.set_num_threads(previous_threads)


def tiny_base(directory: Path) -> Path:
    """保存独立标识的真实微型BERT。参数：新目录；返回：目录；异常：模型保存或IO错误。"""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
    directory.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("退货物流尺码问题我要第号多少")
    (directory / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(directory)
    names = label_names(load_taxonomy())
    torch.manual_seed(7)
    config = BertConfig(vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=24, max_position_embeddings=32,
                        num_labels=17, problem_type="multi_label_classification",
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                        id2label=dict(enumerate(names)), label2id={name: i for i, name in enumerate(names)})
    BertForSequenceClassification(config).save_pretrained(directory)
    files = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
             for path in directory.iterdir() if path.is_file()}
    (directory / "source-manifest.json").write_text(json.dumps(dict(repo="local/tiny-cpu-test", revision="tiny-test-v1", files=files)), encoding="utf-8")
    return directory


def tiny_dataset(directory: Path) -> Path:
    """冻结少量独立家族训练/留出样本。参数：新目录；返回：目录；异常：数据校验或IO错误。"""
    parts = {}
    for split, count in (("train", 5), ("validation", 2), ("test", 2)):
        parts[split] = [dict(id=f"{split}:{i}", text=f"第{split}{i}号我要退货", labels=["退换货"],
                            source="synthetic", group_id=f"{split}:{i}", pool_ids=[], parent_id=None,
                            annotation_status="model_prelabel") for i in range(count)]
    write_dataset(directory, parts, load_taxonomy(), seed=42)
    return directory


@pytest.fixture
def trained_tiny(tmp_path, cpu_offline):
    """执行真实CPU训练并提供输出。参数：临时目录、离线CPU约束；返回：制品和报告；异常：训练错误。"""
    from app.topics.training import train_model
    base = tiny_base(tmp_path / "base")
    dataset = tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "trained"
    report = train_model(dataset, base, output, epochs=1, micro_batch=2, accumulate=2, max_length=16)
    return base, dataset, output, report


def test_real_cpu_training_updates_encoder_and_exports_consistent_probabilities(trained_tiny):
    """捕获冻结编码器、错误标签dtype、漏尾组和失真导出的错误。参数：真实训练制品；返回：无；异常：断言失败。"""
    import torch
    from transformers import AutoModelForSequenceClassification
    from app.topics.inference import TopicPredictor
    base, dataset, output, report = trained_tiny
    before = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True, use_safetensors=True)
    after = AutoModelForSequenceClassification.from_pretrained(output / "model", local_files_only=True, use_safetensors=True)
    assert after.config.num_labels == 17 and all(p.requires_grad for p in after.parameters())
    assert not torch.equal(before.bert.encoder.layer[0].attention.self.query.weight,
                           after.bert.encoder.layer[0].attention.self.query.weight)
    assert report["all_parameters_trainable"] and report["label_dtype"] == "float32"
    assert report["optimizer_steps"] == 2 and report["epochs"][0]["train_samples"] == 5
    assert report["device"] == "cpu" and report["status"] == "completed"
    assert report["base"]["repo"] == "local/tiny-cpu-test"
    assert (output / "model/model.safetensors").is_file()
    predictor = TopicPredictor(output / "model")
    predictions = [json.loads(line) for line in (output / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    reloaded = predictor.predict([row["text"] for row in predictions])
    for original, loaded in zip(predictions, reloaded):
        assert loaded["probabilities"] == pytest.approx(original["probabilities"], abs=1e-7)
    metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["samples"] == 2
    assert metrics["by_source"]["synthetic"]["samples"] == 2
    assert metrics["by_annotation_status"]["model_prelabel"]["samples"] == 2
    assert (output / "test_errors.md").exists()
    manifest = predictor.metadata
    assert manifest["annotation_status"] == "pending"
    assert manifest["base"]["revision"] == "tiny-test-v1"
    assert manifest["data_sha256"] == hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()


def test_tail_group_matches_single_effective_batch_update(tmp_path, cpu_offline):
    """捕获尾组损失仍除以固定累计次数的错误。参数：临时目录、离线CPU；返回：无；异常：断言失败。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from app.topics.training import _train_epoch
    base = tiny_base(tmp_path / "base")
    model1 = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True)
    model2 = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rows = [dict(text="我要退货", labels=["退换货"]) for _ in range(3)]
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=.1)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    _train_epoch(model1, tokenizer, rows, load_taxonomy(), optimizer1, scaler,
                 device=torch.device("cpu"), micro_batch=2, accumulate=8, max_length=16)
    _train_epoch(model2, tokenizer, rows, load_taxonomy(), optimizer2, scaler,
                 device=torch.device("cpu"), micro_batch=3, accumulate=1, max_length=16)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2, atol=1e-7, rtol=1e-5)


def test_train_rejects_existing_directory_and_source_tampering(tmp_path, cpu_offline):
    """捕获覆盖训练目录或伪造来源权重的错误。参数：临时目录、离线CPU；返回：无；异常：断言失败。"""
    from app.topics.training import train_model
    base = tiny_base(tmp_path / "base")
    dataset = tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        train_model(dataset, base, output, epochs=1)
    (base / "vocab.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        train_model(dataset, base, tmp_path / "invalid", epochs=1)


@pytest.mark.parametrize("failure,status", [(RuntimeError("secret path"), "failed"), (KeyboardInterrupt(), "cancelled")])
def test_training_failure_flushes_status_without_complete_model(tmp_path, cpu_offline, monkeypatch, failure, status):
    """捕获失败取消后伪完成标记的错误。参数：目录、隔离、替换器、异常、状态；返回：无；异常：断言失败。"""
    from app.topics import training
    base = tiny_base(tmp_path / "base")
    dataset = tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "failed"

    def fail_epoch(*args, **kwargs):
        """模拟训练运算失败。参数：训练参数；返回：无；异常：指定失败或取消。"""
        raise failure

    monkeypatch.setattr(training, "_train_epoch", fail_epoch)
    with pytest.raises(type(failure)):
        training.train_model(dataset, base, output, epochs=1, max_length=16)
    report = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert report["status"] == status and "secret path" not in json.dumps(report)
    assert not (output / "model/topic_manifest.json").exists()


def test_cli_trains_and_predicts_only_explicit_local_directories(trained_tiny, monkeypatch, capsys):
    """捕获CLI训练预测入口缺失或参数不透传的错误。参数：制品、替换器、输出捕获；返回：无；异常：断言失败。"""
    import sys
    from app.topics.cli import main
    base, dataset, output, _ = trained_tiny
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["topics", "predict", "--model-dir", str(output / "model"), "--text", "我要退货"])
    main()
    prediction = json.loads(capsys.readouterr().out)
    assert len(prediction["results"]) == 1 and len(prediction["results"][0]["probabilities"]) == 17
    new_output = output.with_name("cli-trained")
    monkeypatch.setattr(sys, "argv", ["topics", "train", "--dataset-dir", str(dataset), "--base-dir", str(base),
                                     "--output-dir", str(new_output), "--epochs", "1", "--micro-batch", "2",
                                     "--accumulate", "8", "--max-length", "16", "--seed", "11"])
    main()
    report = json.loads((new_output / "training.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed" and report["optimizer_steps"] == 1 and report["seed"] == 11


def test_validation_early_stopping_patience_two_and_single_test_evaluation(tmp_path, cpu_offline, monkeypatch):
    """捕获验证早停失效或多次消费测试集的错误。参数：目录、隔离、替换器；返回：无；异常：断言失败。"""
    from app.topics import training
    base = tiny_base(tmp_path / "base")
    dataset = tiny_dataset(tmp_path / "dataset")
    evaluate = training._evaluate
    test_evaluations = []

    def stable_validation(*args, **kwargs):
        """保留真实前向，仅控制验证损失为固定值。参数：真实评估参数；返回：真实概率/固定验证分；异常：模型错误。"""
        result = evaluate(*args, **kwargs)
        rows = args[2]
        if rows[0]["id"].startswith("validation:"):
            result["loss"] = 1.0
        else:
            test_evaluations.append(1)
        return result

    monkeypatch.setattr(training, "_evaluate", stable_validation)
    report = training.train_model(dataset, base, tmp_path / "early", epochs=8, max_length=16)
    assert len(report["epochs"]) == 3 and report["best_epoch"] == 1
    assert report["test_evaluations"] == 1 and test_evaluations == [1]


def test_final_package_failure_cannot_leave_loadable_staging(tmp_path, cpu_offline, monkeypatch):
    """捕获发布末尾失败后临时目录仍冒充完成模型的错误。参数：目录、隔离、替换器；返回：无；异常：断言失败。"""
    from app.topics import training
    base = tiny_base(tmp_path / "base")
    dataset = tiny_dataset(tmp_path / "dataset")
    output = tmp_path / "rejected-package"

    def reject_package(directory):
        """模拟真实完整性核验失败。参数：临时模型包；返回：无；异常：ValueError。"""
        raise ValueError("package check failed")

    monkeypatch.setattr(training, "validate_model_package", reject_package)
    with pytest.raises(ValueError):
        training.train_model(dataset, base, output, epochs=1, max_length=16)
    assert json.loads((output / "training.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert not list(output.rglob("topic_manifest.json"))


def test_untracked_base_file_cannot_override_verified_tokenizer(tmp_path, cpu_offline):
    """捕获未登记基础模型文件覆盖已核验输入的错误。参数：目录、离线约束；返回：无；异常：断言失败。"""
    from app.topics.training import train_model
    base = tiny_base(tmp_path / "base")
    dataset = tiny_dataset(tmp_path / "dataset")
    (base / "untracked.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        train_model(dataset, base, tmp_path / "invalid-extra", epochs=1, max_length=16)
