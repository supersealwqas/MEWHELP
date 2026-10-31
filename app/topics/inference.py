"""校验离线完整主题模型包，并提供带不确定和截断标记的多标签推理。"""

import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path

from app.topics.metrics import decode_labels, evaluation_plan
from app.topics.limits import MAX_TOPIC_TEXT_LENGTH
from app.topics.taxonomy import label_names, load_taxonomy, taxonomy_digest


def file_sha256(path: Path) -> str:
    """分块计算文件摘要。参数：path本地文件；返回：SHA256；异常：IO错误。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_texts(texts: list[str]) -> None:
    """校验服务与直接推理共有输入。参数：texts为1至64条、每条1至4000字符的非空白文本；返回：None；异常：数量、类型或长度非法ValueError。"""
    if not isinstance(texts, list) or not 1 <= len(texts) <= 64:
        raise ValueError("文本数量必须为1至64")
    if any(not isinstance(text, str) or not 1 <= len(text) <= MAX_TOPIC_TEXT_LENGTH or not text.strip() for text in texts):
        raise ValueError(f"文本必须为1至{MAX_TOPIC_TEXT_LENGTH}字符的非空字符串")


def validate_model_package(model_dir: Path) -> dict:
    """在加载权重前核对全部模型文件和业务映射。参数：model_dir本地模型包；返回：清单；异常：ValueError。"""
    try:
        directory = Path(model_dir)
        manifest = json.loads((directory / "topic_manifest.json").read_text(encoding="utf-8"))
        required = {"version", "taxonomy", "taxonomy_digest", "threshold", "base", "training_sha256", "data_sha256",
                    "annotation_status", "max_length", "model_files", "training"}
        if not required <= manifest.keys() or not isinstance(manifest["version"], str) or not manifest["version"]:
            raise ValueError("模型清单缺少字段")
        taxonomy = load_taxonomy()
        if (manifest["taxonomy"] != taxonomy or manifest["taxonomy_digest"] != taxonomy_digest(taxonomy)):
            raise ValueError("模型与权威主题表不一致")
        threshold = manifest["threshold"]
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not .1 <= threshold <= .9:
            raise ValueError("模型阈值非法")
        if type(manifest["max_length"]) is not int or manifest["max_length"] < 2:
            raise ValueError("模型截断长度非法")
        training = manifest["training"]
        if not isinstance(training, dict) or type(training.get("seed")) is not int:
            raise ValueError("模型包缺少合法训练种子")
        if training.get("evaluation_plan") != evaluation_plan():
            raise ValueError("模型训练评价方案缺失或与固定规则不一致")
        configuration = training.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError("模型包训练配置缺失")
        for key in ("epochs", "micro_batch", "accumulate", "max_length"):
            if type(configuration.get(key)) is not int or configuration[key] < 1:
                raise ValueError("模型包训练批次或轮数非法")
        if configuration["max_length"] != manifest["max_length"]:
            raise ValueError("模型包训练长度与推理长度不一致")
        fixed = dict(learning_rate=2e-5, weight_decay=.01, early_stopping_patience=2, optimizer="AdamW",
                     problem_type="multi_label_classification", gradient_clip_norm=1.0,
                     gradient_checkpointing=dict(enabled=True, use_reentrant=False),
                     amp=dict(enabled_on="cuda", dtype="float16", init_scale=1024), shuffle="seed_plus_epoch")
        if any(configuration.get(key) != value for key, value in fixed.items()):
            raise ValueError("模型包训练优化与精度配置缺失或不符合本章规则")
        for key in ("training_sha256", "data_sha256"):
            digest = manifest[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("训练或数据摘要非法")
        if not isinstance(manifest["base"], dict) or not manifest["base"].get("revision"):
            raise ValueError("基础模型来源缺失")
        if manifest["annotation_status"] not in {"pending", "completed"}:
            raise ValueError("审核状态非法")
        files = manifest["model_files"]
        if not isinstance(files, dict) or not {"config.json", "tokenizer_config.json"} <= files.keys():
            raise ValueError("模型文件清单不完整")
        if not any(name.endswith(".safetensors") for name in files) or not any(name in files for name in ("tokenizer.json", "vocab.txt")):
            raise ValueError("模型权重或分词词表缺失")
        actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
                  and path.name != "topic_manifest.json"}
        if actual != set(files):
            raise ValueError("模型包缺失或含未登记文件")
        for name, digest in files.items():
            relative = Path(name)
            # 文件名不能越出模型包，也不能借符号链接绕过完整性校验。
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name or (directory / relative).is_symlink()
                    or not (directory / relative).resolve().is_relative_to(directory.resolve())):
                raise ValueError("模型文件路径非法")
            if file_sha256(directory / relative) != digest:
                raise ValueError("模型文件摘要不匹配")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        names = label_names(taxonomy)
        if (config.get("id2label") != {str(i): name for i, name in enumerate(names)}
                or config.get("label2id") != {name: i for i, name in enumerate(names)}
                or config.get("problem_type") != "multi_label_classification"
                or manifest["max_length"] > config.get("max_position_embeddings", 0)):
            raise ValueError("模型配置与17类映射不一致")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("模型包缺失或结构非法") from error


class TopicPredictor:
    """只加载已校验本地模型，以固定主题表和全局阈值提供独立推理。"""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        """校验后离线加载一次。参数：模型包、cpu或cuda设备；返回：无；异常：ValueError/模型加载错误。"""
        self._metadata = validate_model_package(Path(model_dir))
        if device not in {"cpu", "cuda"}:
            raise ValueError("设备仅支持cpu或cuda")
        # 防止模型加载期间联系远端仓库或自动转换服务，不启用远端代码。
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA设备不可用")
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self._model, loading = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False, output_loading_info=True)
        # 训练后的完整包不能像基础掩码模型那样静默随机补齐分类头。
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("模型包权重未完整匹配分类模型")
        if self._model.config.num_labels != 17:
            raise ValueError("模型分类头必须为17维")
        self._model.to(self._device).eval()

    @property
    def metadata(self) -> dict:
        """返回不可反向修改内部状态的清单副本。参数：无；返回：元信息字典；异常：无。"""
        return deepcopy(self._metadata)

    def predict(self, texts: list[str]) -> list[dict]:
        """执行真实分词、概率和业务解码。参数：合法文本列表；返回：标签/概率/未确定/截断；异常：ValueError/运算错误。"""
        import torch
        validate_texts(texts)
        length = self._metadata["max_length"]
        full = self._tokenizer(texts, truncation=False, add_special_tokens=True)
        truncated = [len(ids) > length for ids in full["input_ids"]]
        inputs = self._tokenizer(texts, padding=True, truncation=True, max_length=length, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        # inference_mode的作用域在线程内，不能只在应用启动线程开启。
        with torch.inference_mode():
            logits = self._model(**inputs).logits
            if tuple(logits.shape) != (len(texts), 17) or not torch.isfinite(logits).all():
                raise ValueError("模型输出维数或数值非法")
            probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        results = []
        for values, cut in zip(probabilities, truncated):
            labels = decode_labels(values, self._metadata["threshold"], self._metadata["taxonomy"])
            results.append(dict(labels=labels, probabilities=values, uncertain=not labels, truncated=cut))
        return results
