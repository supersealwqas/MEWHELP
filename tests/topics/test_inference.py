"""验证完整模型包、真实解码和独立HTTP服务的输入、并发与异常边界。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from tests.topics.test_training import cpu_offline, trained_tiny


def test_predictor_validates_text_and_reports_truncation_and_readonly_metadata(trained_tiny):
    """捕获输入越界、静默截断和元信息可变的错误。参数：真实制品；返回：无；异常：断言失败。"""
    from app.topics.inference import TopicPredictor
    predictor = TopicPredictor(trained_tiny[2] / "model")
    result = predictor.predict(["我要退货", "退" * 100])
    assert result[0]["truncated"] is False and result[1]["truncated"] is True
    assert len(result[0]["probabilities"]) == 17
    assert result[0]["uncertain"] == (not result[0]["labels"])
    metadata = predictor.metadata
    metadata["threshold"] = 0
    assert predictor.metadata["threshold"] != 0
    for texts in ([], [" "], [1], ["退" * 2001], ["退"] * 65):
        with pytest.raises(ValueError):
            predictor.predict(texts)


@pytest.mark.parametrize("mutation", ["missing", "weights", "taxonomy", "dimension", "extra", "head",
                                      "training_missing", "training_seed", "training_policy", "training_parameters"])
def test_predictor_rejects_incomplete_or_mismatched_model_package(trained_tiny, mutation):
    """捕获模型缺失、SHA破坏与主题映射错配的错误。参数：制品、破坏方式；返回：无；异常：断言失败。"""
    from app.topics.inference import TopicPredictor
    model_dir = trained_tiny[2] / "model"
    manifest_path = model_dir / "topic_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        (model_dir / "config.json").unlink()
    elif mutation == "weights":
        with (model_dir / "model.safetensors").open("ab") as stream:
            stream.write(b"corrupt")
    elif mutation == "taxonomy":
        manifest["taxonomy"]["labels"][0]["name"] = "未知"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    elif mutation == "extra":
        (model_dir / "untracked.txt").write_text("injected", encoding="utf-8")
    elif mutation == "head":
        from safetensors.torch import load_file, save_file
        path = model_dir / "model.safetensors"
        state = load_file(path)
        state.pop("classifier.weight")
        save_file(state, path, metadata={"format": "pt"})
        manifest["model_files"]["model.safetensors"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    elif mutation.startswith("training_"):
        if mutation == "training_missing":
            manifest.pop("training", None)
        elif mutation == "training_seed":
            manifest.setdefault("training", {})["seed"] = True
        elif mutation == "training_policy":
            manifest.setdefault("training", {})["evaluation_plan"] = {"threshold_source": "test"}
        else:
            manifest.setdefault("training", {})["configuration"] = {"epochs": 0}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    else:
        path = model_dir / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["id2label"].pop("16")
        path.write_text(json.dumps(config), encoding="utf-8")
        manifest["model_files"]["config.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        TopicPredictor(model_dir)


def test_runtime_nonfinite_logits_and_uncertain_use_real_decoder(trained_tiny, monkeypatch):
    """只替代权重计算，捕获NaN输出和强制argmax错误。参数：制品、替换器；返回：无；异常：断言失败。"""
    import torch
    from types import SimpleNamespace
    from app.topics.inference import TopicPredictor
    predictor = TopicPredictor(trained_tiny[2] / "model")

    def low_logits(**inputs):
        """替代纯权重计算。参数：真实tokenizer输入；返回：低分logits；异常：无。"""
        return SimpleNamespace(logits=torch.full((inputs["input_ids"].shape[0], 17), -20.0))

    monkeypatch.setattr(predictor._model, "forward", low_logits)
    result = predictor.predict(["我要退货"])[0]
    assert result["labels"] == [] and result["uncertain"] is True

    def bad_logits(**inputs):
        """替代纯权重计算。参数：真实输入；返回：非法logits；异常：无。"""
        return SimpleNamespace(logits=torch.full((1, 17), float("nan")))

    monkeypatch.setattr(predictor._model, "forward", bad_logits)
    with pytest.raises(ValueError):
        predictor.predict(["我要退货"])


def test_service_loads_once_validates_inputs_and_returns_model_identity(trained_tiny):
    """捕获每请求重载、非法客户端配置和请求上限失守的错误。参数：真实制品；返回：无；异常：断言失败。"""
    from app.topics.inference import TopicPredictor
    from app.topic_service import create_app
    calls = []

    def factory():
        """提供真实预测器并记录实际加载。参数：无；返回：预测器；异常：模型加载错误。"""
        calls.append(1)
        return TopicPredictor(trained_tiny[2] / "model")

    with TestClient(create_app(factory)) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/classify", json={"texts": ["我要退货"]})
        assert response.status_code == 200
        assert response.json()["taxonomy_digest"] == trained_tiny[3]["taxonomy_digest"]
        assert response.json()["model_version"] and len(response.json()["results"]) == 1
        assert client.post("/classify", json={"texts": ["退"] * 64}).status_code == 200
        for body in ({"texts": []}, {"texts": [" "]}, {"texts": [2]}, {"texts": ["退"] * 65},
                     {"texts": ["退" * 2001]}, {"texts": ["退"], "model_dir": "secret"}):
            invalid = client.post("/classify", json=body)
            assert invalid.status_code == 422
            assert "secret" not in invalid.text
    assert calls == [1]


def test_service_bounds_concurrency_and_sanitizes_errors(trained_tiny, monkeypatch):
    """捕获无界并发和异常路径泄漏的错误。参数：真实制品、替换器；返回：无；异常：断言失败。"""
    from app.topics.inference import TopicPredictor
    from app.topic_service import create_app
    predictor = TopicPredictor(trained_tiny[2] / "model")
    entered, release = Event(), Event()

    def blocking_predict(texts):
        """阻塞纯计算边界以验证限流。参数：文本；返回：无；异常：模拟敏感运算错误。"""
        entered.set()
        assert release.wait(10)
        raise RuntimeError("D:/secret/path " + texts[0])

    monkeypatch.setattr(predictor, "predict", blocking_predict)
    with TestClient(create_app(lambda: predictor)) as client, ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.post, "/classify", json={"texts": ["私密输入"]})
        assert entered.wait(10)
        busy = client.post("/classify", json={"texts": ["其他"]})
        assert busy.status_code == 503
        release.set()
        failed = first.result(timeout=10)
        assert failed.status_code == 500
        assert "secret" not in failed.text and "私密输入" not in failed.text


def test_service_import_and_health_need_no_torch_or_chat_configuration(tmp_path, cpu_offline):
    """捕获主模块导入torch或聊天配置依赖的错误。参数：临时目录、离线约束；返回：无；异常：断言失败。"""
    import os
    import subprocess
    import sys
    source = "import sys; import app.topic_service; assert 'torch' not in sys.modules; assert 'app.config' not in sys.modules"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CHAT_")}
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    from app.topic_service import create_app
    with TestClient(create_app(lambda: None)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/classify", json={"texts": ["退货"]}).status_code == 503


async def test_cancelled_request_keeps_slot_until_real_worker_finishes(trained_tiny, monkeypatch):
    """捕获取消请求后旧线程仍运行却放行新推理的错误。参数：制品、替换器；返回：无；异常：断言失败。"""
    import asyncio
    from fastapi import HTTPException
    from starlette.requests import Request
    from app.topics.inference import TopicPredictor
    from app.topic_service import ClassifyRequest, create_app
    predictor = TopicPredictor(trained_tiny[2] / "model")
    real_predict = predictor.predict
    entered, release, finished = Event(), Event(), Event()

    def blocking_predict(texts):
        """仅阻塞首次真实计算，保留分词和解码。参数：文本；返回：真实预测；异常：阻塞失败AssertionError。"""
        if texts[0] == "阻塞首次":
            entered.set()
            try:
                assert release.wait(10)
                return real_predict(texts)
            finally:
                finished.set()
        return real_predict(texts)

    monkeypatch.setattr(predictor, "predict", blocking_predict)
    application = create_app(lambda: predictor)
    endpoint = next(route.endpoint for route in application.routes if getattr(route, "path", None) == "/classify")
    request = Request({"type": "http", "app": application})
    async with application.router.lifespan_context(application):
        task = asyncio.create_task(endpoint(ClassifyRequest(texts=["阻塞首次"]), request))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(HTTPException) as busy:
                await endpoint(ClassifyRequest(texts=["我要退货"]), request)
            assert busy.value.status_code == 503 and not finished.is_set()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 10)
        recovered = await endpoint(ClassifyRequest(texts=["我要退货"]), request)
        assert len(recovered["results"]) == 1


async def test_timeout_returns_504_and_holds_slot_until_worker_finishes(trained_tiny, monkeypatch):
    """捕获服务无限等待或超时提前释放真实计算槽的错误。参数：制品、替换器；返回：无；异常：断言失败。"""
    import asyncio
    import time
    from fastapi import HTTPException
    from starlette.requests import Request
    from app.topics.inference import TopicPredictor
    from app.topic_service import ClassifyRequest, create_app
    predictor = TopicPredictor(trained_tiny[2] / "model")
    real_predict = predictor.predict
    entered, release, finished = Event(), Event(), Event()

    def blocking_predict(texts):
        """只阻塞指定实际预测。参数：文本；返回：真实预测；异常：阻塞失败AssertionError。"""
        entered.set()
        try:
            assert release.wait(10)
            return real_predict(texts)
        finally:
            finished.set()

    monkeypatch.setattr(predictor, "predict", blocking_predict)
    application = create_app(lambda: predictor, inference_timeout=.03)
    endpoint = next(route.endpoint for route in application.routes if getattr(route, "path", None) == "/classify")
    request = Request({"type": "http", "app": application})
    async with application.router.lifespan_context(application):
        started = time.monotonic()
        try:
            with pytest.raises(HTTPException) as timeout:
                await endpoint(ClassifyRequest(texts=["私密输入"]), request)
            assert timeout.value.status_code == 504 and time.monotonic() - started < 2
            assert "私密输入" not in str(timeout.value.detail)
            assert entered.is_set() and not finished.is_set()
            with pytest.raises(HTTPException) as busy:
                await endpoint(ClassifyRequest(texts=["我要退货"]), request)
            assert busy.value.status_code == 503
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 10)
        monkeypatch.setattr(predictor, "predict", real_predict)
        recovered = await endpoint(ClassifyRequest(texts=["我要退货"]), request)
        assert len(recovered["results"]) == 1
