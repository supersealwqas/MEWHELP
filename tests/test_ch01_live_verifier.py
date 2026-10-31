"""CH01 在线验收入口的离线回归：核对当前 API、独立会话和失败退出。"""

import importlib
import json

import httpx
import pytest
from langchain_core.runnables import RunnableLambda


def test_help_and_explicit_service_target(capsys):
    """帮助无需服务即可显示，未指定目标必须退出。参数：捕获器；返回：无；异常：断言失败。"""
    verifier = importlib.import_module("tests.verify_ch01_live")
    with pytest.raises(SystemExit) as help_exit:
        verifier.main(["--help"])
    assert help_exit.value.code == 0
    assert "--base-url" in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing_target:
        verifier.main([])
    assert missing_target.value.code == 2


@pytest.mark.asyncio
async def test_verifier_runs_current_api_with_new_sessions(offline_chat_runtime, monkeypatch):
    """真实应用生命周期和工作流可执行全部三组检查。参数：隔离依赖/替换器；返回：无；异常：断言失败。"""
    verifier = importlib.import_module("tests.verify_ch01_live")
    from app.main import app
    from app.schemas.extract import AfterSalesTicket
    from app.schemas.workflow import AgentDecision

    extractor_api = importlib.import_module("app.api.extract")
    samples = {item["input"]: item for item in verifier.LABELED_SAMPLES}

    def extract_sample(payload):
        """仅替换外部模型输出，保留真实抽取路由。参数：请求；返回：抽取对象；异常：键缺失。"""
        sample = samples[payload["text"]]
        return AfterSalesTicket(order_id=sample["expected_order_id"],
            request_type=sample["expected_request_type"], expected_solution=sample["solution_keywords"][0])

    monkeypatch.setattr(extractor_api, "get_extractor", lambda: RunnableLambda(extract_sample))
    _, repository, _, model = offline_chat_runtime
    model.intents.clear()
    model.intents.extend(["订单"] * 4)
    model.decisions.clear()
    model.decisions.extend([AgentDecision(kind="answer")] * 4)
    model.answers.clear()
    model.answers.extend([["猫粮配送请以订单为准。"], ["你刚才问的是猫粮。"],
                          ["无法直接替你转接人工。"], ["不能保证退款时间或承诺赔偿。"]])
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            assert await verifier.run_verification(client) == 0
    turns = list(repository.turns.values())
    assert len(turns) == 4
    assert all(turn.turn_status == "completed" for turn in turns)
    sessions = [context.session_id for context in repository.contexts]
    assert sessions[0] == sessions[1] and len(set(sessions)) == 3
    assert all(session.startswith("live-") and len(session) <= 64 for session in sessions)
    assert len(repository.turns) == 4  # 验收结束保留专用会话，不清空数据库记录。


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["extract", "missing_done", "server_error", "boundary"])
async def test_any_failed_check_returns_nonzero(defect):
    """抽取、SSE或承诺边界失败均使整次验收失败。参数：缺陷类型；返回：无；异常：断言失败。"""
    verifier = importlib.import_module("tests.verify_ch01_live")
    requests = []

    def respond(request):
        """模拟远端HTTP响应，记录资源边界。参数：请求；返回：响应；异常：JSON解析错误。"""
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path == "/api/extract":
            if defect == "extract":
                return httpx.Response(502)
            sample = next(item for item in verifier.LABELED_SAMPLES if item["input"] == body["text"])
            return httpx.Response(200, json={"order_id": sample["expected_order_id"],
                "request_type": sample["expected_request_type"], "expected_solution": sample["solution_keywords"][0]})
        reply = "猫粮。不能保证到账，也无法转接人工。"
        if defect == "boundary":
            reply = "猫粮。保证今天到账，已转接成功。"
        content = "data: " + json.dumps({"delta": reply}, ensure_ascii=False) + "\n\n"
        if defect == "server_error":
            content += 'event: error\ndata: {"message":"失败"}\n\n'
        if defect != "missing_done":
            content += "data: [DONE]\n\n"
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test") as client:
        assert await verifier.run_verification(client) == 1
    assert all(request.method == "POST" for request in requests)
    assert {request.url.path for request in requests} == {"/api/chat", "/api/extract"}
