"""意图用量的真实来源、分页完整性与原轮归属测试；外部 HTTP 使用受控响应。"""

import importlib.util
from datetime import datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

START = datetime(2026, 10, 30, 8, tzinfo=timezone.utc)
END = datetime(2026, 10, 30, 9, tzinfo=timezone.utc)


def costs_module():
    """加载待实现模块。参数：无；返回：模块；异常：缺少功能时断言失败。"""
    assert importlib.util.find_spec("app.observability.costs"), "尚未实现真实意图用量统计"
    from app.observability import costs
    return costs


def generation(key, trace="trace-1", *, intent="unclassified", task_kind="chat", input=10, output=2):
    """构造原始HTTP观测。参数：ID/原轮/标签/可缺失的原始双向计量；返回：dict；异常：无。"""
    metadata = {"intent": intent, "task_kind": task_kind,
        "usage_origin": "model_reported" if input is not None and output is not None else "inferred_or_unknown",
        "input_usage_present": input is not None, "output_usage_present": output is not None}
    if input is not None:
        metadata["reported_input_tokens"] = input
    if output is not None:
        metadata["reported_output_tokens"] = output
    return {"id": key, "traceId": trace, "type": "GENERATION", "startTime": START.isoformat(),
        "name": "ChatOpenAI", "metadata": metadata, "model": "test-model", "usageDetails": {}, "totalCost": None}


def root(trace="trace-1", *, intent="商品咨询", kind="chat", key="root-1"):
    """构造窗外逻辑根。参数：原轮/意图/执行类别/观测ID；返回：dict；异常：无。"""
    return {"id": key, "traceId": trace, "type": "SPAN", "name": f"execution.{kind}",
        "startTime": "2026-10-30T07:00:00Z", "parentObservationId": "physical-parent",
        "isRootObservation": True, "metadata": {"intent": intent,
        "task_kind": kind if kind in {"summary", "evaluation", "flywheel"} else "chat"}}


def reader_for(handler, offline_settings):
    """只替换HTTP传输。参数：响应器/隔离配置；返回：真实reader；异常：配置错误。"""
    return costs_module().LangfuseObservationReader(offline_settings.model_copy(update={
        "observability_enabled": True, "langfuse_public_key": "test-public",
        "langfuse_secret_key": SecretStr("test-secret")}),
        transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_costs_include_all_pages_and_deduplicate_observations(offline_settings):
    """防止漏页与重复计费。参数：离线配置；返回：None；异常：断言失败。"""
    requests = []

    def handler(request):
        """返回含重复ID的两页。参数：HTTP请求；返回：响应；异常：断言失败。"""
        requests.append(request)
        if request.url.params.get("cursor") == "page-2":
            return httpx.Response(200, json={"data": [generation("g1"), generation("g2")], "meta": {}})
        return httpx.Response(200, json={"data": [generation("g1")], "meta": {"cursor": "page-2"}})

    reader = reader_for(handler, offline_settings)
    rows = [item async for item in reader.iter_observations(start=START, end=END, observation_type="GENERATION")]
    result = costs_module().summarize_intent_usage(rows)
    assert result["groups"][0]["model_call_count"] == 2
    assert result["groups"][0]["total_tokens"] == 24
    assert len(requests) == 2
    query = requests[0].url.params
    assert requests[0].url.path == "/api/public/v2/observations"
    assert {"metadata", "model", "usage", "trace_context"} <= set(query["fields"].split(","))
    assert query["type"] == "GENERATION"
    assert query["fromStartTime"] == START.isoformat()
    assert query["toStartTime"] == END.isoformat()
    assert "parseIoAsJson" not in query and "filter" not in query


def test_preclassification_and_retries_belong_to_original_intent():
    """防止分类前和恢复调用拆成新请求。参数：无；返回：None；异常：断言失败。"""
    result = costs_module().summarize_intent_usage([
        generation("classify"), generation("retry"), generation("answer", intent="订单查询"),
        root(intent="订单查询"), root(intent="unclassified", kind="resume", key="resume-1")])
    assert result["groups"] == [{"task_kind": "chat", "intent": "订单查询", "request_count": 1,
        "model_call_count": 3, "input_tokens": 30, "output_tokens": 6, "total_tokens": 36,
        "unknown_usage_count": 0, "known_token_share": 1.0, "cost": None}]


def test_unknown_usage_is_not_budget_or_zero():
    """防止用预留或零补原始缺失。参数：无；返回：None；异常：断言失败。"""
    unknown = generation("unknown", input=None, output=None)
    unknown["metadata"]["reserved_units"] = 500
    result = costs_module().summarize_intent_usage([unknown, generation("partial", input=7, output=None),
        generation("zero", input=0, output=3), root()])
    group = result["groups"][0]
    assert (group["input_tokens"], group["output_tokens"], group["total_tokens"]) == (7, 3, 10)
    assert group["unknown_usage_count"] == 2
    assert result["coverage"]["fully_reported_call_count"] == 1
    assert result["coverage"]["usage_coverage"] == pytest.approx(1 / 3)


def test_background_work_is_not_chat_intent_cost():
    """防止父级双计及后台污染客服分组。参数：无；返回：None；异常：断言失败。"""
    parent = root()
    parent.update(usageDetails={"input": 10000, "output": 10000}, totalCost=500)
    result = costs_module().summarize_intent_usage([parent, generation("chat"),
        generation("eval", "trace-eval", task_kind="evaluation", input=7, output=3),
        root("trace-eval", kind="evaluation", key="eval-root")])
    groups = {item["task_kind"]: item for item in result["groups"]}
    assert groups["chat"]["total_tokens"] == 12
    assert groups["evaluation"]["total_tokens"] == 10
    assert groups["evaluation"]["intent"] is None
    assert groups["chat"]["known_token_share"] == pytest.approx(12 / 22)
    assert all(item["cost"] is None for item in groups.values())


def test_inferred_usage_does_not_become_reported_tokens():
    """拒绝服务器推算和SDK错误零金额。参数：无；返回：None；异常：断言失败。"""
    row = generation("inferred", input=None, output=None)
    row.update(usageDetails={"input": 400, "output": 100, "total": 500}, totalCost=0, totalUsage=500)
    group = costs_module().summarize_intent_usage([row])["groups"][0]
    assert (group["total_tokens"], group["unknown_usage_count"], group["known_token_share"], group["cost"]) == (0, 1, None, None)


@pytest.mark.parametrize("invalid", [True, "7", 7.1, -1])
def test_invalid_original_measurement_remains_unknown(invalid):
    """不把布尔字符串负数当token。参数：非法计量；返回：None；异常：断言失败。"""
    group = costs_module().summarize_intent_usage([generation("bad", input=invalid)])["groups"][0]
    assert group["input_tokens"] == 0
    assert group["unknown_usage_count"] == 1


@pytest.mark.asyncio
async def test_window_outside_root_is_used_only_for_attribution(offline_settings):
    """窗外根补查不能把窗外用量带入。参数：离线配置；返回：None；异常：断言失败。"""
    requests = []

    def handler(request):
        """模拟同轮补查。参数：HTTP请求；返回：观测页；异常：无。"""
        requests.append(request)
        rows = [root(), generation("outside", input=999)] if request.url.params.get("traceId") else [generation("inside")]
        return httpx.Response(200, json={"data": rows, "meta": {}})

    result = await costs_module().get_intent_usage(START, END, reader=reader_for(handler, offline_settings))
    assert result["groups"][0]["intent"] == "商品咨询"
    assert result["groups"][0]["total_tokens"] == 12
    assert result["coverage"]["complete"] is True
    assert len(requests) == 2
    assert requests[1].url.params["traceId"] == "trace-1"
    assert "fromStartTime" not in requests[1].url.params


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["page", "attribution"])
async def test_partial_read_is_explicit_failure_not_complete_bill(offline_settings, failure_stage):
    """分页或归属失败必须撤回完整账单。参数：配置/故障阶段；返回：None；异常：断言失败。"""
    def handler(request):
        """制造后续页或补查故障。参数：请求；返回：响应；异常：无。"""
        if request.url.params.get("cursor") or request.url.params.get("traceId"):
            return httpx.Response(503, text="secret upstream details")
        return httpx.Response(200, json={"data": [generation("g1")],
            "meta": {"cursor": "next"} if failure_stage == "page" else {}})

    result = await costs_module().get_intent_usage(START, END, reader=reader_for(handler, offline_settings))
    assert result["coverage"]["complete"] is False
    assert result["coverage"]["failure_stage"] == failure_stage
    assert result["coverage"]["collected_model_call_count"] == 1
    assert result["groups"] == []
    assert "secret" not in str(result)


@pytest.mark.asyncio
async def test_reader_refuses_unbounded_query_and_cursor_loop(offline_settings):
    """无界查询和循环游标不能挂死读取。参数：配置；返回：None；异常：断言失败。"""
    def handler(request):
        """固定重复游标。参数：请求；返回：响应；异常：无。"""
        return httpx.Response(200, json={"data": [], "meta": {"cursor": "same"}})

    reader = reader_for(handler, offline_settings)
    with pytest.raises(ValueError):
        _ = [item async for item in reader.iter_observations(start=None, end=None, observation_type=None)]
    with pytest.raises(costs_module().ObservationReadError):
        _ = [item async for item in reader.iter_observations(start=START, end=END, observation_type="GENERATION")]


def test_api_rejects_invalid_windows_and_external_endpoint(monkeypatch):
    """API不允许无界或外部端点输入。参数：替换夹具；返回：None；异常：断言失败。"""
    costs_module()
    from app.main import create_app
    with TestClient(create_app()) as client:
        for params in [{}, {"from": "2026-10-30", "to": "2026-10-31"},
            {"from": END.isoformat(), "to": START.isoformat()},
            {"from": "2026-01-01T00:00:00Z", "to": END.isoformat()},
            {"from": START.isoformat(), "to": END.isoformat(), "base_url": "https://external.invalid"}]:
            assert client.get("/api/observability/costs", params=params).status_code == 422


def test_api_returns_503_with_incomplete_coverage(monkeypatch):
    """平台未启用时明确不可用。参数：替换夹具；返回：None；异常：断言失败。"""
    costs_module()
    from app.main import create_app
    with TestClient(create_app()) as client:
        response = client.get("/api/observability/costs", params={"from": START.isoformat(), "to": END.isoformat()})
    assert response.status_code == 503
    assert response.json()["coverage"]["complete"] is False
    assert response.json()["groups"] == []
