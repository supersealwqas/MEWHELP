"""从本机 Langfuse v2 完整读取观测，按原轮意图汇总模型原始上报的用量。"""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from urllib.parse import urlparse

import httpx

from app.config import Settings, settings


class ObservationReadError(RuntimeError):
    """分页、归属或平台配置不可用；错误消息不包含上游响应或鉴权材料。"""


def validate_window(start: datetime, end: datetime, max_days: int) -> None:
    """校验有限半开窗口。参数：带时区起止时间/最大天数；返回：None；异常：ValueError。"""
    if (start is None or end is None or start.utcoffset() is None or end.utcoffset() is None
            or start >= end or end - start > timedelta(days=max_days)):
        raise ValueError(f"须提供带时区的起止时间，起点早于终点且跨度不超过{max_days}天")


class LangfuseObservationReader:
    """只从服务端配置的本机端点读取原始HTTP字段，负责游标完整性和有界网络等待。"""

    def __init__(self, config: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        """保存读取配置。参数：应用配置/可选HTTP传输；返回：None；异常：无。"""
        self.config = config
        self.transport = transport

    async def iter_observations(self, *, start: datetime | None, end: datetime | None,
            observation_type: str | None, trace_id: str | None = None) -> AsyncIterator[dict]:
        """遍历全部页。参数：时间窗/类型/可选原轮ID，无窗须限定原轮；返回：异步dict流；异常：ValueError或ObservationReadError。"""
        if start is None and end is None:
            if not trace_id:
                raise ValueError("无时间窗只允许按原轮补查归属")
        else:
            validate_window(start, end, self.config.observability_cost_max_days)
        endpoint = self.config.langfuse_base_url.rstrip("/")
        parsed = urlparse(endpoint)
        if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ObservationReadError("观测读取端点必须是本机回环地址")
        public = self.config.langfuse_public_key
        secret = self.config.langfuse_secret_key.get_secret_value()
        if not self.config.observability_enabled or not public or not secret:
            raise ObservationReadError("本机观测未启用或未配置")
        params = {"fields": "core,basic,time,metadata,model,usage,trace_context",
            "limit": self.config.observability_cost_page_size}
        if start is not None:
            params.update(fromStartTime=start.isoformat(), toStartTime=end.isoformat())
        if observation_type is not None:
            params["type"] = observation_type
        if trace_id is not None:
            params["traceId"] = trace_id
        seen_cursors = set()
        try:
            # 禁止重定向和环境代理，避免本机凭据被转发到其他端点。
            async with httpx.AsyncClient(auth=(public, secret), transport=self.transport,
                    timeout=self.config.observability_cost_request_timeout_seconds,
                    follow_redirects=False, trust_env=False) as client:
                while True:
                    response = await client.get(f"{endpoint}/api/public/v2/observations", params=params)
                    response.raise_for_status()
                    page = response.json()
                    if not isinstance(page, dict) or not isinstance(page.get("data"), list) or not isinstance(page.get("meta"), dict):
                        raise ObservationReadError("观测分页响应格式不完整")
                    for row in page["data"]:
                        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                                or not isinstance(row.get("traceId"), str) or not row["traceId"]
                                or not isinstance(row.get("type"), str)
                                or (row.get("metadata") is not None and not isinstance(row["metadata"], dict))):
                            raise ObservationReadError("观测记录缺少可靠身份或元数据")
                        yield row
                    cursor = page["meta"].get("cursor")
                    if cursor is None:
                        break
                    if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                        raise ObservationReadError("观测分页游标无效或重复")
                    seen_cursors.add(cursor)
                    params["cursor"] = cursor
        except (httpx.HTTPError, ValueError):
            # 只暴露固定状态，不能把含鉴权或上游内容的异常文本带入账单。
            raise ObservationReadError("观测平台读取失败") from None


def _reported_tokens(metadata: dict, direction: str) -> int | None:
    """提取原始计量的已知方向。参数：元数据/input或output；返回：非负整数或None；异常：无。"""
    value = metadata.get(f"reported_{direction}_tokens")
    if (metadata.get("usage_origin") in {"model_reported", "inferred_or_unknown"}
            and metadata.get(f"{direction}_usage_present") is True and type(value) is int and value >= 0):
        return value
    return None


def summarize_intent_usage(observations: list[dict]) -> dict:
    """按原轮去重汇总已知用量。参数：原始观测列表含归属SPAN；返回：coverage/groups；异常：无。"""
    unique = {row["id"]: row for row in observations}
    traces = {}
    # 最初聊天根优先，恢复根上的unclassified占位不能覆盖已有意图。
    ordered = sorted(unique.values(), key=lambda row: (
        row.get("name") != "execution.chat", not str(row.get("name", "")).startswith("execution."),
        row.get("startTime", ""), row["id"]))
    for row in ordered:
        metadata = row.get("metadata") or {}
        trace = traces.setdefault(row.get("traceId"), {})
        if metadata.get("task_kind"):
            trace.setdefault("task_kind", metadata["task_kind"])
        intent = metadata.get("intent")
        if isinstance(intent, str) and intent and intent != "unclassified":
            trace.setdefault("intent", intent)
    groups, requests = {}, {}
    fully_reported = model_calls = total_known = 0
    for row in unique.values():
        if row.get("type") != "GENERATION":
            continue
        model_calls += 1
        metadata = row.get("metadata") or {}
        trace = traces.get(row.get("traceId"), {})
        task_kind = metadata.get("task_kind") or trace.get("task_kind") or "chat"
        intent = trace.get("intent", "unclassified") if task_kind == "chat" else None
        key = (task_kind, intent)
        group = groups.setdefault(key, {"task_kind": task_kind, "intent": intent, "request_count": 0,
            "model_call_count": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "unknown_usage_count": 0, "known_token_share": None, "cost": None})
        requests.setdefault(key, set()).add(row.get("traceId") or row["id"])
        group["model_call_count"] += 1
        input_tokens, output_tokens = _reported_tokens(metadata, "input"), _reported_tokens(metadata, "output")
        if input_tokens is None or output_tokens is None:
            group["unknown_usage_count"] += 1
        else:
            fully_reported += 1
        # 0仅是已知部分的加法单位；缺方仍由unknown_usage_count及coverage明确表达。
        group["input_tokens"] += input_tokens if input_tokens is not None else 0
        group["output_tokens"] += output_tokens if output_tokens is not None else 0
        known = (input_tokens or 0) + (output_tokens or 0)
        group["total_tokens"] += known
        total_known += known
    for key, group in groups.items():
        group["request_count"] = len(requests[key])
        group["known_token_share"] = group["total_tokens"] / total_known if total_known else None
    return {"coverage": {"complete": True, "model_call_count": model_calls,
        "fully_reported_call_count": fully_reported, "unknown_usage_count": model_calls - fully_reported,
        "usage_coverage": fully_reported / model_calls if model_calls else None,
        "known_tokens": total_known, "token_basis": "model_reported_known_parts",
        "known_token_share_basis": "all_groups_known_tokens", "cost_basis": "unavailable"},
        "groups": sorted(groups.values(), key=lambda group: (group["task_kind"], group["intent"] or ""))}


async def get_intent_usage(start: datetime, end: datetime, *, reader: LangfuseObservationReader | None = None) -> dict:
    """取得有限窗口完整账单。参数：带时区起止时间/可选reader；返回：汇总或显式不完整状态；异常：ValueError。"""
    reader = reader or LangfuseObservationReader(settings)
    validate_window(start, end, reader.config.observability_cost_max_days)
    generations, attribution = {}, []
    stage = "page"
    try:
        async with asyncio.timeout(reader.config.observability_cost_timeout_seconds):
            async for row in reader.iter_observations(start=start, end=end, observation_type="GENERATION"):
                if row["type"] == "GENERATION":
                    generations[row["id"]] = row
            stage = "attribution"
            for trace_id in sorted({row["traceId"] for row in generations.values()}):
                async for row in reader.iter_observations(start=None, end=None, observation_type="SPAN", trace_id=trace_id):
                    # 补查只提供归属，不能把窗外generation添进当前窗口的用量。
                    if row["type"] != "GENERATION" and row["traceId"] == trace_id:
                        attribution.append(row)
    except (ObservationReadError, TimeoutError):
        return {"coverage": {"complete": False, "failure_stage": stage,
            "collected_model_call_count": len(generations), "message": "观测读取未完整完成，请稍后重试"}, "groups": []}
    result = summarize_intent_usage([*generations.values(), *attribution])
    result["coverage"].update({"from": start.isoformat(), "to": end.isoformat(), "window": "[from,to)"})
    return result
