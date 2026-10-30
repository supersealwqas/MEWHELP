"""本机 Langfuse 生命周期与任务局部追踪，不把 SDK 对象写入工作流状态。

主要接口：TraceIdentity 保存原轮身份；ObservabilityRuntime 统一挂载回调、创建执行根、
立即记录意图并有界关闭；observe 给非 Runnable 工具、检索和精排提供同一观测边界。
"""

import asyncio
import base64
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from time import monotonic
from typing import Any
from urllib.parse import urlparse

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from app.schemas.tools import ToolContext

logger = logging.getLogger(__name__)
_active: ContextVar[Any] = ContextVar("mewhelp_observability_execution", default=None)
_manual: ContextVar[Any] = ContextVar("mewhelp_observability_boundary", default=None)


@dataclass
class TraceIdentity:
    """可序列化原轮身份；首次执行填入根 ID，之后恢复只能引用该根。"""

    trace_id: str  # 服务端确定的 32 位十六进制追踪 ID。
    session_id: str  # 可信会话键。
    turn_id: str  # 可信原始轮次键。
    user_id: str  # 可信用户键。
    root_observation_id: str | None = None  # 首次执行根的 16 位十六进制 ID。


@dataclass
class _Execution:
    """仅在当前请求及其子任务共享的活跃执行；禁止放入 State。"""

    runtime: Any  # 生命周期运行时。
    root: Any  # 仍存活的 SDK 根观测。
    metadata: dict = field(default_factory=dict)  # 同轮子任务读取的最新意图标签。


@dataclass
class _Boundary:
    """非Runnable的活跃父边界，用于把内部模型和后续检索放回真实工具之下。"""

    observation: Any  # Langfuse观测包装器。
    otel_span: Any  # 当前仍存活的OTel span。
    enclosing_id: str  # 建立此边界前的Runnable父观测ID。


class _Observation:
    """隔离观测更新失败，避免导出问题覆盖业务事务或取消。"""

    def __init__(self, runtime=None, wrapped=None):
        """保存观测边界。参数：runtime运行时/wrapped SDK对象；返回：None；异常：无。"""
        self.runtime, self.wrapped = runtime, wrapped

    def update(self, *, output=None, error=None, metadata=None, **kwargs):
        """更新当前观测。参数：输出/错误/元数据及SDK字段；返回：自身；异常：观测错误只记健康状态。"""
        if self.wrapped is not None:
            try:
                if error is not None:
                    if isinstance(error, GraphInterrupt):
                        kwargs.update(level="DEFAULT", status_message="paused")
                    elif isinstance(error, (asyncio.CancelledError, GeneratorExit)):
                        kwargs.update(level="WARNING", status_message="cancelled")
                    else:
                        kwargs.update(level="ERROR", status_message=type(error).__name__)
                if output is not None:
                    kwargs["output"] = output
                if metadata is not None:
                    kwargs["metadata"] = metadata
                self.wrapped.update(**kwargs)
            except Exception as failure:
                self.runtime._failure("instrumentation_failed", failure)
        return self


def _reported_usage(response) -> dict:
    """只读上游原始用量，区分零值与缺失。参数：LLMResult或None；返回：来源标签；异常：无。"""
    candidates = []
    if response is not None:
        output = response.llm_output or {}
        candidates.extend([output.get("token_usage"), output.get("usage")])
        for group in response.generations:
            for generation in group:
                message = getattr(generation, "message", None)
                if message is not None:
                    candidates.append(getattr(message, "usage_metadata", None))
                    details = getattr(message, "response_metadata", {}) or {}
                    candidates.extend([details.get("token_usage"), details.get("usage")])
                candidates.append((generation.generation_info or {}).get("usage_metadata"))
    found = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for direction, names in {"input": ("input_tokens", "prompt_tokens", "input"),
                                 "output": ("output_tokens", "completion_tokens", "output")}.items():
            for name in names:
                value = candidate.get(name)
                if type(value) is int and value >= 0:
                    found[direction] = value
                    break
    return {"usage_origin": "model_reported" if len(found) == 2 else "inferred_or_unknown",
            "input_usage_present": "input" in found, "output_usage_present": "output" in found,
            **{f"reported_{key}_tokens": value for key, value in found.items()}}


class _UsageCallbackHandler(CallbackHandler):
    """沿用 SDK 唯一 generation，仅在结束前补原始用量来源和当前意图。"""

    run_inline = True

    def _get_parent_observation(self, parent_run_id):
        """保留SDK子树并补非Runnable父级。参数：LangChain父运行ID；返回：实际父观测；异常：SDK接口错误。"""
        parent = super()._get_parent_observation(parent_run_id)
        boundary = _manual.get()
        # SDK默认只看LangChain父运行，会跨过工具执行器的手工span；只替换进入手工边界前的父级。
        # 已经进入该边界的MCP/Runnable子树仍按其原父运行组织，不展平、不重复生成generation。
        if boundary is not None and (isinstance(parent, Langfuse) or getattr(parent, "id", None) == boundary.enclosing_id):
            return boundary.observation
        return parent

    def _metadata(self, kwargs):
        """传播本次执行最新分类。参数：回调参数；返回：独立参数；异常：无。"""
        execution = _active.get()
        return {**kwargs, "metadata": {**(kwargs.get("metadata") or {}),
                                      **(execution.metadata if execution else {})}}

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """记录图及节点输入。参数：SDK链回调字段；返回：SDK结果；异常：SDK自行隔离。"""
        values = self._metadata(kwargs)
        if kwargs.get("parent_run_id") is None:
            # SDK会把顶层链metadata再次传播为trace属性；可变分类只放活跃执行根和子观测，避免旧值覆盖根。
            for key in ("intent", "intent_confidence", "route"):
                values["metadata"].pop(key, None)
        return super().on_chain_start(serialized, inputs, run_id=run_id, **values)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        """记录模型完整输入及工具定义。参数：SDK模型回调字段；返回：SDK结果；异常：SDK自行隔离。"""
        return super().on_chat_model_start(serialized, messages, run_id=run_id, **self._metadata(kwargs))

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        """支持非聊天模型的相同归属。参数：SDK文本模型回调字段；返回：SDK结果；异常：SDK自行隔离。"""
        return super().on_llm_start(serialized, prompts, run_id=run_id, **self._metadata(kwargs))

    def _mark_usage(self, run_id, response):
        """结束前标记已有 generation。参数：运行ID/原响应；返回：None；异常：失败只留日志。"""
        try:
            # 固定 SDK 4.15.4 的回调没有公开 pre-end 钩子；此映射已按官方源码核验并由真实导出测试保护。
            generation = self._runs.get(run_id)
            if generation is not None:
                generation.update(metadata=_reported_usage(response))
        except Exception as error:
            logger.warning("observability usage_annotation_failed type=%s", type(error).__name__)

    def on_llm_end(self, response, *, run_id, **kwargs):
        """保留上游计量的存在性。参数：LLMResult/运行ID/SDK字段；返回：SDK结果；异常：SDK自行隔离。"""
        self._mark_usage(run_id, response)
        return super().on_llm_end(response, run_id=run_id, **kwargs)

    def on_llm_error(self, error, *, run_id, **kwargs):
        """错误调用的缺失用量不补零。参数：错误/运行ID/SDK字段；返回：SDK结果；异常：SDK自行隔离。"""
        self._mark_usage(run_id, kwargs.get("response"))
        return super().on_llm_error(error, run_id=run_id, **kwargs)


class _HealthExporter(SpanExporter):
    """包装真实 OTLP 导出结果，显式暴露平台离线或导出失败。"""

    def __init__(self, runtime, exporter):
        """参数：运行时/真实外部导出器；返回：None；异常：无。"""
        self.runtime, self.exporter = runtime, exporter

    def export(self, spans):
        """发送一批真实观测。参数：OTel spans；返回：SpanExportResult；异常：导出错误转失败状态。"""
        try:
            result = self.exporter.export(spans)
            if result != SpanExportResult.SUCCESS:
                self.runtime._failure("export_failed")
            return result
        except Exception as error:
            self.runtime._failure("export_failed", error)
            return SpanExportResult.FAILURE

    def shutdown(self):
        """关闭外部导出器。参数：无；返回：None；异常：由有界关闭线程隔离。"""
        self.exporter.shutdown()


class ObservabilityRuntime:
    """应用级唯一客户端/回调；身份与意图属于任务局部执行。"""

    def __init__(self, settings, *, span_exporter=None):
        """建立本机资源。参数：Settings/可替换OTel导出端；返回：None；异常：非本机地址或缺密钥。"""
        self.enabled = settings.observability_enabled
        self.timeout = settings.observability_shutdown_timeout_seconds
        self.health = {"status": "ready" if self.enabled else "disabled", "last_error_type": None}
        self.client = self.tracer_provider = None
        self.callback_handler = BaseCallbackHandler()
        self._closing = False
        if not self.enabled:
            return
        endpoint = settings.langfuse_base_url.rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
            raise ValueError("Langfuse 追踪端点必须是本机回环地址")
        public, secret = settings.langfuse_public_key, settings.langfuse_secret_key.get_secret_value()
        if not public or not secret:
            raise ValueError("启用观测需要本机项目的公钥与密钥")
        # 自有 provider 不注册无界 atexit；SDK.shutdown 会先注销其自身 atexit 再 flush。
        self.tracer_provider = TracerProvider(sampler=ALWAYS_ON, shutdown_on_exit=False)
        exporter = span_exporter if span_exporter is not None else OTLPSpanExporter(
            endpoint=f"{endpoint}/api/public/otel/v1/traces", timeout=3,
            headers={"Authorization": "Basic " + base64.b64encode(f"{public}:{secret}".encode()).decode(),
                     "x-langfuse-sdk-name": "python", "x-langfuse-sdk-version": "4.15.4",
                     "x-langfuse-public-key": public, "x-langfuse-ingestion-version": "4"})
        self.client = Langfuse(public_key=public, secret_key=secret, base_url=endpoint, sample_rate=1.0,
            timeout=3, tracer_provider=self.tracer_provider, span_exporter=_HealthExporter(self, exporter))
        self.callback_handler = _UsageCallbackHandler(public_key=public)

    def _failure(self, status, error=None):
        """标记导出健康状态，日志不输出凭据/请求。参数：状态/异常；返回：None；异常：无。"""
        self.health.update(status=status, last_error_type=type(error).__name__ if error else None)
        logger.warning("observability %s type=%s", status, self.health["last_error_type"])

    def identity_for(self, context: ToolContext) -> TraceIdentity:
        """由可信原轮稳定派生身份。参数：服务端ToolContext；返回：TraceIdentity；异常：无。"""
        value = "\0".join(("mewhelp:ch09", context.session_id, context.user_id, context.turn_id))
        return TraceIdentity(sha256(value.encode()).hexdigest()[:32], context.session_id, context.turn_id, context.user_id)

    @contextmanager
    def execution(self, identity: TraceIdentity, *, kind: str):
        """创建当前执行根，恢复链接最初根。参数：身份/kind执行或后台类别；返回：观测上下文；异常：业务异常原样传播。"""
        details = {"turn_id": identity.turn_id, "task_kind": kind if kind in {"summary", "flywheel", "evaluation"} else "chat",
                   "execution_kind": kind, "intent": "unclassified"}
        if not self.enabled:
            yield _Observation()
            return
        trace_context = {"trace_id": identity.trace_id}
        if identity.root_observation_id:
            trace_context["parent_span_id"] = identity.root_observation_id
        with self.client.start_as_current_observation(name=f"execution.{kind}", as_type="span",
                trace_context=trace_context, metadata=details) as root:
            if identity.root_observation_id is None:
                identity.root_observation_id = root.id
            current = _Execution(self, _Observation(self, root), details)
            token = _active.set(current)
            boundary_token = _manual.set(None)
            try:
                with propagate_attributes(session_id=identity.session_id, user_id=identity.user_id,
                        trace_name="MewHelp 原轮", metadata={key: value for key, value in details.items() if key != "intent"},
                        as_baggage=False):
                    # 只传播稳定身份：服务端trace元数据会覆盖observation元数据，不能传播待更新的unclassified占位。
                    try:
                        yield current.root
                    except BaseException as error:
                        status = "cancelled" if isinstance(error, (asyncio.CancelledError, GeneratorExit)) else "failed"
                        current.root.update(error=error, metadata={**current.metadata, "execution_status": status})
                        raise
            finally:
                _manual.reset(boundary_token)
                _active.reset(token)

    def record_intent(self, intent: str, confidence: float, route: str) -> None:
        """分类成功立即更新仍存活根及后续调用标签。参数：意图/置信度/路由；返回：None；异常：观测错误隔离。"""
        current = _active.get()
        if current is not None and current.runtime is self:
            current.metadata.update(intent=intent, intent_confidence=str(confidence), route=route)
            current.root.update(metadata=current.metadata)

    @contextmanager
    def observe(self, name: str, *, kind: str, input: dict):
        """为非Runnable操作添加一次观测。参数：名称/类型/真实输入；返回：可更新观测；异常：业务错误透传。"""
        current = _active.get()
        if not self.enabled or current is None:
            yield _Observation()
            return
        parent = otel_trace.get_current_span()
        boundary = _manual.get()
        # 流式模型结束回调可能在另一任务关闭上下文；不能把后续检索挂在已结束的generation上。
        if not parent.is_recording() and boundary is not None:
            parent = boundary.otel_span
        with otel_trace.use_span(parent, end_on_exit=False, record_exception=False, set_status_on_exception=False):
            enclosing_id = f"{parent.get_span_context().span_id:016x}"
            with self.client.start_as_current_observation(name=name, as_type=kind, input=input,
                    metadata=current.metadata) as wrapped:
                observation = _Observation(self, wrapped)
                token = _manual.set(_Boundary(wrapped, otel_trace.get_current_span(), enclosing_id))
                try:
                    yield observation
                except BaseException as error:
                    observation.update(error=error)
                    raise
                finally:
                    _manual.reset(token)

    async def aclose(self) -> None:
        """将SDK阻塞关闭移出事件循环并有界等待。参数：无；返回：None；异常：关闭失败记状态，不覆盖事务。"""
        if not self.enabled or self._closing:
            return
        self._closing = True
        done = threading.Event()

        def shutdown():
            """守护线程调用真实SDK关闭。参数：无；返回：None；异常：失败只更新健康状态。"""
            try:
                # shutdown本身会flush；不能先调用flush，否则其阻塞会留下SDK无界atexit。
                self.client.shutdown()
                self.tracer_provider.shutdown()
            except Exception as error:
                self._failure("export_failed", error)
            finally:
                done.set()

        threading.Thread(target=shutdown, name="mewhelp-observability-shutdown", daemon=True).start()
        deadline = monotonic() + self.timeout
        while not done.is_set() and monotonic() < deadline:
            await asyncio.sleep(min(0.02, max(0, deadline - monotonic())))
        if not done.is_set():
            # Python无法强制杀线程；明确标记未完成导出，进程无需等待该守护线程退出。
            self._failure("shutdown_timeout")


@contextmanager
def observe(name: str, *, kind: str, input: dict):
    """使用当前任务运行时的统一边界。参数：名称/类型/实际输入；返回：观测；异常：业务错误透传。"""
    current = _active.get()
    if current is None:
        yield _Observation()
    else:
        with current.runtime.observe(name, kind=kind, input=input) as observation:
            yield observation
