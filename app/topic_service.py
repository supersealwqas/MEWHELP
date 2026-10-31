"""独立主题分类HTTP服务：生命周期加载一次，本模块导入不加载torch或聊天配置。"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from starlette.concurrency import run_in_threadpool


class ClassifyRequest(BaseModel):
    """仅接收有限长度文本，不允许客户端控制模型目录或设备。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    texts: list[Annotated[str, StringConstraints(min_length=1, max_length=2000)]] = Field(
        min_length=1, max_length=64, description="待分类文本，1至64条，每条1至2000字符")

    @field_validator("texts")
    @classmethod
    def reject_blank(cls, texts):
        """拒绝仅空白问题。参数：texts已验证文本；返回：原数组；异常：ValueError。"""
        if any(not text.strip() for text in texts):
            raise ValueError("文本不能为空白")
        return texts


def _configured_predictor():
    """只从进程环境构建模型。参数：无；返回：预测器或未配置None；异常：模型校验/加载错误。"""
    directory = os.environ.get("TOPIC_MODEL_DIR")
    if not directory:
        return None
    from app.topics.inference import TopicPredictor
    return TopicPredictor(Path(directory), device=os.environ.get("TOPIC_DEVICE", "cpu"))


def create_app(predictor_factory=None) -> FastAPI:
    """创建有界并发独立服务。参数：可选零参数预测器工厂；返回：FastAPI；异常：启动时统一模型加载错误。"""
    factory = predictor_factory or _configured_predictor

    @asynccontextmanager
    async def lifespan(application):
        """进程启动加载模型一次。参数：application应用；返回：生命周期；异常：无敏感信息RuntimeError。"""
        try:
            application.state.predictor = await run_in_threadpool(factory)
        except Exception:
            # 不向客户端或启动异常链泄露目录、文本或底层库错误内容。
            raise RuntimeError("主题模型加载失败，请检查本地模型包") from None
        application.state.inference_slot = BoundedSemaphore(1)
        yield
        application.state.predictor = None

    application = FastAPI(title="CH10 主题分类旁路服务", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_input(request, error):
        """隐藏Pydantic默认错误中的原始输入。参数：request请求、error校验错误；返回：422响应；异常：无。"""
        return JSONResponse(status_code=422, content={"detail": "texts必须为1至64条非空文本，每条最多2000字符"})

    @application.get("/health")
    async def health(request: Request):
        """报告就绪状态和模型身份。参数：request请求；返回：只读健康摘要；异常：未就绪HTTP503。"""
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, "主题模型尚未配置")
        metadata = predictor.metadata
        return dict(status="ready", model_version=metadata["version"], taxonomy_digest=metadata["taxonomy_digest"])

    @application.post("/classify")
    async def classify(body: ClassifyRequest, request: Request):
        """在一个计算槽中分类，不积累无界等待队列。参数：body文本、request请求；返回：模型身份和结果；异常：503/500。"""
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, "主题模型尚未配置")
        slot = request.app.state.inference_slot
        if not slot.acquire(blocking=False):
            raise HTTPException(503, "主题分类服务繁忙，请稍后重试")
        try:
            results = await run_in_threadpool(predictor.predict, body.texts)
            metadata = predictor.metadata
            return dict(model_version=metadata["version"], taxonomy_digest=metadata["taxonomy_digest"], results=results)
        except Exception:
            raise HTTPException(500, "主题分类失败") from None
        finally:
            slot.release()

    return application


app = create_app()
