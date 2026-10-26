"""售后诉求信息结构化提取 API 路由模块。

本模块提供基于大语言模型 Function Calling 能力的结构化信息提取端点，
将用户提交的售后问题描述解析为标准的 AfterSalesTicket 数据模型。
主要包含接口：
- POST /api/extract: 提取订单号、诉求分类及期望解决措施。
"""

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.runnables import Runnable

from app.core.llm import get_chat_model
from app.core.prompts import EXTRACT_PROMPT
from app.schemas.extract import AfterSalesTicket, ExtractRequest

# 注册售后提取 API 路由器，统一前缀为 /api，分组标签为 extract
router = APIRouter(prefix="/api", tags=["extract"])
logger = logging.getLogger(__name__)


def get_extractor() -> Runnable:
    """创建售后信息结构化提取链。

    参数:
        无；模型连接信息来自全局配置。

    返回值:
        Runnable: 接收 text 并输出 AfterSalesTicket 的提示词与模型调用链。

    异常:
        pydantic.ValidationError: 模型客户端参数不合法时抛出。
        ValueError: 结构化输出参数不受模型客户端支持时抛出。
    """
    # 使用函数调用结构化协议；由LangChain强制指定结果工具，不能退回auto自由文本。
    # 单次提取显式限制输出额度，足够容纳工单JSON并避免无界生成。
    model = get_chat_model(temperature=0.0, streaming=True, max_tokens=1024)
    structured_model = model.with_structured_output(
        AfterSalesTicket,
        method="function_calling",
        strict=True,
    )
    return EXTRACT_PROMPT | structured_model


@router.post("/extract", response_model=AfterSalesTicket, summary="售后信息结构化提取")
async def extract(req: ExtractRequest) -> AfterSalesTicket:
    """从用户输入的自然语言售后描述中，结构化提取售后工单信息。

    利用绑定了 AfterSalesTicket 数据契约的大语言模型 Function Calling 能力，
    从原始文本中精准抽取订单号（无则为 None，不可臆造）、诉求类型（退款/换货/维修/投诉/其他）
    以及核心期望解决方案。

    参数:
        req (ExtractRequest): 包含用户待提取售后文本的请求体模型，要求 text 为1至4000个Unicode码点且非纯空白。

    返回值:
        AfterSalesTicket: 结构化解析后的工单对象，包含 order_id、request_type 和 expected_solution 字段。

    异常:
        fastapi.HTTPException: 当上游模型调用超时、网络故障或模型解析失败时，抛出 502 Bad Gateway 异常。
    """
    try:
        # 在异常保护范围内创建调用链，客户端初始化失败也继续返回 HTTP 502。
        extractor = get_extractor()
        # 异步执行抽取链路并返回结构化数据
        ticket: AfterSalesTicket = await extractor.ainvoke({"text": req.text})
        # 未返回工具调用时，解析器可能返回 None；在响应校验前转为明确的上游错误。
        if not isinstance(ticket, AfterSalesTicket):
            raise ValueError("上游未返回有效的售后工单")
        return ticket
    except HTTPException:
        # 透传已显式抛出的 HTTP 业务异常，避免被外层重复包装
        raise
    except Exception as exc:
        # 统一返回稳定错误文案，原始上游异常可能包含连接或鉴权信息，日志仅记类型。
        logger.warning("售后模型提取失败：%s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="上游模型暂时不可用，请稍后重试",
        ) from exc
