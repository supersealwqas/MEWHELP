"""售后诉求信息结构化提取 API 路由模块。

本模块提供基于大语言模型 Function Calling 能力的结构化信息提取端点，
将用户提交的售后问题描述解析为标准的 AfterSalesTicket 数据模型。
主要包含接口：
- POST /api/extract: 提取订单号、诉求分类及期望解决措施。
"""

from fastapi import APIRouter, HTTPException

from app.core.llm import get_llm
from app.core.prompts import EXTRACTION_PROMPT
from app.schemas.after_sales import AfterSalesTicket, ExtractRequest

# 注册售后提取 API 路由器，统一前缀为 /api，分组标签为 extract
router = APIRouter(prefix="/api", tags=["extract"])

# 暴露别名以提升导入语义清晰度
extract_router = router


@router.post("/extract", response_model=AfterSalesTicket, summary="售后信息结构化提取")
async def extract_after_sales_info(request: ExtractRequest) -> AfterSalesTicket:
    """从用户输入的自然语言售后描述中，结构化提取售后工单信息。

    利用绑定了 AfterSalesTicket 数据契约的大语言模型 Function Calling 能力，
    从原始文本中精准抽取订单号（无则为 None，不可臆造）、诉求类型（退款/换货/维修/投诉/其他）
    以及核心期望解决方案。

    参数:
        request (ExtractRequest): 包含用户待提取售后文本的请求体模型，要求 text 非空且非纯空白字符。

    返回值:
        AfterSalesTicket: 结构化解析后的工单对象，包含 order_id、request_type 和 expected_solution 字段。

    异常:
        fastapi.HTTPException: 当上游模型调用超时、网络故障或模型解析失败时，抛出 502 Bad Gateway 异常。
    """
    try:
        # 初始化确定性采样温度 (0.0) 的 LLM 实例
        llm = get_llm(temperature=0.0)
        # 将模型输出强绑定为售后工单结构化数据契约 (采用 function_calling 机制)
        structured_llm = llm.with_structured_output(AfterSalesTicket, method="function_calling")
        # 编排 Prompt 模板与结构化输出链
        chain = EXTRACTION_PROMPT | structured_llm
        # 异步执行抽取链路并返回结构化数据
        ticket: AfterSalesTicket = await chain.ainvoke({"text": request.text})
        return ticket
    except HTTPException:
        # 透传已显式抛出的 HTTP 业务异常，避免被外层重复包装
        raise
    except Exception as exc:
        # 捕获上游模型网络中断、超时、解析失败等底层异常，统一封装为 HTTP 502 错误对外暴露
        raise HTTPException(
            status_code=502,
            detail=f"上游模型提取失败: {str(exc)}",
        ) from exc
