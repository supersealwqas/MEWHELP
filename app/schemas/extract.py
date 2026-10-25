"""售后信息提取相关的 Pydantic 数据契约与结构化模型模块。

包含向售后抽取接口发起的请求入参模型 (ExtractRequest)，以及与 LangChain `with_structured_output`
强绑定的售后工单抽取结构模型 (AfterSalesTicket)，规范提取字段与诉求分类枚举。
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ExtractRequest(BaseModel):
    """售后信息提取接口请求体模型。

    封装用户输入的原始自然语言文本，供大模型进行信息结构化解析。
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户描述售后问题的自然语言文本，最多4000个Unicode码点，不可为空或纯空白字符",
    )

    @field_validator("text")
    @classmethod
    def validate_not_whitespace(cls, value: str) -> str:
        """校验待提取文本不能为纯空白字符。

        参数:
            value (str): 待校验的自然语言文本输入

        返回值:
            str: 校验通过的原始文本

        异常:
            ValueError: 当输入文本为空或仅包含空白字符时抛出
        """
        # 校验去除空白后的有效内容，避免无意义的空白推理请求
        if not value or not value.strip():
            raise ValueError("text 不能为空或纯空白字符")
        return value


class RequestType(str, Enum):
    """售后诉求类型枚举，与参考项目统一五种分类及其中文取值。"""

    REFUND = "退款"  # 退货退款或仅退款诉求
    EXCHANGE = "换货"  # 更换商品或规格诉求
    REPAIR = "维修"  # 商品故障维修诉求
    COMPLAINT = "投诉"  # 服务或商品投诉诉求
    OTHER = "其他"  # 其他售后诉求


class AfterSalesTicket(BaseModel):
    """售后工单结构化数据模型。

    用于 LangChain function calling 结构化输出，严格提取订单号、诉求类型与期望方案。
    若用户未提供订单号，必须严格保持为 None，严禁虚构。
    """

    order_id: str | None = Field(
        default=None,
        description="订单号，若用户未提供必须为 None，不可臆造",
    )
    request_type: RequestType = Field(
        ...,
        description="诉求类型枚举：仅支持退款、换货、维修、投诉、其他",
    )
    expected_solution: str = Field(
        ...,
        min_length=1,
        description="用户的核心诉求或期望解决方案简述",
    )

    @field_validator("order_id", mode="before")
    @classmethod
    def _normalize_missing_order_id(cls, value: object) -> object:
        """将模型用来表示缺失订单号的占位文本归一化为空值。

        参数:
            value: 模型返回的订单号原始值；真实订单号保持原样。
        返回值:
            object: 占位文本转换为 None，其他值交由字段类型继续校验。
        异常:
            无业务异常。
        """
        # 只处理明确的缺失标记，避免清洗或重写真实订单号。
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a", "无"}:
            return None
        return value

    @field_validator("expected_solution")
    @classmethod
    def validate_solution_not_whitespace(cls, value: str) -> str:
        """拒绝模型输出的纯空白处理方案。

        参数:
            value: 已完成类型校验的期望方案。
        返回值:
            str: 含有效文本的原始方案。
        异常:
            ValueError: 方案不包含有效文字时抛出。
        """
        if not value.strip():
            raise ValueError("expected_solution 不能为空或纯空白字符")
        return value
