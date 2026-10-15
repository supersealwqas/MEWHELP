"""应用全局配置管理模块。

使用 pydantic-settings 统一从系统环境变量或根目录下的 .env 文件中加载配置，
集中管理上游大模型连接凭证、Token 裁剪预算以及预留的向量与重排配置。
"""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局应用配置类。

    提供单进程 FastAPI 运行时所需的所有外部环境变量映射与默认值。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 对话模型配置项
    chat_base_url: str = Field(
        min_length=1,
        description="对话模型上游兼容端点地址（支持 OpenAI 协议）",
    )
    chat_model: str = Field(
        min_length=1,
        description="上游对话模型的真实模型名称，换模型需与上游提供方完全一致",
    )
    chat_api_key: str = Field(
        min_length=1,
        description="上游对话模型调用的 API 鉴权密钥",
    )
    token_budget: int = Field(
        default=2000,
        gt=0,
        # 优先使用参考项目的名称，同时兼容已有 .env 中的旧配置键。
        validation_alias=AliasChoices("token_budget", "chat_token_budget"),
        description="多轮会话上下文裁剪保留的 Token 上限预算",
    )

    # 预留向量嵌入模型配置
    embed_base_url: str | None = Field(
        default=None,
        description="向量模型上游端点地址",
    )
    embed_model: str | None = Field(
        default=None,
        description="向量嵌入模型名称",
    )
    embed_api_key: str | None = Field(
        default=None,
        description="向量嵌入模型鉴权密钥",
    )

    # 预留重排模型配置
    rerank_base_url: str | None = Field(
        default=None,
        description="重排模型上游端点地址",
    )
    rerank_model: str | None = Field(
        default=None,
        description="重排模型名称",
    )
    rerank_api_key: str | None = Field(
        default=None,
        description="重排模型鉴权密钥",
    )


# 单例配置实例，方便在整个应用生命周期中直接导入使用
settings = Settings()
