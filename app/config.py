"""应用全局配置管理模块。

使用 pydantic-settings 统一从系统环境变量或根目录下的 .env 文件中加载配置，
集中管理上游大模型连接凭证、Token 裁剪预算以及预留的向量与重排配置。
"""

from pydantic import AliasChoices, Field, SecretStr
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
        default=12000,
        gt=0,
        # 优先使用参考项目的名称，同时兼容已有 .env 中的旧配置键。
        validation_alias=AliasChoices("token_budget", "chat_token_budget"),
        description="对话消息估算预算，CH03 起默认 12000 以容纳三条知识事实，不含系统提示及工具定义，并非模型真实 Token 上限",
    )
    request_token_budget: int = Field(
        default=16000,
        gt=0,
        description="完整输入估算预算，包含系统提示、对话和本阶段工具定义，不含模型输出，并非真实上下文窗口",
    )

    # 本章 MySQL 持久化及工具的有限执行配置。
    mysql_host: str = Field(default="127.0.0.1", description="MySQL 主机地址")
    mysql_port: int = Field(default=3307, ge=1, le=65535, description="MySQL 本机映射端口")
    mysql_database: str = Field(default="mewhelp", description="应用数据库名称")
    mysql_test_database: str = Field(default="mewhelp_test", description="隔离测试库名称")
    mysql_user: str = Field(default="mewhelp", description="非 root 的应用数据库账号")
    mysql_password: SecretStr = Field(default=SecretStr(""), description="应用数据库密码，日志中隐藏")
    tool_timeout_seconds: float = Field(default=5, gt=0, le=60, description="单次工具执行超时秒数")
    tool_max_retries: int = Field(default=1, ge=0, le=1, description="工具最大故障重试次数")
    tool_retry_delay_seconds: float = Field(default=0.2, ge=0, le=5, description="可取消的重试等待秒数")
    tool_max_calls_per_turn: int = Field(
        default=4, ge=1, le=8,
        description="一条用户消息内最多执行的业务工具数，包含同批次及连续决策；执行器内部重试不重复计数",
    )

    # CH03 知识库与 Dense 检索配置。模型与向量库名称是本章固定选型，不在此处提供降级开关。
    knowledge_model_path: str = Field(
        default="models/bge-m3",
        min_length=1,
        description="本地 BGE-M3 模型目录，必须包含权重与 tokenizer，不在请求中联网下载",
    )
    milvus_uri: str = Field(
        default="http://127.0.0.1:19530",
        min_length=1,
        description="Milvus 实例连接地址；运行基线见部署手册，不自动升级或清理原项目资源",
    )
    milvus_database: str = Field(
        default="mewhelp_ch03",
        min_length=1,
        description="本章独立 Milvus 数据库，隔离于原项目默认库",
    )
    milvus_collection: str = Field(
        default="knowledge",
        min_length=1,
        description="本章 Milvus 集合名，初始化前必须校验所有权与 schema",
    )
    milvus_token: SecretStr = Field(
        default=SecretStr(""),
        description="Milvus 鉴权令牌；本地无鉴权时为空，日志与报告中隐藏",
    )
    knowledge_test_database: str = Field(
        default="mewhelp_ch03_test",
        min_length=1,
        description="CH03 真实集成测试专用 MySQL 数据库，不重建 CH02 测试库",
    )
    milvus_test_database: str = Field(
        default="mewhelp_ch03_test",
        min_length=1,
        description="CH03 真实集成测试专用 Milvus 数据库",
    )
    knowledge_policy_path: str = Field(
        default="data/knowledge/retrieval-policy.json",
        min_length=1,
        description="在线 Dense 检索的校准策略文件路径",
    )
    knowledge_relation_policy_path: str = Field(
        default="data/knowledge/relation-policy.json",
        min_length=1,
        description="离线知识关系判定的校准策略文件路径",
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
