"""应用全局配置管理模块。

使用 pydantic-settings 统一从系统环境变量或根目录下的 .env 文件中加载配置，
集中管理上游大模型连接凭证、Token 裁剪预算以及预留的向量与重排配置。
"""

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
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
    observability_enabled: bool = Field(default=False, description="显式启用本机 Langfuse；开发验收固定全量采样")
    langfuse_base_url: str = Field(default="http://127.0.0.1:13000", description="只允许本机回环的 Langfuse Web 地址")
    langfuse_public_key: str = Field(default="", description="本机 Langfuse 项目公钥，只放本地环境配置")
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""), description="本机 Langfuse 项目密钥，不写入日志或前端")
    observability_shutdown_timeout_seconds: float = Field(default=5, gt=0, le=30,
        description="应用等待观测导出关闭的总秒数，超时明确记录未完成导出")

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
    intent_model: str = Field(
        default="gemini-3.5-flash-lite", min_length=1,
        description="CH06单一意图分类模型，沿用聊天上游；置信度仅记录，不启用分级升级",
    )
    judge_model: str | None = Field(
        default=None,
        description="忠实度裁判模型名；留空时沿用 chat_model，显式指定可用不同模型避免同模型自评偏差",
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
    workflow_checkpoint_path: str = Field(default=".data/checkpoints/ch05.sqlite3", min_length=1,
                                          description="CH05 单实例工作流 SQLite checkpoint 文件路径")
    workflow_max_decisions: int = Field(default=5, ge=1, le=5,
        validation_alias=AliasChoices("max_agent_steps", "workflow_max_decisions"),
        description="每轮最多内部决策次数，优先MAX_AGENT_STEPS，兼容WORKFLOW_MAX_DECISIONS")
    workflow_max_model_calls: int = Field(default=9, ge=1, le=9, description="每轮逻辑模型请求总上限，含一次空正文补救")
    workflow_total_token_budget: int = Field(default=240000, gt=0, description="每轮含 SDK 重试预留的保守估算总额，不是精确 Token 账单")
    workflow_stream_usage: bool = Field(default=False, description="显式请求兼容网关流式用量；缺失用量仍记未知")

    # CH07按模型窗口分配历史；旧TOKEN_BUDGET只保留给旧章入口兼容。
    model_context_window: int = Field(default=131072, gt=0, strict=True, description="应用配置的模型窗口，须根据实际模型校准")
    max_output_tokens: int = Field(default=4096, gt=0, strict=True, description="单次模型输出预留上限")
    max_user_input_tokens: int = Field(default=2000, gt=0, strict=True, description="本轮用户消息的估算上限，含协议")
    tool_result_max_tokens: int = Field(default=1200, gt=0, strict=True, description="单个工具结果模型视图上限，原始落库结果不截断")
    tool_call_max_tokens: int = Field(default=384, gt=0, strict=True, description="单个工具申请及关联协议的估算预留")
    context_target_turns: int = Field(default=32, gt=0, strict=True, description="期望保留历史轮数，只用于预算推导，不按轮数触发摘要")
    context_steady_turn_tokens: int = Field(default=1024, gt=0, strict=True, description="校准后的单轮稳态估算用量")
    context_evidence_max_tokens: int = Field(default=8192, ge=0, strict=True, description="检索资料预留上限，实际取与窗口1/16的较小值")
    context_summary_max_tokens: int = Field(default=4096, ge=0, strict=True, description="注入摘要预留上限，实际取与窗口1/24的较小值")
    context_non_ascii_tokens: float = Field(default=2.0, gt=0, allow_inf_nan=False,
        description="非ASCII字符估算系数，改变时必须同步重算预算与用量")

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
    mcp_logistics_url: str = Field(default="http://127.0.0.1:8101/mcp", description="本机物流MCP Streamable HTTP端点")
    mcp_after_sales_url: str = Field(default="http://127.0.0.1:8102/mcp", description="本机售后MCP Streamable HTTP端点")
    mcp_discovery_timeout_seconds: float = Field(default=3, gt=0, le=30, description="每个MCP Server独立发现时限")
    mcp_permissions_path: str = Field(default="data/tools/mcp-permissions.json", description="本地MCP权限JSON热配置文件")
    tool_audit_timeout_seconds: float = Field(default=1, gt=0, le=10, description="独立审计写入时限，失败不影响业务结果")
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
    milvus_hybrid_collection: str = Field(
        default="knowledge_ch04", min_length=1,
        description="CH04 原生 BM25/Dense 混合索引集合，独立于原 CH03 knowledge 集合",
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
    evidence_policy_path: str = Field(default="data/knowledge/evidence-policy.json", min_length=1,
        description="真实生产校准生成的v2证据策略；缺失或不兼容时新搜索失败关闭，没有拍定默认阈值")
    knowledge_relation_policy_path: str = Field(
        default="data/knowledge/relation-policy.json",
        min_length=1,
        description="离线知识关系判定的校准策略文件路径",
    )

    knowledge_retrieval_mode: Literal["hybrid", "dense"] = Field(
        default="hybrid", description="在线检索默认原生 BM25+Dense+重排；dense 为明确 CH03 回退，不因故障自动切换",
    )
    knowledge_timeout_seconds: float = Field(
        default=60, gt=0, le=300, description="FAQ/退款检索独立超时秒数，含逐子句本地推理；其他工具仍用原期限",
    )
    knowledge_rag_test_database: str = Field(
        default="mewhelp_ch04_test", min_length=1, description="新混合检索专用 MySQL 测试库，不能复用旧章测试库",
    )
    milvus_rag_test_database: str = Field(
        default="mewhelp_ch04_test", min_length=1, description="新混合检索专用 Milvus 测试库，与 MySQL 同时切换",
    )
    # CH04 检索参数：每路各 50，融合后 50，默认重排返回 10；不同于 CH03 Dense 的校准分数。
    recall_top_k: int = Field(default=50, ge=1, le=50, strict=True,
                             description="Dense 与 BM25 每路初步召回数量，同时作为 RRF 融合候选上限，默认各 50")
    rerank_top_k: int = Field(default=10, ge=1, le=50, strict=True,
                             description="普通检索重排后进入上下文的最大条数，默认 10，显式查询可覆盖")
    subquery_split: bool = Field(default=True, description="按中文逗号句号分号问号拆子句，每句分别完整召回并重排")
    rerank_min_score: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False,
                                   description="最高归一化重排分低于此值时 FAQ 返回空；不是 COSINE/RRF 阈值或正确概率")
    evidence_confidence_threshold: float = Field(default=0.26, ge=0, le=1, allow_inf_nan=False,
        description="参考 CH09 四信号闸默认值；当前未定义四信号/权重且未接入，不代表本项目已校准")
    rerank_model_path: str = Field(default="models/bge-reranker-large", min_length=1,
                                  description="本地 BAAI/bge-reranker-large 目录，请求中不下载")
    rerank_device: str = Field(default="cuda:0", min_length=1, description="显式重排设备，默认 cuda:0，不在异常时切换模型或设备")
    rerank_batch_size: int = Field(default=4, ge=1, le=32, strict=True, description="本地重排推理批次，默认 4，GPU 并发由锁串行")

    @field_validator("context_non_ascii_tokens", mode="before")
    @classmethod
    def validate_context_factor(cls, value: object) -> object:
        """拒绝布尔系数。参数：value环境或调用值；返回：原值；异常：ValueError，避免True被转为1.0。"""
        if isinstance(value, bool):
            raise ValueError("中文估算系数不能是布尔值")
        return value

    @field_validator("recall_top_k", "rerank_top_k", "rerank_batch_size",
        "model_context_window", "max_output_tokens", "max_user_input_tokens", "tool_result_max_tokens",
        "tool_call_max_tokens", "context_target_turns", "context_steady_turn_tokens",
        "context_evidence_max_tokens", "context_summary_max_tokens", mode="before")
    @classmethod
    def parse_retrieval_integer(cls, value: object) -> object:
        """只将环境中的十进制整数字符串转换为严格限额。

        参数：value 为环境字符串或代码显式值；bool/浮点保持原值交给严格字段拒绝。
        返回：object，纯十进制字符串转 int，其余保持原值。
        异常：无主动业务错误；不合法值由字段校验抛 ValidationError。
        """
        if isinstance(value, str) and value.strip().lstrip("+-").isascii() and value.strip().lstrip("+-").isdigit():
            return int(value)
        return value

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
