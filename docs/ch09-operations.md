# CH09 本机观测运行说明

追踪平台使用独立 Compose 项目 `mewhelp-langfuse`。客服业务继续使用现有 MySQL、Milvus 和模型配置；追踪数据仅导出到本机 Langfuse。

## 初始化与启动

1. 复制 `.env.langfuse.example` 为 `.env.langfuse`，把全部秘密占位符替换为分别生成的随机值。`ENCRYPTION_KEY` 必须是 64 位十六进制字符串。可以用 `uv run python -c "import secrets; print(secrets.token_hex(32))"` 生成单个值。凭据仅保存在忽略文件中，不粘贴到开发日志。
2. 验证配置后启动：

```bash
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml -p mewhelp-langfuse config --quiet
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml -p mewhelp-langfuse up -d
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml -p mewhelp-langfuse ps
```

Web 地址为 `http://127.0.0.1:13000`，健康检查为 `/api/public/health`。首次启动会用本地环境文件中的管理员邮箱、密码创建 `MewHelp CH09` 项目。初始化后改环境值不会重置已有账号或项目密钥。

3. 在应用 `.env` 中设置 `OBSERVABILITY_ENABLED=true`、`LANGFUSE_BASE_URL=http://127.0.0.1:13000`，并填入与初始化项目一致的 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`。`OBSERVABILITY_SHUTDOWN_TIMEOUT_SECONDS=5` 是应用等待导出关闭的总时限。然后照常运行 `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000`。

Web / Worker 固定为服务端 `4.72.0`，Python SDK 固定 `4.15.4`，其 LangChain 集成必需主包固定 `1.4.6`，保持原有 core / openai / graph 版本。ClickHouse、PostgreSQL、Redis 与 MinIO 镜像同样显式固定；实际拉取 digest 和健康证据保存在 `reports/ch09/`。不要把 SDK 版本当成服务端版本，也不要用 `latest` 做验收。

所有发布端口都绑定 `127.0.0.1`；数据库和缓存只在该 Compose 网络内访问。独立卷带项目名前缀，不复用业务卷。已经关闭 OSS telemetry、注册引导与内置 AI 助手。容器直接从官方上游镜像仓库拉取，未经过采集下载指标的 `docker.langfuse.com` 代理。

## 查看原轮完整链路

在聊天页发起真实问题后，在 Langfuse 中打开 `MewHelp CH09`，按 session、user 或 trace ID 找到对应原轮。

- `execution.chat` 是首次执行根；用户原消息的 `rag_metadata.trace_identity` 在第一个模型调用前保存这个根 ID。
- 确认恢复使用 `execution.resume`，同一个 trace ID，并把原根作为父节点。检查点仅保存 JSON 身份。
- 理解、分类、决策、正文及修复调用由同一个 LangChain handler 记录为 generation。完整 messages、工具定义、实际输出、model、usage 由 SDK 记录；工具和本地精排不会被重复计为模型调用。
- 分类成功立即更新存活执行根的 `intent`、`intent_confidence`、`route`。分类前模型消费按 trace ID 关联此根归属，不能只统计带意图标签的后半段模型调用。
- `tool.execute` 保存原申请、可信来源、权限结果、尝试次数、重试、结果及失败；`knowledge.retrieve` 保存实际查询、原始召回分、权威候选和排名；`knowledge.rerank` 保存本地精排输入输出与耗时。
- 后台摘要另建 trace，并固定 `task_kind=summary`。后续飞轮及评估使用相同运行时的 `execution(..., kind="flywheel" / "evaluation")`，将其 Runnable 一次绑定运行时的 `callback_handler`，避免混入客服直接消费。
- 完成回执只回放已保存结果，不调用图或模型，不增加 generation。

`usage_origin=model_reported` 表示收到完整上游输入/输出用量；`inferred_or_unknown` 表示不完整或没有用量。`input_usage_present` / `output_usage_present` 保留原响应字段是否存在，收到的零值仍是有效用量。Langfuse 可能自行推算 token，因此统计必须同时查看这些字段，不能把自动推算当精确账单。金额没有可靠价格时保持未知。

SDK 会把顶层链元数据再传播到整条 trace，因此顶层只保留稳定身份与任务类别，动态意图只更新活跃执行根及后续子观测，避免初始“未分类”覆盖最终结果。非 Runnable 工具边界使用固定 SDK 的 `_get_parent_observation` 兼容钩子，结束前的用量标记使用 `_runs` 映射；升级 SDK 必须运行真实 SDK 回归测试并核对工具子树、意图和原始用量字段。

服务端 v4 支持 Observations v2：`GET /api/public/v2/observations`。查询要显式选择 `core,basic,time,metadata,model,usage,trace_context` 并遍历 `meta.cursor`；不要调用旧版本的 `/api/public/observations`。根可能早于统计时间窗，意图关联要单独补查根。追踪内容仅留本机，公开 SSE 继续走现有白名单。

## 关闭和故障

正常停止应用后，先完成业务和摘要收尾，再关闭观测。运行时在守护线程调用 SDK `shutdown()`（内部已含 flush），没有传入不存在的 timeout 参数。应用最多等待配置秒数，失败以 `observability export_failed` 或 `observability shutdown_timeout` 日志及 `service.observability.health` 显示。

Python 无法强制回收永久阻塞的线程。超时意味着导出未完成，不能承诺 trace 完整；该线程为守护线程，且运行时自有 OTel provider 不注册阻塞退出钩子。独立子进程测试会真正阻塞 exporter 并验证进程能退出，不把协程超时冒称线程已回收。

保留数据停止追踪平台：

```bash
docker compose --env-file .env.langfuse -f docker-compose.langfuse.yml -p mewhelp-langfuse stop
```

不要运行带 `-v` 的删除命令；这会删除追踪数据。更新版本前先备份本项目全部卷并查官方升级说明。
