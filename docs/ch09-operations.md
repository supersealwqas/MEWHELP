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

验收时确认 `docker compose ps` 的六个组件都显示 `healthy`，并确认 Web 与 Worker 的健康接口成功；页面可打开不能代替容器健康检查通过。Web/Worker 的容器内探针使用实际容器主机名，设置4秒请求期限，失败返回非零并记录错误类型或HTTP状态。宿主机发布端口仍只绑定回环地址。

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

## CH09 表结构迁移

迁移只补给定DDL的两张新表、问题池两列和对应索引/外键。先核实 `.env` 的目标应用库名；本机已确认目标是 `127.0.0.1:3307/mewhelp`。在项目根目录的 Git Bash 执行：

```bash
PYTHONIOENCODING=utf-8 uv run python -m app.db.migrate_ch09 --env-file .env --database mewhelp
PYTHONIOENCODING=utf-8 uv run python -m app.db.migrate_ch09 --env-file .env --database mewhelp --apply
PYTHONIOENCODING=utf-8 uv run python -m app.db.migrate_ch09 --env-file .env --database mewhelp
```

第一条预览，第二条应用，最后一条应返回空 `statements`。Windows显式指定stdout的UTF-8编码便于核对中文枚举；数据库连接仍由迁移实现按utf8mb4处理。命令会核对实际库名和既有结构，不用 `create_all` 代替增量迁移。

2026-10-30本机应用库已按以上流程迁移，完整结构核验通过；迁移前后163个会话、1367条消息、17条问题池原件、52条知识块的数量一致。后续服务启动继续严格核验表结构，不重复执行已经完成的DDL。

已有应用库也必须完成前章迁移。2026-10-30真实确认验收发现CH06等待状态约束尚未升级，已运行 `uv run python -m app.db.migrate_ch06 --env-file .env --database mewhelp` 预览，再加 `--apply` 执行，最后复预览为空；补齐等待状态与空退款申请表，原有业务行数未变。新环境须按所缺章次预览并迁移，不能以CH09新表齐全推断前章约束也完整。

## 人工审核与发布状态

审核接口使用 `/api/flywheel/reviews` 列表和 `/api/flywheel/reviews/{id}` 详情。列表可传 `status=待审/通过/驳回`、`after_id`、`limit`（1–200）。主键和返回游标使用十进制字符串，浏览器不能把大整数主键转换成 Number。

对某条缺口提交 `POST /api/flywheel/reviews/{id}/approve`，正文为 `{"answer":"人工核准的完整答案"}`。服务先用实际分词器校验标准问题、类别和答案合计1600字符、格式化后2048 token，再冻结正文；超限返回422，修改后可以重新提交。模型示例仅作参考，发布保存人工输入的原文。

`publication_status` 是从持久状态计算的：`not_started` 表示尚未开始发布，`pending` 表示人工答案已冻结但发布待完成，`published` 表示已通过数据库和向量可见性核验。发布使用稳定来源 `ch09/review/{id}`；最后审核完成事务与知识写锁共用同一连接，等待期间失锁会保留待审。

冻结后通过 `/retry` 重试，或由服务启动和周期恢复继续处理；不能换答案或驳回，重复相同核准答案不会生成新知识身份。尚未冻结的条目可以 `POST /api/flywheel/reviews/{id}/reject`。404表示条目不存在，409表示状态冲突，422表示可修正输入，503表示依赖或发布未完整完成。收到503后应重新读取详情；状态暂时读不到时界面不能猜成发布成功或失败。

详情中的每条原件保留原始 `retrieved_chunks`；展示采用独立的 `retrieval_status` 和 `retrieval_snapshot`。只有 `not_retrieved` 才表示确认未检索，`empty` 表示零命中，`fault` 表示检索故障，`legacy_unavailable/legacy_incomplete` 表示旧证据缺失或不完整。旧SQL NULL本身不能证明未检索；系统不会重搜今天的知识来补造历史。

## 本机验收专用知识

闭环演示使用独立可辨认的“青禾校验卡”问答，来源会保存为 `ch09/review/<实际审核ID>`，不是商城商品或售后政策。核准正文固定为：“仅用于 MewHelp CH09 本机验收演示：青禾校验卡的核对词是‘青禾入库完成’。这不是商城商品、订单或售后政策。”此正文由本验收文档定义并人工核对，不使用模型给出的示例答案作为政策。

先执行第一轮48题生产评估，再通过审核页发布这条演示知识，验证重新提问确实引用新增知识，最后执行同口径第二轮评估。所有正式业务FAQ仍须核对已有业务资料或用户明确提供的事实。

本轮固定问题为“验收演示中，青禾校验卡的核对词是什么？”，对应审核14、来源 `ch09/review/14`。原先带“CH09”的提问触发了型号保留检查，属于检索故障原件；不能将它当成发布前缺知识的成功样例。新的原轮身份和只读核验见 `reports/ch09/acceptance/evidence-manifest-draft.json`。

## 页面、反馈与用量

聊天入口 `/`、审核后台 `/flywheel`、评估页 `/rag-eval` 使用同源资源。当前本机验收应用为 `http://127.0.0.1:18000`，Langfuse 为 `http://127.0.0.1:13000`；8000是上方普通启动命令的示例端口。

在历史会话切回原轮后点击“没帮助”，浏览器向 `/api/chat/{session_id}/turns/{turn_id}/feedback` 提交 `{"user_id":"demo-user","resolved":false}`。只有服务器确认成功后才显示已提交；重复请求仍返回原池主键。同轮自评与用户差评可以各保留一条来源原件，但归并后的问题出现次数只计一次。

后台消费取自 `GET /api/observability/costs` 的完整时间窗口。时间按设备时区输入，结束时刻不含在内；展示输入、输出、已知token、缺失用量数量与覆盖比例，未知金额不填0。服务器数据库时间按UTC保存，页面转换为设备时间。订单1001等内置业务资料带 `is_mock=true`，是本机演示数据，不能当成真实商城订单。

## 生产评估与每日调度

评估只运行生产知识子流程，包括理解、查询、召回/精排、证据判断、知识自评和引用处理。它不覆盖全部客服业务图，不向真实会话和问题池写入评估提问。原始题集为72题，当前成绩只使用其中48道evaluation题；24道校准题不混入成绩。

在项目根目录的Git Bash执行以下命令。评估会调用已配置模型并产生实际用量；输出目录必须尚不存在。

```bash
uv run python -m app.evaluation.cli evaluate-production --dataset data/evaluation/ch04.json --policy data/knowledge/evidence-policy.json --triggered-by 手动 --target application-readonly --output-dir reports/ch09/eval-before
# 完成核准知识发布与同问重验后，保持数据、模型和策略不变：
uv run python -m app.evaluation.cli evaluate-production --dataset data/evaluation/ch04.json --policy data/knowledge/evidence-policy.json --triggered-by 手动 --target application-readonly --output-dir reports/ch09/eval-after
```

本轮用 `.superpowers/sdd/2026-10-30-ch09-observability-flywheel/task11-evaluate.py` 代替 `-m app.evaluation.cli` 入口，仅在进程内读取本地忽略配置并开启Langfuse观测；所有子命令和参数相同，不把密钥写入报告。

每轮完整保存 `manifest.json`、`summary.json`、逐题结果、冻结题集、知识与策略原件，并向应用库 `eval_runs` 写一行。生成或裁判的真实逐题错误保留，完成但有错误与完全无错分开显示，缺少的指标保持空值。若知识/模型/协议在执行中变化或已经确认失锁，产物不可补存为正常成绩。

如果只有最后落表失败、完整产物仍合法，可用以下命令补存，不重新调用模型；来源沿用原始运行记录：

```bash
uv run python -m app.evaluation.cli persist-run --artifact-dir reports/ch09/eval-before
```

生产API为 `POST /api/rag-eval/production/rerun`、`GET /api/rag-eval/production/job`、`GET /api/rag-eval/production/runs`。页面读取全部分页并按比较键判断能否直接算差值。只有知识版本改变时允许比较并标明版本；模型、题集、策略或协议改变时，不直接显示前后增减。下降也照实保留。

`EVALUATION_SCHEDULE_ENABLED` 默认开启；默认 `EVALUATION_HOUR=3`、`EVALUATION_MINUTE=0`、`EVALUATION_TIMEZONE=Asia/Shanghai`。启动只安排下一次，不补跑错过的日期；手动、定时和独立CLI共享数据库执行锁。初始 `admission=pending` 只是当前实例已接受，不能据此认为已获得全库执行权。应用关闭会先停止调度，再回收自己启动的评估进程。

## 真实验收与故障恢复记录

`tests/verify_ch09_live.py` 只读API、数据库和Langfuse，并重新核验清单指向的原件，不自动发问、反馈、发布或评估：

```bash
uv run python tests/verify_ch09_live.py --base-url http://127.0.0.1:18000 --evidence-manifest reports/ch09/acceptance/evidence-manifest.json --output-dir reports/ch09/acceptance/final-read
```

输出目录使用新名称，保留既有失败原件。返回1表示存在失败或缺项，不能写成整体验收通过。浏览器画面与答案语义由单独具名记录负责，脚本只检查其引用和文件摘要，不把自动验证冒称人工观察。

发布失败恢复的演示采用本验收进程中的定向故障开关：只有明确的审核14演示正文和内容指纹匹配时，在向量写入前产生故障；MySQL已经冻结的人工答案和待同步知识保留。其他服务和共享Milvus均不停机。撤销该独占开关后，通过真实“重试发布”恢复，并再次核对MySQL、向量指纹和同问引用。此故障由验收主动注入，应与自然网络故障区分；具体执行时间和状态以最终原件为准。
