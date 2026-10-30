# CH08 工具系统运行手册

CH08 已完成开发与复审修复，当前交付结论以 [2026-10-30 最终复审报告](superpowers/reviews/2026-10-30-ch08-rereview.md) 为准。2026-10-29 的 `live-final-4` 保留九项端到端检查及八项 Codex 内置浏览器操作的历史证据；修复后的模型验收和增补页面另行记录，不能用旧源码指纹证明新版本。实际操作与证据边界见[页面验收记录](ch08-browser-verification.md)。

## 运行环境

在项目原目录使用 Git Bash 和 uv。客服进程保持一个 worker，避免多个进程同时维护同一 SQLite 工作流文件。已核验依赖为 LangGraph 1.2.17、FastAPI 0.143.1、Pydantic 2.14.1、MCP SDK 1.30.0、langchain-mcp-adapters 0.3.2、HTTPX 0.28.1、AnyIO 4.15.1。

`.env` 中的业务模型当前为 `gemini-3.5-flash-lite`。聊天预算会计算本次实际绑定的全部工具定义；窗口不足时在发送模型请求前拒绝，不漏算工具定义。

## 数据库

审计迁移只新增 `tool_audit_logs`，不修改已有工单表；已有同名异构表会被拒绝，不自动修表。先预览并核对输出中的库名，再决定是否应用：

```bash
uv run python -m app.db.migrate_ch08 --env-file .env --database mewhelp_test
uv run python -m app.db.migrate_ch08 --env-file .env --database mewhelp_test --apply
```

本章验收仅写 `mewhelp_test`。真实集成测试会重建该测试库业务表，因此不能与运行中的本章验收器同时执行。不要把应用库传入验收脚本。已有知识库和本地模型权重仅供读取，验收不重建知识库。

## 启动三个独立服务

先核对端口是否空闲，不结束不属于本次任务的进程。三个终端分别运行：

```bash
uv run python -m mcp_servers.logistics --port 8101
uv run python -m mcp_servers.after_sales --port 8102
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

两个 MCP 服务使用真正的 SDK 和 HTTP 传输，业务内容为本地演示数据，会在结果中保留演示标记。内置工具提供订单、商品、FAQ、退款政策和建单；物流、在保、退货进度来自 MCP。

## 热发现与权限

工具用途说明不能授予权限。`data/tools/mcp-permissions.json` 按来源、Server 和名称配置本地权限；模型不会看到未授权的新工具。本章只允许 MCP 读取，外部写工具不放行。

例如新增无用户数据的只读工具后，在完整权限文件中增加：

```json
{"source":"mcp","server":"after_sales","name":"query_service_hours","enabled":true,"access":"read","ownership":"none","result_policy":{"is_mock":true}}
```

保持其他规则不变，建议写临时文件后原子替换。每次决定前刷新发现和权限；只重启修改的 MCP Server，客服无需重启。合法完整配置才替换旧配置；格式错误保留最后有效规则，权限文件缺失则默认拒绝 MCP。单个 Server 离线只撤下该来源，不能偷偷回退旧内置实现。

内置新工具通过已启动的 `service.executor.registry.register_definition(...)` 登记，不提供公开的注册 HTTP 接口。登记要声明可信来源、异步处理器、JSON Schema、读写类型和本地结果策略；名称冲突会被隔离，不覆盖原工具。

## 工单确认与恢复

“帮我建个工单”先追问具体问题；问题描述只取当前诉求中的用户原话，不采用模型或工具编造内容。完整描述最多 2000 字，超长时要求整理，不能静默截断。单纯投诉、转人工、引用、否定或假设不等于建单授权。

完整申请先保存冻结预览，然后暂停。用户点击确认或取消时，只提交原会话、原轮、动作 ID 和决定；浏览器不能覆盖描述及类型。首次决定持久化，相同决定重发复用原结果，相反决定冲突。写入始终经过同一个工具执行器。

刷新后先使用页面的“核实状态”。已确认但尚未继续执行的动作提供“继续处理”；已有工单显示真实编号。执行已经开始但无法核实提交结果时，不再调用建单来试探，显示“待核实”并阻止新建。执行前已确定拒绝则可以继续聊天。取消也恢复原工具调用，保存拒绝结果及审计，不生成工单。

## 错误与回执

参数校验错误、空结果、权限拒绝和真实故障分别保存为结构化结果。当前轮的可信读查询明确故障、且没有成功查询事实时，由程序直接说明查询未完成，不调用正文模型。判定使用服务器保存的原调用事实，来源随后下线或规则更新不会抹掉原故障；正常空结果、权限或参数拒绝、未执行调用及历史故障不会触发这一出口。成功查询与故障混合、或旧检查点缺少调用事实快照时，仍沿用原答复流程。

只有明确的瞬时读取故障最多额外尝试一次；写工具不自动重试。LangGraph 暂停、取消及提交未知不能被当成普通失败吞掉。已确认创建的真实工单号在后续选单、查询故障和恢复时继续保留；整轮未真正完成时，不发送伪造的完成元数据。

完整工具结果保存在消息流水，模型只使用预算内的结果副本。嵌套工单号、状态及必要父路径必须保留；核心回执本身放不下则明确报预算不足。审计采用独立短事务，审计故障不能覆盖已经提交的工单号。

## 验收入口

模型评估先固定 20 组、23 轮标注，每组至少重复两次；评估器不写业务库、不执行业务处理器，只读工具反馈明确标记为模拟。模型正文、程序固定答复和预览候选分别记录；程序答复不算模型原始输出。逐条语义复核完成前保持待审：

```bash
uv run python -m tests.eval_ch08_tools --env-file .env --report <新的模型报告目录>
uv run python -m tests.eval_ch08_tools --review-report <模型报告目录> --review-file <复核JSON>
```

真实端到端验收启动自己拥有的两个 MCP 进程和测试网页，复用同一个运行时，任何必验缺项均非零退出：

```bash
uv run python -m tests.verify_ch08_live --env-file .env --database mewhelp_test --report <新的验收目录> --model-report <已复核模型报告目录> --browser-timeout 3600
```

`browser-ready.json` 提供本次测试网址；浏览器操作记录、截图和数据库核对保存到同一目录，完成后生成 `browser-evidence.json`。不能把默认 pytest 中的 MySQL deselected 或知识库 skipped 记为真实验收通过。

本次按用户要求使用 **Codex 内置浏览器**。验收实例的 `BrowserObserver` 记录实际 HTTP 请求/响应到 `browser-http/`；截图由内置浏览器直接取得，数据库快照由原实例处理 `browser-snapshot-requests/` 中的只读请求。该观察器只在验收脚本启用，不进入生产启动流程。断线和迟到响应测试在本轮目录显式指定会话，并记录注入边界；不替换模型或业务结果。

完整回归需要同时开启真实依赖测试，并等其结束后再启动本章页面验收，避免测试夹具重建 `mewhelp_test` 影响页面会话：

```bash
env MEWHELP_TEST_ENV_FILE=D:/AI/my_AI_project/MEWHELP/.env \
  MEWHELP_RUN_QUALITY_MODEL=1 MEWHELP_RUN_QUALITY_MYSQL=1 \
  MEWHELP_RUN_RAG_INTEGRATION=1 MEWHELP_RUN_RERANK_LIVE=1 \
  uv run --no-sync --offline pytest -q -o addopts= --run-knowledge-integration \
  --basetemp=.pytest-tmp/ch08-full
node --test tests/frontend/*.test.mjs
uv lock --check --offline
uv pip check
```

2026-10-30 修复后的全开关回归为 **2411 passed、零失败零跳过**，前端为 **132 passed**；日志保存在 `.superpowers/sdd/2026-10-30-ch08-review-fixes/closeout-final-regression.log` 和 `program-node.log`。最终模型目录为同目录的 `model-closeout-final/evaluation`，46 条结构与语义均通过，原件核验覆盖 108 次真实模型调用及当前源码。46 条中包含 38 条模型正文、4 条程序固定答复和 4 条预览候选，模拟反馈不证明真实写库。增补页面使用独占 SQLite 和有限脚本模型，详细限制见最终复审报告。旧批次与失败记录保留，不作为当前版本的通过证据。

内置浏览器截图原件为 JPEG；本轮另存像素相同的 PNG，哈希对应关系见 `screenshot-exports.json`。证据目录保留在本机且被 Git 忽略，不随源码提交上传。
