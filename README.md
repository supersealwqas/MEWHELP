# MewHelp

喵购商城客服演示项目：在同一个聊天页面完成“模型选工具 → 执行 → 按结果流式回答”。后端使用 FastAPI、SQLAlchemy 和 Docker MySQL，工具使用 LangChain `@tool`。

## 启动与演示

在项目根目录用 Git Bash 执行。先把 `.env.example` 中的配置补到本地 `.env`，填写已有模型配置及两个本地 MySQL 密码；不要覆盖已有模型凭据。

```bash
uv sync
docker compose up -d --wait mysql
uv run python -m app.db.init_db
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

浏览器访问 [客服聊天页](http://127.0.0.1:8000/)。页面与 API 同源，不需要 npm 或单独启动前端。普通初始化只创建缺失表、补缺失种子，不清空已有聊天。

依次发送：

1. `订单 1001 的物流到哪了`：出现“物流查询”徽章，回答引用返回节点并注明演示数据。
2. `退货政策是什么`：`query_faq` 用 SQL LIKE 命中 FAQ。
3. `邮费是多少`：实际查 `邮费`，结果为空，回答说明未查到。这是**预期的关键词漏召回**。
4. 对照 `运费是多少`：能查到 8 元、满 99 元免运费的演示政策。
5. `请帮我创建人工工单，订单 1001 一直未送达`：在 MySQL 写入待处理工单，回答给出真实保存的工单号。

订单、商品、物流每次在工具内部随机生成，不连接公司业务系统，不建额外业务表；同一单号的不同演示查询可能不一致。FAQ 来自演示政策表；工单会真实写入本地数据库，但没有人工坐席自动接入。

## 页面行为与接口

- Enter 发送，Shift+Enter 换行；保留中文输入法选字行为。
- 工具执行中、重试、成功或失败显示在当前客服气泡的徽章中；悬停可查看工具技术名。
- 最终回答按上游到达的增量显示。首字前有一次工具决策，等待取决于模型与网关。
- 停止回复会关闭本轮网络流并等待工具事务清理；已经提交的工单不会撤销，状态未知时请先核实，避免重复提交。
- 新会话和刷新页面生成新的 session_id；旧记录保留在数据库。此版本没有历史会话列表。
- `POST /api/chat` 仍接收 `session_id` 和 `message`，可选 `user_id` 默认 `demo-user`；标识上限 64 字符。
- SSE 保留 `data: {"delta":"…"}`、`data: [DONE]` 和 `event: error`，增加 `event: tool_status`；完成记录提交后才发送 DONE。
- `POST /api/extract` 保持原有结构化提取功能；可在 [接口文档](http://127.0.0.1:8000/docs) 查看。

每条消息最多执行一个工具。决策后最终模型请求不携带工具，不进入 Agent Loop。FAQ 只做原话关键词 LIKE，不做同义词扩展、向量检索或 RAG。

## 测试

```bash
# 离线回归，不请求真实模型、不连接 MySQL。
uv run --offline --locked pytest -q
node --test tests/frontend/chat-api.test.mjs

# 真实数据库测试：只允许 mewhelp_test；先停止使用测试库的评估服务。
uv run python -m app.db.init_db --test
uv run pytest -m mysql tests/integration -q
```

数据库测试会重建 **mewhelp_test** 中的四表，不操作应用库 mewhelp。覆盖约束、LIKE 转义、种子恢复、工单与结果防重、提交确认丢失、取消及启动恢复。

真实模型评估：

```bash
# 先停止旧评估服务及数据库测试；仅刷新测试库。
uv run python -m app.db.init_db --test --reset
MYSQL_DATABASE=mewhelp_test uv run uvicorn app.main:app --host 127.0.0.1 --port 8001
```

在另一个 Git Bash 终端运行：

```bash
uv run python tests/verify_ch02_live.py --base-url http://127.0.0.1:8001 --report dev-notes/ch02-evaluation.md
```

评估会先核对数据库 FAQ 是否与当前种子完全一致，再发送 14 条真实模型样例（含同会话连续三问），记录工具名、参数、结果、回答、状态顺序和增量数量；会使用模型额度。路由自动判定，回答忠实度需逐例复核。种子改变后必须停评估服务、重新执行测试库 reset，不能用普通幂等初始化替代刷新。

## 文件与运行限制

- `app/models/`、`app/db/`：四表模型、连接及显式初始化。
- `app/repositories/`：会话消息、FAQ 与工单短事务。
- `app/tools/`：五工具、注册、参数校验、5 秒默认超时和最多 1 次重试。
- `app/services/chat.py`：单工具聊天编排、同会话互斥、取消与持久化。
- `app/core/`：模型工厂与适配、完整轮次历史裁剪、提示词。
- `app/static/`：纯文本安全渲染与工具轨迹，无外部 CDN。
- [开发记录](dev-notes/ch02.md)、[真实模型评估](dev-notes/ch02-evaluation.md)。

使用**单实例、单 worker**；不要启动多个服务同时使用同一数据库。启动时将旧 running 轮标为 cancelled，保留已提交工具事实；恢复失败会阻止启动。同会话重叠请求被拒绝，不同会话可以并发。

Compose 项目为 `mewhelp-ch02`，MySQL 默认只监听 `127.0.0.1:3307`，数据保存在命名卷。不要执行 `docker compose down -v`，除非确实要删除本地数据库。

这是本地演示系统，尚无用户鉴权、权限控制和限流。`user_id` 不是可信登录凭证；勿直接暴露到公网。不能直接退款、修改订单或查询真实物流。
