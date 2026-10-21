# MewHelp

喵购商城客服演示项目：在同一个聊天页面完成“模型选工具 → 执行 → 按结果流式回答”。后端使用 FastAPI、SQLAlchemy 和 Docker MySQL，工具使用 LangChain `@tool`；CH03 起 `query_faq` 使用 BGE-M3 + Milvus 的 Dense 语义检索，MySQL 保存权威原文与发布状态。

## 启动与演示

在项目根目录用 Git Bash 执行。先把 `.env.example` 中的配置补到本地 `.env`，填写已有模型配置及两个本地 MySQL 密码；不要覆盖已有模型凭据。

```bash
uv sync
docker compose up -d --wait mysql
uv run python -m app.db.init_db
# CH03 知识库初始化、六份资料导入与向量同步（应用库/应用 Milvus 库）
uv run python -m app.knowledge.cli init
uv run python -m app.knowledge.cli import-docs --manifest data/kb/manifest.json
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

浏览器访问 [客服聊天页](http://127.0.0.1:8000/)。页面与 API 同源，不需要 npm 或单独启动前端。普通初始化只创建缺失表、补缺失种子，不清空已有聊天。

CH03 起 `query_faq` 的在线数据来自 `data/kb` 六份权威文档构建的 Dense 知识库，不再使用 FAQ LIKE；FAQ 表保留为旧数据。启动服务前必须有版本匹配的 `data/knowledge/retrieval-policy.json`，生成/校准、测试库和恢复命令见 [CH03 runbook](docs/ch03-runbook.md)。已有数据库中的旧答案不会被普通初始化覆盖；仅修改种子代码不会自动刷新已保存的数据。手动 SQL 的用途与执行顺序见 [sql/README.md](sql/README.md)。

依次发送：

1. `订单 1001 的物流到哪了`：同一条消息依次执行 `query_order` 和 `query_logistics`，自动使用订单结果里的物流单号，直接回答物流轨迹；页面依次显示订单与物流两个徽章。
2. `物流单号 MOCK213502378238 到哪了`：出现“物流查询”徽章，回答引用返回节点并注明演示数据。
3. `退货政策是什么`：`query_faq` 走向量知识库，命中退货政策。
4. `邮费是多少`：原词“邮费”经 Dense 检索命中 6 元基础运费、满 99 元包邮，并保留偏远地区 12 元附加费不参与包邮的例外；不再要求“邮费零命中”。
5. 对照 `运费是多少`：同样命中知识库，不再使用 10 元旧口径。
6. `请帮我创建人工工单，订单 1001 一直未送达`：在 MySQL 写入待处理工单，回答给出真实保存的工单号。
7. `帮我查订单 2003`：该单属于另一个演示用户，工具返回 `order_not_owned`，页面直接回固定文本“没有找到您的这笔订单，请确认订单号或物流单号后重试。”，不经过模型改写。

订单、商品、物流在工具内部按入参生成可复现的演示数据，不连接公司业务系统，不建额外业务表。订单和物流共用以 `order:{order_id}` 为种子的独立 `random.Random`，商品使用 `product:{keyword}`；同一入参在相同代码和 Python 版本下，重复调用、交错查询、重启进程都返回相同数据。物流时间使用固定演示时间，不读取当前时钟；未发货订单没有承运方和物流单号。FAQ 来自 `data/kb` 构建的语义知识库；工单会真实写入本地数据库，但没有人工坐席自动接入。

每条消息默认最多执行4个业务工具，可通过 `TOOL_MAX_CALLS_PER_TURN` 设置为1至8。支持连续决策和同批多个独立查询，工具均按顺序执行；订单→物流等有依赖的步骤使用前一步的真实结果。`query_logistics` 只接受 `tracking_no`，不接受订单号；用户只给订单号时自动先查订单，有单号才继续查物流，待发货则说明暂无物流信息。`query_order` 与 `query_logistics` 的 `user_id` 仍由执行器从 `ToolContext` 注入，模型看不到也填不了；模型夹带身份字段会被判为非法参数。归属不符与订单不存在返回同一 `order_not_owned`，最终回复由服务端固定文本给出，保证两种情况措辞完全一致、不泄露订单是否存在。

同轮相同工具和参数不会重复执行，每条消息最多尝试创建一张工单。达到调用上限、工具失败或检测到重复时停止继续调用；批次中未执行的项目会留下明确的跳过结果。已经提交的工单编号始终保留，后续步骤失败也不能直接重发建单。执行器原有同一调用的有限网络重试继续保留。

文档示例订单 `1001` 固定为：1 件演示猫粮，单价和总额均为 39.9 元，订单状态“已发货”，物流状态“运输中”，节点“包裹已到达杭州转运中心”，物流单号 `MOCK213502378238`，节点时间 `2026-02-05T16:45:00+00:00`。这份工具结果由测试固定校验；模型的自然语言措辞仍可能变化。

模型可见的工具说明使用业务语言，描述用途、参数要求、返回内容和操作边界，不包含 mock、随机种子等开发实现信息。当前工具显式设置 `@tool(description=...)`，这段说明和参数字段定义会发送给模型，普通代码注释不会发送。内部仍使用可复现的模拟数据；回答是否注明演示来源，由工具结果的 `is_mock` 标识决定。

## 页面行为与接口

可查询的订单来自有限演示清单：默认 `demo-user` 拥有 `1001`–`1007` 和 `A-1001`；`2003` 属于 `demo-other-user`。其他订单号均返回相同的归属拒绝；不是任意订单号都能生成可查询数据。`1005`、`1007` 为待发货示例，没有物流单号。

- Enter 发送，Shift+Enter 换行；保留中文输入法选字行为。
- 工具执行中、重试、成功、失败或未执行显示在当前客服气泡的徽章中；多个调用分别显示，悬停可查看工具技术名。
- 最终回答按上游到达的增量显示。首字前可能有多次工具决策及查询，等待取决于模型与网关。
- 停止回复会关闭本轮网络流并等待工具事务清理；已经提交的工单不会撤销，状态未知时请先核实，避免重复提交。
- 工单成功后，服务端先发送已保存的工单号，再生成补充回答；模型或回答保存失败仍保留已收到的编号和文字。断流、取消或写入状态未知时不自动回填原话、不鼓励重复建单；连接已断开时无法保证新提示送达。
- 新会话和刷新页面生成新的 session_id；旧记录保留在数据库。此版本没有历史会话列表。
- `POST /api/chat` 仍接收 `session_id` 和 `message`，可选 `user_id` 默认 `demo-user`；标识上限 64 字符，问题上限 4000 字符。最终回答上限 12000 字符，超限停止生成并明确标为未完成。
- SSE 保留 `data: {"delta":"…"}`、`data: [DONE]` 和 `event: error`，增加 `event: tool_status`；完成记录提交后才发送 DONE。错误载荷中的可选 `retry_safe=false` 表示应先核实提交结果，不能直接重发原话。
- `POST /api/extract` 保持原有结构化提取功能；可在 [接口文档](http://127.0.0.1:8000/docs) 查看。

工具决策在次数上限内循环，前一步结果回传后再决定下一步；每次都重新核算消息预算，最终回答阶段不携带工具。CH03 起 `query_faq` 只做 BGE-M3 Dense 单路检索：关键词仍取用户原话、不做同义词改写；命中按 MySQL active/done/指纹回读，低分是正常空结果，索引失配或后端异常是工具错误，不伪装为空命中。

## 测试

对话使用两层近似预算：`TOKEN_BUDGET=12000` 限制历史消息、当前问题及工具事实；`REQUEST_TOKEN_BUDGET=16000` 另限制包含系统提示和本阶段工具定义的完整输入。ASCII 每四字符约一单位，非 ASCII 每字符约两单位，协议字段同样计入，JSON 保留真实 Unicode。它们不是模型实际分词数或上下文窗口；16000 也不含模型输出，需结合实际网关限制留出输出空间。历史只能按整轮裁剪；当前问题或工具结果仍超限时明确失败，不丢掉事实继续回答。较长的 FAQ 结果可能触及对话预算，应缩小问题范围或按模型容量同时调整两层配置。

离线测试在首次导入时禁用项目 `.env` 文件来源，并为实际使用配置的模块注入隔离配置；单文件与整套运行都不依赖本机模型凭据。配置读取测试使用虚构配置文件。非法工具申请会留作数据库审计，但不会执行，也不会以损坏的工具协议回传模型。

```bash
# 离线回归，不请求真实模型、不连接 MySQL。
uv run --offline --locked pytest -q
node --test tests/frontend/chat-api.test.mjs

# 真实 MySQL：CH02 mewhelp_test
uv run python -m app.db.init_db --test
uv run pytest -m mysql tests/integration -q

# CH03 专用 MySQL 与 Milvus 测试库；先确认 Docker 和 milvus-standalone 运行
uv run pytest tests/knowledge_integration --run-knowledge-integration -q
```

CH02 数据库测试会重建 **mewhelp_test** 中的四表，不操作应用库 mewhelp。CH03 集成测试只清理 **mewhelp_ch03_test** 与 Milvus 测试库自己的数据，覆盖发布、批次、来源补扫、向量重写与恢复。

CH03 当前验收范围为链路正确性，检索与生成质量留待后续优化。以下入口只操作 `mewhelp_ch03_test` 与 Milvus 测试库，会使用真实 BGE-M3、Milvus、MySQL 和模型额度：

```bash
uv run python tests/verify_ch03_live.py pipeline --test --report dev-notes/ch03-pipeline.md
uv run python tests/verify_ch03_live.py recovery --test --report dev-notes/ch03-recovery.md
```

`pipeline` 要求测试库已导入当前资料，检查知识快照、真实工具调用、SSE 和消息落库；`recovery` 演示向量写入后中断与重跑。检索策略当前是未完成质量校准的链路基线。后续质量优化可使用 `calibrate` 和 `evaluate`：前者重建测试知识库，只有校准全部通过才覆盖策略，后者运行独立固定问法的质量验收。CH02 的 `verify_ch02_live.py` 和 `ch02-evaluation.md` 保留为历史记录，不再要求“邮费零命中”或 10 元口径。完整命令、应用库初始化和启动方式见 [CH03 runbook](docs/ch03-runbook.md)。

## 文件与运行限制

- `app/api/`：接收聊天与结构化提取请求，输出 HTTP/SSE 响应。
- `app/models/`、`app/db/`：客服与知识库表模型、连接及显式初始化。
- `app/repositories/`：会话与工单存储，以及调用知识检索的客服入口。
- `app/knowledge/`：集中处理知识库业务。`jobs.py` 编排导入和挖掘，`lifecycle.py` 决定发布/合并/隔离，`store.py` 负责原文与审计事务，`sync.py` 对照并同步向量，`search.py` 执行在线检索；编码与 Milvus 访问分别在 `embedding.py`、`vector.py`。
- `app/tools/`：五工具、注册、参数校验、5 秒默认超时和最多 1 次重试。
- `app/services/chat.py`：有界多工具聊天编排、同会话互斥、取消与持久化。
- `app/core/`：模型工厂与适配、完整轮次历史裁剪、提示词。
- `app/static/`：纯文本安全渲染与工具轨迹，无外部 CDN。
- `tests/knowledge/`、`tests/knowledge_integration/`：离线知识规则测试与真实存储集成测试，共享替身集中在 `tests/knowledge/fakes.py`。
- [CH02 开发记录](dev-notes/ch02.md)、[CH02 历史评估](dev-notes/ch02-evaluation.md)。
- [CH03 开发记录](dev-notes/ch03.md)、[CH03 校准](dev-notes/ch03-calibration.md)、[CH03 验收](dev-notes/ch03-evaluation.md)、[CH03 运行手册](docs/ch03-runbook.md)。

当前 embedding 使用本地 BGE-M3。`EMBED_BASE_URL`、`EMBED_MODEL`、`EMBED_API_KEY` 仍是预留配置，填写它们不会切换编码器；云端接入与配置切换后续单独实现。

使用**单实例、单 worker**；不要启动多个服务同时使用同一数据库。启动时将旧 running 轮标为 cancelled，保留已提交工具事实；恢复失败会阻止启动。同会话重叠请求被拒绝，不同会话可以并发。

Compose 项目为 `mewhelp-ch02`，MySQL 默认只监听 `127.0.0.1:3307`，数据保存在命名卷。不要执行 `docker compose down -v`，除非确实要删除本地数据库。

这是本地演示系统，尚无用户鉴权、权限控制和限流。`user_id` 可以由客户端填写，所有默认访客共用 `demo-user`；知道同一会话号并指定匹配的演示身份，就能继续读取该会话。不要分享页脚的完整会话号，也不要存放真实个人或业务资料；会话号不能代替可靠的身份认证。开发脚本默认只监听 `127.0.0.1`，勿直接暴露到公网。多人使用前需要另行设计登录与权限隔离。不能直接退款、修改订单或查询真实物流。
