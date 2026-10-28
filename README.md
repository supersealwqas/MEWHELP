# MewHelp

喵购商城客服演示项目：CH05先识别诉求，再按固定流程检索知识、查询业务或给出投诉/闲聊回复。业务查询由模型分步选择只读工具；建单必须由用户点击确认。后端使用FastAPI、SQLAlchemy、MySQL和LangGraph，官方SQLite保存会话工作流；知识沿用BGE-M3、Milvus混合检索和本地重排。运行与验收见[CH05运行手册](docs/ch05-runbook.md)。

## 启动与演示

在项目根目录用 Git Bash 执行。先把 `.env.example` 中的配置补到本地 `.env`，填写已有模型配置及两个本地 MySQL 密码；不要覆盖已有模型凭据。

```bash
uv sync
docker compose -f compose.yaml -f compose.milvus.yaml up -d --wait
uv run python -m app.db.init_db
# 显式补齐质量表和消息引用列；自定义库名时相应替换 mewhelp。
uv run python -m app.db.migrate_rag_quality
uv run python -m app.db.migrate_citations --database mewhelp
# 新环境先建知识表/旧集合，再准备独立 CH04 集合并导入六份资料。
uv run python -m app.knowledge.cli init
uv run python -m app.knowledge.cli prepare-hybrid
uv run python -m app.knowledge.cli import-docs --mode hybrid --manifest data/kb/manifest.json
uv run python -m app.knowledge.cli sync --mode hybrid
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

浏览器访问 [客服聊天页](http://127.0.0.1:8000/)。页面与API同源，不需要npm或单独启动前端。CH05必须使用hybrid模式、单实例单worker及可写的`WORKFLOW_CHECKPOINT_PATH`，默认`.data/checkpoints/ch05.sqlite3`。上述初始化仅用于确认归属的新环境；已有环境按运行手册复用，不能为验收重复同步知识或清库。

在线知识来自 `data/kb` 六份权威文档，不再使用 FAQ LIKE；FAQ 表保留为旧数据。已有 CH03 数据升级时先阅读 [CH04 运行手册](docs/ch04-runbook.md)，明确重新切分与旧索引的关系。显式旧 `dense` 模式仍需版本匹配的 `data/knowledge/retrieval-policy.json`，历史操作见 [CH03 runbook](docs/ch03-runbook.md)。已有聊天不会被普通初始化覆盖；修改种子代码不会自动刷新已保存的数据。手动 SQL 的用途与执行顺序见 [sql/README.md](sql/README.md)。

依次发送：

1. `订单 1001 的物流到哪了`：同一条消息依次执行 `query_order` 和 `query_logistics`，自动使用订单结果里的物流单号，直接回答物流轨迹；页面依次显示订单与物流两个徽章。
2. `物流单号 SF213502378238 到哪了`：出现“调用了 query_logistics”徽章，回答引用返回的承运方和物流节点。
3. `退货政策是什么`：固定先检索退款政策，最高重排分达到0.3才进入回答，并显示可点击的引用编号。
4. `邮费是多少`：经混合检索查询 6 元基础运费、满 99 元包邮，并保留偏远地区 12 元附加费不参与包邮的例外；不再要求“邮费零命中”。
5. 对照 `运费是多少`：同样命中知识库，不再使用 10 元旧口径。
6. `我要投诉你们客服态度太差！`：显示独立的“转人工”和“创建工单”建议；确认弹窗提交后才写MySQL并返回真实工单号。取消或Esc不会建单。
7. `帮我查订单 2003`：该单属于另一个演示用户，工具返回 `order_not_owned`，页面直接回固定文本“没有找到您的这笔订单，请确认订单号或物流单号后重试。”，不经过模型改写。

订单、商品、物流在工具内部按入参生成可复现的演示数据，不连接公司业务系统，不建额外业务表。订单和物流共用以 `order:{order_id}` 为种子的独立 `random.Random`，商品使用 `product:{keyword}`；同一入参在相同代码和 Python 版本下，重复调用、交错查询、重启进程都返回相同数据。物流时间使用固定演示时间，不读取当前时钟；未发货订单没有承运方和物流单号。FAQ 来自 `data/kb` 构建的语义知识库；工单会真实写入本地数据库，但没有人工坐席自动接入。

每条消息最多执行4个业务工具，`TOOL_MAX_CALLS_PER_TURN`可进一步降低上限。支持连续决策和同批多个独立查询，工具均按顺序执行；订单→物流等有依赖的步骤使用前一步的实际结果。`query_logistics`只接受`tracking_no`；用户只给订单号时先查订单。身份由执行器注入，模型看不到也填不了；夹带身份字段会被拒绝。归属不符与订单不存在都返回`order_not_owned`和相同固定回复，不泄露订单是否存在。

同轮相同工具和参数不会重复执行。普通聊天只允许查询，不执行建单；确认建议后使用稳定动作ID提交，重复确认返回同一张票。达到调用上限、工具失败或检测到重复时停止继续调用；批次中未执行的项目会留下明确的跳过结果。已经提交的工单编号始终保留，执行器原有同一调用的有限网络重试继续保留。

文档示例订单 `1001` 固定为：1 件猫粮，单价和总额均为 39.9 元，订单状态“已发货”，发出地浙江省杭州市、目的地浙江省宁波市，物流状态“运输中”，节点“包裹已到达杭州转运中心”，物流单号 `SF213502378238`，节点时间 `2026-02-05T16:45:00+00:00`。这份工具结果由测试固定校验；模型的自然语言措辞仍可能变化。订单 `1002` 从杭州发往新疆乌鲁木齐，总额129元；订单 `1005` 计划从杭州发往成都，总额89元、待发货。订单工具使用 `shipping_origin`、`shipping_destination` 返回省市，待发货时表示计划起终点。

模型可见的工具说明使用业务语言，描述用途、参数要求、返回内容和操作边界，不包含 mock、随机种子等开发实现信息。当前工具显式设置 `@tool(description=...)`，这段说明和公开参数字段定义会发送给模型，普通代码注释不会发送。内部仍使用可复现的模拟数据；`is_mock` 保留在数据库审计和原状态协议中，`prepare_messages` 在两阶段的模型入口统一去除当前及历史工具结果外层标记，业务字段和错误事实保持原样。客服正文与页面徽章不展示模拟来源，提示词也不包含模拟环境信息。

新结果使用普通商品/承运方名称与 `SF` 单号；旧 `MOCK` 单号仍兼容查询并执行相同归属检查。旧会话里已经保存的客服“演示数据”文字不会被改写，验证新行为时应点击“新会话”。这项调整只统一模型接收的业务接口，不表示订单已接入真实业务系统。

## 页面行为与接口

可查询的订单来自有限演示清单：默认 `demo-user` 拥有 `1001`–`1007` 和 `A-1001`；`2003` 属于 `demo-other-user`。其他订单号均返回相同的归属拒绝；不是任意订单号都能生成可查询数据。`1005`、`1007` 为待发货示例，没有物流单号。

- Enter 发送，Shift+Enter 换行；保留中文输入法选字行为。
- 工具执行中、重试、成功、失败或未执行显示在当前客服气泡顶部的标签中，直接展示具体工具名（如“调用了 query_order”）；多个调用分别显示，同一次调用更新原标签。
- 业务回答按上游增量显示；知识先检索和过证据闸，之后同样按模型分片输出正文。页面排版Markdown标题、列表、加粗、引用、代码和表格；用户输入保留原文。Markdown库随本地静态资源提供，无需外部CDN；原始HTML不执行，图片显示替代文字。
- 已完成知识回答的合法 `[n]` 可打开证据面板和登记原文；旧快照保留，当前文档改变时明确提示版本差异。代码、已有链接及未知编号不会转成引用按钮。
- 已完成回答可选择一次赞或踩，仅保存在本浏览器 localStorage；反馈尚未上传到服务器，不参与问题池或评估计分。
- 停止回复会关闭本轮网络流并等待上游任务和短事务清理；该轮标为未完成，后续可以重新发问，不保证无缝续播。
- “转人工”只显示本地固定客服话术，没有真实坐席接入；“创建工单”是独立确认请求。重复确认使用原动作ID；状态未知时用原按钮核实，不换键重复提交。已返回真实票号但审计待同步时仍保留票号。
- 新会话和刷新页面生成新的 session_id；旧记录保留在数据库。此版本没有历史会话列表。
- `POST /api/chat` 仍接收 `session_id` 和 `message`，可选 `user_id` 默认 `demo-user`；标识上限 64 字符，问题上限 4000 字符。最终回答上限 12000 字符，超限停止生成并明确标为未完成。
- SSE保留文本增量、`[DONE]`、错误和工具状态事件；MySQL完成事务和SQLite最终状态均确认后发送`event: answer_metadata`（会话、轮次、状态、引用和建议），再发送DONE。中途停止或错误不会留下可执行的半成品建议。
- `POST /api/actions/tickets`只接收会话/用户、来源轮次、动作ID和`confirmed:true`；原问题与工单类型由服务端读取保存快照。跨归属或伪造动作返回404，输入不合法422，会话忙409，结果未查清503。
- `GET /api/chat/{session_id}/turns/{turn_id}/citations/{n}` 返回本轮证据快照；追加 `/source` 读取登记来源。两者沿用演示 `user_id` 归属检查，不是新增登录认证。
- `POST /api/extract` 保持原有结构化提取功能；可在 [接口文档](http://127.0.0.1:8000/docs) 查看。

CH05固定七类意图、四条路径：商品咨询/退款退货先检索和过证据闸；物流/订单/售后直接进入只读Agent；投诉和闲聊走固定出口。分类与Agent可以使用真实历史，原话不做指代改写。底层知识沿用CH04单轮检索改写与混合召回：每个子句Dense/BM25各50条，RRF合并50条，普通重排最多10条、退款政策最多30条完整证据。最高归一化重排分低于0.3或成功空召回时固定拒答，并将原话按`retrieval_low_conf`入池；故障和预算停止不入低证据池。最终正文独立流式生成，模型不能自行申请建单工具。

## 测试

CH04 提供72题（24校准、48固定验收）、四策略命令与独立 `mewhelp_ch04_eval` 资源。题集依据、筛选含义见 [评估题集说明](docs/ch04-evaluation-set.md)；真实准备、校准、比较及专用测试见 [CH04 运行手册](docs/ch04-runbook.md)。评估不写应用问题池或编造台账。

固定验收集（48题）最终源码候选 `85f1b1a` 的真实报告在 `reports/ch04/20261025-evaluation-02`，状态 `completed_with_errors`（192条中8条错误）。[收尾验收](docs/ch04-closeout.md)列出完整证据、分母、原始失败和独审修复。

| 策略 | Recall@10 | MRR@10 | 覆盖率 | 误拒率 | 无答案正确拒答 | 忠实度宏平均（有效数） | 错误 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.0000 | 0.8542 | 0.5833 | 0.3000 | 1.0000 | 1.0000（28） | 0 |
| bm25 | 1.0000 | 0.9833 | 0.6458 | 0.1750 | 1.0000 | 0.9667（30） | 3 |
| hybrid | 1.0000 | 0.9375 | 0.7083 | 0.1000 | 1.0000 | 1.0000（34） | 2 |
| hybrid_rerank | 1.0000 | 1.0000 | 0.5417 | 0.3000 | 1.0000 | 1.0000（25） | 3 |

召回/MRR分母为40道可答题，覆盖率分母48，无答案正确拒答为各8/8；拒答不进入忠实度分母。8条错误包括生成5、正文引用1、裁判2，全部保留。独审修复看板跨轮配对后，为最终源码版本补跑本轮；第一轮报告保留，没有按分数调题或调参。这些自动评估数字不等于全部回答正确。

对话使用两层近似预算：`TOKEN_BUDGET=12000` 限制历史消息、当前问题及工具事实；`REQUEST_TOKEN_BUDGET=16000` 另限制包含系统提示和本阶段工具定义的完整输入。ASCII 每四字符约一单位，非 ASCII 每字符约两单位，协议字段同样计入，JSON 保留真实 Unicode。它们不是模型实际分词数或上下文窗口；16000 也不含模型输出，需结合实际网关限制留出输出空间。历史只能按整轮裁剪；当前问题或工具结果仍超限时明确失败，不丢掉事实继续回答。较长的 FAQ 结果可能触及对话预算，应缩小问题范围或按模型容量同时调整两层配置。

离线测试在首次导入时禁用项目 `.env` 文件来源，并为实际使用配置的模块注入隔离配置；单文件与整套运行都不依赖本机模型凭据。配置读取测试使用虚构配置文件。非法工具申请会留作数据库审计，但不会执行，也不会以损坏的工具协议回传模型。

```bash
# 离线回归，不请求真实模型、不连接 MySQL。
CHAT_BASE_URL=http://test.invalid/v1 CHAT_MODEL=test-model CHAT_API_KEY=test-key uv run --offline --locked pytest -q
node --test tests/frontend/*.test.mjs

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
- `app/tools/`：工具注册、参数校验、业务工具5秒默认超时、知识查询独立60秒超时和最多1次重试。
- `app/services/chat.py`：有界多工具聊天编排、同会话互斥、取消与持久化。
- `app/core/`：模型工厂与适配、完整轮次历史裁剪、提示词。
- `app/static/`：纯文本安全渲染与工具轨迹，无外部 CDN。
- `tests/knowledge/`、`tests/knowledge_integration/`：离线知识规则测试与真实存储集成测试，共享替身集中在 `tests/knowledge/fakes.py`。
- [CH02 开发记录](dev-notes/ch02.md)、[CH02 历史评估](dev-notes/ch02-evaluation.md)。
- [CH03 开发记录](dev-notes/ch03.md)、[CH03 校准](dev-notes/ch03-calibration.md)、[CH03 验收](dev-notes/ch03-evaluation.md)、[CH03 运行手册](docs/ch03-runbook.md)。
- [CH04 开发记录](dev-notes/ch04.md)、[CH04 运行手册](docs/ch04-runbook.md)、[CH04 题集](docs/ch04-evaluation-set.md)。
- [CH05开发记录](dev-notes/ch05.md)、[CH05运行手册与真实验收](docs/ch05-runbook.md)。
- [CH07开发记录](dev-notes/ch07.md)、[CH07上下文管理运行与验收](docs/ch07-runbook.md)：三层历史、后台追加摘要和多会话回载；已有应用库启用前需按手册增量迁移。

当前 embedding 使用本地 BGE-M3。`EMBED_BASE_URL`、`EMBED_MODEL`、`EMBED_API_KEY` 仍是预留配置，填写它们不会切换编码器；云端接入与配置切换后续单独实现。

使用**单实例、单 worker**；不要启动多个服务同时使用同一数据库。启动时将旧 running 轮标为 cancelled，保留已提交工具事实；恢复失败会阻止启动。同会话重叠请求被拒绝，不同会话可以并发。

Compose 项目为 `mewhelp-ch02`，MySQL 默认只监听 `127.0.0.1:3307`，数据保存在命名卷。不要执行 `docker compose down -v`，除非确实要删除本地数据库。

这是本地演示系统，尚无用户鉴权、权限控制和限流。`user_id` 可以由客户端填写，所有默认访客共用 `demo-user`；知道同一会话号并指定匹配的演示身份，就能继续读取该会话。不要分享页脚的完整会话号，也不要存放真实个人或业务资料；会话号不能代替可靠的身份认证。开发脚本默认只监听 `127.0.0.1`，勿直接暴露到公网。多人使用前需要另行设计登录与权限隔离。不能直接退款、修改订单或查询真实物流。
