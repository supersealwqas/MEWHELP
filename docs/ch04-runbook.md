# CH04 运行与验证手册

本章在 CH03 基础上增加混合检索、本地重排、生成自评、问题池、编造台账、引用溯源及四策略评估。默认运行模式为 `hybrid`，旧 `dense` 路径作为显式兼容入口。以下应用升级命令与独立测试命令各有明确目标；本次开发只在独立测试环境执行数据库操作。

## 环境与数据边界

- Python 3.12、uv、MySQL 8.4、Milvus 2.6.22，客户端 PyMilvus 2.6.17。
- 编码使用本地 BGE-M3，重排使用本地 `bge-reranker-large`；独立工作区应配置两个模型的绝对路径，不复制模型或密钥到 Git。
- 应用库沿用本地 `.env`；CH04 测试库固定 `mewhelp_ch04_test`，评估库固定 `mewhelp_ch04_eval`。测试和评估不能使用 `mewhelp`、`mewhelp_test` 或 `mewhelp_ch03_test`。
- `app.db.init_db` 只负责原四张业务表；质量表使用独立元数据和显式迁移，不随旧业务初始化/测试重置处理。
- 此版本仍为单实例、单 worker 的本地演示，订单/商品/物流为可复现演示数据，工单保存到本地数据库。

## 已有应用的核心升级顺序

在目标工作区的 Git Bash 中执行，先确认 `.env` 指向正确应用库和模型目录。以下是显式操作命令；本次开发测试不会替用户执行应用库迁移。

```bash
# 检查现有知识状态，不加载模型、不修改数据。
uv run python -m app.knowledge.cli status

# 两张质量表：只创建缺失表，已有结构不兼容时明确停止。
uv run python -m app.db.migrate_rag_quality

# 消息引用快照列；请显式填写实际应用库名。
uv run python -m app.db.migrate_citations --database mewhelp

# 保留旧集合，将现有 MySQL 权威记录同步到独立 knowledge_ch04 集合。
uv run python -m app.knowledge.cli prepare-hybrid

# 仅当确实需要使用新字符切分重新导入时执行；准备索引本身不会自动重新切分。
uv run python -m app.knowledge.cli import-docs --mode hybrid --manifest data/kb/manifest.json
uv run python -m app.knowledge.cli sync --mode hybrid
```

`prepare-hybrid` 在知识写锁内进行，校验目标集合结构和归属，不删除旧集合。已有空库需要具名 `--confirm-milvus-database`，确认值必须与实际目标库相同；不能从“库是空的”推断它归本项目所有。

仅保留旧集合不足以保证重新切分后的数据可以即时回退：两个模式共用 MySQL 权威原文，重新导入会改变块和指纹。旧 Dense 模式应在匹配的旧原文/编码版本快照上使用；不得宣称只改环境变量就一定能恢复任意历史状态。

## 检索与回答规则

- 向量和关键词两路各最多50条，经RRF合并最多50条；每个有效子句分别重排，普通结果默认最多10条。
- 退款政策三条查询各取结果并去重，最多30条；仅查询政策，不办理退款。
- 重排最高分低于0.3时返回空知识结果。RRF分数与重排分数含义不同，不混用阈值；0.26预留参数不是已实现的四信号质量闸门。
- 有效知识回答先校验本轮完整证据及编号，再保存并输出。只有生成自评 `useful=false` 才保存原话到问题池；检索/模型故障不当作知识不足入池。
- 证据先按检索顺序编号，进入每个知识区时按首尾顺序摆放，十条为 `[1,3,5,7,9,10,8,6,4,2]`。只调整模型输入的位置，不改变持久化原文或编号；多个工具结果各自排列，不声称跨工具重新评分。
- 已保存的工单回执保留；知识不足时保留已查到的订单等业务事实，再明确说明知识不足。

## 引用与反馈

完成事务提交后，SSE发送 `answer_metadata`，包含 `session_id`、`turn_id`、`status`、`citations`，随后发送DONE。`messages.rag_metadata` 保存模型所见的全部证据，不仅保存正文引用子集；历史未存元数据的消息不伪造引用。

引用接口为 `/api/chat/{session_id}/turns/{turn_id}/citations/{n}`，追加 `/source` 读取白名单来源。未知轮次、未完成回答和归属不符均返回404。来源文件改变时展示旧证据与当前全文的版本差异；未证明与当前原文一致的历史证据不填写版本号。问答挖掘来源不暴露私人对话。

页面仅将合法正文编号转换成按钮；代码、已有链接和无效编号保持原文。证据面板可用Escape关闭。赞/踩每条完成回答只能选择一次，保存在本浏览器localStorage，不发送后端、不触发问题池、不参与模型评估。当前依然使用演示身份，不具备生产登录鉴权。

## 四策略独立评估

题集共72题，六桶各4道校准、8道固定验收，详见 [题集说明](ch04-evaluation-set.md)。`validate` 检查原文切分和证据定位；`prepare` 将冻结资料写入专用评估库；`calibrate` 只用24题检索成绩确定各策略门槛；`compare` 使用冻结门槛完成48题的四策略检索、生成与独立裁判。

数据库固定为 MySQL/Milvus `mewhelp_ch04_eval`，集合固定为 `knowledge_ch04_eval`。数据库账号需在该MySQL库拥有CREATE、SELECT、INSERT、UPDATE、DELETE、REFERENCES、INDEX权限（后两项用于知识块的自引用外键与查询索引），可由管理员预先建立空库。运行会校验身份标记、语料与模型指纹，不接管外来表、不覆盖其他语料版本。准备中断可用同一语料重试，但每次报告必须使用新的目录。

```bash
uv run python -m app.evaluation.cli validate
uv run python -m app.evaluation.cli prepare --target eval --output-dir reports/ch04/prepare-01
uv run python -m app.evaluation.cli calibrate --split calibration --output-dir reports/ch04/calibration-01
uv run python -m app.evaluation.cli compare --split evaluation --policy reports/ch04/calibration-01/policy.json --output-dir reports/ch04/evaluation-01
```

独立工作区可在每个命令后加 `--env-file <原配置路径>`；prepare/calibrate/compare另支持 `--mysql-port`、`--milvus-uri`、`--embedding-model-path`、`--reranker-model-path`，这些显式参数覆盖环境文件但不修改文件。程序从本地加载权重；离线环境设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 防止下载。

每个运行目录保存manifest、冻结语料、逐题results.jsonl和JSON/Markdown摘要。校准目录另有policy.json。四种分数各自选门槛，无法分离时保存null，不把重排0.3或预留0.26当作其他策略阈值。源码、题集、模型、Prompt或预算变化后旧校准文件失效，必须重新校准。

Recall按必需证据组覆盖率计，组内任选、组间同时需要；MRR只表示首条相关证据位置，不表示证据齐全。报告保留每个分母、错误及拒答；检索故障仍计入可答题召回分母，拒答忠实度为null，裁判故障不算满分。普通hybrid_rerank最多10个重排名次，退款最多30，故其Recall@50不代表实际提供50条重排结果。自动裁判使用单独模型调用，不能代替人工标注核查。

## 核心测试命令

```bash
# 离线：不连接数据库或真实模型。
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 CHAT_BASE_URL=http://test.invalid/v1 CHAT_MODEL=test-model CHAT_API_KEY=test-key uv run --offline --locked --no-sync pytest -q
node --test tests/frontend/*.test.mjs

# 专用 MySQL：仅创建必要缺表和本例随机记录，清理本例记录，不清库。
MEWHELP_RUN_QUALITY_MYSQL=1 uv run --no-sync pytest tests/rag/test_quality_mysql.py -q

# 独占测试 Milvus 集合；不覆盖旧集合。
MEWHELP_RUN_RAG_INTEGRATION=1 uv run --no-sync pytest tests/rag/test_hybrid_live.py -q

# 本地重排权重，禁止联网下载；独立工作区可用 MEWHELP_LOCAL_RERANK_MODEL 指定绝对路径。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MEWHELP_RUN_RERANK_LIVE=1 uv run --no-sync pytest tests/rag/test_rerank_live.py -q

# 实际聊天模型自评和独立裁判；会调用已配置网关。
MEWHELP_RUN_QUALITY_MODEL=1 uv run --no-sync pytest tests/rag/test_quality_model_live.py -q
```

真实质量测试可通过 `MEWHELP_TEST_ENV_FILE` 指定既有配置文件；不会把该文件内容写入报告。并行测试使用各自独占 `--basetemp`，避免 Windows 临时目录相互影响。

## 本次验证记录

2026-10-23，初始化隔离修复后的完整离线回归：827通过、47跳过、25排除；前端53通过。真实网关自评/裁判专项1通过（12.19秒）。这是一轮阶段验证，不能替代后续引用/评估实现后的最终完整验证。

首次真实MySQL检查3项因本地3307拒绝连接失败；Docker当时未运行，启动后报本地Secrets Engine套接字错误并退出。数据库环境恢复与后续真实集成结果将单独记录。

Docker恢复后原容器、镜像、卷均为空，日志曾记录GUI恢复出厂设置。根代理只重命名保留了故障套接字目录，没有执行出厂重置，也未恢复旧数据库。为验证新建独立Compose项目 `mewhelp-ch04-verification`：MySQL端口13307，Milvus端口19540，健康端口19091，四个容器均健康。项目代码、配置和本地模型仍在宿主目录。

真实空库检查发现测试父表准备顺序问题，补齐每例独立准备后MySQL3项通过（1.97秒）；真实Milvus中文BM25/RRF专项1项通过。旧失败日志保留，不以重跑结果掩盖缺陷。新的引用迁移及最终真实聊天验证见后续验收记录。
