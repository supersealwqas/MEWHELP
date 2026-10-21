# CH03 知识库运行手册

## 环境

- Python 3.12、uv；本地 BGE-M3 位于 `models/bge-m3`，不重复下载。
- MySQL 8.4（Compose，映射 `127.0.0.1:3307`）；应用库 `mewhelp`。
- 已有 Milvus 2.4.10（`127.0.0.1:19530`）；本项目只用独立数据库 `mewhelp_ch03` 的 `knowledge` 集合，不触碰原项目数据。
- 测试库：MySQL `mewhelp_ch03_test`、Milvus `mewhelp_ch03_test`；代码会同时校验两个库名，拒绝半边测试半边应用。
- 空数据卷首次启动 MySQL 时，`docker/mysql/init/02-ch03-test-db.sql` 自动创建测试库并授权；已有数据卷需由管理员执行该文件中的语句，不要删除已有数据库。
- 先复制 `.env.example` 到 `.env` 并填写本地凭据；不要提交真实 `.env`。

## 当前链路验证流程

当前交付只保证调用、SSE、落库及恢复链路。检索与生成质量已按用户要求暂缓，不能把下述通过结果称为13/7业务质量验收通过。

2026-10-21 最新实测：51 个 active/done 知识块与当前源文件一致，MySQL/Milvus 的 51 组主键及内容/编码指纹全部匹配；正式 `app.main.database_runtime` 启动和资源关闭通过。真实 FAQ、无工具问候、SSE/DONE、当前轮落库及空白请求拒绝通过，详见 [链路报告](../dev-notes/ch03-pipeline.md)；同主键中断重跑及 done 缺向量补齐通过，详见 [恢复报告](../dev-notes/ch03-recovery.md)。最新默认回归 537 passed、38 skipped、21 deselected，前端 28 passed；38 条真实知识集成已在本阶段最终导入前单独全部通过，默认跳过项不重复计为通过。

```bash
uv sync --locked
docker compose up -d --wait mysql
# 确认已有 Milvus 2.4.10 正常
docker ps --format '{{.Names}}\t{{.Image}}'

# 集成夹具会清测试知识，必须在最终导入和演示验收前执行。
PYTHONUTF8=1 uv run --no-sync pytest --run-knowledge-integration -m knowledge_integration tests/knowledge_integration -q
# 首次准备或文档更新时执行；已有51块且hash匹配时，不必重复导入。
PYTHONUTF8=1 uv run --no-sync python -m app.knowledge.cli init --test
PYTHONUTF8=1 uv run --no-sync python -m app.knowledge.cli import-docs --test --manifest data/kb/manifest.json
PYTHONUTF8=1 uv run --no-sync python tests/verify_ch03_live.py pipeline --test --report dev-notes/ch03-pipeline.md
PYTHONUTF8=1 uv run --no-sync python tests/verify_ch03_live.py recovery --test --report dev-notes/ch03-recovery.md
PYTHONUTF8=1 uv run --no-sync python -m app.knowledge.cli status --test
PYTHONUTF8=1 uv run --no-sync pytest -q
node --test tests/frontend/chat-api.test.mjs
```

`--test` 同时切换 MySQL 与 Milvus 测试库，并核对实际 `SELECT DATABASE()`。`pipeline`加载实际交付的policy，先验证已导入语料的source/ordinal/section/hash和active/done状态，再启动独有本机临时TCP端口；验证合法FAQ调用、无工具问候、SSE/DONE/落库一致和空白请求拒绝，不运行答案质量规则。导入会逐批进行文档关系判定，耗时取决于真实模型请求，不能仅凭一段时间没有标准输出判断死锁。

当前`retrieval-policy.json`是**链路基线策略，未完成质量校准**：原`min_score=0.6256920099258423`保持不变，编码版本保持，切分版本匹配`ch03-v3`；`calibration_digest`以`pipeline-baseline-not-quality-calibrated`标明性质和旧策略来源。它解决版本不匹配导致的启动拒绝，不承诺召回或回答质量。

## 后续可选：检索与生成质量校准

```bash
# 关系策略先经过固定标注评估；失败不会覆盖旧策略。
PYTHONUTF8=1 uv run --no-sync python tests/eval_ch03_prompts.py --suite relation --report dev-notes/task-5-relation.json
PYTHONUTF8=1 uv run --no-sync python tests/verify_ch03_live.py calibrate --test --report dev-notes/task-5-calibration.md
PYTHONUTF8=1 uv run --no-sync python tests/verify_ch03_live.py evaluate --test --report dev-notes/task-5-evaluation.md
```

`calibrate`只清测试知识表和测试向量，不删除应用库或聊天消息，随后重新导入文档。当前已有质量失败记录，应后续单独优化，不改写成通过。

校准先用明确标记的低阈值诊断策略，经原 `POST /api/chat` 获取实际工具参数；只对合法且成功执行的`executed_keyword`取Dense分数，失败/被拒绝的申请不采分、不能用于发布策略。无工具负例的只读探针单独标记。逐一检查必要事实与例外的支持块，正式策略下全部校准SSE回归通过才写检索policy。`evaluate`同样走真实TCP SSE和原工具链，独立固定问法不参与阈值选择；`退款审核要几天`已进入校准，不再算独立验收。

报告保留原问、原始及有效keyword、工具申请与结果、raw/有效/截分Top3、逐事实覆盖、每帧耗时、当前轮消息终态和最终回答。HTTP200或直接Dense命中均不能单独算验收通过：还需要成功工具状态、完整增量、唯一末尾 `[DONE]`、无error、整轮completed及落库答案一致。无关负例允许不调用工具，但不能编造资料外结论。

## 应用库初始化与启动

当前聊天支持同轮连续调用工具，默认 `TOOL_MAX_CALLS_PER_TURN=4`（允许1至8）。发送“订单1001的物流到哪了”应在同一回复内依次显示订单查询和物流查询，自动取用订单中的单号；无需用户再发送物流单号。待发货订单应结束于订单查询，不编造物流轨迹。每条消息最多尝试创建一张工单，重复或失败时停止后续调用。

```bash
uv run python -m app.db.init_db
uv run python -m app.knowledge.cli init
uv run python -m app.knowledge.cli import-docs --manifest data/kb/manifest.json
uv run python -m app.knowledge.cli status
TOKEN_BUDGET=12000 REQUEST_TOKEN_BUDGET=16000 \
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

应用启动会加载共享 BGE-M3、Milvus 客户端和版本匹配的检索策略；缺失策略、模型、Milvus 集合或知识原文时明确失败，不会回退 LIKE 或空命中。`data/knowledge/retrieval-policy.json` 的 `embedding_revision` 与 `chunking_version` 必须匹配当前实现（`ch03-v3`）。应用导入前也必须具备通过真实固定标注评估的版本匹配关系策略，不能只改JSON中的版本字符串。

## 历史对话挖掘

```bash
# 单次（先恢复 pending，再领取新来源，去重并同步）
uv run python -m app.knowledge.cli mine-once --test
# 常驻定时：默认北京时间 02:00，启动先恢复
uv run python -m app.knowledge.cli mine-schedule --test
# 只读状态
uv run python -m app.knowledge.cli status --test
# 只补齐/清理向量
uv run python -m app.knowledge.cli sync --test
```

生产使用去掉 `--test`，并保持单 worker；多个写任务通过 MySQL 命名锁互斥。

## 故障与恢复

- 索引完全为空但 MySQL 仍有 active 原文：在线查询明确失败，不再伪装为“没有相关政策”；执行 `uv run python -m app.knowledge.cli sync` 修复后再查询。真正空库或只有已撤回知识时允许空结果，正常低分未命中不受影响。
- MySQL 已 pending、编码前中断：重新执行 `sync`/`import-docs`/`mine-once` 会按主键捡起。
- Milvus upsert 后回填前中断：重跑以同一主键全量 upsert，再条件回填 done。
- MySQL仍done但Milvus向量丢失：同步会按MySQL权威指纹补回。恢复报告逐阶段保存同一主键、三格正文、内容hash、编码输入hash、vector_id、active/pending/done及实际查询结果，恢复演示结束只清理本次临时source。
- 文档撤回/隔离后，旧向量即使删除失败也不会被在线返回；重跑 `sync` 继续清理。
- 文档默认权威：`data/kb` 六份文档覆盖基础运费、偏远附加费、退款到账口径、会员权益、型号和清洗限制；代码种子、当前验收和演示说明不得回写相反事实。
- 历史 CH02 报告中的 10 元与“邮费零命中”只作历史记录，不是当前验收答案。
