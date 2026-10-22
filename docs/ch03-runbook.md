# CH03 知识库运行手册

## 环境

- Python 3.12、uv；本地 BGE-M3 位于 `models/bge-m3`，不重复下载。锁定 FlagEmbedding 1.4.2、PyTorch 2.11.0+cu128，编码使用 `cuda:0`，没有 CPU 降级开关。
- MySQL 8.4（Compose，映射 `127.0.0.1:3307`；2026-10-22 本机实测 8.4.13）；应用库 `mewhelp`。
- Milvus 服务端基线为 **2.6.22**（`compose.milvus.yaml`，`127.0.0.1:19530`），客户端保持 **pymilvus 2.4.15**（`pyproject.toml` 与 `uv.lock`）。本项目只用独立数据库 `mewhelp_ch03` 的 `knowledge` 集合，不触碰原项目数据。
- Milvus 依赖 etcd `v3.5.16`，MinIO 固定为本机已运行镜像的 `sha256:ad134b2dd572ae0d989d6807c3e45516cc75a34883a8021db2ff04be5d1d496a`，不再随 `latest` 变化。
- 测试库：MySQL `mewhelp_ch03_test`、Milvus `mewhelp_ch03_test`；代码会同时校验两个库名，拒绝半边测试半边应用。
- 空数据卷首次启动 MySQL 时，`docker/mysql/init/02-ch03-test-db.sql` 自动创建测试库并授权；已有数据卷需由管理员执行该文件中的语句，不要删除已有数据库。
- 先复制 `.env.example` 到 `.env` 并填写本地凭据；不要提交真实 `.env`。

## 部署版本核对

2026-10-22 本机只读核验：pymilvus 2.4.15 的 `get_server_version()` 返回 2.6.22，列数据库、列集合、读取集合描述和索引均成功；应用集合描述为 `mewhelp-ch03:knowledge:v1`。已安装 SDK 也提供建库、切库与分页查询接口。客户端和服务端版本不同本身不是不兼容证据；这些只读结果也不能替代下面的真实写入、检索和恢复验证。

本次部署基线采用现有 2.6.22 容器与当前客户端锁定值，不包含客户端升级、旧 2.4.10 数据卷迁移或原项目资源重建。历史 2.4.10 验证记录保留原日期，不作为当前组合已通过的证明。需要全新启动本机依赖时使用 `docker compose -f compose.yaml -f compose.milvus.yaml up -d --wait`；已有实例先核对端口与数据卷归属，不要同时在 19530 端口启动第二套服务。

以下核对不创建集合、不导入知识，也不调用聊天模型：

```bash
docker compose -f compose.yaml -f compose.milvus.yaml config --quiet
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
PYTHONUTF8=1 uv run --no-sync python - <<'PY'
from importlib.metadata import version
from app.config import Settings
from pymilvus import MilvusClient

config = Settings()
client = MilvusClient(uri=config.milvus_uri,
                      token=config.milvus_token.get_secret_value(), timeout=5.0)
try:
    print({"pymilvus": version("pymilvus"),
           "milvus": client.get_server_version(timeout=5.0),
           "torch": version("torch"), "FlagEmbedding": version("FlagEmbedding")})
finally:
    client.close()
PY
```

## 当前链路验证流程

当前交付只保证调用、SSE、落库及恢复链路。检索与生成质量已按用户要求暂缓，不能把下述通过结果称为13/7业务质量验收通过。

2026-10-21 历史实测：51 个 active/done 知识块与当时源文件一致，MySQL/Milvus 的 51 组主键及内容/编码指纹全部匹配；正式 `app.main.database_runtime` 启动和资源关闭通过。真实 FAQ、无工具问候、SSE/DONE、当前轮落库及空白请求拒绝通过，详见 [链路报告](../dev-notes/ch03-pipeline.md)；同主键中断重跑及 done 缺向量补齐通过，详见 [恢复报告](../dev-notes/ch03-recovery.md)。当时默认回归 537 passed、38 skipped、21 deselected，前端 28 passed；38 条真实知识集成已在该阶段最终导入前单独全部通过，默认跳过项不重复计为通过。当前部署组合的回归结果应另附执行日期、命令和实际通过数量。

```bash
uv sync --locked
docker compose up -d --wait mysql
# 确认已有 Milvus 2.6.22 正常；客户端版本按上一节核对
docker ps --format '{{.Names}}\t{{.Image}}'

# 集成夹具会清测试知识、向量、会话、消息与工单，必须在最终导入前执行。
PYTHONUTF8=1 uv run --no-sync pytest --run-knowledge-integration -m knowledge_integration tests/knowledge_integration -q
# 首次准备或文档更新时执行；已有51块且hash匹配时，不必重复导入。
PYTHONUTF8=1 uv run --no-sync python -m app.knowledge.cli init --test
PYTHONUTF8=1 uv run --no-sync python -m app.knowledge.cli import-docs --test --manifest data/kb/manifest.json
PYTHONUTF8=1 uv run --no-sync python tests/verify_ch03_live.py pipeline --test --report dev-notes/ch03-pipeline.md
PYTHONUTF8=1 uv run --no-sync python tests/verify_ch03_live.py recovery --test --report dev-notes/ch03-recovery.md
PYTHONUTF8=1 uv run --no-sync python -m app.knowledge.cli status --test
PYTHONUTF8=1 uv run --no-sync pytest -q
node --test tests/frontend/*.test.mjs
```

`--test` 同时切换 MySQL 与 Milvus 测试库，并核对实际 `SELECT DATABASE()`。`pipeline`加载实际交付的policy，先验证已导入语料的source/ordinal/section/hash和active/done状态，再启动独有本机临时TCP端口；验证合法FAQ调用、无工具问候、SSE/DONE/落库一致和空白请求拒绝，不运行答案质量规则。导入会逐批进行文档关系判定，耗时取决于真实模型请求，不能仅凭一段时间没有标准输出判断死锁。

空测试库首次完整导入若仍切分为 51 块，会生成 1275 对文档关系，每次模型请求最多处理 8 对；成功完成至少需要 160 次关系模型请求，输入预算限制还可能增加次数。`publish_documents` 对完全未变化的已有文档会提前返回；清空测试库后无法复用这个快速路径。执行全量导入前应把这部分调用量计入预算。

真实检查按以下范围执行，避免把测试通过误认为所有依赖都已验证：

| 入口 | 实际检查 | 写入与清理范围 |
| --- | --- | --- |
| `pytest --run-knowledge-integration -m knowledge_integration tests/knowledge_integration` | 真实 MySQL/Milvus；编码器和模型为测试替身，不验证 GPU 或上游聊天 | 仅 `mewhelp_ch03_test`：夹具建缺失表，清空三张知识表、`tickets`、`messages`、`conversations`；Milvus 夹具在前后清空 `knowledge` 向量。归属测试创建并删除本次唯一的 `foreign_*` 集合 |
| `app.knowledge.cli init --test` | 初始化测试知识表与集合，不加载模型 | 建缺失表、数据库与集合；遇到归属不明的既存空库按下节明确确认，不要先手工创建空库 |
| `app.knowledge.cli import-docs --test` | 本地 GPU 编码、文档关系模型、MySQL 发布与 Milvus 同步 | 写测试知识与向量，并调用已配置的上游模型，可能产生费用 |
| `verify_ch03_live.py pipeline --test` | 已导入语料、GPU、真实双库与聊天 SSE | 建缺失表、恢复测试库既存未完成轮次、新增会话与消息；真实工具仍可执行。调用上游模型并写报告，不自动清理会话 |
| `verify_ch03_live.py recovery --test` | 本地 GPU、同主键写入中断与缺向量修复 | 发布本次唯一临时来源并删写其向量；同步同时处理测试库其他待恢复/失效记录，最后撤回本次来源并写报告 |
| `app.knowledge.cli status --test` | MySQL 知识状态计数 | 只读，不证明 Milvus 或 GPU 可用 |

以上 CLI 与真实验收脚本均从 `.env` 读取凭据，两个测试库名称必须同为 `mewhelp_ch03_test`。`pytest` 必须显式传入 `--run-knowledge-integration`，默认跳过的项目不计为通过。测试库与演示语料共用这一名称时，先完成清理型集成测试，再导入最终语料；保留测试库数据有要求时，先安排备份或另一套隔离服务，不能只改一个库名绕过校验。

单独核对本地 GPU 编码可运行以下命令。它不连接数据库、不调用聊天模型，离线开关阻止缺失权重时下载；应输出 `dimension: 1024` 且所有数值有限：

```bash
PYTHONUTF8=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python - <<'PY'
import asyncio
import math
from app.config import Settings
from app.knowledge.embedding import BgeM3Encoder

async def main():
    """验证本地 GPU 编码并关闭资源；无参数；返回 None；加载/编码错误直接抛出。"""
    encoder = BgeM3Encoder(Settings())
    try:
        vectors = await encoder.encode(["知识库部署基线检查"])
        assert len(vectors) == 1 and len(vectors[0]) == 1024
        assert all(math.isfinite(value) for value in vectors[0])
        print({"dimension": len(vectors[0]), "finite": True})
    finally:
        await encoder.aclose()

asyncio.run(main())
PY
```

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

Milvus 初始化只会直接使用本次成功新建的数据库，或已有且归属标记、结构校验通过的 `knowledge` 集合。目标库已经存在但没有集合时，归属无法从库名确定，`init` 会在创建集合前停止。先核实这个空库属于本项目，再明确指定其准确名称：

```bash
# 已核实归本项目的既存空应用库
uv run python -m app.knowledge.cli init --confirm-milvus-database mewhelp_ch03
# 已核实归本项目的既存空测试库
uv run python -m app.knowledge.cli init --test --confirm-milvus-database mewhelp_ch03_test
```

建库成功但创建集合中断时，重跑也走上述确认入口。确认参数只用于 `init`，不能绕过已有其他集合、目标集合归属不符或结构不兼容的检查；普通启动不接受确认。

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
