# CH03 知识库运行手册

## 环境

- Python 3.12、uv；本地 BGE-M3 位于 `models/bge-m3`，不重复下载。
- MySQL 8.4（Compose，映射 `127.0.0.1:3307`）；应用库 `mewhelp`。
- 已有 Milvus 2.4.10（`127.0.0.1:19530`）；本项目只用独立数据库 `mewhelp_ch03` 的 `knowledge` 集合，不触碰原项目数据。
- 测试库：MySQL `mewhelp_ch03_test`、Milvus `mewhelp_ch03_test`。
- 先复制 `.env.example` 到 `.env` 并填写本地凭据；不要提交真实 `.env`。

## 测试库流程

```bash
uv sync --locked
docker compose up -d --wait mysql
# 确认已有 Milvus 2.4.10 正常
docker ps --format '{{.Names}}\t{{.Image}}'

uv run python -m app.knowledge.cli init --test
uv run python -m app.knowledge.cli import-docs --test --manifest data/kb/manifest.json
uv run python tests/verify_ch03_live.py calibrate --test --report dev-notes/ch03-calibration.md
uv run python tests/verify_ch03_live.py evaluate --test --report dev-notes/ch03-evaluation.md
uv run python tests/verify_ch03_live.py recovery --test --report dev-notes/ch03-recovery.md
uv run python -m app.knowledge.cli status --test
```

`--test` 同时切换 MySQL 与 Milvus 测试库；工具拒绝半边测试半边应用库。`calibrate` 会清空测试库知识表和测试 Milvus 向量后重新导入六份文档，生成版本匹配的 `data/knowledge/retrieval-policy.json`。

## 应用库初始化与启动

```bash
uv run python -m app.db.init_db
uv run python -m app.knowledge.cli init
uv run python -m app.knowledge.cli import-docs --manifest data/kb/manifest.json
uv run python -m app.knowledge.cli status
TOKEN_BUDGET=12000 REQUEST_TOKEN_BUDGET=16000 \
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

应用启动会加载共享 BGE-M3、Milvus 客户端和版本匹配的检索策略；缺失策略、模型、Milvus 集合或知识原文时明确失败，不会回退 LIKE 或空命中。`data/knowledge/retrieval-policy.json` 的 `embedding_revision` 与 `chunking_version` 必须匹配当前实现（`ch03-v2`）。

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

- MySQL 已 pending、编码前中断：重新执行 `sync`/`import-docs`/`mine-once` 会按主键捡起。
- Milvus upsert 后回填前中断：重跑以同一主键全量 upsert，再条件回填 done。
- 文档撤回/隔离后，旧向量即使删除失败也不会被在线返回；重跑 `sync` 继续清理。
- 文档默认权威：`data/kb` 六份文档覆盖基础运费、偏远附加费、退款到账口径、会员权益、型号和清洗限制；代码种子、当前验收和演示说明不得回写相反事实。
- 历史 CH02 报告中的 10 元与“邮费零命中”只作历史记录，不是当前验收答案。
