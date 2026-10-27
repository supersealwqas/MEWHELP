# SQL 脚本说明

当前应用实现到第三章。第二章客服四表以 `app/models/customer_service.py` 为准，第三章知识库三表以 `app/models/knowledge.py` 为准。在线 FAQ 来自 `data/kb` 权威文档构建的知识库；`ch02-seed.sql` 与 `app/db/seed.py` 中的 6 条 FAQ 保留为旧演示数据。

## 当前应用初始化

推荐在项目根目录执行：

```bash
uv run python -m app.db.init_db
uv run python -m app.knowledge.cli init
uv run python -m app.knowledge.cli import-docs --manifest data/kb/manifest.json
```

第一条命令创建缺失的客服四表，并补充旧 FAQ、演示会话、工具调用消息和人工工单；FAQ 按固定 ID 补缺失项，不覆盖已有记录。第二条创建缺失的知识库三表，并检查或创建独立 Milvus 数据库与集合。第三条导入清单中的六份文档，发布知识原文并同步向量。连接配置、本地模型及检索策略的准备见项目根 [README](../README.md) 和 [CH03 runbook](../docs/ch03-runbook.md)。

如果需要手动检查或在空库中建立第二章客服四表，先选择目标数据库，再按顺序执行：

1. `ch02-ddl.sql`：建立 `conversations`、`messages`、`faq`、`tickets`，包含会话标识、轮次字段、工具结果唯一约束和状态校验。
2. `ch02-seed.sql`：写入 6 条 FAQ，不创建演示会话和工单。它会先删除这 6 个问题的同名记录再插入，其他 FAQ 保留；有其他记录时，全表行数可以超过 6。

第二章的代码初始化与手动 SQL 初始化择一使用；第三章仍需执行上述知识库初始化和资料导入。SQL FAQ 脚本使用自增 ID，代码使用固定 ID；在已有数据的库中交替执行，不能保证 ID 对应同一个问题。FAQ 演示内容包括 7 天无理由退货、48 小时发货、满 99 元包邮、未满 99 元收取 6 元运费等，不代表真实商城政策。

## 第三章脚本

| 脚本 | 当前用途 |
|---|---|
| `ch03-ddl.sql` | 建立知识片段、问答抽取暂存、抽取批次三表；只包含 MySQL 表结构。推荐使用上述 `app.knowledge.cli init` 同时准备 MySQL 与 Milvus。 |
| `ch03-seed.sql` | 可选的合成历史对话，依赖第二章表结构；补充两个固定会话及已完成问答轮次，供挖掘任务演示，不替代权威文档导入。运费示例为 6 元。 |
| `ch03-staging-review.sql` | 不适用于当前 CH03 链路，不应执行。它增加 `approved/rejected` 状态并把 `kept` 改为待人工审核；当前代码只支持 `extracted/kept/discarded`，没有该人工审核流程。 |

## 已有数据库

`CREATE TABLE` 脚本用于新建表，不是旧表升级脚本。`app.db.init_db` 使用的 `create_all()` 同样不会为已有表补列或补约束。已有库如果缺少 `session_id`、`turn_id` 等字段，需要结合存量记录另做迁移，不能直接重复建表。

更新第二章种子文件不会自动刷新数据库里的旧 FAQ。需要重建第二章测试数据时，先停止使用测试库的服务，再按项目根 README 的流程重建 `mewhelp_test`；不要为刷新 FAQ 清空应用库。第三章资料更新后重新执行上述 `import-docs` 命令；知识库测试使用独立的 `mewhelp_ch03_test`，具体流程见 CH03 runbook。

## 后续章节

以下脚本保留作后续章节使用，不需要为了运行当前应用而全部导入：

| 脚本 | 前置条件与用途 |
|---|---|
| `ch04-ddl.sql` | 依赖第二章会话表，建立低置信度问题池和编造个案台账。 |
| `ch06-ticket-type.sql` | 第六章再启用“退款”工单类型；当前代码只支持售后、投诉、咨询。 |
| `ch07-ddl.sql`、`ch07-layers.sql` | CH07依赖会话/消息表；两份按顺序应用，增加摘要投影、两个锚点及只追加摘要表。已有部分结构时使用下方增量命令。 |
| `ch08-ddl.sql` | 建工具审计表，当前工具执行器尚未写入该表。 |
| `ch09-ddl.sql` | 依赖第四章问题池，建审核队列、评估记录并补充关联字段。 |
| `ch10-ddl.sql` | 依赖第四章问题池，建主题分类表。 |

后续 `CREATE TABLE` / `ALTER TABLE` 脚本按对应章节执行一次，不应作为可任意重跑的初始化命令。

CH07已有库升级先预览，再显式应用；重复应用只补缺失结构，不清空原文或重写摘要：

```bash
uv run python -m app.db.migrate_ch07 --env-file .env --database mewhelp
uv run python -m app.db.migrate_ch07 --env-file .env --database mewhelp --apply
```

`create_all()`不会为旧会话表补列，升级已有库必须执行增量迁移。MySQL的DDL隐式提交，迁移中断后检查输出并重跑补齐；摘要业务行的追加和锚点推进则由仓储短事务原子提交。
