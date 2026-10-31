# SQL 脚本说明

当前应用实现到 CH10。业务和摘要表以 `app/models/customer_service.py` 为准，知识库表以 `app/models/knowledge.py` 为准；质量、审核、评估及主题表须按对应迁移核验。在线 FAQ 来自 `data/kb` 权威文档构建的知识库；`ch02-seed.sql` 与 `app/db/seed.py` 中的 6 条 FAQ 保留为旧演示数据。

## 当前应用初始化

新环境在项目根目录按 [首页启动顺序](../README.md#启动与演示)执行基础建表、增量迁移和 hybrid 知识准备。已有库跳过基础初始化，按下表补齐缺失结构，复用已有知识：

| 顺序 | 命令模块 | 执行方式 |
|---|---|---|
| 1 | `app.db.init_db` | 仅新环境建业务、摘要、退款和审计表并补种子；不升级旧表，不包含 CH09 质量表。 |
| 2 | `app.db.migrate_rag_quality`、`app.db.migrate_citations` | 显式补齐 CH04 两张质量表及消息引用列；直接执行，无预览模式。引用迁移需 `--database mewhelp`。 |
| 3 | `app.db.migrate_ch06` | 等待状态约束与独立退款申请表；预览 → `--apply` → 复预览。 |
| 4 | `app.db.migrate_ch07`、`app.db.migrate_ch08` | 按顺序补摘要及工具审计结构；各自预览 → `--apply` → 复预览。 |
| 5 | `app.db.migrate_ch09` | 审核队列、评估记录和问题池关联结构；预览 → `--apply` → 复预览。关闭观测或主题功能仍必需。 |
| 6 | `app.db.migrate_ch10` | 仅启用 `TOPIC_ENABLED=true` 时必需；预览 → `--apply` → 复预览。 |

CH06–CH10 命令均须传 `--env-file .env --database mewhelp`；自定义库名时同步替换并核对目标。每章复预览应返回空 `statements`，已兼容结构重复迁移不会清空数据。完整命令在首页，[CH09 操作手册](../docs/ch09-operations.md#ch09-表结构迁移)和 [CH10 操作手册](../docs/ch10-operations.md)说明章次要求。知识资源由 `app.knowledge.cli init`、`prepare-hybrid`、`import-docs --mode hybrid --manifest data/kb/manifest.json`、`sync --mode hybrid` 准备，不能以旧 CH03 dense 导入命令替代当前 hybrid 初始化。

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

## CH04–CH10 脚本

以下是已实现章节的 DDL 参考。当前应用的必需结构通过上方增量迁移准备，不直接把整组 SQL 反复导入：

| 脚本 | 前置条件与用途 |
|---|---|
| `ch04-ddl.sql` | 依赖第二章会话表，建立低置信度问题池和编造个案台账。 |
| `ch06-ticket-type.sql` | 历史工单类型设计，不是当前退款迁移入口；当前退款使用 `refund_applications`，执行 `app.db.migrate_ch06`。 |
| `ch07-ddl.sql`、`ch07-layers.sql` | CH07依赖会话/消息表；两份按顺序应用，增加摘要投影、两个锚点及只追加摘要表。已有部分结构时使用下方增量命令。 |
| `ch08-ddl.sql` | CH08 工具审计表；执行器以独立短事务写入五种中文状态。已有库使用下方增量迁移。 |
| `ch09-ddl.sql` | 依赖第四章问题池，建审核队列、评估记录并补充关联字段；当前启动必需。 |
| `ch10-ddl.sql` | 依赖第四章问题池，建主题分类表；启用主题功能时必需。 |

原始 `CREATE TABLE` / `ALTER TABLE` 脚本不应作为可任意重跑的初始化命令；已有部分结构优先使用对应迁移模块。

CH07已有库升级先预览，再显式应用；重复应用只补缺失结构，不清空原文或重写摘要：

```bash
uv run python -m app.db.migrate_ch07 --env-file .env --database mewhelp
uv run python -m app.db.migrate_ch07 --env-file .env --database mewhelp --apply
```

`create_all()`不会为旧会话表补列，升级已有库必须执行增量迁移。MySQL的DDL隐式提交，迁移中断后检查输出并重跑补齐；摘要业务行的追加和锚点推进则由仓储短事务原子提交。
# CH08 工具审计增量迁移

既有库使用 `uv run python -m app.db.migrate_ch08 --env-file .env --database <显式库名>` 先预览；确认目标后加 `--apply` 创建 `tool_audit_logs`。
迁移直接读取 `sql/ch08-ddl.sql`，不清表；已存在时核对字段类型、中文枚举、默认值、主键、三个普通索引和无外键约束。结构不兼容会拒绝执行，重复执行无变更。
审计为独立短事务，默认写入时限一秒；它不代替完整工具消息，也不承担业务防重复职责。
