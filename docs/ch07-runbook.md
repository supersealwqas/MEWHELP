# CH07 会话上下文管理运行与验收

本章保留完整聊天原文，只缩减发送给模型的历史副本。默认窗口适合日常多轮聊天；小窗口演示会依次把早期原文移入半压层，再由后台追加摘要。历史侧栏可切换会话、完整回载并继续聊。

实现与测试在 `D:/AI/my_AI_project/MEWHELP` 的 `codex/ch07-session`。继续沿用单进程、单 worker、MySQL、官方 SQLite checkpoint 和现有知识检索资源。只管理同一会话，不增加跨会话画像。过程见 [开发记录](../dev-notes/ch07.md)，要求见 [设计](superpowers/specs/2026-10-28-ch07-context-design.md)。

## 启动前准备

使用 Git Bash 和原项目的 uv 环境；依赖版本沿用 `uv.lock`，已有知识与模型权重按 [CH06 运行手册](ch06-runbook.md) 复用。`.env` 保存现有连接配置，不把密钥写进命令或报告。

2026-10-28 已只读核查应用库 `mewhelp`：尚缺三个会话字段、摘要表及索引。本次只迁移和验收指定测试库，没有把本章部署到应用库。应用库启用本章前，需要由部署者先预览、再显式执行增量迁移：

```bash
cd D:/AI/my_AI_project/MEWHELP
uv run --no-sync --offline python -m app.db.migrate_ch07 --env-file .env --database mewhelp
uv run --no-sync --offline python -m app.db.migrate_ch07 --env-file .env --database mewhelp --apply
```

迁移只补缺失结构，不重建或清空已有消息；再次预览应返回空语句列表。SQL 说明见 [sql/README.md](../sql/README.md)。保留现有 checkpoint 路径；旧已完成/等待会话按身份核对后原地恢复，不通过删除 checkpoint 绕开升级。

## 默认与演示启动命令

默认参数显式列出，避免本机 `.env` 中的演示值影响复现：

```bash
MODEL_CONTEXT_WINDOW=131072 MAX_OUTPUT_TOKENS=4096 \
MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=5 \
TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=10 \
uv run --no-sync --offline uvicorn app.main:app --host 127.0.0.1 --port 8000
```

小窗口保留用户指定的六项配置：

```bash
MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 \
MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 \
TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
uv run --no-sync --offline uvicorn app.main:app --host 127.0.0.1 --port 8000
```

访问 [本机聊天页](http://127.0.0.1:8000/)。切换配置前先正常停止旧服务；历史会话继续保留。“新会话”只切换会话标识，首次发送才创建记录。

## 可复算的窗口预算

以下为当前固定模板和工具定义的估算单位，不是模型供应商的计费 token，也不保证网关实际支持任意长度输入。

| 项目 | 默认 | 小窗口 |
|---|---:|---:|
| 应用窗口 | 131072 | 18000 |
| 固定 system 与工具定义最大开销 | 2968 | 2968 |
| 检索资料预留 | 8192 | 1125 |
| 摘要预留 | 4096 | 750 |
| 输出预留 | 4096 | 2000 |
| 安全余量 | 6554 | 900 |
| 当前原问、工具链和补救峰值 | 8794 | 8794 |
| 扣减后可分配历史 | 96372 | 1463 |
| 最终历史预算 | 32768 | 1463 |
| 层1：最近完整原文 | 22937 | 1024 |
| 层2：较早半压历史 | 9831 | 439 |
| 单次完整模型输入上限 | 120422 | 15100 |

历史预算取“窗口扣减后容量”和 `32 × 1024` 的较小值，然后按整数七三分配。保留 `CONTEXT_NON_ASCII_TOKENS=2.0`、`CONTEXT_STEADY_TURN_TOKENS=1024`，工具申请单次上界为384单位；不是按聊了多少轮触发摘要。

初次默认校准84次真实请求均有usage：估算/实际输入最小1.93、中位2.22，样例完整轮最大526单位。最终默认复测83次均有usage，最小1.91、中位2.19。估算偏保守，能避免低估，但可能较早压缩；换模型、模板、工具或计量系数后须重新校准全部派生预算。原校准记录为 `reports/ch07-default-20261028-r1/calibration.json`。

最终小窗口83次主请求也均有usage，估算/实际最小1.98、中位2.37。两组的实际模型记录与日志角色/正文顺序一致，各阶段固定前缀指纹均只有一个。完整证据核对结果位于 `.superpowers/sdd/2026-10-28-ch07-context/final-evidence-audit.json`。

## 真实验收复现

固定题集是 `tests/fixtures/ch07/dialogue_cases.json`，两组都使用原样21问；尾问不直接提供早期订单编号。真实验收复用现有 `mewhelp_ch04_test` 和知识向量，持测试资源锁，登记随机会话并精确回收六表记录。不要为验收重新导入知识或清空共享表。

先选择一个不存在的新报告目录。HTTP 模式能验证真实模型、消息流、MySQL 与 checkpoint，但不算浏览器交互验收：

```bash
uv run --no-sync --offline python -m tests.verify_ch07_live \
  --env-file .env --profile default --report-dir reports/ch07-default-new --test
uv run --no-sync --offline python -m tests.verify_ch07_live \
  --env-file .env --profile demo --report-dir reports/ch07-demo-new --test
```

真实页面模式在以上命令加 `--browser`，等待输出 `browser ready`。打开新报告目录 `browser-ready.json` 内的 `bootstrap_url`，在该独占测试页面依序发送冻结21问；最后一问前确认测试端点 `/__live__/state` 的 `active_summaries` 为0。这个测试页面额外提供“下次侧栏加载失败”按钮，生产页面没有此入口。

浏览器还需完成：回复时禁止切换；新建第二、第三会话各聊一轮；切回首会话核对原文；刷新再回载；继续首会话一轮；触发一次侧栏失败后仍发送一轮。共25轮。保存真实截图和逐项结果，再向测试专用 `/__live__/finish` 提交包含9项布尔检查、`error: null` 的JSON（检查键见 `final_checks`）；服务自动核对并清理测试数据。驱动可使用应用内浏览器，或已连接的 web-access CDP：

```bash
node tests/fixtures/ch07/browser-driver.mjs reports/ch07-default-new TARGET_ID
```

`TARGET_ID` 必须是本次新建测试标签的真实编号，且标签已打开该次 `bootstrap_url`。脚本不替你创建浏览器会话。脚本会自动完成25轮、截图并通知服务；任何缺项、错误或非布尔检查均不能验收通过。不要把 `--browser` 进程启动成功当成验收成功。

## 回归命令

```bash
CHAT_BASE_URL=http://test.invalid/v1 CHAT_MODEL=test-model CHAT_API_KEY=test-key \
uv run --no-sync --offline pytest -q --basetemp=.superpowers/sdd/2026-10-28-ch07-context/tmp-manual-full
node --test tests/frontend/*.test.mjs
uv run --no-sync --offline pytest -q -m mysql \
  tests/integration/test_ch06_actions.py tests/integration/test_ch07_context.py \
  --basetemp=.superpowers/sdd/2026-10-28-ch07-context/tmp-manual-mysql
```

合并审查修复后的实际结果：后端1839 passed、44 skipped、35 deselected；前端117 passed；显式CH06/CH07 MySQL 10 passed。44项跳过与35项未选中不算通过。日志保留在 `.superpowers/sdd/2026-10-28-ch07-context/fix-{full,node,mysql}.log`。修复前的 `closeout-*` 日志继续保留。

## 验收证据与边界

- 默认最终组：`reports/ch07-default-20261028-r3/`，25轮、9项界面检查、无原文降级、0次摘要调用、两个锚点为0；数据已清理，保留有效截图、日志、原始SSE与checkpoint。
- 小窗口最终组：`reports/ch07-demo-20261028-r3/`，25轮、9项界面检查全部通过；12次原文层边界推进，7次后台调用追加5段摘要，其中2次来源校验补救成功，无最终摘要失败。尾问前摘要已完成，实际注入摘要与回答均保留早期订单及配送诉求；清理恢复且截图有效。8次前台模型调用与后台摘要时间区间重叠，另有受控阻塞模型测试证明回复不等待摘要。
- `demo-r2` 对应旧补救提示，只有7项界面检查；保留原报告，不能将其旧 `passed` 视为当前严格判定通过。`default-r2` 中断于4轮，已按登记回收并保留 `recovery-cleanup.json`。
- 摘要标注评估记录为9/10：有一例纯寒暄被写成“结束对话”。属于可见的语义偏差，可能多占摘要容量；没有宣称所有摘要零幻觉，也没有用单题排除词刷分。见 [摘要语义评估](superpowers/reviews/2026-10-28-ch07-summary-eval.md)。
- 知识不足兜底只算正常完成回复，不算知识答案正确。后台暂未生成摘要时可以临时省略最老半压整轮，但不删除原文或推进摘要锚点；失败退避后可在后续对话或重启扫描时恢复。
- 摘要投影有容量上限；旧摘要段一直保存，但无限会话不保证最早一段始终进入模型。延迟回执与中间轮次形成相交区间时共同分块；单轮或相交整组过大时记录 `oversized_turn`，保留原文和锚点，不拆组或截断用户原话硬塞入摘要。
- 成功工单仍待补回执时，即使恢复标为中断也不能先摘要；使用原动作键完成回执修复后才可正常推进。旧版等待选单在没有冻结快照的那次续跑跳过摘要调度，下一正常轮重新读取数据库冻结，原动作与原轮保持。
- 日志 `log/app.log` 含 `history_ctx`、`model_ctx` 和摘要阶段，保留业务正文且轮转；不输出连接串或密钥。查看真实请求时使用报告中的 `models.json`、`summary-models.json` 和 `turn-*.json` 相互核对。

本章保留本地工作分支、原始验收证据和用户原有草稿；合并、推送和应用库部署不属于本次本地交付。

全分支原验收见 [CH07最终审查](superpowers/reviews/2026-10-28-ch07-final-review.md)，当前合并结论、修复问题及验证见 [合并审查与修复复审](superpowers/reviews/2026-10-28-ch07-merge-readiness.md)。
