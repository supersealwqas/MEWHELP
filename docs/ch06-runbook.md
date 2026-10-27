# CH06 运行与验收说明

CH06 使用 `gemini-3.5-flash-lite` 完成一次问题理解和八类意图分类。confidence 只记录，不据此更换模型。退款、退货、维修等具体订单问题先核对本人订单，再检索政策；缺号或对象不明确时等待用户选单，选择后继续原问题。

## 启动与迁移

在 Git Bash 和项目根目录执行。先启动已有 Docker Desktop 与项目 MySQL/Milvus，确认本地 BGE 权重、知识库和 `.env` 已配置。

```bash
# 只预览，核对 --database 是目标应用库；默认不会写库
uv run python -m app.db.migrate_ch06 --env-file D:/AI/my_AI_project/MEWHELP/.env --database mewhelp

# 正式使用新版本前由部署流程应用增量迁移
uv run python -m app.db.migrate_ch06 --env-file D:/AI/my_AI_project/MEWHELP/.env --database mewhelp --apply

uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

迁移扩充消息状态 CHECK，并新增独立的 `refund_applications` 表；重复执行不会重建或清空业务表。本次验收只对 `mewhelp_ch04_test` 应用迁移，应用库尚未迁移。请勿把旧章的清库初始化脚本用于升级。SQLite 检查点文件必须随应用数据持久保存。

当前会话锁适用于**单 worker、单服务进程**。页面使用演示身份 `demo-user`，订单和物流来源仍为项目演示查询；这不是生产登录鉴权系统。

## 页面流程

1. 输入不带订单号的退款问题，页面显示本人订单卡片。此时流正常结束，状态为 `awaiting_input`，没有完成标记 `[DONE]`，也不代表请求出错。
2. 点选订单后，服务器核对原动作和归属，使用同一会话、原问题编号和原调用账本继续执行。页面继续更新原客服气泡，不再追加一条用户问题。重复点选不会另开一轮。
3. 回答完成后，从服务端原因目录选择退款原因，再点击“确认提交申请”。成功会显示 `RF...` 申请编号和“待处理”。同一动作重试返回原编号；改原因不能覆盖已保存申请。
4. “核实处理状态”从服务器读取当前结果。刷新会恢复本标签会话的等待卡片、完成回答或申请回执；不会在浏览器缓存聊天正文。开始新会话会生成新会话编号。

退款申请仅保存在本地数据库，**不代表批准退款、实际退款或到账**。退款原因在表单中填写，聊天阶段不靠追问原因决定能否开始查单。没有可靠签收时间时说明待核实，不能把下单时间当签收时间。人工工单与退款申请是独立动作。

部署后必须刷新页面加载新版脚本，旧页面不识别选单后正常结束的流协议。

## 等待、恢复和故障

服务器确认 SQLite 已保存暂停点后，才保存 MySQL 等待状态；关闭图流并释放会话锁后再发卡片。启动时先核对检查点，修复尚未接受的等待；已接受但尚未续跑的动作保持原订单，并显示“继续处理”。读取检查点失败会阻止启动，不会把等待误作遗留请求取消。

新问题可以取消尚未接受的旧卡片；已接受、尚待续跑的原问题需先继续处理。迟到的旧事件不能覆盖新回答。HTTP 404 表示来源或身份不匹配，409 表示动作过期、换单或会话竞争，422 表示参数不合法。开流后的错误通过 `error` 事件说明；503 或未知网络结果需要核实原动作，避免换键重复提交。

## 可复现验收

三套冻结标注分别包含32条八类单轮、6组29轮对话、12条理解与扩写专例。expected 标注不会进入模型上下文；对话选单是明确用户输入。原始工具参数分片、结束原因、usage、服务端回退、耗时和调用账本分别记录。报告目录必须新建。

```bash
uv run python -m tests.eval_ch06_prompts --suite intent --env-file D:/AI/my_AI_project/MEWHELP/.env --report-dir reports/ch06-intent-new
uv run python -m tests.eval_ch06_prompts --suite dialogue --env-file D:/AI/my_AI_project/MEWHELP/.env --report-dir reports/ch06-dialogue-new
uv run python -m tests.eval_ch06_prompts --suite understanding --env-file D:/AI/my_AI_project/MEWHELP/.env --report-dir reports/ch06-understanding-new
uv run python -m tests.verify_ch06_live --test --env-file D:/AI/my_AI_project/MEWHELP/.env --report-dir reports/ch06-live-new
```

评估退出0代表达标、2代表验收失败、3代表依赖或运行故障。真实流程限定 `mewhelp_ch04_test` 和 `knowledge_ch04_live`，使用独立 SQLite、随机登记会话和逐记录清理，复用已有知识数据，不初始化共享知识库。

需要浏览器取证时，在真实流程命令后增加 `--browser`。完成自动场景后读取报告中的 `browser-ready.json`，使用 Codex 内置浏览器打开 `bootstrap_url`，它只给本次测试页面预置已登记的随机会话。验收工具条可显式延迟一次真实响应或插入未知事件；不会替代模型或修改业务结果。点击保存证据后，核对页面与报告，再在该报告目录写 `browser-finish.json`（包含 `failures` 数组）结束服务并触发精确清理。这些辅助路由只挂在测试应用的随机本机端口，生产应用没有这些入口。

## 本次证据（2026-10-27）

- [单轮分类：32/32](../reports/ch06-intent-20261027-run1/report.md)
- [多轮对话：29/29](../reports/ch06-dialogue-20261027-run3/report.md)
- [理解与扩写：12/12](../reports/ch06-understanding-20261027-run3/report.md)
- [最终真实流程记录](../reports/ch06-live-20261027-run3/summary.json)
- [浏览器11项核对](../reports/ch06-live-20261027-run3/browser-finish.json)：等待刷新、同轮/同气泡、防重复、回执刷新、旧卡片不复活及未知事件均通过。
- [等待卡片截图](../reports/ch06-live-20261027-run3/browser-waiting.png)、[退款待处理截图](../reports/ch06-live-20261027-run3/browser-refund.png)、[旧帧释放后新回答截图](../reports/ch06-live-20261027-run3/browser-late-frame.png)。
- [最终自行审查](superpowers/reviews/2026-10-27-ch06-final-self-review.md)，按用户要求未派代理。

最终相关后端回归 **620 passed**，前端 **71 passed**，显式选择MySQL标记的真实事务测试 **6 passed**。完整流程 **15/15**（包含浏览器一组11项核对），测试创建的6个会话、53条消息、2份退款申请均已按登记身份清理，五表计数恢复运行前基线，未改应用数据。验证日志保存在最终报告的 `verification/` 目录。

```bash
export PYTEST_DEBUG_TEMPROOT=D:/AI/my_AI_project/MEWHELP/.superpowers/sdd/2026-10-27-ch06-router/pytest-temp-root
uv run pytest -q tests/workflow tests/rag/test_query_rewrite.py tests/rag/test_retrieval.py tests/test_order_ownership.py tests/test_chat_api.py tests/test_multi_tool_chat.py
node --test tests/frontend/chat-api.test.mjs tests/frontend/actions.test.mjs
uv run pytest -q -m mysql tests/integration/test_ch06_actions.py
```

早期失败报告保留在对应 run1/run2 目录。修复包括：明确排除旧编号时只保留新候选、追问同时补足对象与业务、屏蔽助手独自编出的编号、验收器接受同义表达及正确读取理解标记。三套最终输出已逐条复核，不仅依赖词表。

每份分类报告均有“错判样本的 confidence 分布”；解析回退的零分单列。本次最终无错判样本，不能据此声称 confidence 已校准，也未启用模型升级。

代码保留在本地 `codex/ch06-router` 分支，尚未合并、推送或部署。自审另记录一个非阻断显示细节：个别模型回答会生成非数字引用标签，可后续统一；本次不将其计作政策来源。
