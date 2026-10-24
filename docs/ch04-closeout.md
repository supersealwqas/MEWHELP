# CH04 原始范围收尾验收

日期：2026-10-25。最终产品候选：`85f1b1af91ebd1946e5bfb9a925bfe77d9e1e16b`。四项原定验收完成，唯一独审问题已TDD修复，已快进并入本地main；main回归1033通过、前端59通过，原文件及源码摘要核对通过。

## 范围与实现

按本轮[计划](superpowers/plans/2026-10-25-ch04-scope-closeout.md)及[规格](superpowers/specs/2026-10-25-ch04-scope-closeout-design.md)收尾。用户以 goal 明确授权自行审阅、确认保留原内容后合入本地 main；不推送远端。

- 保留 Milvus 原生 BM25、BGE-M3、RRF 和 `bge-reranker-large`；两路各50、RRF50、普通10、退款最多30及0.3重排门槛保持。
- 线上 API、FAQ 与退款工具开放 `category`、`content_type`，权威库先确定范围，两路召回前过滤；空范围正常空返回，越界及失配报故障。
- 当前问题只改写一次，Dense用标准问法、BM25附原问和最多4个同义词、重排保留原问；退款三组及评估四策略共享准备结果。短条件随子句保留，同义词不写知识表。
- 知识成功回答必须有可展示的正文角标；代码、链接、图片及HTML内的编号不算。知识故障不伪装成成功或知识不足，不入问题池；此前已建工单回执保留。

## 已验证证据

报告保存在本地 `reports/ch04/`，不进入 Git。失败记录与成功记录同时保留；可从 [证据目录](../reports/ch04/20261025-scope-evidence-01/)查看。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 收尾工作树离线 Python | 1030通过、47跳过、25排除，23.10秒 | `execution/review-full.log` |
| main合并后完整 Python | 1033通过、44跳过、25排除，30.62秒 | `execution/task-4-tests.log` |
| 前端回归 | main合并后59通过，0失败 | `execution/main-node.log` |
| 真实质量MySQL、BM25/RRF、large重排、模型自评与裁判专项 | 6通过，26.29秒 | `live-special.log` |
| 预先标注的单轮语义样例 | 最终Prompt 6/6；初次失败和定向复验保留 | `semantic-final.jsonl`、`semantic-samples*.jsonl` |
| 真实聊天：问候、MH-W20、MH-W99、物流、工单 | 5/5；MH-W20为2L且引用可读，MH-W99明确拒答并以原话单次入池 | `../20261025-scope-live-02/` |
| 线上品类过滤 | 只返回权威范围主键；未放宽筛选 | `extra-checks.json` |
| 已建工单后知识故障 | 真实MySQL工单保留，故障注入后failed、无DONE、问题池0、retry_safe=false | `extra-checks.json` |
| 浏览器引用与反馈 | 引用快照/全文/章节定位正常；反馈提交与刷新后均锁定；代码/链接不变成引用按钮 | `browser-qa.json`、`browser-*.png` |
| 版本绑定 | 最终源码/校准/比较全部匹配，四策略48题共享输入；旧校准被拒绝 | `final-validation.json`、`version-binding.json` |
| 独审发现的看板配对问题 | 新校准/旧比较、无绑定、不匹配三项红测后修复；真实报告目录再次核对通过 | `execution/review-red.log`、`execution/review-green.log`、`dashboard-pairing-live-artifacts.json` |

浏览器操作使用已完成真实聊天的重放页、产品Markdown/引用组件和真实引用API，不声称本次重新实点了整个聊天页面的发送流程。故障演示为主动注入改写连接错误，不是一次真实网络事故。`scope-live-01` 在聊天请求前因临时启动器路径失败，原报告保留。

## 同版本评估

主集72题，24校准/48固定、六桶各12，52个知识块。未改标注、未使用固定集调参、未运行320题扩展集。每题一次改写，四策略共享；检索错误仍进入可答题召回分母，拒答忠实度为null。

- 最终源码摘要：`2c41d9e548f697dc96d178ea2f4dee81d6400e65bac776dd5e2d6888ba0a23a7`。
- Prompt摘要：`39339c01209b5d5a89ce3a660c82fbe0cfccfba939ccb6b87eb1c46c6de2947e`。
- 语料摘要：`bef3dc1ebbae0f9dae30e5ad70a3e0f9432b5f338a100a61eb8dc55e5d4877f9`。
- 题集摘要：`f91e0e437cdbad149d54daf3287fb74f2f47c9105b348768ab74c594cbd41a8e`。
- 最终校准：`20261025-calibration-02`，96条、0错误，每策略24条有效记录。
- 最终固定比较：`20261025-evaluation-02`，192条，`completed_with_errors`，8条错误全部保留。
- 第一轮候选 `f54a687` 的校准/比较目录 `20261025-calibration-01`、`20261025-evaluation-01` 完整保留。校准 `ch04-c-08` 改写失败四策略各1错，固定比较7错。独审后只修看板配对；为满足最终源码绑定补跑一轮，不调题、Prompt或参数，额外额度与等待由用户goal自主决定授权覆盖。

评估门槛分别由校准集确定，不替换线上0.3重排门槛，也不把BM25/RRF分数当概率。文档提交改变Git提交号但不改变源码/Prompt摘要，完整绑定见manifest与policy。

| 策略 | Recall@10 | MRR@10 | 覆盖率 | 误拒率 | 忠实度宏平均（分子/有效数） | 错误 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 1.0000 | 0.8542 | 28/48 | 12/40 | 1.0000（28/28） | 0 |
| BM25 | 1.0000 | 0.9833 | 31/48 | 7/40 | 0.9667（29/30） | 3 |
| Hybrid | 1.0000 | 0.9375 | 34/48 | 4/40 | 1.0000（34/34） | 2 |
| Hybrid＋重排 | 1.0000 | 1.0000 | 26/48 | 12/40 | 1.0000（25/25） | 3 |

召回及MRR均以40道可答题为分母；四策略库外正确拒答均8/8。忠实度微平均分别为76/76、76/77、91/91、71/71。完整[分桶表](../reports/ch04/20261025-scope-evidence-01/final-bucket-metrics.md)列出六桶×四策略的数值和分母；[原始汇总](../reports/ch04/20261025-evaluation-02/summary.json)保留全部分子、分母及难度分组。

8条错误为：Hybrid e-02/e-15、BM25 e-04/e-14、重排 e-31 的生成错误；重排 e-28 的正文引用错误；重排 e-33 与 BM25 e-38 的裁判错误。完整题号前缀均为 `ch04-`，逐条原因见 `results.jsonl`。不为消除这些错误继续重跑或制定新通过线。

## 独审与合并保护

唯一一次fresh只读后端审查覆盖 `e62a538..f54a687`，结论为1项Important、无Critical/Minor；看板错配门槛问题由主Agent一次TDD修复，完整回归1030通过。[独审记录](superpowers/reviews/2026-10-25-ch04-scope-closeout-review.md)明确其原始范围、所有排除项及处理，没有冒称再次独审。最终85f1b1a的全源码绑定通过补跑验证。

合并前逐字节核对main的59个原文件和优化工作树的7个文件，包括两份 `.env`。本地main从 `e62a538` 快进；不触碰 `codex/ch04-completion` 的HEAD及其未提交内容。合并后的原文件哈希、源码摘要、Git树和测试证据记录在 `preservation-after.json` 与 `execution/task-4-tests.log`；保留收尾工作树及报告副本，不删除用户分支或文件。

main比收尾工作树多通过3项，是根目录存在本地BGE-M3 tokenizer，`tests/knowledge/test_chunking_tokenizer.py` 的3项实际切分检查可以运行；没有开启真实网关/数据库开关。合并检查发现9个文件仅LF/CRLF不同，先核对Git树及规范化内容相同，再对齐已评估候选的物理字节；全源码摘要恢复精确一致，记录见 `main-line-ending-alignment.json`，没有逻辑修改。

最后一次提交只收尾文档与计划状态；运行代码仍是85f1b1a对应的源码摘要，不因文档提交重复模型评估。完整执行台账、红绿日志和审查包先逐文件校验归档，再移除本计划的临时目录；原优化工作树及其他计划目录保持。

## 本机复现命令与资源

本次复用已核验归属的 `mewhelp-ch04-verification`：MySQL `127.0.0.1:13307`、Milvus `http://127.0.0.1:19540`；测试库 `mewhelp_ch04_test`/集合 `knowledge_ch04_live`，评估库 `mewhelp_ch04_eval`/集合 `knowledge_ch04_eval`。资源审计见 `resource-audit.json`，没有清库、删卷、迁移应用库或重启用户服务。

以下为本机归档启动器的等价复现命令。它只读取原 `.env`，在子进程中覆盖专用端口/库/模型路径和12000/16000预算，拒绝离线占位凭据；不会修改 `.env`。先为输出选择未存在的新目录，真实命令会使用模型额度。

```bash
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export UV_PROJECT_ENVIRONMENT=D:/AI/my_AI_project/MEWHELP-worktrees/ch04-completion/.venv
entry=reports/ch04/20261025-scope-evidence-01/execution/live_entry.py
uv run --offline --locked --no-sync python "$entry" -m app.evaluation.cli validate --dataset data/evaluation/ch04.json
uv run --offline --locked --no-sync python "$entry" -m app.evaluation.cli calibrate --split calibration --output-dir reports/ch04/20261025-calibration-03 --mysql-port 13307 --milvus-uri http://127.0.0.1:19540 --embedding-model-path D:/AI/my_AI_project/MEWHELP/models/bge-m3 --reranker-model-path D:/AI/my_AI_project/MEWHELP/models/bge-reranker-large
uv run --offline --locked --no-sync python "$entry" -m app.evaluation.cli compare --split evaluation --policy reports/ch04/20261025-calibration-03/policy.json --output-dir reports/ch04/20261025-evaluation-03 --mysql-port 13307 --milvus-uri http://127.0.0.1:19540 --embedding-model-path D:/AI/my_AI_project/MEWHELP/models/bge-m3 --reranker-model-path D:/AI/my_AI_project/MEWHELP/models/bge-reranker-large
```

首次部署和专用测试命令见[运行手册](ch04-runbook.md)。演示可使用“MH-W20水箱多大”“MH-W99支持哪些功能”；引用点开全文后定位型号章节，反馈点击后不可再次提交。

## 保留的限制

本阶段不支持历史指代消解或跨轮证据继承；模型改写仍可能失败或偏离原意，单次样例通过不等于质量保证。拒答、生成及裁判错误须结合逐题报告判断，不以新增百分比门槛掩盖。赞/踩仅存本浏览器。系统仍是本地单实例演示，订单物流为演示数据，工单真实落本地MySQL，不具备生产登录鉴权。
