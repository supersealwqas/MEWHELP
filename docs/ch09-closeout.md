# CH09 收尾与验收结果

2026-10-31完成本机功能与验收，功能修复提交 `ef86692`。最终复审已通过，满足合并到本地 `main` 的条件；仓库未配置远程。下列结论只针对本机已批准范围。

## 最后合并前复审（2026-10-31）

本轮固定审查范围为 `af1cdfc...506c334`，由三路独立审查分别核对项目规范、需求和功能实现。需求与功能审查均无新增发现；规范审查发现1项P3模块说明错误：`pipeline.py` 已含策略判定及问题池信号构造，旧说明却写成“无策略文件或问题池依赖”。已更正为共享知识流程、不负责聊天或问题池持久化。剔除模块文档后，语法树与受审版本完全一致，没有改变执行逻辑。

|本轮重新执行的验证|结果|
|---|---|
|完整默认回归|2708 passed、44 skipped、92 deselected，403.36秒|
|CH09隔离MySQL|38 passed，73.88秒|
|前端Node测试|132 passed，0 failed|
|真实只读闭环核验|15项passed，failures=[]、pending=[]，退出0|
|注释更正后的生产评估/知识自评回归|73 passed，15.83秒|
|换行检查|按CRLF检查通过；原始证据不改写|

测试日志保存在 `.superpowers/sdd/2026-10-30-ch09-observability-flywheel/merge-review-20261031-HoaWyR/`；本轮只读原件位于 `reports/ch09/acceptance/merge-read-20261031-HoaWyR/summary.json`。原有516份未提交文件已记录字节摘要并核对未变。本轮没有重新调用付费模型、发布知识或改写生产数据。下文保留此前完整现场验收及其版本边界；既有误拒和历史追踪限制仍有效。

## 已完成

- 本机Langfuse记录原轮模型、工具、召回/精排和实际用量；确认恢复沿用同一追踪，完成回放不产生新模型调用。
- 证据不足、知识自评不足、用户差评进入问题池，标准化查重后人工审核；同轮多入口不重复累计出现次数。
- 审核14使用明确标为本机演示的“青禾校验卡”知识。受控向量写入失败后答案冻结，重试复用知识块53与来源 `ch09/review/14`，发布完成后同问答出“青禾入库完成”，公开引用显示核准正文。没有添加虚构商城政策。
- 待审/通过/驳回、超长答案、失败恢复、历史证据缺失、反馈重发、窄屏和两轮趋势均有实际页面原件。
- 生产评估只覆盖知识子流程，默认每日北京时间03:00运行；本次现场已显示下一次为2026-11-01 03:00。未宣称未来定时运行已经执行。

## 最终验证

|验证|结果|本地原件|
|---|---|---|
|修后完整回归|2708 passed、44 skipped、92 deselected，397.03秒|`.superpowers/sdd/2026-10-30-ch09-observability-flywheel/final-suite2.log`|
|CH09专用MySQL|38 passed，70.14秒|同目录 `final-mysql2.log`|
|真实只读闭环核验|15项passed，failures=[]、pending=[]，退出0|`reports/ch09/acceptance/final-read-verified/summary.json`|
|具名页面/语义复核|5类观察通过；由Codex协调者观察，并非用户亲自确认|`reports/ch09/acceptance/manual-observations-verified.json`|
|第二轮Langfuse对账|48题、168次调用全部归evaluation，34次裁判均接回本题原根|`reports/ch09/acceptance/eval-after-attribution.json`|
|运行版本/健康|152份应用源码与启动摘要一致；应用入口200，Langfuse六组件healthy|`reports/ch09/acceptance/preflight-closeout.json`|

重叠测试不累加，跳过或未选中的测试不算通过。完整回归默认不包含mysql，专库38项另行显式运行。

原总审发现裁判消费错误归入客服、公开答案/知识块指纹混用、多查询完整排名记录错误；另补公共持锁连接UTC初始化。四项修复经独立范围复审通过，复审没有新的阻断问题。报告位于 `docs/superpowers/reviews/2026-10-31-ch09-final-review.md` 与 `2026-10-31-ch09-final-fix-review.md`。

## 两轮真实评估

|项目|发布前|发布后|
|---|---:|---:|
|数据库记录ID|1|2|
|完成/调用错误|48 / 0|48 / 0|
|回答题数|32|34|
|拒答题数|16|14|
|可回答题误拒|8/40（20%）|6/40（15%）|
|库外题正确拒答|8/8|8/8|
|Recall@10 / MRR@10|1 / 1|1 / 1|
|已回答内容忠实度|32/32|34/34|
|生产实际token|236669|244673|
|裁判实际token|69913|75258|

首轮artifact：`65c2d9e9-d978-4dc2-844b-439c6c6d9158`；二轮artifact：`326363de-e5b2-4dab-86b5-046589ab589f`。完整原件分别在 `reports/ch09/eval-before/`、`reports/ch09/eval-after/`；比较原件为 `reports/ch09/acceptance/evaluation-comparison.json`。

两轮比较协议一致，冻结配置仅知识版本改变。`ch04-e-38`、`ch04-e-40`由拒答变为回答，但34题的模型改写也发生变化，新增演示FAQ未进入任何评估题的排名、模型证据或引用。因此这里只记录重复监测中的结果波动，不归因为新增FAQ，不宣称质量稳定提升或48题全部答对。

第二轮134次生产调用与34次裁判合计319931 token，逐调用、逐题和总量全部与Langfuse原上报一致。未知用量为0；页面中的估算金额没有作为可靠费用。UI样例 `ch04-e-01` 的裁判为1742输入+160输出=1902，生成与裁判同属trace `7ab99973c06102e6c49708265686fd23`。

## 已知边界和保留原件

- 用户此前已明确选择冻结Top1基线；多信号候选与校准/独立验证原件保留。当前仍有6/40可答题误拒，是后续优化目标，不是本章消除了所有回答质量问题。
- 第一轮32次裁判的69913 token在历史Langfuse中归错客服类别；原trace不改写。原评估逐题用量和分数仍有效，新运行已修复归属。
- 第一轮5题的完整排名记录次序有差异，独立重算确认48题各项分数均未变化；`eval-before-ranking-audit.json`保留证据，不覆盖或付费重跑首轮。
- 首次最终核验的15项业务检查通过，但浏览器截图实际为JPEG、误用PNG后缀。原图和首次失败报告 `final-read/`保留；另存17份真正PNG并逐像素验证一致，转换链见 `screenshot-conversions.json`。最终清单为 `evidence-manifest-verified.json`，没有修改画面来通过验收。
- ClickHouse既有健康探针暂态日志、旧CH03 TINYINT(1)弃用提示继续作为非阻断维护项保留。未知历史数据不清理，CH06和其他参考资料改动未纳入本次提交。

## 使用与复核

- 客服：<http://127.0.0.1:18000/>
- 知识审核：<http://127.0.0.1:18000/flywheel>
- 两轮评估：<http://127.0.0.1:18000/rag-eval>
- Langfuse：<http://127.0.0.1:13000/>

当前应用运行的是修复代码 `ef86692`（PID43500；重启后PID会改变），启动与凭据说明见 `docs/ch09-operations.md`。主应用代码后续未变更，文档提交不改变已加载代码。验收用故障开关已撤销。

只读重新核验以下命令不会发新问题、发布知识或付费重跑评估；输出目录必须选一个新名称：

```bash
USERNAME=Superseal PYTHONIOENCODING=utf-8 PYTHONUTF8=1 uv run --offline python tests/verify_ch09_live.py \
  --base-url http://127.0.0.1:18000 \
  --evidence-manifest reports/ch09/acceptance/evidence-manifest-verified.json \
  --output-dir reports/ch09/acceptance/final-read-new
```

本机原始追踪、截图和完整报告位于忽略的reports目录，未把凭据或整个现场目录推送到远端。版本化交付记录及精选机器摘要与完整本地原件分开保存；迁移、失败原件及既有业务数据均保留。
