# CH05 运行与验收手册

CH05已接入固定工作流、分步只读查询、SQLite会话保存和用户确认建单。它是单实例客服演示：订单、商品和物流仍来自有限演示数据，`demo-user`是演示身份，不是登录认证；“转人工”只显示固定话术，没有真实坐席接入。

## 启动前确认

- 在项目根目录的Git Bash执行，使用当前工作树的独立`uv`环境。不要借用另一个工作树的`UV_PROJECT_ENVIRONMENT`。
- `.env`配置真实聊天模型、MySQL和Milvus；两个本地权重目录分别为BGE-M3和`bge-reranker-large`。验收入口从显式env文件所在目录寻找`models/bge-m3`与`models/bge-reranker-large`并解析为绝对路径。
- 应用MySQL表、质量表及知识索引应已按CH04手册显式准备。知识与向量必须来源一致；缺少或冲突时先处理资源归属，不能运行清库或用生产集合补测。
- 在线必须`KNOWLEDGE_RETRIEVAL_MODE=hybrid`。`WORKFLOW_CHECKPOINT_PATH=.data/checkpoints/ch05.sqlite3`必须是持久化可写文件；一个SQLite文件只能由一个应用实例、一个worker使用。
- 默认消息预算`TOKEN_BUDGET=12000`，完整请求预算`REQUEST_TOKEN_BUDGET=16000`。旧`CHAT_TOKEN_BUDGET`仍兼容；原env若保留较小值，长证据会被完整拒绝。验收首跑实际加载2000，退款证据估算2282而停止；规范配置验收通过CLI明确覆盖为12000/16000，未修改原env。
- 本次最终对照请求模型为`gemini-3.8-flash-high`，上游响应自报`gemini-3.8-flash-n`，两者按原值记录；不据此推断网关别名规则。原env的`gemini-3.5-flash-lite`没有改动。复现对照时使用下面命令的进程级`CHAT_MODEL`，不要误把不同模型的结果合并。

普通本地应用启动命令如下；8000必须空闲。本次真实验收保护用户已有8000服务，使用下文验收器动态分配端口，没有在8000另开实例。

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

完成上述`.env`与资源配置后，本次对照参数对应的启动示例如下；8000仍只是提供给用户的演示命令，本次没有占用该端口实测：

```bash
CHAT_MODEL=gemini-3.8-flash-high TOKEN_BUDGET=12000 REQUEST_TOKEN_BUDGET=16000 uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

打开终端显示的本机地址。前端是本地静态文件，不需要npm安装、构建或独立前端服务。已有服务运行时不要再启动同库第二个应用；启动恢复会处理该库旧running记录。

## 演示输入

1. `订单1001的物流到哪了？请告诉我承运方和最新节点。`：先查订单，再用实际返回的单号查物流。
2. `我要查物流。`，等回复结束，再发`1001`：读取真实同会话历史补参数。
3. `我想退订单1001里的猫粮，请说明无理由退货条件、申请入口和到账规则，再查一下这笔订单现在的状态。`：固定先退款政策检索、证据闸，再查订单和回答。
4. `我要投诉你们客服态度太差！`：显示两个独立按钮。取消或Esc不写库，确认建单后显示真实票号；反复点击原建议仍是同票。
5. `你好呀。`：只做一次意图分类，随后固定欢迎回复。
6. `极光量子路由器 QX-900 的纠错码参数是什么？`：自然低证据时固定拒答并登记问题池，不编参数。

普通聊天不能直接建单。用户未点击按钮可以继续聊天；旧建议保留其原问题，后续消息不替换它。确认描述最多2000字，聊天问题最多4000字，系统不静默裁剪。

## 已跑通的裸循环

Task1的三条命令实际退出0；Task8复用该证据，不重复运行。目标是已准备的`mewhelp_test`，不接图或知识索引。

```bash
uv run python -m examples.ch05_bare_agent --test --env-file D:/AI/my_AI_project/MEWHELP/.env --question "订单 1001 当前状态是什么" --user-id demo-user
uv run python -m examples.ch05_bare_agent --test --env-file D:/AI/my_AI_project/MEWHELP/.env --question "订单 1001 的物流到哪了" --user-id demo-user
uv run python -m examples.ch05_bare_agent --test --env-file D:/AI/my_AI_project/MEWHELP/.env --question "我要查物流" --user-id demo-user
```

## 隔离真实评估

`intent`只接模型，原28例证据保留。`agent`与独立`verify`会实际写测试聊天、问题池及确认票据，所以必须带`--test`。两者固定MySQL/Milvus为`mewhelp_ch04_test`、混合集合为`knowledge_ch04_live`；原应用资源不会被替代。报告目录必须不存在，SQLite建在该目录内。

```bash
CHAT_MODEL=gemini-3.8-flash-high PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync --offline python -m tests.eval_ch05_prompts --suite agent --test --cases tests/fixtures/ch05/agent_cases.json --env-file D:/AI/my_AI_project/MEWHELP/.env --token-budget 12000 --request-token-budget 16000 --report-dir reports/ch05-agent-new-run

CHAT_MODEL=gemini-3.8-flash-high PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync --offline python -m tests.verify_ch05_live --test --env-file D:/AI/my_AI_project/MEWHELP/.env --token-budget 12000 --request-token-budget 16000 --report-dir reports/ch05-verify-new-run
```

正式Agent集冻结10例11轮，样例SHA256为`21a23159f8dcb3f37e4983a4417c6cbac1566afea6b4999b333d49d4d24ad988`。第4例必须使用模型真实第一轮回复；第10例仅在外部工具边界模拟超时，分类、决策和执行器均真实。报告保存模型版本、Prompt/样例哈希、节点、工具参数/结果、预算、未知usage和逐例实际失败。自然语言规则可能漏判或误判，人工复核不能覆盖原规则失败。

本机固定源码的3.8对照目录为`reports/ch05-task8-agent-38-r1`：11轮硬流程和人工语义通过，字面规则10/11、退出1。唯一字面失败是追问“订单编号或快递运单号”，冻结规则只识别“订单号/物流单号/快递单号”；未改题或抹掉失败。33次模型请求，usage共32541 tokens。人工仍记录A03“发发货”文字重复，以及“平台核验”半句的证据编号放在后一相邻句的引用瑕疵，完整原句/证据见该目录`human-review.md`。

Task8修复前同源码同模型的独立最终验收目录`reports/ch05-task8-verify-38-r1`有12条检查记录，全部通过且退出0：固定检索顺序、两个依赖工具、真实分类后的投诉双建议、确认前零票/确认后同票重试、固定闲聊、自然低证据入池一次、停止无遗留任务、新解释器SQLite续聊均通过。首条正文分片早于模型结束125ms；28次请求返回28981已知tokens，1次被停止流usage未知。全部测试数据恢复原计数，临时HTTP、SQLite、模型客户端和父锁关闭。此次核验A03引用落在对应句，但不覆盖Agent对照里保留的引用瑕疵。

最终身份边界修复后只跑一次`reports/ch05-final-verify-38-r1`：原规则11/12、退出1，唯一失败为追问“订单编号或物流运单号”未命中冻结近义词规则。99个流程布尔断言全部true，逐条人工语义核对通过；新命名键下的新解释器续问、两步物流查询、停止无遗留和确认同票均通过。新95个源码指纹匹配，不能用旧Agent报告替代本次指纹。首delta早140ms，28请求/29786已知tokens、1次usage未知；四表0→7/35/1/1→全0，资源关闭。完整原句和版本限制见该目录`human-review.md`与`audit.json`。

3.5系列旧记录全部保留：r1中断、r2四轮失败、r3正文编入口、verify-r1停止泄漏、verify-r2漏政策例外、r5内部字段混入正文、r6从单号推承运方；r4与部分verify通过不能覆盖其他轮失败。该小样例对照不能证明模型整体优劣或所有问法稳定。物流演示时间有时省略时区，原始工具审计保留完整ISO时间。

验收器全程持测试资源锁，拒绝外来running轮次。每次创建唯一会话并登记主键与轮ID；退出按本次问题池主键+会话、票号+会话、消息轮ID+会话、三身份会话的顺序精确清理。不会删共享知识、Milvus集合、其他聊天或报告中的SQLite文件。`cleanup.json`必须显示四表恢复原计数；清理冲突时保留证据，不扩大删除条件。

## 离线回归

Windows默认pytest临时目录在本机权限不足，使用新的工作树内basetemp。下列live开关明确关闭；跳过和排除的测试不算真实验证通过。

```bash
CHAT_BASE_URL=http://127.0.0.1:9/v1 CHAT_MODEL=offline-test CHAT_API_KEY=offline-test MEWHELP_RUN_RAG_INTEGRATION=0 MEWHELP_RUN_RERANK_LIVE=0 MEWHELP_RUN_QUALITY_MODEL=0 MEWHELP_RUN_QUALITY_MYSQL=0 uv run pytest -q --basetemp .tmp/ch05-offline-new-run
node --test --test-reporter=spec tests/frontend/*.test.mjs
```

Task8修复前源码的最终回归一次完成：Python 1273通过、47跳过、25排除（46.88秒）；前端Node 90通过、0跳过。两命令退出0。跳过/排除项没有因此获得真实验证，真实链结果另见上方独立报告。身份边界修复未改前端，因此保留Node90项证据，不重复运行前端套件。

最终身份边界修复后的Python全套单次回归：1287通过、47跳过、25排除（49.44秒），退出0；日志`.tmp/ch05-final-fix-all-offline.log`。受影响workflow/HTTP范围另有274通过、2排除；均明确关闭四个live开关。

2026-10-26追加审核后，前端提交`f79ccf9`统一两个身份字段的Unicode码点计数并补测试注释；目标红测4项预期失败→25/25通过，完整前端更新为98/98通过、0跳过、退出0，日志`.tmp/ch05-minor-frontend.log`。覆盖两个字段32/33/64码点接受、65拒绝、原身份实际按钮确认传递，独立复审通过。后端和Prompt未改，未重复Python全套或真实模型；旧95个验收文件指纹不能代替此次前端修改的浏览器复验。

## 停止与恢复边界

停止流会关闭上游任务、释放会话占用并将未完成轮标为取消。下一条消息从新轮入口开始，按MySQL审计与SQLite游标对账，不自动重放旧工具，也不保证从断点无缝续播。MySQL已提交但SQLite保存失败时不会假发DONE；下一轮恢复依据真实数据库记录。

最终身份边界修复后，聊天与确认入口都会去除session/user标识的首尾空白；用户问题与工单描述仍保存原文。SQLite线程键由核对MySQL归属后的会话主键生成`mewhelp:ch05:{conversation_id}`，不是客户端session字符串，也不占用LangGraph内部`checkpoint_ns`。恢复时必须匹配快照中的conversation/session/user；不匹配或缺失就忽略旧快照历史、游标和未完成轮，从可信MySQL重新读取。相同身份仍增量续聊，每条新消息从START开始。

旧客户端字符串键不自动迁移或兼容，也不删除原SQLite文件；升级后的首次消息会导入该会话的MySQL历史。上文Task8模型评估和1273项Python结果对应修复前源码，Prompt、图与工具语义虽然未变，其源码指纹不代表本次修复版本；本轮结果另存`reports/ch05-final-verify-38-r1`。

确认票据使用来源建议的稳定动作ID。响应丢失后通过原动作重试核实同一票号；不要生成新动作再次提交。`outcome=unknown`表示暂时查不清，保留原键核实；真实票号已经返回但审计待同步时仍应保留该票号。

该版本没有多实例协调、正式鉴权、真实订单/物流供应商和人工坐席服务，不宣称全面生产就绪。逐阶段事实和返工记录见[开发记录](../dev-notes/ch05.md)。
