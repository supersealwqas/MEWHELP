# CH10 主题分类器运行说明

本章把低置信度问题在后台批量归类，实时客服对话不调用这个模型。主题表位于 `data/topics/taxonomy.json`，17个标签及边界由它统一定义；一句话可有多个标签，“其他”只作排他兜底。

## 当前数据与人工审核

2026-10-31只读导出得到21条问题池记录、16个不同原问，均带演示/验收来源标记；没有可证实的真实客户数据。17条旧记录缺少可信原轮映射，统计按池记录分别保留，不能宣称已经准确去重。其余4条可对应3个原轮。

用户已授权“你自己替我决定”，本章完成了AI代审，未冒称用户亲自逐条审核。标签字段保留`model_prelabel`，代审经过另存记录；`human_review_status=pending`表示没有独立人工金标准，不表示等待再次授权。已有知识审核队列的通过/驳回不代表主题标签经过人工确认。

## 导出与准备语料

在原仓库目录使用Git Bash和项目uv环境执行：

```bash
uv run --offline --locked python -m app.topics.cli export-pool \
  --output-dir reports/ch10/pool-20261031

uv run --offline --locked python -m app.topics.cli prepare \
  --source reports/ch10/pool-20261031/pool.json \
  --output-dir reports/ch10/datasets/20261031-v1 \
  --per-label 60 --multi 204 --seed 42 --max-calls 60
```

输出目录不得覆盖；本文列出的池导出已经存在，重导出时应使用新目录。准备失败后原参数重跑，会复用同名 `.resume` 目录中的成功模型批次。原问只读，模拟语料和增强语料不会写回问题池。

预标外发仍限制每题1500字符。准备时，超过该限制的完整脱敏原问会连同来源、分组和池ID写入 `.resume/pending.json`，并进入最终的 `review.json` / `review.md`，不会截断或发送模型；其他短题继续处理。`manifest.json` 的 `metadata.overlong_count` 记录超长隔离数，`pending_count` 记录全部未进入数据集的待复核数。全部原问都超长且未要求生成模拟题时，也会写出空训练/验证/测试文件和完整待复核包。新规则改变了原批次计划时，旧恢复目录会明确报参数不符，应换新制品目录，避免旧序号错误回配。

基础语料先按全局关联组做80/10/10分割；同一原轮、相同正文、父子改写不跨分割。数据增强只处理训练集。`manifest.json`记录摘要、来源、各标签样本量；`review.md`与`review.json`供人工抽审。不能通过修改测试标签或反复调整测试阈值来掩饰错误。

## 固定权重及资源证据

- 来源：`hfl/chinese-roberta-wwm-ext`，revision `5c58d0b8ec1d9014354d691c538661bf00bfdb44`。
- 本地目录：`models/ch10/base/`，`source-manifest.json`保存文件摘要。
- 权重字节数411578458，SHA256 `1ded5a5a1c7841dee6e47942f7b5bf2bcf6f73ff19197580f852f7f638f86b35`。
- 模型配置类型为BERT，使用自动分类模型加载17维分类头；不依名称误用RoBERTa分词器。
- `reports/ch10/gpu-probe-20261031.json`证实RTX4060 Laptop上102280721个参数全部参与梯度计算和更新，batch2、长度128；峰值分配2071200768字节，预留2264924160字节。这是单次资源探测，完整训练证据另行记录。

## 数据修订记录

## 授权代审后的语料

用户明确要求“你自己替我决定”后，两路代理逐项阅读全部1239种基础正文，主审统一有歧义的标签边界。186种正文的标签被纠正，41条无当前诉求或信息不明的基础记录隔离。原版、逐条修改理由与原始模型响应完整保留。这里是用户授权的AI代审，不是用户亲自核对的人工金标准。

首次完整训练使用`reports/ch10/datasets/20261031-v2-reviewed`：训练1050条、验证123条、测试120条，各份覆盖17类。89条增强仅位于训练集，原基础样本的切分归属不变；更改发生在首次正式训练之前，未参考模型测试成绩。代审证据位于`reports/ch10/delegated-review-20261031/audit-completed.json`。

最终版本为`reports/ch10/datasets/20261031-v4-reviewed`：训练1877条、验证256条、测试120条。v3增加独立口语训练/验证语料，v4只增加289条已有训练父句的压缩改写。v4验证和测试与v3逐字不变；原始分组约80/10/10，增强后的数量不再冒称严格80/10/10。新增短句沿用训练父ID和分组，未将“买大了想退”原句植入训练。

数据仍由模拟问题为主，全部原池数据仍属演示来源。制品中`human_review_status=pending`只表示没有独立人工金标；`metadata.delegated_review.status`记载已完成的授权代审，不代表开发停在等待用户确认。

## 全参数训练命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_SAFETENSORS_CONVERSION=1 \
uv run --offline --locked python -m app.topics.cli train \
  --dataset-dir reports/ch10/datasets/20261031-v4-reviewed \
  --base-dir models/ch10/base \
  --output-dir models/ch10/classifier-new-run \
  --epochs 8 --micro-batch 2 --accumulate 8 --max-length 128 --seed 42
```

输出目录必须是新目录。训练前冻结阈值候选0.10至0.90、步长0.05及选优规则；每轮仅用验证集宏F1选最佳，连续2轮不改善早停。测试集仅在最终最佳模型上评一次。`training.json`记录配置与逐轮指标，`test_metrics.json`包含17类精确率/召回率/F1和2×2矩阵、来源分层，`test_errors.md`列出错例；完整服务模型位于`model/`。

最终运行路径为`models/ch10/classifier-20261031-v4`，资源证据位于`reports/ch10/training/20261031-v4-runtime.json`。运行使用同一已审查训练函数，旁观包装只记录耗时、显存与编码器变化，没有替换训练算法。7轮后早停、826次更新；最佳第5轮，阈值0.30，102280721个参数全部参与优化器，编码器检查矩阵每轮确实更新。

## 模型结果和使用限制

模型版本：`ch10-topic-1412cf9dd328ccfb`，推理使用独立CPU服务；导出复载标签一致，最大概率差2.98e-7。

|测试来源|条数|宏F1|Micro F1|整句标签完全正确|
|---|---:|---:|---:|---:|
|混合汇总|120|0.955849|0.964169|0.916667|
|模拟问题|118|0.960374|0.966887|0.923729|
|应用演示|2|0.666667|0.800000|0.500000|

宏F1先计算各类成绩再平均；它不等于“每句话95.58%正确”。各类仅4—22个正例，演示留出只有3类有支持，不能证明真实客户效果。模型共10个测试错例，其中1个未确定；全部错例均完成AI复核。该测试集已用于几个实验的结果报告，后续应另备新的客户留出集。

- 每类精确率、召回率、F1和17张2×2混淆矩阵：`models/ch10/classifier-20261031-v4/test_metrics.json`。
- 可直接查看的矩阵表：`reports/ch10/training/20261031-v4/per-label-confusion.csv`。
- 错例及代审：同目录`error-review.json`；原预测在模型目录上一级的`test_predictions.jsonl`。
- 训练曲线数据：`reports/ch10/training/20261031-v4/epochs.csv`。

“买大了想退”已返回退换货＋尺码。五条固定观察中四条完全正确；“我不退货，只想维修”仍多打质量问题。三诉求、否定及背景成分还会漏标或多标，页面统计用于决定知识补充顺序时，应再看对应原问。本章交付为实验模型旁路，不是已证明可靠的生产分类器。

## 功能演示

在项目根目录执行，无需重新训练即可检查最终模型：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --offline --locked python -m app.topics.cli predict \
  --model-dir models/ch10/classifier-20261031-v4/model \
  --text '买大了想退'
```

独立模型服务启动命令（18010已运行时不要重复启动）：

```bash
TOPIC_MODEL_DIR="$PWD/models/ch10/classifier-20261031-v4/model" \
TOPIC_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv run --offline --locked uvicorn app.topic_service:app \
  --host 127.0.0.1 --port 18010
```

迁移只新增指定表，默认预览；已有业务库不重建，重复apply会核验结构：

```bash
uv run --offline --locked python -m app.db.migrate_ch10 --env-file .env --database mewhelp
uv run --offline --locked python -m app.db.migrate_ch10 --env-file .env --database mewhelp --apply
```

主应用保留原有观测、数据库及会话检查点配置，在现有启动环境加上`TOPIC_ENABLED=true`、`TOPIC_SERVICE_URL=http://127.0.0.1:18010`和`TOPIC_REPORTS_DIR=reports/ch10/batches`。本次本机启动保留CH09的观测凭据及`.data/checkpoints/ch09-task8-validation.sqlite3`，运行地址为`http://127.0.0.1:18000`。

```bash
# 普通本地启动示例；在已配置好的原主应用环境运行。
TOPIC_ENABLED=true TOPIC_SERVICE_URL=http://127.0.0.1:18010 \
TOPIC_REPORTS_DIR=reports/ch10/batches \
uv run --offline --locked uvicorn app.main:app --host 127.0.0.1 --port 18000

# 只处理尚无分类结果的问题；返回202后查询状态。
curl -sS -X POST http://127.0.0.1:18000/api/flywheel/topics/classify \
  -H 'Content-Type: application/json' -d '{"reclassify":false}'
curl -sS http://127.0.0.1:18000/api/flywheel/topics/status
curl -sS 'http://127.0.0.1:18000/api/flywheel/topics?scope=all'
```

打开[主题分布页](http://127.0.0.1:18000/topics)。默认看待补知识问题，可切全部历史、手动归类或勾选重跑。后台默认每60秒检查，达到20条或已知时间的尾批等待24小时即处理；手动运行不受最小批量限制。分类不确定的记录保留待归类，不强塞“其他”。

## 输入与恢复边界（2026-11-01 修复）

聊天入口、主题批处理和本地分类服务统一接受最多4000字符，4001字符仍拒绝。模型内部继续按模型包的token长度截断，并在结果中报告`truncated`；这不改写问题池原话。个人字段脱敏不会增加文本长度，避免合法4000字符问题经清洗后反而超限。短个人值使用等长星号，较长值使用`[个人信息]`。

已补固话、长订单上下文、“姓名是/地址是”和无冒号联系人格式；业务问句、否定、尺码、日期和型号的回归同时保留。规则仍有自然语言覆盖边界，外部预标之前应继续抽查脱敏派生文本，不能把规则测试等同于所有个人资料格式均可识别。

新一轮的高水位读取或自动预检查失败/取消，只记录本次错误，不再把上一轮完成扫描改为失败；恢复连接后可以重试。分页把游标、未分类条件和数量限制下推到数据库，关联读取限定到当前候选；原轮索引一次建立，纯内存派生在线程执行。完整分布仍需处理所选范围的数据，不能将局部性能探针解释为生产端到端压测。

## 实际批处理证据

`reports/ch10/live-final-20261031/summary.json`记录两次真实全池重跑、SQL/API逐类对账和原池完整行摘要。21条原池记录未改写，20条当前分类、1条未确定；按可核实的原轮合并后全部历史20个统计问题，其中19个已归类。默认待补知识18个统计问题，其中17个已归类。多标签各主题数量之和可能超过问题总数。

停止模型服务的演练中，主应用和既有统计仍可读取，页面显示失败/可重试，没有用零覆盖旧结果。截图与观察原件在`reports/ch10/browser-resume/`。最终完整测试和审查结论见`docs/ch10-closeout.md`。

模型、数据与原始运行证据均保留在本机`models/ch10/`、`reports/ch10/`和本章`.superpowers/sdd/`目录，因体积及原问隐私而被Git忽略；换机器时需另外备份这些目录。程序、权威类目、文档与测试在功能分支中保留。

## 最新本机实例（2026-11-01 修复后）

主应用`127.0.0.1:18000`已于北京时间03:43加载提交`1d4de55`，该提交仅同步测试，产品源码与`f37ec79`一致。启动时PID46900，171个app源码摘要验证一致。入口为`.superpowers/sdd/2026-11-01-ch10-field-boundary-fixes/run-local-app.py`，启动清单为同目录的`app-runtime-20261031T194311Z.json`，正常停机标记是同目录的`stop-local-app`。原主PID42640通过旧`merge-fixes/stop-local-app`标记正常退出，旧标记保留，不能再直接用旧入口启动。维护时必须重新核实监听身份，不能只据本文PID结束进程。

分类服务`127.0.0.1:18010`保留原PID43020，创建时间仍为`2026-10-31T17:21:31.946070Z`，其入口仍为`.superpowers/sdd/2026-11-01-ch10-review-fixes/run-topic-service.py`。完整导入链不加载本轮修改的`data.py`和`training.py`模块，因此本轮只更新主应用；原CPU-v4模型保持运行，健康检查为`ready`，版本为`ch10-topic-1412cf9dd328ccfb`，没有重训。入口默认旧模型版本，若今后重新启动，必须显式设置`CH10_SERVING_RUN=20261031-v4`。原观测配置和CH09检查点继续保留。

最新验收位于`.superpowers/sdd/2026-11-01-ch10-field-boundary-fixes/runtime-acceptance-20261031T194310Z-759e75e9b25f/`，确认新主源码身份、原分类器身份与健康、一次普通归类和两个统计范围。普通归类新轮`5bd11ee397e74be99faf691d8d34374a`完成，检查1条、保存0条、未确定1条；默认18/17/1、全部历史20/19/1，原21行完整摘要不变。验收脚本先等待新端口和完整启动清单再发请求；普通归类会生成扫描原件，可能更新派生分类表，重跑前应先检查现有状态。前轮`merge-fixes/runtime-acceptance-20261031T185130Z-ef04e9cbca00/`仍保留为历史原件。固定双标签与4000/4001推理边界继续引用未改分类器的前轮验收`review-fixes/runtime-acceptance-20261031T172236Z-0816a86ea740/`，本轮未重复推理边界或前端视觉验收。
