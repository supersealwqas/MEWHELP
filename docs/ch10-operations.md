# CH10 主题分类器运行说明

本章把低置信度问题在后台批量归类，实时客服对话不调用这个模型。主题表位于 `data/topics/taxonomy.json`，17个标签及边界由它统一定义；一句话可有多个标签，“其他”只作排他兜底。

## 当前数据与人工审核

2026-10-31只读导出得到21条问题池记录、16个不同原问，均带演示/验收来源标记；没有可证实的真实客户数据。17条旧记录缺少可信原轮映射，统计按池记录分别保留，不能宣称已经准确去重。其余4条可对应3个原轮。

模型预标、模拟数据成绩、实际客户成绩和人工审核是不同证据。未经用户实际复核的标签始终标为 `model_prelabel`；当前人工抽审待完成。已有知识审核队列的通过/驳回不代表主题标签经过人工确认。

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

基础语料先按全局关联组做80/10/10分割；同一原轮、相同正文、父子改写不跨分割。数据增强只处理训练集。`manifest.json`记录摘要、来源、各标签样本量；`review.md`与`review.json`供人工抽审。不能通过修改测试标签或反复调整测试阈值来掩饰错误。

## 固定权重及资源证据

- 来源：`hfl/chinese-roberta-wwm-ext`，revision `5c58d0b8ec1d9014354d691c538661bf00bfdb44`。
- 本地目录：`models/ch10/base/`，`source-manifest.json`保存文件摘要。
- 权重字节数411578458，SHA256 `1ded5a5a1c7841dee6e47942f7b5bf2bcf6f73ff19197580f852f7f638f86b35`。
- 模型配置类型为BERT，使用自动分类模型加载17维分类头；不依名称误用RoBERTa分词器。
- `reports/ch10/gpu-probe-20261031.json`证实RTX4060 Laptop上102280721个参数全部参与梯度计算和更新，batch2、长度128；峰值分配2071200768字节，预留2264924160字节。这是单次资源探测，完整训练证据另行记录。

## 后续运行记录

训练、模型服务、迁移、批处理和页面联调正在实施。实际命令、测试结果与报告路径在相应阶段完成后追加，当前不把待执行步骤标为通过。
