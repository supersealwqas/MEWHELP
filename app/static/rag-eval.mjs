/** 评估看板：只负责取数与绘图，所有指标都由后端算好，前端不重算。 */
const $ = (id) => document.getElementById(id);
const COLORS = { dense: "#7fb3d5", bm25: "#6cc28a", hybrid: "#f08a9b", hybrid_rerank: "#ef9a54" };
const STRATEGY_ORDER = ["dense", "bm25", "hybrid", "hybrid_rerank"];

let state = null;
let retrievalMetric = "mrr@10";
let genMetric = "faithfulness_macro";
// 数据刷新与任务状态分别编号，后返回的旧响应不得覆盖新成绩或新任务状态。
let loadRequest = 0;
let jobRevision = 0;
let currentJob = {};
let rerunPending = false;
let polling = false;
let pollInFlight = false;
let pollTimer = null;
// 两个评估入口共用服务端管理器与数据库执行锁，不能各自宣称空闲。
let productionBusy = false;
let productionSync = () => {};

/** 取数值，兼容 {value: x} 与裸数字两种结构。 */
function val(record, key) {
  if (!record) return null;
  const v = record[key];
  if (v && typeof v === "object") return typeof v.value === "number" ? v.value : null;
  return typeof v === "number" ? v : null;
}

/** 百分比或三位小数展示；null 显示为 —。 */
function fmt(x, digits = 3) {
  return x === null || x === undefined ? "—" : Number(x).toFixed(digits);
}

/** 创建元素。 */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* 非 JSON 错误体按状态文本处理 */ }
    throw new Error(detail);
  }
  return response.json();
}

/** 画一组分组柱状图。rows: [{label, sub, values: {strategy: number|null}}] */
function drawChart(container, rows, strategies, labels) {
  container.textContent = "";
  const chart = el("div", "chart");
  for (const row of rows) {
    const group = el("div", "group");
    const bars = el("div", "bars");
    for (const s of strategies) {
      const wrap = el("div", "bar-wrap");
      const value = row.values[s];
      wrap.appendChild(el("div", "bar-val", value === null ? "—" : fmt(value, 2)));
      const bar = el("div", "bar");
      // 数字仍按原比例画柱；未知分数只显示缺失文字，不画成零分。
      bar.style.height = `${Math.max((value ?? 0) * 100, 1.5)}%`;
      if (value === null) bar.classList.add("bar-missing");
      bar.style.background = COLORS[s] || "#999";
      bar.title = `${labels[s] || s}: ${value === null ? "无数据" : fmt(value, 4)}`;
      wrap.appendChild(bar);
      bars.appendChild(wrap);
    }
    group.appendChild(bars);
    const gl = el("div", "group-label", row.label);
    if (row.sub) gl.appendChild(el("small", null, row.sub));
    group.appendChild(gl);
    chart.appendChild(group);
  }
  container.appendChild(chart);
}

function renderLegend(labels, strategies) {
  const box = $("legend");
  box.textContent = "";
  for (const s of strategies) {
    const item = el("span");
    const dot = el("i");
    dot.style.background = COLORS[s];
    item.appendChild(dot);
    item.appendChild(document.createTextNode(labels[s] || s));
    box.appendChild(item);
  }
}

function renderMeta(data) {
  const row = $("metaRow");
  row.textContent = "";
  const meta = data.meta || {};
  const items = [
    ["运行目录", meta.run || "—"],
    ["题集", `${meta.sample_count ?? "—"} 样本`],
    ["语料", `${meta.chunk_count ?? "—"} 块`],
    ["生成模型", meta.model || "—"],
    ["裁判模型", meta.judge_model || "—"],
    ["耗时", meta.elapsed_seconds ? `${Math.round(meta.elapsed_seconds)} s` : "—"],
    ["源码版本", meta.source_revision || "—"],
  ];
  // 有出错样本时单独标出来：状态是 completed_with_errors，不是全绿。
  if (meta.error_total) {
    items.push(["出错样本", `${meta.error_total} 个（见 results.jsonl）`]);
  }
  for (const [k, v] of items) {
    const box = el("div", "meta-item");
    box.appendChild(el("div", "k", k));
    box.appendChild(el("div", "v", String(v)));
    row.appendChild(box);
  }
}

function renderKpi(data) {
  const row = $("kpiRow");
  row.textContent = "";
  const best = data.best || {};
  const overall = data.overall || {};

  // 最佳策略 MRR
  const k1 = el("div", "kpi");
  k1.appendChild(el("div", "k", "最佳整体 MRR@10"));
  k1.appendChild(el("div", "v", fmt(best.mrr, 3)));
  k1.appendChild(el("div", "n", best.label ? `${best.label} · 四策略最高` : "无数据"));
  row.appendChild(k1);

  // 各策略 Recall@10 都高说明召回不是瓶颈
  const recalls = STRATEGY_ORDER.map((s) => val(overall[s], "recall@10")).filter((x) => x !== null);
  const k2 = el("div", "kpi");
  k2.appendChild(el("div", "k", "最低 Recall@10"));
  k2.appendChild(el("div", "v", recalls.length ? fmt(Math.min(...recalls), 3) : "—"));
  k2.appendChild(el("div", "n", "四策略召回下限，接近 1 说明召回不是瓶颈"));
  row.appendChild(k2);

  // 库外问题正确拒答
  const un = STRATEGY_ORDER.map((s) => val(overall[s], "unanswerable_correct_refusal")).filter((x) => x !== null);
  const k3 = el("div", "kpi");
  k3.appendChild(el("div", "k", "库外问题正确拒答"));
  k3.appendChild(el("div", "v", un.length ? `${fmt(Math.min(...un) * 100, 0)}%` : "—"));
  k3.appendChild(el("div", "n", "取四策略最低值"));
  row.appendChild(k3);

  // 校准阈值
  const cal = data.calibration || {};
  const k4 = el("div", "kpi");
  k4.appendChild(el("div", "k", "校准题数"));
  k4.appendChild(el("div", "v", String(cal.case_count ?? "—")));
  const mins = Object.entries(cal.thresholds || {})
    .map(([s, r]) => `${s}=${r.minimum === null ? "null" : fmt(r.minimum, 2)}`);
  k4.appendChild(el("div", "n", mins.length ? mins.join(" · ") : "无校准记录"));
  row.appendChild(k4);
}

function renderRetrieval(data) {
  renderLegend(data.strategy_labels || {}, data.strategies);
  const buckets = data.buckets || [];
  const labels = data.bucket_labels || {};
  const rows = buckets.map((b) => {
    const values = {};
    for (const s of data.strategies) {
      values[s] = val((data.by_bucket[s] || {})[b], retrievalMetric);
    }
    return { label: labels[b] || b, sub: b, values };
  });
  // 总体列单独一组，与分桶区分开。
  const overallValues = {};
  for (const s of data.strategies) overallValues[s] = val((data.overall || {})[s], retrievalMetric);
  rows.push({ label: "总体", sub: "overall", values: overallValues });
  drawChart($("retrievalChart"), rows, data.strategies, data.strategy_labels || {});
}

function renderGen(data) {
  const buckets = data.buckets || [];
  const labels = data.bucket_labels || {};
  const rows = buckets.map((b) => {
    const values = {};
    for (const s of data.strategies) values[s] = val((data.by_bucket[s] || {})[b], genMetric);
    return { label: labels[b] || b, sub: b, values };
  });
  const overallValues = {};
  for (const s of data.strategies) overallValues[s] = val((data.overall || {})[s], genMetric);
  rows.push({ label: "总体", sub: "overall", values: overallValues });
  drawChart($("genChart"), rows, data.strategies, data.strategy_labels || {});
}

function renderOverallTable(data) {
  const table = $("overallTable");
  table.textContent = "";
  const head = el("thead");
  const hr = el("tr");
  for (const title of ["策略", "MRR@10", "Recall@10", "完整证据@10", "覆盖率", "误拒率", "库外正确拒答", "忠实度", "错误"]) {
    hr.appendChild(el("th", null, title));
  }
  head.appendChild(hr);
  table.appendChild(head);

  const body = el("tbody");
  for (const s of data.strategies) {
    const r = (data.overall || {})[s] || {};
    const tr = el("tr");
    tr.appendChild(el("td", null, data.strategy_labels[s] || s));
    const cells = [
      ["mrr@10", 3], ["recall@10", 3], ["all_evidence@10", 3],
      ["coverage", 3], ["false_refusal_rate", 3], ["unanswerable_correct_refusal", 3],
      ["faithfulness_macro", 3],
    ];
    for (const [key, digits] of cells) {
      const v = val(r, key);
      const td = el("td", "num", fmt(v, digits));
      // 误拒率越低越好，其余越高越好，用颜色点出方向。
      if (v !== null) {
        const good = key === "false_refusal_rate" ? v <= 0.15 : v >= 0.8;
        if (good) td.classList.add("good");
        else if (key === "false_refusal_rate" && v >= 0.4) td.classList.add("bad");
      }
      tr.appendChild(td);
    }
    tr.appendChild(el("td", "num", String(r.error_count ?? 0)));
    body.appendChild(tr);
  }
  table.appendChild(body);
}

function renderDifficultyTable(data) {
  const table = $("difficultyTable");
  table.textContent = "";
  const head = el("thead");
  const hr = el("tr");
  for (const title of ["策略", "难度", "样本", "MRR@10", "Recall@10", "覆盖率"]) hr.appendChild(el("th", null, title));
  head.appendChild(hr);
  table.appendChild(head);

  const body = el("tbody");
  for (const s of data.strategies) {
    const block = (data.by_difficulty || {})[s] || {};
    for (const level of ["easy", "medium", "hard"]) {
      const r = block[level];
      if (!r) continue;
      const tr = el("tr");
      tr.appendChild(el("td", null, data.strategy_labels[s] || s));
      tr.appendChild(el("td", null, { easy: "易", medium: "中", hard: "难" }[level] || level));
      tr.appendChild(el("td", "num", String(r.sample_count ?? "—")));
      tr.appendChild(el("td", "num", fmt(val(r, "mrr@10"), 3)));
      tr.appendChild(el("td", "num", fmt(val(r, "recall@10"), 3)));
      tr.appendChild(el("td", "num", fmt(val(r, "coverage"), 3)));
      body.appendChild(tr);
    }
  }
  table.appendChild(body);
}

function renderJob(job) {
  const pill = $("jobPill");
  const msg = $("jobMsg");
  const status = job.status || "idle";
  pill.className = `pill ${status === "running" ? "running" : status === "failed" ? "failed" : "done"}`;
  pill.textContent = { idle: "未运行", running: "运行中", done: "已完成", completed: "已完成", completed_with_errors: "已完成，含错误", busy: "执行位置被占用", persist_failed: "保存失败", failed: "失败", interrupted: "已中断" }[status] || "状态未知";
  msg.textContent = job.message || "";
  $("rerunBtn").disabled = rerunPending || status === "running" || productionBusy;
  productionSync();
  const logBox = $("logBox");
  if (job.log && job.profile !== "production") {
    logBox.hidden = false;
    $("jobLog").textContent = job.log;
  } else {
    logBox.hidden = true;
  }
}

function renderEmpty(data) {
  const box = document.createElement("div");
  box.className = "card empty";
  box.textContent = data.hint || "还没有可展示的评估结果。";
  $("retrievalChart").textContent = "";
  $("retrievalChart").appendChild(box);
}

/** 加载最新成绩；参数：无；返回：Promise<void>；异常：接口错误在页面展示。 */
async function load() {
  const request = ++loadRequest;
  const revision = jobRevision;
  let data;
  try {
    data = await api("/api/rag-eval/overview");
  } catch (error) {
    if (request === loadRequest) $("retrievalChart").textContent = `取数失败：${error.message}`;
    return;
  }
  if (request !== loadRequest) return;
  state = data;
  if (!state.present) {
    renderEmpty(state);
  } else {
    renderMeta(state);
    renderKpi(state);
    renderRetrieval(state);
    renderGen(state);
    renderOverallTable(state);
    renderDifficultyTable(state);
  }
  // 刷新期间若轮询或重跑已推进状态，只更新成绩，不回退任务与按钮。
  if (revision === jobRevision && !rerunPending) updateJob(state.job || {});
}

/** 接收任务状态并协调单一轮询；参数：job后端任务；返回：无；异常：无主动抛出。 */
function updateJob(job) {
  jobRevision += 1;
  currentJob = job;
  renderJob(job);
  polling = job.status === "running";
  if (!polling && pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  if (polling && pollTimer === null) pollJob();
}

/** 轮询状态与日志，终态刷新成绩；参数：无；返回：Promise<void>；异常：网络错误稍后重试。 */
async function pollJob() {
  // 刷新、重跑、定时器共用此入口，任何时刻最多一个job请求在途。
  if (!polling || rerunPending || pollInFlight) return;
  pollInFlight = true;
  const revision = jobRevision;
  let delay = 3000;
  try {
    const job = await api("/api/rag-eval/job");
    if (revision !== jobRevision || rerunPending) return;
    updateJob(job);
    if (!polling) await load();
  } catch {
    if (revision === jobRevision) delay = 5000;
  } finally {
    pollInFlight = false;
    // 即使旧请求被忽略，也由此处续接当前任务，避免断链或生成多个定时器。
    if (polling && !rerunPending && pollTimer === null) {
      pollTimer = setTimeout(() => {
        pollTimer = null;
        pollJob();
      }, delay);
    }
  }
}

for (const button of document.querySelectorAll("#metricSeg button")) {
  button.addEventListener("click", () => {
    retrievalMetric = button.dataset.metric;
    for (const other of document.querySelectorAll("#metricSeg button")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    if (state?.present) renderRetrieval(state);
  });
}

for (const button of document.querySelectorAll("#genSeg button")) {
  button.addEventListener("click", () => {
    genMetric = button.dataset.metric;
    for (const other of document.querySelectorAll("#genSeg button")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    if (state?.present) renderGen(state);
  });
}

$("refreshBtn").addEventListener("click", load);
$("rerunBtn").addEventListener("click", async () => {
  if (rerunPending || productionBusy) return;
  rerunPending = true;
  loadRequest += 1;
  jobRevision += 1;
  polling = false;
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  $("rerunBtn").disabled = true;
  productionSync();
  try {
    await api("/api/rag-eval/rerun", { method: "POST" });
    rerunPending = false;
    // 提交期间发出的overview也属于旧任务，不能在启动成功后中止轮询。
    jobRevision += 1;
    polling = true;
    pollJob();
  } catch (error) {
    rerunPending = false;
    updateJob(currentJob);
    $("jobMsg").textContent = `启动失败：${error.message}`;
  }
});

load();

/** 独立初始化生产历史；旧四策略视图与兼容回归保持原入口。 */
if ($("productionRuns")?.dataset.enabled === "true") initializeProduction();

function initializeProduction() {
  let runs = [], visibleCount = 10, runsRevision = 0, statusRevision = 0;
  let job = {}, starting = false, checking = false, timer = null;
  let statusKnown = false;
  const metrics = [["recall_at_k", "召回率 Recall@10", "recall@10"], ["mrr", "排名质量 MRR@10", "mrr@10"], ["faithfulness", "忠实度", "faithfulness_macro"]];
  const completed = new Set(["completed", "completed_with_errors"]);
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const valueText = (value) => finite(value) ? value.toFixed(4) : "未取得";
  const stamp = (value) => {
    if (!value) return "时间未记录";
    // 无时区的数据库时间是 UTC；带时区的调度时间保留其原始偏移。
    const parsed = new Date(/(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`);
    return Number.isFinite(parsed.getTime()) ? parsed.toLocaleString("zh-CN", { hour12: false }) : "时间未记录";
  };
  const valid = (run) => run?.metrics?.schema_version === 1 && run.metrics.scope === "production_knowledge_subpipeline" && completed.has(run.metrics.status);
  const comparable = (current, previous) => valid(current) && valid(previous) && Boolean(current.metrics.comparison_key) && current.metrics.comparison_key === previous.metrics.comparison_key;
  const busy = () => starting || job.status === "running" || job.database_busy === true;

  function buttons() {
    productionBusy = busy() || !statusKnown;
    $("productionRerun").disabled = productionBusy || rerunPending || currentJob.status === "running";
    $("rerunBtn").disabled = productionBusy || rerunPending || currentJob.status === "running";
  }
  productionSync = buttons;
  buttons();

  function delta(current, previous, key) {
    if (!previous) return { text: "暂无上轮", className: "production-missing" };
    if (!comparable(current, previous)) return { text: "口径不同或结果无效，不可比", className: "production-missing" };
    const now = current.metrics[key], before = previous.metrics[key];
    if (!finite(now) || !finite(before)) return { text: "分数缺失，无法比较", className: "production-missing" };
    const difference = now - before;
    return { text: `${difference > 0 ? "上升 +" : difference < 0 ? "下降 " : "持平 "}${difference.toFixed(4)}`, className: difference < 0 ? "production-down" : difference > 0 ? "production-up" : "production-missing" };
  }

  /** 元信息只显示明确白名单，不渲染内部状态、路径或完整提示词。 */
  function metadataView(run) {
    const m = run.metrics || {}, info = m.metadata || {};
    const details = el("details");
    details.appendChild(el("summary", "", "查看本轮口径与分母"));
    const lines = [
      `产物编号：${m.artifact_id ?? "未记录"}`,
      `指标版本：${m.schema_version ?? "未记录"}；范围：生产知识流程`,
      `题集摘要：${info.dataset_digest ?? "未记录"}`,
      `数据划分：${info.split ?? "未记录"}；题数：${run.dataset_size ?? "未记录"}`,
      `知识版本：${info.corpus_version ?? "未记录"}`,
      `策略版本：${info.policy_version ?? "未记录"}`,
      `生成模型：${info.models?.generation ?? "未记录"}`,
      `裁判模型：${info.models?.judge ?? "未记录"}`,
      `每轮模型调用上限：${info.budget_config?.max_model_calls ?? "未记录"}；决策上限：${info.budget_config?.max_decisions ?? "未记录"}`,
      `单次输入预算：${info.budget_config?.request_token_budget ?? "未记录"}；整轮预算：${info.budget_config?.total_token_budget ?? "未记录"}`,
      `错误样本：${m.error_count ?? "未知"}；拒答样本：${m.refusal_count ?? "未知"}`,
    ];
    for (const [, label, aggregateKey] of metrics) {
      const aggregate = m.aggregate?.[aggregateKey];
      lines.push(`${label}：分子 ${finite(aggregate?.numerator) ? String(aggregate.numerator) : "未知"} / 分母 ${finite(aggregate?.denominator) ? String(aggregate.denominator) : "未知"}`);
    }
    for (const [key, label] of [["production", "生产流程"], ["judge", "测量裁判"]]) {
      const usage = m.usage?.[key];
      if (usage) lines.push(`${label}：已知 token ${usage.known_actual_tokens ?? "未知"}，未知用量记录 ${usage.unknown_usage_records ?? "未知"}`);
    }
    details.appendChild(el("p", "production-meta", lines.join("\n")));
    return details;
  }

  function renderRuns() {
    const latest = runs.at(-1), previous = runs.at(-2);
    $("productionKpis").replaceChildren();
    $("productionRuns").replaceChildren();
    $("productionMore").hidden = runs.length <= visibleCount;
    if (!latest) {
      $("productionComparison").textContent = "还没有已入库的生产评估结果。运行完成并保存后会出现在这里。";
      return;
    }
    for (const [key, label] of metrics) {
      const card = el("div", "fw-stat");
      card.appendChild(el("span", "", label));
      card.appendChild(el("strong", "", valid(latest) ? valueText(latest.metrics[key]) : "结果无效"));
      const change = delta(latest, previous, key);
      card.appendChild(el("span", change.className, `较相邻上轮：${change.text}`));
      $("productionKpis").appendChild(card);
    }
    const explanation = !previous ? "目前只有一轮，尚无差值。" : comparable(latest, previous) ? "两轮口径一致，可比较知识版本变化后的差值。" : "相邻两轮的题集、模型、策略或指标口径不同，或结果无效，因此不计算差值。";
    $("productionComparison").textContent = `最新：${stamp(latest.created_at)}。${explanation}${latest.metrics?.status === "completed_with_errors" ? " 最新轮次含错误样本，请结合分母和错误数判断。" : ""}`;
    const table = el("table", "fw-table production-table");
    const head = el("thead"), header = el("tr");
    for (const label of ["运行与版本", "召回率 / 较上轮", "排名质量 / 较上轮", "忠实度 / 较上轮", "样本 / 错误 / 拒答"]) header.appendChild(el("th", "", label));
    head.appendChild(header); table.appendChild(head);
    const body = el("tbody");
    for (let index = runs.length - 1; index >= Math.max(0, runs.length - visibleCount); index -= 1) {
      const run = runs[index], m = run.metrics || {}, row = el("tr");
      const title = el("td", "", stamp(run.created_at));
      title.appendChild(el("small", "", `${run.triggered_by || "来源未知"} · ${m.status === "completed" ? "已完成" : m.status === "completed_with_errors" ? "已完成，含错误" : "结果状态无效"}`));
      title.appendChild(el("small", "", `知识版本：${m.metadata?.corpus_version ?? "未记录"}`));
      title.appendChild(metadataView(run)); row.appendChild(title);
      for (const [key] of metrics) {
        const cell = el("td", "", valid(run) ? valueText(m[key]) : "结果无效");
        const change = delta(run, runs[index - 1], key);
        cell.appendChild(el("span", `production-delta ${change.className}`, change.text));
        row.appendChild(cell);
      }
      row.appendChild(el("td", "", `${run.dataset_size ?? "未知"} / ${m.error_count ?? "未知"} / ${m.refusal_count ?? "未知"}`));
      body.appendChild(row);
    }
    table.appendChild(body); $("productionRuns").appendChild(table);
  }

  /** 服务端游标按升序，依次读齐后才选最新轮次，避免旧的第一页冒充最新。 */
  async function loadRuns() {
    const revision = ++runsRevision;
    $("productionStatus").textContent = "正在读取生产历史…";
    try {
      const collected = [], cursors = new Set();
      let cursor = null;
      do {
        const query = new URLSearchParams({ limit: "200" });
        if (cursor) query.set("after_id", cursor);
        const data = await api(`/api/rag-eval/production/runs?${query}`);
        if (revision !== runsRevision) return;
        if (!Array.isArray(data.items)) throw new Error("历史记录格式暂不可读");
        collected.push(...data.items);
        cursor = data.next_after_id;
        if (cursor && cursors.has(cursor)) throw new Error("历史分页异常，请重试");
        if (cursor) cursors.add(cursor);
      } while (cursor);
      runs = collected;
      renderRuns();
      $("productionStatus").textContent = `共 ${runs.length} 轮已保存结果 · 数据刷新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
    } catch (error) {
      if (revision === runsRevision) $("productionStatus").textContent = `生产历史读取失败：${error.message}。${runs.length ? "下方保留上次读取的记录。" : ""}`;
    }
  }

  function showJob(extra = "") {
    let message;
    if (starting) message = "正在提交生产评估…";
    else if (job.status === "running") message = job.profile === "legacy" ? "四策略评估正在运行，两个入口共用执行位置。" : job.admission === "pending" ? "本实例已接收，正在确认执行位置；尚未正式开始评估。" : "生产评估正在运行，完成后会自动刷新历史。";
    else message = ({ idle: "暂无正在运行的评估。", completed: "生产评估已完成并保存。", completed_with_errors: "生产评估已保存，但含错误样本，请核对分母与错误数。", done: "四策略评估已完成。", busy: "另一个评估正在占用执行位置，本次未开始。", persist_failed: "评估运行结束，但保存失败；本次尚未进入历史记录。", failed: "评估未成功完成，请检查服务后重试。", interrupted: "上次评估已中断，可核对服务后重新运行。" })[job.status] || "正在确认评估状态。";
    if (job.database_busy && job.status !== "running") message += " 当前有其他进程正在评估，暂不能重跑。";
    if (job.next_run_at) message += ` 下次定时评估：${stamp(job.next_run_at)}。`;
    $("productionJob").textContent = extra || message;
    buttons();
  }
  function schedule(delay = 3000) {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; checkJob(); }, delay);
  }
  async function checkJob() {
    // 启动期间的旧请求可能尚未返回，保留后续确认，避免轮询链断开。
    if (checking || starting) { schedule(500); return; }
    checking = true;
    const revision = statusRevision;
    try {
      const received = await api("/api/rag-eval/production/job");
      if (revision !== statusRevision || starting) return;
      const wasRunning = job.status === "running" || job.database_busy === true;
      const changed = received.status !== job.status || received.finished_at !== job.finished_at;
      job = received; statusKnown = true;
      showJob();
      if (wasRunning && !busy() || changed && completed.has(job.status)) await loadRuns();
      // 页面保持打开时也会发现定时评估；繁忙时加快轮询，闲时每30秒确认一次。
      schedule(busy() ? 3000 : 30000);
    } catch {
      if (revision === statusRevision) { statusKnown = false; showJob("评估状态暂不可读取，稍后自动重试；也可点击刷新记录。"); schedule(5000); }
    } finally { checking = false; }
  }
  $("productionRefresh").addEventListener("click", () => { loadRuns(); checkJob(); });
  $("productionMore").addEventListener("click", () => { visibleCount += 10; renderRuns(); });
  $("productionRerun").addEventListener("click", async () => {
    if (starting || busy() || rerunPending || currentJob.status === "running" || !statusKnown) return;
    starting = true; statusRevision += 1;
    if (timer !== null) { clearTimeout(timer); timer = null; }
    showJob();
    try {
      job = await api("/api/rag-eval/production/rerun", { method: "POST" });
      statusKnown = true;
      starting = false;
      showJob();
    } catch (error) {
      starting = false; statusKnown = false;
      showJob(`启动请求未确认：${error.message}。正在重新确认状态，请勿重复提交。`);
    } finally { schedule(500); }
  });
  loadRuns();
  checkJob();
}
