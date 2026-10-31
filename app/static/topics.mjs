/** CH10 主题统计：只消费服务器汇总，不把未知或请求失败画成零。 */
const byId = (id) => document.getElementById(id);
let revision = 0;
let submitting = false;
let running = false;
let timer = null;
let controller = null;
let hasSnapshot = false;
let snapshotScope = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}
function count(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}
function formatted(value) {
  const number = count(value);
  return number === null ? "未知" : number.toLocaleString("zh-CN");
}
function localTime(value) {
  if (typeof value !== "string" || !value) return null;
  const stamp = new Date(/(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`);
  return Number.isNaN(stamp.getTime()) ? null : stamp.toLocaleString("zh-CN", { hour12: false });
}
function showError(text) {
  byId("topic-error").textContent = text;
  byId("topic-error").hidden = !text;
}
async function request(path, options = {}) {
  let response;
  try { response = await fetch(path, options); }
  catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("暂时无法连接服务，请稍后刷新。");
  }
  let data;
  try { data = await response.json(); }
  catch { throw new Error("服务返回的内容无法读取，请稍后刷新。"); }
  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === "string" ? detail : detail?.message;
    if (response.status === 404) throw new Error("主题统计服务尚未就绪，请稍后刷新。");
    throw new Error(message || ({ 409: "已有分类任务正在运行，请等待完成。", 422: "请求内容不符合要求，请刷新后重试。", 503: "主题分类服务尚未就绪，现有客服对话仍可使用。" })[response.status] || "本次请求没有完成，请重试。");
  }
  return data;
}
function syncControls() {
  byId("topic-run").disabled = submitting || running;
  byId("topic-reclassify").disabled = submitting || running;
  byId("topic-run").textContent = submitting ? "正在提交…" : running ? "正在归类…" : "归类问题";
}
function schedulePoll() {
  clearTimeout(timer);
  timer = null;
  if (running && !document.hidden) timer = setTimeout(refresh, 2500);
}
function renderStatus(value) {
  const status = value?.status;
  running = status === "running" || status === "queued";
  const messages = { idle: "当前没有分类任务", waiting: "正在等待更多问题，也可以手动归类", running: "分类任务正在后台运行", queued: "分类任务正在准备", complete: "最近一次归类已完成", completed: "最近一次归类已完成", done: "最近一次归类已完成", partial: "归类已完成，部分问题仍待确认", failed: "最近一次归类未完成，可以重试", interrupted: "上次分类已中断，可以重新运行", cancelled: "分类已停止，可以重新运行", disabled: "主题分类尚未启用", unavailable: "分类服务暂未就绪" };
  byId("topic-job-status").textContent = messages[status] || "分类任务状态暂未确认";
  syncControls();
  schedulePoll();
}
function renderDistribution(data) {
  if (!Array.isArray(data?.labels) || data.labels.length !== 17 || data.labels.some((item) => typeof item?.label !== "string" || !item.label || count(item.count) === null) || new Set(data.labels.map((item) => item.label)).size !== 17) {
    throw new Error("主题统计尚不完整，请刷新后重试。");
  }
  const total = count(data.total_questions);
  const classified = count(data.classified_questions);
  const remaining = count(data.unclassified_questions);
  if (total === null || classified === null || remaining === null || classified + remaining !== total) throw new Error("问题数量尚未核对完整，请稍后刷新。");
  byId("topic-total").textContent = formatted(total);
  byId("topic-classified").textContent = formatted(classified);
  byId("topic-unclassified").textContent = formatted(remaining);
  byId("topic-coverage").textContent = total ? `已覆盖 ${(classified / total * 100).toFixed(0)}% 的统计问题` : "当前范围内没有问题";
  const labels = [...data.labels].sort((left, right) => right.count - left.count);
  const maximum = Math.max(0, ...labels.map((item) => item.count));
  const chart = byId("topic-chart");
  chart.replaceChildren();
  for (const item of labels) {
    const row = element("li");
    const track = element("div", "topic-track");
    track.setAttribute("aria-hidden", "true");
    const fill = element("div", "topic-fill");
    fill.style.width = `${maximum ? item.count / maximum * 100 : 0}%`;
    track.append(fill);
    row.append(element("span", "topic-row-label", item.label), track, element("span", "topic-row-value", formatted(item.count)));
    chart.append(row);
  }
  const notice = byId("topic-chart-message");
  notice.hidden = maximum > 0;
  notice.textContent = total === 0 ? "这个范围内暂时没有待统计的问题。" : "还没有可展示的主题结果，可以点击“归类问题”开始处理。";
  byId("topic-focus-title").textContent = maximum ? labels[0].label : total ? "先完成问题归类" : "暂时没有积累的问题";
  const ties = labels.filter((item) => item.count === maximum).length;
  byId("topic-focus-text").textContent = maximum ? `${formatted(maximum)} 个问题涉及这个主题${ties > 1 ? `，另有 ${ties - 1} 个主题数量相同` : ""}。可以结合待审原话判断知识补充顺序。` : "新的问题积累后，可以在这里查看分布。";
  const uncertainHistory = count(data.history_uncertain_pool_rows);
  byId("topic-history-note").textContent = uncertainHistory === null ? "历史原轮信息尚未核对完整。" : uncertainHistory ? `其中 ${formatted(uncertainHistory)} 条旧记录无法核实原轮，暂按记录分别计数；这部分不能视为已经准确去重。` : "当前范围的原轮信息已纳入去重统计。";
  const sourceNote = data.source_note || data.provenance_note;
  byId("topic-source-note").hidden = typeof sourceNote !== "string" || !sourceNote;
  byId("topic-source-note").textContent = typeof sourceNote === "string" ? sourceNote : "";
  const version = data.model_version;
  const versionNote = classified === 0 ? "" : version === "mixed" ? "已有结果来自多个模型版本，比较数量时请留意。" : version === "unknown" || version === null ? "已有结果的模型来源暂无法完整核实。" : "";
  byId("topic-version-note").hidden = !versionNote;
  byId("topic-version-note").textContent = versionNote;
  const updated = localTime(data.updated_at || data.last_batch?.completed_at || data.last_batch?.ended_at);
  byId("topic-updated").textContent = updated ? `状态更新：${updated}` : "状态更新时间尚未取得";
  chart.setAttribute("aria-busy", "false");
  byId("topic-stats").setAttribute("aria-busy", "false");
  hasSnapshot = true;
}
/** 筛选和刷新使用独立代次；迟到响应不能覆盖当前范围。 */
async function refresh() {
  const current = ++revision;
  controller?.abort();
  controller = new AbortController();
  clearTimeout(timer);
  timer = null;
  byId("topic-refresh").disabled = true;
  const scope = byId("topic-scope").value;
  try {
    const [data, status] = await Promise.all([
      request(`/api/flywheel/topics?${new URLSearchParams({ scope })}`, { signal: controller.signal }),
      request("/api/flywheel/topics/status", { signal: controller.signal }),
    ]);
    if (current !== revision) return;
    renderDistribution(data);
    snapshotScope = scope;
    renderStatus(status);
    showError("");
  } catch (error) {
    if (current !== revision || error.name === "AbortError") return;
    showError(`${error.message}${hasSnapshot ? ` 下方保留上次成功读取的“${snapshotScope === "all" ? "全部历史问题" : "待补知识的问题"}”统计。` : ""}`);
    if (!hasSnapshot) byId("topic-chart-message").textContent = "暂时无法读取统计，不能把缺失的数据当成零。";
    byId("topic-job-status").textContent = running ? "正在等待任务最新状态…" : "分类任务状态尚未确认";
    schedulePoll();
  } finally {
    if (current === revision) byId("topic-refresh").disabled = false;
  }
}
async function classify() {
  if (submitting || running) return;
  submitting = true;
  syncControls();
  showError("");
  try {
    const result = await request("/api/flywheel/topics/classify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reclassify: byId("topic-reclassify").checked }) });
    renderStatus(result);
    await refresh();
  } catch (error) {
    showError(error.message);
    // 提交响应丢失时先读服务器状态，不能据网络错误断言任务没有启动。
    try { renderStatus(await request("/api/flywheel/topics/status")); } catch { byId("topic-job-status").textContent = "提交结果尚未确认，请刷新查看任务状态。"; }
  } finally {
    submitting = false;
    syncControls();
  }
}
byId("topic-refresh").addEventListener("click", refresh);
byId("topic-scope").addEventListener("change", refresh);
byId("topic-run").addEventListener("click", classify);
document.addEventListener("visibilitychange", () => { if (document.hidden) { clearTimeout(timer); timer = null; } else { refresh(); } });
window.addEventListener("pagehide", () => { clearTimeout(timer); controller?.abort(); });
refresh();
