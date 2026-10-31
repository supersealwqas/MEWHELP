/** 知识改进后台：展示服务器原件、提交人工审核，并区分已知用量与未知覆盖。 */
const $ = (id) => document.getElementById(id);
const drafts = new Map();
let reviews = [], nextAfter = null, selectedId = null, detail = null;
let listRevision = 0, detailRevision = 0, usageRevision = 0, actionPending = false;

/** 创建纯文本节点；用户原话、模型建议和检索正文均不进入 HTML。 */
function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = String(text);
  return result;
}
function number(value) { return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN") : "未知"; }
function score(value) { return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "未取得"; }
function percent(value) { return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "未知"; }
function time(value) {
  if (!value) return "时间未记录";
  // MySQL 会话保存 UTC；无时区的 DATETIME 响应须先标明 UTC，再转成本机时间。
  const parsed = new Date(/(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`);
  return Number.isNaN(parsed.getTime()) ? "时间未记录" : parsed.toLocaleString("zh-CN", { hour12: false });
}
function messageFor(response, data) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (typeof detail?.message === "string") return detail.message;
  return ({ 404: "这条记录已不存在，请刷新列表。", 409: "状态已变化，请核对最新详情。", 422: "内容或时间格式不符合要求，请修改后重试。", 503: "服务暂未完成请求，请核对状态后重试。" })[response.status] || "请求未成功，请稍后重试。";
}
async function request(path, options) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error("网络暂时不可用，请重试。"); }
  let data;
  try { data = await response.json(); }
  catch { throw new Error("服务返回内容不可读取，请重试。"); }
  if (!response.ok) throw new Error(messageFor(response, data));
  return data;
}
function reviewLabel(row) {
  if (row.publication_status === "pending") return "发布待完成 · 答案已冻结";
  if (row.publication_status === "published") return "通过 · 已发布";
  return row.review_status || "状态未知";
}
function renderList() {
  const box = $("review-list");
  box.replaceChildren();
  if (!reviews.length) box.append(node("p", "fw-empty", "这个状态下暂无问题。新的反馈整理后会出现在待审列表。"));
  for (const row of reviews) {
    const button = node("button", "fw-review");
    button.type = "button";
    button.dataset.reviewId = row.id;
    button.setAttribute("aria-current", String(row.id === selectedId));
    const meta = node("span", "fw-review-meta");
    meta.append(node("span", "fw-badge", reviewLabel(row)), node("span", "", `出现 ${number(row.occurrence_count)} 次`));
    button.append(meta, node("span", "fw-review-question", row.normalized_question), node("span", "fw-review-preview", row.ai_suggested_answer ? `示例（仅参考）：${row.ai_suggested_answer}` : "暂无示例答案"));
    button.addEventListener("click", () => selectReview(row.id));
    box.append(button);
  }
  $("more-reviews").hidden = !nextAfter;
}

/** 每次筛选建立新代次；迟到的旧页不覆盖当前列表。 */
async function loadReviews(append = false) {
  const revision = ++listRevision;
  const params = new URLSearchParams({ limit: "30" });
  if ($("review-status").value) params.set("status", $("review-status").value);
  if (append && nextAfter) params.set("after_id", nextAfter);
  $("queue-status").textContent = "正在读取队列…";
  $("more-reviews").disabled = true;
  try {
    const data = await request(`/api/flywheel/reviews?${params}`);
    if (revision !== listRevision) return;
    const incoming = Array.isArray(data.items) ? data.items : [];
    reviews = append ? [...reviews, ...incoming.filter((row) => !reviews.some((old) => old.id === row.id))] : incoming;
    nextAfter = data.next_after_id;
    renderList();
    $("queue-status").textContent = `已显示 ${reviews.length} 条${nextAfter ? "，可继续加载" : ""}`;
  } catch (error) {
    if (revision === listRevision) $("queue-status").textContent = error.message;
  } finally {
    if (revision === listRevision) $("more-reviews").disabled = false;
  }
}

const retrievalLabels = {
  not_retrieved: "本轮未检索",
  ok: "已保存当轮检索证据",
  empty: "已检索，但没有找到结果",
  fault: "本轮检索发生故障；已有部分证据见下方",
  legacy_incomplete: "旧记录不完整；只展示当时已保存的部分",
  legacy_unavailable: "旧记录无法还原，不能确认当轮是否检索",
};
const sourceLabels = { retrieval_low_conf: "检索证据不足", self_check: "回答自查不足", user_feedback: "用户反馈没解决" };

/** 只消费可信展示投影，不能用原始 SQL NULL 或新检索猜测历史。 */
function originalView(original, index) {
  const box = node("details", "fw-original");
  box.append(node("summary", "", `${index + 1}. ${original.raw_question}`));
  box.append(node("p", "fw-muted", `${sourceLabels[original.source] || "来源未识别"} · ${time(original.created_at)}`));
  box.append(node("p", "fw-copy", original.raw_question));
  if (original.reason) box.append(node("p", "fw-copy", `记录原因：${original.reason}`));
  box.append(node("p", "fw-notice", retrievalLabels[original.retrieval_status] || "检索状态未记录，无法确认当轮证据"));
  for (const [attemptIndex, attempt] of (original.retrieval_snapshot?.attempts || []).entries()) {
    const block = node("div", "fw-attempt");
    block.append(node("h5", "", `检索尝试 ${attemptIndex + 1}`), node("p", "fw-copy", `查询：${attempt.query ?? "旧记录未保存"}`));
    block.append(node("p", "fw-muted", `${retrievalLabels[attempt.status] || "状态未知"} · 候选 ${number(attempt.candidate_count)} 条 · 返回上限 ${number(attempt.top_k)}`));
    if (attempt.fault) block.append(node("p", "fw-error", "检索发生故障，以下仅展示已保存的证据。"));
    for (const chunk of attempt.chunks || []) {
      const article = node("article", "fw-chunk");
      article.append(node("p", "fw-muted", `第 ${number(chunk.rank)} 条 · 召回分数 ${score(chunk.recall_score)} · 重排分数 ${score(chunk.rerank_score)}`));
      article.append(node("p", "fw-muted fw-wrap", `来源：${chunk.source_key ?? "未记录"} · 片段 ${chunk.chunk_id ?? "未记录"}`));
      article.append(node("p", "fw-copy", `问题：${chunk.question ?? "未保存"}`), node("p", "fw-copy", `原文：${chunk.answer ?? "未保存"}`));
      block.append(article);
    }
    box.append(block);
  }
  return box;
}

function renderDetail(row, feedback = "", failed = false, stateUnknown = false) {
  const box = $("review-detail");
  const frozen = row.approved_answer !== null && row.approved_answer !== undefined;
  const editable = row.review_status === "待审" && !frozen && !stateUnknown;
  box.replaceChildren(node("h3", "", row.normalized_question), node("p", "fw-muted", `${reviewLabel(row)} · 出现 ${number(row.occurrence_count)} 次 · ${time(row.created_at)}`));
  if (row.ai_suggested_answer) {
    const suggestion = node("details", "fw-suggestion");
    suggestion.append(node("summary", "", "查看模型示例答案（仅供参考）"), node("p", "fw-copy", row.ai_suggested_answer));
    box.append(suggestion);
  }
  const form = node("form", "fw-approval-form");
  const label = node("label", "", frozen ? "已冻结的人工核准答案" : "人工核准答案");
  label.htmlFor = "approved-answer";
  const input = node("textarea");
  input.id = "approved-answer";
  input.value = frozen ? row.approved_answer : (drafts.get(row.id) || "");
  input.readOnly = !editable;
  input.disabled = actionPending;
  input.required = editable;
  input.setAttribute("aria-describedby", "answer-limit");
  const length = node("p", "fw-muted");
  length.id = "answer-limit";
  const updateLength = () => {
    length.textContent = `已输入 ${Array.from(input.value).length} 字符。标准问题、答案和分类合计最多 1600 字符；发布前还会核验模型长度限制。`;
  };
  input.addEventListener("input", () => { drafts.set(row.id, input.value); updateLength(); });
  updateLength();
  form.append(label, input, length);
  if (frozen) form.append(node("p", "fw-notice", row.publication_status === "published" ? "此答案已发布到知识库。" : "答案已冻结，发布尚未完整完成。核对后可重试发布，不能修改正文或驳回。"));
  if (stateUnknown) form.append(node("p", "fw-notice", "暂时无法确认最新状态。输入已保留，请先刷新此条详情再继续审核。"));
  const actions = node("div", "fw-actions");
  if (editable) {
    const approve = node("button", "fw-button", "通过并发布");
    approve.type = "submit";
    approve.id = "approve-review";
    approve.disabled = actionPending;
    const reject = node("button", "fw-button fw-danger", "驳回此问题");
    reject.type = "button";
    reject.id = "reject-review";
    reject.disabled = actionPending;
    reject.addEventListener("click", () => reviewAction("reject", row.id));
    actions.append(approve, reject);
  }
  if (frozen && row.publication_status === "pending") {
    const retry = node("button", "fw-button", "重试发布");
    retry.type = "button";
    retry.id = "retry-review";
    retry.disabled = actionPending || stateUnknown;
    retry.addEventListener("click", () => reviewAction("retry", row.id));
    actions.append(retry);
  }
  const refresh = node("button", "fw-button fw-secondary", "刷新此条详情");
  refresh.type = "button";
  refresh.disabled = actionPending;
  refresh.addEventListener("click", () => selectReview(row.id));
  actions.append(refresh);
  form.append(actions);
  const status = node("p", failed ? "fw-error" : "fw-muted", feedback);
  status.id = "review-action-status";
  status.setAttribute("role", "status");
  form.append(status);
  form.addEventListener("submit", (event) => { event.preventDefault(); drafts.set(row.id, input.value); reviewAction("approve", row.id, input.value); });
  box.append(form, node("h4", "", `用户原件与当轮证据（${row.originals?.length ?? 0} 条）`));
  if (!row.originals?.length) box.append(node("p", "fw-muted", "暂无关联原件；不能据此推断当轮检索情况。"));
  (row.originals || []).forEach((original, index) => box.append(originalView(original, index)));
}

async function selectReview(id) {
  const revision = ++detailRevision;
  selectedId = id;
  detail = null;
  renderList();
  $("review-detail").replaceChildren(node("p", "fw-muted", "正在读取原件…"));
  try {
    const row = await request(`/api/flywheel/reviews/${encodeURIComponent(id)}`);
    if (revision !== detailRevision || selectedId !== id) return;
    detail = row;
    renderDetail(row);
  } catch (error) {
    if (revision !== detailRevision) return;
    const retry = node("button", "fw-button fw-secondary", "重新读取详情");
    retry.type = "button";
    retry.addEventListener("click", () => selectReview(id));
    $("review-detail").replaceChildren(node("p", "fw-error", error.message), retry);
  }
}

/** 动作绑定点击时的记录，失败后回读冻结状态且保留未提交草稿。 */
async function reviewAction(action, id, answer) {
  if (actionPending || selectedId !== id || !detail) return;
  if (action === "approve" && !answer?.trim()) { $("review-action-status").textContent = "请先填写人工核准答案。"; return; }
  const previous = detail;
  const revision = ++detailRevision;
  actionPending = true;
  renderDetail(previous, "正在处理，请稍候…");
  let feedback, failed = false, fresh = null;
  try {
    const options = { method: "POST" };
    if (action === "approve") { options.headers = { "Content-Type": "application/json" }; options.body = JSON.stringify({ answer }); }
    const result = await request(`/api/flywheel/reviews/${encodeURIComponent(id)}/${action}`, options);
    feedback = result.publication_status === "published" ? "已通过并发布，答案已进入知识库。" : action === "reject" ? "已驳回此问题。" : "发布尚未完成，答案已冻结，可重试发布。";
    if (result.publication_status === "published" || action === "reject") drafts.delete(id);
  } catch (error) { failed = true; feedback = `${error.message} 输入已保留。`; }
  try { fresh = await request(`/api/flywheel/reviews/${encodeURIComponent(id)}`); }
  catch { feedback += " 最新状态读取失败。"; }
  actionPending = false;
  if (selectedId === id && revision === detailRevision) {
    detail = fresh || previous;
    renderDetail(detail, feedback, failed || !fresh, !fresh);
  } else if (detail) renderDetail(detail);
  await loadReviews();
}

function stat(label, value, note) {
  const card = node("div", "fw-stat");
  card.append(node("span", "", label), node("strong", "", value));
  if (note) card.append(node("span", "", note));
  return card;
}
function usageTable(groups) {
  const table = node("table", "fw-table");
  const head = node("thead"), header = node("tr");
  for (const label of ["归属", "请求数", "模型调用", "已知输入", "已知输出", "已知 token", "用量不全的调用", "已知量占比", "金额"]) header.append(node("th", "", label));
  head.append(header);
  const body = node("tbody");
  for (const group of groups) {
    const row = node("tr");
    const kind = { summary: "后台 · 对话摘要", flywheel: "后台 · 问题整理", evaluation: "后台 · 质量评估" }[group.task_kind] || `后台 · ${group.task_kind}`;
    const label = group.task_kind === "chat" ? (group.intent === "unclassified" || !group.intent ? "对话 · 未分类" : `对话 · ${group.intent}`) : kind;
    for (const value of [label, number(group.request_count), number(group.model_call_count), number(group.input_tokens), number(group.output_tokens), number(group.total_tokens), number(group.unknown_usage_count), percent(group.known_token_share), group.cost === null || group.cost === undefined ? "暂无可靠金额" : String(group.cost)]) row.append(node("td", "", value));
    body.append(row);
  }
  table.append(head, body);
  return table;
}
async function loadUsage(event) {
  event?.preventDefault();
  const revision = ++usageRevision;
  const start = new Date($("usage-from").value), end = new Date($("usage-to").value);
  if (!Number.isFinite(start.getTime()) || !Number.isFinite(end.getTime()) || start >= end) { $("usage-status").textContent = "请填写有效时间，结束时间须晚于开始时间。"; return; }
  $("usage-status").textContent = "正在读取完整用量…";
  $("refresh-usage").disabled = true;
  // 切换窗口时清空旧数值，避免读取失败后把旧窗口误当当前账单。
  $("usage-summary").replaceChildren();
  $("usage-groups").replaceChildren();
  try {
    const response = await fetch(`/api/observability/costs?${new URLSearchParams({ from: start.toISOString(), to: end.toISOString() })}`);
    const data = await response.json();
    if (revision !== usageRevision) return;
    const coverage = data.coverage;
    if (coverage?.complete === false) { $("usage-status").textContent = `用量读取不完整，暂不展示合计。已读到 ${number(coverage.collected_model_call_count)} 次模型调用，请稍后重试。`; return; }
    if (!response.ok || coverage?.complete !== true) throw new Error(messageFor(response, data));
    $("usage-status").textContent = data.groups.length ? "窗口内观测已完整读取；模型未报告的用量仍为未知。" : "这个时间窗口暂无模型调用。";
    $("usage-summary").append(stat("已知 token", number(coverage.known_tokens), "仅合计模型实际报告的部分"), stat("完整报告用量的调用", percent(coverage.usage_coverage), `${number(coverage.fully_reported_call_count)} / ${number(coverage.model_call_count)} 次调用`), stat("用量不全的调用", number(coverage.unknown_usage_count), "缺失部分未作为零用量估算"));
    if (data.groups.length) $("usage-groups").append(usageTable(data.groups));
  } catch (error) {
    if (revision === usageRevision) $("usage-status").textContent = error instanceof TypeError ? "网络暂时不可用，请重试。" : error.message;
  } finally { if (revision === usageRevision) $("refresh-usage").disabled = false; }
}

$("review-status").addEventListener("change", () => {
  selectedId = null; detail = null; detailRevision += 1;
  $("review-detail").replaceChildren(node("p", "fw-empty", "选择一个问题，查看原件并审核。"));
  loadReviews();
});
$("refresh-reviews").addEventListener("click", () => loadReviews());
$("more-reviews").addEventListener("click", () => loadReviews(true));
$("process-gaps").addEventListener("click", async () => {
  $("process-gaps").disabled = true;
  $("queue-status").textContent = "正在整理新问题，可能需要一些时间…";
  try {
    const result = await request("/api/flywheel/process", { method: "POST" });
    await loadReviews();
    $("queue-status").textContent = `本次处理 ${number(result.processed)} 条，其中归并 ${number(result.merged)} 条，失败 ${number(result.failed)} 条。`;
  } catch (error) { $("queue-status").textContent = error.message; }
  finally { $("process-gaps").disabled = false; }
});
$("usage-form").addEventListener("submit", loadUsage);
const now = new Date();
const localValue = (value) => new Date(value.getTime() - value.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
$("usage-to").value = localValue(now);
$("usage-from").value = localValue(new Date(now.getTime() - 86400000));
loadReviews();
loadUsage();
