/**
 * MewHelp 聊天界面控制器：管理本页会话、消息气泡、逐字展示及请求状态。
 * 刷新保留本标签的会话标识并回读服务端动作；页面不读取密钥、不缓存聊天正文。
 */
import { failurePresentation, streamWorkflowChat, resumeOrderSelection, resumeTicketConfirmation, getCurrentActions } from "./chat-api.mjs";
import { renderMarkdown } from "./markdown.mjs";
import { attachAnswerTools } from "./citations.mjs";
import { attachSuggestedActions, attachOrderSelection, attachTicketPreview, attachRefundAction } from "./actions.mjs";
import { createConversationBrowser } from "./conversations.mjs";

const ui = {
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  stop: document.querySelector("#stop-button"),
  newChat: document.querySelector("#new-chat"),
  refreshStatus: document.querySelector("#refresh-status"),
  messages: document.querySelector("#messages"),
  conversation: document.querySelector("#conversation"),
  welcome: document.querySelector("#welcome"),
  status: document.querySelector("#request-status"),
  feedback: document.querySelector("#feedback"),
  sessionId: document.querySelector("#session-id"),
  historyList: document.querySelector("#conversation-list"),
  historyEmpty: document.querySelector("#history-empty"),
  historyStatus: document.querySelector("#history-status"),
  historyCount: document.querySelector("#history-count"),
  historyMore: document.querySelector("#more-conversations"),
  historyRefresh: document.querySelector("#refresh-conversations"),
  historyDisclosure: document.querySelector("#history-disclosure"),
};
let sessionId = "";
let activeController = null;
let activeRenderer = null;
let pendingActions = 0;
let followOutput = true;
let requestGeneration = 0;
let activeReply = null;
let historyLoading = false;
const replyByTurn = new Map();
const orderCards = new Set();
const conversationBrowser = createConversationBrowser({ onSelect: applyConversation, onChange: paintConversations,
  onLoading(value) { historyLoading = value; paintControls(Boolean(activeController)); } });

/** 保存当前标识并更新页脚；参数无；返回void；异常：存储禁用时忽略。 */
function rememberSession() {
  try { sessionStorage.setItem("mewhelp.session", sessionId); } catch { /* 私密模式仍可聊天。 */ }
  ui.sessionId.textContent = sessionId.startsWith("web-") ? sessionId.slice(4, 16) : sessionId.slice(0, 12);
  ui.sessionId.title = `session_id: ${sessionId}`;
}

/** 按只读快照绘制侧栏，文本不作为HTML；参数浏览状态；返回void；异常：DOM错误。 */
function paintConversations(state) {
  ui.historyList.replaceChildren();
  for (const item of state.items) {
    const wrapper = document.createElement("div"); wrapper.setAttribute("role", "listitem");
    const button = document.createElement("button"); button.type = "button"; button.className = "conversation-item";
    button.disabled = state.busy; button.setAttribute("aria-current", String(item.session_id === sessionId));
    const preview = document.createElement("span"); preview.className = "conversation-preview";
    preview.textContent = item.preview || "尚未提问的会话"; button.append(preview);
    if (item.has_summary) { const badge = document.createElement("span"); badge.className = "conversation-summary"; badge.textContent = "已摘要"; button.append(badge); }
    button.addEventListener("click", async () => {
      const generation = requestGeneration;
      ui.historyStatus.textContent = "正在回载历史…";
      const loaded = await conversationBrowser.select(item);
      if (!loaded && generation === requestGeneration && !conversationBrowser.snapshot().loading)
        ui.historyStatus.textContent = "暂时未能读取历史，当前会话已保留。";
    });
    wrapper.append(button); ui.historyList.append(wrapper);
  }
  ui.historyCount.textContent = state.items.length ? String(state.items.length) : "";
  ui.historyEmpty.hidden = Boolean(state.items.length);
  ui.historyMore.hidden = state.nextBefore === null;
  ui.historyMore.disabled = state.listing || state.busy;
  ui.historyRefresh.disabled = state.listing || state.busy;
}

/** 完整历史到齐后切换；旧快照只恢复正文/引用，可执行动作另查当前状态。参数：会话与原文；返回void；异常：DOM错误。 */
function applyConversation({ conversation, messages }) {
  if (activeController || pendingActions) return;
  requestGeneration++; sessionId = conversation.session_id; rememberSession();
  ui.messages.replaceChildren(); replyByTurn.clear(); orderCards.clear();
  ui.welcome.hidden = Boolean(messages.length); ui.feedback.hidden = true; ui.input.value = "";
  const originals = new Map(), tools = new Map();
  followOutput = false;
  for (const row of messages) {
    if (row.role === "user") { originals.set(row.turn_id, row.content ?? ""); appendMessage("user", row.content ?? ""); }
    if (row.role === "tool" || row.tool_calls?.length) {
      if (!tools.has(row.turn_id)) tools.set(row.turn_id, []);
      tools.get(row.turn_id).push(row.role === "tool" ? row.content ?? "" : JSON.stringify(row.tool_calls, null, 2));
    }
    if (row.role !== "assistant" || (!row.content && row.tool_calls?.length)) continue;
    const reply = appendMessage("assistant", row.content ?? "");
    reply.turnId = row.turn_id; reply.originalMessage = originals.get(row.turn_id) ?? "";
    reply.article.dataset.sourceTurnId = row.turn_id; replyByTurn.set(row.turn_id, reply);
    reply.bubble.innerHTML = renderMarkdown(row.content ?? "");
    if (row.turn_status === "completed") {
      const metadata = { ...row.rag_metadata, ...currentIdentity(), turn_id: row.turn_id,
        citations: Array.isArray(row.rag_metadata?.citations) ? row.rag_metadata.citations : [] };
      attachAnswerTools(reply, row.content ?? "", metadata);
      reply.historyTools = true; // 防止随后当前状态回读重复挂引用/反馈；不复活历史业务动作。
    } else { reply.note.hidden = false; reply.note.textContent = "这轮没有完整结束，可核实处理状态或继续提问。"; }
  }
  for (const [turnId, records] of tools) {
    const reply = replyByTurn.get(turnId);
    if (!reply) continue;
    const details = document.createElement("details"); details.className = "history-records";
    const label = document.createElement("summary"); label.textContent = "查看本轮工具记录";
    const source = document.createElement("pre"); source.textContent = records.join("\n\n");
    details.append(label, source); reply.tools.append(details); reply.tools.hidden = false;
  }
  followOutput = true; scrollToLatest();
  ui.status.textContent = "历史已完整回载，可以继续提问"; ui.historyStatus.textContent = "";
  if (window.matchMedia("(max-width: 860px)").matches) ui.historyDisclosure.open = false;
  updateSendState();
  void syncCurrent(null, requestGeneration);
}

/** 冻结当前身份，异步回调不得读后来的会话；参数无；返回对象；异常无。 */
function currentIdentity() { return Object.freeze({ session_id: sessionId, user_id: "demo-user" }); }

/**
 * 创建新会话并清空页面，不调用后端删除接口。
 * @returns {void} 更新会话标识、空态和输入框。
 * @throws {Error} 浏览器不支持 Web Crypto 时可能抛出运行环境错误。
 */
function newConversation({ restore = false } = {}) {
  if (activeController || pendingActions) return;
  requestGeneration += 1;
  // getRandomValues 兼容非 HTTPS 的局域网测试，不依赖 randomUUID 的安全上下文限制。
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  let randomId = "";
  for (const byte of bytes) randomId += byte.toString(16).padStart(2, "0");
  sessionId = `web-${randomId}`;
  let restoring = false;
  try {
    const stored = sessionStorage.getItem("mewhelp.session");
    if (restore && typeof stored === "string" && stored.trim() && [...stored].length <= 64) {
      sessionId = stored; restoring = true;
    }
  } catch { /* 隐私模式禁用存储时仍可使用当前页面会话。 */ }
  rememberSession();
  conversationBrowser.startNew(sessionId);
  ui.messages.replaceChildren();
  replyByTurn.clear();
  orderCards.clear();
  ui.welcome.hidden = false;
  ui.feedback.hidden = true;
  ui.input.value = "";
  ui.status.textContent = "准备好了，开始聊聊吧";
  ui.historyStatus.textContent = "";
  followOutput = true;
  updateSendState();
  if (restoring) {
    const generation = requestGeneration;
    ui.status.textContent = "正在回载历史，请稍候…";
    void conversationBrowser.restore(sessionId).then(loaded => {
      if (!loaded && generation === requestGeneration) {
        ui.status.textContent = "历史暂未回载，可从侧栏重试或继续提问";
        void syncCurrent(null, generation);
      }
    });
  } else void conversationBrowser.refresh();
}

/**
 * 仅在用户仍靠近底部时跟随新回复，不打断向上阅读历史的操作。
 * @returns {void} 必要时更新消息区滚动位置。
 * @throws {Error} 无主动抛出的业务异常。
 */
function scrollToLatest() {
  if (followOutput) ui.conversation.scrollTop = ui.conversation.scrollHeight;
}

/**
 * 记录用户是否希望继续自动跟随消息。
 * @returns {void} 更新自动滚动开关。
 * @throws {Error} 无主动抛出的业务异常。
 */
function onConversationScroll() {
  const remaining = ui.conversation.scrollHeight - ui.conversation.scrollTop - ui.conversation.clientHeight;
  followOutput = remaining < 80;
}

/**
 * 创建消息气泡；用户输入保持纯文本，客服正文预留 Markdown 容器。
 * @param {"user"|"assistant"} role 消息的发送者。
 * @param {string} text 初始内容，不会被解析为 HTML。
 * @returns {{article: HTMLElement, bubble: HTMLElement, tools: HTMLElement, note: HTMLElement}} 消息节点集合。
 * @throws {Error} 无主动抛出的业务异常。
 */
function appendMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message message--${role}`;
  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "你" : "喵";
  avatar.setAttribute("aria-hidden", "true");
  const body = document.createElement("div");
  body.className = "message-body";
  const author = document.createElement("p");
  author.className = "message-author";
  author.textContent = role === "user" ? "你" : "小喵";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  const tools = document.createElement("div");
  tools.className = "tool-trail";
  tools.hidden = true;
  tools.setAttribute("aria-label", "本轮工具调用");
  tools.setAttribute("aria-live", "polite");
  const content = document.createElement("div");
  content.className = role === "assistant" ? "message-content markdown-content" : "message-content";
  content.textContent = text;
  const stage = document.createElement("p");
  stage.className = "workflow-stage";
  stage.hidden = true;
  stage.setAttribute("aria-live", "polite");
  bubble.append(stage, tools, content);
  const actions = document.createElement("div");
  actions.className = "answer-actions";
  const note = document.createElement("p");
  note.className = "message-note";
  note.hidden = true;
  body.append(author, bubble, actions, note);
  article.append(avatar, body);
  ui.messages.append(article);
  scrollToLatest();
  return { article, bubble: content, tools, note, stage, actions };
}

/**
 * 将已收到的增量排入轻量逐字队列，不等待整个网络请求完成。
 * @param {HTMLElement} node 回复文字节点；调用方负责其消息状态样式。
 * @returns {{append: Function, finish: Function, flush: Function, hasText: Function}} 渲染控制接口。
 * @throws {Error} 无主动抛出的业务异常。
 */
function createTextRenderer(node) {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let pending = [];
  let text = "";
  let frame = null;
  let resolveFinish = null;

  /**
   * 在一帧内显示少量已收到的字符，积压较多时加速，避免长回复滞后。
   * @returns {void} 更新文字并继续调度，队列排空时通知完成。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function paint() {
    frame = null;
    const count = reducedMotion.matches ? pending.length : Math.max(1, Math.ceil(pending.length / 60));
    text += pending.splice(0, count).join("");
    // 只写入禁用原始 HTML 的 Markdown 解析结果，避免将模型文本直接当网页执行。
    node.innerHTML = renderMarkdown(text);
    scrollToLatest();
    if (pending.length) frame = requestAnimationFrame(paint);
    else if (resolveFinish) { resolveFinish(); resolveFinish = null; }
  }

  /**
   * 加入新的文本分块；按 Unicode 码点拆分，避免截断表情的代理对。
   * @param {string} delta SSE 已收到的文本。
   * @returns {void} 启动增量渲染。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function append(delta) {
    pending.push(...Array.from(delta));
    // 后台标签的动画帧可能暂停；此时直接显示已接收内容，不积压到整轮结束。
    if (document.hidden || reducedMotion.matches) flush();
    else if (frame === null) frame = requestAnimationFrame(paint);
  }

  /**
   * 立刻显示已收到的队列内容，用于取消、错误或切到后台标签时收尾。
   * @returns {void} 清空渲染队列，并解除等待。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function flush() {
    if (frame !== null) cancelAnimationFrame(frame);
    if (pending.length) text += pending.join("");
    pending = [];
    frame = null;
    if (text) node.innerHTML = renderMarkdown(text);
    if (resolveFinish) { resolveFinish(); resolveFinish = null; }
    scrollToLatest();
  }

  /**
   * 等待网络结束后余下字符显示完成，后台标签不等待被浏览器暂停的动画。
   * @returns {Promise<void>} 当前积压字符全部显示后完成。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function finish() {
    if (document.hidden) flush();
    return new Promise(function waitForPaint(resolve) {
      // 参数 resolve：Promise 完成回调；返回 void；无业务异常。
      if (!pending.length) resolve();
      else resolveFinish = resolve;
    });
  }

  /**
   * 判断模型是否返回了任何内容，包含尚未绘制的分块。
   * @returns {boolean} 是否已有文本。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function hasText() { return Boolean(text || pending.length); }
  return { append, finish, flush, hasText };
}

/**
 * 根据输入与请求状态更新发送按钮，阻止同一会话并发提交。
 * @returns {void} 更新按钮与输入框校验状态。
 * @throws {Error} 无主动抛出的业务异常。
 */
function updateSendState() {
  ui.input.setCustomValidity("");
  ui.send.disabled = Boolean(activeController) || pendingActions > 0 || historyLoading || !ui.input.value.trim();
}

/**
 * 切换请求期间的交互状态。
 * @param {boolean} busy 是否正在请求或显示本轮回复。
 * @returns {void} 更新按钮、输入框及无障碍状态。
 * @throws {Error} 无主动抛出的业务异常。
 */
function setBusy(busy) {
  conversationBrowser.setBusy(busy || pendingActions > 0);
  paintControls(busy);
}

/** 绘制输入控件，历史回载不改变业务互斥；参数回复状态；返回void；异常：DOM错误。 */
function paintControls(busy) {
  ui.input.readOnly = busy || pendingActions > 0 || historyLoading;
  ui.newChat.disabled = busy || pendingActions > 0;
  ui.refreshStatus.disabled = busy || pendingActions > 0 || historyLoading;
  ui.stop.hidden = !busy;
  ui.form.setAttribute("aria-busy", String(busy || pendingActions > 0 || historyLoading));
  updateSendState();
}

/**
 * 更新具体工具名和调用状态，图标与文本分别创建，不把服务端字段解析为 HTML。
 * @param {HTMLElement} badge 本次调用的标签节点，按 tool_call_id 复用。
 * @param {object} tool 服务端工具状态，或页面停止接收后的 interrupted 状态。
 * @returns {void} 更新标签内容、样式及无障碍说明。
 * @throws {Error} 无主动抛出的业务异常。
 */
function updateToolBadge(badge, tool) {
  const prefixes = { running: "正在调用", retrying: "正在重试", succeeded: "调用了",
    failed: "调用失败", interrupted: "调用中断" };
  // 被服务端跳过的调用也使用 failed；明确标为未执行，不能当作已经调用失败。
  const prefix = tool.status === "failed" && tool.label === "未执行"
    ? "未执行" : prefixes[tool.status];
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0L21.5 5.5a6 6 0 0 1-7.9 7.9l-7.1 7.1a2.1 2.1 0 0 1-3-3l7.1-7.1a6 6 0 0 1 7.9-7.9z");
  icon.append(path);
  const label = document.createElement("span");
  label.textContent = `${prefix} `;
  const name = document.createElement("code");
  name.className = "tool-badge-name";
  name.textContent = tool.name;
  label.append(name);
  if (tool.status === "interrupted") label.append(" · 结果待核实");
  badge.replaceChildren(icon, label);
  badge.dataset.status = tool.status;
  badge.title = `${tool.name} · ${tool.label}`;
  badge.setAttribute("aria-label", badge.textContent);
}

/**
 * 提交消息并将流式结果写入同一个客服气泡。
 * @param {SubmitEvent} event 表单提交事件。
 * @returns {Promise<void>} 本轮成功、失败或取消后恢复操作。
 * @throws {Error} 网络及解析异常在内部转为页面提示，不向事件循环抛出。
 */
async function sendMessage(event) {
  event.preventDefault();
  const message = ui.input.value;
  if (activeController || pendingActions || historyLoading || !message.trim()) return;
  for (const card of orderCards) card.disable("已发出新问题，请以当前处理状态为准。");
  ui.welcome.hidden = true;
  ui.feedback.hidden = true;
  followOutput = true;
  appendMessage("user", message);
  const reply = appendMessage("assistant", "小喵正在思考…");
  reply.originalMessage = message;
  ui.input.value = "";
  await runReply(reply, currentIdentity(), { message });
}

/** 完成后装配独立动作和引用；参数为原气泡/正文/保存快照/可选回执；返回void；异常：非法元数据。 */
function presentCompleted(reply, answer, metadata, receipt = null, ticketReceipt = null) {
  if (!reply.completed) {
    reply.card?.disable("原轮处理已完成。");
    reply.card?.element.remove();
    reply.bubble.innerHTML = renderMarkdown(answer);
    reply.note.hidden = true;
    reply.completed = true;
    if (!reply.historyTools) attachAnswerTools(reply, answer, metadata);
    else reply.bubble.innerHTML = renderMarkdown(answer, metadata.citations);
  }
  const options = { appendMessage, onBusy: (busy) => {
    pendingActions = Math.max(0, pendingActions + (busy ? 1 : -1));
    setBusy(Boolean(activeController));
  } };
  attachSuggestedActions(reply.actions, metadata, { ...options, receipt: ticketReceipt });
  attachRefundAction(reply.actions, metadata, { ...options, receipt });
}

/** 装配原轮选单/继续按钮；参数为原气泡、冻结身份、动作及接受状态；返回void；异常：非法卡片。 */
function presentSelection(reply, identity, action, acceptedOrderId = null, resumeRequired = false) {
  reply.card?.disable();
  reply.card?.element.remove();
  reply.turnId = action.source_turn_id;
  reply.article.dataset.sourceTurnId = reply.turnId;
  replyByTurn.set(reply.turnId, reply);
  reply.card = attachOrderSelection(reply.actions, identity, action, {
    acceptedOrderId, resumeRequired, onSelect: async (frozenIdentity, selectedAction, orderId) => {
      if (activeController || pendingActions || frozenIdentity.session_id !== sessionId) return;
      await runReply(reply, frozenIdentity, { action: selectedAction, orderId, message: reply.originalMessage });
    },
  });
  orderCards.add(reply.card);
}

/** 恢复同一原轮工单卡片；参数为气泡、身份、冻结预览及服务端状态；返回void；异常：非法快照。 */
function presentTicket(reply, identity, action, current = {}) {
  reply.card?.disable();
  reply.card?.element.remove();
  reply.turnId = action.source_turn_id;
  reply.article.dataset.sourceTurnId = reply.turnId;
  replyByTurn.set(reply.turnId, reply);
  reply.card = attachTicketPreview(reply.actions, identity, action, {
    decision: current.ticket_decision, resumeRequired: current.resume_required, receipt: current.ticket_receipt,
    ticketStatus: current.ticket_status,
    onDecide: async (frozenIdentity, selectedAction, decision) => {
      if (activeController || pendingActions || frozenIdentity.session_id !== sessionId) return;
      await runReply(reply, frozenIdentity, { action: selectedAction, decision, message: reply.originalMessage });
    },
  });
  orderCards.add(reply.card);
}

/** 服务端状态优先于缓存卡片，旧请求响应不能污染新问；参数为可选原气泡及代次；返回Promise；异常在当前页面显示。 */
async function syncCurrent(fallbackReply = null, generation = requestGeneration) {
  const identity = currentIdentity();
  try {
    const current = await getCurrentActions(identity);
    if (generation !== requestGeneration || identity.session_id !== sessionId || !current.source_turn_id) return;
    ui.welcome.hidden = true;
    const reply = replyByTurn.get(current.source_turn_id) ?? fallbackReply ?? appendMessage("assistant", "已恢复上次处理状态。");
    reply.turnId = current.source_turn_id;
    reply.article.dataset.sourceTurnId = reply.turnId;
    replyByTurn.set(reply.turnId, reply);
    if (current.turn_status === "completed") {
      presentCompleted(reply, current.answer ?? "", { ...current.completion_metadata, ...identity }, current.refund_receipt, current.ticket_receipt);
      if (current.ticket_confirmation) presentTicket(reply, identity, current.ticket_confirmation, current);
      ui.status.textContent = current.ticket_status === "unknown" ? "工单结果待核实，请先核实原操作" : "已恢复完成结果，可以继续提问";
    } else if (current.ticket_confirmation) {
      reply.bubble.textContent = current.ticket_receipt ? "已核实工单处理结果。" :
        current.ticket_decision === "cancel" && !current.resume_required ? "已取消，未创建工单。" : "请核对下面的工单信息。";
      presentTicket(reply, identity, current.ticket_confirmation, current);
      ui.status.textContent = current.resume_required ? "可以继续原来的工单处理" : "已恢复工单状态";
    } else if (current.turn_status === "awaiting_input" && current.order_selection) {
      reply.bubble.textContent = "请选择本次要处理的订单。";
      presentSelection(reply, identity, current.order_selection);
      ui.status.textContent = "等待选择订单，也可以发送新的问题";
    } else if (current.resume_required && current.order_selection) {
      reply.bubble.textContent = "订单已确认，点击继续处理即可接上原来的问题。";
      presentSelection(reply, identity, current.order_selection, current.accepted_order_id, true);
      ui.status.textContent = "请继续处理已确认的订单";
    } else {
      reply.card?.disable("这张选单卡已失效，请以当前处理状态为准。");
      reply.note.hidden = false;
      reply.note.textContent = current.turn_status === "running" ? "原问题正在处理中，可稍后核实状态。" : "上次处理未完成，可重新提问。";
    }
  } catch (error) {
    if (generation === requestGeneration) {
      ui.feedback.textContent = `暂时无法恢复处理状态：${error.message}`;
      ui.feedback.hidden = false;
    }
  }
}

/** 新问或原轮续跑共用安全渲染，各回调和finally都核验请求代次；参数为气泡/身份/输入；返回Promise；异常在本气泡处理。 */
async function runReply(reply, identity, { message, action = null, orderId = null, decision = null }) {
  const generation = ++requestGeneration;
  const isCurrent = () => generation === requestGeneration && identity.session_id === sessionId;
  reply.article.classList.add("is-pending");
  reply.note.hidden = true;
  const renderer = createTextRenderer(reply.bubble);
  const controller = new AbortController();
  activeController = controller;
  activeRenderer = renderer;
  activeReply = reply;
  ui.status.textContent = "正在等待小喵回复…";
  setBusy(true);

  /**
   * 将接口文本交给逐字队列，首块到达时退出等待状态。
   * @param {string} delta 模型文本增量。
   * @returns {void} 更新展示状态，将文本交给安全 Markdown 渲染器。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function onDelta(delta) {
    if (!isCurrent()) return;
    reply.article.classList.remove("is-pending");
    reply.article.classList.add("is-streaming");
    ui.status.textContent = "小喵正在回复…";
    renderer.append(delta);
  }

  const badges = new Map();
  const toolStatuses = new Map();

  /**
   * 按调用标识更新标签，同一工具的不同调用分别保留。
   * @param {object} status 已经通过接口客户端校验的工具状态。
   * @returns {void} 更新标签及正文尚未到达时的等待提示。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function onToolStatus(status) {
    if (!isCurrent()) return;
    toolStatuses.set(status.tool_call_id, status);
    let badge = badges.get(status.tool_call_id);
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "tool-badge";
      badges.set(status.tool_call_id, badge);
      reply.tools.append(badge);
    }
    reply.tools.hidden = false;
    updateToolBadge(badge, status);
    if (!renderer.hasText()) {
      reply.bubble.textContent = ["running", "retrying"].includes(status.status)
        ? `${status.label}…` : "小喵正在整理结果…";
      ui.status.textContent = reply.bubble.textContent;
    }
    scrollToLatest();
  }

  try {
    const handlers = { sessionId: identity.session_id, userId: identity.user_id, message, signal: controller.signal, onDelta, onToolStatus,
      onStage: (stage) => {
        if (!isCurrent()) return;
        reply.turnId = stage.source_turn_id;
        reply.article.dataset.sourceTurnId = reply.turnId;
        replyByTurn.set(reply.turnId, reply);
        reply.stage.hidden = false;
        reply.stage.textContent = `${stage.label}${stage.status === "running" ? "…" : "完成"}`;
        if (stage.stage === "understand" && stage.status === "completed" && typeof stage.resolved_query === "string" && stage.resolved_query !== message) {
          reply.resolvedQuestion = stage.resolved_query;
        }
        if (reply.resolvedQuestion) reply.stage.textContent += ` · 理解为：${reply.resolvedQuestion}`;
      } };
    const result = action?.type === "confirm_ticket" ? await resumeTicketConfirmation(identity, action, decision, handlers) :
      action ? await resumeOrderSelection(identity, action, orderId, handlers) : await streamWorkflowChat(handlers);
    if (!isCurrent()) return;
    await renderer.finish();
    if (!isCurrent()) return;
    if (result.status === "awaiting_input") {
      if (result.action.type === "confirm_ticket") {
        reply.bubble.textContent = "请核对工单信息，确认提交后才会创建。";
        presentTicket(reply, identity, result.action);
        ui.status.textContent = "等待确认或取消工单";
      } else {
        reply.bubble.textContent = "请选择本次要处理的订单。";
        presentSelection(reply, identity, result.action);
        ui.status.textContent = "等待选择订单，也可以发送新的问题";
      }
    } else if (result.status === "accepted") {
      reply.bubble.textContent = "决定已保存，正在核实处理状态…";
      await syncCurrent(reply, generation);
    } else {
      if (result.metadata) presentCompleted(reply, result.answer, result.metadata);
      if (action?.type === "confirm_ticket") await syncCurrent(reply, generation);
      else ui.status.textContent = "回复完成，可以接着追问";
    }
  } catch (error) {
    if (!isCurrent()) return;
    renderer.flush();
    const stopped = controller.signal.aborted;
    const presentation = failurePresentation({ error, stopped, toolStatuses: toolStatuses.values() });
    for (const [callId, badge] of badges) {
      if (["running", "retrying"].includes(badge.dataset.status)) {
        updateToolBadge(badge, { ...toolStatuses.get(callId), status: "interrupted", label: "结果待核实" });
      }
    }
    if (!renderer.hasText()) reply.bubble.textContent = presentation.status;
    reply.note.textContent = presentation.note;
    reply.note.hidden = false;
    ui.status.textContent = presentation.status;
    if (!stopped) {
      ui.feedback.textContent = presentation.feedback;
      ui.feedback.hidden = false;
    }
    // 普通聊天失败可回填；独立建单结果留在原回复的动作状态中。
    ui.input.value = !action && presentation.restoreMessage ? (message ?? "") : "";
    if (!stopped) await syncCurrent(reply, generation);
  } finally {
    if (!isCurrent()) return;
    reply.article.classList.remove("is-pending", "is-streaming");
    activeController = null;
    activeRenderer = null;
    activeReply = null;
    setBusy(false);
    void conversationBrowser.refresh();
    scrollToLatest();
    ui.input.focus({ preventScroll: true });
  }
}

/**
 * 停止当前请求，并保留已经收到的文字。
 * @returns {void} 触发 fetch 取消与绘制收尾。
 * @throws {Error} 无主动抛出的业务异常。
 */
function stopReply() {
  activeController?.abort();
  activeRenderer?.flush();
  activeReply?.article.classList.remove("is-pending", "is-streaming");
  if (activeReply) {
    activeReply.note.textContent = "已停止接收，可核实服务端保存的处理状态。";
    activeReply.note.hidden = false;
  }
  requestGeneration += 1;
  activeController = activeRenderer = activeReply = null;
  ui.status.textContent = "已停止接收，可以继续提问";
  setBusy(false);
}

/**
 * 处理 Enter 发送；中文输入法选字和 Shift+Enter 不触发发送。
 * @param {KeyboardEvent} event 输入框键盘事件。
 * @returns {void} 需要时通过正常表单路径发送。
 * @throws {Error} 无主动抛出的业务异常。
 */
function onInputKeyDown(event) {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!ui.send.disabled) ui.form.requestSubmit();
  }
}

/**
 * 将示例问题填入输入框，由用户确认发送，避免点击示例就消耗模型额度。
 * @param {MouseEvent} event 欢迎区域的点击事件。
 * @returns {void} 只更新输入，不调用接口。
 * @throws {Error} 无主动抛出的业务异常。
 */
function onSuggestionClick(event) {
  const button = event.target.closest("button[data-prompt]");
  if (!button || activeController) return;
  ui.input.value = button.dataset.prompt;
  updateSendState();
  ui.input.focus();
}

/**
 * 页面进入后台时冲刷队列，避免 requestAnimationFrame 暂停导致完成状态悬挂。
 * @returns {void} 必要时显示已收到的文字。
 * @throws {Error} 无主动抛出的业务异常。
 */
function onVisibilityChange() {
  if (document.hidden) activeRenderer?.flush();
}

ui.form.addEventListener("submit", sendMessage);
ui.input.addEventListener("input", updateSendState);
ui.input.addEventListener("keydown", onInputKeyDown);
ui.stop.addEventListener("click", stopReply);
ui.newChat.addEventListener("click", newConversation);
ui.historyRefresh.addEventListener("click", () => { void conversationBrowser.refresh(); });
ui.historyMore.addEventListener("click", () => { void conversationBrowser.more(); });
ui.refreshStatus.addEventListener("click", () => { if (!activeController && !pendingActions) void syncCurrent(null, requestGeneration); });
ui.welcome.addEventListener("click", onSuggestionClick);
ui.conversation.addEventListener("scroll", onConversationScroll, { passive: true });
document.addEventListener("visibilitychange", onVisibilityChange);
if (window.matchMedia("(max-width: 860px)").matches) ui.historyDisclosure.open = false;
newConversation({ restore: true });
