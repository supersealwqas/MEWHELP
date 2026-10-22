/**
 * MewHelp 聊天界面控制器：管理本页会话、消息气泡、逐字展示及请求状态。
 * 刷新或新建会话会换用新标识；页面不读取密钥、不保存本地聊天记录。
 */
import { failurePresentation, streamChat } from "./chat-api.mjs";
import { renderMarkdown } from "./markdown.mjs";

const ui = {
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  stop: document.querySelector("#stop-button"),
  newChat: document.querySelector("#new-chat"),
  messages: document.querySelector("#messages"),
  conversation: document.querySelector("#conversation"),
  welcome: document.querySelector("#welcome"),
  status: document.querySelector("#request-status"),
  feedback: document.querySelector("#feedback"),
  sessionId: document.querySelector("#session-id"),
};
let sessionId = "";
let activeController = null;
let activeRenderer = null;
let followOutput = true;

/**
 * 创建新会话并清空页面，不调用后端删除接口。
 * @returns {void} 更新会话标识、空态和输入框。
 * @throws {Error} 浏览器不支持 Web Crypto 时可能抛出运行环境错误。
 */
function newConversation() {
  if (activeController) return;
  // getRandomValues 兼容非 HTTPS 的局域网测试，不依赖 randomUUID 的安全上下文限制。
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  let randomId = "";
  for (const byte of bytes) randomId += byte.toString(16).padStart(2, "0");
  sessionId = `web-${randomId}`;
  ui.sessionId.textContent = sessionId.slice(4, 16);
  ui.sessionId.title = `session_id: ${sessionId}`;
  ui.messages.replaceChildren();
  ui.welcome.hidden = false;
  ui.feedback.hidden = true;
  ui.input.value = "";
  ui.status.textContent = "准备好了，开始聊聊吧";
  followOutput = true;
  updateSendState();
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
  bubble.append(tools, content);
  const note = document.createElement("p");
  note.className = "message-note";
  note.hidden = true;
  body.append(author, bubble, note);
  article.append(avatar, body);
  ui.messages.append(article);
  scrollToLatest();
  return { article, bubble: content, tools, note };
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
  ui.send.disabled = Boolean(activeController) || !ui.input.value.trim();
}

/**
 * 切换请求期间的交互状态。
 * @param {boolean} busy 是否正在请求或显示本轮回复。
 * @returns {void} 更新按钮、输入框及无障碍状态。
 * @throws {Error} 无主动抛出的业务异常。
 */
function setBusy(busy) {
  ui.input.readOnly = busy;
  ui.newChat.disabled = busy;
  ui.stop.hidden = !busy;
  ui.form.setAttribute("aria-busy", String(busy));
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
  const message = ui.input.value.trim();
  if (activeController || !message) return;
  ui.welcome.hidden = true;
  ui.feedback.hidden = true;
  followOutput = true;
  appendMessage("user", message);
  const reply = appendMessage("assistant", "小喵正在思考…");
  reply.article.classList.add("is-pending");
  const renderer = createTextRenderer(reply.bubble);
  const controller = new AbortController();
  activeController = controller;
  activeRenderer = renderer;
  ui.input.value = "";
  ui.status.textContent = "正在等待小喵回复…";
  setBusy(true);

  /**
   * 将接口文本交给逐字队列，首块到达时退出等待状态。
   * @param {string} delta 模型文本增量。
   * @returns {void} 更新展示状态，将文本交给安全 Markdown 渲染器。
   * @throws {Error} 无主动抛出的业务异常。
   */
  function onDelta(delta) {
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
    await streamChat({ sessionId, message, signal: controller.signal, onDelta, onToolStatus });
    await renderer.finish();
    ui.status.textContent = "回复完成，可以接着追问";
  } catch (error) {
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
    // 工单已开始或服务端明确不可重试时，不自动回填建单原话。
    ui.input.value = presentation.restoreMessage ? message : "";
  } finally {
    reply.article.classList.remove("is-pending", "is-streaming");
    activeController = null;
    activeRenderer = null;
    setBusy(false);
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
ui.welcome.addEventListener("click", onSuggestionClick);
ui.conversation.addEventListener("scroll", onConversationScroll, { passive: true });
document.addEventListener("visibilitychange", onVisibilityChange);
newConversation();
