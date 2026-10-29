/** 完成回复的独立建议按钮：转人工只显示固定话术，建单须经原生弹窗明确确认。 */
import { confirmTicket, validateSuggestedActions, validateOrderSelection, validateTicketPreview, validateRefundAction,
  getRefundReasons, submitRefund, getCurrentActions } from "./chat-api.mjs";

const attached = new WeakMap();
const refundAttached = new WeakSet();

/** 创建纯文本DOM；参数为文档/标签/文本/类名；返回元素；异常：DOM错误。 */
function textNode(doc, tag, text = "", className = "") {
  const node = doc.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

/** 展示完整工单描述及冻结的确认/取消动作；参数为容器、身份、预览和恢复状态；返回卡片控制器；异常：非法预览。 */
export function attachTicketPreview(container, identity, payload, { decision = null, resumeRequired = false,
  receipt = null, ticketStatus = null, onDecide } = {}) {
  const action = validateTicketPreview(payload);
  const frozenIdentity = Object.freeze({ ...identity });
  const doc = container.ownerDocument;
  const card = textNode(doc, "section", "", "ticket-preview");
  card.dataset.actionId = action.action_id;
  card.dataset.sourceTurnId = action.source_turn_id;
  card.setAttribute("aria-label", "工单预览");
  card.append(textNode(doc, "h3", "工单预览"), textNode(doc, "p", `类型：${action.ticket_type}`, "ticket-type"),
    textNode(doc, "p", action.description, "ticket-description"));
  const controls = textNode(doc, "div", "", "ticket-controls");
  const note = textNode(doc, "p", "确认后将保存为待处理工单。", "action-note");
  note.setAttribute("role", "status");
  card.append(controls, note);
  const buttons = [];
  let disabled = false;

  /** 同步禁用所有动作，连点只能发出一个请求。参数为提示；返回void；异常无。 */
  function disable(message) {
    disabled = true;
    for (const button of buttons) button.disabled = true;
    if (message) note.textContent = message;
  }

  const hasReceipt = receipt?.action_id === action.action_id && validReceipt(receipt);
  if (hasReceipt) note.textContent = `工单已创建：${receipt.ticket_no} · ${receipt.status}`;
  else if (ticketStatus === "cancelled" && decision === "confirm") note.textContent = "本次工单未创建，可以继续提问。";
  else if (decision === "cancel" && !resumeRequired) note.textContent = "已取消，未创建工单。";
  else if (decision === "confirm" && !resumeRequired) note.textContent = "确认决定已保存，正在核实处理结果。请通过“核实状态”查看，不要重复建单。";
  else {
    const choices = resumeRequired && decision ? [[decision, "继续处理"]] : [["confirm", "确认提交"], ["cancel", "取消"]];
    if (resumeRequired) note.textContent = decision === "cancel" ? "取消决定已保存，继续完成原操作。" : "确认决定已保存，可以继续原来的处理。";
    for (const [value, label] of choices) {
      const button = textNode(doc, "button", label, `button ${value === "confirm" ? "button-primary" : "button-secondary"}`);
      button.type = "button";
      buttons.push(button);
      controls.append(button);
      button.addEventListener("click", async () => {
        if (disabled || button.disabled) return;
        disable(value === "confirm" ? "正在提交，请稍候…" : "正在取消，请稍候…");
        try { await onDecide(frozenIdentity, action, value); }
        catch (error) { note.textContent = error.message || "请核实原操作的处理状态。"; }
      });
    }
  }
  container.append(card);
  return { disable, element: card };
}

/** 仅接受服务端实际回执的最小展示字段。参数为回执；返回布尔；异常无。 */
function validReceipt(receipt) {
  return typeof receipt.ticket_no === "string" && receipt.ticket_no.trim() && receipt.ticket_no.length <= 32 &&
    receipt.status === "待处理";
}

/** 绑定本人订单选择，回调始终使用冻结身份；参数为容器/身份/卡片/续跑选项；返回禁用控制器；异常：非法快照。 */
export function attachOrderSelection(container, identity, payload, { onSelect, acceptedOrderId = null, resumeRequired = false } = {}) {
  const action = validateOrderSelection(payload);
  const frozenIdentity = Object.freeze({ ...identity });
  const doc = container.ownerDocument;
  const card = textNode(doc, "section", "", "order-selection");
  card.dataset.actionId = action.action_id;
  card.dataset.sourceTurnId = action.source_turn_id;
  card.setAttribute("aria-label", "选择本次处理的订单");
  const title = textNode(doc, "h3", acceptedOrderId ? `已选择订单 ${acceptedOrderId}` : "请选择要处理的订单");
  const note = textNode(doc, "p", resumeRequired ? "订单已确认，可以继续原来的处理。" : "只显示当前账号的订单。", "action-note");
  note.setAttribute("role", "status");
  const choices = textNode(doc, "div", "", "order-options");
  card.append(title, choices, note);
  const buttons = [];
  let disabled = false;

  /** 冻结当前卡片，防止连续点选或新问后旧卡复活；参数为提示；返回void；异常：无。 */
  function disable(message) {
    disabled = true;
    for (const button of buttons) button.disabled = true;
    if (message) note.textContent = message;
  }

  for (const order of action.orders.filter((order) => !acceptedOrderId || order.order_id === acceptedOrderId)) {
    const button = textNode(doc, "button", resumeRequired ? "继续处理" :
      `${order.product_name} · 订单 ${order.order_id} · ${order.status}`, "button button-secondary order-option");
    button.type = "button";
    button.disabled = Boolean(acceptedOrderId && !resumeRequired);
    buttons.push(button);
    choices.append(button);
    button.addEventListener("click", async () => {
      if (disabled || button.disabled) return;
      disable("已选择，正在继续处理…");
      try { await onSelect(frozenIdentity, action, order.order_id); }
      catch (error) { note.textContent = error.message || "暂时无法继续，请核实当前处理状态。"; }
    });
  }
  container.append(card);
  return { disable, element: card };
}

/** 给已完成退款原轮附加原因表单；参数为容器/元数据/网络与忙碌回调；返回void；异常：非法快照。 */
export function attachRefundAction(container, metadata, { fetchImpl = globalThis.fetch, onBusy = () => {}, receipt = null } = {}) {
  const action = validateRefundAction(metadata.refund_action, metadata.turn_id);
  if (!action || refundAttached.has(container)) return;
  if (metadata.status !== "completed") throw new Error("退款申请必须来源于已完成答复。");
  const identity = Object.freeze({ session_id: metadata.session_id, user_id: metadata.user_id ?? "demo-user" });
  const doc = container.ownerDocument;
  const card = textNode(doc, "section", "", "refund-action");
  card.dataset.actionId = action.action_id;
  card.append(textNode(doc, "h3", `订单 ${action.order_id} · 退款申请`),
    textNode(doc, "p", "提交后将保存为待处理，不代表退款已批准或到账。", "action-note"));
  const form = textNode(doc, "form", "", "refund-form");
  const label = textNode(doc, "label", "退款原因");
  const select = doc.createElement("select");
  select.required = true;
  select.disabled = true;
  select.setAttribute("aria-label", "退款原因");
  label.append(select);
  const button = textNode(doc, "button", "正在加载原因…", "button button-primary action-button");
  button.type = "submit";
  button.disabled = true;
  const note = textNode(doc, "p", "", "action-note");
  note.setAttribute("role", "status");
  form.append(label, button);
  card.append(form, note);
  container.append(card);
  refundAttached.add(container);
  let state = "loading";
  let frozenRequest = null;

  /** 显示真实待处理回执；参数为已校验回执；返回void；异常：无。 */
  function showReceipt(value) {
    if (value.action_id !== action.action_id || value.order_id !== action.order_id || value.status !== "待处理" || !/^RF[0-9A-F]{32}$/.test(value.refund_no)) {
      throw new Error("申请回执与原订单不一致。");
    }
    state = "completed";
    select.disabled = button.disabled = true;
    button.textContent = "申请已提交";
    note.textContent = `申请编号 ${value.refund_no} · 待处理`;
  }

  /** 原因仅从服务端读取，失败后允许重新读取；参数无；返回Promise；异常在本卡片显示。 */
  async function loadReasons() {
    try {
      const reasons = await getRefundReasons(fetchImpl);
      const placeholder = textNode(doc, "option", "请选择退款原因");
      placeholder.value = "";
      select.append(placeholder);
      for (const reason of reasons) {
        const option = textNode(doc, "option", reason.label);
        option.value = reason.code;
        select.append(option);
      }
      state = "ready";
      select.disabled = false;
      button.disabled = false;
      button.textContent = "确认提交申请";
    } catch (error) {
      state = "load_failed";
      button.disabled = false;
      button.textContent = "重新加载原因";
      note.textContent = error.message;
    }
  }
  if (receipt) showReceipt(receipt);
  else void loadReasons();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state === "load_failed") { state = "loading"; button.disabled = true; await loadReasons(); return; }
    if (!["ready", "unknown"].includes(state) || (!frozenRequest && !select.value)) return;
    const previouslyUnknown = state === "unknown";
    frozenRequest ??= Object.freeze({ ...identity, action_id: action.action_id, source_turn_id: action.source_turn_id,
      reason_code: select.value, confirmed: true });
    state = "submitting";
    select.disabled = button.disabled = true;
    button.textContent = "正在提交…";
    onBusy(true);
    try {
      showReceipt(await submitRefund(frozenRequest, fetchImpl));
    } catch (error) {
      let current = null;
      if (error.outcome === "unknown" || previouslyUnknown) {
        try { current = await getCurrentActions(identity, fetchImpl); } catch { /* 保留原键核实。 */ }
      }
      if (current?.refund_receipt?.action_id === action.action_id) showReceipt(current.refund_receipt);
      else {
        state = error.outcome === "unknown" || previouslyUnknown ? "unknown" : "ready";
        note.textContent = error.message;
        button.textContent = state === "unknown" ? "核实原申请" : "确认提交申请";
        if (state === "ready") frozenRequest = null;
      }
    } finally {
      button.disabled = state === "completed";
      select.disabled = state !== "ready";
      onBusy(false);
    }
  });
}

/**
 * 将已完成的合法建议绑定到原回复，快照不随新消息变化。
 * @param {HTMLElement} container 该条回复的按钮容器。
 * @param {object} metadata DONE后交付的完成元数据，身份来自原请求。
 * @param {object} options 网络、追加消息和建单忙碌通知接口。
 * @returns {void} 创建按钮和独立状态；无建议时不产生界面或请求。
 * @throws {Error} 非法动作或完成身份在创建任何节点前被拒绝。
 */
export function attachSuggestedActions(container, metadata, { fetchImpl = globalThis.fetch, appendMessage, onBusy = () => {}, receipt = null } = {}) {
  const suggested = validateSuggestedActions(metadata);
  if (attached.has(container)) { attached.get(container)(receipt); return; }
  if (!suggested.length) return;
  // 与确认客户端及后端统一按Unicode码点计数，避免将emoji代理对算成两个字符。
  if (metadata.status !== "completed" || typeof metadata.session_id !== "string" || !metadata.session_id ||
      Array.from(metadata.session_id).length > 64 || (metadata.user_id !== undefined &&
        (typeof metadata.user_id !== "string" || !metadata.user_id || Array.from(metadata.user_id).length > 64))) {
    throw new Error("动作建议缺少已完成的来源身份。");
  }
  const doc = container.ownerDocument;
  const identity = { session_id: metadata.session_id, user_id: metadata.user_id ?? "demo-user" };
  const row = node("div", "", "suggested-actions");
  row.setAttribute("aria-label", "本条回复的建议操作");
  let updateReceipt = () => {};

  /** 创建安全文本节点；参数为标签/文字/类名；返回DOM元素；异常：DOM创建失败。 */
  function node(tag, text = "", className = "") {
    const element = doc.createElement(tag);
    element.textContent = text;
    element.className = className;
    return element;
  }

  for (const action of suggested) {
    const item = node("div", "", "suggested-action");
    const button = node("button", action.kind === "handoff" ? "转人工客服" : "创建工单", "button button-secondary action-button");
    button.type = "button";
    item.append(button);
    row.append(item);
    if (action.kind === "handoff") {
      let greeted = false;
      button.addEventListener("click", () => {
        if (greeted) return;
        greeted = true;
        button.disabled = true;
        button.textContent = "已转接人工客服";
        // 此按钮只有本地显示；不修改会话身份/状态，也不访问建单或聊天接口。
        appendMessage("assistant", "已转接人工客服");
        appendMessage("assistant", "您好，我是客服小猫，请问有什么可以帮您的");
      });
      continue;
    }

    const note = node("p", "", "action-note");
    note.hidden = true;
    note.setAttribute("role", "status");
    item.append(note);
    const payload = Object.freeze({ ...identity, source_turn_id: action.source_turn_id, action_id: action.action_id, confirmed: true });
    let dialog = null;
    let state = "ready";

    /** 回读或确认后更新同一原动作；参数已保存回执；返回void；异常无。 */
    function showReceipt(saved) {
      if (!saved || saved.action_id !== action.action_id || typeof saved.ticket_no !== "string"
          || !saved.ticket_no.trim() || [...saved.ticket_no].length > 32 || saved.ticket_type !== action.ticket_type
          || saved.status !== "待处理" || (saved.audit_sync_pending !== undefined && typeof saved.audit_sync_pending !== "boolean")) return;
      state = "succeeded"; button.disabled = true; button.textContent = "工单已创建";
      note.hidden = false;
      note.textContent = `工单 ${saved.ticket_no} · ${saved.ticket_type} · ${saved.status}`;
      if (saved.audit_sync_pending) note.textContent += "。工单已创建，记录同步待恢复，无需重复建单。";
    }
    // 重复核实仅更新工单回执，不重建按钮或重置已经显示的人工接入状态。
    updateReceipt = showReceipt;
    showReceipt(receipt);

    button.addEventListener("click", () => {
      if (dialog || !["ready", "unknown"].includes(state)) return;
      dialog = node("dialog", "", "citation-dialog ticket-dialog");
      const panel = dialog;
      const form = node("form", "", "ticket-form");
      const header = node("div", "", "citation-header");
      const title = node("h2", state === "unknown" ? "核实建单结果" : "确认创建工单");
      title.id = `ticket-title-${action.action_id}`;
      panel.setAttribute("aria-labelledby", title.id);
      header.append(title);
      const content = node("div", "", "citation-body");
      const original = node("p", action.question, "ticket-question");
      const tooLong = Array.from(action.question).length > 2000;
      const explanation = node("p", tooLong ? "原问题超过2000字，请缩短问题后重新发送，再使用新回复中的建议。" :
        state === "unknown" ? "将使用同一条原请求核实已有结果，请勿另建工单。" : "确认后将按这条原问题创建工单。", "citation-note");
      content.append(node("p", `工单类型：${action.ticket_type}`, "citation-section"), node("p", "原问题", "ticket-label"), original, explanation);
      const controls = node("div", "", "ticket-controls");
      const cancel = node("button", "暂不创建", "button button-secondary action-button");
      cancel.type = "button";
      cancel.autofocus = true;
      cancel.addEventListener("click", () => panel.close());
      const submit = node("button", state === "unknown" ? "用原请求核实" : "确认建单", "button button-primary action-button");
      submit.type = "submit";
      submit.disabled = tooLong;
      controls.append(cancel, submit);
      form.append(header, content, controls);
      panel.append(form);
      // 原生Esc默认关闭；close统一移除临时弹窗并恢复原建议按钮焦点。
      panel.addEventListener("close", () => { panel.remove(); dialog = null; button.focus({ preventScroll: true }); }, { once: true });
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (tooLong || !panel.open || !["ready", "unknown"].includes(state)) return;
        const wasUnknown = state === "unknown";
        state = "submitting";
        submit.disabled = true;
        button.disabled = true;
        button.textContent = "正在确认…";
        button.setAttribute("aria-busy", "true");
        panel.close();
        note.hidden = false;
        note.textContent = "正在核实建单结果…";
        onBusy(true);
        try {
          const receipt = await confirmTicket(payload, { fetchImpl });
          showReceipt(receipt);
        } catch (error) {
          if (error.outcome === "failed") {
            state = "failed";
            button.textContent = "建单未成功";
            note.textContent = `${error.message}。这条建议已确定未成功，请联系人工客服。`;
          } else if (error.outcome === "unknown" || wasUnknown) {
            // 先前未知的结果不会被后一次409/422覆盖；始终用原键手动核实。
            state = "unknown";
            button.textContent = "核实建单结果";
            note.textContent = "建单结果暂未确认。请先核实；如需重试，请使用本条建议的原请求，避免重复建单。";
          } else {
            state = error.status === 409 ? "ready" : "failed";
            button.textContent = state === "ready" ? "创建工单" : "建单未开始";
            note.textContent = error.message;
          }
        } finally {
          button.disabled = !["ready", "unknown"].includes(state);
          button.setAttribute("aria-busy", "false");
          onBusy(false);
        }
      });
      doc.body.append(panel);
      panel.showModal();
    });
  }
  container.append(row);
  attached.set(container, updateReceipt);
}
