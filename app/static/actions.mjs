/** 完成回复的独立建议按钮：转人工只显示固定话术，建单须经原生弹窗明确确认。 */
import { confirmTicket, validateSuggestedActions } from "./chat-api.mjs";

const attached = new WeakSet();

/**
 * 将已完成的合法建议绑定到原回复，快照不随新消息变化。
 * @param {HTMLElement} container 该条回复的按钮容器。
 * @param {object} metadata DONE后交付的完成元数据，身份来自原请求。
 * @param {object} options 网络、追加消息和建单忙碌通知接口。
 * @returns {void} 创建按钮和独立状态；无建议时不产生界面或请求。
 * @throws {Error} 非法动作或完成身份在创建任何节点前被拒绝。
 */
export function attachSuggestedActions(container, metadata, { fetchImpl = globalThis.fetch, appendMessage, onBusy = () => {} } = {}) {
  const suggested = validateSuggestedActions(metadata);
  if (!suggested.length || attached.has(container)) return;
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
          state = "succeeded";
          button.textContent = "工单已创建";
          note.textContent = `工单 ${receipt.ticket_no} · ${receipt.ticket_type} · ${receipt.status}`;
          if (receipt.audit_sync_pending) note.textContent += "。工单已创建，记录同步待恢复，无需重复建单。";
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
  attached.add(container);
}
