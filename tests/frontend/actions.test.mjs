/** 建议动作的实际事件测试：只替换浏览器DOM与网络边界，不替换生产点击/提交处理器。 */
import test from "node:test";
import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import * as api from "../../app/static/chat-api.mjs";
import * as actions from "../../app/static/actions.mjs";

/** 最小DOM边界；保留真实EventTarget分发，浏览器布局和键盘另由真实页面验收。 */
class Element extends EventTarget {
  constructor(tag, doc) { super(); Object.assign(this, { tagName: tag, ownerDocument: doc, children: [], dataset: {}, disabled: false, open: false, attributes: {}, _text: "" }); }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  append(...nodes) { for (const node of nodes) { node.parentNode = this; this.children.push(node); } }
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name]; }
  focus() { this.ownerDocument.activeElement = this; }
  remove() { this.parentNode.children = this.parentNode.children.filter((node) => node !== this); }
  showModal() { this.open = true; }
  close() { this.open = false; this.dispatchEvent(new Event("close")); }
  requestSubmit() { this.dispatchEvent(new Event("submit", { cancelable: true })); }
  click() {
    if (this.disabled) return;
    this.focus();
    if (this.dispatchEvent(new Event("click", { cancelable: true })) && this.type === "submit") {
      let node = this.parentNode;
      while (node && node.tagName !== "form") node = node.parentNode;
      node?.requestSubmit();
    }
  }
}

const ticketId = "00000000-0000-5000-8000-000000000001";
const handoffId = "00000000-0000-5000-8000-000000000002";
const question = "  原订单破损 <img src=x>  ";
const receipt = { action_id: ticketId, ticket_no: "T-REAL-123", ticket_type: "投诉", status: "待处理" };
const body = { session_id: "old-session", user_id: "original-user", source_turn_id: "old-turn", action_id: ticketId, confirmed: true };

/** 参数：可选原问；返回：合法完成快照；异常：无。 */
function metadata(original = question) {
  return { session_id: "old-session", user_id: "original-user", turn_id: "old-turn", status: "completed", citations: [],
    suggested_actions: [
      { action_id: handoffId, kind: "handoff", source_turn_id: "old-turn", question: original, ticket_type: null },
      { action_id: ticketId, kind: "create_ticket", source_turn_id: "old-turn", question: original, ticket_type: "投诉" },
    ] };
}
/** 参数：根节点；返回：所有后代；异常：无。 */
function all(root) { return [root, ...root.children.flatMap(all)]; }
/** 参数：根节点/按钮文字；返回：匹配按钮；异常：断言错误。 */
function button(root, label) {
  const found = all(root).find((node) => node.tagName === "button" && node.textContent === label);
  assert.ok(found, `缺少按钮：${label}`);
  return found;
}
/** 参数：完成快照/响应函数；返回：DOM和可观察输出；异常：生产实现错误。 */
function setup(meta = metadata(), respond = async () => Response.json(receipt)) {
  const doc = { createElement(tag) { return new Element(tag, doc); } };
  doc.body = doc.createElement("body");
  const container = doc.createElement("section");
  doc.body.append(container);
  const calls = [], messages = [], busy = [];
  const options = { fetchImpl: async (...args) => { calls.push(args); return respond(...args); },
    appendMessage: (role, text) => { assert.equal(role, "assistant"); messages.push(text); }, onBusy: (value) => busy.push(value) };
  actions.attachSuggestedActions(container, meta, options);
  return { doc, container, calls, messages, busy, options };
}
/** 参数：夹具；返回：弹窗；异常：没有弹窗时断言失败。 */
function openTicket(fixture) {
  button(fixture.container, "创建工单").click();
  const dialog = all(fixture.doc.body).find((node) => node.tagName === "dialog");
  assert.ok(dialog?.open, "必须通过实际点击打开确认弹窗");
  return dialog;
}

test("投诉回复装配两个按钮但不主动联网或打开弹窗", () => {
  const f = setup();
  assert.equal(all(f.container).filter((node) => node.tagName === "button").length, 2);
  assert.equal(f.calls.length, 0);
  assert.equal(all(f.doc.body).some((node) => node.open), false);
});

test("转人工零网络且只显示两句固定话，另一个按钮继续可用", () => {
  const f = setup();
  const handoff = button(f.container, "转人工客服");
  handoff.click(); handoff.click();
  assert.deepEqual(f.messages, ["已转接人工客服", "您好，我是客服小猫，请问有什么可以帮您的"]);
  assert.equal(f.calls.length, 0);
  assert.equal(button(f.container, "创建工单").disabled, false);
  openTicket(f);
});

test("取消确认不联网并恢复原按钮焦点，重新点击仍可打开", () => {
  const f = setup();
  const trigger = button(f.container, "创建工单");
  const dialog = openTicket(f);
  assert.ok(dialog.textContent.includes(question));
  assert.ok(dialog.textContent.includes("投诉"));
  button(dialog, "暂不创建").click();
  assert.equal(f.calls.length, 0);
  assert.equal(f.doc.activeElement, trigger);
  openTicket(f);
});

test("真实submit确认后只POST原动作，连击不重复提交且人工按钮不被锁", async () => {
  let resolve;
  const f = setup(metadata(), () => new Promise((done) => { resolve = done; }));
  const trigger = button(f.container, "创建工单");
  const dialog = openTicket(f);
  const submit = button(dialog, "确认建单");
  submit.click(); submit.click(); trigger.click();
  assert.equal(f.calls.length, 1);
  assert.equal(f.calls[0][0], "/api/actions/tickets");
  assert.deepEqual(JSON.parse(f.calls[0][1].body), body);
  assert.equal(button(f.container, "转人工客服").disabled, false);
  button(f.container, "转人工客服").click();
  assert.equal(f.messages.length, 2);
  resolve(Response.json(receipt));
  await setImmediate();
  assert.match(f.container.textContent, /T-REAL-123.*待处理/);
  assert.deepEqual(f.busy, [true, false]);
  trigger.click();
  assert.equal(f.calls.length, 1);
});

test("外部元数据改变不重绑旧建议的身份、原问题或动作键", async () => {
  const meta = metadata();
  const f = setup(meta);
  meta.session_id = "new-session"; meta.user_id = "new-user"; meta.turn_id = "new-turn";
  Object.assign(meta.suggested_actions[1], { question: "新问题", action_id: handoffId });
  const dialog = openTicket(f);
  assert.ok(dialog.textContent.includes(question));
  button(dialog, "确认建单").click();
  await setImmediate();
  assert.deepEqual(JSON.parse(f.calls[0][1].body), body);
});

for (const [length, permitted] of [[2000, true], [2001, false], [4000, false]]) {
  test(`原问题${length}个Unicode字符${permitted ? "可" : "不可"}确认，不裁剪`, async () => {
    const raw = " " + "🐱".repeat(length - 2) + " ";
    const f = setup(metadata(raw));
    const dialog = openTicket(f);
    const submit = button(dialog, "确认建单");
    assert.ok(dialog.textContent.includes(raw));
    assert.equal(submit.disabled, !permitted);
    submit.click();
    await setImmediate();
    assert.equal(f.calls.length, permitted ? 1 : 0);
    if (!permitted) assert.match(dialog.textContent, /2000.*缩短|缩短.*2000/);
  });
}

test("未知结果手动重试只用原键，审计待同步仍保留成功票号", async () => {
  let count = 0;
  const f = setup(metadata(), async () => ++count === 1
    ? Response.json({ detail: { code: "TICKET_OUTCOME_UNKNOWN", message: "结果未知", outcome: "unknown" } }, { status: 503 })
    : Response.json({ ...receipt, audit_sync_pending: true }));
  button(openTicket(f), "确认建单").click();
  await setImmediate();
  assert.match(f.container.textContent, /核实/);
  assert.equal(f.calls.length, 1);
  button(f.container, "核实建单结果").click();
  const dialog = all(f.doc.body).find((node) => node.tagName === "dialog");
  button(dialog, "用原请求核实").click();
  await setImmediate();
  assert.deepEqual(f.calls.map((call) => JSON.parse(call[1].body)), [body, body]);
  assert.match(f.container.textContent, /T-REAL-123/);
  assert.match(f.container.textContent, /工单已创建，记录同步待恢复，无需重复建单。/);
  assert.equal(all(f.container).find((node) => node.tagName === "button" && node.textContent === "工单已创建").disabled, true);
});

test("确定失败不换键重新执行，人工仍可使用", async () => {
  const f = setup(metadata(), async () => Response.json({ detail: { code: "TICKET_REJECTED", message: "建单被拒绝", outcome: "failed" } }, { status: 422 }));
  button(openTicket(f), "确认建单").click();
  await setImmediate();
  assert.match(f.container.textContent, /建单被拒绝/);
  assert.equal(button(f.container, "建单未成功").disabled, true);
  button(f.container, "转人工客服").click();
  assert.equal(f.messages.length, 2);
  assert.equal(f.calls.length, 1);
});

test("无建议旧元数据没有按钮，非法动作整体拒绝且不留下半成品按钮", () => {
  assert.equal(all(setup({ ...metadata(), suggested_actions: undefined }).container).length, 1);
  const meta = metadata(); meta.suggested_actions[1].source_turn_id = "different";
  const f = setup({ ...metadata(), suggested_actions: [] });
  assert.throws(() => actions.attachSuggestedActions(f.container, meta, f.options), /建议/);
  assert.equal(all(f.container).length, 1);
});

test("confirmTicket拒绝非true、额外业务字段和错误身份，不联网", async () => {
  assert.equal(typeof api.confirmTicket, "function", "需要独立确认API客户端");
  let calls = 0;
  for (const change of [{ confirmed: false }, { confirmed: 1 }, { confirmed: "true" }, { description: "替换原问" }, { session_id: "" }, { user_id: null }, { action_id: "bad" }]) {
    await assert.rejects(api.confirmTicket({ ...body, ...change }, { fetchImpl: async () => { calls++; return Response.json(receipt); } }), /参数|确认/);
  }
  assert.equal(calls, 0);
});

for (const [response, outcome] of [
  [() => Response.json({ detail: [{ msg: "invalid" }] }, { status: 422 }), "not_started"],
  [() => Response.json({ detail: { code: "CONVERSATION_BUSY", message: "会话忙", outcome: "not_started" } }, { status: 409 }), "not_started"],
  [() => Response.json({ ...receipt, action_id: handoffId }), "unknown"],
  [() => new Response("<html>error</html>", { status: 502 }), "unknown"],
  [() => { throw new TypeError("network lost"); }, "unknown"],
]) {
  test(`确认API错误归类${outcome}且不冒充成功：${response}`, async () => {
    assert.equal(typeof api.confirmTicket, "function");
    await assert.rejects(api.confirmTicket(body, { fetchImpl: async () => response() }), (error) => {
      assert.equal(error.outcome, outcome); return true;
    });
  });
}
