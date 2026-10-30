/** 执行原聊天控制器并复用生产会话模块，检查回载后的动作与竞态；DOM/网络是测试边界。 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { setImmediate } from "node:timers/promises";
import { createConversationBrowser } from "../../app/static/conversations.mjs";
import { attachOrderSelection, attachTicketPreview } from "../../app/static/actions.mjs";
import { failurePresentation, getCurrentActions, resumeOrderSelection, resumeTicketConfirmation } from "../../app/static/chat-api.mjs";

const script = readFileSync(new URL("../../app/static/chat.js", import.meta.url),"utf8").replace(/^import .*;\r?\n/gm, "");

/** 最小DOM模拟真实事件派发，布局由浏览器验收。 */
class Element extends EventTarget {
  constructor(tag="div",doc=null) { super(); Object.assign(this,{tagName:tag,ownerDocument:doc,children:[],dataset:{},attributes:{},value:"",_text:"",disabled:false,hidden:false,className:""}); this.classList={add(){},remove(){}}; }
  set textContent(value) { this._text=String(value); this.children=[]; }
  get textContent() { return this._text + this.children.map(row=>row.textContent).join(""); }
  set innerHTML(value) { this.textContent=value; }
  get innerHTML() { return this._text; }
  append(...rows) { for(const row of rows) { row.parentElement=this; this.children.push(row); } }
  replaceChildren(...rows) { this._text="";this.children=[];this.append(...rows); }
  remove() { if(this.parentElement)this.parentElement.children=this.parentElement.children.filter(row=>row!==this); }
  setAttribute(key,value) { this.attributes[key]=value; }
  setCustomValidity() {}
  focus() {}
  click() { if(!this.disabled)this.dispatchEvent(new Event("click")); }
}

const conversation={id:"1",session_id:"web-original",preview:"原问",has_summary:true};
const records=[{id:"1",role:"user",content:"第一条原问\n  ",turn_id:"first",turn_status:"completed"},
 {id:"2",role:"assistant",content:"旧退款建议",turn_id:"first",turn_status:"completed",rag_metadata:{citations:[],refund_action:{action_id:"obsolete"}}},
 {id:"3",role:"user",content:"第二条原问",turn_id:"second",turn_status:"completed"},
 {id:"4",role:"assistant",content:"新办理说明",turn_id:"second",turn_status:"completed",rag_metadata:{citations:[{n:1}],refund_action:{action_id:"current"}}}];

/** 执行生产页面脚本，替换外部网络及已有动作渲染边界；参数响应函数；返回观察记录；异常：脚本或断言失败。 */
function setup(respond=null, { storedSession="web-original", ticketReceipt=null, currentActions=null, workflowFetch=null } = {}) {
  const nodes=new Map(), mounted=[], citations=[], requests=[];
  const document=new EventTarget();
  document.querySelector=id=>{if(!nodes.has(id))nodes.set(id,new Element("div",document));return nodes.get(id);};
  document.createElement=tag=>new Element(tag,document);
  document.createElementNS=(_namespace,tag)=>document.createElement(tag);
  const storage=new Map(storedSession === null ? [] : [["mewhelp.session",storedSession]]);
  const sandbox={document, window:{matchMedia:query=>({matches:query.includes("prefers-reduced-motion")})}, crypto:{getRandomValues:array=>array.fill(1)},Uint8Array,URLSearchParams,AbortController,
    sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
    createConversationBrowser:options=>createConversationBrowser({...options,fetch:async(url,opts)=>{requests.push([url,opts]);return respond?respond(url):Response.json(url.includes("/messages")?{session_id:"web-original",items:records,next_after_id:null}:{items:[conversation],next_before_id:null});}}),
    renderMarkdown:value=>value,attachAnswerTools:(reply,answer,metadata)=>citations.push(metadata),
    attachSuggestedActions:(container,metadata,options)=>mounted.push({kind:"suggested",metadata,options}),
    attachRefundAction:(container,metadata,options)=>mounted.push({kind:"refund",metadata,options}),
    attachOrderSelection,attachTicketPreview,
    failurePresentation,
    resumeTicketConfirmation:(identity,action,decision,handlers)=>resumeTicketConfirmation(identity,action,decision,{...handlers,fetchImpl:workflowFetch}),
    resumeOrderSelection:(identity,action,orderId,handlers)=>resumeOrderSelection(identity,action,orderId,{...handlers,fetchImpl:workflowFetch}),
    getCurrentActions:async identity=>workflowFetch?getCurrentActions(identity,workflowFetch):currentActions??({source_turn_id:"second",turn_status:"completed",answer:"新办理说明",
      completion_metadata:{turn_id:"second",citations:[{n:1}],refund_action:{action_id:"current"}},refund_receipt:{refund_no:"RF-SAVED"},ticket_receipt:ticketReceipt}),
  };
  const context=vm.createContext(sandbox);
  vm.runInContext(script,context,{filename:"app/static/chat.js"});
  return {nodes,mounted,citations,requests,context,node:id=>document.querySelector(id)};
}

/** 构造完全本地的协议流；参数为事件/数据对；返回Response；异常：载荷不能序列化。 */
function workflowResponse(events) {
  const text=events.map(([event,data])=>`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("");
  return new Response(text,{headers:{"Content-Type":"text/event-stream"}});
}

/** 收集真实动作组件生成的元素；参数为根节点；返回元素数组；异常：无。 */
function descendants(root) { return [root,...root.children.flatMap(descendants)]; }

const confirmation={type:"confirm_ticket",action_id:"00000000-0000-5000-8000-000000000002",
  source_turn_id:"second",tool_call_id:"ticket-1",ticket_type:"售后",description:"商品破损"};
const selection={type:"select_order",action_id:"00000000-0000-5000-8000-000000000001",source_turn_id:"second",
  orders:[{order_id:"1001",product_name:"猫粮",status:"已发货"}]};
const savedReceipt={action_id:confirmation.action_id,ticket_no:"T-REAL-123",ticket_type:"售后",status:"待处理"};

test("确认建单成功后接续选单立即保留真实票号，刷新恢复一致",async()=>{
  let current={source_turn_id:"second",turn_status:"awaiting_input",resume_required:false,ticket_confirmation:confirmation,ticket_status:"awaiting_input"};
  const submitted=[];
  const workflowFetch=async(url,options)=>{
    if(url.startsWith("/api/actions/current?")) return Response.json(current);
    assert.equal(url,"/api/chat/resume");
    submitted.push(JSON.parse(options.body));
    current={...current,order_selection:selection,ticket_decision:"confirm",ticket_status:"completed",ticket_receipt:savedReceipt};
    return workflowResponse([
      ["tool_status",{tool_call_id:"ticket-1",name:"create_ticket",status:"succeeded",label:"提交完成",is_mock:false}],
      ["select_order",selection],
    ]);
  };
  const f=setup(null,{workflowFetch});await setImmediate();await setImmediate();
  descendants(f.node("#messages")).find(row=>row.tagName==="button"&&row.textContent==="确认提交").click();
  await setImmediate();await setImmediate();
  const assertReceiptAndSelection=page=>{
    assert.match(page.node("#messages").textContent,/T-REAL-123/);
    assert.ok(descendants(page.node("#messages")).some(row=>row.tagName==="button"&&!row.disabled&&row.textContent==="猫粮 · 订单 1001 · 已发货"));
    assert.equal(page.node("#feedback").hidden,true);
  };
  assertReceiptAndSelection(f);
  const refreshed=setup(null,{workflowFetch});await setImmediate();await setImmediate();
  assertReceiptAndSelection(refreshed);
  assert.deepEqual(submitted,[{type:"confirm_ticket",session_id:"web-original",user_id:"demo-user",source_turn_id:"second",
    action_id:confirmation.action_id,decision:"confirm"}]);
});

test("成功回执以accepted结束时核实原动作，页面不残留协议错误",async()=>{
  let current={source_turn_id:"second",turn_status:"awaiting_input",resume_required:false,ticket_confirmation:confirmation,ticket_status:"awaiting_input"};
  const workflowFetch=async(url,options)=>{
    if(url.startsWith("/api/actions/current?")) return Response.json(current);
    assert.equal(url,"/api/chat/resume");
    assert.equal(JSON.parse(options.body).action_id,confirmation.action_id);
    current={...current,turn_status:"running",ticket_decision:"confirm",ticket_status:"completed",ticket_receipt:savedReceipt,resume_required:false};
    return workflowResponse([
      ["message",{delta:"工单已创建，工单号：T-REAL-123。"}],
      ["action_status",{status:"accepted",action_id:confirmation.action_id,source_turn_id:"second"}],
    ]);
  };
  const f=setup(null,{workflowFetch});await setImmediate();await setImmediate();
  descendants(f.node("#messages")).find(row=>row.tagName==="button"&&row.textContent==="确认提交").click();
  await setImmediate();await setImmediate();
  assert.match(f.node("#messages").textContent,/T-REAL-123/);
  assert.equal(f.node("#feedback").hidden,true);
  assert.doesNotMatch(f.node("#request-status").textContent,/错误|不完整|未收到/);
  assert.equal(descendants(f.node("#messages")).filter(row=>row.tagName==="button"&&!row.disabled&&row.textContent==="确认提交").length,0);
  assert.equal(f.node("#message-input").readOnly,false);
});

test("真实回载保留原问与所有正文，只有当前动作带服务端回执恢复",async()=>{
  const f=setup();await setImmediate();await setImmediate();
  assert.match(f.node("#messages").textContent,/第一条原问\n  /);
  assert.match(f.node("#messages").textContent,/旧退款建议/);
  assert.match(f.node("#messages").textContent,/新办理说明/);
  assert.deepEqual(f.citations.map(row=>row.turn_id),["first","second"]);
  assert.equal(f.mounted.filter(row=>row.kind==="refund").length,1);
  assert.equal(f.mounted.find(row=>row.kind==="refund").options.receipt.refund_no,"RF-SAVED");
  assert.ok(f.mounted.every(row=>row.metadata.turn_id==="second"));
  f.mounted[0].options.onBusy(true);
  assert.equal(f.node("#new-chat").disabled,true);
  assert.equal(f.node("#conversation-list").children[0].children[0].disabled,true);
  f.mounted[0].options.onBusy(false);
  f.node("#new-chat").click();
  assert.equal(f.node("#messages").children.length,0);
  assert.equal(f.node("#conversation-list").children.length,1);
});

test("旧历史失败晚到不能给新会话写错误状态",async()=>{
  let reject;let delay=false;
  const f=setup(url=>url.includes("/messages")&&delay?new Promise((_,fail)=>reject=fail):Response.json(url.includes("/messages")?{session_id:"web-original",items:records,next_after_id:null}:{items:[conversation],next_before_id:null}));
  await setImmediate();await setImmediate();delay=true;
  f.node("#conversation-list").children[0].children[0].click();await setImmediate();
  f.node("#new-chat").click();reject(new Error("offline"));await setImmediate();
  assert.equal(f.node("#messages").children.length,0);
  assert.doesNotMatch(f.node("#history-status").textContent,/未能|失败|正在/);
});

test("刷新查找旧会话期间禁止发送，完整原文到齐后才解锁",async()=>{
  let release;
  const f=setup(url=>url.includes("/messages")?Response.json({session_id:"web-original",items:records,next_after_id:null}):new Promise(resolve=>release=resolve));
  await setImmediate();
  try {
    f.node("#message-input").value="恢复期间的新问题";
    f.node("#message-input").dispatchEvent(new Event("input"));
    assert.equal(f.node("#message-input").readOnly,true);
    assert.equal(f.node("#send-button").disabled,true);
    assert.match(f.node("#request-status").textContent,/恢复|回载/);
    f.node("#chat-form").dispatchEvent(new Event("submit",{cancelable:true}));
    assert.equal(f.node("#messages").children.length,0);
  } finally {
    release(Response.json({items:[conversation],next_before_id:null}));
    await setImmediate();await setImmediate();
  }
  assert.match(f.node("#messages").textContent,/第一条原问/);
  assert.match(f.node("#messages").textContent,/新办理说明/);
  assert.equal(f.node("#message-input").readOnly,false);
});

test("普通侧栏刷新等待不锁住已恢复会话的输入",async()=>{
  let release,delay=false;
  const f=setup(url=>delay?new Promise(resolve=>release=resolve):Response.json(url.includes("/messages")?{session_id:"web-original",items:records,next_after_id:null}:{items:[conversation],next_before_id:null}));
  await setImmediate();await setImmediate();delay=true;
  f.node("#refresh-conversations").click();await setImmediate();
  f.node("#message-input").value="继续提问";
  f.node("#message-input").dispatchEvent(new Event("input"));
  assert.equal(f.node("#message-input").readOnly,false);
  assert.equal(f.node("#send-button").disabled,false);
  release(Response.json({items:[conversation],next_before_id:null}));await setImmediate();
});

test("没有保存会话时直接允许输入，不等待历史列表",async()=>{
  let release;
  const f=setup(()=>new Promise(resolve=>release=resolve),{storedSession:null});
  await setImmediate();
  try {
    f.node("#message-input").value="首次提问";
    f.node("#message-input").dispatchEvent(new Event("input"));
    assert.equal(f.node("#message-input").readOnly,false);
    assert.equal(f.node("#send-button").disabled,false);
  } finally { release(Response.json({items:[],next_before_id:null}));await setImmediate(); }
});

test("刷新及再次核实状态都把服务端工单回执传给当前原动作",async()=>{
  const receipt={action_id:"saved-ticket",ticket_no:"T-SAVED",ticket_type:"售后",status:"待处理",audit_sync_pending:true};
  const f=setup(null,{ticketReceipt:receipt});await setImmediate();await setImmediate();
  assert.equal(f.mounted.find(row=>row.kind==="suggested").options.receipt,receipt);
  receipt.audit_sync_pending=false;
  f.node("#refresh-status").click();await setImmediate();
  const mounted=f.mounted.filter(row=>row.kind==="suggested");
  assert.equal(mounted.length,2);
  assert.equal(mounted[1].options.receipt.audit_sync_pending,false);
  assert.equal(f.node("#messages").children.length,4);
});

for (const accepted of [false,true]) {
  test(`工单完成后刷新恢复${accepted?"已确认订单的继续按钮":"当前选单"}并保留票号`,async()=>{
    const current={source_turn_id:"second",turn_status:accepted?"running":"awaiting_input",
      resume_required:accepted,accepted_order_id:accepted?"1001":null,
      order_selection:{type:"select_order",action_id:"00000000-0000-5000-8000-000000000001",source_turn_id:"second",
        orders:[{order_id:"1001",product_name:"猫粮",status:"已发货"}]},
      ticket_confirmation:{type:"confirm_ticket",action_id:"00000000-0000-5000-8000-000000000002",
        source_turn_id:"second",tool_call_id:"ticket-1",ticket_type:"售后",description:"商品破损"},
      ticket_decision:"confirm",ticket_status:"completed",
      ticket_receipt:{action_id:"00000000-0000-5000-8000-000000000002",ticket_no:"T-SAVED",ticket_type:"售后",status:"待处理"}};
    const f=setup(null,{currentActions:current});await setImmediate();await setImmediate();
    // 使用真实动作组件；错误分支会只展示工单卡，或把订单续跑误接到旧建单动作。
    const all=root=>[root,...root.children.flatMap(all)];
    for (let refresh=0;refresh<2;refresh++) {
      const buttons=all(f.node("#messages")).filter(row=>row.tagName==="button"&&!row.disabled);
      assert.ok(buttons.some(row=>row.textContent===(accepted?"继续处理":"猫粮 · 订单 1001 · 已发货")));
      assert.match(f.node("#messages").textContent,/T-SAVED/);
      assert.doesNotMatch(f.node("#messages").textContent,/工单预览/);
      assert.doesNotMatch(f.node("#messages").textContent,/继续原工单处理|确认并创建工单/);
      assert.equal(f.node("#feedback").hidden,true);
      f.node("#refresh-status").click();await setImmediate();
    }
  });
}
