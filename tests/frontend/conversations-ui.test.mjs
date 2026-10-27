/** 执行原聊天控制器并复用生产会话模块，检查回载后的动作与竞态；DOM/网络是测试边界。 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { setImmediate } from "node:timers/promises";
import { createConversationBrowser } from "../../app/static/conversations.mjs";

const script = readFileSync(new URL("../../app/static/chat.js", import.meta.url),"utf8").replace(/^import .*;\r?\n/gm, "");

/** 最小DOM模拟真实事件派发，布局由浏览器验收。 */
class Element extends EventTarget {
  constructor(tag="div") { super(); Object.assign(this,{tagName:tag,children:[],dataset:{},attributes:{},value:"",_text:"",disabled:false,hidden:false,className:""}); this.classList={add(){},remove(){}}; }
  set textContent(value) { this._text=String(value); this.children=[]; }
  get textContent() { return this._text + this.children.map(row=>row.textContent).join(""); }
  set innerHTML(value) { this.textContent=value; }
  get innerHTML() { return this._text; }
  append(...rows) { for(const row of rows) { row.parentElement=this; this.children.push(row); } }
  replaceChildren(...rows) { this._text="";this.children=[];this.append(...rows); }
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
function setup(respond=null) {
  const nodes=new Map(), mounted=[], citations=[], requests=[];
  const document=new EventTarget();
  document.querySelector=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  document.createElement=tag=>new Element(tag);
  const storage=new Map([["mewhelp.session","web-original"]]);
  const sandbox={document, window:{matchMedia:()=>({matches:false})}, crypto:{getRandomValues:array=>array.fill(1)},Uint8Array,URLSearchParams,AbortController,
    sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
    createConversationBrowser:options=>createConversationBrowser({...options,fetch:async(url,opts)=>{requests.push([url,opts]);return respond?respond(url):Response.json(url.includes("/messages")?{session_id:"web-original",items:records,next_after_id:null}:{items:[conversation],next_before_id:null});}}),
    renderMarkdown:value=>value,attachAnswerTools:(reply,answer,metadata)=>citations.push(metadata),
    attachSuggestedActions:(container,metadata,options)=>mounted.push({kind:"suggested",metadata,options}),
    attachRefundAction:(container,metadata,options)=>mounted.push({kind:"refund",metadata,options}),
    getCurrentActions:async identity=>({source_turn_id:"second",turn_status:"completed",answer:"新办理说明",
      completion_metadata:{turn_id:"second",citations:[{n:1}],refund_action:{action_id:"current"}},refund_receipt:{refund_no:"RF-SAVED"}}),
  };
  const context=vm.createContext(sandbox);
  vm.runInContext(script,context,{filename:"app/static/chat.js"});
  return {nodes,mounted,citations,requests,context,node:id=>document.querySelector(id)};
}

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
