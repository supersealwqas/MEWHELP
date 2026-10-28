/** CH07会话浏览控制器：用真实异步响应检查分页、切换代次、只读及操作互斥。 */
import test from "node:test";
import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";

/** 延迟导入以观察缺实现红测；参数无；返回模块；异常：模块缺失。 */
async function subject() { return import("../../app/static/conversations.mjs"); }
const old = { id: "9007199254740993", session_id: "web-old", preview: "旧问题", has_summary: true };
const newer = { id: "9007199254740999", session_id: "web-new", preview: "新问题", has_summary: false };
/** 构造原文页；参数为会话、记录、游标；返回JSON响应；异常无。 */
function page(conversation, items, next = null) { return Response.json({ session_id:conversation.session_id, items, next_after_id:next }); }
/** 原文消息；参数为ID、正文及元数据；返回消息；异常无。 */
function message(id, content="完整原文\n  ", metadata=null) { return { id, role:"assistant", content, turn_id:"turn-a", turn_status:"completed", rag_metadata:metadata }; }

test("按字符串BIGINT分页取完整历史，新会话保留列表且没有写请求", async () => {
  const { createConversationBrowser } = await subject();
  const requests=[], selected=[];
  const browser=createConversationBrowser({fetch:async(url, options)=>{
    requests.push([url,options]); const parsed=new URL(url,"http://local");
    if(parsed.pathname==="/api/conversations") return Response.json({items:parsed.searchParams.has("before_id")?[old]:[newer],next_before_id:parsed.searchParams.has("before_id")?null:newer.id});
    return parsed.searchParams.get("after_id")==="9007199254740994" ? page(old,[message("9007199254740995")]) : page(old,[message("9007199254740994")],"9007199254740994");
  },onSelect:result=>selected.push(result)});
  assert.equal(await browser.refresh(),true); assert.equal(await browser.more(),true);
  assert.equal(await browser.select(old),true);
  assert.deepEqual(selected[0].messages.map(m=>m.id),["9007199254740994","9007199254740995"]);
  assert.equal(selected[0].messages[0].content,"完整原文\n  ");
  assert.equal(browser.startNew("web-empty"),true);
  assert.equal(browser.snapshot().items.length,2);
  assert.ok(requests.every(([, options])=>!options?.method || options.method==="GET"));
});

test("较早历史响应晚到不能覆盖后来选择或新会话", async () => {
  const { createConversationBrowser } = await subject();
  let release; const selected=[];
  const browser=createConversationBrowser({fetch:url=>url.includes(old.id)?new Promise(r=>release=r):Promise.resolve(page(newer,[message("9")])),onSelect:r=>selected.push(r.conversation.session_id)});
  const slow=browser.select(old); await setImmediate();
  assert.equal(await browser.select(newer),true);
  release(page(old,[message("1")])); assert.equal(await slow,false);
  assert.deepEqual(selected,["web-new"]);
  const slowAgain=browser.select(old); await setImmediate(); browser.startNew("web-empty");
  release(page(old,[message("1")])); assert.equal(await slowAgain,false);
  assert.equal(browser.snapshot().sessionId,"web-empty");
});

test("列表失败静默保留旧列表，历史中途失败保留原会话", async () => {
  const { createConversationBrowser } = await subject(); let fail=false; const selected=[];
  const browser=createConversationBrowser({fetch:async url=>{
    if(fail) throw new Error("offline");
    return url.includes("/messages")?page(old,[message("1")],"1"):Response.json({items:[old],next_before_id:null});
  },onSelect:r=>selected.push(r)});
  await browser.refresh(); browser.startNew("keep-me"); fail=true;
  assert.equal(await browser.refresh(),false); assert.equal(browser.snapshot().items.length,1);
  assert.equal(await browser.select(old),false); assert.equal(browser.snapshot().sessionId,"keep-me"); assert.equal(selected.length,0);
});

test("回复与动作互斥阻止切换，新发送使旧回载失效", async () => {
  const { createConversationBrowser } = await subject(); let calls=0, release; const selected=[];
  const browser=createConversationBrowser({fetch:()=>{calls++;return new Promise(r=>release=r);},onSelect:r=>selected.push(r)});
  browser.setBusy(true); assert.equal(await browser.select(old),false); assert.equal(browser.startNew("other"),false); assert.equal(calls,0);
  browser.setBusy(false); const slow=browser.select(old); await setImmediate(); browser.setBusy(true);
  release(page(old,[message("1")])); assert.equal(await slow,false); assert.equal(selected.length,0);
});

test("坏页、循环游标及不匹配会话不清空当前页面", async () => {
  const { createConversationBrowser } = await subject();
  for(const response of [()=>page(newer,[message("1")]),()=>page(old,[message(9007199254740993)]),()=>page(old,[message("1")],"1")]) {
    const selected=[]; const browser=createConversationBrowser({fetch:async()=>response(),onSelect:r=>selected.push(r)});
    browser.startNew("keep-me"); assert.equal(await browser.select(old),false); assert.equal(selected.length,0); assert.equal(browser.snapshot().sessionId,"keep-me");
  }
});

test("刷新恢复可遍历第二页找到旧会话，旧退款元数据只作历史资料传递", async () => {
  const { createConversationBrowser } = await subject(); const selected=[];
  const metadata={refund_action:{action_id:"old"},citations:[{n:1,answer:"证据"}]};
  const browser=createConversationBrowser({fetch:async url=>url.includes("/messages")?page(old,[message("1","原文",metadata)]):Response.json({items:url.includes("before_id")?[old]:[newer],next_before_id:url.includes("before_id")?null:newer.id}),onSelect:r=>selected.push(r)});
  assert.equal(await browser.restore("web-old"),true);
  assert.deepEqual(selected[0].messages[0].rag_metadata,metadata);
});

test("恢复锁覆盖第一列表页、后续列表页及全部原文页",async()=>{
  const {createConversationBrowser}=await subject();const observed=[],selected=[];
  const browser=createConversationBrowser({fetch:async url=>{
    await setImmediate();observed.push(browser.snapshot().loading);
    if(url.includes("/messages")) return url.includes("after_id=1")?page(old,[message("2")]):page(old,[message("1")],"1");
    return Response.json({items:url.includes("before_id")?[old]:[newer],next_before_id:url.includes("before_id")?null:newer.id});
  },onSelect:value=>selected.push(value)});
  assert.equal(await browser.restore("web-old"),true);
  assert.deepEqual(observed,[true,true,true,true]);
  assert.deepEqual(selected[0].messages.map(row=>row.id),["1","2"]);
  assert.equal(browser.snapshot().loading,false);
});

test("恢复列表失败释放输入锁，保留当前会话且不提交空历史",async()=>{
  const {createConversationBrowser}=await subject();let release;const selected=[];
  const browser=createConversationBrowser({fetch:()=>new Promise(resolve=>release=resolve),onSelect:value=>selected.push(value)});
  browser.startNew("web-old");const restoring=browser.restore("web-old");
  try {assert.equal(browser.snapshot().loading,true);}
  finally {release(new Response("offline",{status:503}));await restoring;}
  assert.equal(browser.snapshot().loading,false);
  assert.equal(browser.snapshot().sessionId,"web-old");assert.deepEqual(selected,[]);
});

test("查找旧会话时新建可取消恢复，旧列表响应不回载原文",async()=>{
  const {createConversationBrowser}=await subject();let release;const selected=[],requests=[];
  const browser=createConversationBrowser({fetch:url=>{requests.push(url);return new Promise(resolve=>release=resolve);},onSelect:value=>selected.push(value)});
  const restoring=browser.restore("web-old");
  browser.startNew("web-empty");
  assert.equal(browser.snapshot().loading,false);
  release(Response.json({items:[old],next_before_id:null}));
  assert.equal(await restoring,false);
  assert.equal(browser.snapshot().sessionId,"web-empty");assert.deepEqual(selected,[]);
  assert.equal(requests.length,1);
});
