/** CH07浏览器验收驱动：经已配置web-access CDP操作真实页面；不直接调用聊天API。 */
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { setTimeout as pause } from "node:timers/promises";

const [directory, target] = process.argv.slice(2);
if (!directory || !target) throw new Error("需要报告目录与本次新建浏览器targetId");
const reportDir = resolve(directory), ready = JSON.parse(await readFile(resolve(reportDir,"browser-ready.json"),"utf8"));
const proxy="http://localhost:3456", checks={}, snapshots=[];

/** 执行浏览器表达式；参数JS字符串；返回JSON值；异常：CDP或页面错误。 */
async function evaluate(code) {
  const response=await fetch(`${proxy}/eval?target=${target}`,{method:"POST",body:code});
  const data=await response.json(); if(data.error)throw new Error(data.error); return data.value;
}
/** 读取界面状态；参数无；返回只读快照；异常：CDP错误。 */
async function state() {
  return evaluate(`({session:sessionStorage.getItem("mewhelp.session"),ready:!!document.querySelector("#message-input"),
    busy:document.querySelector("#message-input")?.readOnly,newDisabled:document.querySelector("#new-chat")?.disabled,
    sidebarDisabled:[...document.querySelectorAll(".conversation-item")].every(b=>b.disabled),
    users:[...document.querySelectorAll(".message--user .message-content")].map(n=>n.textContent),
    answer:[...document.querySelectorAll(".message--assistant .message-content")].at(-1)?.textContent,
    status:document.querySelector("#request-status")?.textContent,feedback:document.querySelector("#feedback")?.hidden,
    sidebar:document.querySelector("#conversation-list")?.textContent,probe:window.__ch07Probe})`);
}
/** 有界等待浏览器条件；参数谓词与秒数；返回满足条件快照；异常：超时或取证失败。 */
async function until(predicate, seconds=240) {
  const end=Date.now()+seconds*1000;
  while(Date.now()<end) {
    const value=await state();
    if(value?.probe?.errors?.length) throw new Error(value.probe.errors.join(";"));
    if(predicate(value))return value;
    await pause(300);
  }
  throw new Error("浏览器等待超时");
}
/** 通过真实输入/提交事件发送单轮；参数原问；返回页面完成状态；异常：接口/取证/超时。 */
async function send(query) {
  const before=await (await fetch(ready.url+"/__live__/state")).json();
  await until(s=>s?.ready&&!s.busy);
  await evaluate(`(()=>{const input=document.querySelector("#message-input");input.value=${JSON.stringify(query)};
    input.dispatchEvent(new Event("input",{bubbles:true}));document.querySelector("#chat-form").requestSubmit();return true;})()`);
  const busy=await state(); checks.reply_blocks_switch=(checks.reply_blocks_switch??true)&&busy.newDisabled&&busy.sidebarDisabled;
  const done=await until(s=>s?.probe?.captured>before.captured&&!s.busy);
  snapshots.push({query,...done});
  console.log(JSON.stringify({captured:done.probe.captured,status:done.status,answerChars:done.answer?.length??0}));
  return done;
}
/** 等待后台摘要明确结束；参数无；返回void；异常：超时。 */
async function summariesIdle() {
  const end=Date.now()+125000;
  while(Date.now()<end) { const status=await(await fetch(ready.url+"/__live__/state")).json();if(!status.active_summaries)return;await pause(400); }
  throw new Error("后台摘要未在期限内完成");
}

let error=null;
try {
  await until(s=>s?.ready&&s.session===ready.session_id&&!s.busy);
  for(let i=0;i<ready.queries.length;i++) {
    if(i===ready.queries.length-1)await summariesIdle();
    await send(ready.queries[i]);
  }
  checks.twenty_one_frontend_turns=(await state()).users.length===ready.queries.length;
  await evaluate('document.querySelector("#new-chat").click()');
  const second=await state();checks.new_clears_only_page=second.users.length===0&&second.session!==ready.session_id;
  await send("这是另一个会话，请介绍一下你能帮我处理哪些商城问题。");
  await evaluate('document.querySelector("#new-chat").click()');
  const third=await state();checks.three_distinct_sessions=third.session!==second.session&&third.session!==ready.session_id;
  await send("这个会话我想先了解猫粮应该如何保存。");
  await evaluate('document.querySelector("#refresh-conversations").click()');
  await until(s=>s.sidebar?.includes(ready.queries[0]));
  await evaluate(`(()=>{const panel=document.querySelector("#history-disclosure");if(!panel.open)panel.querySelector("summary").click();
    const button=[...document.querySelectorAll(".conversation-item")].find(b=>b.textContent.includes(${JSON.stringify(ready.queries[0])}));button.click();return true;})()`);
  let restored=await until(s=>s.session===ready.session_id&&!s.busy&&s.users.length===ready.queries.length);
  checks.full_original_history=JSON.stringify(restored.users)===JSON.stringify(ready.queries);
  await fetch(`${proxy}/navigate?target=${target}`,{method:"POST",body:ready.url+"/"});
  restored=await until(s=>s.session===ready.session_id&&!s.busy&&s.users.length===ready.queries.length);
  checks.refresh_restores_history=JSON.stringify(restored.users)===JSON.stringify(ready.queries);
  await send("继续聊一下，查订单1001现在的运输进度。");
  checks.continues_old_session=(await state()).session===ready.session_id;
  await summariesIdle();
  await until(s=>!s.busy);
  await evaluate('window.__ch07Probe.failNextList=true;document.querySelector("#refresh-conversations").click()');
  await pause(500);
  const failure=await state();checks.sidebar_failure_keeps_chat=failure.users.length===ready.queries.length+1&&!failure.busy&&failure.session===ready.session_id;
  await send("谢谢，我暂时没有其他问题。");
  checks.chat_after_sidebar_failure=(await state()).users.length===ready.queries.length+2;
  await evaluate('document.querySelector("#refresh-conversations").click();const panel=document.querySelector("#history-disclosure");if(!panel.open)panel.querySelector("summary").click()');
  await pause(500);
  const file=resolve(reportDir,"browser.png");
  const screenshot=await fetch(`${proxy}/screenshot?target=${target}&file=${encodeURIComponent(file)}`,{signal:AbortSignal.timeout(40000)});
  const result=await screenshot.json();
  if(!screenshot.ok || result.error) throw new Error(`截图失败：${result.error??screenshot.status}`);
  const bytes=await readFile(file);
  if(bytes.length<8 || bytes.subarray(0,8).toString("hex")!=="89504e470d0a1a0a")throw new Error("截图文件不是有效PNG");
  console.log(JSON.stringify({screenshot:file,checks}));
} catch(caught) {
  error=String(caught); checks.driver_completed=false;
  console.log(JSON.stringify({error,checks}));
} finally {
  await writeFile(resolve(reportDir,"browser-snapshots.json"),JSON.stringify({checks,snapshots,error},null,2),"utf8");
  await fetch(ready.url+"/__live__/finish",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({checks,error,target})});
}
process.exitCode=Object.values(checks).every(Boolean)?0:2;
