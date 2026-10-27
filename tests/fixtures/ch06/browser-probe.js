/** 仅本机CH06验收使用：记录真实网络；显式按钮控制晚到帧和未知事件，不替代任何模型。 */
(() => {
  const key = "ch06.browser-evidence";
  const data = JSON.parse(sessionStorage.getItem(key) || '{"requests":[],"responses":[],"checks":{},"errors":[]}');
  const original = window.fetch.bind(window);
  let delay = false;
  let unknown = false;
  let release = null;
  let status;
  const save = () => sessionStorage.setItem(key, JSON.stringify(data));
  window.addEventListener("error", (event) => { data.errors.push(event.message); save(); });
  window.fetch = async (url, init) => {
    const delayed = delay && url === "/api/chat";
    const addUnknown = unknown && url === "/api/chat";
    delay = delayed ? false : delay;
    unknown = addUnknown ? false : unknown;
    const item = {url:String(url), body:init?.body ?? null, delayed, addUnknown};
    data.requests.push(item);
    save();
    const response = await original(url, delayed ? {...init,signal:undefined} : init);
    if (!delayed && !addUnknown) {
      response.clone().text().then(body => { data.responses.push({url:String(url),status:response.status,body}); save(); });
      return response;
    }
    let body = await response.text();
    data.responses.push({url:String(url),status:response.status,body});
    save();
    if (delayed) {
      status.textContent = "旧选单帧已阻塞，可停止并发新问";
      await new Promise(resolve => { release = resolve; });
    }
    if (addUnknown) body = 'event: workflow_stage\ndata: {"stage":"future_stage","status":"running"}\n\nevent: future_event\ndata: {}\n\n' + body;
    return new Response(body,{status:response.status,headers:response.headers});
  };
  const bar = document.createElement("div");
  bar.setAttribute("aria-label","CH06验收工具");
  status = document.createElement("output");
  status.id = "qa-status";
  status.textContent = "记录真实网络";
  const control = (id,text,handler) => {
    const button=document.createElement("button");
    button.id=id; button.textContent=text; button.type="button";
    button.addEventListener("click",handler); bar.append(button);
  };
  control("qa-delay","阻塞下一条选单帧",()=>{delay=true;status.textContent="已准备阻塞下一次聊天响应";});
  control("qa-release","释放旧帧",()=>{release?.();release=null;status.textContent="旧帧已释放";});
  control("qa-unknown","下次插入未知事件",()=>{unknown=true;status.textContent="下一次聊天插入未知事件";});
  control("qa-save","保存验收证据",async()=>{
    const session_id=sessionStorage.getItem("mewhelp.session");
    const response=await original("/__live__/evidence",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...data,session_id,visible_text:document.body.innerText})});
    status.textContent=response.ok?"验收证据已保存":"证据保存失败";
  });
  bar.append(status);
  document.body.append(bar);
})();
