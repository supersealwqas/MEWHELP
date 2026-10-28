/** 只在CH07独占测试页面加载：透明复制浏览器收到的SSE，原响应/业务逻辑不变。 */
(() => {
  const original = window.fetch.bind(window);
  window.__ch07Probe = { captured: 0, errors: [], failNextList: false };
  // 仅独占验收页面提供可点击故障开关，内置浏览器无需修改页面内部状态。
  function installFaultButton() {
    const button = document.createElement("button");
    button.textContent = "测试：下次侧栏加载失败";
    button.type = "button";
    button.addEventListener("click", () => {
      window.__ch07Probe.failNextList = true;
      button.textContent = "测试：已安排一次侧栏失败";
    });
    document.body.append(button);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", installFaultButton, { once: true });
  else installFaultButton();
  window.fetch = async (url, options) => {
    if (window.__ch07Probe.failNextList && String(url).startsWith("/api/conversations?")) {
      window.__ch07Probe.failNextList = false;
      throw new TypeError("本次侧栏故障注入");
    }
    const response = await original(url, options);
    if (String(url) === "/api/chat") {
      const request = JSON.parse(options.body);
      response.clone().text().then(async body => {
        const result = await original("/__live__/record", { method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
          session_id:request.session_id,message:request.message,response:body,http_status:response.status}) });
        if (!result.ok) throw new Error(`取证失败${result.status}`);
        window.__ch07Probe.captured = (await result.json()).captured;
      }).catch(error => window.__ch07Probe.errors.push(error.message));
    }
    return response;
  };
})();
