/** 完成回答的引用阅读面板与原轮反馈；负面反馈以服务端成功回执为准。 */
import { renderMarkdown } from "./markdown.mjs";

/** 创建纯文本节点；参数为标签、文字与可选类名；返回元素，不解释HTML。 */
function element(tag, text = "", className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

/** 读取已归属校验的来源；参数为同源路径；返回JSON，HTTP错误抛出中文提示。 */
async function readCitation(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" });
  if (!response.ok) throw new Error(response.status === 404 ? "这条引用当前不可访问。" : "来源暂时无法读取，请稍后重试。");
  return response.json();
}

/** 打开原生可键盘关闭的面板；参数为快照路径、引用编号；失败时保留可关闭的错误面板。 */
async function openCitation(url, number) {
  const dialog = element("dialog", "", "citation-dialog");
  const header = element("div", "", "citation-header");
  const title = element("h2", `引用 [${number}]`);
  title.id = "citation-dialog-title";
  dialog.setAttribute("aria-labelledby", title.id);
  const close = element("button", "关闭", "button citation-close");
  close.type = "button";
  close.addEventListener("click", () => dialog.close());
  header.append(title, close);
  const content = element("div", "正在读取引用…", "citation-body");
  content.setAttribute("aria-live", "polite");
  dialog.append(header, content);
  dialog.addEventListener("close", () => dialog.remove(), { once: true });
  document.body.append(dialog);
  dialog.showModal();
  try {
    const citation = await readCitation(url);
    if (!dialog.open) return;
    const sourceType = citation.source_type === "published_qa" ? "已发布问答知识" : citation.source_type === "document" ? "知识文档" : "知识证据";
    const section = element("p", citation.section_path || sourceType, "citation-section");
    const original = element("div", "", "markdown-content citation-snapshot");
    original.innerHTML = renderMarkdown(citation.answer);
    const note = element("p", "这是回答生成时使用的完整知识片段。", "citation-note");
    content.replaceChildren(section, original, note);
    if (citation.source_type === "published_qa") {
      content.append(element("p", "展示已发布的知识正文，不公开原始私人对话。", "citation-note"));
    }
    if (citation.source_type !== "document") return;
    const full = element("button", "查看文档全文", "button citation-full");
    full.type = "button";
    content.append(full);
    full.addEventListener("click", async function showFullSource() {
      full.disabled = true;
      full.textContent = "正在读取全文…";
      try {
        const source = await readCitation(`${url}/source`);
        if (!dialog.open) return;
        const message = source.source_changed === true ? "原文已更新。上方仍是当时的引用，下方是当前文档。"
          : source.source_changed === false ? "文档与回答时核对的版本一致。"
          : "当时未记录可核对的文件版本，下方为当前文档，可能与上方引用不同。";
        const version = element("p", message, "citation-version");
        const heading = element("h3", source.title);
        const documentBody = element("pre", "", "citation-document");
        let target = null;
        source.content.split(/\r?\n/).forEach((line, index) => {
          const row = element("span", `${line}\n`);
          if (index + 1 === source.section_line) {
            row.className = "citation-target";
            row.tabIndex = -1;
            target = row;
          }
          documentBody.append(row);
        });
        full.replaceWith(version, heading, documentBody);
        if (target) { target.scrollIntoView({ block: "center" }); target.focus({ preventScroll: true }); }
        else version.append(" 未能唯一定位原章节，请阅读全文核对。");
      } catch (error) {
        full.disabled = false;
        full.textContent = "重新读取全文";
        note.textContent = error.message;
      }
    });
  } catch (error) {
    content.textContent = error.message;
  }
}

/** 仅在收到完成帧后启用引用与原轮反馈；参数为气泡、完整回答和已验证元数据。 */
export function attachAnswerTools(reply, answer, metadata) {
  reply.bubble.innerHTML = renderMarkdown(answer, metadata.citations);
  const prefix = `/api/chat/${encodeURIComponent(metadata.session_id)}/turns/${encodeURIComponent(metadata.turn_id)}/citations/`;
  reply.bubble.addEventListener("click", function showCitation(event) {
    const button = event.target.closest("button[data-citation]");
    if (!button || !reply.bubble.contains(button)) return;
    const n = Number(button.dataset.citation);
    if (metadata.citations.some((item) => item.n === n)) openCitation(`${prefix}${n}`, n);
  });
  const feedback = element("div", "", "answer-feedback");
  feedback.setAttribute("aria-label", "评价这条回答");
  const useful = element("button", "👍 有帮助", "feedback-choice");
  const unhelpful = element("button", "👎 没帮助", "feedback-choice");
  const status = element("span", "没解决的问题可提交给客服改进", "feedback-note");
  status.setAttribute("aria-live", "polite");
  // 新键绑定用户且只接受服务端回执；旧版的本地 down 不能冒充已入池。
  const key = `mewhelp:answer-feedback:v2:${metadata.user_id}:${metadata.session_id}:${metadata.turn_id}`;
  let choice = null;
  let pending = false;
  try {
    const stored = JSON.parse(localStorage.getItem(key));
    if (stored?.choice === "up" || (stored?.choice === "down" && typeof stored.pool_id === "string" && /^[1-9][0-9]*$/.test(stored.pool_id))) choice = stored.choice;
  } catch { /* 禁止本地存储时，后端仍提供原轮幂等保证。 */ }
  function paintChoice() {
    useful.disabled = pending || Boolean(choice);
    unhelpful.disabled = pending || choice === "down";
    useful.setAttribute("aria-pressed", String(choice === "up"));
    unhelpful.setAttribute("aria-pressed", String(choice === "down"));
    if (choice === "down") status.textContent = "已提交 · 我们会用这条反馈改进回答";
    else if (choice === "up") status.textContent = "谢谢反馈 · 已记在本浏览器";
  }
  useful.type = unhelpful.type = "button";
  useful.addEventListener("click", () => {
    if (pending || choice) return;
    choice = "up";
    try { localStorage.setItem(key, JSON.stringify({ choice })); } catch { /* 仍保留当前页面选择。 */ }
    paintChoice();
  });
  unhelpful.addEventListener("click", async () => {
    if (pending || choice === "down") return;
    pending = true;
    paintChoice();
    status.textContent = "正在提交…";
    try {
      const response = await fetch(`/api/chat/${encodeURIComponent(metadata.session_id)}/turns/${encodeURIComponent(metadata.turn_id)}/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: metadata.user_id, resolved: false }),
      });
      let receipt;
      try { receipt = await response.json(); }
      catch { throw new Error("提交暂未确认，请再试一次。"); }
      if (!response.ok || typeof receipt?.pool_id !== "string" || !/^[1-9][0-9]*$/.test(receipt.pool_id)) {
        throw new Error(response.status === 409 ? "这轮还未完整结束，请稍后重试。" : "提交暂未成功，请再试一次。");
      }
      choice = "down";
      try { localStorage.setItem(key, JSON.stringify({ choice, pool_id: receipt.pool_id })); }
      catch { /* 回执已成功；刷新后的重复请求由服务器去重。 */ }
    } catch (error) {
      status.textContent = error instanceof TypeError ? "网络暂时不可用，请重试。" : error.message;
    } finally {
      pending = false;
      const message = status.textContent;
      paintChoice();
      if (choice !== "down") status.textContent = message;
    }
  });
  feedback.append(useful, unhelpful, status);
  reply.bubble.parentElement.after(feedback);
  paintChoice();
}
