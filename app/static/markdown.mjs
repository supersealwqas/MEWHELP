/** 聊天 Markdown 渲染模块；集中管理正文的格式和安全边界。 */
import MarkdownIt from "./vendor/markdown-it.mjs";

// 禁止正文里的原始 HTML，并保留普通换行；不启用会输出额外 HTML 的高亮插件。
const markdown = new MarkdownIt({ html: false, breaks: true });
const defaultValidateLink = markdown.validateLink.bind(markdown);

/**
 * 沿用库的危险链接检查，再限制聊天中允许的链接协议。
 * @param {string} url 已由 Markdown 解析器解码、规范化的链接地址。
 * @returns {boolean} 仅网页、邮件和相对地址返回 true。
 * @throws {Error} 无主动抛出的业务异常。
 */
markdown.validateLink = function validateChatLink(url) {
  if (!defaultValidateLink(url)) return false;
  const destination = url.trim();
  // 首个路径段里的冒号视为协议分隔，连带拒绝夹入控制字符编码的伪造协议。
  return /^(?:https?:|mailto:)/i.test(destination) || !/^[^/?#]*:/.test(destination);
};

/**
 * 在新标签打开链接，避免覆盖当前聊天并阻止新页面操控来源页。
 * @param {object[]} tokens 解析器生成的当前行内标记序列。
 * @param {number} index 当前链接起始标记的下标。
 * @param {object} options Markdown 渲染配置。
 * @param {object} env 本次渲染环境，由解析器传入。
 * @param {object} renderer Markdown 渲染器，负责属性转义。
 * @returns {string} 已转义属性的链接起始 HTML。
 * @throws {Error} 无主动抛出的业务异常。
 */
markdown.renderer.rules.link_open = function renderLink(tokens, index, options, env, renderer) {
  tokens[index].attrSet("target", "_blank");
  tokens[index].attrSet("rel", "noopener noreferrer");
  return renderer.renderToken(tokens, index, options);
};

/**
 * 将图片替换为安全的替代文字，避免正文自动请求外部资源。
 * @param {object[]} tokens 解析器生成的当前行内标记序列。
 * @param {number} index 当前图片标记的下标。
 * @param {object} options Markdown 渲染配置。
 * @param {object} env 本次渲染环境，由解析器传入。
 * @param {object} renderer Markdown 渲染器，负责提取替代文字。
 * @returns {string} 已转义的纯文本，不包含图片元素或资源地址。
 * @throws {Error} 无主动抛出的业务异常。
 */
markdown.renderer.rules.image = function renderImageText(tokens, index, options, env, renderer) {
  const alt = renderer.renderInlineAsText(tokens[index].children, options, env);
  // 替代文字也来自不可信正文，必须转义后才可进入 innerHTML。
  return markdown.utils.escapeHtml(alt);
};

/**
 * 将表格对齐转换为静态 CSS 类，兼容页面禁止内联样式的安全策略。
 * @param {object[]} tokens 解析器生成的当前表格标记序列。
 * @param {number} index 当前表头或单元格起始标记的下标。
 * @param {object} options Markdown 渲染配置。
 * @param {object} env 本次渲染环境，由解析器传入。
 * @param {object} renderer Markdown 渲染器，负责属性转义。
 * @returns {string} 不含内联 style 的表头或单元格起始 HTML。
 * @throws {Error} 无主动抛出的业务异常。
 */
function renderTableCell(tokens, index, options, env, renderer) {
  const token = tokens[index];
  const styleIndex = token.attrIndex("style");
  if (styleIndex >= 0) {
    // 只接受解析器提供的三种标准对齐值，任何内联样式都会被删除。
    const alignment = /^text-align:(left|center|right)$/.exec(token.attrs[styleIndex][1]);
    token.attrs.splice(styleIndex, 1);
    if (alignment) token.attrJoin("class", `markdown-align-${alignment[1]}`);
  }
  return renderer.renderToken(tokens, index, options);
}

markdown.renderer.rules.th_open = renderTableCell;
markdown.renderer.rules.td_open = renderTableCell;

// 只改行内普通文本；代码、图片和已有链接中的数字保持原样。
markdown.core.ruler.after("inline", "answer_citations", function citationTokens(state) {
  const allowed = new Set(state.env.citations?.map((item) => item.n) || []);
  if (!allowed.size) return;
  for (const block of state.tokens) {
    if (block.type !== "inline" || !block.children) continue;
    const output = [];
    let linkDepth = 0;
    for (const token of block.children) {
      if (token.type === "link_open") linkDepth += 1;
      if (token.type === "link_close") linkDepth -= 1;
      if (token.type !== "text" || linkDepth) { output.push(token); continue; }
      let start = 0;
      for (const match of token.content.matchAll(/\[([1-9]\d*)\]/g)) {
        const number = Number(match[1]);
        if (!Number.isSafeInteger(number) || !allowed.has(number)) continue;
        const prefix = new state.Token("text", "", 0);
        prefix.content = token.content.slice(start, match.index);
        output.push(prefix);
        const citation = new state.Token("citation_ref", "button", 0);
        citation.content = String(number);
        output.push(citation);
        start = match.index + match[0].length;
      }
      const suffix = new state.Token("text", "", 0);
      suffix.content = token.content.slice(start);
      output.push(suffix);
    }
    block.children = output;
  }
});
markdown.renderer.rules.citation_ref = function renderCitation(tokens, index) {
  const number = tokens[index].content;
  return `<button type="button" class="citation-ref" data-citation="${number}" aria-label="查看引用 ${number}">[${number}]</button>`;
};

/**
 * 将聊天原文转换为显示用 HTML。
 * @param {string} text 当前累计的聊天原文，可为空或未闭合的流式片段。
 * @returns {string} 禁止原始 HTML 和危险协议的显示用 HTML；空输入返回空字符串。
 * @throws {TypeError} text 不是字符串时由 Markdown 解析器抛出。
 */
export function renderMarkdown(text, citations = []) {
  return markdown.render(text, { citations });
}
