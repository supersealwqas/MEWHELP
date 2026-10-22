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

/**
 * 将聊天原文转换为显示用 HTML。
 * @param {string} text 当前累计的聊天原文，可为空或未闭合的流式片段。
 * @returns {string} 禁止原始 HTML 和危险协议的显示用 HTML；空输入返回空字符串。
 * @throws {TypeError} text 不是字符串时由 Markdown 解析器抛出。
 */
export function renderMarkdown(text) {
  return markdown.render(text);
}
