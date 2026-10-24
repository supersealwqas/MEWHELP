/** Markdown 渲染离线测试：验证聊天格式、流式片段与不可信内容的安全边界。 */
import test from "node:test";
import assert from "node:assert/strict";
import { renderMarkdown } from "../../app/static/markdown.mjs";

/** 无参数；返回 void；标题或强调未转换为可显示的 HTML 时断言失败。 */
test("标题、加粗和斜体能显示为 Markdown 格式", function textFormatting() {
  const html = renderMarkdown("## 喂养建议\n\n**少量多餐**，并*逐步换粮*。");
  assert.match(html, /<h2>喂养建议<\/h2>/);
  assert.match(html, /<strong>少量多餐<\/strong>/);
  assert.match(html, /<em>逐步换粮<\/em>/);
});

/** 无参数；返回 void；列表层级或引用结构丢失时断言失败。 */
test("有序列表、无序列表和引用保留各自的结构", function listsAndQuote() {
  const html = renderMarkdown("1. 早餐\n2. 晚餐\n\n- 主粮\n  - 幼猫粮\n\n> 请准备饮水");
  assert.match(html, /<ol>[\s\S]*<li>早餐<\/li>[\s\S]*<li>晚餐<\/li>[\s\S]*<\/ol>/);
  assert.match(html, /<ul>[\s\S]*主粮[\s\S]*<ul>[\s\S]*幼猫粮[\s\S]*<\/ul>/);
  assert.match(html, /<blockquote>\n<p>请准备饮水<\/p>\n<\/blockquote>/);
});

/** 无参数；返回 void；表格被输出为管道字符或列内容错位时断言失败。 */
test("商品对比表格保留表头和单元格", function tableFormatting() {
  const html = renderMarkdown("| 商品 | 适用 |\n| --- | --- |\n| 幼猫粮 | 12 月龄前 |");
  assert.match(html, /<table>[\s\S]*<thead>[\s\S]*<th>商品<\/th>/);
  assert.match(html, /<tbody>[\s\S]*<td>幼猫粮<\/td>[\s\S]*<td>12 月龄前<\/td>/);
});

/** 无参数；返回 void；表格对齐依赖被页面安全规则禁止的内联样式时断言失败。 */
test("表格左右居中对齐使用 CSS 类并完全移除内联样式", function tableAlignmentClasses() {
  const html = renderMarkdown("| 左侧 | 居中 | 右侧 |\n| :--- | :---: | ---: |\n| 甲 | 乙 | 丙 |");
  assert.match(html, /<th class="markdown-align-left">左侧<\/th>/);
  assert.match(html, /<th class="markdown-align-center">居中<\/th>/);
  assert.match(html, /<th class="markdown-align-right">右侧<\/th>/);
  assert.match(html, /<td class="markdown-align-left">甲<\/td>/);
  assert.match(html, /<td class="markdown-align-center">乙<\/td>/);
  assert.match(html, /<td class="markdown-align-right">丙<\/td>/);
  assert.doesNotMatch(html, /\sstyle=/i);
});

/** 无参数；返回 void；代码被解释为标签或 Markdown 时断言失败。 */
test("围栏代码和行内代码中的标签及 Markdown 都是纯文本", function codeEscaping() {
  const html = renderMarkdown('`<img src=x onerror=alert(1)>`\n\n```html\n<script>alert(1)</script>\n**原样保留** & text\n```');
  assert.match(html, /<code>&lt;img src=x onerror=alert\(1\)&gt;<\/code>/);
  assert.match(html, /<pre><code class="language-html">&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.match(html, /\*\*原样保留\*\* &amp; text/);
  assert.doesNotMatch(html, /<(?:script|img|strong)\b/i);
});

/** 无参数；返回 void；用户输入的普通换行被浏览器折叠时断言失败。 */
test("段落内换行仍然可见", function lineBreaks() {
  assert.equal(renderMarkdown("第一行\n第二行"), "<p>第一行<br>\n第二行</p>\n");
});

/** 无参数；返回 void；原始 HTML、事件属性或实体能生成标签时断言失败。 */
test("原始 HTML 和编码后的标签只显示为文字", function rawHtmlEscaping() {
  const html = renderMarkdown('<script>alert(1)</script>\n<img src=x onerror=alert(1)>\n<svg onload=alert(1)>\n&lt;iframe src=x&gt;');
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(html, /&lt;iframe src=x&gt;/);
  assert.doesNotMatch(html, /<(?:script|img|svg|iframe)\b/i);
});

// 协议白名单既限制危险脚本，也避免聊天正文启动本地应用或内嵌图片资源。
for (const destination of [
  "javascript:alert(1)", "JaVaScRiPt:alert(1)", "javascript&#58;alert(1)",
  "java&#x09;script:alert(1)", "vbscript:alert(1)", "file:///C:/secret.txt",
  "data:text/html;base64,PHNjcmlwdD4=", "data:image/png;base64,aGVsbG8=", "vscode:open",
]) {
  /** 无参数；返回 void；危险或未允许的协议成为可点击链接时断言失败。 */
  test(`不生成危险或自定义协议的链接：${destination}`, function dangerousLink() {
    assert.doesNotMatch(renderMarkdown(`[查看](${destination})`), /<a\b/i);
  });
}

/** 无参数；返回 void；正常链接丢失，或点击链接会替换聊天页时断言失败。 */
test("网页、邮件和相对链接在新标签打开并隔离来源页面", function allowedLinks() {
  const html = renderMarkdown("[网页](https://example.com/help) [本地](/help) [邮件](mailto:help@example.com) [目录](./guide) [锚点](#notice)");
  const anchors = html.match(/<a\b[^>]*>/g) ?? [];
  assert.equal(anchors.length, 5);
  assert.match(html, /href="https:\/\/example\.com\/help"/);
  assert.match(html, /href="mailto:help@example\.com"/);
  assert.match(html, /href="\/help"/);
  for (const anchor of anchors) {
    assert.match(anchor, /target="_blank"/);
    assert.match(anchor, /rel="noopener noreferrer"/);
  }
});

/** 无参数；返回 void；链接标题中的引号产生额外属性时断言失败。 */
test("链接标题的引号不能逃逸到 HTML 属性", function linkAttributeEscaping() {
  const html = renderMarkdown('[安全](https://example.com "&quot; onmouseover=&quot;alert(1)")');
  assert.match(html, /title="&quot; onmouseover=&quot;alert\(1\)"/);
  assert.doesNotMatch(html, /" onmouseover="/i);
});

/** 无参数；返回 void；图片生成可加载资源的元素或未转义替代文字时断言失败。 */
test("Markdown 图片仅保留安全替代文字，不自动加载远程图片", function imageAsText() {
  const html = renderMarkdown('![猫咪](https://example.com/tracker.png)\n![<img src=x onerror=alert(1)>](https://example.com/other.png)');
  assert.match(html, /猫咪/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.doesNotMatch(html, /<(?:img|iframe|video|audio|source|object|embed)\b/i);
  assert.doesNotMatch(html, /https:\/\/example\.com/);
});

/** 无参数；返回 void；未闭合片段报错或在代码围栏中生成执行标签时断言失败。 */
test("流式返回的未闭合 Markdown 片段可以逐次安全重绘", function incompleteStreaming() {
  assert.equal(renderMarkdown("**逐步"), "<p>**逐步</p>\n");
  assert.equal(renderMarkdown("**逐步补齐**"), "<p><strong>逐步补齐</strong></p>\n");
  assert.equal(renderMarkdown("[查看订单](https://"), "<p>[查看订单](https://</p>\n");
  const html = renderMarkdown('```html\n<script>alert(1)</script>\n');
  assert.match(html, /<pre><code class="language-html">&lt;script&gt;/);
  assert.doesNotMatch(html, /<script\b/i);
});

/** 无参数；返回 void；空白初始流生成多余正文时断言失败。 */
test("空回复不生成正文", function emptyInput() {
  assert.equal(renderMarkdown(""), "");
});

/** 无参数；返回 void；单编号角标未渲染成可点击按钮时断言失败。 */
test("单编号角标渲染为引用按钮", function singleCitation() {
  const html = renderMarkdown("支持 7 天无理由退货[1]。", [{ n: 1 }]);
  assert.match(html, /data-citation="1"/);
  assert.match(html, /class="citation-ref"/);
});

/** 无参数；返回 void；合并写法 [1, 4] 未被拆成独立按钮时断言失败。 */
test("合并角标 [1, 4] 拆成多个引用按钮", function mergedCitations() {
  const html = renderMarkdown("商品需保持完好[1, 4]。", [{ n: 1 }, { n: 4 }]);
  assert.match(html, /data-citation="1"/);
  assert.match(html, /data-citation="4"/);
  assert.equal((html.match(/citation-ref/g) || []).length, 2);
});

/** 无参数；返回 void；中文逗号合并没有同样处理时断言失败。 */
test("中文逗号合并角标 [1，4] 同样拆成按钮", function mergedCitationsFullWidth() {
  const html = renderMarkdown("商品需保持完好[1，4]。", [{ n: 1 }, { n: 4 }]);
  assert.equal((html.match(/citation-ref/g) || []).length, 2);
});

/** 无参数；返回 void；未登记编号被渲染成按钮时断言失败。 */
test("不在引用清单中的编号保持原文不生成按钮", function unknownCitation() {
  const html = renderMarkdown("答案[9]。", [{ n: 1 }]);
  assert.doesNotMatch(html, /data-citation="9"/);
  assert.match(html, /\[9\]/);
});

/** 无参数；返回 void；代码块内角标被转换时断言失败。 */
test("代码块中的角标保持原样不生成按钮", function citationInCode() {
  const html = renderMarkdown("```\n数组索引 arr[1]\n```", [{ n: 1 }]);
  assert.doesNotMatch(html, /citation-ref/);
});

/** 无参数；返回 void；混合单写与合并写法未被同时处理时断言失败。 */
test("单写与合并混用同时生效", function mixedCitationStyles() {
  const html = renderMarkdown("第一点[1][4]，第二点[10, 12]。", [{ n: 1 }, { n: 4 }, { n: 10 }, { n: 12 }]);
  assert.equal((html.match(/citation-ref/g) || []).length, 4);
});
