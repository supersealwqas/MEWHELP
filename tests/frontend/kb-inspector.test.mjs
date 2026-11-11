/** 知识检查器回归：运行真实页面脚本，替换DOM、网络和时钟边界。 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import vm from 'node:vm';
import { setImmediate } from 'node:timers/promises';

const moduleUrl = new URL('../../app/static/kb-inspector.mjs', import.meta.url);
const script = existsSync(moduleUrl) ? readFileSync(moduleUrl, 'utf8') : '';
const html = readFileSync(new URL('../../app/static/rag-eval.html', import.meta.url), 'utf8');

/** 最小节点：保留事件、属性、后代和纯文本语义，不模拟布局。 */
class Element extends EventTarget {
  constructor(tag = 'div') { super(); Object.assign(this, { tagName: tag, children: [], attrs: {}, style: {}, dataset: {}, value: '', checked: false, disabled: false, hidden: false, open: false, scrollTop: 0, scrollLeft: 0, _text: '' }); }
  set textContent(value) { if (this.ownerDocument?.activeElement && this.children.some(child => child.contains(this.ownerDocument.activeElement))) this.ownerDocument.activeElement = null; this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(x => x.textContent).join(''); }
  set innerHTML(value) { throw new Error('正文禁止innerHTML'); }
  appendChild(node) { this.children.push(node); return node; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  click() { if (!this.disabled) this.dispatchEvent(new Event('click')); }
  contains(other) { return this === other || this.children.some(child => child.contains(other)); }
  focus(options) { this.ownerDocument.activeElement = this; this.lastFocusOptions = options; }
}

function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
const row = (id = '9007199254740993', text = '<img src=x onerror=alert(1)>') => ({ id, milvus: { id, text, content_hash: 'hash', embedding_revision: 'rev' }, mysql: { id, category: null, answer: text, ordinal: 0, is_active: false }, sync: { status: 'ok', label: '同步一致', reasons: [] }, read_at: '2026-11-11T01:00:00+00:00' });
const page = (ids = ['9007199254740993'], more = true) => ({ rows: ids.map(id => row(id)), next_after_id: ids.at(-1) || '0', has_more: more, warnings: [], read_at: '页读取成功时间' });
const collection = { database: 'real-db', collection: 'real-collection', row_count: 55, dense_dimension: 1024, fields: [{ name: 'sparse_vector', type: 'SPARSE_FLOAT_VECTOR', readable: false, note: '不能回读' }], indexes: [{ field_name: 'vector', metric_type: 'COSINE' }], functions: [{ name: 'bm25', type: 'BM25' }], read_at: '概况读取时间' };
const recall = { query: '已提交问题', parameters: { strategy: 'weighted', dense_weight: .5, sparse_weight: .5, candidate_limit: 50, rerank_top_k: 10, rerank: true }, dense: [{ ...row('1'), rank: 1, cosine_score: .7 }], sparse: [{ ...row('2'), rank: 1, bm25_score: 3.4 }], fused: [{ ...row('1'), rank: 1, weighted_score: .6 }], counts: { dense: 1, sparse: 1, fused: 1, union: 2, eligible: 1 }, warnings: [], read_at: '召回读取时间' };
const frame = (event, data) => `event: ${event}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;

/** 安装真实脚本并返回可控制请求完成顺序的测试边界。 */
function setup(hidden = false) {
  const nodes = new Map(), requests = [], timers = new Map(); let timerId = 0;
  const document = new EventTarget(); document.hidden = hidden;
  document.createElement = tag => Object.assign(new Element(tag), { ownerDocument: document }); document.createElementNS = (_, tag) => document.createElement(tag);
  document.getElementById = id => { if (!nodes.has(id)) nodes.set(id, document.createElement('div')); return nodes.get(id); };
  const clipboard = { writeText: async value => { clipboard.last = value; } };
  const context = vm.createContext({ document, navigator: { clipboard }, TextDecoder, AbortController, console,
    fetch(path, options) { const pending = deferred(); requests.push({ path, options, ...pending }); return pending.promise; },
    setTimeout(callback, delay) { const id = ++timerId; timers.set(id, { callback, delay }); return id; }, clearTimeout(id) { timers.delete(id); },
  });
  vm.runInContext(script, context, { filename: 'kb-inspector.mjs' });
  const node = document.getElementById;
  return { nodes, requests, timers, document, node, clipboard,
    async respond(index, body, status = 200) { assert.ok(requests[index], `请求${index}未发出`); requests[index].resolve(Response.json(body, { status })); await setImmediate(); },
    async fail(index) { requests[index].reject(new Error('网络断开')); await setImmediate(); },
    async tick(delay) { const entry = [...timers].find(([, timer]) => timer.delay === delay); assert.ok(entry, `缺少${delay}ms定时器`); timers.delete(entry[0]); entry[1].callback(); await setImmediate(); },
    async stream(index) { let controller, cancelled = false; requests[index].resolve(new Response(new ReadableStream({ start(c) { controller = c; }, cancel() { cancelled = true; } }))); await setImmediate(); return { async push(text) { controller.enqueue(new TextEncoder().encode(text)); await setImmediate(); }, async end() { if (!cancelled) controller.close(); await setImmediate(); } }; },
    async change(id, value, type = 'change') { node(id).value = String(value); node(id).dispatchEvent(new Event(type)); await setImmediate(); },
    async visible(hidden) { document.hidden = hidden; document.dispatchEvent(new Event('visibilitychange')); await setImmediate(); },
  };
}
const posts = f => f.requests.filter(r => r.options?.method === 'POST');
const descendants = node => [node, ...node.children.flatMap(descendants)];
async function initial(f) { await f.respond(0, collection); await f.respond(1, page()); }
async function submit(f, query = '已提交问题') { f.node('kbQuery').value = query; f.node('kbSearch').click(); await setImmediate(); }

// 缺少模块加载或控件会使入口检查失败；空实现不能发出首次读取。
test('页面加载独立同源模块，首开只读取实时概况和20条列表', async () => {
  assert.match(html, /src="\/static\/kb-inspector\.mjs"/);
  assert.match(html, /href="\/static\/kb-inspector\.css"/);
  const f = setup(); assert.deepEqual(f.requests.map(r => r.path), ['/kb/collection', '/kb/records?after_id=0&limit=20']);
  await initial(f); assert.match(f.node('kbOverview').textContent, /real-db.*55.*1024/); assert.match(f.node('kbSchema').textContent, /sparse_vector.*不能回读/);
  assert.match(f.node('kbRows').textContent, /9007199254740993/); assert.equal(posts(f).length, 0);
});

test('分页保留字符串游标历史，删除造成空页时回首页', async () => {
  const f = setup(); await initial(f); f.node('kbNext').click(); assert.equal(f.requests[2].path, '/kb/records?after_id=9007199254740993&limit=20');
  await f.respond(2, page(['9007199254740999'], false)); f.node('kbPrev').click(); assert.equal(f.requests[3].path, '/kb/records?after_id=0&limit=20'); await f.respond(3, page());
  f.node('kbNext').click(); await f.respond(4, page([], false)); assert.equal(f.requests[5].path, '/kb/records?after_id=0&limit=20'); await f.respond(5, page([], false)); assert.match(f.node('kbRows').textContent, /暂无记录/);
});

test('详情按ID定位，纯文本显示原始空值并绘制真实向量与复制', async () => {
  const f = setup(); await initial(f); f.node('kbId').value = '9007199254740993'; f.node('kbLocate').click(); assert.equal(f.requests[2].path, '/kb/records/9007199254740993');
  await f.respond(2, { ...row(), milvus: { ...row().milvus, vector: [3, 4, 0] } });
  assert.match(f.node('kbDetail').textContent, /Milvus.*<img src=x onerror=alert\(1\)>.*MySQL.*NULL/s);
  assert.match(f.node('kbDetail').textContent, /min：0.*max：4.*norm：5/);
  const svg = descendants(f.node('kbDetail')).find(n => n.tagName === 'svg'); assert.ok(svg); assert.equal(descendants(svg).find(n => n.tagName === 'polyline').attrs.points.split(' ').length, 3);
  descendants(f.node('kbDetail')).find(n => n.tagName === 'button').click(); await setImmediate(); assert.equal(f.clipboard.last, '[3,4,0]');
  f.clipboard.writeText = async () => { throw new Error('denied'); }; descendants(f.node('kbDetail')).find(n => n.tagName === 'button').click(); await setImmediate(); assert.match(f.node('kbDetailStatus').textContent, /复制失败/);
});

test('列表和详情乱序响应不能覆盖新选择，失败保留成功数据和时间', async () => {
  const f = setup(); await initial(f); f.node('kbRefresh').click(); f.node('kbNext').click();
  await f.respond(4, page(['99'], false)); await f.respond(3, page(['old'])); await f.respond(2, collection); assert.match(f.node('kbRows').textContent, /99/); assert.doesNotMatch(f.node('kbRows').textContent, /old/);
  f.node('kbId').value = '1'; f.node('kbLocate').click(); f.node('kbId').value = '2'; f.node('kbLocate').click(); await f.respond(6, row('2', '新详情')); await f.respond(5, row('1', '旧详情')); assert.match(f.node('kbDetail').textContent, /新详情/); assert.doesNotMatch(f.node('kbDetail').textContent, /旧详情/);
  f.node('kbRefresh').click(); await f.fail(7); await f.fail(8); await f.fail(9); assert.match(f.node('kbListStatus').textContent, /失败.*页读取成功时间/); assert.match(f.node('kbRows').textContent, /99/); assert.match(f.node('kbDetail').textContent, /新详情/);
});

test('五秒浏览轮询不检索，暂停和隐藏停止，恢复立即读取，详情删除标不可用', async () => {
  const f = setup(); await initial(f); await f.tick(5000); assert.equal(posts(f).length, 0); await f.respond(2, collection); await f.respond(3, page());
  f.node('kbPause').click(); assert.equal([...f.timers].filter(([,t]) => t.delay === 5000).length, 0); const n = f.requests.length; await f.visible(true); await f.visible(false); assert.equal(f.requests.length, n);
  f.node('kbPause').click(); assert.equal(f.requests.length, n + 2); await f.respond(n, collection); await f.respond(n + 1, page());
  await f.visible(true); assert.equal(f.timers.size, 0); await f.visible(false); assert.equal(f.requests.length, n + 4); await f.respond(n + 2, collection); await f.respond(n + 3, page());
  f.node('kbId').value = '1'; f.node('kbLocate').click(); await f.respond(n + 4, row('1')); f.node('kbRefresh').click(); await f.respond(n + 5, collection); await f.respond(n + 6, page()); await f.respond(n + 7, { detail: '不存在' }, 404); assert.match(f.node('kbDetailStatus').textContent, /不可用.*删除/);
});

test('缺MySQL保留Milvus正文及具体同步原因，无效ID不发请求', async () => {
  const f = setup(); await f.respond(0, collection); await f.respond(1, { ...page(), rows: [{ ...row(), mysql: null, sync: { label: 'MySQL不可用', reasons: ['读取超时'] } }], warnings: ['数据库暂不可用'] });
  assert.match(f.node('kbRows').textContent, /MySQL不可用/); assert.match(f.node('kbListStatus').textContent, /数据库暂不可用/);
  f.node('kbId').value = '90071992547409930x'; f.node('kbLocate').click(); assert.equal(f.requests.length, 2); assert.match(f.node('kbDetailStatus').textContent, /ID/);
});

test('首次仅点击或Enter检索，Shift+Enter和输入草稿不请求，默认参数精确', async () => {
  const f = setup(); await initial(f); await f.change('kbDenseLimit', 20); assert.equal(posts(f).length, 0); f.node('kbDenseLimit').value = '50';
  f.node('kbQuery').value = '  已提交问题  '; const shift = new Event('keydown'); Object.assign(shift, { key: 'Enter', shiftKey: true }); f.node('kbQuery').dispatchEvent(shift); assert.equal(posts(f).length, 0);
  const enter = new Event('keydown', { cancelable: true }); Object.assign(enter, { key: 'Enter', shiftKey: false }); f.node('kbQuery').dispatchEvent(enter); await setImmediate(); assert.equal(enter.defaultPrevented, true);
  assert.deepEqual(JSON.parse(posts(f)[0].options.body), { query: '已提交问题', strategy: 'weighted', dense_weight: .5, sparse_weight: .5, dense_limit: 50, sparse_limit: 50, candidate_limit: 50, rerank_top_k: 10, rerank: true });
  await f.change('kbQuery', '新草稿', 'input'); assert.equal(posts(f).length, 1);
});

test('权重联动合计为1，非weighted禁用，调参600ms刷新已提交query且串行只保留最新快照', async () => {
  const f = setup(); await initial(f); await submit(f); const s = await f.stream(2);
  f.node('kbQuery').value = '未提交草稿'; await f.change('kbDenseWeight', .8, 'input'); assert.equal(Number(f.node('kbSparseWeight').value), .2);
  await f.change('kbDenseLimit', 30); await f.tick(600); assert.equal(posts(f).length, 1);
  await f.change('kbDenseLimit', 31); await f.tick(600); assert.equal(posts(f).length, 1);
  await s.push(frame('complete', { timings: { total_ms: 1 } })); await s.end(); assert.equal(posts(f).length, 2); const actual = JSON.parse(posts(f)[1].options.body); assert.equal(actual.query, '已提交问题'); assert.equal(actual.dense_limit, 31); assert.equal(actual.dense_weight + actual.sparse_weight, 1);
  await f.change('kbStrategy', 'rrf'); assert.equal(f.node('kbDenseWeight').disabled, true); assert.equal(f.node('kbSparseWeight').disabled, true);
});

test('明确提交新query清除旧防抖事件，旧流不能覆盖新快照', async () => {
  const f = setup(); await initial(f); await submit(f, '旧问题'); const s = await f.stream(2); await f.change('kbDenseLimit', 12);
  await submit(f, '新问题'); assert.equal([...f.timers].filter(([,t]) => t.delay === 600).length, 0);
  await s.push(frame('status', { message: '旧进度' }) + frame('recall', recall) + frame('complete', { timings: {} })); await s.end(); assert.equal(JSON.parse(posts(f)[1].options.body).query, '新问题'); assert.doesNotMatch(f.node('kbSearchStatus').textContent, /旧进度/); assert.equal(f.node('kbRecall').textContent, '');
});

test('POST SSE跨chunk/CRLF先显示召回各路，再显示重排排名变化和真实耗时', async () => {
  const f = setup(); await initial(f); await submit(f); const s = await f.stream(2);
  await s.push(frame('status', { stage: 'recall', message: '召回进行中' })); assert.match(f.node('kbSearchStatus').textContent, /召回进行中/);
  const data = frame('recall', recall); await s.push(data.slice(0, 23)); assert.equal(f.node('kbRecall').textContent, ''); await s.push(data.slice(23, -1)); await s.push(data.slice(-1));
  assert.match(f.node('kbRecall').textContent, /weighted_score.*0.6/s); assert.match(f.node('kbSearchMeta').textContent, /召回读取时间.*1.*50/s);
  f.node('kbTabDense').click(); assert.match(f.node('kbRecall').textContent, /cosine_score.*0.7/s); f.node('kbTabSparse').click(); assert.match(f.node('kbRecall').textContent, /bm25_score.*3.4/s); assert.doesNotMatch(f.node('kbRecall').textContent, /weighted_score/);
  await s.push(frame('rerank', { rows: [{ ...row('1'), rank: 1, recall_rank: 3, rerank_score: .9, rank_change: 2 }], scored_ids: ['1'], skipped: [{ id: '2', reason: '未进入Top-K' }] })); assert.match(f.node('kbRerank').textContent, /rerank_score.*0.9.*\+2.*未进入Top-K/s);
  await s.push(frame('complete', { timings: { encoding_ms: 1, recall_ms: 2, rerank_ms: 3, total_ms: 6 } })); await s.end(); assert.match(f.node('kbSearchMeta').textContent, /total_ms：6/); assert.match(f.node('kbSearchStatus').textContent, /完成/);
  descendants(f.node('kbRerank')).find(n => n.tagName === 'button').click(); assert.equal(f.requests[3].path, '/kb/records/1');
});

for (const terminal of ['cut', 'error', 'invalid']) test(`SSE ${terminal}保留初步结果并明确错误`, async () => {
  const f = setup(); await initial(f); await submit(f); const s = await f.stream(2); await s.push(frame('recall', recall));
  if (terminal === 'error') await s.push(frame('error', { stage: 'rerank', message: '重排暂不可用' }));
  if (terminal === 'invalid') await s.push('event: rerank\ndata: invalid\n\n');
  await s.end(); assert.match(f.node('kbRecall').textContent, /9007199254740993|weighted_score/); assert.match(f.node('kbSearchStatus').textContent, /失败|错误|断开|终止/);
});

test('关闭重排明确显示已关闭，零权重对照行只存在单路tab', async () => {
  const f = setup(); await initial(f); f.node('kbRerankEnabled').checked = false; await submit(f); const s = await f.stream(2);
  await s.push(frame('recall', { ...recall, parameters: { ...recall.parameters, dense_weight: 1, sparse_weight: 0, rerank: false } })); assert.doesNotMatch(f.node('kbRecall').textContent, /<img.*2/); assert.match(f.node('kbRerank').textContent, /已关闭/);
  await s.push(frame('rerank', { rows: [], skipped: [{ id: '1', reason: '未启用重排' }] }) + frame('complete', { timings: {} })); await s.end(); assert.match(f.node('kbRerank').textContent, /已关闭/);
});

test('数量输入本身600ms刷新已提交原问，展示重排独立读取时间', async () => {
  const f = setup(); await initial(f); await submit(f); const s = await f.stream(2);
  await s.push(frame('recall', recall) + frame('rerank', { rows: [], read_at: '重排读取时间', scored_ids: [], skipped: [] }));
  assert.match(f.node('kbSearchMeta').textContent, /重排读取时间/);
  await s.push(frame('complete', { timings: {} })); await s.end(); f.node('kbQuery').value = '草稿';
  await f.change('kbCandidateLimit', 9, 'input'); await f.tick(600); assert.equal(JSON.parse(posts(f)[1].options.body).query, '已提交问题'); assert.equal(JSON.parse(posts(f)[1].options.body).candidate_limit, 9);
});

test('中文输入法确认回车不提交，非法量和空问题不发送POST', async () => {
  const f = setup(); await initial(f); f.node('kbQuery').value = '中文'; const enter = new Event('keydown', { cancelable: true }); Object.assign(enter, { key: 'Enter', isComposing: true }); f.node('kbQuery').dispatchEvent(enter); assert.equal(posts(f).length, 0); assert.equal(enter.defaultPrevented, false);
  f.node('kbDenseLimit').value = '1.5'; await submit(f); assert.equal(posts(f).length, 0); assert.match(f.node('kbSearchStatus').textContent, /整数/); f.node('kbDenseLimit').value = '50'; await submit(f, ' '); assert.equal(posts(f).length, 0); assert.match(f.node('kbSearchStatus').textContent, /1–2000/);
});

test('首开隐藏不读取，恢复后同一GET飞行时五秒时钟不堆叠请求', async () => {
  const f = setup(true); assert.equal(f.requests.length, 0); assert.equal(f.timers.size, 0); await f.visible(false); await initial(f); await f.tick(5000); await f.tick(5000); assert.equal(f.requests.length, 4); assert.equal(posts(f).length, 0); await f.respond(2, collection); await f.respond(3, page());
  await f.visible(true); assert.equal(f.timers.size, 0); await f.visible(false); assert.equal(f.requests.length, 6);
});

test('完整1024向量不裁剪，流中文UTF8字节跨chunk仍正确显示', async () => {
  const f = setup(); await initial(f); f.node('kbId').value = '1'; f.node('kbLocate').click(); const vector = Array.from({ length: 1024 }, (_, i) => i / 1024);
  await f.respond(2, { ...row('1'), milvus: { ...row('1').milvus, vector } });
  const polyline = descendants(f.node('kbDetail')).find(n => n.tagName === 'polyline'); assert.equal(polyline.attrs.points.split(' ').length, 1024); descendants(f.node('kbDetail')).find(n => n.tagName === 'button').click(); await setImmediate(); assert.deepEqual(JSON.parse(f.clipboard.last), vector);
  await submit(f); let controller; f.requests[3].resolve(new Response(new ReadableStream({ start(c) { controller = c; } }))); await setImmediate();
  const bytes = new TextEncoder().encode(frame('status', { message: '中文进行中' })); const split = bytes.findIndex(b => b >= 128) + 1; controller.enqueue(bytes.slice(0, split)); await setImmediate(); controller.enqueue(bytes.slice(split)); await setImmediate(); assert.match(f.node('kbSearchStatus').textContent, /中文进行中/); controller.close(); await setImmediate();
});

test('问题长度按Unicode字符匹配后端契约，不裁剪合法2000字符原问', async () => {
  const f = setup(); await initial(f); const query = '🐱'.repeat(2000); await submit(f, query); assert.equal(posts(f).length, 1); assert.equal(JSON.parse(posts(f)[0].options.body).query, query);
  const g = setup(); await initial(g); await submit(g, '🐱'.repeat(2001)); assert.equal(posts(g).length, 0); assert.match(g.node('kbSearchStatus').textContent, /1–2000/);
});

/** 根据已渲染详情取得数组展开、复制和滚动节点，不窥探生产模块内部状态。 */
function vectorView(f) {
  const all = descendants(f.node('kbDetail'));
  return { fold: all.find(n => n.tagName === 'details'), summary: all.find(n => n.tagName === 'summary'), copy: all.find(n => n.tagName === 'button'), array: all.find(n => n.tagName === 'pre') };
}
const detail = (id = '1', vector = [3, 4, 0]) => ({ ...row(id), milvus: { ...row(id).milvus, vector } });
async function selectDetail(f, id = '1', data = detail(id)) { f.node('kbId').value = id; f.node('kbLocate').click(); await f.respond(f.requests.length - 1, data); }

test('五秒同ID无字段变化保留向量展开/滚动/焦点及列表按钮，成功时间仍刷新', async () => {
  const f = setup(); await initial(f); await selectDetail(f); const original = vectorView(f), listButton = descendants(f.node('kbRows')).find(n => n.tagName === 'button');
  original.fold.open = true; original.array.scrollTop = 80; original.array.scrollLeft = 12; original.copy.focus(); f.node('kbRows').scrollTop = 55;
  await f.tick(5000); await f.respond(3, collection); await f.respond(4, { ...page(), rows: page().rows.map(item => ({ ...item, read_at: '列表行新时间' })), read_at: '列表新成功时间' }); await f.respond(5, { ...detail(), read_at: '详情新成功时间' });
  const current = vectorView(f); assert.equal(current.fold, original.fold, '无实质变化不能重建details'); assert.equal(current.fold.open, true); assert.equal(current.array.scrollTop, 80); assert.equal(current.array.scrollLeft, 12); assert.equal(f.document.activeElement, original.copy); assert.equal(descendants(f.node('kbRows')).find(n => n.tagName === 'button'), listButton); assert.equal(f.node('kbRows').scrollTop, 55);
  assert.match(f.node('kbDetailStatus').textContent, /详情新成功时间/); assert.match(f.node('kbListStatus').textContent, /列表新成功时间/); assert.equal(posts(f).length, 0);
  listButton.focus(); await f.tick(5000); await f.respond(6, collection); await f.respond(7, page()); await f.respond(8, detail()); assert.equal(f.document.activeElement, listButton);
});

for (const focusTarget of ['copy', 'summary', 'array']) test(`同ID字段更新可见，保留数组展开位置和${focusTarget}焦点；换ID重置`, async () => {
  const f = setup(); await initial(f); await selectDetail(f); const original = vectorView(f); original.fold.open = true; original.array.scrollTop = 60; original.array.scrollLeft = 9; original[focusTarget].focus();
  await f.tick(5000); await f.respond(3, collection); await f.respond(4, { ...page(), rows: [row('99', '列表新正文')] }); const changed = detail('1', [0, 6, 8]); changed.mysql.answer = '详情新答案'; await f.respond(5, changed);
  const current = vectorView(f); assert.notEqual(current.fold, original.fold); assert.equal(current.fold.open, true); assert.equal(current.array.scrollTop, 60); assert.equal(current.array.scrollLeft, 9); assert.equal(f.document.activeElement, current[focusTarget]); assert.equal(current[focusTarget].lastFocusOptions?.preventScroll, true); assert.match(f.node('kbDetail').textContent, /详情新答案/); assert.equal(current.array.textContent, '[0,6,8]'); assert.match(f.node('kbRows').textContent, /列表新正文/);
  await selectDetail(f, '2', detail('2')); const next = vectorView(f); assert.equal(next.fold.open, false); assert.equal(next.array.scrollTop, 0); assert.equal(next.array.scrollLeft, 0); assert.notEqual(f.document.activeElement, next[focusTarget]);
});

test('字段变化时保留详情外其他控件焦点，故障与删除不关闭原数组或伪报新成功', async () => {
  const f = setup(); await initial(f); await selectDetail(f); vectorView(f).fold.open = true; f.node('kbQuery').focus();
  await f.tick(5000); await f.respond(3, collection); await f.respond(4, page()); await f.respond(5, { ...detail('1', [9, 0, 0]), read_at: '最后详情成功' }); assert.equal(f.document.activeElement, f.node('kbQuery')); assert.equal(vectorView(f).fold.open, true);
  const current = vectorView(f); await f.tick(5000); await f.respond(6, collection); await f.respond(7, page()); await f.fail(8); assert.equal(vectorView(f).fold, current.fold); assert.match(f.node('kbDetailStatus').textContent, /读取失败.*最后详情成功/);
  await f.tick(5000); await f.respond(9, collection); await f.respond(10, page()); await f.respond(11, { detail: '已删除' }, 404); assert.equal(vectorView(f).fold, current.fold); assert.equal(current.fold.open, true); assert.match(f.node('kbDetailStatus').textContent, /不可用.*最后详情成功/);
});
