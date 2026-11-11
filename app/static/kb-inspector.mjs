/** 实时知识检查器：只读浏览两库原始字段，独立串行运行分阶段检索调试。 */
const kb = id => document.getElementById(id);
const limits = ['kbDenseLimit', 'kbSparseLimit', 'kbCandidateLimit', 'kbRerankTopK'];
let cursor = '0', cursorHistory = [], nextCursor = '0', hasMore = false, selectedId = null;
let paused = false, browseTimer = null;
// 概况、列表、详情分别编号；自动刷新跳过飞行请求，用户主动操作可使旧请求失效。
const resources = {
  collection: { revision: 0, flights: 0, success: null },
  list: { revision: 0, flights: 0, success: null },
  detail: { revision: 0, flights: 0, success: null },
};
let submittedQuery = null, searchRevision = 0, searchFlight = false, pendingSearch = null, debounceTimer = null;
let recallData = null, activeTab = 'fused', searchParameters = null, searchMeta = '', searchTiming = '';
let renderedListContent = null, renderedDetailContent = null, detailView = null;

/** 创建纯文本节点；参数为标签、文本、局部类名；返回DOM节点。 */
function node(tag, text, cls) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (cls) result.className = cls;
  return result;
}

/** 显示真实原始值：null为NULL，缺字段为未取得，不推测业务值。 */
function raw(value) {
  if (value === null) return 'NULL';
  if (value === undefined) return '未取得';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

/** 比较记录的实质字段：读取时间单独更新，不用于触发原文DOM重建。 */
function recordContent(record) {
  const { read_at, ...fields } = record;
  return fields;
}

/** 设置状态文字和错误配色；参数为控件ID、文字和错误标志。 */
function status(id, text, error = false) {
  kb(id).textContent = text;
  kb(id).className = error ? 'fw-muted kb-error' : 'fw-muted';
}

/** 读取同源JSON；非成功响应抛出HTTP状态，不展示底层网络细节。 */
async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw Object.assign(new Error(`HTTP ${response.status}`), { status: response.status });
  return response.json();
}

/** 原始字段表：保留字段名和值，跳过另行可折叠展示的向量。 */
function rawFields(target, data) {
  const list = node('dl', undefined, 'kb-raw');
  for (const [key, value] of Object.entries(data || {})) {
    if (key === 'vector') continue;
    list.appendChild(node('dt', key)); list.appendChild(node('dd', raw(value)));
  }
  target.appendChild(list);
}

/** 按ID打开公共详情区的键盘可用按钮；ID始终保留字符串。 */
function recordButton(id) {
  const button = node('button', String(id), 'kb-id');
  button.setAttribute('type', 'button'); button.setAttribute('aria-label', `查看知识 ${id} 详情`);
  button.addEventListener('click', () => { selectedId = String(id); kb('kbId').value = selectedId; loadDetail(); });
  return button;
}

/** 真实命中表：明确区分各路分数列，缺分数保持空值，不重算融合排名。 */
function recordTable(target, rows, scoreColumns = []) {
  target.textContent = '';
  if (!rows.length) { target.appendChild(node('p', '暂无记录', 'fw-muted')); return; }
  const table = node('table', undefined, 'kb-table'), head = node('thead'), header = node('tr'), body = node('tbody');
  const headers = scoreColumns.length ? ['ID', ...scoreColumns, '同步状态', 'Milvus text'] : ['ID', 'MySQL category', '同步状态', 'Milvus text'];
  for (const title of headers) header.appendChild(node('th', title));
  head.appendChild(header); table.appendChild(head);
  for (const item of rows) {
    const tr = node('tr'), idCell = node('td'); idCell.appendChild(recordButton(item.id)); tr.appendChild(idCell);
    if (scoreColumns.length) {
      for (const key of scoreColumns) {
        const value = key === 'rank_change' && typeof item[key] === 'number' && item[key] > 0 ? `+${item[key]}` : raw(item[key]);
        tr.appendChild(node('td', value, 'num'));
      }
    } else tr.appendChild(node('td', raw(item.mysql?.category)));
    const sync = node('td', item.sync?.label || '状态未取得');
    if (item.sync?.reasons?.length) sync.appendChild(node('small', item.sync.reasons.join('；')));
    const preview = node('td'); preview.appendChild(node('div', raw(item.milvus?.text), 'kb-preview'));
    tr.appendChild(sync); tr.appendChild(preview); body.appendChild(tr);
  }
  table.appendChild(body); target.appendChild(table);
}

/** 渲染真实collection概况与公开结构，不使用推测计数或默认维度。 */
function renderCollection(data) {
  const box = kb('kbOverview'); box.textContent = '';
  for (const [label, key] of [['数据库', 'database'], ['集合', 'collection'], ['实时记录数', 'row_count'], ['Dense 维度', 'dense_dimension']]) box.appendChild(node('div', `${label}：${raw(data[key])}`));
  const schema = kb('kbSchema'); schema.textContent = '';
  for (const [label, key] of [['Milvus fields', 'fields'], ['Milvus functions', 'functions'], ['Milvus indexes', 'indexes']]) {
    schema.appendChild(node('h3', label));
    for (const item of data[key] || []) rawFields(schema, item);
  }
}

/** 读取概况：失败保留旧概况及成功读取时间；自动调用不重叠。 */
async function loadCollection(automatic = false) {
  const state = resources.collection;
  if (automatic && state.flights) return;
  const revision = ++state.revision; state.flights++;
  try {
    const data = await getJson('/kb/collection');
    if (revision !== state.revision) return;
    renderCollection(data); state.success = data.read_at;
    status('kbCollectionStatus', `概况读取时间：${raw(state.success)}`);
  } catch {
    if (revision === state.revision) status('kbCollectionStatus', `概况读取失败；上次成功：${raw(state.success)}`, true);
  } finally { state.flights--; }
}

/** 更新分页按钮与游标页号；只有当前页返回后才允许继续下一页。 */
function pageControls() {
  kb('kbPrev').disabled = cursorHistory.length === 0;
  kb('kbNext').disabled = !hasMore;
  kb('kbPage').textContent = `第 ${cursorHistory.length + 1} 页 · 每页 20 条`;
}

/** 读取20条实时页；旧页响应失效，删除造成空页时重置首页。 */
async function loadList(automatic = false) {
  const state = resources.list;
  if (automatic && state.flights) return;
  const revision = ++state.revision, requestCursor = cursor; state.flights++;
  pageControls(); kb('kbRows').setAttribute('aria-busy', 'true');
  try {
    const data = await getJson(`/kb/records?after_id=${requestCursor}&limit=20`);
    if (revision !== state.revision) return;
    if (!data.rows.length && requestCursor !== '0') {
      cursor = '0'; cursorHistory = []; loadList(); return;
    }
    const content = JSON.stringify(data.rows.map(recordContent));
    // 同页数据没有变化时保留原按钮和局部滚动位置，避免五秒刷新打断键盘操作。
    if (content !== renderedListContent) { recordTable(kb('kbRows'), data.rows); renderedListContent = content; }
    nextCursor = String(data.next_after_id); hasMore = data.has_more;
    state.success = data.read_at; pageControls();
    status('kbListStatus', `列表读取时间：${raw(state.success)}${data.warnings?.length ? `；${data.warnings.join('；')}` : ''}`, Boolean(data.warnings?.length));
  } catch {
    if (revision === state.revision) status('kbListStatus', `列表读取失败；上次成功：${raw(state.success)}`, true);
  } finally {
    state.flights--;
    if (revision === state.revision) kb('kbRows').setAttribute('aria-busy', 'false');
  }
}

/** 显示所有实际向量分量、真实极值及欧氏长度；复制失败明确提示。 */
function renderVector(target, vector) {
  target.appendChild(node('h3', 'Milvus vector'));
  if (!Array.isArray(vector) || !vector.length || !vector.every(value => typeof value === 'number' && Number.isFinite(value))) {
    target.appendChild(node('p', '未取得可绘制向量', 'fw-muted')); return;
  }
  const min = Math.min(...vector), max = Math.max(...vector), norm = Math.hypot(...vector);
  target.appendChild(node('p', `实际分量：${vector.length}；min：${min}；max：${max}；norm：${norm}`, 'fw-muted'));
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 600 120'); svg.setAttribute('class', 'kb-vector'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `实际向量${vector.length}个分量折线，最小值${min}，最大值${max}`);
  const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  polyline.setAttribute('points', vector.map((value, index) => `${index * 600 / Math.max(vector.length - 1, 1)},${max === min ? 60 : 110 - (value - min) * 100 / (max - min)}`).join(' '));
  svg.appendChild(polyline); target.appendChild(svg);
  const fold = node('details', undefined, 'kb-fold'), summary = node('summary', `展开完整 ${vector.length} 分量数组`); fold.appendChild(summary);
  const text = JSON.stringify(vector), copy = node('button', '复制完整向量', 'act ghost');
  copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(text); status('kbDetailStatus', '完整向量已复制。'); }
    catch { status('kbDetailStatus', '复制失败，请展开数组后手动选择复制。', true); }
  });
  const array = node('pre', text, 'kb-vector-json'); array.setAttribute('tabindex', '0');
  fold.appendChild(copy); fold.appendChild(array); target.appendChild(fold);
  return { fold, summary, copy, array };
}

/** 展示记录的两库原始字段与同步原因；NULL不转换成业务默认值。 */
function renderDetail(data) {
  const box = kb('kbDetail'), sameRecord = detailView?.id === String(data.id), previous = sameRecord ? detailView.vector : null;
  const active = document.activeElement;
  const position = previous ? { open: previous.fold.open, top: previous.array.scrollTop, left: previous.array.scrollLeft } : null;
  const boxPosition = sameRecord ? { top: box.scrollTop, left: box.scrollLeft } : { top: 0, left: 0 };
  box.textContent = '';
  box.appendChild(node('h3', `知识 ID：${String(data.id)}`));
  box.appendChild(node('p', `${data.sync?.label || '同步状态未取得'}${data.sync?.reasons?.length ? `：${data.sync.reasons.join('；')}` : ''}`));
  for (const source of ['milvus', 'mysql']) {
    box.appendChild(node('h3', source === 'milvus' ? 'Milvus 原始字段' : 'MySQL 原始字段'));
    if (data[source] === null || data[source] === undefined) box.appendChild(node('p', `${source === 'milvus' ? 'Milvus' : 'MySQL'}：${raw(data[source])}；${data.sync?.label || '请查看同步状态'}`));
    else rawFields(box, data[source]);
  }
  const vector = renderVector(box, data.milvus?.vector);
  detailView = { id: String(data.id), vector };
  if (previous && vector) {
    vector.fold.open = position.open;
    // 字段发生变化时才重建；只恢复原数组相关焦点，不抢走其他控件的焦点。
    for (const key of ['summary', 'copy', 'array']) if (active === previous[key]) vector[key].focus({ preventScroll: true });
    vector.array.scrollTop = position.top; vector.array.scrollLeft = position.left;
  }
  box.scrollTop = boxPosition.top; box.scrollLeft = boxPosition.left;
}

/** 读取已选详情，保持独立编号与飞行状态；404标不可用并保留上次原文。 */
async function loadDetail(automatic = false) {
  if (selectedId === null) return;
  const state = resources.detail;
  if (automatic && state.flights) return;
  const revision = ++state.revision, id = selectedId; state.flights++;
  kb('kbDetail').setAttribute('aria-busy', 'true');
  try {
    const data = await getJson(`/kb/records/${id}`);
    if (revision !== state.revision) return;
    const content = JSON.stringify(recordContent(data));
    if (content !== renderedDetailContent) { renderDetail(data); renderedDetailContent = content; }
    state.success = data.read_at;
    status('kbDetailStatus', `详情 ID ${id}；读取时间：${raw(state.success)}`);
  } catch (error) {
    if (revision === state.revision) status('kbDetailStatus', error.status === 404 ? `知识 ${id} 当前不可用，可能已删除；上次成功：${raw(state.success)}` : `详情 ID ${id} 读取失败；上次成功：${raw(state.success)}`, true);
  } finally { state.flights--; if (revision === state.revision) kb('kbDetail').setAttribute('aria-busy', 'false'); }
}

/** 校验完整signed INT64主键字符串后定位，不通过Number破坏精度。 */
function locate() {
  const id = kb('kbId').value.trim();
  if (!/^[1-9]\d{0,18}$/.test(id) || (id.length === 19 && id > '9223372036854775807')) { status('kbDetailStatus', 'ID 需为 1 到 9223372036854775807 的完整整数。', true); return; }
  selectedId = id; loadDetail();
}

/** 独立浏览刷新，绝不调用模型检索接口。 */
function refreshBrowse(automatic = false) { loadCollection(automatic); loadList(automatic); loadDetail(automatic); }

/** 只保留一个五秒浏览时钟；隐藏或手动暂停时清除，恢复立即刷新。 */
function scheduleBrowse() {
  clearTimeout(browseTimer); browseTimer = null;
  if (paused || document.hidden) return;
  browseTimer = setTimeout(() => { browseTimer = null; refreshBrowse(true); scheduleBrowse(); }, 5000);
}

/** 权重联动与模式禁用；使用两位小数保持滑块合计恰为1。 */
function weights(changed = null) {
  if (changed) {
    const counterpart = changed === 'kbDenseWeight' ? 'kbSparseWeight' : 'kbDenseWeight';
    kb(counterpart).value = (1 - Number(kb(changed).value)).toFixed(2);
  }
  for (const id of ['kbDenseWeight', 'kbSparseWeight']) {
    kb(id).disabled = kb('kbStrategy').value !== 'weighted';
    kb(`${id}Value`).textContent = Number(kb(id).value).toFixed(2);
  }
}

/** 捕获提交快照：查询使用最近明确提交原问，限制严格整数1–50。 */
function snapshot(query) {
  // Python接口按Unicode码点计数；JS字符串长度会把表情算成两个字符，不能据此拒绝合法原问。
  if (!query || Array.from(query).length > 2000) throw new Error('请提交 1–2000 字的问题。');
  const values = limits.map(id => Number(kb(id).value));
  if (values.some(value => !Number.isInteger(value) || value < 1 || value > 50)) throw new Error('召回量、候选量和 Top-K 需为 1–50 的整数。');
  return { query, strategy: kb('kbStrategy').value, dense_weight: Number(kb('kbDenseWeight').value), sparse_weight: Number(kb('kbSparseWeight').value), dense_limit: values[0], sparse_limit: values[1], candidate_limit: values[2], rerank_top_k: values[3], rerank: kb('kbRerankEnabled').checked };
}

/** 切换真实召回分组；只有fused数组代表融合命中，零权重对照不混入。 */
function renderRecall() {
  for (const [tab, id] of [['fused', 'kbTabFused'], ['dense', 'kbTabDense'], ['sparse', 'kbTabSparse']]) kb(id).setAttribute('aria-pressed', String(activeTab === tab));
  if (!recallData) return;
  const columns = activeTab === 'dense' ? ['rank', 'cosine_score'] : activeTab === 'sparse' ? ['rank', 'bm25_score'] : ['rank', 'dense_rank', 'sparse_rank', 'cosine_score', 'bm25_score', ...(searchParameters.strategy === 'weighted' ? ['weighted_score'] : searchParameters.strategy === 'rrf' ? ['rrf_score'] : [])];
  recordTable(kb('kbRecall'), recallData[activeTab] || [], columns);
}

/** 实际参数、读取时间、候选上限、数量及耗时合并显示。 */
function renderSearchMeta() { kb('kbSearchMeta').textContent = `${searchMeta}${searchTiming ? `\n${searchTiming}` : ''}`; }

/** 接收流事件，只渲染最新已经提交快照的事件。 */
function searchEvent(event, data, revision) {
  if (revision !== searchRevision) return;
  if (event === 'status') status('kbSearchStatus', data.message);
  if (event === 'recall') {
    recallData = data; searchParameters = data.parameters; activeTab = 'fused'; renderRecall();
    searchMeta = `已提交问题：${data.query}\n实际参数：${JSON.stringify(data.parameters)}\n读取时间：${raw(data.read_at)}\n候选数/上限：${raw(data.counts?.fused)} / ${raw(data.parameters?.candidate_limit)}；Dense：${raw(data.counts?.dense)}；BM25：${raw(data.counts?.sparse)}；并集：${raw(data.counts?.union)}；可重排：${raw(data.counts?.eligible)}${data.warnings?.length ? `\n${data.warnings.join('；')}` : ''}`;
    renderSearchMeta(); status('kbSearchStatus', '初步召回已显示，等待后续阶段。');
    if (!searchParameters.rerank) kb('kbRerank').textContent = '重排已关闭。';
  }
  if (event === 'rerank') {
    searchMeta += `\n重排读取时间：${raw(data.read_at)}`; renderSearchMeta();
    if (searchParameters.rerank) recordTable(kb('kbRerank'), data.rows || [], ['rank', 'recall_rank', 'rerank_score', 'rank_change']);
    else kb('kbRerank').textContent = '重排已关闭。';
    kb('kbRerank').appendChild(node('p', `实际评分候选：${(data.scored_ids || []).map(String).join('、') || '无'}`, 'fw-muted'));
    for (const item of data.skipped || []) kb('kbRerank').appendChild(node('p', `ID ${String(item.id)}：${item.reason}`, 'fw-muted'));
  }
  if (event === 'complete') {
    searchTiming = `耗时（毫秒）：${Object.entries(data.timings || {}).map(([key, value]) => `${key}：${raw(value)}`).join('；')}`;
    renderSearchMeta(); status('kbSearchStatus', '检索完成。');
  }
  if (event === 'error') status('kbSearchStatus', `检索失败（${raw(data.stage)}）：${data.message}`, true);
}

/** 按SSE行处理任意UTF-8分段及CRLF；只有complete/error是终止事件。 */
async function consumeStream(response, revision) {
  if (!response.body) throw new Error('响应没有可读取的数据流。');
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', event = 'message', dataLines = [], terminal = false;
  const line = text => {
    if (terminal) return;
    if (text === '') {
      if (dataLines.length) {
        const data = JSON.parse(dataLines.join('\n')); searchEvent(event, data, revision);
        if (event === 'complete' || event === 'error') terminal = true;
      }
      event = 'message'; dataLines = []; return;
    }
    if (text.startsWith(':')) return;
    const index = text.indexOf(':'), field = index === -1 ? text : text.slice(0, index);
    let value = index === -1 ? '' : text.slice(index + 1); if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    if (field === 'data') dataLines.push(value);
  };
  try {
    while (!terminal) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf('\n')) !== -1) {
        let text = buffer.slice(0, end); buffer = buffer.slice(end + 1);
        if (text.endsWith('\r')) text = text.slice(0, -1);
        line(text);
      }
      if (done) break;
    }
    if (!terminal) throw new Error('数据流已断开，缺少成功或失败终止帧。初步结果已保留。');
  } finally {
    try { await reader.cancel(); } catch { /* 断网后取消可能失败，不能覆盖原始终止状态。 */ }
    reader.releaseLock();
  }
}

/** 单个POST流，结束后只执行最新待处理快照，旧流事件不回退新提交状态。 */
async function runSearch(request) {
  searchFlight = true; kb('kbSearch').textContent = '提交最新检索';
  try {
    const response = await fetch('/kb/retrieval-debug', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: JSON.stringify(request.parameters) });
    if (!response.ok) throw new Error(`检索请求失败（HTTP ${response.status}）。`);
    await consumeStream(response, request.revision);
  } catch (error) {
    if (request.revision === searchRevision) status('kbSearchStatus', `检索错误：${error instanceof SyntaxError ? '数据流格式无效，初步结果已保留。' : error.message}`, true);
  } finally {
    searchFlight = false; kb('kbSearch').textContent = '检索';
    if (pendingSearch) { const next = pendingSearch; pendingSearch = null; runSearch(next); }
  }
}

/** 明确快照入队时立刻废弃旧事件；无论变化多少次仅保留一个新快照。 */
function enqueueSearch(parameters) {
  const request = { parameters, revision: ++searchRevision };
  recallData = null; searchParameters = parameters; searchMeta = `已提交问题：${parameters.query}\n请求参数：${JSON.stringify(parameters)}`; searchTiming = '';
  kb('kbRecall').textContent = ''; kb('kbRerank').textContent = parameters.rerank ? '' : '重排已关闭。'; renderSearchMeta();
  status('kbSearchStatus', searchFlight ? '已保留最新检索，等待当前请求结束。' : '正在检索…');
  if (searchFlight) pendingSearch = request; else runSearch(request);
}

/** 显式提交取消旧防抖，以本次输入成为之后调参使用的原问。 */
function submitSearch() {
  clearTimeout(debounceTimer); debounceTimer = null;
  try { const parameters = snapshot(kb('kbQuery').value.trim()); submittedQuery = parameters.query; enqueueSearch(parameters); }
  catch (error) { status('kbSearchStatus', error.message, true); }
}

/** 仅参数变化防抖600ms，不读取textarea中新草稿。 */
function parametersChanged() {
  clearTimeout(debounceTimer); debounceTimer = null;
  if (submittedQuery === null) return;
  try {
    const parameters = snapshot(submittedQuery);
    debounceTimer = setTimeout(() => { debounceTimer = null; enqueueSearch(parameters); }, 600);
  } catch (error) { status('kbSearchStatus', error.message, true); }
}

// 初始化以接口契约为准，测试和真实DOM都使用相同默认值。
kb('kbStrategy').value = 'weighted'; kb('kbDenseWeight').value = '0.5'; kb('kbSparseWeight').value = '0.5';
for (const id of limits) kb(id).value = id === 'kbRerankTopK' ? '10' : '50';
kb('kbRerankEnabled').checked = true; weights();
kb('kbRefresh').addEventListener('click', () => refreshBrowse());
kb('kbPause').addEventListener('click', () => {
  paused = !paused; kb('kbPause').textContent = paused ? '恢复自动刷新' : '暂停自动刷新'; kb('kbPause').setAttribute('aria-pressed', String(paused));
  if (!paused && !document.hidden) refreshBrowse(true); scheduleBrowse();
});
document.addEventListener('visibilitychange', () => { if (!document.hidden && !paused) refreshBrowse(true); scheduleBrowse(); });
kb('kbNext').addEventListener('click', () => { if (!hasMore) return; cursorHistory.push(cursor); cursor = nextCursor; hasMore = false; loadList(); });
kb('kbPrev').addEventListener('click', () => { if (!cursorHistory.length) return; cursor = cursorHistory.pop(); hasMore = false; loadList(); });
kb('kbLocate').addEventListener('click', locate);
kb('kbId').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); locate(); } });
kb('kbSearch').addEventListener('click', submitSearch);
kb('kbQuery').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); submitSearch(); } });
for (const id of [...limits, 'kbRerankEnabled', 'kbStrategy']) kb(id).addEventListener('change', () => { weights(); parametersChanged(); });
for (const id of limits) kb(id).addEventListener('input', parametersChanged);
for (const id of ['kbDenseWeight', 'kbSparseWeight']) kb(id).addEventListener('input', () => { weights(id); parametersChanged(); });
for (const [tab, id] of [['fused', 'kbTabFused'], ['dense', 'kbTabDense'], ['sparse', 'kbTabSparse']]) kb(id).addEventListener('click', () => { activeTab = tab; renderRecall(); });
if (!document.hidden) refreshBrowse();
scheduleBrowse();
