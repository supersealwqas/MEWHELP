/** 本人会话只读浏览：分页原文、切换代次与请求互斥，不缓存正文、不执行业务动作。 */

/** 验证精确无符号BIGINT字符串；参数value；返回boolean；异常无。 */
function validId(value) {
  return typeof value === "string" && /^[1-9][0-9]{0,19}$/.test(value) && BigInt(value) <= 18446744073709551615n;
}

/** 创建可测试的浏览控制器；参数含fetch与同步视图回调；返回控制接口；异常由异步入口转为false。 */
export function createConversationBrowser({ fetch: fetchImpl = globalThis.fetch, userId = "demo-user", onSelect = () => {},
  onChange = () => {}, onLoading = () => {} } = {}) {
  let items = [], nextBefore = null, sessionId = "", busy = false, listEpoch = 0, historyEpoch = 0;
  let listing = false, loading = false, historyController = null;
  /** 返回只读副本；参数无；返回状态；异常无。 */
  function snapshot() { return { items: structuredClone(items), nextBefore, sessionId, busy, listing, loading }; }
  /** 通知侧栏更新；参数无；返回void；异常：视图编程错误。 */
  function changed() { onChange(snapshot()); }
  /** 撤销旧历史回载，旧响应不能提交；参数无；返回void；异常无。 */
  function invalidate() {
    historyEpoch++; historyController?.abort(); historyController = null;
    loading = false; onLoading(false);
  }
  /** 只读请求并限制网络等待；参数URL和可选控制器；返回JSON；异常：网络、超时或响应错误。 */
  async function get(url, controller = new AbortController()) {
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetchImpl(url, { method: "GET", signal: controller.signal });
      if (!response.ok) throw new Error("历史读取失败");
      return await response.json();
    } finally { clearTimeout(timer); }
  }
  /** 读取列表一页，只有最新请求能替换列表；参数是否追加；返回成功与否；异常已隔离。 */
  async function loadList(append) {
    if (append && (listing || nextBefore === null)) return false;
    const epoch = ++listEpoch, before = append ? nextBefore : null;
    listing = true; changed();
    try {
      const params = new URLSearchParams({ user_id: userId, limit: "50" });
      if (before !== null) params.set("before_id", before);
      const page = await get(`/api/conversations?${params}`);
      if (epoch !== listEpoch) return false;
      if (!Array.isArray(page.items) || !page.items.every(row => validId(row.id) && typeof row.session_id === "string"
          && typeof row.preview === "string" && typeof row.has_summary === "boolean")
          || !(page.next_before_id === null || validId(page.next_before_id))) throw new Error("会话列表格式错误");
      let last = before === null ? null : BigInt(before);
      for (const row of page.items) {
        if (last !== null && BigInt(row.id) >= last) throw new Error("列表游标没有前进");
        last = BigInt(row.id);
      }
      if (page.next_before_id !== null && page.next_before_id !== page.items.at(-1)?.id) throw new Error("列表游标不一致");
      items = append ? [...items, ...page.items] : page.items;
      nextBefore = page.next_before_id;
      return true;
    } catch { return false; } // 侧栏读取失败不清空旧列表、不干扰当前聊天。
    finally { if (epoch === listEpoch) { listing = false; changed(); } }
  }
  /** 完整取回历史后才提交切换；参数会话条目；返回成功与否；异常隔离，保留原会话。 */
  async function select(conversation) {
    if (busy || !validId(conversation?.id) || typeof conversation.session_id !== "string") return false;
    invalidate();
    const epoch = historyEpoch, controller = new AbortController();
    historyController = controller; loading = true; onLoading(true); changed();
    try {
      const messages = []; let after = "0";
      do {
        const params = new URLSearchParams({ user_id:userId, after_id:after, limit:"100" });
        const page = await get(`/api/conversations/${conversation.id}/messages?${params}`, controller);
        if (epoch !== historyEpoch || busy) return false;
        if (page.session_id !== conversation.session_id || !Array.isArray(page.items)
            || !(page.next_after_id === null || validId(page.next_after_id))) throw new Error("历史页目标错误");
        let last = BigInt(after);
        for (const row of page.items) {
          if (!validId(row.id) || BigInt(row.id) <= last || !["user","assistant","tool"].includes(row.role)
              || !(row.content === null || typeof row.content === "string")) throw new Error("历史记录格式错误");
          last = BigInt(row.id);
        }
        if (page.next_after_id !== null && (page.next_after_id !== page.items.at(-1)?.id || BigInt(page.next_after_id) <= BigInt(after)))
          throw new Error("历史游标没有前进");
        messages.push(...page.items); after = page.next_after_id;
      } while (after !== null);
      if (epoch !== historyEpoch || busy) return false;
      onSelect({ conversation: structuredClone(conversation), messages });
      sessionId = conversation.session_id;
      return true;
    } catch { return false; }
    finally { if (epoch === historyEpoch) { loading = false; historyController = null; onLoading(false); changed(); } }
  }
  /** 禁止回复/动作期间切换，并使之前的回载失效；参数布尔值；返回void；异常无。 */
  function setBusy(value) { const entered = value && !busy; busy = Boolean(value); if (entered) invalidate(); changed(); }
  /** 只切换本地空会话标识，列表不删、后端不写；参数新会话标识；返回是否成功；异常无。 */
  function startNew(value) { if (busy) return false; invalidate(); sessionId = value; changed(); return true; }
  /** 从分页列表找到旧session再回载；参数会话标识；返回是否恢复；异常在内部隔离。 */
  async function restore(value) {
    if (busy) return false;
    invalidate();
    const epoch = historyEpoch;
    // 查找旧session也是恢复的一部分；普通侧栏refresh不进入此输入锁。
    loading = true; onLoading(true); changed();
    try {
      if (!await loadList(false) || epoch !== historyEpoch || busy) return false;
      while (!items.some(row => row.session_id === value) && nextBefore !== null) {
        if (!await loadList(true) || epoch !== historyEpoch || busy) return false;
      }
      const found = items.find(row => row.session_id === value);
      return found ? await select(found) : false;
    } finally {
      // select或新建已接管代次时，由它们释放锁，旧恢复不得解锁后来的请求。
      if (epoch === historyEpoch) { loading = false; onLoading(false); changed(); }
    }
  }
  return { snapshot, refresh: () => loadList(false), more: () => loadList(true), select, startNew, setBusy, restore };
}
