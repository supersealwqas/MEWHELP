/** 评估页行为回归：执行页面同源模块，仅替换DOM、网络与定时器边界。 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { setImmediate } from "node:timers/promises";

const script = readFileSync(new URL("../../app/static/rag-eval.mjs", import.meta.url), "utf8");
const OVERVIEW = "/api/rag-eval/overview";
const JOB = "/api/rag-eval/job";
const RERUN = "/api/rag-eval/rerun";

/** 最小DOM保留真实事件分发与文本渲染；不模拟布局或浏览器安全策略。 */
class Element extends EventTarget {
  /** 建立节点；参数：无；返回：Element；异常：无主动抛出。 */
  constructor() { super(); Object.assign(this, { children: [], style: {}, dataset: {}, disabled: false, className: "", _text: "" }); }
  /** 替换文本并清空后代；参数：value文本；返回：无；异常：无。 */
  set textContent(value) { this._text = String(value); this.children = []; }
  /** 读取本节点与后代文本；参数：无；返回：字符串；异常：非法后代引发TypeError。 */
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  /** 追加节点；参数：node子节点；返回：子节点；异常：无主动抛出。 */
  appendChild(node) { this.children.push(node); return node; }
  /** 模拟启用按钮点击；参数：无；返回：无；异常：事件错误由EventTarget报告。 */
  click() { if (!this.disabled) this.dispatchEvent(new Event("click")); }
}

/** 建立可控制完成顺序的真实Promise；参数：无；返回：Promise与完成函数；异常：无。 */
function deferred() {
  let resolve, reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

/** 构造后端任务字段；参数：status状态，message显示信息；返回：任务对象；异常：无。 */
function job(status, message = status) { return { status, message, log: `日志：${message}` }; }

/** 构造有成绩的看板响应；参数：status任务状态，run运行名；返回：完整看板字段；异常：无。 */
function overview(status, run = "old-run") {
  return { present: true, hint: "", job: job(status), meta: { run, sample_count: 1, chunk_count: 1, model: "offline", judge_model: "offline", elapsed_seconds: 1, source_revision: "test", error_total: 0 },
    best: { name: "dense", label: "Dense", mrr: 0.8, coverage: 1 }, calibration: { case_count: 1, thresholds: {} },
    strategies: ["dense"], strategy_labels: { dense: "Dense" }, buckets: [], bucket_labels: {},
    by_bucket: { dense: {} }, by_difficulty: { dense: {} }, overall: { dense: {} } };
}

/** 执行生产脚本；参数：无；返回：可观察DOM、请求队列和手动时钟；异常：脚本或断言错误。 */
function setup() {
  const nodes = new Map(), requests = [], timers = new Map();
  let timerId = 0;
  const document = {
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); },
    createElement() { const node = new Element(); node.classList = { add(name) { node.className += ` ${name}`; } }; return node; },
    createTextNode(text) { const node = new Element(); node.textContent = text; return node; },
    querySelectorAll() { return []; },
  };
  const context = vm.createContext({ document,
    fetch(path, options) {
      const pending = deferred();
      requests.push({ path, options, ...pending });
      return pending.promise;
    },
    setTimeout(callback, delay) { const id = ++timerId; timers.set(id, { callback, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
  });
  vm.runInContext(script, context, { filename: "app/static/rag-eval.mjs" });
  return { requests, timers, node: document.getElementById,
    async respond(index, body, status = 200) {
      assert.ok(requests[index], `缺少第${index + 1}个网络请求`);
      requests[index].resolve(Response.json(body, { status }));
      await setImmediate();
    },
    async fail(index) { requests[index].reject(new TypeError("网络断开")); await setImmediate(); },
    async tick() {
      assert.equal(timers.size, 1, "页面必须仅保留一个轮询定时器");
      const [id, timer] = timers.entries().next().value;
      timers.delete(id); timer.callback(); await setImmediate();
    },
  };
}

// 去掉load完成后的轮询会使首开/刷新恢复测试失败；终态不刷新会使运行目录仍是旧成绩。
for (const present of [false, true]) {
  test(`首次打开延迟返回running（${present ? "已有" : "没有"}成绩）后持续轮询并刷新新成绩`, async () => {
    const f = setup();
    assert.equal(f.requests.length, 1);
    assert.equal(f.timers.size, 0);
    await f.respond(0, present ? overview("running") : { present: false, hint: "等待首轮成绩", job: job("running") });
    assert.equal(f.node("jobPill").textContent, "运行中");
    assert.equal(f.node("rerunBtn").disabled, true);
    assert.equal(f.requests[1]?.path, JOB, "取得running后必须恢复任务轮询");
    await f.respond(1, job("running", "比较中"));
    assert.equal(f.node("jobMsg").textContent, "比较中");
    await f.tick();
    assert.equal(f.requests[2]?.path, JOB);
    await f.respond(2, job("done"));
    assert.equal(f.requests[3]?.path, OVERVIEW, "终态必须重新读取成绩");
    await f.respond(3, overview("done", "new-run"));
    assert.match(f.node("metaRow").textContent, /new-run/);
    assert.equal(f.node("jobPill").textContent, "已完成");
    assert.equal(f.node("rerunBtn").disabled, false);
    assert.equal(f.timers.size, 0);
  });
}

// 手动刷新或重复发现running不能增加并行轮询；旧job响应不能覆盖刷新发现的新终态。
test("手动刷新发现running，重复刷新只保留一个轮询链，旧job不覆盖新终态", async () => {
  const f = setup();
  await f.respond(0, overview("idle"));
  f.node("refreshBtn").click();
  await f.respond(1, overview("running"));
  assert.equal(f.requests[2]?.path, JOB);
  f.node("refreshBtn").click();
  await f.respond(3, overview("running"));
  assert.equal(f.requests.filter((request) => request.path === JOB).length, 1, "旧轮询未结束时不得再发一个job请求");
  await f.respond(2, job("running", "旧日志"));
  assert.equal(f.timers.size, 1);
  await f.tick();
  f.node("refreshBtn").click();
  await f.respond(5, overview("done", "manual-new-run"));
  await f.respond(4, job("running", "过期响应"));
  assert.equal(f.node("jobPill").textContent, "已完成");
  assert.equal(f.node("rerunBtn").disabled, false);
  assert.match(f.node("metaRow").textContent, /manual-new-run/);
  assert.equal(f.timers.size, 0);
});

// 去掉overview请求序号校验会让后返回的旧请求回退最新成绩和任务状态。
test("两个overview乱序返回时旧成绩与running状态不能覆盖最新刷新", async () => {
  const f = setup();
  f.node("refreshBtn").click();
  await f.respond(1, overview("done", "latest-run"));
  await f.respond(0, overview("running", "stale-run"));
  assert.match(f.node("metaRow").textContent, /latest-run/);
  assert.doesNotMatch(f.node("metaRow").textContent, /stale-run/);
  assert.equal(f.node("jobPill").textContent, "已完成");
  assert.equal(f.node("rerunBtn").disabled, false);
  assert.equal(f.timers.size, 0);
  assert.equal(f.requests.filter((request) => request.path === JOB).length, 0);
});

// 去掉错误分支重试会永久停在running；终态后继续调度会重复请求并锁住按钮。
for (const status of ["done", "failed", "interrupted"]) {
  test(`轮询网络错误后重试，${status}终态刷新数据且停止定时器`, async () => {
    const f = setup();
    await f.respond(0, overview("idle"));
    f.node("rerunBtn").click();
    assert.equal(f.requests[1].path, RERUN);
    assert.equal(f.requests[1].options.method, "POST");
    await f.respond(1, job("running"));
    assert.equal(f.requests[2]?.path, JOB);
    await f.fail(2);
    await f.tick();
    assert.equal(f.requests[3]?.path, JOB);
    await f.respond(3, job(status));
    assert.equal(f.requests[4]?.path, OVERVIEW);
    await f.respond(4, overview(status, "terminal-run"));
    assert.equal(f.node("rerunBtn").disabled, false);
    assert.match(f.node("metaRow").textContent, /terminal-run/);
    assert.equal(f.timers.size, 0);
  });
}

// 去掉启动中的锁会让并发overview重启按钮，从而允许重复POST；旧overview不能盖掉轮询完成状态。
test("刷新与重跑并发时保持按钮锁，旧overview不覆盖完成状态且仅一个轮询", async () => {
  const f = setup();
  await f.respond(0, overview("idle"));
  f.node("refreshBtn").click();
  f.node("rerunBtn").click();
  f.node("refreshBtn").click();
  await f.respond(3, overview("idle"));
  assert.equal(f.node("rerunBtn").disabled, true, "POST尚未完成时刷新不能解锁重跑按钮");
  f.node("rerunBtn").click();
  assert.equal(f.requests.filter((request) => request.path === RERUN).length, 1);
  await f.respond(2, job("running"));
  assert.equal(f.requests[4]?.path, JOB);
  await f.respond(4, job("done"));
  await f.respond(5, overview("done", "rerun-new-run"));
  await f.respond(1, overview("running", "before-rerun"));
  assert.equal(f.node("jobPill").textContent, "已完成");
  assert.equal(f.node("rerunBtn").disabled, false);
  assert.match(f.node("metaRow").textContent, /rerun-new-run/);
  assert.equal(f.timers.size, 0);
});

// 启动成功也必须废弃提交期间发出的overview，否则其idle会中止刚开始的轮询。
test("重跑提交期间的刷新晚于启动成功返回，不能解锁按钮或中止新任务轮询", async () => {
  const f = setup();
  await f.respond(0, overview("idle"));
  f.node("rerunBtn").click();
  f.node("refreshBtn").click();
  await f.respond(1, job("running"));
  assert.equal(f.requests[3]?.path, JOB);
  await f.respond(2, overview("idle", "before-start"));
  assert.equal(f.node("rerunBtn").disabled, true, "启动前的overview不能解锁已启动任务");
  await f.respond(3, job("running", "新任务运行中"));
  assert.equal(f.node("jobPill").textContent, "运行中");
  assert.equal(f.node("jobMsg").textContent, "新任务运行中");
  assert.equal(f.timers.size, 1);
  await f.tick();
  assert.equal(f.requests[4]?.path, JOB, "启动后的轮询必须继续");
});

// running刷新不得再创建定时器；正在读取的overview不能回退轮询已取得的新日志。
test("已有轮询定时器时重复刷新不重复请求，旧overview不回退新日志", async () => {
  const f = setup();
  await f.respond(0, overview("running"));
  await f.respond(1, job("running", "第一阶段"));
  f.node("refreshBtn").click();
  await f.respond(2, overview("running"));
  assert.equal(f.requests.filter((request) => request.path === JOB).length, 1);
  assert.equal(f.timers.size, 1);
  f.node("refreshBtn").click();
  await f.tick();
  await f.respond(4, job("running", "第二阶段"));
  await f.respond(3, overview("running"));
  assert.equal(f.node("jobMsg").textContent, "第二阶段");
  assert.equal(f.timers.size, 1);
});

// 去掉启动失败处理会让按钮永久不可用；此处还通过第二次真实点击证明可以恢复重跑。
test("重跑启动失败显示原因、恢复按钮，并允许再次成功启动", async () => {
  const f = setup();
  await f.respond(0, overview("idle"));
  f.node("rerunBtn").click();
  await f.respond(1, { detail: "后台不可用" }, 503);
  assert.match(f.node("jobMsg").textContent, /启动失败.*后台不可用/);
  assert.equal(f.node("rerunBtn").disabled, false);
  assert.equal(f.timers.size, 0);
  f.node("rerunBtn").click();
  await f.respond(2, job("running"));
  assert.equal(f.requests[3]?.path, JOB);
  await f.respond(3, job("running"));
  assert.equal(f.node("rerunBtn").disabled, true);
  assert.equal(f.timers.size, 1);
});
