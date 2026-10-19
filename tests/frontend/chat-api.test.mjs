/** 聊天 SSE 客户端离线测试：使用 Node 自带的测试、响应流与断言，无 npm 依赖。 */
import test from "node:test";
import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import { failurePresentation, streamChat } from "../../app/static/chat-api.mjs";

const encoder = new TextEncoder();
const headers = { "Content-Type": "text/event-stream; charset=utf-8" };

/**
 * 将任意网络分包构造成浏览器标准 Response。
 * @param {Array<string|Uint8Array>} parts 按接收顺序排列的字节或字符串分块。
 * @returns {Response} 完全本地的 SSE 响应。
 * @throws {Error} 无主动抛出的业务异常。
 */
function makeResponse(parts) {
  return new Response(new ReadableStream({
    /** 参数 controller 为流控制器；返回 void；编码失败时抛出 TypeError。 */
    start(controller) {
      for (const part of parts) controller.enqueue(typeof part === "string" ? encoder.encode(part) : part);
      controller.close();
    },
  }), { headers });
}

/**
 * 运行本地响应并收集增量。
 * @param {Response} response 不访问网络的模拟响应。
 * @param {string[]} deltas 用于保存按顺序到达的文本。
 * @returns {Promise<string>} 客户端完整回复。
 * @throws {Error} 透传协议或 HTTP 错误供断言验证。
 */
function receive(response, deltas = []) {
  return streamChat({
    sessionId: "web-test-session",
    message: "你好",
    signal: new AbortController().signal,
    /** 参数 delta 为有效文本；返回 void；无业务异常。 */
    onDelta(delta) { deltas.push(delta); },
  }, async function fakeFetch() {
    // 无参数消费；返回本地 Response；无网络异常。
    return response;
  });
}

/** 无参数；返回 Promise<void>；请求路径、请求体或增量不符时抛出断言异常。 */
test("以 POST JSON 发送消息、沿用会话标识并申请 SSE 响应", async function requestContract() {
  const signal = new AbortController().signal;
  const deltas = [];
  const answer = await streamChat({
    sessionId: "same-session", message: "第二轮问题", signal,
    /** 参数 delta 为回复增量；返回 void；无业务异常。 */
    onDelta(delta) { deltas.push(delta); },
  }, async function inspectFetch(url, options) {
    // 参数为 fetch 地址和配置；返回 Response；契约错误时断言失败。
    assert.equal(url, "/api/chat");
    assert.equal(options.method, "POST");
    assert.equal(options.headers.Accept, "text/event-stream");
    assert.equal(options.headers["Content-Type"], "application/json");
    assert.equal(options.signal, signal);
    assert.deepEqual(JSON.parse(options.body), { session_id: "same-session", message: "第二轮问题" });
    return makeResponse(['data: {"delta":"猫"}\n\ndata: {"delta":"粮"}\n\ndata: [DONE]\n\n']);
  });
  assert.equal(answer, "猫粮");
  assert.deepEqual(deltas, ["猫", "粮"]);
});

/** 无参数；返回 Promise<void>；中文、表情或跨包 CRLF 损坏时断言失败。 */
test("逐字节网络分包仍能保留中文、表情与 CRLF 帧边界", async function splitUtf8() {
  const bytes = encoder.encode(': keepalive\r\n\r\ndata: {"delta":"你好🐱"}\r\n\r\ndata: [DONE]\r\n\r\n');
  const parts = [];
  for (const byte of bytes) parts.push(new Uint8Array([byte]));
  assert.equal(await receive(makeResponse(parts)), "你好🐱");
});

/** 无参数；返回 Promise<void>；多行 data、空块或非消息事件处理错误时断言失败。 */
test("支持多行 data 并忽略心跳、未知事件和空增量", async function multilineAndHeartbeat() {
  const deltas = [];
  const response = makeResponse([
    ': 心跳\n\nevent: ping\ndata: heartbeat\n\n',
    'data: {"delta":\ndata: "您好"}\n\ndata: {"delta":""}\n\n',
    'data: [DONE]\n\n',
  ]);
  assert.equal(await receive(response, deltas), "您好");
  assert.deepEqual(deltas, ["您好"]);
});

/** 无参数；返回 Promise<void>；客户端先缓存完整响应再通知页面时断言失败。 */
test("完成帧尚未到达时，页面已经收到首个增量", async function arrivesBeforeDone() {
  let source;
  const response = new Response(new ReadableStream({
    /** 参数 controller 为流控制器；返回 void；无业务异常。 */
    start(controller) { source = controller; },
  }), { headers });
  const deltas = [];
  const result = receive(response, deltas);
  source.enqueue(encoder.encode('data: {"delta":"首个字"}\n\n'));
  await setImmediate();
  try {
    assert.deepEqual(deltas, ["首个字"]);
  } finally {
    source.enqueue(encoder.encode('data: {"delta":"继续"}\n\ndata: [DONE]\n\n'));
    source.close();
    assert.equal(await result, "首个字继续");
  }
});

/** 无参数；返回 Promise<void>；错误帧被忽略或部分内容被丢弃时断言失败。 */
test("流中错误会失败，同时保留已经交给页面的部分回复", async function errorEvent() {
  const deltas = [];
  const response = makeResponse([
    'data: {"delta":"部分内容"}\n\n',
    'event: error\ndata: {"error":"上游模型暂时不可用"}\n\n',
  ]);
  await assert.rejects(receive(response, deltas), /上游模型暂时不可用/);
  assert.deepEqual(deltas, ["部分内容"]);
});

/** 无参数；返回 Promise<void>；错误帧的安全重试标志丢失时断言失败。 */
test("保留服务端错误的 retry_safe，同时保留已收到的工单号", async function retrySafetySignal() {
  const deltas = [];
  const response = makeResponse([
    'data: {"delta":"工单 T-123 已创建。"}\n\n',
    'event: error\ndata: {"error":"后续回复失败，请核实工单。","retry_safe":false}\n\n',
  ]);
  await assert.rejects(receive(response, deltas), {
    message: "后续回复失败，请核实工单。", retry_safe: false,
  });
  assert.deepEqual(deltas, ["工单 T-123 已创建。"]);
});

// 工单已经开始或成功时，断流和用户取消都不能鼓励再次提交原话。
for (const status of ["running", "retrying", "succeeded"]) {
  for (const stopped of [false, true]) {
    /** 无参数；返回 void；未知或成功的写操作仍被提示重试时断言失败。 */
    test(`工单 ${status} 后${stopped ? "取消" : "断流"}先核实结果且不回填`, function uncertainTicket() {
      const presentation = failurePresentation({
        error: new TypeError("连接失败，请重试。"), stopped,
        toolStatuses: [{ name: "create_ticket", status }],
      });
      assert.equal(presentation.restoreMessage, false);
      assert.match(presentation.note, /核实.*结果|结果.*核实/);
      assert.match(presentation.note, /避免重复/);
      assert.doesNotMatch(`${presentation.note}${presentation.status}${presentation.feedback}`, /重试|重新发送|新建会话/);
    });
  }
}

/** 无参数；返回 void；服务端明确不可重试但页面仍鼓励重复提交时断言失败。 */
test("服务端 retry_safe=false 优先于缺失或失败徽章", function explicitUnsafeRetry() {
  for (const toolStatuses of [[], [{ name: "create_ticket", status: "failed" }]]) {
    const presentation = failurePresentation({
      error: Object.assign(new Error("请重试。"), { retry_safe: false }),
      stopped: false, toolStatuses,
    });
    assert.equal(presentation.restoreMessage, false);
    assert.match(presentation.status, /核实/);
    assert.doesNotMatch(`${presentation.note}${presentation.feedback}`, /重试|重新发送|新建会话/);
  }
});

/** 无参数；返回 void；普通失败或确定未执行的工单被禁止手动重试时断言失败。 */
test("普通失败与已确定失败的工具仍可回填后手动重试", function ordinaryFailure() {
  for (const toolStatuses of [[], [{ name: "query_order", status: "succeeded" }],
    [{ name: "create_ticket", status: "failed" }]]) {
    const presentation = failurePresentation({
      error: new Error("服务暂时不可用。"), stopped: false, toolStatuses,
    });
    assert.equal(presentation.restoreMessage, true);
    assert.match(presentation.status, /重试/);
    assert.equal(presentation.feedback, "服务暂时不可用。");
  }
});

// 参数化覆盖非法帧：错误条件必须显式报告，不能当作正常完成。
for (const [name, frame, pattern] of [
  ["无效 JSON", 'data: not-json\n\n', /无法解析/],
  ["非文本增量", 'data: {"delta":12}\n\n', /缺少文本 delta/],
  ["空回复", 'data: [DONE]\n\n', /没有返回有效回复/],
  ["纯空白回复", 'data: {"delta":"  "}\n\ndata: [DONE]\n\n', /没有返回有效回复/],
  ["缺少完成帧", 'data: {"delta":"部分"}\n\n', /未收到完成标识/],
  ["末尾残缺帧", 'data: {"delta":"部分"}\n\ndata: [DO', /未收到完成标识/],
]) {
  /** 无参数；返回 Promise<void>；未按预期拒绝非法输入时断言失败。 */
  test(name, async function invalidFrame() {
    await assert.rejects(receive(makeResponse([frame])), pattern);
  });
}

/** 无参数；返回 Promise<void>；非 SSE 响应被接受时断言失败。 */
test("拒绝非 SSE 成功响应", async function wrongContentType() {
  await assert.rejects(receive(new Response("<html>代理页</html>")), /未返回 SSE/);
});

/** 无参数；返回 Promise<void>；HTTP 错误提示缺少状态码或详情时断言失败。 */
test("显示 HTTP 状态码，并妥善处理 JSON 和 HTML 错误响应", async function httpFailures() {
  await assert.rejects(receive(new Response(JSON.stringify({ detail: "上游暂时不可用" }), { status: 502 })), /HTTP 502：上游暂时不可用/);
  await assert.rejects(receive(new Response(JSON.stringify({ detail: [] }), { status: 422 })), /HTTP 422：消息或会话参数/);
  await assert.rejects(receive(new Response("<html>错误页</html>", { status: 500 })), /HTTP 500：请求未成功/);
});

/** 无参数；返回 Promise<void>；读到结束帧后没有释放网络流时断言失败。 */
test("收到 DONE 后取消不再需要的响应流", async function releaseReader() {
  let cancelled = false;
  const response = new Response(new ReadableStream({
    /** 参数 controller 为流控制器；返回 void；无业务异常。 */
    start(controller) {
      controller.enqueue(encoder.encode('data: {"delta":"完成"}\n\ndata: [DONE]\n\n'));
    },
    /** 无参数；返回 void；记录资源已释放，无业务异常。 */
    cancel() { cancelled = true; },
  }), { headers });
  assert.equal(await receive(response), "完成");
  assert.equal(cancelled, true);
});

/** 无参数；返回 Promise<void>；取消错误被伪装成成功时断言失败。 */
test("透传请求取消，让页面恢复输入状态", async function abortedRequest() {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(streamChat({
    sessionId: "session", message: "测试停止", signal: controller.signal,
    /** 参数 delta 未使用；返回 void；本例不应收到任何增量。 */
    onDelta(delta) { assert.fail(`意外增量：${delta}`); },
  }, async function abortedFetch(url, options) {
    // 参数为请求地址和配置；抛出取消原因；不访问网络。
    options.signal.throwIfAborted();
    return makeResponse([]);
  }), { name: "AbortError" });
});

// 状态帧尚未送达不代表服务端没有开始写入；这些传输故障必须保守提示核实。
for (const mode of ["fetch-disconnect", "early-eof", "broken-status"]) {
  /** 无参数；返回 Promise<void>；没有状态帧就鼓励重复提交时断言失败。 */
  test(`无工具徽章时 ${mode} 也不回填建单原话`, async function missingStatus() {
    await assert.rejects(streamChat({
      sessionId: "s", message: "请创建工单", signal: new AbortController().signal,
      /** 无有效增量；返回 void；收到意外文字时断言失败。 */
      onDelta() { assert.fail("不应收到文字"); },
    }, async function disconnectedFetch() {
      // 无参数；返回本地响应；模拟真实 fetch 的连接错误。
      if (mode === "fetch-disconnect") throw new TypeError("连接失败");
      return makeResponse(mode === "early-eof" ? [] : [
        'event: tool_status\ndata: {"name":"create_ticket","status":"running"}\n\n',
      ]);
    }), function checkUnsafe(error) {
      // 参数 error 为客户端异常；返回 boolean；提示可重试时断言失败。
      const presentation = failurePresentation({ error, stopped: false });
      assert.equal(presentation.restoreMessage, false);
      assert.doesNotMatch(presentation.feedback, /重试|重新发送/);
      return true;
    });
  });
}

/** 无参数；返回 Promise<void>；明确安全的服务端错误被客户端错误覆盖时断言失败。 */
test("服务端明确允许重试时保留原话回填", async function safeRetry() {
  await assert.rejects(receive(makeResponse([
    'event: error\ndata: {"error":"暂时不可用","retry_safe":true}\n\n',
  ])), function checkSafe(error) {
    // 参数 error 为服务端异常；返回 boolean；丢失明确安全状态时断言失败。
    assert.equal(failurePresentation({ error, stopped: false }).restoreMessage, true);
    return true;
  });
});
