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
 * @param {object[]} toolStatuses 用于保存每个调用按顺序到达的完整状态。
 * @returns {Promise<string>} 客户端完整回复。
 * @throws {Error} 透传协议或 HTTP 错误供断言验证。
 */
function receive(response, deltas = [], toolStatuses = []) {
  return streamChat({
    sessionId: "web-test-session",
    message: "你好",
    signal: new AbortController().signal,
    /** 参数 delta 为有效文本；返回 void；无业务异常。 */
    onDelta(delta) { deltas.push(delta); },
    /** 参数 status 为已校验工具状态；返回 void；无业务异常。 */
    onToolStatus(status) { toolStatuses.push(status); },
  }, async function fakeFetch() {
    // 无参数消费；返回本地 Response；无网络异常。
    return response;
  });
}

/**
 * 将完整工具状态编码为现有 SSE 协议，不调用服务或模型。
 * @param {object} status 包含调用 ID、工具名、状态、中文标签和演示标记的载荷。
 * @returns {string} 可与文本增量交错发送的状态帧。
 * @throws {TypeError} 载荷无法序列化时抛出。
 */
function toolStatusFrame(status) {
  return `event: tool_status\ndata: ${JSON.stringify(status)}\n\n`;
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

/** 无参数；返回 Promise<void>；工具状态丢失、乱序或正文被缓存到完成帧时断言失败。 */
test("两个工具按顺序送达状态，完成帧之前已经收到回答增量", async function arrivesBeforeDone() {
  let source;
  const response = new Response(new ReadableStream({
    /** 参数 controller 为流控制器；返回 void；无业务异常。 */
    start(controller) { source = controller; },
  }), { headers });
  const expectedStatuses = [
    { tool_call_id: "order-1", name: "query_order", status: "running", label: "正在查询", is_mock: true },
    { tool_call_id: "order-1", name: "query_order", status: "succeeded", label: "查询完成", is_mock: true },
    { tool_call_id: "logistics-1", name: "query_logistics", status: "running", label: "正在查询", is_mock: true },
    { tool_call_id: "logistics-1", name: "query_logistics", status: "succeeded", label: "查询完成", is_mock: true },
  ];
  const deltas = [];
  const toolStatuses = [];
  const result = receive(response, deltas, toolStatuses);
  try {
    // 状态先于后续工具和正文到达，页面不能等到整轮结束才显示查询进度。
    source.enqueue(encoder.encode(toolStatusFrame(expectedStatuses[0])));
    await setImmediate();
    assert.deepEqual(toolStatuses, [expectedStatuses[0]]);
    assert.deepEqual(deltas, []);
    const remaining = expectedStatuses.slice(1).map(toolStatusFrame).join("");
    source.enqueue(encoder.encode(remaining + 'data: {"delta":"首个字"}\n\n'));
    await setImmediate();
    assert.deepEqual(toolStatuses, expectedStatuses);
    assert.deepEqual(deltas, ["首个字"]);
  } finally {
    source.enqueue(encoder.encode('data: {"delta":"继续"}\n\ndata: [DONE]\n\n'));
    source.close();
    assert.equal(await result, "首个字继续");
  }
});

/** 无参数；返回 Promise<void>；同名调用被合并、ID 丢失或状态混入正文时断言失败。 */
test("同名工具的不同调用 ID 及各自重试状态完整送达页面", async function repeatedToolNames() {
  const expectedStatuses = [
    { tool_call_id: "product-a", name: "query_product", status: "running", label: "正在查询", is_mock: true },
    { tool_call_id: "product-a", name: "query_product", status: "succeeded", label: "查询完成", is_mock: true },
    { tool_call_id: "product-b", name: "query_product", status: "running", label: "正在查询", is_mock: true },
    { tool_call_id: "product-b", name: "query_product", status: "retrying", label: "正在重试", is_mock: true },
    { tool_call_id: "product-b", name: "query_product", status: "failed", label: "查询失败", is_mock: true },
  ];
  const deltas = [];
  const toolStatuses = [];
  const answer = await receive(makeResponse([
    ...expectedStatuses.map(toolStatusFrame),
    'data: {"delta":"第二件商品暂时查不到。"}\n\ndata: [DONE]\n\n',
  ]), deltas, toolStatuses);
  assert.deepEqual(toolStatuses, expectedStatuses);
  assert.deepEqual(deltas, ["第二件商品暂时查不到。"]);
  assert.equal(answer, "第二件商品暂时查不到。");
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

// 使用真实解码器接收已建单状态和回执，再让后续只读操作失败或中断。
for (const mode of ["只读故障", "连接提前结束", "用户停止"]) {
  /** 无参数；返回 Promise<void>；后续查询覆盖已建单事实、丢失回执或允许回填时断言失败。 */
  test(`工单回执后${mode}仍保留编号且不回填建单原话`, async function failureAfterTicketAndQuery() {
    const controller = new AbortController();
    let source;
    const response = new Response(new ReadableStream({
      /** 参数 stream 为响应流控制器；返回 void；无业务异常。 */
      start(stream) { source = stream; },
    }), { headers });
    const deltas = [];
    const latestStatuses = new Map();
    const result = streamChat({
      sessionId: "ticket-query-session", message: "请创建工单并查订单", signal: controller.signal,
      /** 参数 delta 为已接收文本；返回 void；无业务异常。 */
      onDelta(delta) { deltas.push(delta); },
      /** 参数 status 为调用最新状态；返回 void；按页面约定保留各调用。 */
      onToolStatus(status) { latestStatuses.set(status.tool_call_id, status); },
    }, async function fakeFetch(url, options) {
      // 参数为请求地址和配置；返回本地流；用响应流错误模拟 fetch 收到取消信号。
      options.signal.addEventListener("abort", function abortResponse() {
        // 无参数；返回 void；将真实 AbortError 传给等待读取的客户端。
        source.error(options.signal.reason);
      }, { once: true });
      return response;
    });
    const rejected = assert.rejects(result, function checkFailure(error) {
      // 参数 error 为客户端异常；返回 boolean；只读失败不能清除先前工单的防重复保护。
      if (mode === "用户停止") assert.equal(error.name, "AbortError");
      else assert.match(error.message, mode === "只读故障" ? /订单查询失败/ : /未收到完成标识/);
      if (mode === "只读故障") assert.equal(error.retry_safe, true);
      const presentation = failurePresentation({
        error, stopped: controller.signal.aborted, toolStatuses: latestStatuses.values(),
      });
      assert.equal(presentation.restoreMessage, false);
      assert.match(presentation.note, /避免重复/);
      assert.doesNotMatch(`${presentation.note}${presentation.status}${presentation.feedback}`, /重试|重新发送|新建会话/);
      return true;
    });
    try {
      source.enqueue(encoder.encode([
        toolStatusFrame({ tool_call_id: "ticket-1", name: "create_ticket", status: "running", label: "正在提交", is_mock: false }),
        toolStatusFrame({ tool_call_id: "ticket-1", name: "create_ticket", status: "succeeded", label: "提交完成", is_mock: false }),
        'data: {"delta":"工单 T-123 已创建。"}\n\n',
        toolStatusFrame({ tool_call_id: "order-2", name: "query_order", status: "running", label: "正在查询", is_mock: true }),
      ].join("")));
      await setImmediate();
      // 后续查询仍未结束时，工单号已经交给页面，且成功工单与新查询独立保留。
      assert.deepEqual(deltas, ["工单 T-123 已创建。"]);
      assert.equal(latestStatuses.get("ticket-1").status, "succeeded");
      assert.equal(latestStatuses.get("order-2").status, "running");
      if (mode === "用户停止") controller.abort();
      else {
        if (mode === "只读故障") {
          // 即使后续只读错误单独声称可重试，页面仍须顾及之前已提交的工单。
          source.enqueue(encoder.encode(
            toolStatusFrame({ tool_call_id: "order-2", name: "query_order", status: "failed", label: "查询失败", is_mock: true }) +
            'event: error\ndata: {"error":"订单查询失败，请重试。","retry_safe":true}\n\n'));
        }
        source.close();
      }
      await rejected;
      assert.deepEqual(deltas, ["工单 T-123 已创建。"]);
      assert.equal(latestStatuses.get("ticket-1").status, "succeeded");
      assert.equal(latestStatuses.get("order-2").status, mode === "只读故障" ? "failed" : "running");
    } finally {
      // 断言失败也关闭开放流，避免测试留下等待读取的请求。
      controller.abort();
      await rejected;
    }
  });
}

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
test("普通聊天 retry_safe=false 不能被误说成工单已提交", function explicitUnsafeRetry() {
  for (const toolStatuses of [[], [{ name: "create_ticket", status: "failed" }]]) {
    const presentation = failurePresentation({
      error: Object.assign(new Error("请重试。"), { retry_safe: false }),
      stopped: false, toolStatuses,
    });
    assert.equal(presentation.restoreMessage, true);
    assert.doesNotMatch(`${presentation.note}${presentation.status}${presentation.feedback}`, /工单/);
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

// 普通聊天不执行写工具；传输故障不能误报工单可能提交。
for (const mode of ["fetch-disconnect", "early-eof", "broken-status"]) {
  /** 无参数；返回 Promise<void>；没有状态帧就鼓励重复提交时断言失败。 */
  test(`普通聊天 ${mode} 可回填且不误报建单`, async function missingStatus() {
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
      assert.equal(presentation.restoreMessage, true);
      assert.doesNotMatch(`${presentation.note}${presentation.status}${presentation.feedback}`, /工单/);
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

const validAction = { action_id: "00000000-0000-5000-8000-000000000001", kind: "create_ticket",
  source_turn_id: "old-turn", question: "原问题", ticket_type: "投诉" };
const completion = { session_id: "old-session", turn_id: "old-turn", status: "completed", citations: [], suggested_actions: [validAction] };
/** 参数：完成载荷；返回：SSE帧；异常：序列化错误。 */
function metadataFrame(value) { return `event: answer_metadata\ndata: ${JSON.stringify(value)}\n\n`; }

test("建议仅在DONE后交付并绑定本次原始用户，旧元数据缺动作时归为空", async () => {
  let source;
  const seen = [];
  const response = new Response(new ReadableStream({ start(controller) { source = controller; } }), { headers });
  const promise = streamChat({ sessionId: "old-session", userId: "original-user", message: "原问题", onDelta() {},
    onAnswerMetadata: (value) => seen.push(value) }, async () => response);
  source.enqueue(encoder.encode('data: {"delta":"完成回复"}\n\n' + metadataFrame(completion)));
  await setImmediate();
  assert.deepEqual(seen, []);
  source.enqueue(encoder.encode("data: [DONE]\n\n")); source.close();
  assert.equal(await promise, "完成回复");
  assert.equal(seen[0].user_id, "original-user");
  assert.deepEqual(seen[0].suggested_actions, [validAction]);
  const old = { ...completion }; delete old.suggested_actions;
  await streamChat({ sessionId: "old-session", message: "原问题", onDelta() {}, onAnswerMetadata: (value) => seen.push(value) },
    async () => makeResponse(['data: {"delta":"旧回复"}\n\n' + metadataFrame(old) + "data: [DONE]\n\n"]));
  assert.deepEqual(seen[1].suggested_actions, []);
});

for (const [name, suggested] of [
  ["非数组", null], ["未知类型", [{ ...validAction, kind: "other" }]],
  ["重复动作", [validAction, validAction]], ["不规范UUID", [{ ...validAction, action_id: validAction.action_id.toUpperCase().replace("00000001", "ABCDEF01") }]],
  ["来源不符", [{ ...validAction, source_turn_id: "new-turn" }]], ["不是原问题", [{ ...validAction, question: "另一个问题" }]],
  ["人工带类型", [{ ...validAction, kind: "handoff" }]], ["建单无类型", [{ ...validAction, ticket_type: null }]],
  ["附加链接", [{ ...validAction, url: "https://example.com" }]], ["超长", [{ ...validAction, question: "字".repeat(4001) }]],
]) {
  test(`非法SSE动作${name}不得留下可执行建议`, async () => {
    const seen = [];
    await assert.rejects(streamChat({ sessionId: "old-session", message: "原问题", onDelta() {}, onAnswerMetadata: (value) => seen.push(value) },
      async () => makeResponse(['data: {"delta":"回复"}\n\n' + metadataFrame({ ...completion, suggested_actions: suggested }) + "data: [DONE]\n\n"])), /建议|完成信息/);
    assert.deepEqual(seen, []);
  });
}

for (const ending of ["", 'event: error\ndata: {"error":"失败"}\n\n']) {
  test(`有效动作后${ending ? "错误" : "断流"}不提前交付`, async () => {
    const seen = [];
    await assert.rejects(streamChat({ sessionId: "old-session", message: "原问题", onDelta() {}, onAnswerMetadata: (value) => seen.push(value) },
      async () => makeResponse(['data: {"delta":"部分回复"}\n\n' + metadataFrame(completion) + ending])));
    assert.deepEqual(seen, []);
  });
}

test("收到同包正文时取消，后续完成帧也不能交付可执行建议", async () => {
  const seen = [];
  const controller = new AbortController();
  await assert.rejects(streamChat({ sessionId: "old-session", message: "原问题", signal: controller.signal,
    onDelta() { controller.abort(); }, onAnswerMetadata: (value) => seen.push(value) },
    async () => makeResponse(['data: {"delta":"部分回复"}\n\n' + metadataFrame(completion) + "data: [DONE]\n\n"])), { name: "AbortError" });
  assert.deepEqual(seen, []);
});
