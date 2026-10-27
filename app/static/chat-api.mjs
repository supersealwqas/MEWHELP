/**
 * 聊天接口客户端：以 POST 发送会话消息，增量解码 UTF-8/SSE，并校验完成帧。
 * 本模块不依赖 DOM，便于使用 Node 内置测试验证分包、错误与中断行为。
 */

/**
 * 将单个 SSE 帧拆为事件名和数据，忽略注释及无关字段。
 * @param {string} frame 不含末尾空行的完整事件帧，支持 LF 和 CRLF。
 * @returns {{event: string, data: string}} 事件名与多行合并的数据。
 * @throws {Error} 无主动抛出的业务异常。
 */
function parseEvent(frame) {
  let event = "message";
  const data = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return { event, data: data.join("\n") };
}

/**
 * 将 HTTP 失败转换为用户可读提示，不展示整段服务器响应或 HTML。
 * @param {Response} response 非成功状态的接口响应。
 * @returns {Promise<string>} 包含 HTTP 状态码的错误提示。
 * @throws {Error} 响应解析失败在内部处理，不影响后续错误展示。
 */
async function httpErrorMessage(response) {
  let detail = "请求未成功，请稍后重试。";
  try {
    const body = await response.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (typeof body.detail?.message === "string") detail = body.detail.message;
    else if (response.status === 422) detail = "消息或会话参数不符合接口要求。";
  } catch {
    // 代理返回 HTML 或空响应时只显示状态码，不把原始页面塞入聊天界面。
  }
  return `HTTP ${response.status}：${detail}`;
}

/**
 * 普通聊天失败可重发；仅为明确收到的旧版写工具状态保留防重复兼容提示。
 * @param {object} options 本轮失败、取消及各调用的最新状态。
 * @param {Error} options.error 接口错误；普通聊天不会因断流推断工单写入。
 * @param {boolean} options.stopped 是否由用户停止接收。
 * @param {Iterable<object>} options.toolStatuses 各工具调用最新状态。
 * @returns {{restoreMessage: boolean, note: string, status: string, feedback: string}} 页面提示及原话回填决策。
 * @throws {Error} 无主动抛出的业务异常。
 */
export function failurePresentation({ error, stopped, toolStatuses = [] }) {
  const ticketMayExist = Array.from(toolStatuses).some(function hasPendingOrSavedTicket(tool) {
    // 参数 tool 为调用最新状态；返回 boolean；无业务异常。failed 表示服务端已确定未执行。
    return tool.name === "create_ticket" && ["running", "retrying", "succeeded"].includes(tool.status);
  });
  if (ticketMayExist) {
    return {
      restoreMessage: false,
      note: "工单可能已提交，请先核实已提交结果，避免重复建单。已收到的文字和工单号会保留在本页。",
      status: stopped ? "回复已停止，请先核实工单结果" : "回复未完成，请先核实工单结果",
      // 网络错误的通用文案可能带有重试建议，这里统一采用适用于写操作的提示。
      feedback: "请保留本页已收到的工单号和回复，核实提交结果后再继续。",
    };
  }
  return {
    restoreMessage: true,
    note: stopped
      ? "已停止接收；是否保存本轮上下文，以服务端实际完成情况为准。"
      : "回复未完成；请稍后重试。",
    status: stopped ? "已停止，可编辑问题后重新发送" : "发送未完成，请稍后重试",
    feedback: error instanceof TypeError
      ? "连接失败，请确认后端服务仍在运行，再重新发送。"
      : (error?.message || "请求失败，请稍后重试。"),
  };
}

/**
 * 发起一轮聊天，在每个有效增量帧到达时立即通知调用方。
 * @param {object} options 当前轮请求参数。
 * @param {string} options.sessionId 当前页面内保持不变的会话标识。
 * @param {string} options.message 非空用户消息。
 * @param {AbortSignal} options.signal 用于手动停止请求的取消信号。
 * @param {(delta: string) => void} options.onDelta 增量文本回调，不接收 HTML。
 * @param {(status: object) => void} [options.onToolStatus] 工具状态回调，与回答正文分离。
 * @param {typeof fetch} fetcher 默认使用浏览器 fetch；测试可注入离线替身。
 * @returns {Promise<string>} 收到 [DONE] 后返回本轮完整回复。
 * @throws {Error} HTTP 失败、SSE 错误、非法数据、空回复或缺少完成帧时抛出。
 * @throws {DOMException} 请求被取消时透传 AbortError。
 */
export async function streamChat(options, fetcher = globalThis.fetch) {
  if (options.signal?.aborted) {
    // 普通聊天不执行建单，取消和断流不会代表工单提交。
    const error = new DOMException("请求已取消", "AbortError");
    error.retry_safe = true;
    throw error;
  }
  try {
    return (await receiveChatStream(options, fetcher)).answer;
  } catch (error) {
    // 新工作流只建议动作；写入结果未知由独立确认API处理。
    if (typeof error.retry_safe !== "boolean") error.retry_safe = true;
    throw error;
  }
}

/**
 * 发送普通聊天并解码响应；只在未取消且收到DONE后交付完成元数据。
 * @param {object} options 会话、消息、取消信号和增量回调。
 * @param {typeof fetch} fetcher 浏览器 fetch 或离线网络替身。
 * @returns {Promise<string>} 收到完成帧后的完整回答。
 * @throws {Error} HTTP、网络或协议错误；取消时抛出 AbortError。
 */
async function receiveChatStream(options, fetcher) {
  const { sessionId, userId = "demo-user", message, signal } = options;
  // EventSource 不支持此处需要的 POST JSON，因此使用 fetch 的可读响应流。
  const response = await fetcher("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ session_id: sessionId, message, ...(userId === "demo-user" ? {} : { user_id: userId }) }),
    signal,
  });
  return receiveWorkflowStream(response, options);
}

/** 发送支持等待的新工作流；参数为冻结身份、问题与回调；返回三种终态对象；异常：网络/协议错误。 */
export async function streamWorkflowChat(options, fetcher = globalThis.fetch) {
  if (options.signal?.aborted) throw new DOMException("请求已取消", "AbortError");
  return receiveChatStream(options, fetcher);
}

/** 解码工作流流；参数为Response及冻结身份/原轮/回调；返回completed、awaiting_input或accepted；异常：非法帧或无终态EOF。 */
export async function receiveWorkflowStream(response, { sessionId, userId = "demo-user", message, sourceTurnId,
  signal, onDelta, onToolStatus, onAnswerMetadata, onStage, onSelectOrder } = {}) {
  if (!response.ok) {
    const error = new Error(await httpErrorMessage(response));
    // 参数校验失败发生在业务执行前，用户修改输入后可以重新发送。
    if (response.status === 422) error.retry_safe = true;
    throw error;
  }
  if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body) {
    throw new Error("服务器未返回 SSE 数据流，请确认正在访问本项目的聊天接口。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let answer = "";
  let metadata = null;
  let waiting = null;
  let accepted = null;
  try {
    while (true) {
      const { value, done } = await reader.read();
      // stream:true 保留跨网络分包的中文多字节字符；EOF 时再冲刷解码器。
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
        // 取消可能发生在正文回调期间；同一个已缓存分包中的DONE也不能继续装配按钮。
        if (signal?.aborted) throw signal.reason ?? new DOMException("请求已取消", "AbortError");
        const frame = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const event = parseEvent(frame);
        if (!event.data || !["message", "error", "tool_status", "answer_metadata", "workflow_stage", "select_order", "action_status"].includes(event.event)) continue;
        if (event.event === "message" && event.data === "[DONE]") {
          if (waiting || accepted) throw new Error("工作流终态冲突，请查询当前处理状态。");
          if (!answer.trim()) throw new Error("模型没有返回有效回复，请重新提问。");
          // 兼容旧服务的DONE；仅含新完成元数据的回答提供引用和反馈入口。
          if (metadata) onAnswerMetadata?.(metadata);
          return { status: "completed", answer, metadata };
        }

        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          throw new Error("收到无法解析的 SSE 数据，请检查接口响应。");
        }
        if (event.event === "error") {
          const error = new Error(typeof payload?.error === "string" ? payload.error : "模型暂时不可用，请稍后重试。");
          if (typeof payload?.retry_safe === "boolean") error.retry_safe = payload.retry_safe;
          error.status = payload?.status_code;
          error.outcome = payload?.outcome;
          throw error;
        }
        if (event.event === "workflow_stage") {
          if (!["understand", "classify", "order", "expand", "policy", "answer"].includes(payload?.stage)) continue;
          if (!validText(payload.source_turn_id, 36) || !["running", "completed"].includes(payload.status) ||
              !validText(payload.label, 200) || (sourceTurnId && payload.source_turn_id !== sourceTurnId)) {
            throw new Error("处理阶段与原轮不一致。");
          }
          onStage?.(payload);
          continue;
        }
        if (event.event === "select_order") {
          if (waiting || accepted || metadata) throw new Error("选单终态重复或冲突。");
          waiting = validateOrderSelection(payload, sourceTurnId);
          continue;
        }
        if (event.event === "action_status") {
          if (waiting || accepted || metadata || payload?.status !== "accepted" || !UUID_V5.test(payload.action_id) ||
              !validText(payload.source_turn_id, 36) || (sourceTurnId && payload.source_turn_id !== sourceTurnId)) {
            throw new Error("订单接受状态不完整。");
          }
          accepted = Object.freeze({ ...payload });
          continue;
        }
        if (waiting || accepted) throw new Error("等待状态后收到其他正文，请核实当前处理状态。");
        if (event.event === "answer_metadata") {
          if (metadata || payload?.session_id !== sessionId || typeof payload?.turn_id !== "string" ||
              !payload.turn_id || (sourceTurnId && payload.turn_id !== sourceTurnId) || payload.status !== "completed" || !Array.isArray(payload.citations) ||
              payload.citations.some((item) => !Number.isSafeInteger(item?.n) || item.n < 1 ||
                typeof item.answer !== "string" || typeof item.question !== "string") ||
              new Set(payload.citations.map((item) => item.n)).size !== payload.citations.length) {
            throw new Error("回答完成信息不完整，请保留本页文字并稍后核实。");
          }
          const suggested = validateSuggestedActions(payload);
          if (typeof message === "string" && suggested.some((action) => action.question !== message)) throw new Error("动作建议与本轮原问题不符。");
          const refund = validateRefundAction(payload.refund_action, payload.turn_id);
          if (refund && typeof message === "string" && refund.question !== message) throw new Error("退款建议与原问题不符。");
          // 绑定发出这一轮请求时的身份，不读取后续页面会话或服务端提供的其他用户。
          metadata = { ...payload, user_id: userId, suggested_actions: suggested, ...(refund ? { refund_action: refund } : {}) };
          continue;
        }
        if (event.event === "tool_status") {
          if (typeof payload?.tool_call_id !== "string" || typeof payload?.name !== "string" ||
              typeof payload?.label !== "string" || typeof payload?.is_mock !== "boolean" ||
              !["running", "retrying", "succeeded", "failed"].includes(payload?.status)) {
            throw new Error("工具状态数据不完整，请稍后重试。");
          }
          onToolStatus?.(payload);
          continue;
        }
        if (typeof payload?.delta !== "string") {
          throw new Error("SSE 增量缺少文本 delta 字段。");
        }
        answer += payload.delta;
        if (payload.delta) onDelta?.(payload.delta);
      }
      // 网络断开不能伪装为成功；已收到的部分回复由页面保留并标记未完成。
      if (done) {
        if (signal?.aborted) throw new DOMException("请求已取消", "AbortError");
        if (buffer.trim()) throw new Error("连接在事件中途结束，未收到完成标识。请核实处理状态。");
        if (waiting) {
          onSelectOrder?.(waiting);
          return { status: "awaiting_input", answer, action: waiting };
        }
        if (accepted) return { status: "accepted", answer, action: accepted };
        throw new Error("连接提前结束，未收到完成标识。请检查网络后重试。");
      }
    }
  } finally {
    // 完成、错误和取消路径都释放响应流；忽略已经关闭的连接在清理时产生的异常。
    try { await reader.cancel(); } catch { /* 已断开的流无需再次取消。 */ }
    reader.releaseLock();
  }
}

const UUID_V5 = /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const TICKET_TYPES = ["售后", "投诉", "咨询"];

/** 校验本人订单卡片最小快照；参数为载荷及可选原轮；返回冻结快照；异常：非法结构。 */
export function validateOrderSelection(action, sourceTurnId) {
  if (!action || action.type !== "select_order" || Object.keys(action).length !== 4 || !UUID_V5.test(action.action_id) ||
      !validText(action.source_turn_id, 36) || (sourceTurnId && action.source_turn_id !== sourceTurnId) ||
      !Array.isArray(action.orders) || !action.orders.length || action.orders.some((row) =>
        !row || Object.keys(row).length !== 3 || !validText(row.order_id, 64) || !validText(row.product_name, 500) || !validText(row.status, 200)) ||
      new Set(action.orders.map((row) => row.order_id)).size !== action.orders.length) {
    throw new Error("订单选择信息不完整，请查询当前状态。");
  }
  return Object.freeze({ ...action, orders: Object.freeze(action.orders.map((row) => Object.freeze({ ...row }))) });
}

/** 校验独立退款动作；参数为快照及原轮；返回冻结快照或null；异常：非法结构。 */
export function validateRefundAction(action, turnId) {
  if (action == null) return null;
  if (Object.keys(action).length !== 5 || action.kind !== "submit_refund" || !UUID_V5.test(action.action_id) ||
      action.source_turn_id !== turnId || !validText(action.question, 4000) || !validText(action.order_id, 64)) {
    throw new Error("退款建议信息不完整，请核实来源。");
  }
  return Object.freeze({ ...action });
}

/** 用原身份和动作续跑；参数为冻结身份/卡片/完整订单/回调；返回流终态；异常：请求或协议失败。 */
export async function resumeOrderSelection(identity, action, orderId, handlers = {}) {
  validateOrderSelection(action);
  if (!action.orders.some((row) => row.order_id === orderId)) throw new Error("请选择原卡片中的订单。");
  const response = await (handlers.fetchImpl ?? globalThis.fetch)("/api/chat/resume", {
    method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, signal: handlers.signal,
    body: JSON.stringify({ ...identity, source_turn_id: action.source_turn_id, action_id: action.action_id, order_id: orderId }),
  });
  return receiveWorkflowStream(response, { ...handlers, sessionId: identity.session_id, userId: identity.user_id, sourceTurnId: action.source_turn_id });
}

/** 回读服务端当前动作，优先于旧页面卡片；参数为冻结身份和fetch；返回可信状态；异常：HTTP/结构错误。 */
export async function getCurrentActions(identity, fetchImpl = globalThis.fetch) {
  const response = await fetchImpl(`/api/actions/current?${new URLSearchParams(identity)}`);
  if (!response.ok) throw new Error(await httpErrorMessage(response));
  const result = await response.json();
  if (!result || typeof result.resume_required !== "boolean" || (result.source_turn_id !== null && !validText(result.source_turn_id, 36)) ||
      ![null, "running", "awaiting_input", "completed", "failed", "cancelled"].includes(result.turn_status)) throw new Error("当前处理状态不完整。");
  if (result.order_selection) result.order_selection = validateOrderSelection(result.order_selection, result.source_turn_id);
  result.refund_action = validateRefundAction(result.refund_action, result.source_turn_id);
  return result;
}

/** 拉取唯一服务端原因目录；参数为fetch；返回代码/中文标签；异常：接口或目录格式错误。 */
export async function getRefundReasons(fetchImpl = globalThis.fetch) {
  const response = await fetchImpl("/api/actions/refund-reasons");
  if (!response.ok) throw new Error(await httpErrorMessage(response));
  const reasons = await response.json();
  if (!Array.isArray(reasons) || !reasons.length || reasons.some((item) => !validText(item?.code, 32) || !validText(item?.label, 200)) ||
      new Set(reasons.map((item) => item.code)).size !== reasons.length) throw new Error("退款原因目录暂不可用。");
  return reasons;
}

/** 显式提交本地申请；参数为原身份/动作/原因/确认及fetch；返回待处理回执；异常：未开始或结果未知。 */
export async function submitRefund(payload, fetchImpl = globalThis.fetch) {
  if (payload?.confirmed !== true || !UUID_V5.test(payload.action_id) || !validText(payload.reason_code, 32)) {
    throw ticketError("请选择原因并明确确认提交。", "not_started");
  }
  let response, result;
  try {
    response = await fetchImpl("/api/actions/refunds", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    result = await response.json();
  } catch { throw ticketError("申请结果尚未确认，请用原申请核实。", "unknown"); }
  if (!response.ok) throw ticketError(result?.detail?.message ?? "申请未完成，请核实原操作。", response.status === 503 ? "unknown" : "not_started", response.status);
  if (result?.action_id !== payload.action_id || result.reason_code !== payload.reason_code || result.status !== "待处理" ||
      !/^RF[0-9A-F]{32}$/.test(result.refund_no) || !validText(result.order_id, 64)) throw ticketError("申请回执尚未确认，请用原申请核实。", "unknown");
  return result;
}

/** 校验建议完整快照；参数为完成元数据；返回独立快照数组；异常：Error，非法建议。 */
export function validateSuggestedActions(metadata) {
  const suggested = metadata?.suggested_actions === undefined ? [] : metadata.suggested_actions;
  const keys = ["action_id", "kind", "source_turn_id", "question", "ticket_type"];
  if (!Array.isArray(suggested) || suggested.length > 2 || suggested.some((action) =>
    !action || Object.keys(action).length !== keys.length || keys.some((key) => !Object.hasOwn(action, key)) ||
    typeof action.action_id !== "string" || !UUID_V5.test(action.action_id) ||
    action.source_turn_id !== metadata.turn_id || !validText(action.source_turn_id, 36) ||
    !validText(action.question, 4000) ||
    (action.kind === "handoff" ? action.ticket_type !== null :
      action.kind !== "create_ticket" || !TICKET_TYPES.includes(action.ticket_type))) ||
    new Set(suggested.map((action) => action.kind)).size !== suggested.length ||
    new Set(suggested.map((action) => action.action_id)).size !== suggested.length) {
    throw new Error("动作建议信息不完整，请保留本页文字并稍后核实。");
  }
  return suggested.map((action) => Object.freeze({ ...action }));
}

/** 校验与后端一致的Unicode字符数；参数为值/上限；返回布尔；异常：无。 */
function validText(value, limit) {
  return typeof value === "string" && Boolean(value.trim()) && Array.from(value).length <= limit;
}

/** 构造确认错误；参数为安全文字/结果状态/HTTP状态；返回Error；异常：无。 */
function ticketError(message, outcome, status = 0) {
  return Object.assign(new Error(message), { outcome, status });
}

/**
 * 只向固定接口确认保存的建议，不接受客户端描述或类型。
 * @param {object} payload 原会话、用户、来源轮、动作键和confirmed:true。
 * @param {object} options 可注入fetchImpl和取消signal。
 * @returns {Promise<object>} 已校验的真实回执，含可选审计同步状态。
 * @throws {Error} 参数错误为not_started；已发请求的未知响应为unknown；业务失败保留服务端分类。
 */
export async function confirmTicket(payload, { fetchImpl = globalThis.fetch, signal } = {}) {
  const keys = ["session_id", "user_id", "source_turn_id", "action_id", "confirmed"];
  if (!payload || Object.keys(payload).some((key) => !keys.includes(key)) || payload.confirmed !== true ||
      !validText(payload.session_id, 64) || !validText(payload.user_id === undefined ? "demo-user" : payload.user_id, 64) ||
      !validText(payload.source_turn_id, 36) || typeof payload.action_id !== "string" || !UUID_V5.test(payload.action_id)) {
    throw ticketError("确认参数不正确，未提交建单请求。", "not_started");
  }
  if (signal?.aborted) throw ticketError("确认已取消，未提交建单请求。", "not_started");
  let response, result;
  try {
    response = await fetchImpl("/api/actions/tickets", { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ ...payload, user_id: payload.user_id ?? "demo-user" }), signal });
    result = await response.json();
  } catch {
    // 请求发出后即使读取或取消失败，也不能将可能已落库的工单误报为失败。
    throw ticketError("建单结果暂未确认，请先核实，或使用本条建议的原请求重试。", "unknown", response?.status);
  }
  if (!response.ok) {
    const detail = result?.detail;
    if ([404, 409, 422].includes(response.status) && Array.isArray(detail)) {
      throw ticketError("确认参数不符合要求，未开始建单。请检查原问题和会话。", "not_started", response.status);
    }
    const allowed = response.status === 503 ? ["unknown"] : response.status === 422 ? ["not_started", "failed"] :
      [404, 409].includes(response.status) ? ["not_started"] : [];
    if (typeof detail?.message === "string" && typeof detail.code === "string" && allowed.includes(detail.outcome)) {
      throw ticketError(detail.message, detail.outcome, response.status);
    }
    throw ticketError("建单结果暂未确认，请使用原请求核实。", "unknown", response.status);
  }
  if (result?.action_id !== payload.action_id || !validText(result.ticket_no, 32) ||
      !TICKET_TYPES.includes(result.ticket_type) || result.status !== "待处理" ||
      (result.audit_sync_pending !== undefined && typeof result.audit_sync_pending !== "boolean")) {
    throw ticketError("未收到有效工单回执，请使用原请求核实。", "unknown", response.status);
  }
  return { ...result, audit_sync_pending: result.audit_sync_pending ?? false };
}
