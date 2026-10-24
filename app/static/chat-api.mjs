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
    else if (response.status === 422) detail = "消息或会话参数不符合接口要求。";
  } catch {
    // 代理返回 HTML 或空响应时只显示状态码，不把原始页面塞入聊天界面。
  }
  return `HTTP ${response.status}：${detail}`;
}

/**
 * 根据写操作状态生成失败提示，避免在结果未知或已提交时引导重复建单。
 * @param {object} options 本轮失败、取消及各调用的最新状态。
 * @param {Error} options.error 接口错误；retry_safe 为服务端可选的安全重试判断。
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
  if (error?.retry_safe === false || ticketMayExist) {
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
    // 尚未发出请求的取消可以安全回填；发出请求后的断流不能据此推断写入结果。
    const error = new DOMException("请求已取消", "AbortError");
    error.retry_safe = true;
    throw error;
  }
  try {
    return await receiveChatStream(options, fetcher);
  } catch (error) {
    // 状态帧可能尚未送达；仅凭没有徽章，不能认定服务端没有创建工单。
    if (typeof error.retry_safe !== "boolean") error.retry_safe = false;
    throw error;
  }
}

/**
 * 发送请求并解码响应；由 streamChat 统一标记无法确认提交结果的错误。
 * @param {object} options 会话、消息、取消信号和增量回调。
 * @param {typeof fetch} fetcher 浏览器 fetch 或离线网络替身。
 * @returns {Promise<string>} 收到完成帧后的完整回答。
 * @throws {Error} HTTP、网络或协议错误；取消时抛出 AbortError。
 */
async function receiveChatStream({ sessionId, message, signal, onDelta, onToolStatus, onAnswerMetadata }, fetcher) {
  // EventSource 不支持此处需要的 POST JSON，因此使用 fetch 的可读响应流。
  const response = await fetcher("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });
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
  try {
    while (true) {
      const { value, done } = await reader.read();
      // stream:true 保留跨网络分包的中文多字节字符；EOF 时再冲刷解码器。
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(buffer)) !== null) {
        const frame = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const event = parseEvent(frame);
        if (!event.data || !["message", "error", "tool_status", "answer_metadata"].includes(event.event)) continue;
        if (event.event === "message" && event.data === "[DONE]") {
          if (!answer.trim()) throw new Error("模型没有返回有效回复，请重新提问。");
          // 兼容旧服务的DONE；仅含新完成元数据的回答提供引用和反馈入口。
          if (metadata) onAnswerMetadata?.(metadata);
          return answer;
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
          throw error;
        }
        if (event.event === "answer_metadata") {
          if (metadata || payload?.session_id !== sessionId || typeof payload?.turn_id !== "string" ||
              !payload.turn_id || payload.status !== "completed" || !Array.isArray(payload.citations) ||
              payload.citations.some((item) => !Number.isSafeInteger(item?.n) || item.n < 1 ||
                typeof item.answer !== "string" || typeof item.question !== "string") ||
              new Set(payload.citations.map((item) => item.n)).size !== payload.citations.length) {
            throw new Error("回答完成信息不完整，请保留本页文字并稍后核实。");
          }
          metadata = payload;
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
        if (payload.delta) onDelta(payload.delta);
      }
      // 网络断开不能伪装为成功；已收到的部分回复由页面保留并标记未完成。
      if (done) throw new Error("连接提前结束，未收到完成标识。请检查网络后重试。");
    }
  } finally {
    // 完成、错误和取消路径都释放响应流；忽略已经关闭的连接在清理时产生的异常。
    try { await reader.cancel(); } catch { /* 已断开的流无需再次取消。 */ }
    reader.releaseLock();
  }
}
