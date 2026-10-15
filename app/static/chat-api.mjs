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
 * 发起一轮聊天，在每个有效增量帧到达时立即通知调用方。
 * @param {object} options 当前轮请求参数。
 * @param {string} options.sessionId 当前页面内保持不变的会话标识。
 * @param {string} options.message 非空用户消息。
 * @param {AbortSignal} options.signal 用于手动停止请求的取消信号。
 * @param {(delta: string) => void} options.onDelta 增量文本回调，不接收 HTML。
 * @param {typeof fetch} fetcher 默认使用浏览器 fetch；测试可注入离线替身。
 * @returns {Promise<string>} 收到 [DONE] 后返回本轮完整回复。
 * @throws {Error} HTTP 失败、SSE 错误、非法数据、空回复或缺少完成帧时抛出。
 * @throws {DOMException} 请求被取消时透传 AbortError。
 */
export async function streamChat({ sessionId, message, signal, onDelta }, fetcher = globalThis.fetch) {
  // EventSource 不支持此处需要的 POST JSON，因此使用 fetch 的可读响应流。
  const response = await fetcher("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });
  if (!response.ok) throw new Error(await httpErrorMessage(response));
  if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body) {
    throw new Error("服务器未返回 SSE 数据流，请确认正在访问本项目的聊天接口。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let answer = "";
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
        if (!event.data || !["message", "error"].includes(event.event)) continue;
        if (event.event === "message" && event.data === "[DONE]") {
          if (!answer.trim()) throw new Error("模型没有返回有效回复，请重新提问。");
          return answer;
        }

        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          throw new Error("收到无法解析的 SSE 数据，请检查接口响应。");
        }
        if (event.event === "error") {
          throw new Error(typeof payload?.error === "string" ? payload.error : "模型暂时不可用，请稍后重试。");
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
