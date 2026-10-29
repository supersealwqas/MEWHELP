"""为Codex内置浏览器验收记录真实HTTP原文，并提供仅测试实例使用的受控暂停。

不替换模型、工具、响应事实或数据库；所有注入需在本轮目录显式指定会话并留痕。
"""

import asyncio
from contextlib import aclosing
import json
from pathlib import Path
import time
from uuid import uuid4

from tests.ch08_live_runtime import write_json


class BrowserObserver:
    """包装测试ASGI实例，保存请求响应；不会被生产create_app默认启用。"""

    def __init__(self, app, runtime, report: Path):
        """安装一次测试暂停边界。参数：测试app/原运行时/证据目录；返回：None；异常：文件错误。"""
        self.app, self.report = app, report
        self.directory = report / "browser-http"
        self.directory.mkdir(exist_ok=True)
        self.used = set()
        self.original_run_graph = runtime.service._run_graph
        runtime.service._run_graph = self.run_graph

    def control(self):
        """读取本轮显式故障配置。参数：无；返回：配置或空字典；异常：坏文件报错。"""
        path = self.report / "browser-fault-control.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    async def run_graph(self, initial, context, outcome):
        """首次恢复在决定保存后、执行开始前暂停，取消或显式放行后保留生产关闭行为。

        参数：原图输入、可信上下文、原生命周期标记；返回：原事件流；异常：原取消/图错误透传。
        """
        from langgraph.types import Command
        control = self.control()
        identifier = control.get("id")
        if (isinstance(initial, Command) and identifier and identifier not in self.used
                and context.session_id == control.get("session_id")
                and control.get("mode") in {"disconnect_before_write", "late_reply"}):
            self.used.add(identifier)
            write_json(self.report / ("browser-fault-" + identifier + ".json"), {
                "control": control, "paused_at": time.time(), "turn_id": context.turn_id,
                "boundary": "生产首次决定已持久化；原图尚未继续，尚未调用写处理器"})
            deadline = time.monotonic() + 120
            while not (self.report / ("browser-release-" + identifier)).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("浏览器受控暂停未获测试放行")
                await asyncio.sleep(0.1)
        async with aclosing(self.original_run_graph(initial, context, outcome)) as stream:
            async for event in stream:
                yield event

    async def __call__(self, scope, receive, send):
        """记录所见ASGI消息并原样转发。参数：ASGI协议三件；返回：None；异常：原异常透传。"""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # late_reply只让测试页的原请求继续接收；停止/切会话逻辑仍由真实前端执行。
        if scope["path"] == "/__ch08_browser_probe.js":
            control = self.control()
            script = """const control=CONTROL; const original=window.fetch;
window.fetch=(input,init={})=>{
  if(control.mode==='late_reply' && String(input).endsWith('/api/chat/resume') &&
     JSON.parse(init.body||'{}').session_id===control.session_id){
    const options={...init}; delete options.signal; return original(input,options);
  } return original(input,init);
};""".replace("CONTROL", json.dumps(control, ensure_ascii=False)).encode()
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/javascript"), (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": script})
            return
        if scope["path"] == "/":
            async def inject(message):
                """只给测试HTML添加观察脚本。参数：响应帧；返回：None；异常：发送错误。"""
                if message["type"] == "http.response.start":
                    message = {**message, "headers": [(k, v) for k, v in message["headers"] if k.lower() != b"content-length"]}
                elif message["type"] == "http.response.body":
                    message = {**message, "body": message.get("body", b"").replace(b"</head>",
                        b'<script src="/__ch08_browser_probe.js"></script></head>')}
                await send(message)
            return await self.app(scope, receive, inject)
        if scope["path"] not in {"/api/chat", "/api/chat/resume", "/api/actions/current"}:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        query = scope["query_string"].decode()
        row = {"method": scope["method"], "url": scope["scheme"] + "://" + headers[b"host"].decode() + scope["path"] + ("?" + query if query else ""),
               "started_at": time.time(), "ended_at": None, "status": None, "request_body": None,
               "response_body": "", "disconnected": False, "origin": "test_asgi_wire_observer"}
        body, chunks = [], []
        async def observed_receive():
            """保留请求正文和断线。参数：无；返回：原ASGI消息；异常：接收错误。"""
            message = await receive()
            if message["type"] == "http.request":
                body.append(message.get("body", b""))
            elif message["type"] == "http.disconnect":
                row["disconnected"] = True
            return message
        async def observed_send(message):
            """保存已交付发送层的状态和正文。参数：原消息；返回：None；异常：发送错误。"""
            await send(message)
            if message["type"] == "http.response.start":
                row["status"] = message["status"]
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
        try:
            await self.app(scope, observed_receive, observed_send)
        finally:
            row["ended_at"] = time.time()
            raw = b"".join(body)
            row["request_body"] = json.loads(raw) if raw else None
            row["response_body"] = b"".join(chunks).decode("utf-8")
            write_json(self.directory / (uuid4().hex + ".json"), row)
