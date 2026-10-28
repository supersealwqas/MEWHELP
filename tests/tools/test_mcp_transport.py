"""以两个独立官方SDK进程验证真实Streamable HTTP发现、结构化调用与错误。"""

import asyncio
import importlib
from pathlib import Path
import socket
import subprocess
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import ToolException
import pytest


def free_port():
    """分配测试本机端口。参数：无；返回：端口号；异常：socket错误。"""
    with socket.socket() as channel:
        channel.bind(("127.0.0.1", 0))
        return channel.getsockname()[1]


async def test_two_servers_streamable_http_calls_and_errors(tmp_path):
    """真实官方服务可独立调用，不加载应用数据库。参数：隔离日志目录；返回：None；异常：启动/协议/断言错误。"""
    # 实现缺失时立即给出目标红灯，不等一个不存在的服务直到超时。
    for name in ("logistics", "after_sales"):
        importlib.import_module(f"mcp_servers.{name}")
    processes, logs, connections = [], [], {}
    try:
        for name in ("logistics", "after_sales"):
            port = free_port()
            log = (tmp_path / f"{name}.log").open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen([sys.executable, "-m", f"mcp_servers.{name}", "--port", str(port)],
                cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            processes.append(process)
            connections[name] = {"transport": "http", "url": f"http://127.0.0.1:{port}/mcp"}
        client = MultiServerMCPClient(connections, handle_tool_errors=False)
        async with asyncio.timeout(25):
            while True:
                try:
                    tools = {tool.name: tool for tool in await client.get_tools()}
                    break
                except Exception:
                    assert all(process.poll() is None for process in processes), "MCP子进程提前退出，请查看临时日志"
                    await asyncio.sleep(0.1)
        assert set(tools) == {"query_logistics", "query_warranty", "query_return_progress"}
        calls = [("query_logistics", {"tracking_no": "SF213502378238"}),
                 ("query_warranty", {"order_id": "1001"}), ("query_return_progress", {"order_id": "1001"})]
        for name, arguments in calls:
            result = await tools[name].ainvoke({"type": "tool_call", "id": name, "name": name, "args": arguments})
            assert result.status == "success" and result.artifact["structured_content"]["order_id"] == "1001"
        missing = await tools["query_warranty"].ainvoke({"type": "tool_call", "id": "absent", "name": "query_warranty", "args": {"order_id": "absent"}})
        assert missing.artifact["structured_content"]["found"] is False
        with pytest.raises(ToolException):
            await tools["query_warranty"].ainvoke({"type": "tool_call", "id": "invalid", "name": "query_warranty", "args": {"order_id": ""}})
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                await asyncio.to_thread(process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, timeout=5)
        for log in logs:
            log.close()
