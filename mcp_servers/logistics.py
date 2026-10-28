"""物流演示MCP服务：通过独立Streamable HTTP端点提供确定性物流轨迹。"""

import argparse
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools.mock_data import logistics_snapshot


def create_server(*, host: str = "127.0.0.1", port: int = 8101) -> FastMCP:
    """建立物流服务。参数：本机地址和端口；返回：FastMCP；异常：SDK配置错误。"""
    server = FastMCP("logistics", host=host, port=port, stateless_http=True, json_response=True)

    @server.tool(description="按物流单号tracking_no查询承运方、配送状态、轨迹和更新时间。只有订单号时先query_order取非空tracking_no；未发货不查物流。返回演示数据。")
    async def query_logistics(tracking_no: Annotated[str, Field(min_length=1, max_length=64, description="明确的非空物流单号")]) -> dict[str, Any]:
        """查询确定性物流演示事实。参数：非空物流号；返回：详情或found=false；异常：SDK参数错误。"""
        try:
            data = logistics_snapshot(tracking_no.strip())
        except KeyError:
            return {"found": False, "tracking_no": tracking_no, "message": "没有对应演示物流记录"}
        return {**data, "found": True, "trajectory": [{"time": data["updated_at"], "node": data["latest_node"]}]}

    return server


def main() -> None:
    """启动独立本机服务。参数：命令行--port；返回：None；异常：端口或SDK错误。"""
    parser = argparse.ArgumentParser(description="物流演示MCP服务")
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()
    create_server(port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
