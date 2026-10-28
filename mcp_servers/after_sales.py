"""售后演示MCP服务：独立提供在保和退货进度查询，不访问真实业务或数据库。"""

import argparse
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools.mock_data import order_snapshot


def create_server(*, host: str = "127.0.0.1", port: int = 8102) -> FastMCP:
    """建立售后服务。参数：本机地址和端口；返回：FastMCP；异常：SDK配置错误。"""
    server = FastMCP("after_sales", host=host, port=port, stateless_http=True, json_response=True)

    @server.tool(description="根据order_id查询订单商品是否在保及演示保修截止时间。这是商品在保状态查询，不是退款政策，不执行售后申请。")
    async def query_warranty(order_id: Annotated[str, Field(min_length=1, max_length=64, description="用户明确的订单号")]) -> dict[str, Any]:
        """查询订单演示保修信息。参数：订单号；返回：在保事实或found=false；异常：SDK参数错误。"""
        try:
            order = order_snapshot(order_id.strip())
        except KeyError:
            return {"found": False, "order_id": order_id, "message": "没有对应演示保修记录"}
        return {"found": True, "order_id": order["order_id"], "product": order["product"],
                "warranty_status": "in_warranty", "expires_at": "2027-01-01", "note": "固定演示信息，不代表真实保修承诺"}

    @server.tool(description="根据order_id查询已发起退货的演示办理进度、当前阶段及必要说明，不执行退款，也不查询能否退货的政策。")
    async def query_return_progress(order_id: Annotated[str, Field(min_length=1, max_length=64, description="用户明确的订单号")]) -> dict[str, Any]:
        """查询确定性演示退货进度。参数：订单号；返回：进度或found=false；异常：SDK参数错误。"""
        try:
            order = order_snapshot(order_id.strip())
        except KeyError:
            return {"found": False, "order_id": order_id, "message": "没有对应演示退货记录"}
        return {"found": True, "order_id": order["order_id"], "return_status": "processing",
                "updated_at": "2026-10-28T10:00:00+08:00", "note": "演示退货处理中，不代表退款已到账"}

    return server


def main() -> None:
    """启动独立本机服务。参数：命令行--port；返回：None；异常：端口或SDK错误。"""
    parser = argparse.ArgumentParser(description="售后演示MCP服务")
    parser.add_argument("--port", type=int, default=8102)
    args = parser.parse_args()
    create_server(port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
