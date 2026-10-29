"""CH08热发现验收专用售后服务，可在重启时追加一个新的只读营业时间工具。"""

import argparse

from mcp.server.fastmcp import FastMCP

from mcp_servers.after_sales import create_server


def add_service_hours(server: FastMCP) -> None:
    """仅在验收服务上注册新的只读用途。

    参数：server为本验收拥有的售后MCP服务；返回：None；
    异常：SDK注册或重名错误。
    """
    @server.tool(description="查询售后人工服务的营业时间和服务时区，不办理工单或退款。")
    async def query_service_hours() -> dict[str, str]:
        """返回固定的售后服务时间供热更新验证。

        参数：无；返回：服务名、营业时间与时区字典；异常：无。
        """
        return {"service": "售后人工服务", "service_hours": "每日09:00—18:00", "timezone": "Asia/Shanghai"}


def main() -> None:
    """启动验收独占服务，只有显式开关才发布新工具。

    参数：命令行--port为独占本机端口，--hot控制新增工具；返回：None；
    异常：端口占用或SDK启动错误。
    """
    parser = argparse.ArgumentParser(description="CH08售后服务热发现验收")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--hot", action="store_true")
    args = parser.parse_args()
    server = create_server(port=args.port)
    if args.hot:
        add_service_hours(server)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
