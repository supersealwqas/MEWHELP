"""应用业务日志初始化模块。

集中配置 `app` 命名空间下的日志输出，供各业务模块通过
`logging.getLogger(__name__)` 直接使用，避免每个模块重复挂载 handler。

设计要点：
- 日志格式统一包含时间、级别、logger 名称与消息；
- 通过 handler 判重保证 `--reload` 或重复导入时不会叠加输出；
- 业务日志只写自己的 handler，不向 root 冒泡，避免与 uvicorn 日志重复。
"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """初始化应用业务日志。

    为 `app` 及其子 logger（如 `app.api.chat`）挂载唯一的控制台 handler，
    并设置最低输出级别。重复调用不会重复添加 handler。

    参数:
        level (int): 业务日志的最低输出级别，默认 logging.INFO。

    返回值:
        None: 无返回值。

    异常:
        无业务异常抛出。
    """
    app_logger = logging.getLogger("app")
    # 已配置过则直接复用，避免 --reload 或测试重复导入时叠加 handler。
    if app_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    app_logger.addHandler(handler)
    app_logger.setLevel(level)
    # 只走自己的 handler，向上冒泡会与 root/uvicorn 的 handler 重复输出。
    app_logger.propagate = False
