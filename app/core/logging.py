"""应用业务日志初始化模块。

集中配置 `app` 命名空间下的日志输出，供各业务模块通过
`logging.getLogger(__name__)` 直接使用，避免每个模块重复挂载 handler。

设计要点：
- 日志格式统一包含时间、级别、logger 名称与消息；
- 通过 handler 判重保证 `--reload` 或重复导入时不会叠加输出；
- 业务日志只写自己的 handler，不向 root 冒泡，避免与 uvicorn 日志重复。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: int = logging.INFO, *, log_path: Path | str = "log/app.log") -> None:
    """初始化应用业务日志。

    为 `app` 及其子 logger（如 `app.api.chat`）挂载唯一的控制台 handler，
    并设置最低输出级别。重复调用不会重复添加 handler。

    参数:
        level (int): 业务日志的最低输出级别，默认 logging.INFO。
        log_path: UTF-8轮转文件路径，默认项目log/app.log。

    返回值:
        None: 无返回值。

    异常:
        OSError: 日志目录或文件无法创建，不能声称日志已落盘。
    """
    app_logger = logging.getLogger("app")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not any(getattr(handler, "_mewhelp_console", False) for handler in app_logger.handlers):
        handler = logging.StreamHandler()
        handler._mewhelp_console = True
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)
    path = Path(log_path).resolve()
    if not any(isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(path)
               for handler in app_logger.handlers):
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)
    app_logger.setLevel(level)
    # 只走自己的 handler，向上冒泡会与 root/uvicorn 的 handler 重复输出。
    app_logger.propagate = False
