"""CH07上下文文件日志：验证UTF-8、轮转和重复初始化，不连接外部模型。"""

import logging
from logging.handlers import RotatingFileHandler

from app.core.logging import setup_logging


def test_context_log_is_utf8_rotating_and_setup_is_idempotent(tmp_path):
    """重复初始化不能写两遍，中文正文必须原样。参数：tmp_path隔离日志；返回：None；异常：断言失败。"""
    logger = logging.getLogger("app")
    original = list(logger.handlers)
    path = tmp_path / "app.log"
    try:
        setup_logging(log_path=path)
        setup_logging(log_path=path)
        handlers = [handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)
                    and handler.baseFilename == str(path.resolve())]
        assert len(handlers) == 1 and handlers[0].maxBytes > 0 and handlers[0].backupCount > 0
        logger.info('model_ctx {"summary":"订单1001尚未解决","messages":["原话  "]}')
        handlers[0].flush()
        text = path.read_text(encoding="utf-8")
        assert text.count("订单1001尚未解决") == 1 and "原话  " in text
    finally:
        for handler in logger.handlers[:]:
            if handler not in original:
                logger.removeHandler(handler)
                handler.close()

