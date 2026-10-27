"""官方SQLite保存器与图生命周期入口。"""
from contextlib import asynccontextmanager
from pathlib import Path
from dataclasses import asdict
import logging
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.workflow.graph import WorkflowDependencies, build_workflow
from app.workflow.context import shared_budget
from app.core.context_budget import ContextBudgetExceeded

@asynccontextmanager
async def workflow_runtime(deps: WorkflowDependencies, checkpoint_path: Path):
    """在实例生命周期持有官方文件保存器，退出时关闭连接。

    参数：deps图依赖、checkpoint_path专用SQLite文件路径，不接受内存连接或URI。
    返回：异步上下文中的编译根图，默认递归保险上限64，与业务限制独立。
    异常：ValueError，内存/URI路径；OSError/SQLite错误，文件不可访问；无内存后备。
    """
    raw_path = str(checkpoint_path)
    if raw_path == ":memory:" or raw_path.startswith("file:") or "mode=memory" in raw_path:
        raise ValueError("工作流必须使用专用SQLite持久化文件")
    path = Path(checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        budget = shared_budget(deps.settings)
    except ContextBudgetExceeded as error:
        logging.getLogger(__name__).error("上下文预算不足：%s", error)
        raise
    logging.getLogger(__name__).info("context budget startup %s", asdict(budget))
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield build_workflow(deps, checkpointer=saver)
