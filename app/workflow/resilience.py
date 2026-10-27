"""工作流外部依赖故障边界：统一可降级异常与不含敏感原文的诊断记录。"""

from httpx import HTTPError
from openai import APIError

from app.knowledge.schemas import KnowledgeUnavailable
from app.workflow.model import WorkflowOutputError
from app.workflow.state import ConversationState

# 只承接外部服务和正文协议故障；取消、数据库提交及程序断言不在此列。
DEPENDENCY_ERRORS = (APIError, HTTPError, OSError, KnowledgeUnavailable, WorkflowOutputError)


def degradation(state: ConversationState, stage: str, error: Exception) -> dict:
    """追加安全诊断，不暴露上游地址、凭据或错误原文。

    参数：state当前状态；stage固定阶段名；error外部依赖异常。
    返回：可合并到State的诊断更新。异常：无。
    """
    return {"degradations": [*state.get("degradations", []),
                             {"stage": stage, "error_type": type(error).__name__}]}
