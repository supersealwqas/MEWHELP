"""工具结果的只读模型视图：本地字段规则、中文枚举和可测量的预算裁剪。"""

from copy import deepcopy

from app.core.context_budget import ContextBudgetExceeded
from app.schemas.tools import ToolResult
from app.tools.contracts import ResultPolicy

_FACT_KEYS = {"id", "status", "state", "found", "ticket_type", "currency", "amount", "price", "stock",
              "order_status", "refund_status", "tracking_status", "created_at", "updated_at"}


def _protected(key: str, value: object, policy: ResultPolicy) -> bool:
    """识别不可随长文本裁剪的标识和明确事实。参数：字段/值/规则；返回：是否保护；异常：无。"""
    return (key in _FACT_KEYS or key.endswith(("_id", "_no")) or key in policy.enum_labels
            or isinstance(value, (bool, int, float)))


def _contains_facts(value: object, policy: ResultPolicy) -> bool:
    """递归识别容器内不可丢失的事实。参数：结果子树/本地规则；返回：是否含关键事实；异常：无。"""
    if isinstance(value, dict):
        return any(_protected(key, item, policy) or _contains_facts(item, policy) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_facts(item, policy) for item in value)
    return isinstance(value, (bool, int, float))


def _translate(value: object, policy: ResultPolicy) -> object:
    """递归翻译本地声明的枚举。参数：结果树/本地规则；返回：独立中文视图；异常：无。"""
    if isinstance(value, list):
        return [_translate(item, policy) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    output = {}
    for key, item in value.items():
        if key in policy.enum_labels:
            labels = policy.enum_labels[key]
            # 未知枚举保留原值和明确说明，不套用最接近的已知业务状态。
            output[key] = labels[item] if isinstance(item, str) and item in labels else {
                "value": deepcopy(item), "label": "未知枚举值"}
        else:
            output[key] = _translate(item, policy)
    return output


def _candidates(value: object, policy: ResultPolicy, estimator) -> list:
    """寻找可缩短的正文和明细列表，不动关键标识。参数：结果树/规则/估算器；返回：候选；异常：非法JSON。"""
    candidates = []
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else []
    for key, item in items:
        if isinstance(key, str) and _protected(key, item, policy):
            continue
        if isinstance(item, str) and len(item) > 16:
            candidates.append((estimator.value_tokens(item), value, key, item[:len(item) // 2] + "…"))
        elif isinstance(item, list) and len(item) > 1 and not _contains_facts(item, policy):
            # 有关键事实的列表保留全部来源位置，只向内部裁剪说明，不能整体减半丢掉其他回执。
            candidates.append((estimator.value_tokens(item), value, key, item[:max(1, len(item) // 2)]))
        if isinstance(item, (dict, list)):
            candidates.extend(_candidates(item, policy, estimator))
    return candidates


def _removable(value: object, policy: ResultPolicy, estimator) -> list:
    """递归收集可删除的非关键字段，保留事实父路径与列表位置。参数：子树/规则/估算器；返回：候选；异常：非法JSON。"""
    candidates = []
    if isinstance(value, dict):
        for key, item in value.items():
            if _protected(key, item, policy):
                continue
            if not _contains_facts(item, policy):
                candidates.append((estimator.value_tokens(item), value, key))
            else:
                candidates.extend(_removable(item, policy, estimator))
    elif isinstance(value, list):
        for item in value:
            candidates.extend(_removable(item, policy, estimator))
    return candidates


def model_tool_result(result: ToolResult, policy: ResultPolicy, *, max_tokens: int, estimator) -> dict:
    """生成不含后台元数据的有界模型结果副本。

    参数：result完整可信结果；policy本地字段/枚举规则；max_tokens正整数预算；estimator统一value_tokens估算器。
    返回：dict，保留成功/错误语义、标识和关键事实，裁剪时包含truncated=True；中文由调用方以ensure_ascii=False序列化。
    异常：ValueError，预算非法；ContextBudgetExceeded，连关键事实或安全错误本身都放不下；TypeError，结果非JSON。
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("工具结果预算必须为正整数")
    data = result.data
    if data is not None and policy.fields:
        selected = [key for key in policy.fields if key in data]
        selected.extend(key for key, value in data.items() if key not in selected
                        and (_protected(key, value, policy) or _contains_facts(value, policy)))
        data = {key: data[key] for key in selected}
    view = {"ok": result.ok, "data": _translate(data, policy),
            "error": result.error.model_dump(exclude_none=True) if result.error is not None else None}
    if estimator.value_tokens(view) <= max_tokens:
        return view
    # 所有裁剪只操作副本；显式提示结果不完整，不把省略的明细误称为查询落空。
    view["truncated"] = True
    while estimator.value_tokens(view) > max_tokens:
        candidates = _candidates(view["data"], policy, estimator)
        if candidates:
            _, parent, key, replacement = max(candidates, key=lambda candidate: candidate[0])
            parent[key] = replacement
            continue
        removable = _removable(view["data"], policy, estimator)
        if not removable:
            raise ContextBudgetExceeded("工具关键事实超过模型结果预算，不能静默丢弃")
        _, parent, key = max(removable, key=lambda candidate: candidate[0])
        del parent[key]
    return view
