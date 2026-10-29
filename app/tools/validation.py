"""严格校验公开工具JSON Schema，拒绝外部引用并返回有界中文字段错误。"""

from copy import deepcopy
from datetime import datetime
from itertools import islice
import re
from typing import Any

from jsonschema import Draft7Validator, Draft201909Validator, Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from langchain_core.tools import BaseTool
from referencing import Registry

_DRAFTS = {
    "http://json-schema.org/draft-07/schema": Draft7Validator,
    "https://json-schema.org/draft/2019-09/schema": Draft201909Validator,
    "https://json-schema.org/draft/2020-12/schema": Draft202012Validator,
}
_FORMATS = {"date", "date-time", "uuid", "ipv4", "ipv6"}
_FORMAT_CHECKER = FormatChecker()
_MESSAGES = {
    "required": "缺少必填字段，请补充该信息", "type": "字段类型不正确，请按参数定义填写",
    "additionalProperties": "包含不允许的额外字段", "enum": "字段取值不在允许范围内",
    "const": "字段值与要求不符", "minLength": "文本过短，请补充信息", "maxLength": "文本超过允许长度",
    "minimum": "数值低于允许下限", "maximum": "数值超过允许上限", "format": "字段格式不正确",
}


@_FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _date_time(value: Any) -> bool:
    """本地检查带时区时间，避免缺可选依赖时静默跳过。参数：JSON值；返回：是否合法；异常：ValueError，日期无效。"""
    if not isinstance(value, str):
        return True
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})", value):
        return False
    return datetime.fromisoformat(value.upper()).tzinfo is not None


def public_schema(tool: BaseTool) -> dict:
    """提取公开参数并保留原严格约束。参数：内置BaseTool；返回：独立schema；异常：schema非法。"""
    visible = tool.tool_call_schema
    schema = deepcopy(visible if isinstance(visible, dict) else visible.model_json_schema())
    full = tool.args_schema
    if full is not None:
        original = full if isinstance(full, dict) else full.model_json_schema()
        if "additionalProperties" in original:
            schema["additionalProperties"] = original["additionalProperties"]
    check_schema(schema)
    return schema


def _validator_class(schema: dict):
    """选取明确支持的草案。参数：公开schema；返回：校验器类；异常：ValueError，草案未知。"""
    dialect = schema.get("$schema")
    if dialect is None:
        return Draft202012Validator
    if not isinstance(dialect, str) or dialect.rstrip("#") not in _DRAFTS:
        raise ValueError("不支持此JSON Schema草案")
    return _DRAFTS[dialect.rstrip("#")]


def _check_references(value: Any) -> None:
    """检查所有嵌套schema的引用和格式。参数：schema树；返回：None；异常：ValueError，不安全或未知声明。"""
    if not isinstance(value, dict):
        return
    for key in ("$ref", "$dynamicRef", "$recursiveRef"):
        if key in value and (not isinstance(value[key], str) or not value[key].startswith("#")):
            raise ValueError("工具schema只允许文档内引用")
    if "$schema" in value:
        _validator_class(value)
    if "format" in value and (not isinstance(value["format"], str) or value["format"] not in _FORMATS):
        raise ValueError("工具schema包含不支持的格式")
    # 只遍历草案定义的子schema位置，业务属性名和examples/default只是数据。
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"):
        mapping = value.get(key, {})
        if isinstance(mapping, dict):
            for item in mapping.values():
                _check_references(item)
    for key in ("items", "additionalItems", "contains", "additionalProperties", "unevaluatedProperties", "unevaluatedItems",
                "propertyNames", "not", "if", "then", "else", "allOf", "anyOf", "oneOf", "prefixItems"):
        item = value.get(key)
        if isinstance(item, list):
            for nested in item:
                _check_references(nested)
        else:
            _check_references(item)


def check_schema(schema: dict) -> None:
    """登记前检查schema自身。参数：根对象schema；返回：None；异常：ValueError，不合法或引用越界。"""
    if not isinstance(schema, dict):
        raise ValueError("工具参数定义必须是JSON对象")
    _check_references(schema)
    try:
        _validator_class(schema).check_schema(schema)
    except SchemaError as error:
        raise ValueError("工具参数schema不合法") from error


def _pointer(parts: list) -> str:
    """将字段位置编码为JSON Pointer。参数：字段/数组下标列表；返回：路径；异常：无。"""
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts) if parts else ""


def validate_arguments(schema: dict, arguments: object) -> list[dict]:
    """校验原始参数，不做类型强转。参数：已登记schema及原始输入；返回：最多5项字段错误；异常：ValueError，schema不合法。"""
    check_schema(schema)
    validator = _validator_class(schema)(schema, format_checker=_FORMAT_CHECKER, registry=Registry())
    rows = []
    for error in islice(validator.iter_errors(arguments), 5):
        path = list(error.absolute_path)
        if error.validator == "required" and isinstance(error.instance, dict):
            # 不解析依赖的英文错误字符串，直接从本字段schema找出缺项。
            missing = next((key for key in error.validator_value if key not in error.instance
                            and not any(row["path"] == _pointer([*path, key]) and row["rule"] == "required" for row in rows)), None)
            # 根规则、allOf 或本地引用可能重复要求同一字段；已报告的缺项直接跳过。
            if missing is None:
                continue
            path.append(missing)
        elif error.validator == "additionalProperties" and isinstance(error.instance, dict):
            extra = [key for key in error.instance.keys() - error.schema.get("properties", {}).keys()
                     if not any(re.search(pattern, key) for pattern in error.schema.get("patternProperties", {}))]
            if extra:
                path.append(sorted(extra)[0])
        rows.append({"path": _pointer(path), "rule": str(error.validator),
                     "message": _MESSAGES.get(error.validator, "字段不符合参数约束，请补充或修正")[:200]})
    return rows
