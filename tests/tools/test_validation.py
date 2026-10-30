"""验证公开 JSON Schema 的严格类型、字段路径和本地引用边界。"""

import pytest

from app.tools.business import create_ticket, query_order


def test_public_schema_has_no_identity():
    """从真实内置工具提取参数且保留禁止额外字段。参数：无；返回：None；异常：断言失败。"""
    from app.tools.validation import public_schema
    schema = public_schema(query_order)
    assert schema["required"] == ["order_id"]
    assert set(schema["properties"]) == {"order_id"}
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("arguments,path,rule", [
    ({"ticket_type": "咨询"}, "/description", "required"),
    ({"ticket_type": "咨询", "description": 1001}, "/description", "type"),
    ({"ticket_type": "咨询", "description": "x" * 2001}, "/description", "maxLength"),
    ({"ticket_type": "未知", "description": "问题"}, "/ticket_type", "enum"),
    ({"ticket_type": "咨询", "description": "问题", "user_id": "other"}, "/user_id", "additionalProperties"),
])
def test_strict_arguments_have_actionable_paths(arguments, path, rule):
    """返回具体字段错误，不偷偷强转。参数：非法参数、路径与规则；返回：None；异常：断言失败。"""
    from app.tools.validation import public_schema, validate_arguments
    errors = validate_arguments(public_schema(create_ticket), arguments)
    assert any(row["path"] == path and row["rule"] == rule for row in errors)
    assert all(0 < len(row["message"]) <= 200 for row in errors)


def test_optional_values_remain_optional_and_errors_are_bounded():
    """省略合法字段可执行，错误列表有限。参数：无；返回：None；异常：断言失败。"""
    from app.tools.validation import validate_arguments
    schema = {"type": "object", "properties": {"note": {"type": "string"}}}
    assert validate_arguments(schema, {}) == []
    schema["required"] = [f"missing_{i}" for i in range(20)]
    assert len(validate_arguments(schema, {})) == 5


def test_local_ref_and_oneof():
    """本地组合schema保留嵌套字段路径。参数：无；返回：None；异常：断言失败。"""
    from app.tools.validation import check_schema, validate_arguments
    schema = {"type": "object", "$defs": {"row": {"type": "object", "properties": {"qty": {
        "oneOf": [{"type": "integer", "minimum": 1}, {"const": "全部"}]}}}},
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/row"}}}}
    check_schema(schema)
    assert validate_arguments(schema, {"items": [{"qty": "全部"}]}) == []
    assert validate_arguments(schema, {"items": [{"qty": 0}]})[0]["path"] == "/items/0/qty"


@pytest.mark.parametrize("schema", [
    {"properties": {"value": {"$ref": "#/$defs/missing"}}},
    {"properties": {"value": {"$ref": "#missing"}}},
    {"properties": {"value": {"$dynamicRef": "#missing"}}},
    {"$defs": {"row": {"properties": {"value": {"$ref": "#/$defs/missing"}}}}},
    {"$defs": {"row": {"$id": "nested", "properties": {"value": {"$ref": "#missing"}}}}},
    {"$ref": "#/custom", "custom": {"$ref": "#missing"}},
    {"$defs": {"row": {"maxLength": 3}}, "properties": {"value": {"$ref": "#/$defs/row/maxLength/missing"}}},
])
def test_unresolvable_local_reference_is_rejected_before_use(schema):
    """漏掉本地指针、锚点或嵌套引用检查会让坏定义进入注册表。

    参数：schema 含不存在目标的定义；返回：None；异常：断言失败。
    """
    from app.tools.validation import check_schema
    with pytest.raises(ValueError):
        check_schema({"type": "object", **schema})


@pytest.mark.parametrize("draft", ["http://json-schema.org/draft-07/schema#",
                                   "https://json-schema.org/draft/2019-09/schema",
                                   "https://json-schema.org/draft/2020-12/schema"])
@pytest.mark.parametrize("target", [{"properties": 1}, {"type": 42}])
def test_malformed_reference_target_is_rejected_as_schema_error(draft, target):
    """只核实引用目标存在而忽略其结构时，坏配置会逃过登记的 ValueError 隔离。

    参数：draft 支持的根草案，target 扩展位置中的坏规则；返回：None；异常：断言失败。
    """
    from app.tools.validation import check_schema
    with pytest.raises(ValueError):
        check_schema({"$schema": draft, "type": "object", "$ref": "#/custom", "custom": target})


@pytest.mark.parametrize("schema,arguments", [
    ({"$defs": {"a/b~c": {"type": "string"}}, "properties": {"value": {"$ref": "#/$defs/a~1b~0c"}}}, {"value": "x"}),
    ({"$defs": {"row": {"$anchor": "row", "type": "string"}}, "properties": {"value": {"$ref": "#row"}}}, {"value": "x"}),
    ({"properties": {"child": {"$ref": "#"}}}, {"child": {"child": {}}}),
    ({"$dynamicAnchor": "node", "properties": {"child": {"$dynamicRef": "#node"}}}, {"child": {}}),
    ({"$schema": "https://json-schema.org/draft/2019-09/schema", "$recursiveAnchor": True,
      "properties": {"child": {"$recursiveRef": "#"}}}, {"child": {}}),
    ({"$schema": "http://json-schema.org/draft-07/schema#", "definitions": {"row": {"$id": "#row", "type": "string"}},
      "properties": {"value": {"$ref": "#row"}}}, {"value": "x"}),
    ({"$schema": "http://json-schema.org/draft-07/schema#", "custom": {"type": "array", "items": [{"type": "integer"}]},
      "properties": {"value": {"$ref": "#/custom"}}}, {"value": [1]}),
    ({"$id": "https://schemas.invalid/root", "properties": {"value": {"$id": "nested", "type": "object",
      "$defs": {"leaf": {"type": "string"}}, "properties": {"leaf": {"$ref": "#/$defs/leaf"}}}}}, {"value": {"leaf": "x"}}),
])
def test_resolvable_local_references_keep_their_scope_and_recursion(schema, arguments):
    """把合法锚点、嵌套资源或递归一律拒绝会破坏原有合法参数。

    参数：schema 合法本地引用定义，arguments 有限合法实例；返回：None；异常：断言失败或生产校验异常。
    """
    from app.tools.validation import check_schema, validate_arguments
    schema = {"type": "object", **schema}
    check_schema(schema)
    assert validate_arguments(schema, arguments) == []


@pytest.mark.parametrize("fragment", [
    {"$ref": "https://example.invalid/private"}, {"$dynamicRef": "file:///private.json"},
    {"$schema": "https://example.invalid/schema"}, {"format": "unknown-format"},
])
def test_external_ref_and_unknown_dialect_or_format_rejected(fragment, monkeypatch):
    """无论远端能否响应都在登记时拒绝外部引用。参数：非法schema片段及替换器；返回：None；异常：断言失败。"""
    from app.tools.validation import check_schema

    def no_network(*args, **kwargs):
        """网络访问即测试失败。参数：任意连接；返回：无；异常：AssertionError。"""
        raise AssertionError("schema 校验不应发起网络请求")

    monkeypatch.setattr("socket.create_connection", no_network)
    with pytest.raises(ValueError):
        check_schema({"type": "object", **fragment})


@pytest.mark.parametrize("draft", ["http://json-schema.org/draft-07/schema#", "https://json-schema.org/draft/2019-09/schema", "https://json-schema.org/draft/2020-12/schema"])
def test_supported_drafts_enforce_formats(draft):
    """不同草案都启用本地格式检查。参数：草案地址；返回：None；异常：断言失败。"""
    from app.tools.validation import validate_arguments
    schema = {"$schema": draft, "type": "object", "properties": {"date": {"type": "string", "format": "date"}}}
    assert validate_arguments(schema, {"date": "2026-10-28"}) == []
    assert validate_arguments(schema, {"date": "2026-99-99"})[0]["rule"] == "format"


def test_old_tool_error_codes_get_stable_categories():
    """旧消息恢复仍能区分参数、权限和故障。参数：无；返回：None；异常：断言失败。"""
    from app.schemas.tools import ToolError
    assert ToolError(code="INVALID_ARGUMENTS", message="参数错").category == "invalid_arguments"
    assert ToolError(code="order_not_owned", message="没有权限").category == "permission_denied"
    assert ToolError(code="TOOL_TIMEOUT", message="超时").category == "fault"


def test_literal_schema_keyword_property_and_examples_allowed():
    """业务字段/示例中的schema单词不是校验器指令。参数：无；返回：None；异常：断言失败。"""
    from app.tools.validation import check_schema, validate_arguments
    schema = {"type": "object", "properties": {"format": {"type": "string"}, "$ref": {"type": "string"}},
              "examples": [{"format": "custom", "$ref": "https://example.invalid"}]}
    check_schema(schema)
    assert validate_arguments(schema, {"format": "custom", "$ref": "只是数据"}) == []


def test_required_errors_have_distinct_paths():
    """多个缺字段不能全部错报第一个字段。参数：无；返回：None；异常：断言失败。"""
    from app.tools.validation import validate_arguments
    errors = validate_arguments({"type": "object", "required": ["first", "second"]}, {})
    assert [row["path"] for row in errors] == ["/first", "/second"]


@pytest.mark.parametrize("nested", [False, True])
def test_repeated_required_rules_return_missing_paths_without_crashing(nested):
    """同字段在组合规则中重复缺失不能抛异常或漏掉其他缺项。

    参数：nested 是否把规则放到嵌套对象；返回：None；异常：断言失败或生产校验异常。
    """
    from app.tools.validation import validate_arguments
    schema = {"type": "object", "required": ["order_id"],
              "allOf": [{"required": ["order_id"]}, {"required": ["order_id", "reason"]}]}
    arguments, prefix = {}, ""
    if nested:
        schema = {"type": "object", "properties": {"order": schema}}
        arguments, prefix = {"order": {}}, "/order"
    errors = validate_arguments(schema, arguments)
    assert {(row["path"], row["rule"]) for row in errors} == {
        (prefix + "/order_id", "required"), (prefix + "/reason", "required")}
    assert len(errors) <= 5


@pytest.mark.parametrize("value,valid", [("2026-10-28T10:00:00+08:00", True), ("2026-10-28T02:00:00Z", True),
                                        ("2026-10-28T02:00:00", False), ("not-a-date", False)])
def test_date_time_never_silently_skips_format_check(value, valid):
    """无可选第三方format包时仍校验时间及显式时区。参数：输入/预期有效性；返回：None；异常：断言失败。"""
    from app.tools.validation import validate_arguments
    errors = validate_arguments({"type": "object", "properties": {"at": {"type": "string", "format": "date-time"}}}, {"at": value})
    assert (errors == []) is valid
