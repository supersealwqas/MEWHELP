"""验证本地权限热更新、缺省拒绝与调用前订单归属边界。"""

import json

import pytest

from app.schemas.tools import ToolContext
from app.tools.contracts import ToolIdentity


def rule(name="query_warranty", server="after_sales", **overrides):
    """生成本地授权输入。参数：名称/来源/覆盖项；返回：JSON规则；异常：无。"""
    return {"source": "mcp", "server": server, "name": name, "enabled": True, "access": "read",
            "ownership": "order_id", "result_policy": {"is_mock": True}, **overrides}


def write_policy(path, *rules):
    """写测试拥有的权限文件。参数：路径与规则；返回：None；异常：文件写入错误。"""
    path.write_text(json.dumps({"version": 1, "tools": list(rules)}, ensure_ascii=False), encoding="utf-8")


def identity(name="query_warranty", server="after_sales"):
    """生成被申请MCP工具身份。参数：名称/来源；返回：ToolIdentity；异常：身份非法。"""
    return ToolIdentity("mcp", server, name)


def context():
    """提供当前用户的可信身份。参数：无；返回：ToolContext；异常：无。"""
    return ToolContext(conversation_id=1, session_id="s", user_id="demo-user", turn_id="t", user_message="查询在保情况", tool_call_id="c")


def test_unlisted_or_remote_write_is_denied(tmp_path):
    """无授权/远端写工具永不放行。参数：隔离目录；返回：None；异常：断言失败。"""
    from app.tools.permissions import ToolPermissions
    path = tmp_path / "policy.json"
    permissions = ToolPermissions(path)
    permissions.reload()
    assert not permissions.allowed(identity())
    write_policy(path, rule(access="write"))
    permissions.reload()
    result = permissions.check(identity(), {"order_id": "1001"}, context())
    assert result.error.category == "permission_denied"
    assert not permissions.allowed(identity(server="other"))


def test_bad_reload_keeps_last_complete_policy(tmp_path):
    """坏文件不部分更新，合法撤销立即生效。参数：隔离目录；返回：None；异常：断言失败。"""
    from app.tools.permissions import ToolPermissions
    path = tmp_path / "policy.json"
    write_policy(path, rule())
    permissions = ToolPermissions(path)
    permissions.reload()
    assert permissions.allowed(identity())
    write_policy(path, rule(enabled=False), rule(name="bad", readOnlyHint=True))
    permissions.reload()
    assert permissions.allowed(identity())
    write_policy(path, rule(enabled=False))
    permissions.reload()
    assert not permissions.allowed(identity())


def test_deleted_permission_file_fails_closed(tmp_path):
    """运行中移走权限文件即撤销调用。参数：隔离目录；返回：None；异常：断言失败。"""
    from app.tools.permissions import ToolPermissions
    path = tmp_path / "policy.json"
    write_policy(path, rule())
    permissions = ToolPermissions(path)
    permissions.reload()
    path.unlink()
    permissions.reload()
    assert not permissions.allowed(identity())


@pytest.mark.parametrize("name,server,ownership,args", [
    ("query_warranty", "after_sales", "order_id", {"order_id": "2003"}),
    ("query_return_progress", "after_sales", "order_id", {"order_id": "2003"}),
    ("query_logistics", "logistics", "tracking_no", {"tracking_no": "SF500290570519"}),
])
def test_cross_owner_is_denied(name, server, ownership, args, tmp_path):
    """外人业务对象在本地被拒绝。参数：工具身份/校验类型/参数/目录；返回：None；异常：断言失败。"""
    from app.tools.permissions import ToolPermissions
    path = tmp_path / "policy.json"
    write_policy(path, rule(name, server, ownership=ownership))
    permissions = ToolPermissions(path)
    permissions.reload()
    result = permissions.check(identity(name, server), args, context())
    assert result.error.code == "order_not_owned"
    assert permissions.check(identity(name, server), {"order_id": "1001", "tracking_no": "SF213502378238"}, context()) is None


def test_duplicate_rules_are_not_partially_applied(tmp_path):
    """冲突本地授权整份拒绝。参数：隔离目录；返回：None；异常：断言失败。"""
    from app.tools.permissions import ToolPermissions
    path = tmp_path / "policy.json"
    write_policy(path, rule(), rule(enabled=False))
    permissions = ToolPermissions(path)
    permissions.reload()
    assert not permissions.allowed(identity())
