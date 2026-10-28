"""审计输入只保留DDL字段，摘要有界且完整公开参数不被裁剪。"""

def test_audit_summary_is_bounded_and_arguments_are_preserved():
    """中文长结果只裁审计摘要。参数：无；返回：None；异常：断言失败。"""
    from app.repositories.tool_audit import ToolAuditEntry
    entry = ToolAuditEntry(tool_name="query_order", tool_source="builtin", status="成功",
                           arguments={"order_id": "1001"}, result_summary="中" * 5000,
                           error_message="错" * 600)
    assert len(entry.result_summary) == 4000 and len(entry.error_message) == 512
    assert entry.arguments == {"order_id": "1001"}
