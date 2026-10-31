"""主题统计与存储边界：精确轮次并集、全池覆盖和非法写入。"""

import importlib
from types import SimpleNamespace as NS

import pytest


def module():
    """加载待实现仓储。参数：无；返回：模块；异常：断言失败。"""
    assert importlib.util.find_spec("app.repositories.topics"), "缺少主题仓储"
    return importlib.import_module("app.repositories.topics")


def test_distribution_counts_turn_union_and_legacy_ids():
    """按行或相同文本合并会误计。参数：无；返回：None；异常：断言失败。"""
    rows = [dict(question_id=str(i), group_id=g, history_uncertain=h, review_status=s, text="同文")
            for i, g, h, s in [(1, "turn:a", False, None), (2, "turn:a", False, "待审"),
                               (3, "turn:b", False, None), (4, "legacy-pool:4", True, None),
                               (5, "turn:c", False, "通过"), (6, "turn:d", False, "驳回")]]
    classes = [dict(question_id=str(i), labels=labels) for i, labels in
               [(1, ["物流"]), (2, ["物流", "运费"]), (3, ["物流"]), (5, ["其他"])]]
    result = module().aggregate_distribution(rows, classes, "pending")
    assert result["total_questions"] == 3
    assert result["classified_questions"] == 2 and result["unclassified_questions"] == 1
    assert result["pool_rows"] == 4 and result["classified_pool_rows"] == 3
    assert result["known_original_turns"] == 2 and result["history_uncertain_pool_rows"] == 1
    counts = {row["label"]: row["count"] for row in result["labels"]}
    assert len(counts) == 17 and counts["物流"] == 2 and counts["运费"] == 1 and counts["其他"] == 0
    assert module().aggregate_distribution(rows, classes, "all")["total_questions"] == 5


def test_distribution_includes_all_pages():
    """统计截断到一页会漏掉最后组。参数：无；返回：None；异常：断言失败。"""
    rows = [dict(question_id=str(i), group_id=f"legacy-pool:{i}", history_uncertain=True, review_status=None)
            for i in range(1, 402)]
    result = module().aggregate_distribution(rows, [dict(question_id="401", labels=["发票"])], "all")
    assert result["total_questions"] == 401 and result["classified_questions"] == 1


@pytest.mark.parametrize("rows", [[{"question_id": "1", "labels": ["其他", "物流"]}],
    [{"question_id": "1", "labels": ["物流", "物流"]}], [{"question_id": "1", "labels": ["未知"]}],
    [{"question_id": "1", "labels": []}], [{"question_id": "1", "labels": ["物流"]}] * 2])
async def test_save_rejects_invalid_or_duplicate_before_connection(rows):
    """非法或重复输入不能触达数据库。参数：rows错误输入；返回：None；异常：断言失败。"""
    repo = module().TopicRepository(None, None)
    with pytest.raises(ValueError):
        await repo.save(rows, lease=NS())
