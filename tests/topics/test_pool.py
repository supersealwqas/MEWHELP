"""问题池轮次映射测试，禁止按原话猜测来源。"""

from types import SimpleNamespace as NS
import pytest

from app.topics.pool import derive_pool


def pool(identifier, source="self_check"):
    """构造池记录。参数：主键、入口；返回：记录；异常：无。"""
    return NS(id=identifier, conversation_id=1, source=source, raw_question="尺寸是多少", matched_review_id=None, created_at=None)


def message(identifier, mapping, conversation=1):
    """构造精确来源消息。参数：主键、映射、会话；返回：记录；异常：无。"""
    return NS(id=identifier, conversation_id=conversation, role="user", turn_id=f"turn{identifier}", content="尺寸是多少", rag_metadata={"knowledge_pool": {"pool_ids": mapping}})


def test_legacy_pool_never_guesses_turn():
    """相同原话和会话不足以认定原轮。参数：无；返回：无；异常：断言失败。"""
    result = derive_pool([pool(1)], [message(10, {})], [], [])
    assert result[0]["group_id"] == "legacy-pool:1" and result[0]["history_uncertain"]


def test_same_turn_multiple_sources_share_group():
    """入口不同但精确同轮时共享匿名家族。参数：无；返回：无；异常：断言失败。"""
    result = derive_pool([pool(1), pool(2, "user_feedback")], [message(10, {"self_check": 1, "user_feedback": 2})], [], [])
    assert result[0]["group_id"] == result[1]["group_id"]
    assert not result[0]["history_uncertain"]
    assert result[0]["provenance"]["pool_source"] == "self_check"


def test_ambiguous_exact_mapping_is_rejected():
    """重复精确映射必须报错。参数：无；返回：无；异常：断言失败。"""
    with pytest.raises(ValueError):
        derive_pool([pool(1)], [message(10, {"self_check": 1}), message(11, {"self_check": 1})], [], [])
    result = derive_pool([pool(1)], [message(10, {"self_check": 1}, conversation=2)], [], [])
    assert result[0]["history_uncertain"]


def test_pool_preserves_review_and_demo_provenance():
    """审核知识通过不能被误作标签人审。参数：无；返回：无；异常：断言失败。"""
    record = pool(1)
    record.matched_review_id = 5
    result = derive_pool([record], [], [NS(id=1, session_id="demo-ch10", user_id="test")], [NS(id=5, review_status="通过")])
    assert result[0]["review_status"] == "通过" and result[0]["source"] == "application_demo"
