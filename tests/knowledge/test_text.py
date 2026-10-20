"""知识文本格式与内容指纹的离线契约测试。"""

from app.knowledge.schemas import KnowledgeText
from app.knowledge.text import content_fingerprint, format_knowledge


def _text(**overrides: str) -> KnowledgeText:
    """构造测试用三格知识文本。

    参数:
        **overrides: 需要覆盖的 category/questions/answer 字段。
    返回值:
        KnowledgeText: 合法测试文本。
    异常:
        pydantic.ValidationError: 覆盖值不合法时抛出。
    """
    values = {
        "category": "开票与配送",
        "questions": "运费与包邮",
        "answer": "未满99元收6元。",
    }
    values.update(overrides)
    return KnowledgeText(**values)


def test_format_only_three_fields() -> None:
    """向量化文本只包含分类、问法、答案三格，不拼接元数据。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    assert format_knowledge(_text()) == "分类：开票与配送\n问法：运费与包邮\n答案：未满99元收6元。"


def test_fingerprint_changes_only_for_embedding_inputs() -> None:
    """相同三格得到相同指纹，任一格变化都会改变指纹。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    base = content_fingerprint(_text())
    assert base == content_fingerprint(_text())
    assert len(base) == 64
    for field, value in (
        ("category", "会员权益"),
        ("questions", "退货运费"),
        ("answer", "未满99元收10元。"),
    ):
        assert content_fingerprint(_text(**{field: value})) != base


def test_fingerprint_preserves_negation_and_punctuation() -> None:
    """否定条件与标点属于业务事实，不能被指纹规范化吞掉。

    参数:
        无
    返回值:
        None
    异常:
        无
    """
    assert content_fingerprint(_text(answer="偏远地区不包邮。")) != content_fingerprint(
        _text(answer="偏远地区包邮。")
    )
