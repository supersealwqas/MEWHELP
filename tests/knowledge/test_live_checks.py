"""真实验收工具的阈值与事实校验逻辑测试。"""

from __future__ import annotations

import pytest
import json
from pathlib import Path
from types import SimpleNamespace
import copy
import tempfile
from tests import verify_ch03_live as live_checks

from app.knowledge.schemas import VectorHit
from tests.verify_ch03_live import ScoredQuery, choose_threshold, validate_answer, _positive_ids
from tests.verify_ch03_live import validate_recovery_snapshots


def _hit(identifier: int, score: float) -> VectorHit:
    """构造命中。参数：主键/分数；返回：命中；异常：校验失败。"""
    return VectorHit(
        id=identifier,
        score=score,
        content_hash="a" * 64,
        embedding_revision="rev",
    )


def test_choose_threshold_retains_positive_and_rejects_negative() -> None:
    """正例保留且负例低于阈值。参数：无；返回：None；异常：无。"""
    scores = [
        ScoredQuery("positive", [1], False, [_hit(1, 0.91), _hit(2, 0.3)]),
        ScoredQuery("negative", [], True, [_hit(9, 0.4)]),
    ]
    assert choose_threshold(scores) == 0.91


def test_choose_threshold_rejects_overlap() -> None:
    """正负例分数重叠时不能生成策略。参数：无；返回：None；异常：无。"""
    scores = [
        ScoredQuery("positive", [1], False, [_hit(1, 0.5)]),
        ScoredQuery("negative", [], True, [_hit(9, 0.7)]),
    ]
    with pytest.raises(ValueError):
        choose_threshold(scores)


def test_choose_threshold_requires_positive_recall() -> None:
    """正例未进入 Top3 时停止校准。参数：无；返回：None；异常：无。"""
    scores = [ScoredQuery("positive", [1], False, [_hit(2, 0.9)])]
    with pytest.raises(ValueError):
        choose_threshold(scores)


def test_validate_answer_rejects_missing_or_forbidden_fact() -> None:
    """缺事实或含禁止结论均失败。参数：无；返回：None；异常：无。"""
    case = {
        "must_include": ["6 元"],
        "must_not_include": ["10 元"],
    }
    assert validate_answer(case, {"answer": "未满99元收6元。", "hits": [{"id": 1}]}) == []
    assert validate_answer(case, {"answer": "未满99元收10元。", "hits": [{"id": 1}]})


def test_validate_empty_case_requires_no_hits() -> None:
    """负例必须空命中。参数：无；返回：None；异常：无。"""
    case = {"expect_empty": True, "must_not_include": ["6 元"]}
    assert validate_answer(case, {"answer": "没有查到相关政策。", "hits": []}) == []
    assert validate_answer(case, {"answer": "没有查到。", "hits": [{"id": 1}]})


def test_validate_accepts_safe_paraphrase_and_markdown() -> None:
    """同义事实与 Markdown 强调不应被误判。参数：无；返回：None；异常：无。"""
    case = {
        "must_include": ["不承诺"],
        "must_include_any": [["到账时效", "到账时间"]],
    }
    observed = {
        "answer": "内部垫芯**不可机洗**；平台不承诺具体的到账时间。",
        "hits": [{"id": 1}],
    }
    assert validate_answer(case, observed) == []


@pytest.mark.parametrize("ident,answer", [
    ("shipping", "未满199元收16元，偏远112元附加费不参与包邮。"),
    ("member_discount", "银卡95折，金卡9折。"),
    ("member_discount", "银卡19折，金卡995折。"),
    ("member_discount", "银卡不享9折，金卡不享95折。"),
    ("shipping", "未满99元收12元基础运费，满99元包邮，偏远加收6元附加费不参与包邮。"),
    ("shipping", "满99元收6元基础运费，未满免基础运费，偏远另收12元且不参与包邮。"),
    ("refund_processing", "退款处理时限为13个工作日。"),
    ("refund_processing", "发票3个工作日；退款将在3个工作日内到账。"),
    ("lite_app", "Lite不含App联网，但支持远程监控与健康记录。"),
    ("lite_app", "Pro不含App联网，没有远程监控与健康记录。Lite能够连接手机。"),
    ("negative_weather", "明天会下雨。"),
    ("negative_weather", "没查到相关政策，明天会下雨，概率80%。"),
])
def test_rejects_misbound_numbers_and_scope(ident: str, answer: str) -> None:
    """拒绝尾部数值假匹配与主体错绑。参数：案例ID/错误回答；返回：None；异常：断言失败。"""
    cases = [case for name in ("acceptance_cases.json", "calibration_cases.json")
             for case in json.loads((Path("tests/fixtures/ch03") / name).read_text(encoding="utf-8"))["cases"]]
    case = next(case for case in cases if case["id"] == ident)
    assert validate_answer(case, {"answer": answer, "hits": [] if case.get("expect_empty") else [{"id": 1}]})


def test_separated_digits_are_not_combined() -> None:
    """空白不能把分开的数字拼成金额。参数：无；返回：None；异常：断言失败。"""
    assert validate_answer({"must_include": ["600积分"]}, {"answer": "6 00积分"})


def test_cleaning_equivalent_negative_wording_is_allowed() -> None:
    """不可以与不可机洗表达同一否定事实。参数：无；返回：None；异常：断言失败。"""
    case = {"answer_rule": "nest_cleaning", "must_include": ["内部垫芯不可机洗"]}
    assert validate_answer(case, {"answer": "内部垫芯不可以机洗，只能局部擦洗晾干。"}) == []


def test_shipping_requires_paid_amount_condition() -> None:
    """订单金额不能替代实付包邮门槛。参数：无；返回：None；异常：断言失败。"""
    case = {"answer_rule": "shipping"}
    assert validate_answer(case, {"answer": "单笔订单满99元包邮，未满收6元基础运费，偏远另收12元且不参与包邮。"})


def test_threshold_preserves_every_required_id() -> None:
    """必要低分事实不能被阈值截掉。参数：无；返回：None；异常：断言失败。"""
    assert choose_threshold([ScoredQuery("two", [1, 2], False,
        [_hit(1, .9), _hit(2, .6), _hit(3, .5)])]) == .6


def test_threshold_rejects_required_fact_outside_top3() -> None:
    """第四名必要事实不能冒充Top3覆盖。参数：无；返回：None；异常：断言失败。"""
    with pytest.raises(ValueError):
        choose_threshold([ScoredQuery("two", [1, 2], False,
            [_hit(1, .9), _hit(3, .8), _hit(4, .7), _hit(2, .6)])])


def test_fact_support_cannot_use_unrelated_text_in_same_section() -> None:
    """同章错误数字不能作为支持事实。参数：无；返回：None；异常：断言失败。"""
    chunk = SimpleNamespace(id=1, is_active=True, source_key="doc:billing-shipping.md",
        section_path="开票与配送 / 运费与包邮", text=SimpleNamespace(answer="199元包邮，16元运费"))
    case = {"source": "billing-shipping.md", "section": chunk.section_path,
            "required_facts": ["shipping_base_fee"]}
    assert _positive_ids([chunk], case) == []


def test_threshold_support_groups_allow_alternative_but_require_each_fact() -> None:
    """一项事实可有多个支持块，所有事实仍须覆盖。参数：无；返回：None；异常：断言失败。"""
    score = ScoredQuery("groups", [], False, [_hit(1, .9), _hit(2, .6)],
        {"fee": [1, 8], "exception": [2, 9]})
    assert choose_threshold([score]) == .6


@pytest.mark.parametrize("fault", ["id", "content_hash", "encoding_hash", "embedding_revision", "pending_visible", "missing_not_repaired"])
def test_recovery_rejects_changed_identity_or_visibility(fault: str) -> None:
    """不能仅凭done宣称恢复正确。参数：损坏字段；返回：None；异常：断言失败。"""
    base = {"id": 7, "content_hash": "a", "encoding_hash": "b", "embedding_revision": "rev", "is_active": True,
            "vector_id": "7", "vectorize_status": "done", "raw_ids": [7], "visible_ids": [7]}
    snapshots = {phase: copy.deepcopy(base) for phase in ("pending", "recovered", "missing", "repaired")}
    snapshots["pending"].update(vector_id=None, vectorize_status="pending", visible_ids=[])
    snapshots["missing"].update(raw_ids=[], visible_ids=[])
    if fault in ("id", "content_hash", "encoding_hash", "embedding_revision"):
        snapshots["repaired"][fault] = "different"
    elif fault == "pending_visible":
        snapshots["pending"]["visible_ids"] = [7]
    else:
        snapshots["repaired"]["raw_ids"] = []
    assert validate_recovery_snapshots(snapshots)


async def test_rejected_tool_call_cannot_publish_calibration_policy(monkeypatch) -> None:
    """失败工具的参数不能借Direct探针发布策略。参数：替换器；返回：None；异常：断言失败。"""
    temporary = Path(".pytest-tmp")
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as directory:
        root = Path(directory)
        (root / "calibration_cases.json").write_text(json.dumps({"cases": [{"id": "p", "query": "配送"}]}), encoding="utf-8")
        (root / "facts.json").write_text('{"cases":[]}', encoding="utf-8")
        policy = root / "policy.json"
        policy.write_text("old-valid-policy", encoding="utf-8")

        async def prepared(config):
            """替代外部导入。参数：配置；返回：摘要；异常：无。"""
            return {}

        async def observations(*args):
            """复现有参数但被工具拒绝的证据。参数：调用参数；返回：诊断；异常：无。"""
            return [{"effective_keyword": "邮费", "executed_keyword": None,
                     "exchange_failures": ["非法关键词"], "tool_pairs": [{"result": {"ok": False}}],
                     "valid_candidates": [_hit(1, .9).model_dump(mode="json")],
                     "support_groups": {"fact": [1]}, "passed": True}]

        monkeypatch.setattr(live_checks, "_prepare_kb", prepared)
        monkeypatch.setattr(live_checks, "_observe_cases", observations)
        result = await live_checks.calibrate(SimpleNamespace(), root, policy, root / "report.md")
        assert result["passed"] is False
        assert policy.read_text(encoding="utf-8") == "old-valid-policy"


def test_changed_document_hash_cannot_be_reused_as_current_snapshot() -> None:
    """旧知识内容不能冒称匹配新语料。参数：无；返回：None；异常：断言失败。"""
    from app.knowledge.schemas import ChunkDraft, KnowledgeText
    draft = ChunkDraft(source_key="doc:test.md", ordinal=0,
        text=KnowledgeText(category="分类", questions="标题", answer="新版答案"),
        section_path="标题", content_type="faq", is_key_clause=False)
    row = SimpleNamespace(id=1, source_key="doc:test.md", ordinal=0, section_path="标题",
        content_hash="old-hash", is_active=True, vectorize_status="done", vector_id="1",
        embedding_revision=live_checks.EMBEDDING_REVISION)
    assert live_checks.validate_document_snapshot([draft], [row])
