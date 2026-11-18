"""邮费真实验收器的费用与承担方回归，防止错误金额、金额错位或承担方颠倒产生假通过。"""

import pytest

from tests.verify_shipping_rules import CASES, inspect_result


@pytest.mark.parametrize("answer", [
    "满199元包邮，未满收16元基础运费，偏远地区加收112元附加运费。",
    "满6元包邮，未满收99元基础运费，偏远地区加收12元附加运费。",
    "满99元包邮，未满收12元基础运费，偏远地区加收6元附加运费。",
])
def test_incorrect_shipping_amounts_fail(answer):
    """错误数字或正确数字放错规则都须报失败。

    参数：answer含错误金额的答复；返回：None。
    异常：AssertionError，验收器误放错误答复。
    """
    result = {"answer": answer, "refused": False, "citations": [{"n": 1}]}
    assert inspect_result(CASES[0], result)


@pytest.mark.parametrize("answer", [
    "单笔实付满99元包邮，未满收取6元基础运费；偏远地区另加12元附加运费。",
    "实付满 **99 元** 即可包邮，未满收取 **6 元** 基础运费。偏远地区加收 **12 元** 附加运费。",
    "单笔实付满99.00元包邮；未满时基础运费为6.00元。偏远地区附加运费为12.00元。",
])
def test_correct_shipping_amounts_pass(answer):
    """正确金额支持普通文字、Markdown和小数写法。

    参数：answer有依据的正确表述；返回：None。
    异常：AssertionError，验收器误拒等价表述。
    """
    result = {"answer": answer, "refused": False, "citations": [{"n": 1}]}
    assert inspect_result(CASES[0], result) == []


@pytest.mark.parametrize("extra, valid", [
    ("无理由退货运费由平台承担。", False),
    ("非质量问题换货的往返运费由买家承担。", True),
    ("无理由退货的退回运费由买家承担。", True),
])
def test_related_shipping_rules_must_preserve_the_payer(extra, valid):
    """补充相关规则可以接受，但不能把无理由退货运费的承担方说反。

    参数：extra补充规则；valid预期事实是否正确；返回：None；异常：AssertionError，判定不符。
    """
    result = {"answer": "单笔实付满99元包邮，未满收取6元基础运费；偏远地区另加12元附加运费。" + extra,
              "refused": False, "citations": [{"n": 1}]}
    assert (not inspect_result(CASES[0], result)) is valid
