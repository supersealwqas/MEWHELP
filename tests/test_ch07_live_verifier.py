"""CH07真实验收判定器：必须有原文/流/落库证据，不能只信页面或进程退出码。"""

import importlib
import logging

import pytest


def test_verifier_rejects_empty_stream_and_changed_source():
    """空答案及落库原话改变均应失败。参数：无；返回：None；异常：断言或缺模块。"""
    module = importlib.import_module("tests.verify_ch07_live")
    data = {"query":"原问", "http_status":200, "events":[{"event":"answer_metadata","data":{}},{"event":"done"}],
        "messages":[{"role":"user","content":"被改写", "tool_calls":None},{"role":"assistant","content":"", "tool_calls":None}],
        "state":{"raw_query":"原问", "completion_metadata":{"status":"completed"}}}
    result = module.inspect_record(data)
    assert "empty_answer" in result["failures"] and "original_or_saved_answer" in result["failures"]


def test_profile_default_and_demo_share_the_frozen_questions():
    """两个配置必须使用同一组不少于21轮，尾问不直接泄露早期编号。参数：无；返回：None；异常：断言或缺模块。"""
    module = importlib.import_module("tests.verify_ch07_live")
    cases = module.load_cases()
    assert len(cases["queries"]) >= 21
    assert cases["early_order_id"] in cases["queries"][0]
    assert cases["early_order_id"] not in cases["queries"][-1]


def test_final_checks_reject_failure_in_switched_conversation():
    """侧栏切出的会话空答也不能验收通过。参数：无；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch07_live")
    cases = {"queries":["第一问"],"early_order_id":"1001"}
    report = {"turns":[{"session_id":"main","query":"第一问","answer":"有回答","failures":[]},
        {"session_id":"second","query":"另一问","answer":"","failures":["empty_answer"]}],
        "cleanup":{"restored":True},"summary_cleanup_restored":True,"saved_segments":[]}
    checks = module.final_checks(report, cases, "main", [{"history_context":{}}], "default", False)
    assert checks["all_answered"] is True
    assert checks["all_recorded_answered"] is False


def test_default_checks_include_compression_after_switching_back():
    """切回续聊时的压缩也属于默认验收。参数：无；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch07_live")
    report = {"turns":[{"session_id":"main","query":"第一问","answer":"有回答","failures":[]}],
        "cleanup":{"restored":True},"summary_cleanup_restored":True,"saved_segments":[]}
    states = [{"history_context":{}},{"history_context":{"layer1_from_msg_id":12}}]
    checks = module.final_checks(report, {"queries":["第一问"]}, "main", states, "default", False)
    assert checks["no_compression"] is False


def test_report_log_captures_non_propagating_application_logger(tmp_path, monkeypatch):
    """生产app日志不向root冒泡时报告仍须落原日志。参数：临时目录/替身；返回：None；异常：断言失败。"""
    module = importlib.import_module("tests.verify_ch07_live")
    logger = logging.getLogger("app")
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.INFO)

    def unavailable_config(args):
        """只模拟外部配置失败并发出业务日志。参数args；返回无；异常ValueError。"""
        logging.getLogger("app.workflow.context").info("history_ctx sentinel")
        raise ValueError("expected test failure")

    monkeypatch.setattr(module, "build_config", unavailable_config)
    directory = tmp_path / "report"
    assert module.main(["--test","--env-file","unused.env","--profile","default","--report-dir",str(directory)]) == 3
    assert "history_ctx sentinel" in (directory / "app.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("defect", ["missing_check", "reported_error", "non_boolean"])
def test_browser_acceptance_requires_all_checks_and_no_error(defect):
    """浏览器缺项、驱动报错或非布尔结果均不能通过验收。

    参数：defect为缺项、错误或非法结果三种报告缺陷；返回：None；异常：断言失败。
    """
    module = importlib.import_module("tests.verify_ch07_live")
    browser = {"error": None, "checks": {
        "reply_blocks_switch": True, "twenty_one_frontend_turns": True,
        "new_clears_only_page": True, "three_distinct_sessions": True,
        "full_original_history": True, "refresh_restores_history": True,
        "continues_old_session": True, "sidebar_failure_keeps_chat": True,
        "chat_after_sidebar_failure": True,
    }}
    report = {"turns": [{"session_id": "main", "query": "第一问", "answer": "有回答", "failures": []}],
        "cleanup": {"restored": True}, "summary_cleanup_restored": True,
        "saved_segments": [], "browser": browser}
    cases = {"queries": ["第一问"]}
    assert module.final_checks(report, cases, "main", [{}], "default", True)["browser"] is True
    if defect == "missing_check":
        del browser["checks"]["chat_after_sidebar_failure"]
    elif defect == "reported_error":
        browser["error"] = "截图或驱动执行失败"
    else:
        browser["checks"]["full_original_history"] = "未核验"
    assert module.final_checks(report, cases, "main", [{}], "default", True)["browser"] is False
