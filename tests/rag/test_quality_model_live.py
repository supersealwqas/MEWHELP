"""显式真实网关验证 useful 自评与证据边界；默认不联网、不产生模型费用。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.core.llm import ChatModelAdapter
from app.knowledge.answering import KNOWLEDGE_ANSWER_RULES, decide_answer, EvidenceCitation


@pytest.mark.skipif(os.getenv("MEWHELP_RUN_QUALITY_MODEL") != "1", reason="需要显式 MEWHELP_RUN_QUALITY_MODEL=1，调用实际网关")
async def test_real_model_useful_true_and_false_use_only_current_evidence(monkeypatch) -> None:
    """真实模型须按足够/无证据输出 useful，不把处理时限混成到账。参数：替换器；返回：None；异常：网络/断言错误。"""
    from app.core import llm

    path = Path(os.getenv("MEWHELP_TEST_ENV_FILE", ".env"))
    values = dotenv_values(path)
    config = Settings(_env_file=path, **{key.lower(): values[key] for key in ("CHAT_BASE_URL", "CHAT_MODEL", "CHAT_API_KEY")})
    monkeypatch.setattr(llm, "settings", config)
    model = ChatModelAdapter()
    citation = EvidenceCitation(n=1, chunk_id=1, section_path="售后 / 退款处理", question="退款处理时限",
                                answer="退款处理时限为3个工作日，不承诺退款到账时间。")
    messages = [SystemMessage(content=KNOWLEDGE_ANSWER_RULES),
                HumanMessage(content='本轮证据hits:[{"n":1,"answer":"退款处理时限为3个工作日，不承诺退款到账时间。"}]\n当前问题：退款处理要几个工作日？')]
    answer = await model.answer_knowledge(messages)
    assert answer.useful is True and "3" in answer.answer
    decide_answer(answer, [citation])
    missing = await model.answer_knowledge([SystemMessage(content=KNOWLEDGE_ANSWER_RULES),
        HumanMessage(content="本轮证据hits:[]\n当前问题：型号MH-UNKNOWN-999在零下50度能稳定工作多久？")])
    assert missing.useful is False
    assert decide_answer(missing, [])[1].source == "self_check"
    from app.evaluation.faithfulness import FaithfulnessJudge
    judgment = await FaithfulnessJudge(model_name=config.chat_model).judge(
        "退款多久能到账", "保证2个工作日到账[1]", [citation])
    assert any(not fact.supported for fact in judgment.facts)
