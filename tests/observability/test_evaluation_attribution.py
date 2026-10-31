"""使用真实模型客户端和 Langfuse 内存导出验证评估跨阶段归属及原输入不变。

只替换 HTTP 传输和知识检索外部边界，不访问模型网关、数据库或观测服务。
"""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.evaluation.faithfulness import FaithfulnessError, FaithfulnessJudge
from app.evaluation.production import ProductionEvaluationRuntime
from app.observability.costs import summarize_intent_usage
from app.workflow.model import WorkflowModelAdapter
from tests.observability.test_runtime import generations, metadata, observed_runtime
from tests.workflow.graph_fakes import dependencies, ranked
from tests.workflow.test_model import call_response, intent_response, sdk_runtime, understanding_response


def exported_usage(exporter):
    """将真实完成观测转换为用量汇总器入口，不补写类别或计量。

    参数：exporter 为本次内存导出器。
    返回：dict，真实用量汇总。
    异常：导出属性或汇总结构不合法时直接抛出。
    """
    return summarize_intent_usage([
        {"id": f"{span.context.span_id:016x}", "traceId": f"{span.context.trace_id:032x}",
         "type": span.attributes.get("langfuse.observation.type", "span").upper(),
         "name": span.name, "metadata": metadata(span)}
        for span in exporter.get_finished_spans()
    ])


@pytest.mark.parametrize("judge_failure", [False, True])
async def test_execute_and_judge_keep_each_case_trace_and_evaluation_usage(
        observed_runtime, sdk_runtime, judge_failure):
    """裁判丢失类别、离开本题追踪或沿用上题身份时失败。

    参数：observed_runtime 真实观测及内存导出；sdk_runtime 离线 HTTP；judge_failure 是否返回坏裁判协议。
    返回：None。
    异常：预期坏协议由 FaithfulnessError 表示，其余为断言或客户端错误。
    """
    observation, exporter = observed_runtime
    deps = dependencies(intents=["商品咨询"], ranked=[ranked(answer="完整政策正文")])

    def model_factory(**options):
        """挂生产回调，保留真实 SDK 序列化。参数：模型选项；返回：模型；异常：配置错误。"""
        options.setdefault("stream_usage", True)
        return sdk_runtime.factory(callbacks=[observation.callback_handler], **options)

    runtime = ProductionEvaluationRuntime(deps.search, deps.settings,
        model=WorkflowModelAdapter(model_factory=model_factory, stream_usage=True),
        query_expander=deps.query_expander, observability=observation)
    runtime.faithfulness_judge = FaithfulnessJudge(model_factory, usage_observer=runtime.observe_judge_usage)
    question_turns = []
    for query in ("政策条件是什么？", "政策原文是什么？"):
        facts = {"facts": [{"claim": "完整政策正文", "supported": True,
                            "reason": "完整原文支持", "evidence_numbers": [1]}]}
        sdk_runtime.responses.extend([
            understanding_response(query, usage=True), intent_response("商品咨询", usage=True),
            call_response("submit_knowledge_answer", json.dumps({"useful": True, "reason": "",
                "answer": "完整政策正文[1]", "citation_numbers": [1]}, ensure_ascii=False), usage=True),
            call_response("wrong_tool" if judge_failure else "submit_faithfulness",
                          json.dumps(facts, ensure_ascii=False), usage=True),
        ])
        result = await runtime.execute(query, {}, retrieval_only=False)
        assert result["refused"] is False and result["answer"] == "完整政策正文[1]"
        question_turns.append((runtime.last_attempt["capture"].turn_id, result.get("trace_id")))
        # 明确跨过 execute 的上下文退出点，再进入正式裁判方法。
        if judge_failure:
            with pytest.raises(FaithfulnessError):
                await runtime.judge(query, result["answer"], result["selected"])
        else:
            judgment = await runtime.judge(query, result["answer"], result["selected"])
            assert judgment.model_dump() == facts
        assert runtime.last_judge_usage["usage"]["total_tokens"] == 112
        assert result["budget"]["actual_tokens"] == 336
        assert json.loads(sdk_runtime.requests[-1]["messages"][-1]["content"]) == {
            "query": query, "answer": "完整政策正文[1]",
            "citations": [item.model_dump() for item in result["selected"]],
        }

    # 退出评估后产生普通调用，证明 evaluation 标签没有泄露到后续客服。
    chat_model = FakeMessagesListChatModel(responses=[AIMessage(content="普通回答", usage_metadata={
        "input_tokens": 11, "output_tokens": 3, "total_tokens": 14})], callbacks=[observation.callback_handler])
    await chat_model.ainvoke("普通问候")
    observation.client.flush()
    calls = generations(exporter)
    assert len(calls) == 9
    evaluation_calls = [span for span in calls if metadata(span).get("task_kind") == "evaluation"]
    assert len(evaluation_calls) == 8
    assert len({span.context.trace_id for span in evaluation_calls}) == 2
    assert len(set(question_turns)) == 2
    for turn_id, trace_id in question_turns:
        case_calls = [span for span in evaluation_calls if metadata(span).get("turn_id") == turn_id]
        assert len(case_calls) == 4 and len({span.context.trace_id for span in case_calls}) == 1
        assert trace_id == f"{case_calls[0].context.trace_id:032x}"
        roots = [span for span in exporter.get_finished_spans()
                 if span.name == "execution.evaluation" and metadata(span).get("turn_id") == turn_id]
        # SDK 为显式 trace_context 创建虚拟父级，按实际起始时间区分本题初次与续接执行。
        assert len(roots) == 2
        original, continued = sorted(roots, key=lambda span: span.start_time)
        assert continued.parent.span_id == original.context.span_id
    summary = exported_usage(exporter)
    assert [(group["task_kind"], group["request_count"], group["model_call_count"], group["total_tokens"])
            for group in summary["groups"]] == [("chat", 1, 1, 14), ("evaluation", 2, 8, 896)]
    assert summary["coverage"]["unknown_usage_count"] == 0


async def test_standalone_judge_is_explicitly_evaluation(observed_runtime, sdk_runtime):
    """独立裁判没有 execute 前置调用时也不能落入客服用量。

    参数：observed_runtime 真实观测；sdk_runtime 离线 HTTP。
    返回：None。
    异常：断言或真实客户端协议错误。
    """
    from tests.rag.test_faith_cases import _citation
    observation, exporter = observed_runtime
    deps = dependencies()

    def model_factory(**options):
        """挂观测到离线真实客户端。参数：模型选项；返回：模型；异常：配置错误。"""
        return sdk_runtime.factory(callbacks=[observation.callback_handler], stream_usage=True, **options)

    runtime = ProductionEvaluationRuntime(deps.search, deps.settings, model=deps.model,
        query_expander=deps.query_expander, observability=observation)
    runtime.faithfulness_judge = FaithfulnessJudge(model_factory, usage_observer=runtime.observe_judge_usage)
    sdk_runtime.responses.append(call_response("submit_faithfulness", json.dumps({"facts": [{
        "claim": "原证据1", "supported": True, "reason": "原文", "evidence_numbers": [1]}]}), usage=True))
    await runtime.judge("问题", "原证据1[1]", [_citation()])
    observation.client.flush()
    assert [metadata(span).get("task_kind") for span in generations(exporter)] == ["evaluation"]
    assert exported_usage(exporter)["groups"][0]["total_tokens"] == 112
