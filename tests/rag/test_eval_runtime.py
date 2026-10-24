"""评估资源身份与实际模型证据预算的纯离线保护测试。"""

import pytest

from app.knowledge.retrieval import RetrievalCandidate
from tests.knowledge.fakes import make_settings
from tests.rag.test_retrieval import _chunk


async def test_runtime_prepare_passes_only_current_question_to_model():
    """正式评估准备接入真实改写解析，不读取标注。参数：无；返回：None；异常：断言失败。"""
    import json
    from app.evaluation.runtime import EvaluationRuntime
    from tests.rag.test_query_rewrite import RewriteModel

    runtime = EvaluationRuntime(None, [], make_settings(), evidence_budget=8000, metadata={})
    model = RewriteModel()
    runtime.query_rewriter.model_factory = model.factory
    prepared = await runtime.prepare("饮水机脏了咋洗")
    assert prepared.raw == "饮水机脏了咋洗" and len(model.inputs) == 1
    assert json.loads(model.inputs[0][-1].content) == {"query": "饮水机脏了咋洗", "keyword": None}
    assert prepared.groups == [["饮水机怎么清洗"]]


def test_only_named_eval_resources_are_allowed() -> None:
    """任何半边应用资源配置必须被拒绝。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.runtime import eval_config, validate_resources

    config = eval_config(make_settings(), target="eval")
    assert config.mysql_database == config.milvus_database == "mewhelp_ch04_eval"
    assert config.milvus_hybrid_collection == "knowledge_ch04_eval"
    validate_resources(config)
    with pytest.raises(ValueError):
        validate_resources(config.model_copy(update={"mysql_database": "mewhelp"}))
    with pytest.raises(ValueError):
        eval_config(make_settings(), target="production")


def test_evidence_budget_drops_whole_chunks_and_matches_messages() -> None:
    """模型见到的JSON与保存快照完全一致，预算不足只去掉整块。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.runtime import generation_input
    from app.core.memory import _estimate_message_tokens
    import json

    chunks = [_chunk(1), _chunk(2)]
    chunks[1] = chunks[1].model_copy(update={"text": chunks[1].text.model_copy(update={"answer": "长" * 2000})})
    messages, citations, dropped = generation_input("问题", [RetrievalCandidate(chunk=c, score=0.1) for c in chunks],
                                                    evidence_budget=400, request_budget=3000)
    assert [item.chunk_id for item in citations] == [1]
    assert dropped == [2]
    payload = json.loads(messages[-1].content)
    # 模型只看 n 与正文；chunk_id 是服务端内部主键，不进模型上下文。
    assert payload["hits"] == [citation.model_dump(exclude={"chunk_id"}) for citation in citations]
    assert all("chunk_id" not in hit for hit in payload["hits"])
    assert _estimate_message_tokens(messages) <= 3000


def test_evaluation_handoff_keeps_original_query_and_budgeted_evidence() -> None:
    """评估请求须明确交接生成任务，原话和完整实际证据仍一致且计入请求预算。

    参数：无；原问题刻意含空白、否定和换行。
    返回：None，交接与预算均符合要求时正常结束。
    异常：AssertionError，原问/证据改变或交接未进入预算内实际输入。
    """
    import json
    from app.core.memory import _estimate_message_tokens
    from app.evaluation.runtime import generation_input

    raw = '  MH-W99容量是多少？\n不要换成MH-W20。  '
    candidates = [RetrievalCandidate(chunk=_chunk(1), score=0.5)]
    messages, citations, dropped = generation_input(raw, candidates, evidence_budget=8000, request_budget=16000)
    payload = json.loads(messages[-1].content)
    assert payload["query"] == raw
    assert isinstance(payload.get("instruction"), str) and payload["instruction"]
    assert payload["hits"] == [item.model_dump(exclude={"chunk_id"}) for item in citations]
    assert _estimate_message_tokens(messages) <= 16000 and dropped == []


def test_existing_database_identity_rejects_unowned_or_foreign_tables() -> None:
    """同名库不等于本项目可接管资源。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.runtime import check_identity, EVAL_MARKER

    with pytest.raises(ValueError):
        check_identity({"knowledge_chunks"}, None)
    with pytest.raises(ValueError):
        check_identity({"ch04_eval_identity", "knowledge_chunks", "messages"}, {"marker": EVAL_MARKER})
    with pytest.raises(ValueError):
        check_identity({"ch04_eval_identity", "knowledge_chunks"}, {"marker": "foreign"})
    check_identity({"ch04_eval_identity", "knowledge_chunks"}, {"marker": EVAL_MARKER,
                    "collection_name": "knowledge_ch04_eval"})


async def test_bootstrap_connects_without_requiring_database_to_exist(monkeypatch) -> None:
    """首次建库连接不得提前选择不存在的评估库。参数：monkeypatch；返回：None；异常：断言失败。"""
    from app.evaluation import runtime
    from sqlalchemy import URL
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    original = URL.create("mysql+asyncmy", username="offline", database="mewhelp_ch04_eval")
    monkeypatch.setattr(runtime, "create_database", lambda config: (SimpleNamespace(url=original, dispose=AsyncMock()), None))

    def observe(url, **kwargs):
        """核对真实URL变换结果后停止外部连接。参数：url/kwargs；返回：不返回；异常：LookupError或断言失败。"""
        assert url.database is None
        raise LookupError("connection boundary reached")

    monkeypatch.setattr(runtime, "create_async_engine", observe)
    with pytest.raises(LookupError, match="connection boundary"):
        async with runtime._resource_lock(runtime.eval_config(make_settings(), target="eval"), create=True):
            pass


async def test_identity_reflection_explicitly_selects_eval_schema(monkeypatch) -> None:
    """无库连接执行USE不更新SQLAlchemy默认schema，反射必须显式传入专用库。参数：monkeypatch；返回：None；异常：断言失败。"""
    from app.evaluation import runtime
    from types import SimpleNamespace

    def table_names(*, schema=None):
        """检查反射边界传入的真实schema。参数：schema；返回：不返回；异常：LookupError。"""
        assert schema == "mewhelp_ch04_eval"
        raise LookupError("schema confirmed")

    async def run_sync(callback):
        """执行真实反射回调。参数：callback；返回：调用结果；异常：原样传播。"""
        return callback(object())

    monkeypatch.setattr(runtime, "inspect", lambda connection: SimpleNamespace(get_table_names=table_names))
    with pytest.raises(LookupError, match="schema confirmed"):
        await runtime._read_identity(SimpleNamespace(run_sync=run_sync), create=True)


def test_evidence_order_after_budget_preserves_stable_numbers() -> None:
    """先裁低相关整块，再首尾排列，不重编号。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.runtime import generation_input
    import json

    candidates = [RetrievalCandidate(chunk=_chunk(index), score=0.5) for index in range(1, 11)]
    messages, citations, dropped = generation_input("问题", candidates, evidence_budget=8000, request_budget=16000)
    assert [item.n for item in citations] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert json.loads(messages[-1].content)["hits"] == [item.model_dump(exclude={"chunk_id"}) for item in citations]
    assert dropped == []


async def test_prepare_repairs_missing_indexes_on_existing_table() -> None:
    """DDL中断后的既存表必须补齐普通索引，不能被表级checkfirst跳过。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.runtime import _ensure_chunk_table
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            # 使用真实SQL引擎保留“表存在但无普通索引”的部分建表状态。
            connection.execute(text("CREATE TABLE knowledge_chunks (id INTEGER PRIMARY KEY, source_key TEXT, ordinal INTEGER, is_active INTEGER, vectorize_status TEXT)"))

            class ConnectionBoundary:
                """将异步run_sync边界接到真实内存SQL连接。"""

                async def run_sync(self, callback):
                    """执行实际SQLAlchemy DDL。参数：callback；返回：结果；异常：SQL错误。"""
                    return callback(connection)

            await _ensure_chunk_table(ConnectionBoundary())
            await _ensure_chunk_table(ConnectionBoundary())
            indexes = {item["name"]: item["column_names"] for item in inspect(connection).get_indexes("knowledge_chunks")}
            assert indexes == {"idx_chunks_active_status": ["is_active", "vectorize_status"],
                               "idx_chunks_source_key": ["source_key"], "idx_chunks_ordinal": ["source_key", "ordinal"]}
    finally:
        engine.dispose()


def test_judge_model_falls_back_to_chat_model_when_unset(monkeypatch) -> None:
    """未配置 JUDGE_MODEL 时裁判沿用对话模型，保持既有行为不变。

    参数：monkeypatch 清掉主机环境里的 JUDGE_MODEL。
    返回：None。
    异常：AssertionError，默认值不再是对话模型。
    """
    from app.config import Settings

    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    s = Settings(_env_file=None, chat_base_url="http://test.invalid/v1",
                 chat_model="answer-model", chat_api_key="k")
    assert s.judge_model is None
    assert (s.judge_model or s.chat_model) == "answer-model"


def test_judge_model_can_differ_from_chat_model() -> None:
    """显式配置 JUDGE_MODEL 时裁判使用独立模型，避免同模型自评偏差。

    参数：无。
    返回：None。
    异常：AssertionError，裁判模型未按配置切换。
    """
    from app.config import Settings

    s = Settings(_env_file=None, chat_base_url="http://test.invalid/v1",
                 chat_model="answer-model", chat_api_key="k", judge_model="judge-model")
    assert (s.judge_model or s.chat_model) == "judge-model"


async def test_evaluation_runtime_passes_configured_judge_model(monkeypatch) -> None:
    """评估运行时必须把配置的裁判模型名传给 FaithfulnessJudge。

    参数：monkeypatch 替换裁判构造。
    返回：None。
    异常：AssertionError，运行时仍用对话模型当裁判。
    """
    import app.evaluation.runtime as runtime_module
    from app.config import Settings

    observed = {}

    class Recorder:
        """记录构造时收到的模型名。"""

        def __init__(self, *, model_name=None):
            """保存模型名。参数：model_name 裁判模型；返回：None；异常：无。"""
            observed["judge"] = model_name

    monkeypatch.setattr(runtime_module, "FaithfulnessJudge", Recorder)
    config = Settings(_env_file=None, chat_base_url="http://test.invalid/v1",
                      chat_model="answer-model", chat_api_key="k", judge_model="judge-model")
    runtime_module.EvaluationRuntime(None, [], config, evidence_budget=100, metadata={})
    assert observed["judge"] == "judge-model"
