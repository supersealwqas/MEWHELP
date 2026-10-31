"""评估资源身份与实际模型证据预算的纯离线保护测试。"""

import pytest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.knowledge.retrieval import RetrievalCandidate
from tests.knowledge.fakes import make_settings
from tests.rag.test_retrieval import _chunk


class EvaluationConnection:
    """模拟会话级命名锁，区分原连接、重新连接及锁转移。"""

    def __init__(self):
        """建立独占连接事实。参数：无；返回：None；异常：无。"""
        self.current, self.owner, self.lost = 41, 41, False
        self.release_result = 1
        self.queries = []

    async def scalar(self, statement, parameters=None):
        """返回真实SQL边界所需事实。参数：statement/parameters SQL及绑定；返回：标量；异常：ConnectionError。"""
        from app.evaluation.runtime import EVAL_DATABASE
        sql = str(statement)
        self.queries.append(sql)
        if self.lost:
            raise ConnectionError("isolated lost connection")
        if "CONNECTION_ID" in sql:
            return self.current
        if "IS_USED_LOCK" in sql:
            return self.owner
        if "RELEASE_LOCK" in sql:
            if isinstance(self.release_result, Exception):
                raise self.release_result
            return self.release_result
        return 1 if "GET_LOCK" in sql else EVAL_DATABASE

    async def execute(self, statement, parameters=None):
        """模拟控制SQL，断线后释放也失败。参数：statement/parameters；返回：None；异常：ConnectionError。"""
        await self.scalar(statement, parameters)

    async def commit(self):
        """结束短事务。参数：无；返回：None；异常：ConnectionError，断线时。"""
        if self.lost:
            raise ConnectionError("isolated lost connection")

    def in_transaction(self):
        """返回没有未提交业务。参数：无；返回：False；异常：无。"""
        return False


@pytest.fixture
def evaluation_boundaries(monkeypatch):
    """仅隔离数据库、索引及模型，保留真实锁、运行时和评估编排。

    参数：monkeypatch 外部依赖替换器；返回：连接、配置、上下文工厂和工作记录。
    异常：替换目标错误或断言失败。
    """
    import app.evaluation.runtime as module
    from sqlalchemy import URL
    from app.evaluation.faithfulness import FaithfulnessDecision, FactJudgment
    from app.knowledge.answering import KnowledgeAnswer
    from app.knowledge.query_rewrite import QueryRewrite
    from app.knowledge.retrieval import EvaluationRetrieval

    connection, chunk, calls = EvaluationConnection(), _chunk(1), []
    control = SimpleNamespace(loss_at=None, fail_stage=False)

    @asynccontextmanager
    async def connect():
        """提供隔离物理连接。参数：无；返回：连接；异常：调用方异常原样传播。"""
        yield connection

    @asynccontextmanager
    async def storage(settings):
        """隔离存储会话。参数：settings 专用配置；返回：仓储边界；异常：调用方异常原样传播。"""
        yield object()

    def record(stage):
        """记录新工作并在指定边界断线。参数：stage 阶段；返回：None；异常：RuntimeError，可模拟模型同时失败。"""
        calls.append(stage)
        if control.loss_at == stage:
            connection.lost = True
            if control.fail_stage:
                raise RuntimeError("isolated model failure")

    async def rewrite(query):
        """提供合法模型改写。参数：query 原问；返回：改写；异常：RuntimeError，可控外部失败。"""
        record("rewrite")
        return QueryRewrite(standard_query=query, synonyms=[])

    async def retrieve(*args, **kwargs):
        """提供真实候选协议。参数：检索参数；返回：排名及证据；异常：RuntimeError，可控外部失败。"""
        record("retrieve")
        item = RetrievalCandidate(chunk=chunk, score=0.9)
        return EvaluationRetrieval(ranked=[item], selected=[item])

    async def answer(messages):
        """提供可引用答案。参数：messages 模型消息；返回：知识答案；异常：RuntimeError，可控外部失败。"""
        record("generate")
        return KnowledgeAnswer(useful=True, reason="", answer="权威答案1。[1]", citation_numbers=[1])

    async def judge(query, answer, citations):
        """提供合法逐事实裁判。参数：问答及证据；返回：裁判；异常：RuntimeError，可控外部失败。"""
        record("judge")
        return FaithfulnessDecision(facts=[FactJudgment(claim="权威答案1。", supported=True,
            reason="证据1支持", evidence_numbers=[1])])

    engine = SimpleNamespace(connect=connect, dispose=AsyncMock())
    target = SimpleNamespace(url=URL.create("mysql+asyncmy", username="offline", database=module.EVAL_DATABASE),
                             dispose=AsyncMock())
    resource = SimpleNamespace(aclose=AsyncMock(), ensure_ready=AsyncMock(), count_tokens=len)
    replacements = {
        "create_database": lambda config: (target, None), "create_async_engine": lambda *args, **kwargs: engine,
        "BgeM3Encoder": lambda config: resource, "BgeReranker": lambda config: resource,
        "MilvusHybridIndex": lambda config: resource, "knowledge_storage": storage,
        "_read_identity": AsyncMock(return_value={"corpus_digest": module.corpus_digest([chunk])}),
        "_all_chunks": AsyncMock(return_value=[chunk]), "local_corpus": lambda *args, **kwargs: [chunk],
        "_verify_vectors": AsyncMock(), "runtime_metadata": lambda *args, **kwargs: {},
        "EvaluationSearch": lambda *args: SimpleNamespace(retrieve=retrieve),
        "ChatModelAdapter": lambda: SimpleNamespace(answer_knowledge=answer),
        "QueryRewriter": lambda **kwargs: SimpleNamespace(rewrite=rewrite),
        "FaithfulnessJudge": lambda **kwargs: SimpleNamespace(judge=judge),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(module, name, replacement)
    config = module.eval_config(make_settings(), target="eval")
    return SimpleNamespace(connection=connection, calls=calls, control=control, config=config,
                           runtime=module.evaluation_runtime, engine=engine, target=target)


@pytest.mark.parametrize("stage,expected", [
    ("initial", []), ("rewrite", ["rewrite"]), ("retrieve", ["rewrite", "retrieve"]),
    ("generate", ["rewrite", "retrieve", "generate"]),
    ("judge", ["rewrite", "retrieve", "generate", "judge"]),
])
async def test_lock_loss_stops_next_stage_and_invalidates_artifacts(tmp_path, evaluation_boundaries, stage, expected):
    """失锁必须停止后续阶段，不能逐题吞掉或发布完成成绩。参数：目录、隔离边界、阶段和预期工作；返回：None；异常：断言失败。"""
    import json
    from app.evaluation.runner import evaluate_dataset
    from app.knowledge.errors import LeaseLost
    from tests.rag.test_eval_runner import case

    boundary, output = evaluation_boundaries, tmp_path / "lost"
    boundary.control.loss_at = stage
    with pytest.raises(LeaseLost):
        async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused") as runtime:
            if stage == "initial":
                boundary.connection.lost = True
            await evaluate_dataset([case(), case()], runtime, output, strategies=["dense"],
                                   evidence_resolver=lambda *args: [{1}])
    assert boundary.calls == expected
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "invalidated"
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"] == "invalidated"


@pytest.mark.parametrize("current,owner", [(41, 99), (41, None), (99, 99)])
async def test_lock_owner_and_original_connection_are_both_required(tmp_path, evaluation_boundaries, current, owner):
    """新连接即使持同名锁也不能接续旧轮次。参数：隔离边界与当前连接/持锁者；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset
    from app.knowledge.errors import LeaseLost
    from tests.rag.test_eval_runner import case

    boundary = evaluation_boundaries
    with pytest.raises(LeaseLost):
        async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused") as runtime:
            boundary.connection.current, boundary.connection.owner = current, owner
            await evaluate_dataset([case()], runtime, tmp_path / "replaced", strategies=["dense"],
                                   evidence_resolver=lambda *args: [{1}])
    assert boundary.calls == []


@pytest.mark.parametrize("between_cases", [False, True])
async def test_lock_loss_after_saved_row_stops_new_work(tmp_path, evaluation_boundaries, between_cases):
    """题间或策略间失锁保留旧行并阻止新增工作。参数：隔离边界及切换种类；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset
    from app.knowledge.errors import LeaseLost
    from tests.rag.test_eval_runner import case

    boundary, output = evaluation_boundaries, tmp_path / "between"

    def lose_after_row(row):
        """已落盘后模拟断线。参数：row 完成行；返回：None；异常：无。"""
        boundary.connection.lost = True

    with pytest.raises(LeaseLost):
        async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused") as runtime:
            await evaluate_dataset([case(), case()] if between_cases else [case()], runtime, output,
                strategies=["dense"] if between_cases else ["dense", "bm25"],
                evidence_resolver=lambda *args: [{1}], progress=lose_after_row)
    assert boundary.calls == ["rewrite", "retrieve", "generate", "judge"]
    assert len((output / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize("filename", ["summary.json", "manifest.json"])
async def test_lock_loss_during_completion_revokes_completed_artifacts(tmp_path, monkeypatch, evaluation_boundaries, filename):
    """完成文件写出期间断线，也要撤销已发布完成状态。参数：隔离边界及触发文件；返回：None；异常：断言失败。"""
    import json
    from app.evaluation import runner
    from app.knowledge.errors import LeaseLost
    from tests.rag.test_eval_runner import case

    boundary, output, original = evaluation_boundaries, tmp_path / "completion", runner.atomic_json

    def lose_after_write(path, value):
        """执行真实落盘后模拟断线。参数：path/value 文件和内容；返回：None；异常：文件错误。"""
        original(path, value)
        if path.name == filename and value.get("status") == "completed":
            boundary.connection.lost = True

    monkeypatch.setattr(runner, "atomic_json", lose_after_write)
    with pytest.raises(LeaseLost):
        async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused") as runtime:
            await runner.evaluate_dataset([case()], runtime, output, strategies=["dense"],
                                          evidence_resolver=lambda *args: [{1}])
    for name in ("summary.json", "manifest.json"):
        assert json.loads((output / name).read_text(encoding="utf-8"))["status"] == "invalidated"
    assert "invalidated" in (output / "summary.md").read_text(encoding="utf-8")


async def test_lock_loss_with_model_error_still_aborts_run(tmp_path, evaluation_boundaries):
    """模型失败与失锁同时发生时，不能只记题目错误继续运行。参数：隔离边界；返回：None；异常：断言失败。"""
    from app.evaluation.runner import evaluate_dataset
    from app.knowledge.errors import LeaseLost
    from tests.rag.test_eval_runner import case

    boundary = evaluation_boundaries
    boundary.control.loss_at, boundary.control.fail_stage = "rewrite", True
    with pytest.raises(LeaseLost):
        async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused") as runtime:
            await evaluate_dataset([case(), case()], runtime, tmp_path / "model-failure",
                strategies=["dense", "bm25"], evidence_resolver=lambda *args: [{1}])
    assert boundary.calls == ["rewrite"]


async def test_lock_cleanup_preserves_original_exception(tmp_path, evaluation_boundaries):
    """断线后的释放错误不能覆盖原始失败，资源仍全部释放。参数：隔离边界；返回：None；异常：断言失败。"""
    boundary, original = evaluation_boundaries, LookupError("original failure")
    with pytest.raises(LookupError) as caught:
        async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused"):
            boundary.connection.lost = True
            raise original
    assert caught.value is original
    assert boundary.engine.dispose.await_count == boundary.target.dispose.await_count == 1


@pytest.mark.parametrize("failure", ["after_runner", "release_zero", "release_null", "release_error", "readonly_markdown"])
async def test_cli_late_lock_loss_invalidates_finished_run(tmp_path, monkeypatch, evaluation_boundaries, capsys, failure):
    """运行器返回后失锁，CLI必须撤销完成报告及不可用校准策略。参数：隔离边界；返回：None；异常：断言失败。"""
    import json
    from app import config
    from app.evaluation import runner, dataset
    from app.evaluation.cli import build_parser, run
    from pathlib import Path
    from tests.rag.test_eval_runner import case

    boundary, output, original = evaluation_boundaries, tmp_path / "late", runner.evaluate_dataset
    sample = case()
    sample.split = "calibration"
    sample.model_dump = lambda **kwargs: {"id": sample.id, "split": sample.split}
    blocked_paths, write_text = set(), Path.write_text

    def write_display(path, text, *args, **kwargs):
        """只隔离展示文件的权限错误，其余产物真实落盘。参数：文件、内容及写入参数；返回：字符数；异常：PermissionError。"""
        if path in blocked_paths:
            raise PermissionError("isolated readonly Markdown")
        return write_text(path, text, *args, **kwargs)

    async def lose_after_evaluation(*args, **kwargs):
        """保留真实评估，再模拟退出上下文前断线。参数：评估参数；返回：原报告；异常：真实评估错误。"""
        result = await original(*args, **kwargs, evidence_resolver=lambda *args: [{1}])
        if failure == "after_runner":
            boundary.connection.lost = True
        else:
            boundary.connection.release_result = {"release_zero": 0, "release_null": None,
                "release_error": ConnectionError("release disconnected"), "readonly_markdown": 0}[failure]
        if failure == "readonly_markdown":
            blocked_paths.add(output / "summary.md")
        return result

    monkeypatch.setattr(config, "Settings", lambda: boundary.config)
    monkeypatch.setattr(dataset, "load_dataset", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(runner, "evaluate_dataset", lose_after_evaluation)
    monkeypatch.setattr(Path, "write_text", write_display)
    args = build_parser().parse_args(["calibrate", "--output-dir", str(output), "--strategies", "dense"])
    assert await run(args) == 1
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"] == "invalidated"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "invalidated"
    assert not (output / "policy.json").exists()
    assert json.loads((output / "startup-error.json").read_text(encoding="utf-8"))["error_type"] == "LeaseLost"
    assert '"status": "completed"' not in capsys.readouterr().out


@pytest.mark.parametrize("stage", ["rewrite", "completion"])
async def test_runner_readonly_markdown_preserves_lock_loss(tmp_path, monkeypatch, evaluation_boundaries, stage, caplog):
    """展示清理失败不能覆盖整轮失锁，前后两种报告边界都保留权威失效状态。

    参数：临时目录、权限边界替换器、外部运行资源、失锁阶段和日志捕获。
    返回：None；异常：AssertionError，原锁异常或权威状态被二次写盘失败破坏。
    """
    import json
    from pathlib import Path
    from app.evaluation.runner import evaluate_dataset
    from app.knowledge.errors import LeaseLost
    from tests.rag.test_eval_runner import case

    boundary, output, write_text = evaluation_boundaries, tmp_path / "readonly", Path.write_text
    boundary.control.loss_at = stage

    def readonly_display(path, text, *args, **kwargs):
        """模拟展示文件被占用，其余产物继续真实写盘。参数：文件和写入数据；返回：字符数；异常：PermissionError。"""
        if path == output / "summary.md":
            if boundary.connection.lost:
                raise PermissionError("isolated readonly Markdown")
            result = write_text(path, text, *args, **kwargs)
            boundary.connection.lost = True
            return result
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", readonly_display)
    with pytest.raises(LeaseLost):
        async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused") as runtime:
            await evaluate_dataset([case()], runtime, output, strategies=["dense"],
                                   evidence_resolver=lambda *args: [{1}])
    for name in ("manifest.json", "summary.json"):
        assert json.loads((output / name).read_text(encoding="utf-8"))["status"] == "invalidated"
    assert "PermissionError" in caplog.text


async def test_healthy_lock_completes_all_cases_and_strategies(tmp_path, evaluation_boundaries):
    """合法持锁仍完成全部比较，单题改写继续由策略共享。参数：隔离边界；返回：None；异常：断言失败。"""
    import json
    from app.evaluation.runner import evaluate_dataset
    from tests.rag.test_eval_runner import case

    boundary, output = evaluation_boundaries, tmp_path / "healthy"
    async with boundary.runtime(boundary.config, manifest_path=tmp_path / "unused") as runtime:
        report = await evaluate_dataset([case(), case()], runtime, output, strategies=["dense", "bm25"],
                                        evidence_resolver=lambda *args: [{1}])
    assert report["status"] == "completed"
    assert boundary.calls == ["rewrite", "retrieve", "generate", "judge", "retrieve", "generate", "judge"] * 2
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["completed_sample_count"] == 4


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
