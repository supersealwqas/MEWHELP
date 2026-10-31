"""评估CLI参数和校准策略版本绑定的离线测试。"""

import pytest


def test_production_cli_requires_explicit_readonly_target_and_policy():
    """生产路径不得回落旧评估准备或省略冻结策略。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["evaluate-production", "--target", "application-readonly", "--policy", "p.json",
        "--output-dir", "out", "--triggered-by", "定时"])
    assert args.triggered_by == "定时"
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate-production", "--policy", "p.json", "--output-dir", "out"])
    assert parser.parse_args(["persist-run", "--artifact-dir", "out"]).artifact_dir.name == "out"


@pytest.mark.parametrize("triggered_by", ["手动", "定时"])
async def test_persist_run_never_repeats_production_evaluation(tmp_path, monkeypatch, triggered_by):
    """补存只核验现成产物和写仓储，不能重新构建生产运行时。参数：目录/外部数据库替身；返回：None；异常：断言失败。"""
    from contextlib import asynccontextmanager
    from app.evaluation.cli import build_parser, run
    from tests.flywheel.test_eval_runs import completed_artifact
    artifact = completed_artifact(tmp_path / "run", triggered_by)
    stored = []

    class Repository:
        """隔离数据库写入，保留CLI本地完整验证。"""
        async def save_completed_run(self, path, triggered_by):
            """记录实际补存请求。参数：路径/来源；返回：保存结果；异常：无。"""
            stored.append((path, triggered_by))
            return {"id": "11", "saved": True}

    @asynccontextmanager
    async def repository(config):
        """隔离真实数据库连接。参数：配置；返回：仓储；异常：无。"""
        yield Repository()

    def forbidden(*args, **kwargs):
        """任何生产资源构造均违反无付费补存合同。参数：任意；返回：无；异常：AssertionError。"""
        raise AssertionError("补存不得重新评估")

    monkeypatch.setattr("app.repositories.evaluations.evaluation_repository_runtime", repository)
    monkeypatch.setattr("app.evaluation.production.production_runtime", forbidden)
    assert await run(build_parser().parse_args(["persist-run", "--artifact-dir", str(artifact)])) == 0
    assert stored == [(artifact, triggered_by)]


def test_cli_requires_explicit_prepare_target_and_fixed_split() -> None:
    """prepare不默认指向任意库，固定验收不能误当校准。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["validate"]).command == "validate"
    assert parser.parse_args(["prepare", "--target", "eval"]).target == "eval"
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare"])
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate", "--split", "evaluation"])
    with pytest.raises(SystemExit):
        parser.parse_args(["compare", "--split", "calibration", "--policy", "policy.json"])
    assert parser.parse_args(["compare", "--policy", "p.json", "--strategies", "bm25", "dense"]).strategies == ["bm25", "dense"]


def test_cli_can_override_verification_ports_without_editing_env() -> None:
    """独占验证服务端口通过显式参数覆盖，不改用户环境文件。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.cli import build_parser

    args = build_parser().parse_args(["prepare", "--target", "eval", "--mysql-port", "13307", "--milvus-uri", "http://127.0.0.1:19540"])
    assert args.mysql_port == 13307
    assert args.milvus_uri == "http://127.0.0.1:19540"


def test_policy_mismatch_and_missing_strategy_fail_closed() -> None:
    """题集/语料/Prompt/模型版本不同时不能套用旧校准门槛。参数：无；返回：None；异常：断言失败。"""
    from app.evaluation.policy import POLICY_BINDINGS, POLICY_VERSION, validate_policy

    metadata = {key: "same" for key in POLICY_BINDINGS}
    policy = {"version": POLICY_VERSION, "metadata": metadata,
              "calibration_case_count": 24, "thresholds": {"bm25": {"minimum": None}}}
    assert validate_policy(policy, metadata, ["bm25"]) == {"bm25": None}
    with pytest.raises(ValueError):
        validate_policy(policy, {**metadata, "corpus_digest": "changed"}, ["bm25"])
    with pytest.raises(ValueError):
        validate_policy(policy, metadata, ["dense"])


async def test_startup_failure_does_not_write_into_competing_run(tmp_path, monkeypatch, offline_settings) -> None:
    """检测到目录竞争时不得给另一运行追加失败文件。参数：tmp_path/monkeypatch；返回：None；异常：断言失败。"""
    from app.evaluation.cli import build_parser, run
    from app.evaluation import runtime
    from app import config

    output = tmp_path / "competing"

    def competing_runtime(*args, **kwargs):
        """模拟其他进程抢先创建同名目录。参数：任意运行时参数；返回：不返回；异常：RuntimeError。"""
        output.mkdir()
        (output / "sentinel").write_text("other-run", encoding="utf-8")
        raise RuntimeError("启动失败")

    monkeypatch.setattr(runtime, "evaluation_runtime", competing_runtime)
    monkeypatch.setattr(config, "Settings", lambda: offline_settings)
    args = build_parser().parse_args(["compare", "--policy", "unused.json", "--output-dir", str(output)])
    assert await run(args) == 1
    assert {path.name for path in output.iterdir()} == {"sentinel"}


def test_validate_runs_without_chat_credentials_or_env_file(tmp_path) -> None:
    """只验证本地题集不依赖聊天配置，也不偷偷注入假凭据。参数：tmp_path；返回：None；异常：断言失败。"""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CHAT_")}
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run([sys.executable, "-m", "app.evaluation.cli", "validate",
                             "--dataset", str(root / "data/evaluation/ch04.json"),
                             "--manifest", str(root / "data/kb/manifest.json")], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert '"case_count": 72' in result.stdout


def test_policy_accepts_any_calibration_count_when_unspecified() -> None:
    """不再写死 24 题：不传题数时不校验数量，只校验版本与绑定。

    参数：无。
    返回：None。
    异常：AssertionError，非 24 题的 policy 仍被拒绝。
    """
    from app.evaluation.policy import POLICY_BINDINGS, POLICY_VERSION, validate_policy

    metadata = {key: "same" for key in POLICY_BINDINGS}
    policy = {"version": POLICY_VERSION, "metadata": metadata,
              "calibration_case_count": 104, "thresholds": {"bm25": {"minimum": None}}}
    assert validate_policy(policy, metadata, ["bm25"]) == {"bm25": None}


def test_policy_rejects_count_that_differs_from_current_dataset() -> None:
    """显式给出本次题数时必须一致，避免把 24 题的校准套到 104 题的题集上。

    参数：无。
    返回：None。
    异常：AssertionError，题数不一致未报错。
    """
    from app.evaluation.policy import POLICY_BINDINGS, POLICY_VERSION, validate_policy

    metadata = {key: "same" for key in POLICY_BINDINGS}
    policy = {"version": POLICY_VERSION, "metadata": metadata,
              "calibration_case_count": 24, "thresholds": {"bm25": {"minimum": None}}}
    assert validate_policy(policy, metadata, ["bm25"], calibration_case_count=24) == {"bm25": None}
    with pytest.raises(ValueError):
        validate_policy(policy, metadata, ["bm25"], calibration_case_count=104)
