"""离线测试配置隔离回归：在独立进程中验证导入顺序和显式配置来源。"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest


@pytest.mark.parametrize("import_order", ["delayed", "early", "stale"])
def test_offline_settings_ignore_dotenv_and_cached_bindings(import_order: str) -> None:
    """验证离线模型与服务始终使用虚拟配置，不受导入时机或旧引用影响。

    参数:
        import_order: delayed 为延迟导入，early 为提前导入，stale 为替换后的旧引用。
    返回值:
        None: 断言独立 pytest 进程中的配置来源和模型构造参数。
    异常:
        AssertionError: 临时哨兵配置泄漏到离线服务或模型时抛出。
        OSError、subprocess.TimeoutExpired: 临时文件或子进程不可用时抛出。
    """
    early_import = "from app import config, main\nfrom app.core import llm\nfrom app.db import session\n"
    source = """def test_offline_configuration(offline_chat_runtime):
    \"\"\"检查消费者配置。参数：离线服务夹具；返回：None；异常：断言失败。\"\"\"
    from app import config, main
    from app.core import llm
    from app.db import session
    import pytest
    model = llm.get_chat_model()
    assert model.model_name == "test-model"
    assert str(model.openai_api_base).rstrip("/") == "http://test.invalid/v1"
    assert model.openai_api_key.get_secret_value() == "test-key"
    assert config.settings.token_budget == 2000
    assert config.settings.mysql_host == "127.0.0.1"
    assert config.settings.mysql_password.get_secret_value() == ""
    service, _, _, _ = offline_chat_runtime
    assert service.settings is config.settings
    assert main.settings is config.settings
    assert llm.settings is config.settings
    assert session.settings is config.settings
    # 未显式提供数据库配置时，隔离后的空密码必须阻止创建连接资源。
    with pytest.raises(ValueError, match="MYSQL_PASSWORD"):
        session.create_database()
"""
    if import_order != "delayed":
        source = early_import + source
    if import_order == "stale":
        # 先换配置再导入消费者，同时覆盖模块引用和函数默认参数缓存的旧对象。
        source = """from app import config
stale = config.Settings(
    _env_file=None, chat_base_url="http://stale.invalid/v1",
    chat_model="stale-model", chat_api_key="stale-key", token_budget=17,
    mysql_host="stale.invalid", mysql_password="stale-password",
)
config.settings = stale
""" + source
    _run_isolated_pytest(source)


def test_mysql_marker_keeps_explicit_dotenv_semantics() -> None:
    """验证 MySQL 标记保留显式配置文件读取，同时不启动数据库夹具。

    参数:
        无。
    返回值:
        None: 断言显式文件中的虚拟账号和数据库参数被保留。
    异常:
        AssertionError: 全局隔离污染显式配置读取或未跳过离线运行时时抛出。
        OSError、subprocess.TimeoutExpired: 临时文件或子进程不可用时抛出。
    """
    _run_isolated_pytest("""from pathlib import Path
import pytest

@pytest.mark.mysql
def test_explicit_dotenv(offline_chat_runtime):
    \"\"\"检查显式文件。参数：离线服务夹具；返回：None；异常：断言失败。\"\"\"
    import asyncio
    from app.config import Settings
    from app.db.session import create_database
    config = Settings(_env_file=Path.cwd() / ".env")
    assert offline_chat_runtime is None
    assert config.chat_model == "dotenv-sentinel-model"
    assert config.chat_api_key == "dotenv-sentinel-key"
    assert config.mysql_host == "dotenv-sentinel.invalid"
    assert config.mysql_password.get_secret_value() == "dotenv-sentinel-password"
    assert config.token_budget == 31
    # 引擎仅保存连接参数，不调用 connect/begin，避免启动真实数据库连接。
    engine, _ = create_database(config, database="isolated_test")
    try:
        assert engine.url.host == "dotenv-sentinel.invalid"
        assert engine.url.password == "dotenv-sentinel-password"
        assert engine.url.database == "isolated_test"
    finally:
        asyncio.run(engine.dispose())
""")


def _run_isolated_pytest(source: str) -> None:
    """在仅含虚拟配置的临时目录中运行项目真实测试夹具。

    参数:
        source: 单个同步探针测试模块的 Python 源码，不执行模型请求或数据库连接。
    返回值:
        None: 子进程成功结束后返回。
    异常:
        AssertionError: 探针测试失败时附带仅含虚拟配置的输出。
        OSError、subprocess.TimeoutExpired: 临时目录或子进程不可用时抛出。
    """
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    # 不继承开发机的业务配置；子进程只可看到本测试写入的哨兵文件。
    for key in tuple(environment):
        if key.upper().startswith(("CHAT_", "EMBED_", "RERANK_", "MYSQL_", "TOOL_")) or key.upper() in {
            "TOKEN_BUDGET", "REQUEST_TOKEN_BUDGET",
        }:
            environment.pop(key)
    environment["PYTHONPATH"] = str(root)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with TemporaryDirectory(prefix="mewhelp-isolation-") as directory:
        workspace = Path(directory)
        shutil.copyfile(root / "tests" / "conftest.py", workspace / "conftest.py")
        (workspace / ".env").write_text(
            "CHAT_BASE_URL=http://dotenv-sentinel.invalid/v1\n"
            "CHAT_MODEL=dotenv-sentinel-model\n"
            "CHAT_API_KEY=dotenv-sentinel-key\n"
            "TOKEN_BUDGET=31\n"
            "MYSQL_HOST=dotenv-sentinel.invalid\n"
            "MYSQL_PASSWORD=dotenv-sentinel-password\n",
            encoding="utf-8",
        )
        (workspace / "pytest.ini").write_text(
            "[pytest]\nmarkers = mysql: 仅验证显式配置，不连接数据库\n",
            encoding="utf-8",
        )
        (workspace / "test_probe.py").write_text(source, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "test_probe.py"],
            cwd=workspace, env=environment, text=True, encoding="utf-8",
            errors="replace", capture_output=True, timeout=60, check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
