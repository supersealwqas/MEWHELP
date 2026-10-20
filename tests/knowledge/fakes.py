"""CH03 知识库测试替身：不加载 torch 的编码器与离线配置构造工具。"""

from __future__ import annotations

from app.config import Settings


def make_settings(**overrides: object) -> Settings:
    """构造离线测试配置。

    参数:
        **overrides: 需要覆盖的 Settings 字段；未指定字段使用固定的离线测试值。
    返回值:
        Settings: 不读取真实 .env 的完整配置实例。
    异常:
        pydantic.ValidationError: 覆盖值与字段约束冲突时抛出。
    """
    values: dict[str, object] = {
        "chat_base_url": "http://test.invalid/v1",
        "chat_model": "test-model",
        "chat_api_key": "test-key",
        "knowledge_model_path": "models/bge-m3",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakeEncoder:
    """实现与 BgeM3Encoder 相同接口的测试编码器。

    记录输入文本，返回固定的 1024 维单位向量，不导入或加载 torch。
    """

    def __init__(self, revision: str = "fake-encoder-v1") -> None:
        """初始化测试编码器。

        参数:
            revision: 测试用编码规范标识，供 Contract 测试断言版本传递。
        返回值:
            None
        异常:
            无
        """
        self.revision = revision
        self.inputs: list[str] = []
        self.closed = False
        self._vector = [1.0] + [0.0] * 1023

    def count_tokens(self, text: str) -> int:
        """按字符数模拟 token 计数。

        参数:
            text: 待计数文本。
        返回值:
            int: 文本字符数。
        异常:
            无
        """
        return len(text)

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """记录输入并返回固定向量。

        参数:
            texts: 待编码文本列表。
        返回值:
            list[list[float]]: 与输入等长的 1024 维单位向量。
        异常:
            无
        """
        self.inputs.extend(texts)
        return [list(self._vector) for _ in texts]

    async def aclose(self) -> None:
        """标记测试编码器已关闭。

        参数:
            无
        返回值:
            None
        异常:
            无
        """
        self.closed = True
