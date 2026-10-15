"""Application configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chat_base_url: str = "http://8.219.192.36:9000"
    chat_model: str = "claude-opus-4-8"
    chat_api_key: str = "test-token"
    chat_token_budget: int = 2000

    embed_base_url: str | None = None
    embed_model: str | None = None
    embed_api_key: str | None = None

    rerank_base_url: str | None = None
    rerank_model: str | None = None
    rerank_api_key: str | None = None


settings = Settings()
