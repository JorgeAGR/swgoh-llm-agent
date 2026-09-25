"""Runtime configuration, read from environment variables and an optional `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Chat model served by any OpenAI-compatible endpoint.
    llm_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL")
    )
    llm_api_key: str = Field(
        default="not-needed",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"),
    )
    llm_temperature: float | None = Field(
        default=None, validation_alias="LLM_TEMPERATURE"
    )

    # Embeddings. "local" runs a small ONNX model via fastembed; "openai" calls
    # an OpenAI-compatible /embeddings route (defaults to the chat endpoint).
    embedding_provider: str = Field(
        default="local", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", validation_alias="EMBEDDING_MODEL"
    )
    embedding_base_url: str | None = Field(
        default=None, validation_alias="EMBEDDING_BASE_URL"
    )
    embedding_api_key: str | None = Field(
        default=None, validation_alias="EMBEDDING_API_KEY"
    )

    data_dir: Path = Field(
        default=PROJECT_ROOT / "data", validation_alias="SWGOH_DATA_DIR"
    )

    @property
    def units_path(self) -> Path:
        return self.data_dir / "units.json"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
