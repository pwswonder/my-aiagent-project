from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite:///var/ai_paper_agent_v2.db"
    storage_root: Path = Path("var/storage")
    max_upload_bytes: int = 50 * 1024 * 1024
    worker_poll_seconds: float = 1.0
    max_job_attempts: int = 3

    llm_provider: Literal["openai", "azure"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4"
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None
    azure_openai_api_version: str = "2025-04-01-preview"
    embedding_model: str = "text-embedding-3-small"
    github_token: str | None = None
    llm_timeout_seconds: float = Field(default=90.0, ge=10.0, le=300.0)

    sandbox_image: str = "ai-paper-agent-sandbox:latest"
    sandbox_runner_url: str | None = None
    sandbox_storage_volume: str = "ai-paper-agent-storage"
    sandbox_timeout_seconds: int = Field(default=120, ge=10, le=600)
    sandbox_memory: str = "2g"
    sandbox_cpus: float = 1.0
    sandbox_pids_limit: int = 128

    def ensure_directories(self) -> None:
        for path in (
            self.storage_root,
            self.storage_root / "documents",
            self.storage_root / "artifacts",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
