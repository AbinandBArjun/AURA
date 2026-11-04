from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the project root independently of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "AURA"
    environment: str = "development"

    database_url: str = (
        "postgresql+psycopg://aura:aura_dev_password@localhost:5433/aura"
    )

    qdrant_url: str = "http://localhost:6333"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
