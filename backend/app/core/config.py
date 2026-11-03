from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AURA"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://aura:aura_dev_password@localhost:5432/aura"
    qdrant_url: str = "http://localhost:6333"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
