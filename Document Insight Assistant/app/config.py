from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Document Insight Assistant"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    api_prefix: str = "/api"
    backend_base_url: str = Field(default="http://127.0.0.1:8000", alias="BACKEND_BASE_URL")
    allowed_origins: str = Field(default="", alias="ALLOWED_ORIGINS")
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    local_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )
    local_dummy_dimension: int = Field(default=384, alias="LOCAL_DUMMY_DIMENSION")
    vectorstore_dir: Path = Field(
        default=BASE_DIR / "data" / "vectorstore",
        alias="VECTORSTORE_DIR",
    )
    max_upload_size_bytes: int = Field(default=10_485_760, alias="MAX_UPLOAD_SIZE_BYTES")
    chunk_size_chars: int = Field(default=1000, alias="CHUNK_SIZE_CHARS")
    chunk_overlap_chars: int = Field(default=200, alias="CHUNK_OVERLAP_CHARS")
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_min_score: float = Field(default=0.2, alias="RETRIEVAL_MIN_SCORE")
    answer_provider: str = Field(default="local", alias="ANSWER_PROVIDER")
    openai_answer_model: str = Field(default="", alias="OPENAI_ANSWER_MODEL")
    answer_max_sources: int = Field(default=3, alias="ANSWER_MAX_SOURCES")
    answer_snippet_chars: int = Field(default=220, alias="ANSWER_SNIPPET_CHARS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        allowed_embedding_providers = {"openai", "local", "local_dummy"}
        allowed_answer_providers = {"local", "openai"}
        if self.embedding_provider not in allowed_embedding_providers:
            allowed_values = ", ".join(sorted(allowed_embedding_providers))
            raise ValueError(f"EMBEDDING_PROVIDER must be one of: {allowed_values}.")
        if self.answer_provider not in allowed_answer_providers:
            allowed_values = ", ".join(sorted(allowed_answer_providers))
            raise ValueError(f"ANSWER_PROVIDER must be one of: {allowed_values}.")
        if not self.backend_base_url.strip():
            raise ValueError("BACKEND_BASE_URL cannot be empty.")
        if self.chunk_size_chars <= 0:
            raise ValueError("CHUNK_SIZE_CHARS must be greater than 0.")
        if self.chunk_overlap_chars < 0:
            raise ValueError("CHUNK_OVERLAP_CHARS cannot be negative.")
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS.")
        if self.max_upload_size_bytes <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be greater than 0.")
        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be greater than 0.")
        if self.retrieval_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be greater than 0.")
        if not 0.0 <= self.retrieval_min_score <= 1.0:
            raise ValueError("RETRIEVAL_MIN_SCORE must be between 0.0 and 1.0.")
        if self.answer_max_sources <= 0:
            raise ValueError("ANSWER_MAX_SOURCES must be greater than 0.")
        if self.answer_snippet_chars <= 0:
            raise ValueError("ANSWER_SNIPPET_CHARS must be greater than 0.")
        if self.local_dummy_dimension <= 0:
            raise ValueError("LOCAL_DUMMY_DIMENSION must be greater than 0.")
        return self

    @property
    def cors_allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
