from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from pathlib import Path

# Identify base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # LLM Settings (Groq)
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.0)

    # Vector Database Settings
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    collection_name: str = Field(default="sec_filings")

    # Local Embeddings Settings
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME"
    )

    # Storage Paths
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate a single global settings object
settings = Settings()