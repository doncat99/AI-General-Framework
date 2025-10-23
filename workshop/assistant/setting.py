# agents/assistant/config.py
from __future__ import annotations
from pathlib import Path
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Your app-level config (provides config.USER_FILES)
from config import config


class IngestSettings(BaseSettings):
    """
    Centralized settings for the assistant. Uses Pydantic v2-style config
    so we can safely ignore unrelated env vars in your .env/.environment.
    """

    # Tell pydantic-settings to 1) read .env, 2) ignore unknown keys.
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",            # <-- crucial: ignore extra env vars
        env_nested_delimiter="__", # safe default; no nested models here, but fine
    )

    # ---- Mem0 (HTTP API) ----
    MEM0_API_KEY: str = Field(default="", env="MEM0_API_KEY")
    MEM0_BASE_URL: str = Field(default="https://api.mem0.ai", env="MEM0_BASE_URL")
    MEM0_DEFAULT_USER_ID: str = Field(default="user-default", env="MEM0_DEFAULT_USER_ID")
    MEM0_DEFAULT_TEAM_ID: str = Field(default="team-default", env="MEM0_DEFAULT_TEAM_ID")
    MEM0_VISIBILITY: str = Field(default="private", env="MEM0_VISIBILITY")  # private|team|public

    # ---- Local Qdrant for Mem0 vector store (code-defined defaults) ----
    MEM0_QDRANT_PATH: str = Field(
        default_factory=lambda: os.path.join(config.USER_FILES, "assistant", "mem0_qdrant"),
        env="MEM0_QDRANT_PATH",
    )
    MEM0_QDRANT_COLLECTION: str = Field(
        default="mem0migrations",
        env="MEM0_QDRANT_COLLECTION",
    )

    # ---- Local lexical fallback store for memories ----
    MEM_FALLBACK_DIR: str = Field(
        default_factory=lambda: os.path.join(config.USER_FILES, "assistant", "mem_fallback"),
        env="MEM_FALLBACK_DIR",
    )

    # ---- RAG / Chroma (optional) ----
    CHROMA_DIR: str = Field(default="chroma_db", env="CHROMA_DIR")
    CHROMA_COLLECTION: str = Field(default="docs", env="CHROMA_COLLECTION")
    EMBEDDING_BACKEND: str = Field(default="sbert", env="EMBEDDING_BACKEND")  # sbert|openai|cohere
    SBERT_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="SBERT_MODEL")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_EMBED_MODEL: str = Field(default="text-embedding-3-small", env="OPENAI_EMBED_MODEL")
    COHERE_API_KEY: str = Field(default="", env="COHERE_API_KEY")

    # ---- Neo4j ----
    NEO4J_URL: str = Field(default="", env="NEO4J_URL")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="12345678", env="NEO4J_PASSWORD")
    NEO4J_DB: str = Field(default="neo4j", env="NEO4J_DB")

    # ---- Runtime ----
    BATCH_SIZE: int = Field(default=64, env="INGEST_BATCH")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")


settings = IngestSettings()

# ---- Normalize/ensure paths exist ----
_qdrant_path = Path(settings.MEM0_QDRANT_PATH).expanduser()
_qdrant_path.mkdir(parents=True, exist_ok=True)

_chroma_dir = Path(settings.CHROMA_DIR).expanduser()
_chroma_dir.mkdir(parents=True, exist_ok=True)

_fallback_dir = Path(settings.MEM_FALLBACK_DIR).expanduser()
_fallback_dir.mkdir(parents=True, exist_ok=True)

# ---- Export to environment for modules that use getenv() (e.g., memory.py) ----
os.environ.setdefault("MEM0_QDRANT_PATH", str(_qdrant_path))
os.environ.setdefault("MEM0_QDRANT_COLLECTION", settings.MEM0_QDRANT_COLLECTION)
os.environ.setdefault("MEM_FALLBACK_DIR", str(_fallback_dir))

# Prefer local embedder by default (prevents accidental OpenAI use)
os.environ.setdefault("MEM0_EMBEDDER_PROVIDER", os.getenv("MEM0_EMBEDDER_PROVIDER", "ollama"))
os.environ.setdefault("MEM0_EMBEDDER", os.getenv("MEM0_EMBEDDER", os.environ["MEM0_EMBEDDER_PROVIDER"]))

# Optional: allow auto-recreate on dim mismatch (default off)
os.environ.setdefault(
    "MEM0_QDRANT_RECREATE_ON_MISMATCH",
    os.getenv("MEM0_QDRANT_RECREATE_ON_MISMATCH", "0"),
)

# ---- App constants ----
APP_NAME = os.getenv("APP_NAME", "adk_mem0_assistant")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default_user")

# Vector store (optional RAG)
ENABLE_RAG = os.getenv("ENABLE_RAG", "1") == "1"
CHROMA_DIR = str(_chroma_dir)
