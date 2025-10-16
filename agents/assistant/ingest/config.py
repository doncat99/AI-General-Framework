from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings


class IngestSettings(BaseSettings):
    # mem0
    MEM0_API_KEY: str = Field(default="", env="MEM0_API_KEY")
    MEM0_BASE_URL: str = Field(default="https://api.mem0.ai", env="MEM0_BASE_URL")
    MEM0_DEFAULT_USER_ID: str = Field(default="user-default", env="MEM0_DEFAULT_USER_ID")
    MEM0_DEFAULT_TEAM_ID: str = Field(default="team-default", env="MEM0_DEFAULT_TEAM_ID")
    MEM0_VISIBILITY: str = Field(default="private", env="MEM0_VISIBILITY")  # private|team|public

    # chroma
    CHROMA_DIR: str = Field(default="chroma_db", env="CHROMA_DIR")
    CHROMA_COLLECTION: str = Field(default="docs", env="CHROMA_COLLECTION")
    EMBEDDING_BACKEND: str = Field(default="sbert", env="EMBEDDING_BACKEND")  # sbert|openai|cohere
    SBERT_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="SBERT_MODEL")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_EMBED_MODEL: str = Field(default="text-embedding-3-small", env="OPENAI_EMBED_MODEL")
    COHERE_API_KEY: str = Field(default="", env="COHERE_API_KEY")

    # neo4j
    NEO4J_URL: str = Field(default="", env="NEO4J_URL")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="12345678", env="NEO4J_PASSWORD")
    NEO4J_DB: str = Field(default="neo4j", env="NEO4J_DB")

    # runtime
    BATCH_SIZE: int = Field(default=64, env="INGEST_BATCH")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"

settings = IngestSettings()
