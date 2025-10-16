from __future__ import annotations
from typing import List
import chromadb
from chromadb.utils import embedding_functions as ef
from loguru import logger
from agents.assistant.ingest.sinks.base import BaseSink
from agents.assistant.ingest.models import MemoryRecord
from agents.assistant.ingest.config import settings

def _embedding_fn():
    if settings.EMBEDDING_BACKEND.lower() == "openai":
        return ef.OpenAIEmbeddingFunction(api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_EMBED_MODEL)
    if settings.EMBEDDING_BACKEND.lower() == "cohere":
        return ef.CohereEmbeddingFunction(api_key=settings.COHERE_API_KEY, model_name="embed-english-v3.0")
    return ef.SentenceTransformerEmbeddingFunction(model_name=settings.SBERT_MODEL)

class ChromaSink(BaseSink):
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        emb = _embedding_fn()
        try:
            self.col = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                embedding_function=emb,
                metadata={"hnsw:space": "cosine"},
            )
        except TypeError:
            self.col = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            self.col._embedding_function = emb  # type: ignore

    def upsert(self, records: List[MemoryRecord]) -> None:
        if not records:
            return
        ids = [r.id for r in records]
        docs = [r.text for r in records]
        metas = [r.metadata for r in records]
        try:
            try:
                self.col.delete(ids=ids)
            except Exception:
                pass
            self.col.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"chroma sink: upserted {len(ids)} docs into '{settings.CHROMA_COLLECTION}'")
        except Exception as e:
            logger.error(f"chroma upsert error: {e}")
