# gateway/mem0/sinks/chroma_sink.py
from __future__ import annotations

from typing import List, Dict, Any

from loguru import logger

from ...retrievers.chroma_retriever import ServerChromaRetriever  # uses Ollama embedder under the hood
from ..models import MemoryRecord
from .base import BaseSink


class ChromaSink(BaseSink):
    """
    Sink that writes MemoryRecord items into the shared Chroma collection,
    using the SAME Ollama embedder + persistence path as the server retriever.
    """

    def __init__(self, retriever: ServerChromaRetriever) -> None:
        self.retriever = retriever

    def upsert(self, records: List[MemoryRecord]) -> int:
        if not records:
            return 0

        texts = [r.text for r in records]
        ids = [r.id for r in records]
        metas: List[Dict[str, Any]] = []
        for r in records:
            md: Dict[str, Any] = dict(r.metadata or {})
            # keep traceability fields
            md["user_id"] = r.user_id
            md["team_id"] = r.team_id
            md["visibility"] = r.visibility
            md["external_id"] = r.id
            metas.append(md)

        # retriever.upsert_texts delegates to add() (upsert semantics for duplicate ids)
        n = self.retriever.upsert_texts(texts=texts, metadatas=metas, ids=ids, batch_size=256)
        logger.info(f"chroma sink: upserted {n} docs into collection '{self.retriever.collection_name}'")
        return n
