# gateway/mem0/chroma_retriever.py
from __future__ import annotations

import os
from typing import Callable, List, Dict, Any, Optional
import logging

import chromadb
from ollama import Client as OllamaClient

logger = logging.getLogger("mem0-retriever")

EmbeddingsFn = Callable[[List[str]], List[List[float]]]


def get_ollama_embedder(
    host: Optional[str] = None,
    model: Optional[str] = None,
) -> EmbeddingsFn:
    """
    Create an embeddings function backed by Ollama.

    Env:
      OLLAMA_HOST (e.g., http://ollama:11434)
      OLLAMA_EMBED_MODEL (e.g., nomic-embed-text)
    """
    host = host or os.getenv("OLLAMA_HOST", "http://ollama:11434")
    model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    client = OllamaClient(host=host)

    def _embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs: List[List[float]] = []
        # Ollama Python API embeds one text at a time
        for t in texts:
            # NOTE: Some versions use 'prompt' instead of 'input'
            # The official client exposes client.embeddings(model=..., prompt=...)
            resp = client.embeddings(model=model, prompt=t)
            emb = resp.get("embedding")
            if not emb:
                raise RuntimeError("Ollama returned empty embedding.")
            vecs.append(emb)
        return vecs

    logger.info(f"Ollama embedder ready (host={host}, model={model})")
    return _embed


class ServerChromaRetriever:
    """
    Server-local Chroma retriever using Ollama embeddings.

    Persistence path & collection should match your mem0 config so both share the same index.
    """

    def __init__(
        self,
        persist_dir: str,
        collection: str = "rag_memories",
        embed_fn: Optional[EmbeddingsFn] = None,
        distance: str = "cosine",  # 'cosine' | 'l2' | 'ip'
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection
        self.embed_fn = embed_fn or get_ollama_embedder()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata or {"hnsw:space": distance},
        )
        logger.info(
            "ServerChromaRetriever ready (dir='%s', collection='%s', distance='%s')",
            persist_dir, collection, distance
        )

    # ---------- Ingestion ----------
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 256,
    ) -> int:
        if not texts:
            return 0
        if ids and len(ids) != len(texts):
            raise ValueError("ids length must match texts length")
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")

        metadatas = metadatas or [{} for _ in texts]
        next_id_start = self.count()
        ids = ids or [f"doc_{i}" for i in range(next_id_start, next_id_start + len(texts))]

        added = 0
        for i in range(0, len(texts), batch_size):
            chunk_texts = texts[i:i + batch_size]
            chunk_metas = metadatas[i:i + batch_size]
            chunk_ids = ids[i:i + batch_size]
            embeddings = self.embed_fn(chunk_texts)
            self.col.add(
                documents=chunk_texts,
                metadatas=chunk_metas,
                ids=chunk_ids,
                embeddings=embeddings,
            )
            added += len(chunk_texts)
        return added

    def upsert_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 256,
    ) -> int:
        # Chroma add() upserts existing ids
        return self.add_texts(texts, metadatas, ids, batch_size=batch_size)

    # ---------- Retrieval ----------
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        where: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        if not query:
            return []
        q_emb = self.embed_fn([query])[0]
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=max(1, int(k)),
            where=where or {},
            include=["documents"],
        )
        return (res.get("documents") or [[]])[0]

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        q_emb = self.embed_fn([query])[0]
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=max(1, int(k)),
            where=where or {},
            include=["documents", "metadatas", "distances", "ids"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        out = []
        for doc, meta, dist, _id in zip(docs, metas, dists, ids):
            out.append({"id": _id, "text": doc, "metadata": meta, "distance": dist})
        return out

    # ---------- Maintenance ----------
    def delete(self, ids: List[str]) -> int:
        if not ids:
            return 0
        self.col.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        try:
            return self.col.count()
        except Exception:
            return 0
