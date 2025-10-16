# agents/assistant/retrievers/chroma_retriever.py
from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional
from loguru import logger

import chromadb

# ---- Embedding function interfaces ----
EmbeddingsFn = Callable[[List[str]], List[List[float]]]

def openai_embedder(model: str = "text-embedding-3-small") -> EmbeddingsFn:
    """
    Minimal OpenAI embedding function.
    Requires OPENAI_API_KEY in environment.
    """
    try:
        from openai import OpenAI  # openai>=1.0
    except Exception as e:
        raise RuntimeError("openai package is required for openai_embedder") from e

    client = OpenAI()

    def _embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    return _embed


def sbert_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingsFn:
    """
    SentenceTransformers embedder (no external API calls).
    pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers package is required for sbert_embedder") from e

    model = SentenceTransformer(model_name)

    def _embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # normalize for cosine similarity
        vecs = model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    return _embed


class ChromaRetriever:
    """
    A lightweight, retriever over Chroma.

    - Manages a persistent Chroma collection
    - Lets you add/update texts with metadatas
    - Performs similarity search using your embedding function

    Usage:
        retriever = ChromaLiteRetriever(
            persist_dir="path/to/chroma",
            collection="docs",
            embed_fn=openai_embedder(),  # or sbert_embedder()
        )
        retriever.add_texts(["hello world"], metadatas=[{"source":"foo"}], ids=["doc1"])
        hits = retriever.similarity_search("hello", k=4)
    """

    def __init__(
        self,
        persist_dir: str,
        collection: str = "default",
        embed_fn: Optional[EmbeddingsFn] = None,
        metadata: Optional[Dict[str, Any]] = None,
        distance: str = "cosine",  # "cosine" | "l2" | "ip"
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection
        self.embed_fn = embed_fn or sbert_embedder()
        self.client = chromadb.PersistentClient(path=persist_dir)

        # NOTE: You don’t need to set dimension explicitly if you ALWAYS pass embeddings in .add()
        #       Chroma will infer from the first insert.
        self.col = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata or {"hnsw:space": distance},
        )
        logger.info(f"ChromaLiteRetriever ready (dir='{persist_dir}', collection='{collection}')")

    # ---------- Ingestion ----------
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 256,
    ) -> None:
        if not texts:
            return

        if ids and len(ids) != len(texts):
            raise ValueError("ids length must match texts length")
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")

        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [f"doc_{i}" for i in range(self._count(), self._count() + len(texts))]

        # Chunk to avoid big payloads
        for i in range(0, len(texts), batch_size):
            chunk_texts = texts[i:i + batch_size]
            chunk_metas = metadatas[i:i + batch_size]
            chunk_ids = ids[i:i + batch_size]
            try:
                embeddings = self.embed_fn(chunk_texts)
                self.col.add(
                    documents=chunk_texts,
                    metadatas=chunk_metas,
                    ids=chunk_ids,
                    embeddings=embeddings,
                )
            except Exception as e:
                logger.error(f"Chroma add_texts failed at chunk {i}: {e}")

    def upsert_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        # Chroma add() acts like upsert for existing IDs
        self.add_texts(texts, metadatas, ids)

    # ---------- Retrieval ----------
    def similarity_search(self, query: str, k: int = 4, where: Optional[Dict[str, Any]] = None) -> List[str]:
        if not query:
            return []
        try:
            q_emb = self.embed_fn([query])[0]
            res = self.col.query(
                query_embeddings=[q_emb],
                n_results=max(1, int(k)),
                where=where or {},
                include=["documents", "metadatas", "distances", "ids"],
            )
            # Format: return only page_content-like strings
            docs = (res.get("documents") or [[]])[0]
            return docs
        except Exception as e:
            logger.warning(f"Chroma similarity_search failed: {e}")
            return []

    def similarity_search_with_scores(
        self, query: str, k: int = 4, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        try:
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
        except Exception as e:
            logger.warning(f"Chroma similarity_search_with_scores failed: {e}")
            return []

    # ---------- Maintenance ----------
    def delete(self, ids: List[str]) -> None:
        try:
            self.col.delete(ids=ids)
        except Exception as e:
            logger.warning(f"Chroma delete failed: {e}")

    def _count(self) -> int:
        try:
            return self.col.count()
        except Exception:
            return 0
