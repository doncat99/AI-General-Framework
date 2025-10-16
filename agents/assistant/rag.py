# agents/assistant/rag.py
from __future__ import annotations
from typing import List
from .retrievers.chroma_retriever import ChromaRetriever

class RAG:
    def __init__(self, retriever: ChromaRetriever):
        self.retriever = retriever

    def gather_context(self, question: str, k: int = 4) -> List[str]:
        return self.retriever.similarity_search(question, k=k)
