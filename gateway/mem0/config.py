# gateway/mem0/config.py
"""
Mem0 configuration:
- LLM (extraction): OpenAI-compatible client pointed at OpenRouter
- Embeddings: Ollama
- Vector store: Chroma
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from copy import deepcopy


def build_mem0_config() -> Dict[str, Any]:
    # Vector store (Chroma)
    mem0_chroma_path = os.getenv("MEM0_CHROMA_PATH", "/app/data/chroma_db")
    collection_name = os.getenv("MEM0_COLLECTION", "rag_memories")

    # Embeddings via Ollama (Mem0 resolves host from env OLLAMA_BASE_URL/OLLAMA_HOST)
    ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # --- LLM for memory extraction (OpenAI-compatible; default to OpenRouter) ---
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Prefer OPENAI_API_KEY, fall back to OPENROUTER_API_KEY if not set
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    # NOTE: For OpenRouter, use a provider-prefixed model id (e.g., 'openai/gpt-4o-mini', 'google/gemini-2.0-flash')
    extract_model = os.getenv("MEM0_EXTRACT_MODEL", "openai/gpt-4o-mini")

    cfg: Dict[str, Any] = {
        # Embeddings (Ollama)
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": ollama_embed_model,
                # Do NOT include base_url/host here; Mem0 resolves from env (OLLAMA_BASE_URL/OLLAMA_HOST)
            },
        },
        # LLM for extraction (OpenAI-compatible via OpenRouter)
        "llm": {
            "provider": "openai",          # <<— IMPORTANT: must be 'openai', not 'openrouter'
            "config": {
                "model": extract_model,
                "api_key": openrouter_api_key,
                # "url": openrouter_base_url,
                # "organization": os.getenv("OPENAI_ORG_ID"),  # optional if you ever need it
            },
        },
        # Vector store (Chroma)
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection_name,
                "path": mem0_chroma_path,
            },
        },
        # Persistence dirs (non-secret)
        "data_dir": os.getenv("MEM0_DATA_DIR", "/app/data"),
        "config_dir": os.getenv("MEM0_CONFIG_DIR", "/app/config"),
    }
    return cfg


def sanitized_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = deepcopy(cfg or build_mem0_config())
    try:
        if data.get("llm", {}).get("config", {}).get("api_key"):
            data["llm"]["config"]["api_key"] = "****"
    except Exception:
        pass
    return data
