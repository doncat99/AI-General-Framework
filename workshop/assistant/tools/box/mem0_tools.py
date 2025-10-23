# agents/assistant/tools/mem0_tools.py
from __future__ import annotations

import os
import json
import httpx
from typing import Any, Dict, Mapping, Optional
from loguru import logger

from ..base_tool import BaseToolPack, ToolSpec

DEFAULT_MEM0_URL = os.getenv("MEM0_SERVER_URL", "http://mem0:8001")

_MAX_PREVIEW = int(os.getenv("MEM0_LOG_MAX_CHARS", "400"))

def _preview(x: Any) -> str:
    try:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    return s if len(s) <= _MAX_PREVIEW else s[:_MAX_PREVIEW] + "…"

def _resolve_url(args: Dict[str, Any]) -> str:
    """
    base_url precedence: args['base_url'] > env MEM0_SERVER_URL > default
    """
    return (args.get("base_url")
            or os.getenv("MEM0_SERVER_URL")
            or DEFAULT_MEM0_URL).rstrip("/")


class Mem0Tools(BaseToolPack):
    """
    Tool pack that exposes Mem0 service endpoints as ADK tools.
    Pass base_url in defaults or rely on env/Mem0 service DNS.
    """

    # -------------------------
    # Handlers (async dict->dict)
    # -------------------------
    @staticmethod
    async def _health(args: Dict[str, Any]) -> Dict[str, Any]:
        base = _resolve_url(args)
        logger.debug(f"[mem0] GET {base}/health")
        async with httpx.AsyncClient(base_url=base, timeout=20.0) as cli:
            r = await cli.get("/health")
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] health → status={js.get('status')} retriever_ready={js.get('retriever_ready')}")
            return js

    @staticmethod
    async def _config(args: Dict[str, Any]) -> Dict[str, Any]:
        base = _resolve_url(args)
        logger.debug(f"[mem0] GET {base}/config")
        async with httpx.AsyncClient(base_url=base, timeout=20.0) as cli:
            r = await cli.get("/config")
            r.raise_for_status()
            js = r.json()
            logger.debug(f"[mem0] config → {_preview(js)}")
            return js

    @staticmethod
    async def _reload(args: Dict[str, Any]) -> Dict[str, Any]:
        base = _resolve_url(args)
        logger.info(f"[mem0] POST {base}/reload")
        async with httpx.AsyncClient(base_url=base, timeout=60.0) as cli:
            r = await cli.post("/reload")
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] reload → {_preview(js)}")
            return js

    @staticmethod
    async def _add(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        args:
          message (str, required)
          user_id (str)
          agent_id (str)
          metadata (dict)
        """
        base = _resolve_url(args)
        payload = {
            "message": args["message"],
            "user_id": args.get("user_id", "default_user"),
            "agent_id": args.get("agent_id", "rag_agent"),
            "metadata": args.get("metadata") or {},
        }
        logger.info(f"[mem0] POST {base}/add ← uid={payload['user_id']} msg={_preview(payload['message'])}")
        async with httpx.AsyncClient(base_url=base, timeout=30.0) as cli:
            r = await cli.post("/add", json=payload)
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] add → {_preview(js)}")
            return js

    @staticmethod
    async def _search(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        args:
          query (str, required)
          user_id (str)
          agent_id (str)
          limit (int)
        """
        base = _resolve_url(args)
        payload = {
            "query": args["query"],
            "user_id": args.get("user_id", "default_user"),
            "agent_id": args.get("agent_id", "rag_agent"),
            "limit": int(args.get("limit", 5)),
        }
        logger.info(f"[mem0] POST {base}/search ← uid={payload['user_id']} q={_preview(payload['query'])} limit={payload['limit']}")
        async with httpx.AsyncClient(base_url=base, timeout=30.0) as cli:
            r = await cli.post("/search", json=payload)
            r.raise_for_status()
            js = r.json()
            # count results from either shape
            count = 0
            if isinstance(js, dict):
                if isinstance(js.get("memories"), list):
                    count = len(js["memories"])
                elif isinstance(js.get("results"), list):
                    count = len(js["results"])
            logger.info(f"[mem0] search → {count} hit(s)")
            if count:
                logger.debug(f"[mem0] search sample → {_preview(js)}")
            return js

    @staticmethod
    async def _index_add(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        args:
          texts (List[str], required)
          metadatas (List[dict])
          ids (List[str])
          batch_size (int)
        """
        base = _resolve_url(args)
        payload = {
            "texts": list(args["texts"]),
            "metadatas": args.get("metadatas"),
            "ids": args.get("ids"),
            "batch_size": int(args.get("batch_size", 256)),
        }
        logger.info(f"[mem0] POST {base}/index/add ← n_texts={len(payload['texts'])}")
        async with httpx.AsyncClient(base_url=base, timeout=None) as cli:
            r = await cli.post("/index/add", json=payload)
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] index_add → {_preview(js)}")
            return js

    @staticmethod
    async def _index_upsert(args: Dict[str, Any]) -> Dict[str, Any]:
        base = _resolve_url(args)
        payload = {
            "texts": list(args["texts"]),
            "metadatas": args.get("metadatas"),
            "ids": args.get("ids"),
            "batch_size": int(args.get("batch_size", 256)),
        }
        logger.info(f"[mem0] POST {base}/index/upsert ← n_texts={len(payload['texts'])}")
        async with httpx.AsyncClient(base_url=base, timeout=None) as cli:
            r = await cli.post("/index/upsert", json=payload)
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] index_upsert → {_preview(js)}")
            return js

    @staticmethod
    async def _index_search(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        args:
          query (str, required)
          k (int)
          where (dict)
        """
        base = _resolve_url(args)
        payload = {
            "query": args["query"],
            "k": int(args.get("k", 4)),
            "where": args.get("where") or {},
        }
        logger.info(f"[mem0] POST {base}/index/search ← q={_preview(payload['query'])} k={payload['k']}")
        async with httpx.AsyncClient(base_url=base, timeout=30.0) as cli:
            r = await cli.post("/index/search", json=payload)
            r.raise_for_status()
            js = r.json()
            cnt = len(js.get("documents", [])) if isinstance(js, dict) else 0
            logger.info(f"[mem0] index_search → {cnt} doc(s)")
            return js

    @staticmethod
    async def _index_search_with_scores(args: Dict[str, Any]) -> Dict[str, Any]:
        base = _resolve_url(args)
        payload = {
            "query": args["query"],
            "k": int(args.get("k", 4)),
            "where": args.get("where") or {},
        }
        logger.info(f"[mem0] POST {base}/index/search_with_scores ← q={_preview(payload['query'])} k={payload['k']}")
        async with httpx.AsyncClient(base_url=base, timeout=30.0) as cli:
            r = await cli.post("/index/search_with_scores", json=payload)
            r.raise_for_status()
            js = r.json()
            cnt = len(js.get("hits", [])) if isinstance(js, dict) else 0
            logger.info(f"[mem0] index_search_with_scores → {cnt} hit(s)")
            return js

    @staticmethod
    async def _index_delete(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        args:
          ids (List[str], required)
        """
        base = _resolve_url(args)
        payload = {"ids": list(args["ids"])}
        logger.info(f"[mem0] DELETE {base}/index/delete ← n_ids={len(payload['ids'])}")
        async with httpx.AsyncClient(base_url=base, timeout=30.0) as cli:
            r = await cli.request("DELETE", "/index/delete", json=payload)
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] index_delete → {_preview(js)}")
            return js

    @staticmethod
    async def _index_count(args: Dict[str, Any]) -> Dict[str, Any]:
        base = _resolve_url(args)
        logger.debug(f"[mem0] GET {base}/index/count")
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as cli:
            r = await cli.get("/index/count")
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] index_count → {js.get('count')}")
            return js

    @staticmethod
    async def _sink_upsert(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        args:
          records (List[MemoryRecord-like], required)
          sinks (List[str])  # ["chroma","mem0","neo4j"] subset or omit for all
        """
        base = _resolve_url(args)
        payload = {"records": list(args["records"]), "sinks": args.get("sinks")}
        logger.info(f"[mem0] POST {base}/sink/upsert ← n_records={len(payload['records'])} sinks={payload.get('sinks')}")
        async with httpx.AsyncClient(base_url=base, timeout=None) as cli:
            r = await cli.post("/sink/upsert", json=payload)
            r.raise_for_status()
            js = r.json()
            logger.info(f"[mem0] sink_upsert → {_preview(js)}")
            return js

    # -------------------------
    # Schemas
    # -------------------------
    _SCHEMAS: Dict[str, Dict[str, Any]] = {
        "mem0_health": {"type": "object", "properties": {"base_url": {"type": "string"}}, "required": []},
        "mem0_config": {"type": "object", "properties": {"base_url": {"type": "string"}}, "required": []},
        "mem0_reload": {"type": "object", "properties": {"base_url": {"type": "string"}}, "required": []},
        "mem0_add": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "user_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "metadata": {"type": "object"},
                "base_url": {"type": "string"},
            },
            "required": ["message"],
        },
        "mem0_search": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "user_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "limit": {"type": "integer"},
                "base_url": {"type": "string"},
            },
            "required": ["query"],
        },
        "mem0_index_add": {
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}},
                "metadatas": {"type": "array", "items": {"type": "object"}},
                "ids": {"type": "array", "items": {"type": "string"}},
                "batch_size": {"type": "integer"},
                "base_url": {"type": "string"},
            },
            "required": ["texts"],
        },
        "mem0_index_upsert": {
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}},
                "metadatas": {"type": "array", "items": {"type": "object"}},
                "ids": {"type": "array", "items": {"type": "string"}},
                "batch_size": {"type": "integer"},
                "base_url": {"type": "string"},
            },
            "required": ["texts"],
        },
        "mem0_index_search": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
                "where": {"type": "object"},
                "base_url": {"type": "string"},
            },
            "required": ["query"],
        },
        "mem0_index_search_with_scores": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
                "where": {"type": "object"},
                "base_url": {"type": "string"},
            },
            "required": ["query"],
        },
        "mem0_index_delete": {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string"}}, "base_url": {"type": "string"}},
            "required": ["ids"],
        },
        "mem0_index_count": {"type": "object", "properties": {"base_url": {"type": "string"}}, "required": []},
        "mem0_sink_upsert": {
            "type": "object",
            "properties": {
                "records": {"type": "array", "items": {"type": "object"}},
                "sinks": {"type": "array", "items": {"type": "string"}},
                "base_url": {"type": "string"},
            },
            "required": ["records"],
        },
    }

    # -------------------------
    # Declare tools
    # -------------------------
    def tools(self) -> Mapping[str, ToolSpec]:
        return {
            "mem0_health": ToolSpec(self._health, "Check Mem0 service health.", self._SCHEMAS["mem0_health"]),
            "mem0_config": ToolSpec(self._config, "Get sanitized Mem0 server configuration.", self._SCHEMAS["mem0_config"]),
            "mem0_reload": ToolSpec(self._reload, "Reload Mem0 server (memory, retriever, sinks).", self._SCHEMAS["mem0_reload"]),
            "mem0_add": ToolSpec(self._add, "Add a memory to Mem0 (AsyncMemory.add).", self._SCHEMAS["mem0_add"]),
            "mem0_search": ToolSpec(self._search, "Semantic memory search via Mem0.", self._SCHEMAS["mem0_search"]),
            "mem0_index_add": ToolSpec(self._index_add, "Add texts into the Chroma index (no overwrite).", self._SCHEMAS["mem0_index_add"]),
            "mem0_index_upsert": ToolSpec(self._index_upsert, "Upsert texts into the Chroma index.", self._SCHEMAS["mem0_index_upsert"]),
            "mem0_index_search": ToolSpec(self._index_search, "Search Chroma index for similar documents.", self._SCHEMAS["mem0_index_search"]),
            "mem0_index_search_with_scores": ToolSpec(self._index_search_with_scores, "Search index with scores.", self._SCHEMAS["mem0_index_search_with_scores"]),
            "mem0_index_delete": ToolSpec(self._index_delete, "Delete by IDs from index.", self._SCHEMAS["mem0_index_delete"]),
            "mem0_index_count": ToolSpec(self._index_count, "Return count of indexed documents.", self._SCHEMAS["mem0_index_count"]),
            "mem0_sink_upsert": ToolSpec(self._sink_upsert, "Upsert MemoryRecord batch into sinks (chroma/mem0/neo4j).", self._SCHEMAS["mem0_sink_upsert"]),
        }

    # -------------------------
    # Convenience factory (call-time env read)
    # -------------------------
    @classmethod
    def from_env(cls, *, base_url: Optional[str] = None, prefix: Optional[str] = None) -> "Mem0Tools":
        """
        Build a Mem0Tools pack using env defaults if args not provided.
        Reads env at call time to avoid import-time side effects.
        """
        defaults: Dict[str, Any] = {}
        if base_url or os.getenv("MEM0_SERVER_URL"):
            defaults["base_url"] = base_url or os.getenv("MEM0_SERVER_URL")

        return cls(
            defaults=defaults,
            name_prefix=(prefix if prefix is not None else os.getenv("TOOL_PREFIX_MEM0", "")),
        )
