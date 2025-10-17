# agents/assistant/tools/agent_tools.py
from __future__ import annotations

import os
import time
import typing as t
from datetime import datetime
from pathlib import Path
import requests
import logging

from mem0 import Memory

logger = logging.getLogger(__name__)


# If your project offers RAG helpers, import them. Otherwise fallback is no-op.
_RAG_FUNC: t.Optional[t.Callable[..., t.Dict[str, t.Any]]] = None
for mod_name, fn_name in [
    ("retrieval_logic", "rag_answer"),
    ("rag_chain", "rag_answer"),
    ("rag_chain", "run_rag_query"),
]:
    try:
        mod = __import__(mod_name, fromlist=[fn_name])
        _RAG_FUNC = getattr(mod, fn_name, None) or _RAG_FUNC
    except Exception:
        pass


# =========================================================
# Minimal Mem0 client wrapper (aligned to your mem0 source)
# =========================================================

class _Mem0Client:
    """
    Thin wrapper around mem0.Memory.
    - add(...) requires `messages=...` and supports `infer`, ids, metadata
    - search(...) uses `limit` (not `k`)
    - there's NO delete_by_filters; we emulate via get_all → delete each
    """
    def __init__(self, memory: t.Any | None = None):
        # Your Memory() doesn’t take api_key; it’s locally configured.
        self._mem = memory or (Memory() if Memory else None)

    def ensure(self) -> None:
        if self._mem is None:
            raise RuntimeError("mem0 not available. Ensure `mem0` is installed/importable.")

    def upsert(
        self,
        *,
        text: str,
        user_id: str,
        metadata: t.Optional[dict] = None,
        infer: bool = False,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """
        Store a memory item. If infer=True, mem0 will run its fact-extraction/update pipeline.
        """
        self.ensure()
        return self._mem.add(
            messages=text,                 # accepts str or list[dict]
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata or {},
            infer=infer,
        )

    def search(
        self,
        *,
        query: str,
        user_id: t.Optional[str] = None,
        top_k: int = 5,
        filters: t.Optional[dict] = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        self.ensure()
        return self._mem.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=top_k,                   # NOT k=
            filters=filters or {},
            threshold=threshold,
        )

    def delete(
        self,
        *,
        memory_id: t.Optional[str] = None,
        user_id: t.Optional[str] = None,
        filters: t.Optional[dict] = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        """
        Delete a single memory by ID, or emulate delete-by-filters by listing then deleting.
        """
        self.ensure()
        if memory_id:
            return self._mem.delete(memory_id=memory_id)

        # Emulate delete_by_filters: get_all → delete
        results = self._mem.get_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            filters=filters or {},
            limit=1000,
        )
        # v1.1 returns {"results":[...]}; legacy could return list
        items = results.get("results", results) if isinstance(results, dict) else results
        if not isinstance(items, list):
            items = []

        deleted_ids: list[str] = []
        for it in items:
            mid = it.get("id") if isinstance(it, dict) else None
            if mid:
                self._mem.delete(memory_id=mid)
                deleted_ids.append(mid)

        return {"message": f"Deleted {len(deleted_ids)} memories", "deleted_ids": deleted_ids}


_MEM0 = _Mem0Client()


# =========================================================
# Built-in Handlers (domain-agnostic)
# =========================================================

def _ok(data: t.Any, **extra) -> t.Dict[str, t.Any]:
    return {"ok": True, "data": data, **extra}

def _err(message: str, **extra) -> t.Dict[str, t.Any]:
    return {"ok": False, "error": message, **extra}


def handle_memory_upsert(args: dict) -> dict:
    """
    Store a memory about the user/context.
    """
    try:
        text = args["text"].strip()
        if not text:
            return _err("empty text")
        user_id = args.get("user_id") or "default"
        res = _MEM0.upsert(
            text=text,
            user_id=user_id,
            metadata=args.get("metadata") or {},
            infer=bool(args.get("infer", False)),
            agent_id=args.get("agent_id"),
            run_id=args.get("run_id"),
        )
        return _ok(res)
    except Exception as e:
        return _err(f"mem0.upsert failed: {e!s}")


def handle_memory_search(args: dict) -> dict:
    """
    Retrieve memories by semantic search.
    """
    try:
        query = args["query"].strip()
        if not query:
            return _err("empty query")
        res = _MEM0.search(
            query=query,
            user_id=args.get("user_id"),
            agent_id=args.get("agent_id"),
            run_id=args.get("run_id"),
            top_k=int(args.get("top_k", 5)),
            filters=args.get("filters") or {},
            threshold=args.get("threshold"),
        )
        return _ok(res)
    except Exception as e:
        return _err(f"mem0.search failed: {e!s}")


def handle_memory_delete(args: dict) -> dict:
    """
    Delete a specific memory or delete by filters.
    """
    try:
        res = _MEM0.delete(
            memory_id=args.get("memory_id"),
            user_id=args.get("user_id"),
            agent_id=args.get("agent_id"),
            run_id=args.get("run_id"),
            filters=args.get("filters") or {},
        )
        return _ok(res)
    except Exception as e:
        return _err(f"mem0.delete failed: {e!s}")


def handle_rag_query(args: dict) -> dict:
    """
    Answer with your local RAG pipeline if available.
    """
    try:
        query = args["query"].strip()
        if not query:
            return _err("empty query")
        top_k = int(args.get("top_k", 5))
        kwargs = {k: v for k, v in args.items() if k not in {"query", "top_k"}}
        if _RAG_FUNC is None:
            return _err("RAG is not wired (no retrieval function found).")
        res = _RAG_FUNC(query=query, top_k=top_k, **kwargs)  # type: ignore[misc]
        return _ok(res)
    except Exception as e:
        return _err(f"RAG error: {e!s}")


def handle_web_get(args: dict) -> dict:
    """
    Simple HTTP GET (for adapters that allow net access).
    """
    try:
        if requests is None:
            return _err("`requests` not installed.")
        url = args["url"]
        if not url:
            return _err("missing url")
        timeout = float(args.get("timeout", 12))
        headers = args.get("headers") or {}
        r = requests.get(url, timeout=timeout, headers=headers)
        return _ok({"status": r.status_code, "headers": dict(r.headers), "text": r.text})
    except Exception as e:
        return _err(f"GET failed: {e!s}")


def handle_summarize(args: dict) -> dict:
    """
    Lightweight on-agent summarization. If your ADK agent supports tool-calling back into itself,
    you can switch this to a model call. Here we just do a naive split/trim.
    """
    try:
        text = args["text"]
        max_chars = int(args.get("max_chars", 1000))
        summary = text.strip()
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
        return _ok({"summary": summary, "length": len(summary)})
    except Exception as e:
        return _err(f"summarize failed: {e!s}")


def handle_math_eval(args: dict) -> dict:
    """
    Safe(ish) math: evaluates simple arithmetic expressions.
    """
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return _err("missing expression")
    try:
        import ast
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd, ast.FloorDiv, ast.LShift, ast.RShift,
        )
        node = ast.parse(expr, mode="eval")
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return _err("disallowed expression")
        return _ok({"expression": expr, "result": eval(compile(node, "<math>", "eval"))})
    except Exception as e:
        return _err(f"math eval failed: {e!s}")


def handle_files_list(args: dict) -> dict:
    """
    List files under a directory (non-recursive by default).
    """
    try:
        root = Path(args.get("root") or ".").expanduser().resolve()
        recursive = bool(args.get("recursive", False))
        max_items = int(args.get("max_items", 100))
        files: t.List[str] = []
        if recursive:
            for p in root.rglob("*"):
                if len(files) >= max_items:
                    break
                files.append(str(p))
        else:
            for p in root.iterdir():
                if len(files) >= max_items:
                    break
                files.append(str(p))
        return _ok({"root": str(root), "files": files})
    except Exception as e:
        return _err(f"files list failed: {e!s}")


def handle_time_now(args: dict) -> dict:
    tz = args.get("timezone") or "local"
    now = datetime.now()
    return _ok({"timezone": tz, "iso": now.isoformat(timespec="seconds"), "epoch": int(time.time())})


# =========================================================
# Shim callables exposed to the ADK / agent
#   - Accept either a single dict arg OR kwargs (we normalize to dict)
#   - Attach metadata so frameworks that introspect callables can read schema
# =========================================================

def _ensure_object_schema(params: t.Dict[str, t.Any] | None) -> t.Dict[str, t.Any]:
    params = dict(params or {})
    if params.get("type") != "object":
        params["type"] = "object"
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return params


_SCHEMA_MEMORY_UPSERT = _ensure_object_schema({
    "properties": {
        "text": {"type": "string", "description": "The memory content."},
        "user_id": {"type": "string", "description": "User identifier.", "default": "default"},
        "agent_id": {"type": "string"},
        "run_id": {"type": "string"},
        "infer": {"type": "boolean", "default": False},
        "metadata": {"type": "object", "additionalProperties": True},
    },
    "required": ["text"],
})

_SCHEMA_MEMORY_SEARCH = _ensure_object_schema({
    "properties": {
        "query": {"type": "string"},
        "user_id": {"type": "string"},
        "agent_id": {"type": "string"},
        "run_id": {"type": "string"},
        "top_k": {"type": "integer", "default": 5},
        "threshold": {"type": "number"},
        "filters": {"type": "object", "additionalProperties": True},
    },
    "required": ["query"],
})

_SCHEMA_MEMORY_DELETE = _ensure_object_schema({
    "properties": {
        "memory_id": {"type": "string"},
        "user_id": {"type": "string"},
        "agent_id": {"type": "string"},
        "run_id": {"type": "string"},
        "filters": {"type": "object", "additionalProperties": True},
    },
})

_SCHEMA_RAG_QUERY = _ensure_object_schema({
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 5},
    },
    "required": ["query"],
})

_SCHEMA_WEB_GET = _ensure_object_schema({
    "properties": {
        "url": {"type": "string"},
        "timeout": {"type": "number", "default": 12},
        "headers": {"type": "object", "additionalProperties": True},
    },
    "required": ["url"],
})

_SCHEMA_TEXT_SUMMARIZE = _ensure_object_schema({
    "properties": {
        "text": {"type": "string"},
        "max_chars": {"type": "integer", "default": 1000},
    },
    "required": ["text"],
})

_SCHEMA_MATH_EVAL = _ensure_object_schema({
    "properties": {
        "expression": {"type": "string"},
    },
    "required": ["expression"],
})

_SCHEMA_FILES_LIST = _ensure_object_schema({
    "properties": {
        "root": {"type": "string", "default": "."},
        "recursive": {"type": "boolean", "default": False},
        "max_items": {"type": "integer", "default": 100},
    },
})

_SCHEMA_TIME_NOW = _ensure_object_schema({
    "properties": {
        "timezone": {"type": "string", "default": "local"},
    },
})


def _wrap_tool(name: str, description: str, schema: dict, handler: t.Callable[[dict], dict]):
    """
    Create a callable that many agent frameworks can consume directly via add_tool(callable).
    The callable:
      - Accepts either a single dict 'args' OR kwargs (both normalized to dict)
      - Returns the handler result
      - Carries metadata attributes for frameworks that introspect callables
    """
    def _call(args: t.Optional[dict] = None, **kwargs) -> dict:
        payload = {}
        if isinstance(args, dict):
            payload.update(args)
        if kwargs:
            payload.update(kwargs)
        return handler(payload)

    _call.__name__ = name
    _call.__qualname__ = name
    _call.__doc__ = description
    setattr(_call, "_tool_name", name)
    setattr(_call, "_tool_description", description)
    setattr(_call, "_tool_parameters", schema)
    setattr(_call, "_tool_schema", schema)
    return _call


# Underscore names (match ADK logs) + dotted aliases
memory_upsert = _wrap_tool(
    "memory_upsert",
    "Store a memory (text + optional metadata) for a user.",
    _SCHEMA_MEMORY_UPSERT,
    handle_memory_upsert,
)
memory_search = _wrap_tool(
    "memory_search",
    "Semantic search over stored memories (mem0).",
    _SCHEMA_MEMORY_SEARCH,
    handle_memory_search,
)
memory_delete = _wrap_tool(
    "memory_delete",
    "Delete a memory by id or by filters (user_id/agent_id/run_id + metadata).",
    _SCHEMA_MEMORY_DELETE,
    handle_memory_delete,
)
rag_query = _wrap_tool(
    "rag_query",
    "Query the local knowledge base (RAG). Requires project RAG wiring.",
    _SCHEMA_RAG_QUERY,
    handle_rag_query,
)
web_get = _wrap_tool(
    "web_get",
    "HTTP GET a URL and return status, headers, and text.",
    _SCHEMA_WEB_GET,
    handle_web_get,
)
text_summarize = _wrap_tool(
    "text_summarize",
    "Summarize a chunk of text locally (agent may replace with model call).",
    _SCHEMA_TEXT_SUMMARIZE,
    handle_summarize,
)
math_eval = _wrap_tool(
    "math_eval",
    "Evaluate a simple arithmetic expression safely.",
    _SCHEMA_MATH_EVAL,
    handle_math_eval,
)
files_list = _wrap_tool(
    "files_list",
    "List files under a directory (optionally recursive).",
    _SCHEMA_FILES_LIST,
    handle_files_list,
)
time_now = _wrap_tool(
    "time_now",
    "Get current timestamp information.",
    _SCHEMA_TIME_NOW,
    handle_time_now,
)

# Dotted aliases (if your agent resolves dots)
memory_upsert_dot = _wrap_tool("memory.upsert", memory_upsert.__doc__ or "", _SCHEMA_MEMORY_UPSERT, handle_memory_upsert)
memory_search_dot = _wrap_tool("memory.search", memory_search.__doc__ or "", _SCHEMA_MEMORY_SEARCH, handle_memory_search)
memory_delete_dot = _wrap_tool("memory.delete", memory_delete.__doc__ or "", _SCHEMA_MEMORY_DELETE, handle_memory_delete)
rag_query_dot = _wrap_tool("rag.query", rag_query.__doc__ or "", _SCHEMA_RAG_QUERY, handle_rag_query)
web_get_dot = _wrap_tool("web.get", web_get.__doc__ or "", _SCHEMA_WEB_GET, handle_web_get)
text_summarize_dot = _wrap_tool("text.summarize", text_summarize.__doc__ or "", _SCHEMA_TEXT_SUMMARIZE, handle_summarize)
math_eval_dot = _wrap_tool("math.eval", math_eval.__doc__ or "", _SCHEMA_MATH_EVAL, handle_math_eval)
files_list_dot = _wrap_tool("files.list", files_list.__doc__ or "", _SCHEMA_FILES_LIST, handle_files_list)
time_now_dot = _wrap_tool("time.now", time_now.__doc__ or "", _SCHEMA_TIME_NOW, handle_time_now)


# =========================================================
# Public API — TWO registry shapes + tolerant registrar
# =========================================================

def get_registry() -> t.Mapping[str, t.Callable[..., dict]]:
    """
    Return a *Mapping* of {name: callable(args_dict)->dict}
    """
    return {
        # underscore names (seen in logs)
        "memory_upsert": memory_upsert,
        "memory_search": memory_search,
        "memory_delete": memory_delete,
        "rag_query": rag_query,
        "web_get": web_get,
        "text_summarize": text_summarize,
        "math_eval": math_eval,
        "files_list": files_list,
        "time_now": time_now,

        # dotted aliases (harmless if ignored)
        "memory.upsert": memory_upsert_dot,
        "memory.search": memory_search_dot,
        "memory.delete": memory_delete_dot,
        "rag.query": rag_query_dot,
        "web.get": web_get_dot,
        "text.summarize": text_summarize_dot,
        "math.eval": math_eval_dot,
        "files.list": files_list_dot,
        "time.now": time_now_dot,
    }


def _build_adk_registry(mapping: t.Mapping[str, t.Callable[..., dict]]) -> t.Dict[str, t.Dict[str, t.Any]]:
    """
    Convert {name: callable} -> {name: {"handler","description","parameters"}}
    """
    out: dict[str, dict[str, t.Any]] = {}
    for name, fn in mapping.items():
        desc = getattr(fn, "_tool_description", None) or (fn.__doc__ or "")
        params = getattr(fn, "_tool_schema", None) or getattr(fn, "_tool_parameters", None) or {"type": "object", "properties": {}, "required": []}
        params = _ensure_object_schema(params)
        out[name] = {
            "handler": fn,
            "description": desc,
            "parameters": params,
        }
    return out


# Module-level registries (so external loaders/agents can introspect easily)
REGISTRY: t.Mapping[str, t.Callable[..., dict]] = get_registry()
EXPORTED_TOOLS: t.Dict[str, t.Dict[str, t.Any]] = _build_adk_registry(REGISTRY)


def get_adk_registry() -> t.Dict[str, t.Dict[str, t.Any]]:
    """Return the richer ADK-style registry (with handler/description/parameters)."""
    return EXPORTED_TOOLS


def register_with_agent(agent: t.Any, registry: t.Optional[t.Mapping[str, t.Any]] = None) -> None:
    """
    Tolerant registrar:
      - If `registry` is provided and is ADK-style (dict with 'handler'), register with schema when possible.
      - Else if `registry` is {name: callable}, register the callables.
      - Else fall back to module REGISTRY/EXPORTED_TOOLS.
    """
    # Source the registry to use
    reg = registry or EXPORTED_TOOLS or REGISTRY

    def _try_schema_register(name: str, desc: str, params: dict, handler: t.Callable[..., t.Any]) -> bool:
        # add_function_tool(name, description, parameters, handler)
        if hasattr(agent, "add_function_tool"):
            try:
                agent.add_function_tool(name, desc, _ensure_object_schema(params), handler)
                logger.debug(f"Registered tool via add_function_tool: {name}")
                return True
            except Exception:
                pass
        # register_tool(name, description, parameters, handler)
        if hasattr(agent, "register_tool"):
            try:
                agent.register_tool(name, desc, _ensure_object_schema(params), handler)  # positional
                logger.debug(f"Registered tool via register_tool(...): {name}")
                return True
            except TypeError:
                try:
                    agent.register_tool(name=name, description=desc, parameters=_ensure_object_schema(params), handler=handler)  # kwargs
                    logger.debug(f"Registered tool via register_tool(kwargs): {name}")
                    return True
                except Exception:
                    pass
        return False

    def _fallback_add(handler: t.Callable[..., t.Any], name_hint: str) -> None:
        # Fallback to add_tool(callable)
        add_tool = getattr(agent, "add_tool", None)
        if callable(add_tool):
            try:
                add_tool(handler)
                logger.debug(f"Registered tool via add_tool(callable): {name_hint}")
                return
            except Exception as e:
                logger.warning(f"add_tool(callable) failed for {name_hint}: {e}")

        # Absolute last resort: try add_tool(name, handler)
        if callable(add_tool):
            try:
                add_tool(name_hint, handler)  # type: ignore[misc]
                logger.debug(f"Registered tool via add_tool(name, handler): {name_hint}")
            except Exception as e:
                logger.warning(f"add_tool(name, handler) failed for {name_hint}: {e}")

    # Path A: ADK-style registry {"handler","description","parameters"}
    if isinstance(reg, dict) and reg and all(isinstance(v, dict) and "handler" in v for v in reg.values()):
        for name, spec in reg.items():
            handler = spec["handler"]
            desc = spec.get("description", "") or getattr(handler, "__doc__", "") or ""
            params = spec.get("parameters", {}) or getattr(handler, "_tool_schema", {}) or {}
            if not _try_schema_register(name, desc, params, handler):
                _fallback_add(handler, name)
        return

    # Path B: plain {name: callable}
    if isinstance(reg, dict) and reg and all(callable(v) for v in reg.values()):
        for name, fn in reg.items():
            desc = getattr(fn, "_tool_description", None) or (fn.__doc__ or "")
            params = getattr(fn, "_tool_schema", None) or getattr(fn, "_tool_parameters", None) or {"type": "object", "properties": {}, "required": []}
            if not _try_schema_register(name, desc, params, fn):
                _fallback_add(fn, name)
        return

    # Path C: iterable of callables
    if isinstance(reg, (list, tuple)):
        for fn in reg:
            if callable(fn):
                _fallback_add(fn, getattr(fn, "__name__", "tool"))
        return

    logger.warning("register_with_agent: unrecognized registry shape; no tools registered.")
