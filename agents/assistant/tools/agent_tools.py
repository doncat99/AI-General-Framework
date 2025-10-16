# agents/assistant/tools/agent_tools.py
from __future__ import annotations

import os
import time
import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---- optional deps (guarded) ----
try:
    from mem0 import Memory  # pip install mem0ai
except Exception:  # pragma: no cover
    Memory = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

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
# Tool protocol
# =========================================================

ToolHandler = t.Callable[[t.Dict[str, t.Any]], t.Dict[str, t.Any]]

@dataclass
class ToolSpec:
    """
    A single callable tool that can be registered with your ADK agent.
    """
    name: str
    description: str
    # JSON schema (properties/required) for ADK function tool
    parameters: t.Dict[str, t.Any]
    handler: ToolHandler
    # Optional: category/tags to make organization easier
    tags: t.List[str] = field(default_factory=list)

    def to_adk_function(self) -> t.Dict[str, t.Any]:
        """
        Convert to an ADK-compatible tool schema.
        For Google ADK (Function Tool), the expected structure is usually:
        {
          "name": ...,
          "description": ...,
          "parameters": {
             "type": "object",
             "properties": {...},
             "required": [...]
          }
        }
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }


class ToolRegistry:
    """
    Collects ToolSpec objects, deduplicates, and exposes registration helpers.
    """
    def __init__(self, namespace: str = "builtin"):
        self.namespace = namespace
        self._tools: t.Dict[str, ToolSpec] = {}

    @property
    def tools(self) -> t.List[ToolSpec]:
        return list(self._tools.values())

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            # choose your policy: overwrite or raise; we overwrite for simplicity
            # (external plugins can be namespaced if desired)
            pass
        self._tools[spec.name] = spec

    def merge(self, other: "ToolRegistry", *, namespace: str | None = None) -> None:
        """
        Merge another registry. Optionally apply a namespace prefix to avoid collisions.
        """
        prefix = (namespace or other.namespace or "").strip()
        for name, spec in other._tools.items():
            new_name = f"{prefix}.{name}" if prefix else name
            self._tools[new_name] = ToolSpec(
                name=new_name,
                description=spec.description if "/" else spec.description,  # keep exact
                parameters=spec.parameters,
                handler=spec.handler,
                tags=(spec.tags + ([prefix] if prefix else [])),
            )

    def as_adk_tools(self) -> t.List[t.Dict[str, t.Any]]:
        return [t_.to_adk_function() for t_ in self.tools]


# =========================================================
# Minimal Mem0 client wrapper
# =========================================================

class _Mem0Client:
    """
    Thin wrapper around mem0.Memory.
    Reads MEM0_API_KEY from env by default.
    You can inject a custom Memory instance via constructor for tests.
    """
    def __init__(self, memory: t.Any | None = None):
        api_key = os.getenv("MEM0_API_KEY", "")
        self._mem = memory or (Memory(api_key=api_key) if (Memory and api_key) else None)

    def ensure(self) -> None:
        if self._mem is None:
            raise RuntimeError(
                "mem0 not available. Install `mem0ai` and set MEM0_API_KEY in env."
            )

    # Generic shapes based on mem0 examples:
    #   upsert(text, user_id, metadata)
    #   search(query, user_id, k, filters)
    #   delete(memory_id) or delete_by_filters(user_id, metadata)
    def upsert(self, *, text: str, user_id: str, metadata: t.Optional[dict] = None) -> dict:
        self.ensure()
        return self._mem.add(text=text, user_id=user_id, metadata=metadata or {})

    def search(self, *, query: str, user_id: t.Optional[str] = None, top_k: int = 5,
               filters: t.Optional[dict] = None) -> dict:
        self.ensure()
        return self._mem.search(query=query, user_id=user_id, k=top_k, filters=filters or {})

    def delete(self, *, memory_id: t.Optional[str] = None, user_id: t.Optional[str] = None,
               filters: t.Optional[dict] = None) -> dict:
        self.ensure()
        if memory_id:
            return self._mem.delete(memory_id=memory_id)
        return self._mem.delete_by_filters(user_id=user_id, metadata=filters or {})


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
        user_id = args.get("user_id") or "default"
        metadata = args.get("metadata") or {}
        res = _MEM0.upsert(text=text, user_id=user_id, metadata=metadata)
        return _ok(res)
    except Exception as e:
        return _err(f"mem0.upsert failed: {e!s}")


def handle_memory_search(args: dict) -> dict:
    """
    Retrieve memories by semantic search.
    """
    try:
        query = args["query"].strip()
        user_id = args.get("user_id")
        top_k = int(args.get("top_k", 5))
        filters = args.get("filters") or {}
        res = _MEM0.search(query=query, user_id=user_id, top_k=top_k, filters=filters)
        return _ok(res)
    except Exception as e:
        return _err(f"mem0.search failed: {e!s}")


def handle_memory_delete(args: dict) -> dict:
    """
    Delete a specific memory or delete by filters.
    """
    try:
        memory_id = args.get("memory_id")
        user_id = args.get("user_id")
        filters = args.get("filters") or {}
        res = _MEM0.delete(memory_id=memory_id, user_id=user_id, filters=filters)
        return _ok(res)
    except Exception as e:
        return _err(f"mem0.delete failed: {e!s}")


def handle_rag_query(args: dict) -> dict:
    """
    Answer with your local RAG pipeline if available.
    """
    try:
        query = args["query"].strip()
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
            # naive "lead" summary — swap to model call in your agent layer if desired
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
        # VERY minimal safe evaluator (no names, no attributes)
        import ast
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                   ast.USub, ast.UAdd, ast.FloorDiv, ast.LShift, ast.RShift)
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
# Registry (BUILT-IN)
# =========================================================

REGISTRY = ToolRegistry(namespace="builtin")

# ---- Memory tools
REGISTRY.register(ToolSpec(
    name="memory.upsert",
    description="Store a memory (text + optional metadata) for a user.",
    parameters={
        "properties": {
            "text": {"type": "string", "description": "The memory content."},
            "user_id": {"type": "string", "description": "User identifier.", "default": "default"},
            "metadata": {"type": "object", "description": "Arbitrary key/values.", "additionalProperties": True},
        },
        "required": ["text"],
    },
    handler=handle_memory_upsert,
    tags=["memory", "mem0"],
))

REGISTRY.register(ToolSpec(
    name="memory.search",
    description="Semantic search over stored memories (mem0).",
    parameters={
        "properties": {
            "query": {"type": "string"},
            "user_id": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "filters": {"type": "object", "additionalProperties": True},
        },
        "required": ["query"],
    },
    handler=handle_memory_search,
    tags=["memory", "mem0"],
))

REGISTRY.register(ToolSpec(
    name="memory.delete",
    description="Delete a memory by id or by filters (user_id + metadata).",
    parameters={
        "properties": {
            "memory_id": {"type": "string"},
            "user_id": {"type": "string"},
            "filters": {"type": "object", "additionalProperties": True},
        },
        "required": [],
    },
    handler=handle_memory_delete,
    tags=["memory", "mem0"],
))

# ---- RAG
REGISTRY.register(ToolSpec(
    name="rag.query",
    description="Query the local knowledge base (RAG). Requires project RAG wiring.",
    parameters={
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
    handler=handle_rag_query,
    tags=["retrieval", "rag"],
))

# ---- Web + Utils
REGISTRY.register(ToolSpec(
    name="web.get",
    description="HTTP GET a URL and return status, headers, and text.",
    parameters={
        "properties": {
            "url": {"type": "string"},
            "timeout": {"type": "number", "default": 12},
            "headers": {"type": "object", "additionalProperties": True},
        },
        "required": ["url"],
    },
    handler=handle_web_get,
    tags=["web"],
))

REGISTRY.register(ToolSpec(
    name="text.summarize",
    description="Summarize a chunk of text locally (agent may replace with model call).",
    parameters={
        "properties": {
            "text": {"type": "string"},
            "max_chars": {"type": "integer", "default": 1000},
        },
        "required": ["text"],
    },
    handler=handle_summarize,
    tags=["nlp"],
))

REGISTRY.register(ToolSpec(
    name="math.eval",
    description="Evaluate a simple arithmetic expression safely.",
    parameters={
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
    },
    handler=handle_math_eval,
    tags=["math"],
))

REGISTRY.register(ToolSpec(
    name="files.list",
    description="List files under a directory (optionally recursive).",
    parameters={
        "properties": {
            "root": {"type": "string", "default": "."},
            "recursive": {"type": "boolean", "default": False},
            "max_items": {"type": "integer", "default": 100},
        },
        "required": [],
    },
    handler=handle_files_list,
    tags=["io"],
))

REGISTRY.register(ToolSpec(
    name="time.now",
    description="Get current timestamp information.",
    parameters={
        "properties": {
            "timezone": {"type": "string", "default": "local"},
        },
        "required": [],
    },
    handler=handle_time_now,
    tags=["util"],
))


# =========================================================
# ADK adapter
# =========================================================

def register_with_agent(agent: t.Any, registry: ToolRegistry = REGISTRY) -> None:
    """
    Wire all tools into your ADK agent.

    Your ADK wrapper should expose something like:
      agent.register_tool(name, description, parameters_schema, handler)

    If your ADK expects a different API, adjust here in one place.
    """
    if not hasattr(agent, "register_tool"):
        raise AttributeError("agent must expose `register_tool(name, description, parameters, handler)`")

    for spec in registry.tools:
        agent.register_tool(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            handler=spec.handler,
        )


# =========================================================
# Accessor for plugin loader (convention)
# =========================================================

def get_registry() -> ToolRegistry:
    """
    External loaders can import this to merge toolpacks.
    """
    return REGISTRY
