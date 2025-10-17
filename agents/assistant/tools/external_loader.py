# agents/assistant/tools/external_loader.py
from __future__ import annotations
from typing import Dict, Callable, Any, Iterable, Mapping, List, Tuple
from loguru import logger
import importlib
import inspect
import re


def _ensure_async_callable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap sync funcs to async; preserve docstring."""
    if inspect.iscoroutinefunction(fn):
        return fn

    async def _async_wrapper(*args, __fn=fn, **kwargs):
        res = __fn(*args, **kwargs)
        if inspect.isawaitable(res):
            return await res
        return res

    _async_wrapper.__doc__ = getattr(fn, "__doc__", None)
    return _async_wrapper


def _set_func_name(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Make the callable carry the exported name (helps some registrars)."""
    try:
        fn.__name__ = name
        fn.__qualname__ = name
        return fn
    except Exception:
        async def _wrapper(*args, __fn=fn, **kwargs):
            res = __fn(*args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res
        _wrapper.__name__ = name
        _wrapper.__qualname__ = name
        _wrapper.__doc__ = getattr(fn, "__doc__", None)
        return _wrapper


_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_]+")

def _sanitize_name(n: str) -> str:
    n = (n or "tool").strip()
    n = _SANITIZE_RE.sub("_", n)
    return n or "tool"

def _uniquify(name: str, used: set[str]) -> str:
    base = _sanitize_name(name)
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    used.add(f"{base}_{i}")
    return f"{base}_{i}"


def _adapt_tool_like(obj: Any) -> Tuple[str, Callable[..., Any]] | None:
    """Adapt LangChain-like objects exposing .arun/.run into (name, callable)."""
    name = getattr(obj, "name", None) or getattr(obj, "__name__", None) or "tool"
    if hasattr(obj, "arun") and callable(getattr(obj, "arun")):
        async def _runner(*args, __t=obj, **kwargs):
            return await __t.arun(*args, **kwargs)
        _runner.__doc__ = getattr(obj, "__doc__", None)
        return str(name), _runner
    if hasattr(obj, "run") and callable(getattr(obj, "run")):
        def _runner_sync(*args, __t=obj, **kwargs):
            return __t.run(*args, **kwargs)
        _runner_sync.__doc__ = getattr(obj, "__doc__", None)
        return str(name), _runner_sync
    return None


def _collect_candidates(mod) -> List[Tuple[str, Callable[..., Any]]]:
    """
    Accept any of:
      - get_registry() -> Mapping[str, callable]
      - get_agent_tools() -> Iterable[callable|tool-like]
      - module-level REGISTRY / TOOLS / registry
    Return list[(raw_name, callable)]
    """
    candidates: List[Tuple[str, Callable[..., Any]]] = []

    # 1) get_registry()
    get_registry = getattr(mod, "get_registry", None)
    if callable(get_registry):
        try:
            reg = get_registry()
            if isinstance(reg, Mapping):
                for raw_name, fn in reg.items():
                    if callable(fn):
                        candidates.append((str(raw_name), fn))
                    else:
                        adapted = _adapt_tool_like(fn)
                        if adapted:
                            candidates.append(adapted)
        except Exception as e:
            logger.warning(f"Calling {mod.__name__}.get_registry failed: {e}")

    # 2) get_agent_tools()
    if not candidates:
        get_tools = getattr(mod, "get_agent_tools", None)
        if callable(get_tools):
            try:
                items = get_tools() or []
                if isinstance(items, Mapping):
                    for raw_name, fn in items.items():
                        if callable(fn):
                            candidates.append((str(raw_name), fn))
                        else:
                            adapted = _adapt_tool_like(fn)
                            if adapted:
                                candidates.append(adapted)
                else:
                    for t in items:
                        if callable(t):
                            nm = getattr(t, "__tool_name__", None) or getattr(t, "__name__", None) or "tool"
                            candidates.append((str(nm), t))
                        else:
                            adapted = _adapt_tool_like(t)
                            if adapted:
                                candidates.append(adapted)
            except Exception as e:
                logger.warning(f"Calling {mod.__name__}.get_agent_tools failed: {e}")

    # 3) module-level registries
    if not candidates:
        for attr in ("REGISTRY", "TOOLS", "registry"):
            if hasattr(mod, attr):
                obj = getattr(mod, attr)
                if isinstance(obj, Mapping):
                    for raw_name, fn in obj.items():
                        if callable(fn):
                            candidates.append((str(raw_name), fn))
                        else:
                            adapted = _adapt_tool_like(fn)
                            if adapted:
                                candidates.append(adapted)
                elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                    for t in obj:
                        if callable(t):
                            nm = getattr(t, "__tool_name__", None) or getattr(t, "__name__", None) or "tool"
                            candidates.append((str(nm), t))
                        else:
                            adapted = _adapt_tool_like(t)
                            if adapted:
                                candidates.append(adapted)
    return candidates


def _extract_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Pull a JSON Schema-like parameters object off the callable if present."""
    for attr in ("_tool_parameters", "_tool_schema", "parameters", "schema"):
        obj = getattr(fn, attr, None)
        if isinstance(obj, dict):
            obj = dict(obj)  # copy
            if obj.get("type") != "object":
                obj["type"] = "object"
            obj.setdefault("properties", {})
            obj.setdefault("required", [])
            return obj
    return {"type": "object", "properties": {}, "required": []}


def _extract_desc(fn: Callable[..., Any]) -> str:
    return getattr(fn, "_tool_description", None) or (fn.__doc__ or "").strip() or ""


def load_external_tools() -> Dict[str, Dict[str, Any]]:
    """
    Import `agents.assistant.tools.agent_tools` and return an **ADK-compatible registry**:

        { "<name>": { "handler": <async callable>, "description": "...", "parameters": {...} } }

    This shape prevents the “not a compatible registry; skipping merge” warning.
    """
    try:
        mod = importlib.import_module("agents.assistant.tools.agent_tools")
    except Exception as e:
        logger.warning(f"No external tools found; continuing without plugins. {e}")
        return {}

    # If the module already exposes an ADK-style registry, use it verbatim.
    for attr in ("EXPORTED_TOOLS", "exported_tools", "ADK_REGISTRY"):
        reg = getattr(mod, attr, None)
        if isinstance(reg, Mapping) and reg and all(isinstance(v, Mapping) and "handler" in v for v in reg.values()):
            # Ensure handlers are async & names are stable
            out: Dict[str, Dict[str, Any]] = {}
            used: set[str] = set()
            for raw_name, spec in reg.items():
                name = _uniquify(str(raw_name), used)
                handler = spec["handler"]
                fn_async = _ensure_async_callable(handler)
                fn_named = _set_func_name(fn_async, name)
                out[name] = {
                    "handler": fn_named,
                    "description": spec.get("description", "") or _extract_desc(handler),
                    "parameters": spec.get("parameters", {}) or _extract_schema(handler),
                }
            logger.info(f"Loaded {len(out)} external tools via agents.assistant.tools.agent_tools")
            return out

    # Otherwise, synthesize from whatever we can find.
    candidates = _collect_candidates(mod)
    if not candidates:
        logger.info("No external tools found in agent_tools.")
        return {}

    used: set[str] = set()
    registry: Dict[str, Dict[str, Any]] = {}
    for raw_name, fn in candidates:
        # normalize -> async callable
        fn_async = _ensure_async_callable(fn)
        # enforce unique, sanitized export name
        name = _uniquify(raw_name, used)
        # ensure the callable itself carries that name for schema/trace readability
        fn_named = _set_func_name(fn_async, name)
        registry[name] = {
            "handler": fn_named,
            "description": _extract_desc(fn),
            "parameters": _extract_schema(fn),
        }

    logger.info(f"Loaded {len(registry)} external tools via agents.assistant.tools.agent_tools")
    return registry
