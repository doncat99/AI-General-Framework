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
    """
    Ensure the callable carries the chosen tool name so ADK schema
    doesn’t see multiple anonymous wrappers like '_runner'.
    """
    try:
        fn.__name__ = name
        fn.__qualname__ = name
        return fn
    except Exception:
        # Fallback thin wrapper (preserve docstring)
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
    if not n:
        n = "tool"
    return n


def _uniquify(name: str, used: set[str]) -> str:
    base = _sanitize_name(name)
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    final = f"{base}_{i}"
    used.add(final)
    return final


def _adapt_tool_like(obj: Any) -> Tuple[str, Callable[..., Any]] | None:
    """
    Adapt LangChain-like objects or any object exposing .arun/.run into (name, callable).
    """
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
    Look for any of these in the module, in order:
      - get_registry() -> Mapping[str, callable]   (preferred)
      - get_agent_tools() -> Iterable[callable|tool-like]
      - module-level REGISTRY / TOOLS / registry
    Return a flat list[(raw_name, callable)]
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
            else:
                logger.warning(f"{mod.__name__}.get_registry did not return a Mapping; ignoring.")
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


def load_external_tools() -> Dict[str, Callable[..., Any]]:
    """
    Import `agents.assistant.tools.agent_tools` (or the equivalent installed package)
    and normalize its exports into {tool_name: async_callable} with **unique names**.
    """
    try:
        mod = importlib.import_module("agents.assistant.tools.agent_tools")
    except Exception as e:
        logger.warning(f"No external tools found; continuing without plugins. {e}")
        return {}

    candidates = _collect_candidates(mod)
    if not candidates:
        logger.info("No external tools found in agent_tools.")
        return {}

    used: set[str] = set()
    tools: Dict[str, Callable[..., Any]] = {}

    for raw_name, fn in candidates:
        # normalize -> async callable
        fn_async = _ensure_async_callable(fn)
        # enforce unique, sanitized export name
        name = _uniquify(raw_name, used)
        # ensure the callable itself carries that name for ADK schema
        fn_named = _set_func_name(fn_async, name)
        tools[name] = fn_named

    logger.info(f"Loaded {len(tools)} external tools via agents.assistant.tools.agent_tools")
    return tools
