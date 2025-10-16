# agents/assistant/tools/external_loader.py
from __future__ import annotations
from typing import Dict, Any, Callable
from loguru import logger
import importlib
import inspect

def _mk_async(handler: Callable) -> Callable:
    """Wrap sync/async handlers into a uniform async(payload) callable."""
    async def _runner(payload: Dict[str, Any] | None = None, *args, **kwargs):
        payload = payload or {}
        try:
            res = handler(payload)  # prefer dict payload signature
        except TypeError:
            # some tools are call(*args, **kwargs)
            res = handler(*args, **kwargs)
        if inspect.isawaitable(res):
            res = await res
        return res
    return _runner

def _coerce_to_mapping(candidate: Any) -> Dict[str, Callable]:
    """
    Accept ToolRegistry, dict[name->handler], iterable of tools/callables, or a single callable.
    Returns {name: async_callable}.
    """
    mapping: Dict[str, Callable] = {}

    # ToolRegistry (has .tools list of ToolSpec objects with .name/.handler)
    if hasattr(candidate, "tools"):
        for spec in candidate.tools:
            name = getattr(spec, "name", None) or "tool"
            handler = getattr(spec, "handler", None)
            if handler is None:
                continue
            mapping[str(name)] = _mk_async(handler)
        return mapping

    # dict[name -> handler]
    if isinstance(candidate, dict):
        for name, handler in candidate.items():
            mapping[str(name)] = _mk_async(handler)
        return mapping

    # iterable (objects or callables)
    try:
        for t in candidate or []:
            name = getattr(t, "name", None) or getattr(t, "__name__", None) or "tool"
            handler = getattr(t, "handler", None) or getattr(t, "arun", None) or getattr(t, "run", None) or t
            if handler is None:
                continue
            mapping[str(name)] = _mk_async(handler)
        return mapping
    except TypeError:
        # single callable fallback
        if callable(candidate):
            name = getattr(candidate, "__name__", "tool")
            mapping[name] = _mk_async(candidate)
            return mapping
        return {}

def load_external_tools() -> Dict[str, Any]:
    """
    Flexible loader that tries several module/attribute combos:
      - agent_tools.get_registry()  (preferred)
      - agent_tools.REGISTRY
      - agent_tools.get_agent_tools()  (legacy)
    Returns a {name: async_callable} mapping.
    """
    module_candidates = [
        "agent.assistant.tools.agent_tools",  # your package layout
        "agent.tools.agent_tools",
        "tools.agent_tools",
        # last-resort relative import path if this module is a package sibling
        # note: importlib can't resolve purely relative like ".agent_tools" from here,
        # so we keep absolute module candidates above.
    ]
    attr_candidates = ("get_registry", "REGISTRY", "get_agent_tools")

    for mod_path in module_candidates:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue

        for attr in attr_candidates:
            obj = getattr(mod, attr, None)
            if obj is None:
                continue
            try:
                candidate = obj() if callable(obj) and attr.startswith("get_") else obj
                mapping = _coerce_to_mapping(candidate)
                if mapping:
                    logger.info(f"Loaded {len(mapping)} external tools via {mod_path}.{attr}")
                    return mapping
            except Exception as e:
                logger.warning(f"Failed loading tools from {mod_path}.{attr}: {e}")

    logger.warning("No external tools found; continuing without plugins.")
    return {}
