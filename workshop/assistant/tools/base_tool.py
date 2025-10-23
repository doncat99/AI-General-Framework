# agents/assistant/tools/base_tool.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional
import inspect


def _ensure_object_schema(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize to JSON Schema object shape:
      {"type":"object","properties":{...},"required":[...]}
    """
    params = dict(params or {})
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    params.setdefault("required", [])
    return params


async def _invoke(handler: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """
    Try handler(**kwargs); fallback handler(kwargs). Handle sync/async.
    """
    try:
        res = handler(**kwargs)
    except TypeError:
        res = handler(kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


@dataclass(frozen=True)
class ToolSpec:
    """
    Declarative tool spec used by BaseToolPack.
    """
    handler: Callable[..., Any]            # async or sync; will be wrapped
    description: str
    parameters: Dict[str, Any]             # JSON schema (OpenAI-style; for your UIs/logs)


class BaseToolPack:
    """
    Base class for tool collections. Subclasses just override tools()
    and optionally pass defaults (e.g., base_url) and a name prefix.
    """

    def __init__(
        self,
        *,
        defaults: Optional[Dict[str, Any]] = None,
        name_prefix: Optional[str] = None,
    ) -> None:
        self._defaults = dict(defaults or {})
        self._prefix = (name_prefix or "").strip()

    # ---- registration ----
    def register(self, client) -> int:
        """
        Register all tools in this pack into the ADK tool client.
        Returns the number of tools registered.
        """
        count = 0
        for name, spec in self.tools().items():
            final_name = f"{self._prefix}{name}" if self._prefix else name
            wrapped = self._wrap_with_defaults(spec.handler)
            schema = _ensure_object_schema(spec.parameters)
            client.register_tool(
                name=final_name,
                description=spec.description,
                parameters=schema,
                handler=wrapped,
            )
            count += 1
        return count

    # ---- subclass API ----
    def tools(self) -> Mapping[str, ToolSpec]:
        """
        Return mapping: tool_name -> ToolSpec.
        Subclasses must implement.
        """
        raise NotImplementedError

    # ---- helpers ----
    def _wrap_with_defaults(self, func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
        """
        Create an async(**kwargs)->dict callable that merges self._defaults into kwargs
        (without overwriting explicit values), then calls the underlying handler.
        """
        async def _call(**kwargs: Dict[str, Any]) -> Any:
            merged = dict(self._defaults)
            merged.update(kwargs or {})
            return await _invoke(func, merged)

        _call.__name__ = getattr(func, "__name__", "tool")
        _call.__doc__ = getattr(func, "__doc__", None)
        return _call
