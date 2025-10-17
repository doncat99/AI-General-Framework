# agents/assistant/adk_client.py
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Callable, Iterable, Any, Optional, Mapping, Dict
import os
import inspect
from loguru import logger

from utilities.base.base_agent import BaseAgent


def _normalize_name(name: str | None) -> str:
    base = (name or "tool").strip()
    # allow only [A-Za-z0-9_]; replace others with '_'
    base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in base)
    return base or "tool"


def _mk_async_named(handler: Callable[..., Any], tool_name: str) -> Callable[..., Any]:
    """
    Create a clean async wrapper with a SIMPLE signature so ADK can parse it:
        async def tool(args: dict) -> dict:
            ...
    No defaulted closure args (like __fn=handler), so no param pollution.
    """
    if inspect.iscoroutinefunction(handler):
        async def tool(args: Any):
            return await handler(args)
    else:
        async def tool(args: Any):
            res = handler(args)
            if inspect.isawaitable(res):
                return await res
            return res

    tool.__name__ = tool_name
    tool.__qualname__ = tool_name
    tool.__doc__ = getattr(handler, "__doc__", None)
    # keep a reference for duplicate filtering
    setattr(tool, "__wrapped_handler__", handler)
    return tool


def _ensure_object_schema(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Normalize a function-tool parameters schema to JSON Schema object form.
    Accepts shapes like {"properties":{...},"required":[...]} and injects type=object.
    """
    params = dict(params or {})
    if "type" not in params:
        params["type"] = "object"
    if "properties" not in params:
        params["properties"] = {}
    if "required" not in params:
        params["required"] = []
    return params


class ADKClient(BaseAgent):
    """
    ADK-only client built on BaseAgent.

    Supports schema registration when BaseAgent exposes it; otherwise falls back to
    a simple wrapper signature so ADK's auto function calling can infer cleanly.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        agent_name: str = "assistant",
        instruction: Optional[str] = None,
        app_name: str = "ADK_APP",
        user_id: str = "user",
        session_id: str = "default_session",
        use_cache: bool = True,
        cache_dir: str = ".cache",
    ) -> None:
        resolved_model = (
            model_name
            or os.getenv("GOOGLE_MODEL")
            or os.getenv("ADK_MODEL")
            or "google/gemini-1.5-flash"
        )

        # Try to source a project-wide default system prompt; fall back if missing
        default_instruction = "You are a helpful, concise assistant."
        try:
            from agents.assistant.prompts.prompt import get_system_prompt
            default_instruction = get_system_prompt() or default_instruction
        except Exception:
            pass

        resolved_instruction = instruction or default_instruction

        # BaseAgent manages the ADK Agent + Runner + Session creation
        super().__init__(
            agent_name=agent_name,
            model_name=resolved_model,
            instruction=resolved_instruction,
            tools=[],
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        # Track used tool names to avoid ADK "Duplicate function declaration"
        self._used_tool_names: set[str] = set()
        # Track normalized-name collisions to skip alias duplicates (e.g., memory_upsert vs memory.upsert)
        self._normalized_names: set[str] = set()
        # Track registered callables to avoid double-registration by identity
        self._registered_handler_ids: set[int] = set()

        logger.info(
            f"ADKClient initialized (model='{resolved_model}', agent='{agent_name}', app='{app_name}')"
        )

    # ---------- name & duplicate utilities ----------

    def _maybe_skip_alias(self, proposed: str) -> bool:
        """
        If a proposed name normalizes to a name we've already registered,
        treat it as an alias collision and skip the duplicate registration.
        Avoids creating memory_upsert_2 when both memory_upsert and memory.upsert are present.
        """
        base = _normalize_name(proposed)
        if base in self._normalized_names:
            logger.debug(f"Skipping duplicate alias '{proposed}' (normalized='{base}')")
            return True
        return False

    def _dedupe_name(self, proposed: str) -> str:
        base = _normalize_name(proposed)
        if base not in self._used_tool_names:
            self._used_tool_names.add(base)
            self._normalized_names.add(base)
            return base
        # True name collision with a different handler; suffix
        i = 2
        while f"{base}_{i}" in self._used_tool_names:
            i += 1
        final = f"{base}_{i}"
        self._used_tool_names.add(final)
        self._normalized_names.add(final)
        return final

    def _already_registered_handler(self, handler: Callable[..., Any]) -> bool:
        hid = id(handler)
        if hid in self._registered_handler_ids:
            return True
        self._registered_handler_ids.add(hid)
        return False

    def _supports_schema_registration(self) -> bool:
        """
        Return True if BaseAgent exposes a schema-capable API.
        Prefer add_function_tool; otherwise verify add_tool signature supports 4 args.
        """
        if hasattr(self, "add_function_tool"):
            return True
        add_tool = getattr(self, "add_tool", None)
        if not callable(add_tool):
            return False
        try:
            sig = inspect.signature(add_tool)
            # For a bound method, required positional params exclude 'self'.
            req_pos = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect._empty
            ]
            # Expecting (name, description, parameters, handler) → 4 required args
            return len(req_pos) >= 4 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        except Exception:
            return False

    def _register_with_schema(self, tool_name: str, description: str, parameters: Dict[str, Any], handler: Callable[..., Any]) -> bool:
        """
        Try all known schema-bearing registrations. Return True on success.
        """
        if not self._supports_schema_registration():
            return False

        parameters = _ensure_object_schema(parameters)

        # Preferred: dedicated function-tool API if present
        if hasattr(self, "add_function_tool"):
            try:
                self.add_function_tool(tool_name, description, parameters, handler)
                logger.info(f"Registered tool (schema via add_function_tool): {tool_name}")
                return True
            except Exception as e:
                logger.debug(f"add_function_tool failed for {tool_name}: {e}")

        # Fallback schema path via add_tool(name, desc, params, handler) if supported
        add_tool = getattr(self, "add_tool", None)
        if callable(add_tool):
            try:
                add_tool(tool_name, description, parameters, handler)  # type: ignore[misc]
                logger.info(f"Registered tool (schema via add_tool): {tool_name}")
                return True
            except TypeError as e:
                # Don't escalate — many BaseAgent impls only accept a single callable here.
                logger.debug(f"schema add_tool failed for {tool_name}: {e}")
            except Exception as e:
                logger.debug(f"schema add_tool failed for {tool_name}: {e}")

        return False

    def register_tool(self, *args, **kwargs) -> None:
        """
        Accepted forms:
          - register_tool(func)
          - register_tool(name, func)
          - register_tool(name, description, parameters, handler)
          - register_tool(name=..., description=..., parameters=..., handler=...)

        If schema-based registration isn't supported by BaseAgent, we fallback to a simple
        `(args: dict) -> dict` async wrapper so ADK auto-calling can parse the signature.
        """
        # ---- KW schema form ----
        if kwargs:
            name = kwargs.get("name")
            description = kwargs.get("description", "") or ""
            parameters = kwargs.get("parameters") or {}
            handler = kwargs.get("handler")
            if not (name and callable(handler)):
                raise TypeError("Schema registration requires 'name' and callable 'handler'.")

            # Skip alias duplicates (e.g., dotted vs underscore)
            if self._maybe_skip_alias(str(name)):
                return
            if self._already_registered_handler(handler):
                logger.debug(f"Skipping duplicate handler for '{name}' (already registered).")
                return

            tool_name = self._dedupe_name(str(name))
            # Prefer registering the raw handler with schema if supported
            if self._register_with_schema(tool_name, description, parameters, handler):
                return

            # Fallback: simple wrapper, no schema path
            self.add_tool(_mk_async_named(handler, tool_name))
            logger.info(f"Registered tool without explicit schema (fallback): {tool_name}")
            return

        # ---- positional forms ----
        if len(args) == 1 and callable(args[0]):
            func = args[0]
            # Avoid duplicate handlers
            if self._already_registered_handler(func):
                logger.debug("Skipping duplicate handler registration (func).")
                return
            proposed = getattr(func, "__tool_name__", None) or getattr(func, "__name__", None) or "tool"
            if self._maybe_skip_alias(str(proposed)):
                return
            tool_name = self._dedupe_name(str(proposed))
            self.add_tool(_mk_async_named(func, tool_name))
            logger.info(f"Registered tool: {tool_name} (no explicit schema)")
            return

        if len(args) == 2 and callable(args[1]):
            name, func = args  # type: ignore[misc]
            if self._maybe_skip_alias(str(name)):
                return
            if self._already_registered_handler(func):
                logger.debug(f"Skipping duplicate handler for '{name}'.")
                return
            tool_name = self._dedupe_name(str(name))
            self.add_tool(_mk_async_named(func, tool_name))
            logger.info(f"Registered tool: {tool_name} (no explicit schema)")
            return

        if len(args) == 4:
            name, description, parameters, handler = args
            if not callable(handler):
                raise TypeError("handler must be callable")
            if self._maybe_skip_alias(str(name)):
                return
            if self._already_registered_handler(handler):
                logger.debug(f"Skipping duplicate handler for '{name}'.")
                return
            tool_name = self._dedupe_name(str(name))
            if self._register_with_schema(tool_name, str(description or ""), dict(parameters or {}), handler):
                return
            # Fallback
            self.add_tool(_mk_async_named(handler, tool_name))
            logger.info(f"Registered tool without explicit schema (fallback): {tool_name}")
            return

        raise TypeError(
            "register_tool expects one of: (func), (name, func), "
            "(name, description, parameters, handler), or keyword schema."
        )

    def register_tools(
        self,
        funcs: Iterable[Callable[..., Any]] | Mapping[str, Callable[..., Any]]
    ) -> None:
        if isinstance(funcs, Mapping):
            for name, fn in funcs.items():
                # Skip alias duplicates by normalized name before calling register_tool
                if self._maybe_skip_alias(str(name)):
                    continue
                self.register_tool(name, fn)
            return
        for fn in funcs:
            self.register_tool(fn)

    async def run(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        if system_prompt:
            prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
        txt = await self.run_async(prompt)
        return txt or "I don’t have a response yet."

    def set_instruction(self, new_instruction: str) -> None:
        self.update_instruction(new_instruction)
        logger.info("Agent instruction updated.")
