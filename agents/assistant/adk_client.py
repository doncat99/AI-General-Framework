# agents/assistant/adk_client.py
from __future__ import annotations
from typing import Callable, Iterable, Any, Optional, Mapping
import os
import inspect
from loguru import logger

from utilities.base.base_agent import BaseAgent


def _normalize_name(name: str | None) -> str:
    base = (name or "tool").strip()
    # allow only [A-Za-z0-9_]; replace others with '_'
    base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in base)
    return base or "tool"


def _ensure_async(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap sync to async; preserve docstring."""
    if inspect.iscoroutinefunction(fn):
        return fn

    async def _aw(*args, __fn=fn, **kwargs):
        res = __fn(*args, **kwargs)
        if inspect.isawaitable(res):
            return await res
        return res

    _aw.__doc__ = getattr(fn, "__doc__", None)
    return _aw


def _force_func_name(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    """
    Ensure the callable's __name__/__qualname__ are exactly `name`.
    If we cannot mutate, return a thin async wrapper with the desired identity.
    """
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


class ADKClient(BaseAgent):
    """
    ADK-only client built on BaseAgent (Google ADK + LiteLLM via OpenRouter).

    - Keeps a single underlying ADK agent/session.
    - Tools can be registered dynamically (delegates to BaseAgent.add_tool()).
    - run(prompt, system_prompt=...) lets you pass a system message at call-time.

    Model resolution order:
      explicit model_name -> env GOOGLE_MODEL -> env ADK_MODEL -> "google/gemini-1.5-flash"
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
            from agents.assistant.prompts.prompt import get_system_prompt  # optional
            default_instruction = get_system_prompt() or default_instruction
        except Exception:
            pass

        resolved_instruction = instruction or default_instruction

        # BaseAgent manages the ADK Agent + Runner + Session creation
        super().__init__(
            agent_name=agent_name,
            model_name=resolved_model,
            instruction=resolved_instruction,
            tools=[],  # start empty; you can register later
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        # Track used tool names to avoid ADK "Duplicate function declaration"
        self._used_tool_names: set[str] = set()

        logger.info(
            f"ADKClient initialized (model='{resolved_model}', agent='{agent_name}', app='{app_name}')"
        )

    # ---------- name utilities ----------

    def _dedupe_name(self, proposed: str) -> str:
        base = _normalize_name(proposed)
        if base not in self._used_tool_names:
            self._used_tool_names.add(base)
            return base
        i = 2
        while f"{base}_{i}" in self._used_tool_names:
            i += 1
        final = f"{base}_{i}"
        self._used_tool_names.add(final)
        return final

    # ---------- Tool registration ----------

    def register_tool(self, *args) -> None:
        """
        Accept both:
          - register_tool(func)
          - register_tool(name, func)
        Ensures:
          - unique, stable tool name (ADK schema id)
          - callable is async
          - callable object __name__/__qualname__ == exported tool name
        """
        if len(args) == 1:
            func = args[0]
            if not callable(func):
                raise TypeError("Tool must be a callable.")

            proposed_name = getattr(func, "__tool_name__", None) or getattr(func, "__name__", None) or "tool"
            tool_name = self._dedupe_name(str(proposed_name))

            fn_async = _ensure_async(func)
            fn_named = _force_func_name(fn_async, tool_name)

            self.add_tool(fn_named)
            logger.info(f"Registered tool: {tool_name} (total={len(self.tools)})")
            return

        if len(args) == 2:
            name, func = args
            if not callable(func):
                raise TypeError("Tool must be a callable.")

            tool_name = self._dedupe_name(str(name))

            fn_async = _ensure_async(func)
            fn_named = _force_func_name(fn_async, tool_name)

            self.add_tool(fn_named)
            logger.info(f"Registered tool: {tool_name} (total={len(self.tools)})")
            return

        raise TypeError("register_tool expects (func) or (name, func)")

    def register_tools(
        self,
        funcs: Iterable[Callable[..., Any]] | Mapping[str, Callable[..., Any]]
    ) -> None:
        """
        Register multiple tools.

        - Iterable[callable]: each item is a tool function.
        - Mapping[str, callable]: name->function mapping.
        """
        if isinstance(funcs, Mapping):
            for name, fn in funcs.items():
                self.register_tool(name, fn)
            return

        for fn in funcs:
            self.register_tool(fn)

    # ---------- Convenience run ----------

    async def run(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        """
        Single-turn ask; returns assistant text.
        If system_prompt is provided, it is inlined at the top of the message.
        Otherwise, uses the instruction from BaseAgent (which we set from prompts by default).
        """
        if system_prompt:
            prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"
        txt = await self.run_async(prompt)
        return txt or "I don’t have a response yet."

    # ---------- Optional helpers ----------

    def set_instruction(self, new_instruction: str) -> None:
        """Update the system instruction for subsequent calls."""
        self.update_instruction(new_instruction)
        logger.info("Agent instruction updated.")
