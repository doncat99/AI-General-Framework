# agents/assistant/adk_tool_client.py
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import inspect
from typing import Callable, Any, Optional, Dict, List
from loguru import logger

from utilities.base.base_agent import BaseAgent
from agents.assistant.tools.base_tool import BaseToolPack

# turn correlation (set by AssistantAgent.ainvoke)
try:
    from contextvars import ContextVar
except Exception:
    ContextVar = None  # py <3.7 won't happen here

TURN_ID = ContextVar("turn_id", default="-") if ContextVar else None

# ---------- logging knobs ----------
_MAX_PREVIEW = int((__import__("os").getenv("TOOL_LOG_MAX_CHARS") or "400"))
_LOG_ARGS = (__import__("os").getenv("TOOL_LOG_ARGS") or "true").lower() in ("1","true","yes","y","on")
_LOG_RESULT = (__import__("os").getenv("TOOL_LOG_RESULT") or "true").lower() in ("1","true","yes","y","on")

def _normalize_name(name: str | None) -> str:
    base = (name or "tool").strip()
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in base) or "tool"

def _preview(obj: Any) -> str:
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= _MAX_PREVIEW else s[:_MAX_PREVIEW] + "…"

def _mk_async_named(handler: Callable[..., Any], tool_name: str) -> Callable[..., Any]:
    """
    Wrap any callable into ADK-friendly signature:
        async def tool(args: dict) -> dict
    with rich logging around the call.
    """
    if inspect.iscoroutinefunction(handler):
        async def tool(args: Any):
            tid = TURN_ID.get("-") if TURN_ID else "-"
            if _LOG_ARGS:
                logger.info(f"[turn {tid}] [tool] {tool_name} ← { _preview(args) }")
            try:
                res = await handler(args)
                if _LOG_RESULT:
                    logger.info(f"[turn {tid}] [tool] {tool_name} → { _preview(res) }")
                return res
            except Exception as e:
                logger.exception(f"[turn {tid}] [tool] {tool_name} ✖ {e}")
                raise
    else:
        async def tool(args: Any):
            tid = TURN_ID.get("-") if TURN_ID else "-"
            if _LOG_ARGS:
                logger.info(f"[turn {tid}] [tool] {tool_name} ← { _preview(args) }")
            try:
                res = handler(args)
                if inspect.isawaitable(res):
                    res = await res
                if _LOG_RESULT:
                    logger.info(f"[turn {tid}] [tool] {tool_name} → { _preview(res) }")
                return res
            except Exception as e:
                logger.exception(f"[turn {tid}] [tool] {tool_name} ✖ {e}")
                raise

    tool.__name__ = tool_name
    tool.__qualname__ = tool_name
    tool.__doc__ = getattr(handler, "__doc__", None)
    setattr(tool, "__wrapped_handler__", handler)  # for introspection/debug
    return tool


class AdkToolClient(BaseAgent):
    """
    Minimal client for your tools framework.

    - Registers tools as callables via BaseAgent.add_tool / set_tools
    - Dedupes names and handlers
    - Provides batch registration to avoid repeated agent rebuilds
    - Relies on BaseAgent.run_async(prompt, system_prompt=None) and set_system_prompt()
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
        default_instruction = "You are a helpful, concise assistant."
        try:
            from agents.assistant.prompts.prompt import get_system_prompt
            default_instruction = get_system_prompt() or default_instruction
        except Exception:
            pass

        super().__init__(
            agent_name=agent_name,
            model_name=model_name,
            instruction=(instruction or default_instruction),
            tools=[],
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

        # duplicate tracking
        self._used_tool_names: set[str] = set()
        self._normalized_names: set[str] = set()
        self._registered_handler_ids: set[int] = set()

        logger.info(
            f"AdkToolClient initialized (model='{model_name}', agent='{agent_name}', app='{app_name}')"
        )

    # ---------- duplicate/name management ----------

    def _maybe_skip_alias(self, proposed: str) -> bool:
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

    # ---------- public registration API ----------

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """
        Register a single tool. Always wraps to async(dict)->dict and uses BaseAgent.add_tool().
        """
        if not callable(handler):
            raise TypeError("handler must be callable")

        if self._maybe_skip_alias(name):
            return
        if self._already_registered_handler(handler):
            logger.debug(f"Skipping duplicate handler for '{name}' (already registered).")
            return

        tool_name = self._dedupe_name(name)
        wrapper = _mk_async_named(handler, tool_name)
        # stash metadata for introspection / UIs (ADK itself ignores it)
        setattr(wrapper, "__tool_description__", description or "")
        setattr(wrapper, "__tool_parameters__", dict(parameters or {}))
        self.add_tool(wrapper)
        logger.info(f"Registered tool: {tool_name}")

    def register_tools_batch(self, defs: List[Dict[str, Any]]) -> int:
        """
        Batch-register a list of tool definitions in one shot using BaseAgent.set_tools()
        to minimize potential agent rebuilds.

        Each item in `defs` must have: name, description, parameters, handler.
        """
        new_wrappers: List[Callable[..., Any]] = []
        added = 0

        for d in defs:
            name = str(d["name"])
            handler = d["handler"]
            if self._maybe_skip_alias(name) or self._already_registered_handler(handler):
                continue
            tool_name = self._dedupe_name(name)
            wrapper = _mk_async_named(handler, tool_name)
            setattr(wrapper, "__tool_description__", d.get("description", "") or "")
            setattr(wrapper, "__tool_parameters__", dict(d.get("parameters") or {}))
            new_wrappers.append(wrapper)
            added += 1

        if new_wrappers:
            # single update through BaseAgent to avoid thrashing
            self.set_tools(self.tools + new_wrappers)
            logger.info(f"Batch-registered {added} tools.")
        return added

    def register_toolpack(self, pack: "BaseToolPack") -> int:
        """
        Let a pack call client.register_tool(...) repeatedly.
        Packs can choose to use register_tools_batch(...) internally.
        """
        try:
            n = pack.register(self)
            logger.info(f"Registered {n} tools from {pack.__class__.__name__}.")
            return n
        except Exception as e:
            logger.exception(f"Failed registering pack {pack.__class__.__name__}: {e}")
            return 0
