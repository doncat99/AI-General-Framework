# utilities/base/base_agent.py
from __future__ import annotations

import os
import pickle
import hashlib
from typing import Optional, List, Callable, Any
from pathlib import Path
from functools import wraps
import asyncio

from loguru import logger

from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from google.adk.agents import LlmAgent as Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import litellm

from config import config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _as_bool(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "y", "on")


# Activate OpenInference unless disabled via config/env
# if not _as_bool(os.getenv("OPENINFERENCE_DISABLED")) and not _as_bool(os.getenv("OTEL_SDK_DISABLED")):
GoogleADKInstrumentor().instrument()

# optional: verbose HTTP/debug for LiteLLM routing
if config.DEBUG:
    try:
        litellm._turn_on_debug()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Simple pickle cache decorator
# ---------------------------------------------------------------------------

def pickle_cache(cache_subdir: str):
    """Decorator to cache function results using pickle files on disk."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not getattr(self, "use_cache", False) or not getattr(self, "cache_dir", None):
                return await func(self, *args, **kwargs)
            cache_dir = Path(self.cache_dir) / cache_subdir
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
            fp = cache_dir / f"{key}.pkl"
            if fp.exists():
                return pickle.loads(fp.read_bytes())
            result = await func(self, *args, **kwargs)
            if result is not None:
                fp.write_bytes(pickle.dumps(result))
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    ADK-based base agent that supports dynamic tool updates.

    Key points:
    - Builds one Agent/Runner up-front.
    - `set_tools` tries to update tools IN-PLACE (`self.agent.tools = ...`).
      If the ADK version refuses, we fall back to `_build_agent_and_runner()`.
    - Safe in both sync and async environments (lazy session creation).
    """

    def __init__(
        self,
        agent_name: str,
        model_name: Optional[str],
        instruction: str,
        tools: Optional[List[Callable[..., Any]]] = None,
        app_name: str = "default_app",
        user_id: str = "default_user",
        session_id: str = "default_session",
        use_cache: bool = True,
        cache_dir: str = ".cache",
    ):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # Persist config for rebuilds or in-place updates
        self._agent_name = agent_name
        self._model_name = model_name or ""
        if not self._model_name:
            raise ValueError("model_name must be provided")
        self._instruction = instruction  # canonical "system prompt"

        # The current tool list we consider "source of truth"
        self.tools: List[Callable[..., Any]] = list(tools or [])

        # In-memory sessions are perfect for local/dev
        self.session_service = InMemorySessionService()
        self._session_task: Optional[asyncio.Task] = None

        # Create or schedule the session
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop → safe to block synchronously
            asyncio.run(self._create_session())
        else:
            # Loop running → schedule and await later in run_async()
            self._session_task = loop.create_task(self._create_session())

        # Build agent/runner once
        self._build_agent_and_runner()

    async def _create_session(self):
        # idempotent; ADK will handle duplicates
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )

    def _normalize_model(self, model_name: str) -> str:
        """Normalize to OpenRouter-style id once."""
        return model_name if model_name.startswith("openrouter/") else f"openrouter/{model_name}"

    def _build_agent_and_runner(self) -> None:
        """(Re)build Agent + Runner from current config + self.tools."""
        model_id = self._normalize_model(self._model_name)
        self.agent = Agent(
            name=self._agent_name,
            model=LiteLlm(
                model=model_id,
                api_key=config.OPEN_ROUTER_API_KEY,
                api_base=config.OPEN_ROUTER_BASE_URL,
            ),
            instruction=self._instruction,
            tools=self.tools,
        )
        self.runner = Runner(agent=self.agent, app_name=self.app_name, session_service=self.session_service)

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def set_tools(self, tools: List[Callable[..., Any]]) -> None:
        """Replace the tool list (in-place if possible, otherwise rebuild)."""
        self.tools = list(tools or [])
        try:
            # Many ADK versions allow this (pydantic model kept mutable for lists)
            self.agent.tools = self.tools
        except Exception:
            # Fall back to a full rebuild if in-place update isn't supported
            self._build_agent_and_runner()

    def add_tool(self, tool: Callable[..., Any]) -> None:
        """Append a tool; prefer in-place update, rebuild only if needed."""
        if not callable(tool):
            raise TypeError("Tool must be callable")
        self.tools.append(tool)
        try:
            self.agent.tools = self.tools
        except Exception:
            self._build_agent_and_runner()

    def remove_tool(self, tool: Callable[..., Any]) -> None:
        """Remove a tool if present."""
        try:
            self.tools.remove(tool)
        except ValueError:
            return
        try:
            self.agent.tools = self.tools
        except Exception:
            self._build_agent_and_runner()

    # ------------------------------------------------------------------
    # System prompt / instruction
    # ------------------------------------------------------------------

    def set_system_prompt(self, new_prompt: str) -> None:
        """
        Canonical way to set the agent/system prompt.
        Tries to update the underlying ADK Agent in-place; rebuilds if required.
        """
        self._instruction = new_prompt or ""
        try:
            # Many ADK versions accept direct mutation of pydantic model fields
            self.agent.instruction = self._instruction
        except Exception:
            # Fall back to a full rebuild to propagate the new instruction
            self._build_agent_and_runner()
        logger.info("[agent] System prompt updated.")

    def set_instruction(self, new_instruction: str) -> None:
        """Alias kept for callers that use 'instruction' terminology."""
        self.set_system_prompt(new_instruction)

    # ------------------------------------------------------------------
    # Run (single entrypoint)
    # ------------------------------------------------------------------

    # @pickle_cache(cache_subdir="run_async")
    async def run_async(self, message_text: str, *, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Main entrypoint. If `system_prompt` is provided, it is applied via set_system_prompt()
        before running this turn (and persists for subsequent turns).
        """
        # Update system prompt if provided
        if system_prompt is not None:
            try:
                self.set_system_prompt(system_prompt)
            except Exception:
                logger.debug("[agent] Could not set system prompt dynamically; continuing without it.")

        # If session creation was scheduled at __init__, await it now
        if self._session_task and not self._session_task.done():
            await self._session_task

        user_msg = types.Content(role="user", parts=[types.Part(text=message_text)])
        final_text: Optional[str] = None
        try:
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=user_msg,
            ):
                if event.is_final_response():
                    try:
                        final_text = event.content.parts[0].text
                    except Exception:
                        final_text = None
            return final_text
        except Exception as e:
            logger.error(f"[agent] Error running agent: {e}")
            return None
