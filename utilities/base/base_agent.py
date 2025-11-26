# utilities/base/base_agent.py
from __future__ import annotations

import os
import asyncio
import inspect
from typing import Optional, List, Callable, Any, Tuple, Dict

from loguru import logger

from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from google.adk.agents import LlmAgent as Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import config, langfuse  # <- langfuse client
from utilities.base.prompt_manager import PromptManager, PromptVars  # <- decoupled PM

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _as_bool(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "y", "on")


# Activate OpenInference unless disabled via config/env
if not _as_bool(os.getenv("OPENINFERENCE_DISABLED")):
    try:
        GoogleADKInstrumentor().instrument()
    except Exception as e:
        logger.debug(f"[agent] OpenInference instrumentation skipped: {e}")

# optional: verbose HTTP/debug for LiteLLM routing
try:
    import litellm  # type: ignore
    if config.DEBUG:
        try:
            litellm._turn_on_debug()
        except Exception:
            pass
except Exception:
    litellm = None  # type: ignore


# ---------------------------------------------------------------------------
# BaseAgent (with decoupled Prompt Manager + robust fallback)
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    ADK-based base agent that supports dynamic tool updates + Langfuse Prompt Management.

    - Prompt management fully delegated to PromptManager (compile + fallbacks).
    - Robust handling of ADK runner (async-iterable vs awaitable).
    - Direct LiteLLM fallback if the ADK stream fails or yields no text.
    - Correct (sync) Langfuse span context usage.
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
        api_key: Optional[str] = config.OPENROUTER_API_KEY,
        base_url: Optional[str] = config.OPENROUTER_BASE_URL,
    ):
        self.app_name = app_name
        self.user_id = user_id
        self.session_id = session_id
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.base_url = base_url

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

        # Prompt Manager (decoupled)
        self.pm = PromptManager()

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
                api_key=self.api_key,
                api_base=self.base_url,
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
    # Direct LLM fallback (LiteLLM → OpenRouter)
    # ------------------------------------------------------------------

    async def _direct_llm_fallback(self, messages: List[Dict[str, str]], timeout: int = 60) -> Optional[str]:
        """
        One-shot completion if ADK streaming fails. Requires `litellm` to be available
        and OPENROUTER_API_KEY configured in `config`.
        """
        if not litellm:
            logger.error("[agent] Fallback requested but litellm is not installed.")
            return None
        try:
            resp = await litellm.acompletion(
                model=self._normalize_model(self._model_name),
                api_key=config.OPENROUTER_API_KEY,
                api_base=config.OPENROUTER_BASE_URL or None,
                messages=messages,
                timeout=timeout,
            )
            txt = (resp.choices[0].message.get("content") or "").strip()
            return txt or None
        except Exception as e:
            logger.error(f"[agent] Fallback LLM failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Run (single entrypoint)
    # ------------------------------------------------------------------

    async def run_async(
        self,
        message_text: str,
        *,
        system_prompt: Optional[str] = None,
        prompt_vars: Optional[PromptVars] = None,
    ) -> Optional[str]:
        """
        Main entrypoint.

        If `prompt_vars` is provided with a valid `prompt_name`, we will:
          - fetch & compile the managed prompt via PromptManager,
          - set the agent's system prompt to the compiled system text (or `system_prompt` if given),
          - optionally prepend any compiled 'user' template content to this turn's message,
            depending on `prompt_vars.apply` ('system'|'user'|'both').

        Otherwise, if `system_prompt` is provided, we just set that for this turn.

        The updated instruction persists for subsequent turns (explicit by design).
        """
        # If session creation was scheduled at __init__, await it now
        if self._session_task and not self._session_task.done():
            await self._session_task

        # Resolve prompt via PromptManager (or fallbacks)
        resolved_instruction = (system_prompt or self._instruction or "").strip()
        resolved_user_text = message_text
        lf_prompt_obj = None
        lf_cfg: Dict[str, Any] = {}

        if prompt_vars is not None:
            resolved_instruction, resolved_user_text, lf_prompt_obj, lf_cfg = self.pm.compile_prompt(
                pv=prompt_vars,
                fallback_instruction=system_prompt or self._instruction,
                user_message_text=message_text,
            )

        # Apply the instruction for this and future turns
        try:
            self.set_system_prompt(resolved_instruction)
        except Exception:
            logger.debug("[agent] Could not set system prompt dynamically; continuing without it.")

        user_msg = types.Content(role="user", parts=[types.Part(text=resolved_user_text)])
        final_text: Optional[str] = None

        # Trace this turn in Langfuse (SYNC context manager!)
        span_ctx = None
        try:
            span_ctx = langfuse.start_as_current_span(name=self.app_name) if langfuse else None
        except Exception:
            span_ctx = None

        # Helper to run the ADK runner and collect final text
        async def _run_adk() -> Optional[str]:
            text: Optional[str] = None
            stream = self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=user_msg,
            )
            # Some ADK versions return async generator; others return awaitable
            if hasattr(stream, "__aiter__"):
                async for event in stream:
                    if event.is_final_response():
                        try:
                            text = event.content.parts[0].text
                        except Exception:
                            text = None
            else:
                if inspect.isawaitable(stream):
                    result = await stream
                    try:
                        text = result.content.parts[0].text  # type: ignore[attr-defined]
                    except Exception:
                        text = None
            return text

        try:
            if span_ctx:
                with span_ctx as span:  # sync context
                    final_text = await _run_adk()

                    # best-effort prompt linking metadata
                    try:
                        meta = {
                            "provider": "openrouter",
                            "model": self._normalize_model(self._model_name),
                            "adk": True,
                        }
                        # If PM provided a concrete prompt and linking is allowed
                        if lf_prompt_obj is not None and getattr(prompt_vars, "link_prompt", True):
                            meta.update({
                                "langfuse_prompt_name": getattr(lf_prompt_obj, "name", None),
                                "langfuse_prompt_version": getattr(lf_prompt_obj, "version", None),
                                "langfuse_fetch_type": getattr(prompt_vars, "fetch_type", None),
                                "langfuse_apply": getattr(prompt_vars, "apply", None),
                            })
                        # Attach inputs/outputs
                        span.update_trace(
                            input={"system_instruction": resolved_instruction, "user_text": resolved_user_text},
                            output=final_text or "",
                            user_id=self.user_id,
                            session_id=self.session_id,
                            tags=["agent", "adk", "lite-llm"],
                            metadata=meta,
                            version="1.0.0",
                        )
                    except Exception:
                        pass

                try:
                    langfuse.flush()
                except Exception:
                    pass
            else:
                # no span available; just run
                final_text = await _run_adk()

            # If ADK produced nothing, fallback via LiteLLM
            if not (final_text and final_text.strip()):
                logger.warning("[agent] Runner produced no final text; using direct LLM fallback.")
                fb_msgs: List[Dict[str, str]] = []
                sys_txt = resolved_instruction
                if sys_txt:
                    fb_msgs.append({"role": "system", "content": sys_txt})
                fb_msgs.append({"role": "user", "content": resolved_user_text})
                final_text = await self._direct_llm_fallback(fb_msgs)

            return final_text

        except Exception as e:
            logger.error(f"[agent] Error running agent: {e}")
            # Hard fallback on exceptions too
            fb_msgs: List[Dict[str, str]] = []
            sys_txt = resolved_instruction
            if sys_txt:
                fb_msgs.append({"role": "system", "content": sys_txt})
            fb_msgs.append({"role": "user", "content": resolved_user_text})
            return await self._direct_llm_fallback(fb_msgs)
