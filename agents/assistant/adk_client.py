# agents/assistant/adk_client.py
from __future__ import annotations
from typing import Callable, Iterable, Any, Optional, List
import os
from loguru import logger

from utilities.base.base_agent import BaseAgent


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
        resolved_instruction = instruction or "You are a helpful, concise assistant."

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
        logger.info(
            f"ADKClient initialized (model='{resolved_model}', agent='{agent_name}', app='{app_name}')"
        )

    # ---------- Tool registration ----------

    def register_tool(self, func: Callable[..., Any]) -> None:
        """
        Register a Python callable as a tool (docstring + type hints recommended).
        Requires BaseAgent to provide add_tool() and .tools property.
        """
        self.add_tool(func)
        logger.info(f"Registered tool: {getattr(func, '__name__', 'tool')} (total={len(self.tools)})")

    def register_tools(self, funcs: Iterable[Callable[..., Any]]) -> None:
        for f in funcs:
            self.register_tool(f)

    # ---------- Convenience run ----------

    async def run(self, prompt: str, *, system_prompt: Optional[str] = None) -> str:
        """
        Single-turn ask; returns assistant text.
        If system_prompt is provided, it is inlined at the top of the message.
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
